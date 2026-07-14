from __future__ import annotations

import json
from functools import partial
from pathlib import Path
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.data_access import ShoppingDataStore, build_data_tools
from app.prompts import (
    DATA_WORKER_PROMPT,
    POLICY_WORKER_PROMPT,
    RESPONSE_WORKER_PROMPT,
    SUPERVISOR_PROMPT,
)
from app.state import ShoppingState
from app.utils import dump_json, extract_json_payload, serialize_message, timestamp_utc
from provider import get_chat_model
from rag.embeddings import SentenceTransformerEmbeddings
from rag.vector_store import ChromaPolicyStore, build_policy_tool


class ShoppingAssistant:
    """Shopping assistant composed of a supervisor and three workers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.model = get_chat_model(self.settings)
        self.data_store = ShoppingDataStore(self.settings.orders_path)
        self.data_tools = build_data_tools(self.data_store)
        embeddings = SentenceTransformerEmbeddings(self.settings.embedding_model_name)
        self.policy_store = ChromaPolicyStore(self.settings.chroma_dir, embeddings)
        self.policy_store.ensure_index(self.settings.policy_path)
        self.policy_tool = build_policy_tool(self.policy_store, self.settings.top_k)
        self.graph = build_graph(
            self.model,
            self.policy_tool,
            self.data_tools,
            self.settings.top_k,
        )

    def ask(
        self,
        question: str,
        trace_file: Path | None = None,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        if rebuild_index:
            self.policy_store.rebuild(self.settings.policy_path)

        state = self.graph.invoke({"question": question, "trace": []})
        payload = {
            "status": _state_status(state),
            "route": state.get("route", {}),
            "policy_result": state.get("policy_result", {}),
            "data_result": state.get("data_result", {}),
            "final_answer": state.get("final_answer", ""),
            "trace": state.get("trace", []),
        }
        if trace_file:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_file.write_text(dump_json(payload), encoding="utf-8")
        return payload

    def run_batch(
        self,
        test_file: Path,
        output_dir: Path,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        cases = json.loads(test_file.read_text(encoding="utf-8"))
        output_dir.mkdir(parents=True, exist_ok=True)
        if rebuild_index:
            self.policy_store.rebuild(self.settings.policy_path)

        results: list[dict[str, Any]] = []
        for case in cases:
            case_id = str(case["id"])
            try:
                result = self.ask(
                    str(case["question"]),
                    trace_file=output_dir / f"{case_id}.json",
                )
                actual_route = set(result["route"].get("selected_workers", []))
                route_ok = actual_route == set(case.get("expected_route", []))
                status_ok = result["status"] == case.get("expected_status", "ok")
                answer = result["final_answer"].casefold()
                missing = [
                    text
                    for text in case.get("expected_contains", [])
                    if str(text).casefold() not in answer
                ]
                passed = route_ok and status_ok and not missing
                results.append(
                    {
                        "id": case_id,
                        "passed": passed,
                        "expected_route": case.get("expected_route", []),
                        "actual_route": sorted(actual_route),
                        "expected_status": case.get("expected_status", "ok"),
                        "actual_status": result["status"],
                        "missing_expected_text": missing,
                    }
                )
            except Exception as exc:
                error_result = {"id": case_id, "passed": False, "error": str(exc)}
                results.append(error_result)
                (output_dir / f"{case_id}.json").write_text(
                    dump_json(error_result),
                    encoding="utf-8",
                )

        passed = sum(item["passed"] for item in results)
        summary = {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": results,
        }
        (output_dir / "summary.json").write_text(
            dump_json(summary),
            encoding="utf-8",
        )
        return summary


def build_graph(
    model: BaseChatModel,
    policy_tool: BaseTool,
    data_tools: list[BaseTool],
    top_k: int = 4,
) -> Any:
    builder = StateGraph(ShoppingState)
    builder.add_node("supervisor", partial(supervisor_node, model=model))
    builder.add_node(
        "worker_1_policy",
        partial(worker_1_policy_node, model=model, policy_tool=policy_tool, top_k=top_k),
    )
    builder.add_node(
        "worker_2_data",
        partial(worker_2_data_node, model=model, tools=data_tools),
    )
    builder.add_node(
        "worker_3_response",
        partial(worker_3_response_node, model=model),
    )
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "policy": "worker_1_policy",
            "data": "worker_2_data",
            "response": "worker_3_response",
        },
    )
    builder.add_conditional_edges(
        "worker_1_policy",
        lambda state: "data" if state["route"].get("needs_data") else "response",
        {"data": "worker_2_data", "response": "worker_3_response"},
    )
    builder.add_edge("worker_2_data", "worker_3_response")
    builder.add_edge("worker_3_response", END)
    return builder.compile()


def supervisor_node(
    state: ShoppingState,
    *,
    model: BaseChatModel,
) -> ShoppingState:
    question = state["question"]
    response = model.invoke(
        [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=question)]
    )
    route = _normalize_route(extract_json_payload(_message_text(response)), question)
    return {
        "route": route,
        "trace": [
            {
                "timestamp": timestamp_utc(),
                "node": "supervisor",
                "output": route,
            }
        ],
    }


def worker_1_policy_node(
    state: ShoppingState,
    *,
    model: BaseChatModel,
    policy_tool: BaseTool,
    top_k: int,
) -> ShoppingState:
    question = state["question"]
    hits = policy_tool.invoke({"query": question, "top_k": top_k})
    if not hits:
        result = {
            "status": "not_found",
            "summary": "Không tìm thấy chính sách liên quan.",
            "facts": [],
            "citations": [],
        }
    else:
        response = model.invoke(
            [
                SystemMessage(content=POLICY_WORKER_PROMPT),
                HumanMessage(
                    content=f"Câu hỏi: {question}\nPolicy chunks:\n{dump_json(hits)}"
                ),
            ]
        )
        payload = extract_json_payload(_message_text(response))
        allowed_citations = [hit["citation"] for hit in hits if hit.get("citation")]
        citations = [
            item
            for item in _string_list(payload.get("citations"))
            if item in allowed_citations
        ]
        result = {
            "status": "ok",
            "summary": str(payload.get("summary") or _message_text(response)),
            "facts": _string_list(payload.get("facts")),
            "citations": citations or allowed_citations,
        }
    return {
        "policy_result": result,
        "trace": [
            {
                "timestamp": timestamp_utc(),
                "node": "worker_1_policy",
                "tool": policy_tool.name,
                "hits": hits,
                "output": result,
            }
        ],
    }


def worker_2_data_node(
    state: ShoppingState,
    *,
    model: BaseChatModel,
    tools: list[BaseTool],
) -> ShoppingState:
    agent_error = None
    try:
        payload, messages, tool_results = _run_data_agent(
            model,
            state["question"],
            tools,
        )
    except Exception as exc:
        payload, messages, tool_results = {}, [], []
        agent_error = str(exc)
    if not tool_results:
        tool_results = _fallback_data_tools(state["question"], tools)
    result = _normalize_data_result(payload, tool_results)
    return {
        "data_result": result,
        "trace": [
            {
                "timestamp": timestamp_utc(),
                "node": "worker_2_data",
                "tools": [item["tool"] for item in tool_results],
                "agent_error": agent_error,
                "messages": [serialize_message(message) for message in messages],
                "tool_results": tool_results,
                "output": result,
            }
        ],
    }


def worker_3_response_node(
    state: ShoppingState,
    *,
    model: BaseChatModel,
) -> ShoppingState:
    route = state.get("route", {})
    policy_result = state.get("policy_result", {})
    data_result = state.get("data_result", {})

    if route.get("status") == "clarification_needed":
        final_answer = (
            "Status: clarification_needed\n"
            f"Question: {route.get('clarification_question')}"
        )
    elif data_result.get("status") == "clarification_needed":
        final_answer = (
            "Status: clarification_needed\n"
            f"Question: {data_result.get('summary', 'Vui lòng cung cấp thêm định danh.')}"
        )
    elif data_result.get("status") == "not_found":
        final_answer = (
            "Status: not_found\n"
            f"Message: {data_result.get('summary', 'Không tìm thấy dữ liệu yêu cầu.')}"
        )
    elif policy_result.get("status") == "not_found":
        final_answer = (
            "Status: not_found\n"
            f"Message: {policy_result.get('summary', 'Không tìm thấy chính sách liên quan.')}"
        )
    else:
        context = {
            "question": state["question"],
            "route": route,
            "policy_result": policy_result,
            "data_result": data_result,
        }
        response = model.invoke(
            [
                SystemMessage(content=RESPONSE_WORKER_PROMPT),
                HumanMessage(content=dump_json(context)),
            ]
        )
        final_answer = _ensure_success_format(
            _message_text(response),
            policy_result,
            data_result,
        )

    return {
        "final_answer": final_answer,
        "trace": [
            {
                "timestamp": timestamp_utc(),
                "node": "worker_3_response",
                "status": _answer_status(final_answer),
                "final_answer": final_answer,
            }
        ],
    }


def _run_data_agent(
    model: BaseChatModel,
    question: str,
    tools: list[BaseTool],
) -> tuple[dict[str, Any], list[BaseMessage], list[dict[str, Any]]]:
    model_with_tools = model.bind_tools(tools)
    tool_by_name = {item.name: item for item in tools}
    messages: list[BaseMessage] = [
        SystemMessage(content=DATA_WORKER_PROMPT),
        HumanMessage(content=question),
    ]
    tool_results: list[dict[str, Any]] = []

    for _ in range(6):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        calls = getattr(response, "tool_calls", [])
        if not calls:
            return extract_json_payload(_message_text(response)), messages, tool_results
        for call in calls:
            name = call.get("name", "")
            selected_tool = tool_by_name.get(name)
            if selected_tool is None:
                result: Any = {"status": "error", "message": f"Unknown tool: {name}"}
            else:
                try:
                    result = selected_tool.invoke(call.get("args", {}))
                except Exception as exc:
                    result = {"status": "error", "message": str(exc)}
            tool_results.append({"tool": name, "result": result})
            messages.append(
                ToolMessage(
                    content=dump_json(result),
                    tool_call_id=str(call.get("id", name)),
                    name=name,
                )
            )
    return {}, messages, tool_results


def _fallback_data_tools(question: str, tools: list[BaseTool]) -> list[dict[str, Any]]:
    """Use real tools if a provider returns no tool call."""
    tool_by_name = {item.name: item for item in tools}
    order_id, customer_id = _extract_entities(question)
    selected_name = ""
    arguments: dict[str, Any] = {}
    normalized = question.casefold()

    if order_id:
        selected_name = "get_order_detail_by_order_id"
        arguments = {"order_id": order_id}
    elif customer_id:
        if "đơn" in normalized or "order" in normalized:
            selected_name = "get_orders_by_customer_id"
            arguments = {"customer_id": customer_id}
        elif "voucher" in normalized and not any(
            word in normalized for word in ("quota", "hạng", "tối đa")
        ):
            selected_name = "get_vouchers_by_customer_id"
            arguments = {"customer_id": customer_id, "only_active": "còn" in normalized}
        else:
            selected_name = "get_customer_by_id"
            arguments = {"customer_id": customer_id}

    selected_tool = tool_by_name.get(selected_name)
    if selected_tool is None:
        return []
    try:
        result = selected_tool.invoke(arguments)
    except Exception as exc:
        result = {"status": "error", "message": str(exc)}
    return [{"tool": selected_name, "result": result}]


def _normalize_data_result(
    payload: dict[str, Any],
    tool_results: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_results = [item.get("result", {}) for item in tool_results]
    missing = [item for item in raw_results if item.get("status") == "not_found"]
    errors = [item for item in raw_results if item.get("status") == "error"]
    if missing:
        entities = [
            str(value)
            for item in missing
            for key, value in item.items()
            if key.endswith("_id")
        ]
        return {
            "status": "not_found",
            "summary": f"Không tìm thấy dữ liệu cho: {', '.join(entities)}.",
            "facts": [],
            "records": raw_results,
            "missing_fields": [],
            "not_found_entities": entities,
        }
    if errors:
        return {
            "status": "not_found",
            "summary": "Không thể hoàn tất tra cứu dữ liệu.",
            "facts": [],
            "records": raw_results,
            "missing_fields": [],
            "not_found_entities": [],
        }
    if not tool_results:
        return {
            "status": "clarification_needed",
            "summary": str(
                payload.get("summary")
                or "Vui lòng cung cấp mã đơn hàng hoặc mã khách hàng cần tra cứu."
            ),
            "facts": [],
            "records": [],
            "missing_fields": _string_list(payload.get("missing_fields")),
            "not_found_entities": [],
        }
    return {
        "status": "ok",
        "summary": str(payload.get("summary") or "Đã tra cứu dữ liệu thành công."),
        "facts": _string_list(payload.get("facts")) or [dump_json(item) for item in raw_results],
        "records": raw_results,
        "missing_fields": _string_list(payload.get("missing_fields")),
        "not_found_entities": _string_list(payload.get("not_found_entities")),
    }


def _normalize_route(payload: dict[str, Any], question: str) -> dict[str, Any]:
    fallback = _fallback_route(question)
    order_id, customer_id = _extract_entities(question)
    normalized = question.casefold()
    personal = "của tôi" in normalized

    if personal and not order_id and ("đơn" in normalized or "order" in normalized):
        return _clarification_route("Vui lòng cung cấp mã đơn hàng để tôi kiểm tra.")
    if personal and not customer_id and ("voucher" in normalized or "khách" in normalized):
        return _clarification_route("Vui lòng cung cấp mã khách hàng để tôi kiểm tra.")

    status = payload.get("status")
    needs_policy = payload.get("needs_policy")
    needs_data = payload.get("needs_data")
    if status not in {"ok", "clarification_needed"}:
        return fallback
    if status == "clarification_needed":
        question_text = str(
            payload.get("clarification_question")
            or fallback.get("clarification_question")
            or "Vui lòng cung cấp thêm thông tin."
        )
        return _clarification_route(question_text)
    if not isinstance(needs_policy, bool) or not isinstance(needs_data, bool):
        return fallback

    mixed_terms = ("trả", "hoàn", "hủy", "từ chối", "cửa sổ", "chính sách")
    if order_id:
        needs_data = True
        needs_policy = any(term in normalized for term in mixed_terms)
    elif customer_id:
        needs_data = True
        needs_policy = "chính sách" in normalized or "quy định" in normalized
    elif not personal:
        needs_policy, needs_data = True, False
    if not needs_policy and not needs_data:
        return fallback
    return _ok_route(needs_policy, needs_data)


def _fallback_route(question: str) -> dict[str, Any]:
    order_id, customer_id = _extract_entities(question)
    normalized = question.casefold()
    if order_id:
        mixed = any(
            term in normalized
            for term in ("trả", "hoàn", "hủy", "từ chối", "cửa sổ", "chính sách")
        )
        return _ok_route(mixed, True)
    if customer_id:
        return _ok_route(False, True)
    if "của tôi" in normalized:
        identifier = "mã đơn hàng" if "đơn" in normalized else "mã khách hàng"
        return _clarification_route(f"Vui lòng cung cấp {identifier} để tôi kiểm tra.")
    return _ok_route(True, False)


def _ok_route(needs_policy: bool, needs_data: bool) -> dict[str, Any]:
    selected = []
    if needs_policy:
        selected.append("policy")
    if needs_data:
        selected.append("data")
    return {
        "status": "ok",
        "needs_policy": needs_policy,
        "needs_data": needs_data,
        "clarification_question": None,
        "selected_workers": selected,
    }


def _clarification_route(question: str) -> dict[str, Any]:
    return {
        "status": "clarification_needed",
        "needs_policy": False,
        "needs_data": False,
        "clarification_question": question,
        "selected_workers": [],
    }


def _extract_entities(question: str) -> tuple[str | None, str | None]:
    customer_match = re.search(r"\bC\d{3,}\b", question, re.IGNORECASE)
    order_match = re.search(
        r"(?:đơn(?:\s+hàng)?|order)\s*(?:mã\s*)?#?\s*(\d{3,})",
        question,
        re.IGNORECASE,
    )
    return (
        order_match.group(1) if order_match else None,
        customer_match.group(0).upper() if customer_match else None,
    )


def _route_after_supervisor(state: ShoppingState) -> str:
    route = state["route"]
    if route.get("status") == "clarification_needed":
        return "response"
    if route.get("needs_policy"):
        return "policy"
    if route.get("needs_data"):
        return "data"
    return "response"


def _ensure_success_format(
    answer: str,
    policy_result: dict[str, Any],
    data_result: dict[str, Any],
) -> str:
    answer = answer.strip()
    if answer.startswith("Answer:") and "Evidence:" in answer:
        return answer
    evidence = []
    if policy_result:
        citations = "; ".join(policy_result.get("citations", []))
        evidence.append(f"- Policy: {policy_result.get('summary', '')} ({citations})")
    if data_result:
        evidence.append(f"- Order data: {data_result.get('summary', '')}")
    return f"Answer: {answer}\nEvidence:\n" + "\n".join(evidence)


def _state_status(state: dict[str, Any]) -> str:
    if state.get("route", {}).get("status") == "clarification_needed":
        return "clarification_needed"
    for key in ("data_result", "policy_result"):
        status = state.get(key, {}).get("status")
        if status in {"clarification_needed", "not_found"}:
            return status
    return "ok"


def _answer_status(answer: str) -> str:
    if answer.startswith("Status: clarification_needed"):
        return "clarification_needed"
    if answer.startswith("Status: not_found"):
        return "not_found"
    return "ok"


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]
