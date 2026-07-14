from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


class ShoppingDataStore:
    """In-memory indexes over the local shopping dataset."""

    def __init__(self, json_path: Path) -> None:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.metadata = payload.get("metadata", {})
        self.customers = payload.get("customers", [])
        self.orders = payload.get("orders", [])
        self.vouchers = payload.get("vouchers", [])

        self.customer_by_id = {
            str(item["customer_id"]).upper(): item for item in self.customers
        }
        self.order_by_id = {str(item["order_id"]): item for item in self.orders}
        self.orders_by_customer_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.vouchers_by_customer_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for order in self.orders:
            self.orders_by_customer_id[str(order["customer_id"]).upper()].append(order)
        for voucher in self.vouchers:
            self.vouchers_by_customer_id[str(voucher["customer_id"]).upper()].append(voucher)
        for orders in self.orders_by_customer_id.values():
            orders.sort(key=lambda item: item.get("created_at", ""), reverse=True)

    def get_customer_by_id(self, customer_id: str) -> dict[str, Any]:
        customer_id = customer_id.strip().upper()
        customer = self.customer_by_id.get(customer_id)
        if customer is None:
            return {"status": "not_found", "customer_id": customer_id}
        return {"status": "ok", "customer": customer}

    def get_orders_by_customer_id(self, customer_id: str, limit: int = 10) -> dict[str, Any]:
        customer_id = customer_id.strip().upper()
        if customer_id not in self.customer_by_id:
            return {"status": "not_found", "customer_id": customer_id}
        limit = max(0, min(limit, 100))
        return {
            "status": "ok",
            "customer_id": customer_id,
            "orders": self.orders_by_customer_id.get(customer_id, [])[:limit],
        }

    def get_order_detail_by_order_id(self, order_id: str) -> dict[str, Any]:
        order_id = order_id.strip()
        order = self.order_by_id.get(order_id)
        if order is None:
            return {"status": "not_found", "order_id": order_id}
        return {"status": "ok", "order": order}

    def get_vouchers_by_customer_id(
        self,
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        customer_id = customer_id.strip().upper()
        if customer_id not in self.customer_by_id:
            return {"status": "not_found", "customer_id": customer_id}
        vouchers = self.vouchers_by_customer_id.get(customer_id, [])
        if only_active:
            vouchers = [
                item
                for item in vouchers
                if item.get("status") in {"active", "restored"}
                and item.get("remaining_uses", 0) > 0
            ]
        return {
            "status": "ok",
            "customer_id": customer_id,
            "vouchers": vouchers,
        }


def build_data_tools(store: ShoppingDataStore) -> list:
    @tool
    def get_customer_by_id(customer_id: str) -> dict[str, Any]:
        """Tra cứu hồ sơ, hạng và hạn mức voucher bằng mã khách hàng, ví dụ C001."""
        return store.get_customer_by_id(customer_id)

    @tool
    def get_orders_by_customer_id(
        customer_id: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Lấy các đơn gần nhất của một mã khách hàng, mới nhất đứng trước."""
        return store.get_orders_by_customer_id(customer_id, limit)

    @tool
    def get_order_detail_by_order_id(order_id: str) -> dict[str, Any]:
        """Tra cứu trạng thái, giao hàng và đổi trả của đúng một mã đơn hàng."""
        return store.get_order_detail_by_order_id(order_id)

    @tool
    def get_vouchers_by_customer_id(
        customer_id: str,
        only_active: bool = False,
    ) -> dict[str, Any]:
        """Lấy voucher của khách; đặt only_active=true nếu chỉ cần mã còn dùng được."""
        return store.get_vouchers_by_customer_id(customer_id, only_active)

    return [
        get_customer_by_id,
        get_orders_by_customer_id,
        get_order_detail_by_order_id,
        get_vouchers_by_customer_id,
    ]
