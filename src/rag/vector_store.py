from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain_core.tools import tool

from rag.parser import parse_policy_markdown


class ChromaPolicyStore:
    """Persistent Chroma index backed by explicit local embeddings."""

    def __init__(
        self,
        persist_directory: Path,
        embedding_model: Any,
        collection_name: str = "policy_chunks",
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Any:
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ensure_index(self, markdown_path: Path) -> None:
        if self.collection.count() == 0:
            self.rebuild(markdown_path)

    def rebuild(self, markdown_path: Path) -> None:
        chunks = parse_policy_markdown(markdown_path.read_text(encoding="utf-8"))
        if not chunks:
            raise ValueError(f"No policy chunks found in {markdown_path}")

        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        documents = [chunk["rendered_text"] for chunk in chunks]
        metadatas = [
            {
                "source": markdown_path.name,
                "section_h2": chunk["section_h2"],
                "section_h3": chunk["section_h3"],
                "citation": chunk["citation"],
            }
            for chunk in chunks
        ]
        self.collection.add(
            ids=[f"policy-{index:04d}" for index in range(len(chunks))],
            documents=documents,
            metadatas=metadatas,
            embeddings=self.embedding_model.embed_documents(documents),
        )

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        query = query.strip()
        count = self.collection.count()
        if not query or count == 0 or top_k <= 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedding_model.embed_query(query)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {
                "citation": (metadata or {}).get("citation", ""),
                "content": document,
                "distance": distance,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]


def build_policy_tool(store: ChromaPolicyStore, default_top_k: int = 4) -> Any:
    @tool
    def search_policy(query: str, top_k: int = default_top_k) -> list[dict[str, Any]]:
        """Tìm các đoạn chính sách VinShop liên quan và trả nội dung kèm citation."""
        return store.search(query, top_k)

    return search_policy
