
from __future__ import annotations

from typing import Any

from app.indexing.bm25_index import BM25Index


class KeywordRetriever:
    """Retrieve relevant documents using BM25 keyword matching."""

    def __init__(self, bm25_index: BM25Index) -> None:
        self.bm25_index = bm25_index

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the top matching documents for a query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        return self.bm25_index.search(
            query=query,
            top_k=top_k,
        )