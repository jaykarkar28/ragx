
from __future__ import annotations

from typing import Any

from app.retrieval.dense import DenseRetriever
from app.retrieval.keyword import KeywordRetriever


class HybridRetriever:
    """Coordinate dense and keyword retrieval."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        keyword_retriever: KeywordRetriever,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.keyword_retriever = keyword_retriever

    def retrieve_candidates(
        self,
        query: str,
        dense_top_k: int = 5,
        keyword_top_k: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """Retrieve candidates independently from both retrievers."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if dense_top_k <= 0 or keyword_top_k <= 0:
            raise ValueError("top_k values must be positive.")

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=dense_top_k,
        )

        keyword_results = self.keyword_retriever.retrieve(
            query=query,
            top_k=keyword_top_k,
        )

        return {
            "dense": dense_results,
            "keyword": keyword_results,
        }