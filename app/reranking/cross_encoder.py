
from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved documents using a cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        """Load the cross-encoder lazily."""
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score and reorder documents by query relevance."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not documents:
            return []

        if any(
            not isinstance(document.get("text"), str)
            for document in documents
        ):
            raise ValueError("Every document must contain text.")

        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked