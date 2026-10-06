
from __future__ import annotations

from typing import Any


class ReciprocalRankFusion:
    """Combine multiple ranked result lists using RRF."""

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be positive.")

        self.k = k

    @staticmethod
    def _document_key(document: dict[str, Any]) -> str:
        """Create a stable key for deduplicating retrieved documents."""
        return str(document["text"])

    def fuse(
        self,
        results: dict[str, list[dict[str, Any]]],
        top_k: int = 5,
        weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Fuse ranked results from multiple retrieval methods."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if weights is None:
            weights = {name: 1.0 for name in results}

        fused: dict[str, dict[str, Any]] = {}

        for source, documents in results.items():
            weight = weights.get(source, 1.0)

            if weight < 0:
                raise ValueError("Fusion weights cannot be negative.")

            for rank, document in enumerate(documents, start=1):
                key = self._document_key(document)

                if key not in fused:
                    fused[key] = {
                        "id": document.get("id"),
                        "text": document["text"],
                        "metadata": document.get("metadata", {}),
                        "fused_score": 0.0,
                        "sources": [],
                    }

                fused[key]["fused_score"] += weight / (self.k + rank)

                if source not in fused[key]["sources"]:
                    fused[key]["sources"].append(source)

        ranked = sorted(
            fused.values(),
            key=lambda item: item["fused_score"],
            reverse=True,
        )

        return ranked[:top_k]