
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Index:
    """Keyword-based document retrieval using BM25."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
        self.index: BM25Okapi | None = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Convert text into lowercase word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from all stored documents."""
        self.tokenized_corpus = [
            self.tokenize(document["text"])
            for document in self.documents
        ]

        if not any(self.tokenized_corpus):
            self.index = None
            return

        self.index = BM25Okapi(self.tokenized_corpus)

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add documents and rebuild the BM25 index."""
        if not texts:
            return []

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All documents must be strings.")

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(texts) != len(metadatas):
            raise ValueError(
                "Texts and metadatas must have the same length."
            )

        start_id = len(self.documents)
        ids = list(range(start_id, start_id + len(texts)))

        for text, metadata in zip(texts, metadatas):
            self.documents.append({
                "text": text,
                "metadata": metadata,
            })

        self._rebuild_index()
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant documents using BM25 scores."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_tokens = self.tokenize(query)

        if not query_tokens or self.index is None:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )

        results = []

        for idx in ranked_indices:
            if len(results) >= top_k:
                break

            results.append({
                "id": idx,
                "text": self.documents[idx]["text"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(scores[idx]),
            })

        return results

    def save(self, directory: str | Path) -> None:
        """Save the document corpus for later index reconstruction."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with (directory / "bm25_documents.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                self.documents,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path) -> BM25Index:
        """Load documents and rebuild the BM25 index."""
        path = Path(directory) / "bm25_documents.json"

        if not path.exists():
            raise FileNotFoundError(
                f"BM25 document file not found: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            documents = json.load(file)

        instance = cls()
        instance.documents = documents
        instance._rebuild_index()
        return instance

    @property
    def size(self) -> int:
        """Number of stored documents."""
        return len(self.documents)