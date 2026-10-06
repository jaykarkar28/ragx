
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorStore:
    """Local FAISS vector store using cosine similarity."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[dict[str, Any]] = []

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array.")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0

        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def add_documents(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Add text chunks and their embeddings to the index."""
        if len(texts) == 0:
            return []

        if embeddings.shape[0] != len(texts):
            raise ValueError("Texts and embeddings must have the same length.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        if metadatas is None:
            metadatas = [{} for _ in texts]

        if len(metadatas) != len(texts):
            raise ValueError("Texts and metadatas must have the same length.")

        vectors = self._normalize_vectors(embeddings)
        start_id = len(self.documents)
        ids = list(range(start_id, start_id + len(texts)))

        self.index.add(vectors)

        for text, metadata in zip(texts, metadatas):
            self.documents.append({
                "text": text,
                "metadata": metadata,
            })

        return ids

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the most similar documents for a query embedding."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape != (1, self.dimension):
            raise ValueError(
                f"Expected query shape (1, {self.dimension}), "
                f"got {query.shape}."
            )

        query = self._normalize_vectors(query)
        scores, indices = self.index.search(
            query,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            document = self.documents[int(idx)]

            results.append({
                "id": int(idx),
                "text": document["text"],
                "metadata": document["metadata"],
                "score": float(score),
            })

        return results

    def save(self, directory: str | Path) -> None:
        """Save the FAISS index and document metadata."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))

        with (directory / "documents.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(self.documents, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        """Load a previously saved vector store."""
        directory = Path(directory)

        index_path = directory / "index.faiss"
        documents_path = directory / "documents.json"

        if not index_path.exists() or not documents_path.exists():
            raise FileNotFoundError(
                "Vector store files were not found in the directory."
            )

        index = faiss.read_index(str(index_path))

        with documents_path.open("r", encoding="utf-8") as file:
            documents = json.load(file)

        if index.ntotal != len(documents):
            raise ValueError(
                "FAISS index size does not match the document metadata."
            )

        store = cls(dimension=index.d)
        store.index = index
        store.documents = documents

        return store

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return self.index.ntotal