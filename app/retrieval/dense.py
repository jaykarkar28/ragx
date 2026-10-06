
from __future__ import annotations

from typing import Any

from app.indexing.embeddings import EmbeddingModel
from app.indexing.vector_store import VectorStore


class DenseRetriever:
    """Retrieve relevant documents using dense vector similarity."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Embed a query and retrieve the most similar documents."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = self.embedding_model.embed_query(query)

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )