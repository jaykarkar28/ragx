
"""Local embedding generation for the RAGX framework."""

from functools import cached_property

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingModel:
    """Generate normalized embeddings using a local Sentence Transformer."""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.normalize_embeddings = normalize_embeddings

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

    @cached_property
    def model(self) -> SentenceTransformer:
        """Load and cache the embedding model on first use."""
        return SentenceTransformer(self.model_name)

    @property
    def embedding_dimension(self) -> int:
        """Return the model's embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Convert a list of texts into a 2D embedding matrix."""
        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All items in texts must be strings")

        if any(not text.strip() for text in texts):
            raise ValueError("Texts must not contain empty strings")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        return np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

    def embed_text(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text."""
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a search query."""
        return self.embed_text(query)

    def embed_documents(self, documents: list) -> np.ndarray:
        """Generate embeddings from LoadedDocument objects."""
        texts = [document.content for document in documents]
        return self.embed_texts(texts)