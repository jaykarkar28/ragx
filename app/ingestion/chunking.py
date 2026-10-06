
"""Recursive character-based chunking for the RAGX pipeline."""

from app.ingestion.loaders import LoadedDocument


class TextChunker:
    """Split documents into overlapping, context-preserving chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Prefer natural text boundaries before splitting arbitrarily.
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _find_split_point(
        self,
        text: str,
        start: int,
        target_end: int,
    ) -> int:
        """Find a natural boundary near the target end."""
        if target_end >= len(text):
            return len(text)

        min_boundary = start + self.chunk_size // 2

        for separator in self.separators:
            if not separator:
                continue

            boundary = text.rfind(
                separator,
                min_boundary,
                target_end,
            )

            if boundary != -1:
                return boundary + len(separator)

        return target_end

    def chunk_document(
        self,
        document: LoadedDocument,
    ) -> list[LoadedDocument]:
        """Split one document into overlapping chunks."""
        text = document.content.strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            metadata = document.metadata.copy()
            metadata.update(
                {
                    "chunk_index": 0,
                    "char_start": 0,
                    "char_end": len(text),
                }
            )
            return [
                LoadedDocument(
                    content=text,
                    metadata=metadata,
                )
            ]

        chunks = []
        start = 0
        chunk_index = 0
        text_length = len(text)

        while start < text_length:
            target_end = min(
                start + self.chunk_size,
                text_length,
            )

            end = self._find_split_point(
                text,
                start,
                target_end,
            )

            chunk_text = text[start:end].strip()

            if chunk_text:
                metadata = document.metadata.copy()
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                    }
                )

                chunks.append(
                    LoadedDocument(
                        content=chunk_text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

            if end >= text_length:
                break

            next_start = end - self.chunk_overlap

            # Ensure forward progress, even with unusual boundaries.
            start = max(next_start, start + 1)

        return chunks

    def chunk_documents(
        self,
        documents: list[LoadedDocument],
    ) -> list[LoadedDocument]:
        """Chunk multiple documents and combine their results."""
        all_chunks = []

        for document in documents:
            all_chunks.extend(self.chunk_document(document))

        return all_chunks