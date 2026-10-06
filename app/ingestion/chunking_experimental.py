
"""Experimental boundary-aware chunking for the RAGX pipeline."""

from app.ingestion.loaders import LoadedDocument


class BoundaryAwareTextChunker:
    """Split text at natural boundaries while preserving overlap."""

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

        # Prefer paragraph, line, sentence, then word boundaries.
        separators = [
            ("\n\n", 2),
            ("\n", 1),
            (". ", 1),
            ("? ", 1),
            ("! ", 1),
            (" ", 0),
        ]

        for separator, boundary_offset in separators:
            position = text.rfind(
                separator,
                min_boundary,
                target_end + 1,
            )

            if position != -1:
                return position + boundary_offset

        # No suitable boundary before the target. Look ahead to
        # the next whitespace so we do not split a word.
        next_space = len(text)

        for whitespace in (" ", "\n", "\t"):
            position = text.find(whitespace, target_end)

            if position != -1:
                next_space = min(next_space, position)

        if next_space < len(text):
            return next_space

        # An unbroken token longer than the chunk size cannot
        # be split without breaking that token.
        return target_end

    def _align_start(
        self,
        text: str,
        proposed_start: int,
        previous_start: int,
    ) -> int:
        """Align the next chunk start to a word boundary."""
        if proposed_start <= previous_start:
            proposed_start = previous_start + 1

        if proposed_start >= len(text):
            return len(text)

        # If the proposed position is inside a word, move
        # backward to preserve overlap and start at a word.
        if (
            proposed_start > 0
            and not text[proposed_start - 1].isspace()
            and not text[proposed_start].isspace()
        ):
            while (
                proposed_start > previous_start + 1
                and not text[proposed_start - 1].isspace()
            ):
                proposed_start -= 1

        # Skip whitespace so chunks begin with actual content.
        while (
            proposed_start < len(text)
            and text[proposed_start].isspace()
        ):
            proposed_start += 1

        return proposed_start

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

            if end <= start:
                end = min(
                    start + self.chunk_size,
                    text_length,
                )

            raw_chunk = text[start:end]
            chunk_text = raw_chunk.strip()

            if chunk_text:
                # Keep metadata aligned with the actual trimmed text.
                leading_trim = len(raw_chunk) - len(raw_chunk.lstrip())
                trailing_trim = len(raw_chunk) - len(raw_chunk.rstrip())

                actual_start = start + leading_trim
                actual_end = end - trailing_trim

                metadata = document.metadata.copy()
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "char_start": actual_start,
                        "char_end": actual_end,
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

            proposed_start = end - self.chunk_overlap
            next_start = self._align_start(
                text,
                proposed_start,
                start,
            )

            if next_start <= start:
                next_start = end

            start = next_start

        # Merge a tiny final addition into the previous chunk when
        # the chunks overlap and the merged size remains reasonable.
        if len(chunks) >= 2:
            previous = chunks[-2]
            final = chunks[-1]

            previous_start = previous.metadata.get("char_start")
            previous_end = previous.metadata.get("char_end")
            final_start = final.metadata.get("char_start")
            final_end = final.metadata.get("char_end")

            if all(
                value is not None
                for value in (
                    previous_start,
                    previous_end,
                    final_start,
                    final_end,
                )
            ):
                new_content_length = final_end - previous_end
                merged_length = final_end - previous_start

                max_new_content = max(
                    1,
                    int(self.chunk_size * 0.20),
                )
                max_merged_length = int(
                    self.chunk_size * 1.10
                )

                should_merge = (
                    final_start <= previous_end
                    and 0 <= new_content_length <= max_new_content
                    and merged_length <= max_merged_length
                )

                if should_merge:
                    previous.content = text[
                        previous_start:final_end
                    ].strip()
                    previous.metadata["char_end"] = final_end
                    chunks.pop()

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
