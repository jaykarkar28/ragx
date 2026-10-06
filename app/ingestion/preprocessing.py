
"""Text preprocessing utilities for the RAGX ingestion pipeline."""

import re
import unicodedata

from app.ingestion.loaders import LoadedDocument


class TextPreprocessor:
    """Clean and normalize extracted document text."""

    def __init__(
        self,
        normalize_unicode: bool = True,
        fix_hyphenation: bool = True,
        preserve_paragraphs: bool = True,
    ):
        self.normalize_unicode = normalize_unicode
        self.fix_hyphenation = fix_hyphenation
        self.preserve_paragraphs = preserve_paragraphs

    def preprocess_text(self, text: str) -> str:
        """Clean a single text string."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return ""

        # Normalize Unicode characters.
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # Normalize line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove null bytes and unwanted control characters.
        text = text.replace("\x00", "")
        text = "".join(
            char
            for char in text
            if char in "\n\t" or unicodedata.category(char)[0] != "C"
        )

        # Fix words split by hyphenation at line endings.
        if self.fix_hyphenation:
            text = re.sub(
                r"(?<=\w)-\n(?=\w)",
                "",
                text,
            )

        # Replace tabs and repeated horizontal whitespace.
        text = text.replace("\t", " ")
        text = re.sub(r"[^\S\n]+", " ", text)

        # Remove whitespace around line breaks.
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        if self.preserve_paragraphs:
            # Keep paragraph breaks but remove excessive blank lines.
            text = re.sub(r"\n{3,}", "\n\n", text)
        else:
            # Flatten line breaks into spaces.
            text = re.sub(r"\s+", " ", text)

        return text.strip()

    def preprocess_document(
        self,
        document: LoadedDocument,
    ) -> LoadedDocument:
        """Preprocess a document while preserving its metadata."""
        cleaned_content = self.preprocess_text(document.content)

        return LoadedDocument(
            content=cleaned_content,
            metadata=document.metadata.copy(),
        )

    def preprocess_documents(
        self,
        documents: list[LoadedDocument],
    ) -> list[LoadedDocument]:
        """Preprocess multiple documents and skip empty results."""
        processed = []

        for document in documents:
            cleaned_document = self.preprocess_document(document)

            if cleaned_document.content:
                processed.append(cleaned_document)

        return processed