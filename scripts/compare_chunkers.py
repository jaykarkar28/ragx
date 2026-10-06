
"""Compare baseline and experimental chunking on real documents."""

from pathlib import Path

from app.config import settings
from app.ingestion.chunking import TextChunker
from app.ingestion.chunking_experimental import BoundaryAwareTextChunker
from app.ingestion.loaders import DocumentLoader
from app.ingestion.preprocessing import TextPreprocessor


def is_word_char(char: str) -> bool:
    """Return whether a character can be part of a word."""
    return char.isalnum() or char == "_"


def find_chunk_offset(
    source: str,
    chunk: str,
    expected_start: int,
) -> int:
    """Find a chunk in its source, preferring the expected position."""
    candidates = []
    position = source.find(chunk)

    while position != -1:
        candidates.append(position)
        position = source.find(chunk, position + 1)

    if not candidates:
        return -1

    return min(candidates, key=lambda pos: abs(pos - expected_start))


def get_chunk_offsets(
    source: str,
    chunks: list,
    chunk_overlap: int,
    use_metadata: bool,
) -> list[tuple[int, int]]:
    """Get source offsets for chunks from metadata or content matching."""
    offsets = []
    previous_start = 0
    previous_end = 0

    for chunk in chunks:
        if use_metadata:
            start = chunk.metadata.get("char_start")
            end = chunk.metadata.get("char_end")

            if start is not None and end is not None:
                offsets.append((start, end))
                previous_start = start
                previous_end = end
                continue

        expected_start = max(0, previous_end - chunk_overlap)
        start = find_chunk_offset(
            source,
            chunk.content,
            expected_start,
        )

        if start == -1:
            offsets.append((-1, -1))
            continue

        end = start + len(chunk.content)
        offsets.append((start, end))
        previous_start = start
        previous_end = end

    return offsets


def inspect_chunks(
    name: str,
    chunker,
    documents: list,
    chunk_overlap: int,
    use_metadata: bool = False,
) -> dict:
    """Inspect chunks and report source-boundary diagnostics."""
    total_chunks = 0
    mid_word_starts = 0
    mid_word_ends = 0
    unlocated_chunks = 0
    overlaps = []
    short_chunks = []
    chunk_lengths = []

    print(f"\n{'=' * 70}")
    print(name)
    print("=" * 70)

    for document_index, document in enumerate(documents):
        source = document.content.strip()
        chunks = chunker.chunk_document(document)
        offsets = get_chunk_offsets(
            source,
            chunks,
            chunk_overlap,
            use_metadata,
        )

        print(
            f"\nDocument {document_index + 1} "
            f"| Page: {document.metadata.get('page', 'N/A')} "
            f"| Chunks: {len(chunks)}"
        )

        for index, (chunk, (start, end)) in enumerate(
            zip(chunks, offsets)
        ):
            total_chunks += 1
            length = len(chunk.content)
            chunk_lengths.append(length)

            if length < chunker.chunk_size * 0.5 and len(source) > chunker.chunk_size:
                short_chunks.append(
                    (document_index + 1, index, length)
                )

            if start == -1:
                unlocated_chunks += 1
                print(
                    f"  Chunk {index}: could not locate in source "
                    f"| Length: {length}"
                )
                continue

            # A split is mid-word only when both adjacent source
            # characters are word characters.
            if start > 0 and is_word_char(source[start - 1]) and is_word_char(source[start]):
                mid_word_starts += 1

            if end < len(source) and is_word_char(source[end - 1]) and is_word_char(source[end]):
                mid_word_ends += 1

            if index > 0:
                previous_start, previous_end = offsets[index - 1]
                if previous_start != -1:
                    overlap = max(0, previous_end - start)
                    overlaps.append(overlap)

            print(
                f"  Chunk {index} | Length: {length} "
                f"| Start: {start} | End: {end}"
            )
            print(f"    Preview: {chunk.content[:100]!r}")

    average_overlap = (
        sum(overlaps) / len(overlaps) if overlaps else 0
    )

    print(f"\n{name} diagnostics")
    print(f"Total chunks: {total_chunks}")
    print(f"Actual mid-word starts: {mid_word_starts}")
    print(f"Actual mid-word endings: {mid_word_ends}")
    print(f"Unlocated chunks: {unlocated_chunks}")
    print(f"Average overlap: {average_overlap:.1f} characters")
    print(f"Target overlap: {chunk_overlap}")
    print(f"Short chunks (<50% of target): {len(short_chunks)}")
    print(
        "Chunk length range: "
        f"{min(chunk_lengths, default=0)}–"
        f"{max(chunk_lengths, default=0)}"
    )

    if short_chunks:
        print("Short chunk details:")
        for doc_index, chunk_index, length in short_chunks:
            print(
                f"  Document {doc_index}, chunk {chunk_index}: "
                f"{length} characters"
            )

    return {
        "chunks": total_chunks,
        "mid_word_starts": mid_word_starts,
        "mid_word_ends": mid_word_ends,
        "unlocated": unlocated_chunks,
        "average_overlap": average_overlap,
        "short_chunks": len(short_chunks),
    }


def main():
    pdf_path = settings.documents_dir / "demo_sample.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"Could not find PDF: {pdf_path}")

    print(f"Loading: {pdf_path}")

    loader = DocumentLoader()
    documents = loader.load_file(pdf_path)

    preprocessor = TextPreprocessor()
    processed_documents = preprocessor.preprocess_documents(
        documents
    )

    print(f"Loaded sections: {len(documents)}")
    print(f"Processed sections: {len(processed_documents)}")

    baseline = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    experimental = BoundaryAwareTextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    baseline_results = inspect_chunks(
        "BASELINE",
        baseline,
        processed_documents,
        settings.chunk_overlap,
    )

    experimental_results = inspect_chunks(
        "EXPERIMENTAL",
        experimental,
        processed_documents,
        settings.chunk_overlap,
        use_metadata=True,
    )

    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print("=" * 70)

    for key in baseline_results:
        print(
            f"{key}: baseline={baseline_results[key]} "
            f"| experimental={experimental_results[key]}"
        )


if __name__ == "__main__":
    main()
