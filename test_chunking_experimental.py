
from app.ingestion.chunking import TextChunker
from app.ingestion.chunking_experimental import BoundaryAwareTextChunker
from app.ingestion.loaders import LoadedDocument
from app.config import settings


def inspect_chunks(name, chunker, document):
    chunks = chunker.chunk_document(document)

    print(f"\n{'=' * 70}")
    print(f"{name}: {len(chunks)} chunks")
    print("=" * 70)

    for chunk in chunks:
        print(
            f"\nIndex: {chunk.metadata['chunk_index']} | "
            f"Start: {chunk.metadata['char_start']} | "
            f"End: {chunk.metadata['char_end']}"
        )
        print(repr(chunk.content[:100]))
        print("...")
        print(repr(chunk.content[-100:]))

    return chunks


def main():
    sample_text = (
        "Retrieval-Augmented Generation (RAG) can improve grounding. "
        "However, it does not eliminate hallucinations. "
        "Retrieval errors and generation errors can still occur.\n\n"
        "Evaluation is important for measuring retrieval quality. "
        "Metrics such as Recall@K, MRR, and nDCG help assess performance. "
        "Chunk boundaries can affect retrieval results."
    )

    document = LoadedDocument(
        content=sample_text * 8,
        metadata={"source": "chunking_test"},
    )

    baseline = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    experimental = BoundaryAwareTextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    inspect_chunks("Baseline", baseline, document)
    inspect_chunks("Experimental", experimental, document)


if __name__ == "__main__":
    main()
