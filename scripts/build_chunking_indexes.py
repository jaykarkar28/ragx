
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import EmbeddingModel
from app.indexing.vector_store import VectorStore


ROOT_DIR = Path(__file__).resolve().parents[1]
STRATEGY = "boundary_aware"

CHUNK_DIR = (
    ROOT_DIR
    / "experiments"
    / "chunking_comparison"
    / STRATEGY
)

OUTPUT_DIR = (
    ROOT_DIR
    / "experiments"
    / "chunking_comparison"
    / "indexes"
    / STRATEGY
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    chunks_path = CHUNK_DIR / "chunks.json"

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No chunks found.")

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk.get("metadata", {}) for chunk in chunks]

    if any(not text.strip() for text in texts):
        raise ValueError("Empty chunk text found.")

    # Protect existing experiment outputs from accidental overwrite.
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_texts(
        texts,
        show_progress_bar=True,
    )

    vector_store = VectorStore(
        dimension=embeddings.shape[1]
    )
    vector_ids = vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    vector_store.save(OUTPUT_DIR / "vector_store")

    bm25_index = BM25Index()
    bm25_ids = bm25_index.add_documents(
        texts=texts,
        metadatas=metadatas,
    )
    bm25_index.save(OUTPUT_DIR / "bm25_index")

    if vector_ids != bm25_ids:
        raise ValueError("FAISS and BM25 document IDs do not match.")

    print(f"Strategy: {STRATEGY}")
    print(f"Chunks indexed: {len(texts)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Vector index size: {vector_store.size}")
    print(f"BM25 index size: {bm25_index.size}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
