
from __future__ import annotations

import logging

from app.config import settings
from app.ingestion.chunking import TextChunker
from app.ingestion.loaders import DocumentLoader
from app.ingestion.preprocessing import TextPreprocessor
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import EmbeddingModel
from app.indexing.vector_store import VectorStore
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def ingest_documents() -> None:
    """Load, preprocess, chunk, index, and save documents."""
    logger.info("Starting document ingestion")

    # 1. Load documents
    loader = DocumentLoader()
    documents = loader.load_directory(settings.documents_dir)

    if not documents:
        raise ValueError(
            f"No supported documents found in {settings.documents_dir}"
        )

    logger.info("Loaded %d document sections", len(documents))

    # 2. Preprocess documents
    preprocessor = TextPreprocessor()
    processed_documents = preprocessor.preprocess_documents(documents)

    if not processed_documents:
        raise ValueError("No text remained after preprocessing")

    logger.info("Preprocessed %d sections", len(processed_documents))

    # 3. Split documents into chunks
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = chunker.chunk_documents(processed_documents)

    if not chunks:
        raise ValueError("No chunks were generated")

    # Assign a stable, human-readable ID to each chunk.
    for index, chunk in enumerate(chunks):
        filename = chunk.metadata.get("filename", "document")
        page = chunk.metadata.get("page", 0)
        chunk_index = chunk.metadata.get("chunk_index", index)

        chunk.metadata["chunk_id"] = (
            f"{filename}_p{page}_c{chunk_index}"
        )

    logger.info("Generated %d chunks", len(chunks))

    # 4. Generate embeddings
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_documents(chunks)

    logger.info("Generated embeddings with shape %s", embeddings.shape)

    # 5. Build the FAISS vector store
    vector_store = VectorStore(
        dimension=embedding_model.embedding_dimension
    )

    vector_ids = vector_store.add_documents(
        texts=[chunk.content for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )

    # 6. Build the BM25 index
    bm25_index = BM25Index()

    bm25_ids = bm25_index.add_documents(
        texts=[chunk.content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
    )

    if vector_ids != bm25_ids:
        raise RuntimeError("Dense and BM25 document IDs do not match")

    # 7. Save both indexes
    vector_store.save(settings.vector_store_path)
    bm25_index.save(settings.bm25_index_path)

    logger.info("FAISS index saved to %s", settings.vector_store_path)
    logger.info("BM25 index saved to %s", settings.bm25_index_path)

    # 8. Print an ID mapping for evaluation dataset preparation
    print("\nIngestion completed successfully.")
    print(f"Documents/sections: {len(processed_documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")

    print("\nChunk ID mapping:")
    for index, chunk in enumerate(chunks):
        print(
            f"Index ID: {index} | "
            f"Chunk ID: {chunk.metadata['chunk_id']} | "
            f"Source: {chunk.metadata.get('filename')} | "
            f"Page: {chunk.metadata.get('page', 'N/A')} | "
            f"Chunk: {chunk.metadata.get('chunk_index')}"
        )

    print("\nPreview of first chunk:")
    print(chunks[0].content[:500])


if __name__ == "__main__":
    ingest_documents()
