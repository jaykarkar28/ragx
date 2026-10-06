from __future__ import annotations

import time
from typing import Any

from app.config import settings
from app.generation.answer import RAGAnswerGenerator
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import EmbeddingModel
from app.indexing.vector_store import VectorStore
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.dense import DenseRetriever
from app.retrieval.keyword import KeywordRetriever


class RAGPipeline:
    """Reusable production RAG pipeline."""

    def __init__(self) -> None:
        vector_store = VectorStore.load(
            settings.vector_store_path
        )

        bm25_index = BM25Index.load(
            settings.bm25_index_path
        )

        if vector_store.size == 0:
            raise RuntimeError("FAISS vector store is empty.")

        if bm25_index.size == 0:
            raise RuntimeError("BM25 index is empty.")

        if vector_store.size != bm25_index.size:
            raise RuntimeError(
                "FAISS and BM25 index sizes do not match: "
                f"FAISS={vector_store.size}, "
                f"BM25={bm25_index.size}"
            )

        embedding_model = EmbeddingModel()

        dense_retriever = DenseRetriever(
            embedding_model=embedding_model,
            vector_store=vector_store,
        )

        keyword_retriever = KeywordRetriever(
            bm25_index=bm25_index,
        )

        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            keyword_retriever=keyword_retriever,
        )

        self.hybrid_retriever = hybrid_retriever

        self.fusion = ReciprocalRankFusion(
            k=settings.rrf_k
        )

        self.reranker = None

        if settings.reranker_enabled:
            self.reranker = CrossEncoderReranker(
                model_name=settings.reranker_model,
            )

        self.answer_generator = RAGAnswerGenerator()

    def retrieve(
        self,
        query: str,
    ) -> tuple[list[dict[str, Any]], float]:
        """Retrieve and rerank documents for a query."""

        retrieval_start = time.perf_counter()

        candidate_k = max(
            20,
            settings.dense_top_k,
            settings.keyword_top_k,
            settings.hybrid_top_k,
            settings.reranker_top_k,
        )

        candidates = self.hybrid_retriever.retrieve_candidates(
            query=query,
            dense_top_k=candidate_k,
            keyword_top_k=candidate_k,
        )

        fused_results = self.fusion.fuse(
            candidates,
            top_k=candidate_k,
        )

        if self.reranker is not None:
            documents = self.reranker.rerank(
                query=query,
                documents=fused_results,
                top_k=settings.generation_top_k,
            )
        else:
            documents = fused_results[:settings.generation_top_k]

        retrieval_latency_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        return documents, retrieval_latency_ms

    def generate(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], float]:
        """Generate a grounded answer from retrieved documents."""

        generation_start = time.perf_counter()

        generated = self.answer_generator.generate(
            query=query,
            documents=documents,
        )

        generation_latency_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        return generated, generation_latency_ms

    def query(
        self,
        query: str,
    ) -> dict[str, Any]:
        """Run retrieval, reranking, and answer generation."""

        total_start = time.perf_counter()

        documents, retrieval_latency_ms = self.retrieve(query)

        generated, generation_latency_ms = self.generate(
            query=query,
            documents=documents,
        )

        total_latency_ms = (
            time.perf_counter() - total_start
        ) * 1000

        return {
            "query": generated["query"],
            "answer": generated["answer"],
            "source_ids": generated["source_ids"],
            "sources": documents,
            "model": generated["model"],
            "latency": {
                "retrieval_ms": round(retrieval_latency_ms, 2),
                "generation_ms": round(generation_latency_ms, 2),
                "total_ms": round(total_latency_ms, 2),
            },
        }