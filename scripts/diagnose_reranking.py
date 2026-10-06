
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.evaluation.dataset import EvaluationDataset
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import EmbeddingModel
from app.indexing.vector_store import VectorStore
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.keyword import KeywordRetriever


TARGET_QUERY = "Does RAG eliminate hallucinations?"

POOL_SIZES = (10, 20)

RELEVANT_IDS = {"23", "46"}

INSPECT_IDS = {"22", "23", "35", "46"}


def get_document_id(document: dict[str, Any]) -> str:
    """Return a document ID as a string."""
    return str(document.get("id"))


def print_fused_candidates(
    documents: list[dict[str, Any]],
    relevant_ids: set[str],
) -> None:
    """Print fused candidates and relevance information."""
    print("\n--- FUSED CANDIDATES ---")

    for rank, document in enumerate(documents, start=1):
        chunk_id = get_document_id(document)
        relevant = chunk_id in relevant_ids

        print(
            f"Fused rank={rank:2} | "
            f"ID={chunk_id:>12} | "
            f"Relevant={str(relevant):5} | "
            f"Score={document.get('fused_score')}"
        )


def print_reranked_results(
    documents: list[dict[str, Any]],
    relevant_ids: set[str],
    fused_documents: list[dict[str, Any]],
) -> None:
    """Print reranked results with original fused positions."""
    print("\n--- RERANKED TOP 10 ---")

    fused_ranks = {
        get_document_id(document): rank
        for rank, document in enumerate(fused_documents, start=1)
    }

    for rank, document in enumerate(documents[:10], start=1):
        chunk_id = get_document_id(document)
        relevant = chunk_id in relevant_ids
        score = document.get("rerank_score")
        fused_rank = fused_ranks.get(chunk_id, "N/A")

        print(
            f"Rerank rank={rank:2} | "
            f"ID={chunk_id:>12} | "
            f"Relevant={str(relevant):5} | "
            f"Score={score:.6f} | "
            f"Fused rank={fused_rank}"
        )

        text = document.get("text", "")
        print(f"  {text[:350].replace(chr(10), ' ')}")


def print_relevant_movement(
    fused_documents: list[dict[str, Any]],
    reranked_documents: list[dict[str, Any]],
    relevant_ids: set[str],
) -> None:
    """Show how relevant documents move after reranking."""
    fused_ranks = {
        get_document_id(document): rank
        for rank, document in enumerate(fused_documents, start=1)
    }

    reranked_ranks = {
        get_document_id(document): rank
        for rank, document in enumerate(reranked_documents, start=1)
    }

    print("\n--- RELEVANT DOCUMENT MOVEMENT ---")

    for chunk_id in sorted(relevant_ids):
        print(
            f"ID={chunk_id}: "
            f"Fused rank={fused_ranks.get(chunk_id, 'Not retrieved')}, "
            f"Reranked rank={reranked_ranks.get(chunk_id, 'Not retrieved')}"
        )


def print_top5_summary(
    fused_documents: list[dict[str, Any]],
    reranked_documents: list[dict[str, Any]],
    relevant_ids: set[str],
) -> None:
    """Summarize relevant chunks in the top five."""
    fused_top5_relevant = sum(
        get_document_id(document) in relevant_ids
        for document in fused_documents[:5]
    )

    reranked_top5_relevant = sum(
        get_document_id(document) in relevant_ids
        for document in reranked_documents[:5]
    )

    print("\n--- TOP 5 SUMMARY ---")
    print(f"Relevant chunks in fused top 5: {fused_top5_relevant}")
    print(f"Relevant chunks in reranked top 5: {reranked_top5_relevant}")


def print_full_chunk_text(
    documents: list[dict[str, Any]],
    inspect_ids: set[str],
) -> None:
    """Print the complete text of selected chunks."""
    documents_by_id = {
        get_document_id(document): document
        for document in documents
    }

    print("\n" + "=" * 100)
    print("FULL CHUNK TEXT INSPECTION")
    print("=" * 100)

    for chunk_id in sorted(inspect_ids, key=int):
        document = documents_by_id.get(chunk_id)

        print("\n" + "=" * 100)
        print(f"CHUNK ID: {chunk_id}")
        print("=" * 100)

        if document is None:
            print("Chunk not found in the candidate pool.")
            continue

        print(f"FUSED SCORE: {document.get('fused_score', 'N/A')}")
        print(f"RERANK SCORE: {document.get('rerank_score', 'N/A')}")
        print("\nFULL TEXT:\n")
        print(document.get("text", "<No text found>"))


def run_context_experiment(
    query: str,
    fused_documents: list[dict[str, Any]],
    reranker: CrossEncoderReranker,
) -> None:
    """Compare original chunks with a context-augmented chunk."""
    documents_by_id = {
        get_document_id(document): document
        for document in fused_documents
    }

    required_ids = {"22", "23", "35", "46"}
    missing_ids = required_ids - documents_by_id.keys()

    print("\n" + "=" * 100)
    print("CONTROLLED CONTEXT EXPERIMENT")
    print("=" * 100)

    if missing_ids:
        print(
            "Cannot run the context experiment. "
            f"Missing chunks: {sorted(missing_ids)}"
        )
        return

    chunk22 = documents_by_id["22"]
    chunk23 = documents_by_id["23"]
    chunk35 = documents_by_id["35"]
    chunk46 = documents_by_id["46"]

    experiments: list[dict[str, Any]] = []

    variants = [
        ("23_original", "Chunk 23 - original", chunk23),
        ("35_competitor", "Chunk 35 - competitor", chunk35),
        ("46_direct_answer", "Chunk 46 - direct answer", chunk46),
    ]

    for variant_id, label, document in variants:
        experiment_document = document.copy()
        experiment_document["id"] = variant_id
        experiment_document["experiment_label"] = label
        experiments.append(experiment_document)

    augmented = chunk23.copy()
    augmented["id"] = "23_augmented"
    augmented["text"] = (
        chunk22.get("text", "").rstrip()
        + "\n"
        + chunk23.get("text", "").lstrip()
    )
    augmented["experiment_label"] = "Chunk 22 + 23 - augmented"
    experiments.append(augmented)

    print(f"Query: {query}")
    print("Scoring all variants with the same cross-encoder.")

    results = reranker.rerank(
        query=query,
        documents=experiments,
        top_k=len(experiments),
    )

    print("\n--- SCORE COMPARISON ---")

    for rank, result in enumerate(results, start=1):
        print(
            f"Rank={rank} | "
            f"Variant={result.get('experiment_label')} | "
            f"Score={result.get('rerank_score'):.6f}"
        )

    print("\n--- EXPERIMENT TEXTS ---")

    for document in experiments:
        print("\n" + "-" * 80)
        print(document["experiment_label"])
        print("-" * 80)
        print(document.get("text", "<No text found>"))


def run_diagnostic(
    query: str,
    pool_size: int,
    dense_retriever: DenseRetriever,
    keyword_retriever: KeywordRetriever,
    fusion: ReciprocalRankFusion,
    reranker: CrossEncoderReranker,
    relevant_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run retrieval, fusion, and reranking for one pool size."""
    print("\n" + "=" * 100)
    print(f"CANDIDATE POOL: {pool_size}")
    print("=" * 100)

    dense_results = dense_retriever.retrieve(
        query=query,
        top_k=pool_size,
    )

    keyword_results = keyword_retriever.retrieve(
        query=query,
        top_k=pool_size,
    )

    candidates = {
        "dense": dense_results,
        "keyword": keyword_results,
    }

    fused_documents = fusion.fuse(
        candidates,
        top_k=pool_size,
    )

    reranked_documents = reranker.rerank(
        query=query,
        documents=fused_documents,
        top_k=pool_size,
    )

    print_fused_candidates(fused_documents, relevant_ids)

    print_reranked_results(
        reranked_documents,
        relevant_ids,
        fused_documents,
    )

    print_relevant_movement(
        fused_documents,
        reranked_documents,
        relevant_ids,
    )

    print_top5_summary(
        fused_documents,
        reranked_documents,
        relevant_ids,
    )

    return fused_documents, reranked_documents


def main() -> None:
    """Run the full reranking diagnostic."""
    print("Loading indexes and evaluation data...")

    vector_store = VectorStore.load(settings.vector_store_path)
    bm25_index = BM25Index.load(settings.bm25_index_path)

    dataset = EvaluationDataset.from_jsonl(
        settings.evaluation_dataset_path
    )

    record = next(
        (item for item in dataset if item.query == TARGET_QUERY),
        None,
    )

    if record is None:
        raise ValueError(
            f"Query not found in evaluation dataset: {TARGET_QUERY}"
        )

    # Known ground-truth IDs for this specific diagnostic query.
    relevant_ids = RELEVANT_IDS

    embedding_model = EmbeddingModel()

    dense_retriever = DenseRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
    )

    keyword_retriever = KeywordRetriever(
        bm25_index=bm25_index,
    )

    fusion = ReciprocalRankFusion(
        k=settings.rrf_k,
    )

    reranker = CrossEncoderReranker(
        model_name=settings.reranker_model,
    )

    print("\n" + "=" * 100)
    print(f"QUERY: {TARGET_QUERY}")
    print(f"RELEVANT IDS: {sorted(relevant_ids)}")
    print("=" * 100)

    pool20_results = None

    for pool_size in POOL_SIZES:
        fused_documents, reranked_documents = run_diagnostic(
            query=TARGET_QUERY,
            pool_size=pool_size,
            dense_retriever=dense_retriever,
            keyword_retriever=keyword_retriever,
            fusion=fusion,
            reranker=reranker,
            relevant_ids=relevant_ids,
        )

        if pool_size == 20:
            pool20_results = (
                fused_documents,
                reranked_documents,
            )

    if pool20_results is not None:
        fused_documents, reranked_documents = pool20_results

        # Print complete text of chunks 22, 23, 35, and 46.
        print_full_chunk_text(
            fused_documents,
            INSPECT_IDS,
        )

        # Compare original and context-augmented text variants.
        run_context_experiment(
            query=TARGET_QUERY,
            fused_documents=fused_documents,
            reranker=reranker,
        )

    print("\n" + "=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
