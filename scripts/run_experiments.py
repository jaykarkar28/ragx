from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure the project root is available for imports.
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.evaluation.dataset import EvaluationDataset
from app.evaluation.runner import EvaluationRunner
from app.indexing.bm25_index import BM25Index
from app.indexing.embeddings import EmbeddingModel
from app.indexing.vector_store import VectorStore
from app.reranking.cross_encoder import CrossEncoderReranker
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.keyword import KeywordRetriever


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def validate_indexes(
    vector_store: VectorStore,
    bm25_index: BM25Index,
) -> None:
    """Check that the two indexes contain the same number of chunks."""
    if vector_store.size == 0 or bm25_index.size == 0:
        raise ValueError("One or both indexes are empty. Run ingestion first.")

    if vector_store.size != bm25_index.size:
        raise ValueError(
            "FAISS and BM25 index sizes do not match: "
            f"FAISS={vector_store.size}, BM25={bm25_index.size}"
        )

    logger.info(
        "Loaded matching indexes containing %d chunks",
        vector_store.size,
    )


def validate_ground_truth(
    dataset: EvaluationDataset,
    index_size: int,
) -> None:
    """Warn if ground-truth IDs do not match positional index IDs."""
    valid_ids = {str(index) for index in range(index_size)}

    unknown_ids = sorted({
        doc_id
        for record in dataset
        for doc_id in record.relevant_docs
        if doc_id not in valid_ids
    })

    if unknown_ids:
        logger.warning(
            "Ground-truth IDs do not match positional index IDs. "
            "Unknown IDs: %s",
            unknown_ids,
        )
        logger.warning(
            "Evaluation may produce misleading or zero scores. "
            "Use positional IDs such as '0', '1', '2', etc."
        )
    else:
        logger.info("Ground-truth IDs match the index ID range")


def save_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the complete experiment report as JSON."""
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    logger.info("JSON report saved to %s", output_path)


def save_csv_summary(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Save average retrieval metrics for every stage and K value as CSV."""
    rows = []

    for run in report["retrieval_runs"]:
        k = run["k"]

        for stage, metrics in run["average_metrics"].items():
            # Do not include answer evaluation in the retrieval CSV.
            if stage == "answer_evaluation":
                continue

            for metric, value in metrics.items():
                rows.append({
                    "k": k,
                    "stage": stage,
                    "metric": metric,
                    "value": value,
                })

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "k",
                "stage",
                "metric",
                "value",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    logger.info("CSV summary saved to %s", output_path)


def print_summary(report: dict[str, Any]) -> None:
    """Print a readable comparison of average retrieval metrics."""

    print("\n" + "=" * 72)
    print("RAGX RETRIEVAL EXPERIMENT SUMMARY")
    print("=" * 72)

    for run in report["retrieval_runs"]:
        print(f"\nMetrics at K={run['k']}")
        print("-" * 72)

        metrics = run["average_metrics"]

        metric_names = sorted({
            name
            for stage_name, stage_metrics in metrics.items()
            if stage_name != "answer_evaluation"
            for name in stage_metrics
        })

        header = f"{'Stage':<14}" + "".join(
            f"{name:>14}"
            for name in metric_names
        )

        print(header)
        print("-" * len(header))

        for stage, values in metrics.items():
            if stage == "answer_evaluation":
                continue

            line = f"{stage:<14}" + "".join(
                f"{values.get(name, 0.0):>14.4f}"
                for name in metric_names
            )

            print(line)

    print("\n" + "=" * 72)

    if report.get("answer_evaluation") is not None:
        answer_eval = report["answer_evaluation"]

        print("\nANSWER EVALUATION")
        print("-" * 72)
        print(
            f"Faithfulness:     "
            f"{answer_eval['faithfulness']:.4f}"
        )
        print(
            f"Answer Relevance: "
            f"{answer_eval['answer_relevance']:.4f}"
        )
        print(
            f"Grounded Rate:    "
            f"{answer_eval['grounded_rate']:.4f}"
        )

    print("=" * 72)


def run_experiments() -> dict[str, Any]:
    """Run retrieval evaluation and one final answer evaluation."""

    logger.info("Starting RAGX retrieval experiments")

    # ------------------------------------------------------------------
    # 1. Load indexes
    # ------------------------------------------------------------------

    vector_store = VectorStore.load(
        settings.vector_store_path
    )

    bm25_index = BM25Index.load(
        settings.bm25_index_path
    )

    validate_indexes(
        vector_store,
        bm25_index,
    )

    # ------------------------------------------------------------------
    # 2. Load evaluation dataset
    # ------------------------------------------------------------------

    dataset = EvaluationDataset.from_jsonl(
        settings.evaluation_dataset_path
    )

    if len(dataset) == 0:
        raise ValueError("Evaluation dataset is empty.")

    logger.info(
        "Loaded %d evaluation queries",
        len(dataset),
    )

    validate_ground_truth(
        dataset,
        vector_store.size,
    )

    # ------------------------------------------------------------------
    # 3. Initialize retrieval components
    # ------------------------------------------------------------------

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

    fusion = ReciprocalRankFusion(
        k=settings.rrf_k
    )

    reranker = None

    if settings.reranker_enabled:
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
        )

    candidate_k = max(
        20,
        settings.dense_top_k,
        settings.keyword_top_k,
        settings.hybrid_top_k,
        settings.reranker_top_k,
        max(settings.evaluation_k_values),
    )

    # ------------------------------------------------------------------
    # 4. Retrieval evaluation at each K
    # ------------------------------------------------------------------

    retrieval_runs = []

    for k in settings.evaluation_k_values:
        logger.info(
            "Evaluating retrieval metrics at K=%d",
            k,
        )

        runner = EvaluationRunner(
            dataset=dataset,
            hybrid_retriever=hybrid_retriever,
            fusion=fusion,
            reranker=reranker,
            k=k,
            candidate_k=candidate_k,
            evaluate_answers=False,
        )

        result = runner.run()

        # Keep only retrieval information.
        retrieval_runs.append({
            "k": k,
            "queries_evaluated": result["queries_evaluated"],
            "candidate_k": result["candidate_k"],
            "average_metrics": {
                stage: metrics
                for stage, metrics
                in result["average_metrics"].items()
                if stage != "answer_evaluation"
            },
            "per_query": [
                {
                    "query": query_result["query"],
                    "stages": query_result["stages"],
                }
                for query_result in result["per_query"]
            ],
        })

    # ------------------------------------------------------------------\
    # 5. Answer evaluation
    #
    # Use the dedicated reference-answer dataset.
    # ------------------------------------------------------------------

    answer_dataset = EvaluationDataset.from_jsonl(
        settings.answer_evaluation_dataset_path
    )

    if len(answer_dataset) == 0:
        raise ValueError("Answer evaluation dataset is empty.")

    logger.info(
        "Loaded %d answer-evaluation queries",
        len(answer_dataset),
    )

    validate_ground_truth(
        answer_dataset,
        vector_store.size,
    )

    production_k = settings.generation_top_k

    logger.info(
        "Running final answer generation/evaluation at K=%d",
        production_k,
    )

    answer_runner = EvaluationRunner(
        dataset=answer_dataset,
        hybrid_retriever=hybrid_retriever,
        fusion=fusion,
        reranker=reranker,
        k=production_k,
        candidate_k=candidate_k,
        evaluate_answers=True,
    )

    answer_result = answer_runner.run()

    answer_evaluation = (
        answer_result["average_metrics"]
        .get("answer_evaluation")
    )

    # ------------------------------------------------------------------
    # 6. Build final report
    # ------------------------------------------------------------------

    timestamp = datetime.now().astimezone().isoformat()

    report = {
        "project": "RAGX",
        "experiment_timestamp": timestamp,
        "dataset_path": str(
            settings.evaluation_dataset_path
        ),
        "vector_store_path": str(
            settings.vector_store_path
        ),
        "bm25_index_path": str(
            settings.bm25_index_path
        ),
        "index_size": vector_store.size,
        "queries_evaluated": len(dataset),
        "candidate_k": candidate_k,
        "reranker_enabled": reranker is not None,
        "production_generation_k": production_k,
        "retrieval_runs": retrieval_runs,
        "answer_evaluation": answer_evaluation,
        "answer_per_query": answer_result["per_query"],
    }

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------

    settings.experiment_report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename_timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = (
        settings.experiment_report_dir
        / f"retrieval_experiment_{filename_timestamp}.json"
    )

    csv_path = (
        settings.experiment_report_dir
        / f"retrieval_summary_{filename_timestamp}.csv"
    )

    save_json_report(
        report,
        json_path,
    )

    save_csv_summary(
        report,
        csv_path,
    )

    print_summary(report)

    logger.info(
        "Experiment completed successfully"
    )

    return report


if __name__ == "__main__":
    run_experiments()