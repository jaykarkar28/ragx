
from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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

BASE_DIR = (
    ROOT_DIR / "experiments" / "chunking_comparison"
)
INDEX_DIR = BASE_DIR / "indexes"
REPORT_DIR = BASE_DIR / "reports"

STRATEGIES = ["baseline", "boundary_aware"]


def validate_strategy_indexes(
    vector_store: VectorStore,
    bm25_index: BM25Index,
    dataset: EvaluationDataset,
) -> None:
    """Validate index sizes and ground-truth document IDs."""
    if vector_store.size == 0 or bm25_index.size == 0:
        raise ValueError("One or both strategy indexes are empty.")

    if vector_store.size != bm25_index.size:
        raise ValueError(
            "FAISS and BM25 sizes do not match: "
            f"{vector_store.size} vs {bm25_index.size}"
        )

    valid_ids = {str(i) for i in range(vector_store.size)}

    unknown_ids = sorted({
        doc_id
        for record in dataset
        for doc_id in record.relevant_docs
        if doc_id not in valid_ids
    })

    if unknown_ids:
        raise ValueError(
            f"Ground-truth IDs are outside the index range: {unknown_ids}"
        )


def evaluate_strategy(
    strategy: str,
    embedding_model: EmbeddingModel,
    reranker: CrossEncoderReranker | None,
    timestamp: str,
) -> dict[str, Any]:
    """Evaluate one chunking strategy."""
    logger.info("Evaluating strategy: %s", strategy)

    strategy_dir = BASE_DIR / strategy
    index_dir = INDEX_DIR / strategy

    vector_store = VectorStore.load(index_dir / "vector_store")
    bm25_index = BM25Index.load(index_dir / "bm25_index")

    dataset = EvaluationDataset.from_jsonl(
        strategy_dir / "ground_truth.jsonl"
    )

    validate_strategy_indexes(
        vector_store,
        bm25_index,
        dataset,
    )

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

    fusion = ReciprocalRankFusion(k=settings.rrf_k)

    candidate_k = max(
        20,
        settings.dense_top_k,
        settings.keyword_top_k,
        settings.hybrid_top_k,
        settings.reranker_top_k,
        max(settings.evaluation_k_values),
    )

    runs = []

    for k in settings.evaluation_k_values:
        runner = EvaluationRunner(
            dataset=dataset,
            hybrid_retriever=hybrid_retriever,
            fusion=fusion,
            reranker=reranker,
            k=k,
            candidate_k=candidate_k,
        )

        result = runner.run()
        runs.append({
            "k": k,
            **result,
        })

    report = {
        "project": "RAGX",
        "strategy": strategy,
        "experiment_timestamp": timestamp,
        "dataset_path": str(strategy_dir / "ground_truth.jsonl"),
        "vector_store_path": str(index_dir / "vector_store"),
        "bm25_index_path": str(index_dir / "bm25_index"),
        "index_size": vector_store.size,
        "queries_evaluated": len(dataset),
        "candidate_k": candidate_k,
        "reranker_enabled": reranker is not None,
        "runs": runs,
    }

    return report


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def build_comparison_rows(
    reports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create side-by-side average metric rows."""
    rows = []

    for strategy, report in reports.items():
        for run in report["runs"]:
            for stage, metrics in run["average_metrics"].items():
                for metric, value in metrics.items():
                    rows.append({
                        "strategy": strategy,
                        "k": run["k"],
                        "stage": stage,
                        "metric": metric,
                        "value": value,
                    })

    return rows


def save_comparison_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "strategy",
        "k",
        "stage",
        "metric",
        "value",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_comparison(
    reports: dict[str, dict[str, Any]],
) -> None:
    print("\n" + "=" * 90)
    print("RAGX CHUNKING STRATEGY COMPARISON")
    print("=" * 90)

    for strategy, report in reports.items():
        print(f"\nStrategy: {strategy}")
        print("-" * 90)

        for run in report["runs"]:
            print(f"\nK = {run['k']}")
            print(f"{'Stage':<14}", end="")

            metrics = run["average_metrics"]
            metric_names = sorted({
                metric
                for stage_metrics in metrics.values()
                for metric in stage_metrics
            })

            for metric in metric_names:
                print(f"{metric:>14}", end="")
            print()

            for stage, values in metrics.items():
                print(f"{stage:<14}", end="")
                for metric in metric_names:
                    print(f"{values.get(metric, 0.0):>14.4f}", end="")
                print()

    print("\n" + "=" * 90)


def main() -> None:
    timestamp = datetime.now().astimezone().isoformat()
    filename_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    embedding_model = EmbeddingModel()

    reranker = None
    if settings.reranker_enabled:
        reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
        )

    reports = {}

    for strategy in STRATEGIES:
        reports[strategy] = evaluate_strategy(
            strategy=strategy,
            embedding_model=embedding_model,
            reranker=reranker,
            timestamp=timestamp,
        )

    rows = build_comparison_rows(reports)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    combined_report = {
        "project": "RAGX",
        "experiment_timestamp": timestamp,
        "strategies": reports,
    }

    json_path = (
        REPORT_DIR
        / f"chunking_comparison_{filename_timestamp}.json"
    )
    csv_path = (
        REPORT_DIR
        / f"chunking_comparison_{filename_timestamp}.csv"
    )

    save_json(combined_report, json_path)
    save_comparison_csv(rows, csv_path)
    print_comparison(reports)

    logger.info("Comparison JSON saved to %s", json_path)
    logger.info("Comparison CSV saved to %s", csv_path)


if __name__ == "__main__":
    main()
