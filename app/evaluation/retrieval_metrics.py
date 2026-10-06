
from __future__ import annotations

import math
from typing import Any, Sequence


def precision_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any],
    k: int,
) -> float:
    """Calculate Precision@K."""
    if k <= 0:
        raise ValueError("k must be positive.")

    if not retrieved:
        return 0.0

    top_k = retrieved[:k]
    relevant_count = sum(item in relevant for item in top_k)

    return relevant_count / k


def recall_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any],
    k: int,
) -> float:
    """Calculate Recall@K."""
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant:
        return 0.0

    relevant_count = sum(item in relevant for item in retrieved[:k])
    return relevant_count / len(relevant)


def reciprocal_rank(
    retrieved: Sequence[Any],
    relevant: set[Any],
) -> float:
    """Calculate the reciprocal rank of the first relevant result."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank

    return 0.0


def average_precision(
    retrieved: Sequence[Any],
    relevant: set[Any],
) -> float:
    """Calculate average precision for a single query."""
    if not relevant:
        return 0.0

    score = 0.0
    relevant_seen = 0

    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            relevant_seen += 1
            score += relevant_seen / rank

    return score / len(relevant)


def ndcg_at_k(
    retrieved: Sequence[Any],
    relevant: set[Any],
    k: int,
) -> float:
    """Calculate binary-relevance NDCG@K."""
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )

    ideal_relevant_count = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg if idcg else 0.0


def evaluate_retrieval(
    retrieved: Sequence[Any],
    relevant: set[Any],
    k: int = 5,
) -> dict[str, float]:
    """Calculate a collection of retrieval metrics for one query."""
    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant),
        "map": average_precision(retrieved, relevant),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
    }
