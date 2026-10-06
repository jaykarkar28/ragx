# Chunking Strategy Decision

## Decision

**Selected strategy:** Baseline chunking

## Experiment

Compared:
- Baseline chunking
- Boundary-aware chunking

Both strategies were evaluated using the same 12-query ground-truth dataset and the same retrieval pipeline:
- Dense retrieval
- Keyword/BM25 retrieval
- Reciprocal Rank Fusion
- Cross-encoder reranking

## Results

| K | Metric | Baseline | Boundary-aware | Winner |
|---|---|---:|---:|---|
| 1 | MAP | 0.7246 | 0.7084 | Baseline |
| 1 | Recall | 0.5347 | 0.5347 | Tie |
| 1 | NDCG | 0.9167 | 0.9167 | Tie |
| 3 | MAP | 0.7246 | 0.7084 | Baseline |
| 3 | Recall | 0.6944 | 0.6667 | Baseline |
| 3 | NDCG | 0.7494 | 0.7247 | Baseline |
| 5 | MAP | 0.7246 | 0.7084 | Baseline |
| 5 | Recall | 0.7153 | 0.7153 | Tie |
| 5 | NDCG | 0.7513 | 0.7417 | Baseline |
| 10 | MAP | 0.7246 | 0.7084 | Baseline |
| 10 | Recall | 0.7778 | 0.8194 | Boundary-aware |
| 10 | NDCG | 0.7810 | 0.7851 | Boundary-aware |

## Rationale

Baseline chunking was selected because it provides stronger overall ranking quality across the evaluated retrieval depths.

Boundary-aware chunking showed an advantage at K=10 for Recall and NDCG, but this advantage was limited and did not outweigh the baseline's stronger MAP performance and its stronger results at K=3 and K=5.

At K=5, both strategies achieved the same Recall (0.7153), while baseline achieved better MAP and NDCG.

## Final Status

Chunking strategy evaluation: **COMPLETE**

Production/default strategy: **Baseline**

Next phase: **Generation and answer-quality evaluation**
