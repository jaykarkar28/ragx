# RAGX — Evaluation-Driven RAG System

RAGX is an evaluation-driven Retrieval-Augmented Generation (RAG) system designed to investigate, benchmark, and improve retrieval quality before generating answers.

The system combines dense retrieval, sparse keyword retrieval, hybrid retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, grounded answer generation, and automated evaluation.

## Key Features

- Document ingestion and preprocessing
- Configurable document chunking
- Semantic retrieval using Sentence Transformers
- Sparse retrieval using BM25
- Hybrid retrieval combining dense and keyword search
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Grounded RAG answer generation
- Reference-answer evaluation
- Faithfulness and answer-relevance evaluation
- Retrieval benchmark using Precision, Recall, MRR, MAP, and NDCG
- Chunking strategy comparison
- Failure and hallucination testing
- FastAPI production API
- JSON and CSV experiment reports

## Architecture

```text
                         ┌─────────────────────┐
                         │    Documents        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Ingestion &          │
                         │ Preprocessing        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     Chunking        │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       ▼                         ▼
              ┌────────────────┐        ┌────────────────┐
              │ Dense Embedding│        │   BM25 Index   │
              │    + FAISS     │        │                │
              └───────┬────────┘        └───────┬────────┘
                      │                         │
                      └────────────┬────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │  Hybrid Retrieval   │
                         │      + RRF           │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Encoder       │
                         │ Reranking           │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Retrieved Context   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Groq LLM Generation │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Grounded Answer     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │ Evaluation & Benchmarking   │
                    └─────────────────────────────┘
```

## Technology Stack

### Retrieval
- FAISS
- BM25
- Sentence Transformers
- Reciprocal Rank Fusion
- Cross-Encoder

### Generation
- Groq
- `openai/gpt-oss-120b`
- LangChain

### Backend
- Python
- FastAPI
- Pydantic

### Evaluation
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- NDCG@K
- Faithfulness
- Answer Relevance
- Grounded Rate
- Reference Answer Score

## Project Structure

```text
ragx/
├── app/
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── ingestion/
│   │   ├── loaders.py
│   │   ├── preprocessing.py
│   │   └── chunking.py
│   ├── indexing/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── bm25_index.py
│   ├── retrieval/
│   │   ├── dense.py
│   │   ├── keyword.py
│   │   ├── hybrid.py
│   │   └── fusion.py
│   ├── reranking/
│   │   └── cross_encoder.py
│   ├── generation/
│   │   ├── prompts.py
│   │   └── answer.py
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── answer_metrics.py
│   │   ├── dataset.py
│   │   └── runner.py
│   ├── config.py
│   └── pipeline.py
│
├── datasets/
│   ├── documents/
│   ├── ground_truth.jsonl
│   ├── answer_ground_truth.jsonl
│   └── failure_cases.jsonl
│
├── experiments/
│   ├── configs/
│   ├── reports/
│   ├── vector_store/
│   └── bm25_index/
│
├── dashboard/
├── scripts/
└── tests/
```

## Retrieval Evaluation

RAGX evaluates multiple retrieval stages across K = 1, 3, 5, and 10.

### Reranked Retrieval Results

| K | MAP | MRR | NDCG | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| 1 | 0.8444 | 0.9583 | 0.9167 | 0.9167 | 0.6944 |
| 3 | 0.8444 | 0.9583 | 0.8530 | 0.3889 | 0.8472 |
| 5 | 0.8444 | 0.9583 | 0.8530 | 0.2333 | 0.8472 |
| 10 | 0.8444 | 0.9583 | 0.8894 | 0.1333 | 0.9306 |

The reranking stage produced the strongest MAP and MRR among the evaluated retrieval stages.

## Chunking Experiment

Two chunking strategies were evaluated:

1. Baseline chunking
2. Boundary-aware chunking

The baseline strategy was selected as the production/default strategy because it produced stronger overall MAP and ranking quality across the evaluated K values.

Boundary-aware chunking showed an advantage for high-K recall in specific cases, but this was not sufficient to replace the baseline strategy as the default.

## Answer Evaluation

The final answer-generation evaluation achieved:

```text
Faithfulness:      1.0000
Answer Relevance:  1.0000
Grounded Rate:     1.0000
Reference Score:   0.9583
```

The evaluation uses retrieved context as the grounding source and also compares generated answers against reference answers for semantic correctness and coverage.

## Failure & Hallucination Testing

Five targeted failure cases were tested.

| Test | Expected | Result |
|---|---|---|
| Reranking question | Answer | PASS |
| RAG hallucination question | Answer | PASS |
| Unrelated factual question | Refuse | PASS |
| Chunking question | Answer | PASS |
| Unrelated technical question | Refuse | PASS |

The system correctly refused to answer questions when the retrieved context did not contain sufficient information.

## Example

### Query

```text
What is the purpose of reranking in RAG?
```

### Retrieved Sources

```text
[46, 45, 34, 47, 22]
```

### Generated Answer

```text
Reranking scores the retrieved candidates and reorders them so they more
accurately match the specific query before the generation step.
```

## API

The system exposes a FastAPI interface.

### Health Check

```text
GET /api/health
```

### Query

```text
POST /api/query
```

Example request:

```json
{
  "query": "What is RAG?"
}
```

Example response:

```json
{
  "query": "What is RAG?",
  "answer": "...",
  "source_ids": [40, 22, 21, 41, 34],
  "sources": [],
  "model": "openai/gpt-oss-120b"
}
```

## Configuration

Configuration is managed through environment variables and `app/config.py`.

Example:

```env
GROQ_API_KEY=your_api_key
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=1024
```

## Running the Project

Activate the virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

Run the API:

```powershell
uvicorn app.api.routes:app --reload
```

Run retrieval experiments:

```powershell
python scripts/run_experiments.py
```

Run failure-case evaluation:

```powershell
python scripts/run_failure_cases.py
```

## Evaluation Philosophy

RAGX follows an evaluation-first approach:

```text
Build Retrieval
      ↓
Measure Retrieval
      ↓
Compare Strategies
      ↓
Select Production Strategy
      ↓
Generate Answers
      ↓
Evaluate Answers
      ↓
Test Failure Cases
      ↓
Iterate
```

The goal is not simply to build a RAG chatbot, but to understand **where retrieval quality improves or fails and how those retrieval decisions affect generated answers**.

## Performance & Observability

RAGX includes basic latency instrumentation for retrieval, generation, and total request execution.

### Warm-Query Latency Benchmark

A five-query benchmark was executed using the production RAG pipeline.

| Metric | Approx. Average |
|---|---:|
| Retrieval latency | ~0.70 s |
| Generation latency | ~0.76 s |
| Total latency | ~1.47 s |

The first query experienced cold-start overhead because the retrieval models were being initialized. Subsequent queries stabilized around 1.4–1.5 seconds per request.

### Observability Metrics

Each pipeline query now returns:

```json
{
  "latency": {
    "retrieval_ms": 700.0,
    "generation_ms": 760.0,
    "total_ms": 1470.0
  }
}

## Current Status

### Completed

- [x] Document ingestion
- [x] Preprocessing
- [x] Chunking
- [x] Embedding generation
- [x] FAISS indexing
- [x] BM25 indexing
- [x] Dense retrieval
- [x] Keyword retrieval
- [x] Hybrid retrieval
- [x] RRF fusion
- [x] Cross-encoder reranking
- [x] Retrieval evaluation
- [x] Chunking comparison
- [x] Generation layer
- [x] Answer evaluation
- [x] Reference-answer evaluation
- [x] Failure/hallucination testing
- [x] FastAPI API
- [x] Production RAG pipeline
- [x] Experiment reporting

### Future Improvements

- Latency and performance benchmarking
- Detailed observability
- Automated regression testing
- Interactive evaluation dashboard
- Larger failure-case dataset
- Production deployment
- Continuous evaluation pipeline

## Key Takeaway

RAGX demonstrates an evaluation-driven approach to building production-oriented RAG systems, combining retrieval experimentation, reranking, grounded generation, quantitative evaluation, and failure-case testing rather than treating RAG as only a document-search chatbot.