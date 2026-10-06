import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RAGPipeline


pipeline = RAGPipeline()

queries = [
    "What is RAG?",
    "What is the purpose of reranking in RAG?",
    "What is an embedding?",
    "What is hybrid retrieval?",
    "What is the difference between ML and DL?",
]

for query in queries:
    result = pipeline.query(query)
    latency = result["latency"]

    print(f"\nQuery: {query}")
    print(f"Retrieval:  {latency['retrieval_ms']:.2f} ms")
    print(f"Generation: {latency['generation_ms']:.2f} ms")
    print(f"Total:      {latency['total_ms']:.2f} ms")