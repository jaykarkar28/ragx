import json
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline import RAGPipeline


pipeline = RAGPipeline()

with open("datasets/failure_cases.jsonl", encoding="utf-8") as file:
    cases = [json.loads(line) for line in file if line.strip()]

for i, case in enumerate(cases, start=1):
    result = pipeline.query(case["query"])

    print(f"\n[{i}] {case['query']}")
    print(f"Expected: {case['expected_behavior']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['source_ids']}")