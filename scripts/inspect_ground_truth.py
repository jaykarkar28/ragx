import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from app.indexing.vector_store import VectorStore
from app.config import settings

vs = VectorStore.load(settings.vector_store_path)

with open(settings.evaluation_dataset_path, encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

for record in data:
    print("QUERY:", record["query"])
    print("DOCS:", record["relevant_docs"])

    for doc_id in record["relevant_docs"]:
        print(f"[{doc_id}] {vs.documents[int(doc_id)]['text']}")

    print("=" * 100)
