
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.indexing.vector_store import VectorStore


def main() -> None:
    vector_store = VectorStore.load(settings.vector_store_path)

    output_path = ROOT_DIR / "experiments" / "all_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = [
        {
            "id": str(index),
            "text": document["text"],
            "metadata": document.get("metadata", {}),
        }
        for index, document in enumerate(vector_store.documents)
    ]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print(f"Total chunks: {len(chunks)}")
    print(f"Exported to: {output_path}")


if __name__ == "__main__":
    main()
