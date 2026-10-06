from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.ingestion.loaders import DocumentLoader
from app.ingestion.preprocessing import TextPreprocessor
from app.ingestion.chunking import TextChunker
from app.ingestion.chunking_experimental import BoundaryAwareTextChunker


EVIDENCE_PATH = ROOT_DIR / "datasets" / "evidence_ground_truth.jsonl"
OUTPUT_DIR = ROOT_DIR / "experiments" / "chunking_comparison"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_evidence_spans(
    text: str,
    evidence: str,
) -> list[tuple[int, int]]:
    """
    Find evidence in source text while allowing whitespace and line-break
    differences introduced by PDF extraction.
    """
    pattern = re.escape(evidence.strip())
    pattern = re.sub(r"\\\s+", r"\\s+", pattern)

    return [
        match.span()
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]


def get_chunk_ids_for_evidence(
    chunks: list,
    source_docs: list,
    evidence_records: list[dict[str, Any]],
) -> tuple[set[str], list[dict[str, Any]]]:
    relevant_ids: set[str] = set()
    diagnostics: list[dict[str, Any]] = []

    for evidence_record in evidence_records:
        target_page = evidence_record.get("page")
        evidence_text = evidence_record["text"]
        matched_anywhere = False
        matched_chunk_ids: list[str] = []

        for source_doc in source_docs:
            source_metadata = source_doc.metadata
            source_page = source_metadata.get("page")

            if target_page is not None and source_page != target_page:
                continue

            source_text = source_doc.content.strip()
            spans = find_evidence_spans(source_text, evidence_text)

            if not spans:
                continue

            matched_anywhere = True

            for span_start, span_end in spans:
                for chunk_id, chunk in enumerate(chunks):
                    metadata = chunk.metadata

                    if metadata.get("page") != source_page:
                        continue

                    if metadata.get("filename") != source_metadata.get("filename"):
                        continue

                    chunk_start = metadata.get("char_start")
                    chunk_end = metadata.get("char_end")

                    if chunk_start is None or chunk_end is None:
                        continue

                    # Include chunks whose text range overlaps the evidence.
                    if chunk_start < span_end and chunk_end > span_start:
                        relevant_ids.add(str(chunk_id))
                        matched_chunk_ids.append(str(chunk_id))

        diagnostics.append({
            "page": target_page,
            "evidence": evidence_text,
            "found_in_source": matched_anywhere,
            "matched_chunk_ids": sorted(
                set(matched_chunk_ids),
                key=int,
            ),
        })

    return relevant_ids, diagnostics


def save_chunks(path: Path, chunks: list) -> None:
    records = []

    for index, chunk in enumerate(chunks):
        records.append({
            "id": str(index),
            "text": chunk.content,
            "metadata": chunk.metadata,
        })

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def run_strategy(
    name: str,
    chunker,
    source_docs: list,
    evidence_records: list[dict[str, Any]],
) -> None:
    chunks = chunker.chunk_documents(source_docs)

    strategy_dir = OUTPUT_DIR / name
    strategy_dir.mkdir(parents=True, exist_ok=True)

    save_chunks(strategy_dir / "chunks.json", chunks)

    mapped_records = []
    mapping_report = []

    for record in evidence_records:
        relevant_ids, diagnostics = get_chunk_ids_for_evidence(
            chunks,
            source_docs,
            record["evidence"],
        )

        mapped_records.append({
            "query": record["query"],
            "relevant_docs": sorted(relevant_ids, key=int),
        })

        mapping_report.append({
            "query": record["query"],
            "relevant_docs": sorted(relevant_ids, key=int),
            "evidence_diagnostics": diagnostics,
        })

    write_jsonl(
        strategy_dir / "ground_truth.jsonl",
        mapped_records,
    )

    with (strategy_dir / "mapping_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(mapping_report, file, ensure_ascii=False, indent=2)

    print(f"\nStrategy: {name}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Queries mapped: {len(mapped_records)}")

    missing = [
        item
        for query in mapping_report
        for item in query["evidence_diagnostics"]
        if not item["found_in_source"]
    ]

    queries_without_labels = [
        item["query"]
        for item in mapped_records
        if not item["relevant_docs"]
    ]

    print(f"Evidence anchors not found: {len(missing)}")
    print(f"Queries without relevant chunks: {len(queries_without_labels)}")

    if missing:
        print("\nUnmatched evidence:")
        for item in missing:
            print(f"- Page {item['page']}: {item['evidence']}")

    if queries_without_labels:
        print("\nQueries without mapped labels:")
        for query in queries_without_labels:
            print(f"- {query}")


def main() -> None:
    evidence_records = load_jsonl(EVIDENCE_PATH)

    documents = DocumentLoader().load_directory(settings.documents_dir)
    processed_docs = TextPreprocessor().preprocess_documents(documents)

    baseline = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    boundary_aware = BoundaryAwareTextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    run_strategy(
        "baseline",
        baseline,
        processed_docs,
        evidence_records,
    )

    run_strategy(
        "boundary_aware",
        boundary_aware,
        processed_docs,
        evidence_records,
    )

    print(f"\nOutputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()