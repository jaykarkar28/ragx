import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationRecord:
    query: str
    relevant_docs: list[str]
    reference_answer: str | None = None


class EvaluationDataset:
    def __init__(self, records: list[EvaluationRecord]):
        self.records = records

    @classmethod
    def from_jsonl(cls, file_path: str | Path) -> "EvaluationDataset":
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        records = []

        # utf-8-sig handles UTF-8 files with or without a BOM.
        with path.open("r", encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {exc}"
                    ) from exc

                query = record.get("query")
                relevant_docs = record.get("relevant_docs")
                reference_answer = record.get("reference_answer")

                if not isinstance(query, str) or not query.strip():
                    raise ValueError(
                        f"Line {line_number}: 'query' must be a non-empty string"
                    )

                if (
                    not isinstance(relevant_docs, list)
                    or not relevant_docs
                    or not all(
                        isinstance(doc_id, str) and doc_id.strip()
                        for doc_id in relevant_docs
                    )
                ):
                    raise ValueError(
                        f"Line {line_number}: 'relevant_docs' must be a "
                        "non-empty list of non-empty strings"
                    )

                if reference_answer is not None:
                    if (
                        not isinstance(reference_answer, str)
                        or not reference_answer.strip()
                    ):
                        raise ValueError(
                            f"Line {line_number}: 'reference_answer' must be "
                            "a non-empty string when provided"
                        )
                    reference_answer = reference_answer.strip()

                records.append(
                    EvaluationRecord(
                        query=query.strip(),
                        relevant_docs=relevant_docs,
                        reference_answer=reference_answer,
                    )
                )

        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)
