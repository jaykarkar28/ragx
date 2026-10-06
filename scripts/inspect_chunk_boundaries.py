
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

CHUNK_FILES = [
    ROOT / "experiments" / "all_chunks.json",
    ROOT / "experiments" / "vector_store" / "documents.json",
]

TARGET_IDS = [21, 22, 23, 24, 34, 35, 36]
MAX_OVERLAP_CHECK = 300


def load_json(path: Path) -> Any:
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("chunks", "documents", "data", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return value

        if all(isinstance(value, dict) for value in data.values()):
            records = []
            for key, value in data.items():
                record = dict(value)
                record.setdefault("id", key)
                records.append(record)
            return records

    raise ValueError("Unsupported JSON structure.")


def get_chunk_id(record: dict) -> str | None:
    value = record.get("id")

    if value is None:
        value = record.get("chunk_id")

    if value is None:
        return None

    return str(value)


def get_text(record: dict) -> str:
    value = record.get("text", "")

    if value is None:
        return ""

    return str(value)


def get_metadata(record: dict) -> dict:
    metadata = record.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def get_field(record: dict, field: str):
    metadata = get_metadata(record)

    if field in record:
        return record[field]

    return metadata.get(field)


def normalize_source(value: Any) -> str | None:
    if value is None:
        return None

    return str(value).replace("\\", "/").strip()


def find_common_overlap(
    left_text: str,
    right_text: str,
    max_length: int = MAX_OVERLAP_CHECK,
) -> str:
    left = left_text[-max_length:]
    right = right_text[:max_length]

    max_size = min(len(left), len(right))

    for size in range(max_size, 0, -1):
        if left[-size:] == right[:size]:
            return left[-size:]

    return ""


def print_record(record: dict):
    chunk_id = get_chunk_id(record)
    text = get_text(record)
    metadata = get_metadata(record)

    print("\n" + "=" * 90)
    print(f"CHUNK ID: {chunk_id}")
    print("=" * 90)

    print("\nMETADATA")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    print("\nSELECTED FIELDS")
    for field in (
        "source",
        "page",
        "chunk_index",
        "char_start",
        "char_end",
    ):
        print(f"{field:15}: {get_field(record, field)}")

    print("\nTEXT")
    print("-" * 90)
    print(text)
    print("-" * 90)

    print(f"Characters: {len(text)}")
    print(f"Starts with: {text[:60]!r}")
    print(f"Ends with:   {text[-60:]!r}")


def inspect_pair(left: dict, right: dict):
    left_id = get_chunk_id(left)
    right_id = get_chunk_id(right)

    left_text = get_text(left)
    right_text = get_text(right)

    left_source = normalize_source(get_field(left, "source"))
    right_source = normalize_source(get_field(right, "source"))

    left_index = get_field(left, "chunk_index")
    right_index = get_field(right, "chunk_index")

    left_start = get_field(left, "char_start")
    left_end = get_field(left, "char_end")
    right_start = get_field(right, "char_start")
    right_end = get_field(right, "char_end")

    overlap = find_common_overlap(left_text, right_text)

    print("\n" + "#" * 90)
    print(f"PAIR: {left_id} -> {right_id}")
    print("#" * 90)

    same_source = (
        left_source is not None
        and right_source is not None
        and left_source == right_source
    )

    print(f"Same source: {same_source}")
    print(f"Left source:  {left_source}")
    print(f"Right source: {right_source}")

    consecutive_index = False
    if left_index is not None and right_index is not None:
        try:
            consecutive_index = int(right_index) == int(left_index) + 1
        except (TypeError, ValueError):
            pass

    print(f"Chunk indexes: {left_index} -> {right_index}")
    print(f"Consecutive indexes: {consecutive_index}")

    print(f"Character ranges: [{left_start}, {left_end}] -> "
          f"[{right_start}, {right_end}]")

    if all(value is not None for value in (left_end, right_start)):
        try:
            difference = int(right_start) - int(left_end)
            print(f"Start/end difference: {difference}")

            if difference == 0:
                print("Ranges meet exactly.")
            elif difference < 0:
                print(f"Ranges overlap by {-difference} characters.")
            else:
                print(f"Possible gap of {difference} characters.")
        except (TypeError, ValueError):
            print("Could not calculate character-range difference.")

    print(f"Exact text overlap length: {len(overlap)}")

    if overlap:
        print("Detected overlap:")
        print(repr(overlap))

    print("\nLEFT CHUNK END")
    print(repr(left_text[-180:]))

    print("\nRIGHT CHUNK START")
    print(repr(right_text[:180]))

    if same_source and consecutive_index:
        print("\nRESULT: Same source and consecutive chunk indexes.")
    elif same_source:
        print("\nRESULT: Same source, but consecutive indexes not confirmed.")
    else:
        print("\nRESULT: Adjacency is not established.")


def main():
    print("RAGX CHUNK BOUNDARY INSPECTION")
    print(f"Project root: {ROOT}")

    selected_file = None
    records = None

    for path in CHUNK_FILES:
        data = load_json(path)

        if data is None:
            print(f"\nNot found: {path}")
            continue

        try:
            candidate_records = normalize_records(data)
        except ValueError as error:
            print(f"\nCould not read {path}: {error}")
            continue

        if candidate_records:
            selected_file = path
            records = candidate_records
            break

    if records is None:
        raise FileNotFoundError(
            "No usable chunk JSON file was found."
        )

    print(f"\nUsing: {selected_file}")
    print(f"Total records: {len(records)}")

    by_id = {}
    for record in records:
        chunk_id = get_chunk_id(record)
        if chunk_id is not None:
            by_id[chunk_id] = record

    print("\n\nSELECTED CHUNKS")
    for chunk_id in TARGET_IDS:
        record = by_id.get(str(chunk_id))

        if record is None:
            print(f"\nChunk {chunk_id} not found.")
            continue

        print_record(record)

    print("\n\nBOUNDARY COMPARISONS")
    pairs = [
        (21, 22),
        (22, 23),
        (23, 24),
        (34, 35),
        (35, 36),
    ]

    for left_id, right_id in pairs:
        left = by_id.get(str(left_id))
        right = by_id.get(str(right_id))

        if left is None or right is None:
            print(f"\nSkipping {left_id} -> {right_id}: missing chunk.")
            continue

        inspect_pair(left, right)

    print("\n\nINSPECTION COMPLETE")


if __name__ == "__main__":
    main()
