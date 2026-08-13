import argparse
import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalized_text_with_source_positions(text: str) -> tuple[str, list[int]]:
    output = []
    source_positions = []
    in_whitespace = True
    pending_whitespace_position = None

    for position, character in enumerate(text):
        if character.isspace():
            if not in_whitespace:
                pending_whitespace_position = position
            in_whitespace = True
            continue

        if output and in_whitespace:
            output.append(" ")
            source_positions.append(pending_whitespace_position if pending_whitespace_position is not None else position)
        output.append(character)
        source_positions.append(position)
        in_whitespace = False
        pending_whitespace_position = None

    return "".join(output), source_positions


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def locate_chunk(source_normalized: str, source_positions: list[int], chunk_text: str) -> tuple[int, int]:
    chunk_normalized = normalize_text(chunk_text)
    if not chunk_normalized:
        raise ValueError("empty chunk text")

    first = source_normalized.find(chunk_normalized)
    if first < 0:
        raise ValueError(f"chunk text not found in source: {chunk_normalized[:120]!r}")
    second = source_normalized.find(chunk_normalized, first + 1)
    if second >= 0:
        raise ValueError(f"chunk text is ambiguous in source: {chunk_normalized[:120]!r}")

    start = source_positions[first]
    end = source_positions[first + len(chunk_normalized) - 1] + 1
    return start, end


def overlaps(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map canonical evidence spans to one chunk store.")
    parser.add_argument(
        "--canonical",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/carol_canonical.json",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "data/raw/a_christmas_carol.txt",
    )
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    canonical = json.loads(args.canonical.read_text(encoding="utf-8"))
    source = args.source.read_text(encoding="utf-8")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    expected_sha256 = canonical.get("source", {}).get("sha256")
    if source_sha256 != expected_sha256:
        raise ValueError(f"source hash mismatch: expected {expected_sha256}, got {source_sha256}")

    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    if isinstance(chunks, dict):
        chunk_items = list(chunks.items())
    elif isinstance(chunks, list):
        chunk_items = [(chunk.get("chunk_id"), chunk) for chunk in chunks]
    else:
        raise ValueError("chunks file must contain an object or list")

    source_normalized, source_positions = normalized_text_with_source_positions(source)
    located_chunks = []
    for fallback_id, chunk in chunk_items:
        chunk_id = chunk.get("chunk_id") or fallback_id
        chunk_text = chunk.get("text")
        if not chunk_id or not chunk_text:
            raise ValueError(f"invalid chunk record: {chunk}")
        char_start, char_end = locate_chunk(source_normalized, source_positions, chunk_text)
        located_chunks.append({
            "chunk_id": chunk_id,
            "char_start": char_start,
            "char_end": char_end,
        })

    mapped_questions = []
    for question in canonical.get("questions", []):
        evidence_spans = question.get("gold_evidence_spans", [])
        chunk_evidence = {}
        for chunk in located_chunks:
            matched_evidence = [
                evidence["evidence_id"]
                for evidence in evidence_spans
                if overlaps(
                    chunk["char_start"],
                    chunk["char_end"],
                    evidence["char_start"],
                    evidence["char_end"],
                )
            ]
            if matched_evidence:
                chunk_evidence[chunk["chunk_id"]] = matched_evidence

        mapped_question = dict(question)
        mapped_question["gold_chunk_evidence"] = chunk_evidence
        mapped_questions.append(mapped_question)

    output = {
        "schema_version": "1.0",
        "derived_from": str(args.canonical),
        "chunk_store": str(args.chunks),
        "mapping_policy": "Records which canonical evidence spans overlap each chunk; evidence spans remain the evaluation unit.",
        "questions": mapped_questions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Mapped {len(mapped_questions)} questions across {len(located_chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
