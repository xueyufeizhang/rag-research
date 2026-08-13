from __future__ import annotations

import re
import math
from collections.abc import Callable, Iterable


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right > 0 else 0.0


def calc_set_metrics(
    retrieved: list[str],
    gold: list[str],
    normalizer: Callable[[str], str] | None = None,
) -> dict:
    normalize = normalizer or (lambda value: value.strip())
    retrieved_set = {normalize(value) for value in retrieved if value}
    gold_set = {normalize(value) for value in gold if value}
    matched = retrieved_set & gold_set

    recall = len(matched) / len(gold_set) if gold_set else 0.0
    precision = len(matched) / len(retrieved_set) if retrieved_set else 0.0
    return {
        "matched": sorted(matched),
        "matched_count": len(matched),
        "gold_count": len(gold_set),
        "retrieved_count": len(retrieved_set),
        "recall": recall,
        "precision": precision,
        "f1": harmonic_mean(precision, recall),
        "hit": bool(matched),
    }


def calc_evidence_metrics(
    retrieved_chunk_ids: list[str],
    chunk_to_evidence: dict[str, list[str]],
    evidence_to_answer_points: dict[str, list[int]],
    answer_point_count: int,
) -> dict:
    """Evaluate ordered chunks against chunk-independent evidence spans.

    Precision uses retrieved chunks as its unit: a chunk is relevant when it
    overlaps at least one gold evidence span. Recall uses canonical evidence as
    its unit, so overlapping chunk boundaries cannot inflate the denominator.
    """
    retrieved = dedupe_preserving_order(retrieved_chunk_ids)
    gold_evidence = set(evidence_to_answer_points)
    covered_evidence: set[str] = set()
    covered_answer_points: set[int] = set()
    relevant_chunks = []
    novel_relevant_chunks = []
    first_relevant_rank = None

    for rank, chunk_id in enumerate(retrieved, start=1):
        chunk_evidence = set(chunk_to_evidence.get(chunk_id, [])) & gold_evidence
        if not chunk_evidence:
            continue

        relevant_chunks.append(chunk_id)
        if first_relevant_rank is None:
            first_relevant_rank = rank

        newly_covered = chunk_evidence - covered_evidence
        if newly_covered:
            novel_relevant_chunks.append(chunk_id)
        covered_evidence.update(chunk_evidence)
        for evidence_id in chunk_evidence:
            covered_answer_points.update(evidence_to_answer_points.get(evidence_id, []))

    chunk_precision = len(relevant_chunks) / len(retrieved) if retrieved else 0.0
    evidence_recall = len(covered_evidence) / len(gold_evidence) if gold_evidence else 0.0
    answer_point_recall = (
        len(covered_answer_points) / answer_point_count if answer_point_count else 0.0
    )
    redundant_relevant_count = len(relevant_chunks) - len(novel_relevant_chunks)
    redundancy_rate = (
        redundant_relevant_count / len(relevant_chunks) if relevant_chunks else 0.0
    )
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    discounted_gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved, start=1)
        if chunk_to_evidence.get(chunk_id)
    )
    ideal_relevant_count = min(len(retrieved), len(chunk_to_evidence))
    ideal_discounted_gain = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )
    ndcg_at_k = (
        discounted_gain / ideal_discounted_gain if ideal_discounted_gain else 0.0
    )

    return {
        "chunk_precision": chunk_precision,
        "evidence_recall": evidence_recall,
        "coverage_f1": harmonic_mean(chunk_precision, evidence_recall),
        "answer_point_recall": answer_point_recall,
        "chunk_hit": bool(relevant_chunks),
        "k": len(retrieved),
        "precision_at_k": chunk_precision,
        "evidence_recall_at_k": evidence_recall,
        "answer_point_recall_at_k": answer_point_recall,
        "reciprocal_rank": reciprocal_rank,
        "ndcg_at_k": ndcg_at_k,
        "first_relevant_rank": first_relevant_rank,
        "redundancy_rate": redundancy_rate,
        "retrieved_count": len(retrieved),
        "relevant_retrieved_count": len(relevant_chunks),
        "novel_relevant_count": len(novel_relevant_chunks),
        "redundant_relevant_count": redundant_relevant_count,
        "gold_evidence_count": len(gold_evidence),
        "matched_evidence_count": len(covered_evidence),
        "gold_answer_point_count": answer_point_count,
        "matched_answer_point_count": len(covered_answer_points),
        "relevant_chunk_ids": relevant_chunks,
        "novel_relevant_chunk_ids": novel_relevant_chunks,
        "matched_evidence_ids": sorted(covered_evidence),
        "matched_answer_point_indices": sorted(covered_answer_points),
    }


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def build_entity_normalizer(
    canonical_names: Iterable[str],
    aliases: dict[str, str] | None = None,
) -> Callable[[str], str]:
    canonical_lookup = {
        normalize_whitespace(name).casefold(): normalize_whitespace(name)
        for name in canonical_names
        if name
    }
    for alias, canonical in (aliases or {}).items():
        canonical_lookup[normalize_whitespace(alias).casefold()] = normalize_whitespace(canonical)

    def normalize(name: str) -> str:
        cleaned = normalize_whitespace(name).strip("\"'")
        return canonical_lookup.get(cleaned.casefold(), cleaned)

    return normalize


def normalize_relation_key(
    key: str,
    entity_normalizer: Callable[[str], str] | None = None,
) -> str:
    normalize_entity = entity_normalizer or normalize_whitespace
    parts = key.split("||")
    if len(parts) != 2:
        return normalize_whitespace(key)
    endpoints = [normalize_entity(part) for part in parts]
    return "||".join(sorted(endpoints, key=str.casefold))


def build_chunk_evidence_map(
    question: dict,
    relevant_chunk_evidence: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    evidence_to_answer_points = {
        evidence["evidence_id"]: evidence.get("supports_answer_points", [])
        for evidence in question.get("gold_evidence_spans", [])
    }
    chunk_to_evidence = {
        chunk_id: [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id in evidence_to_answer_points
        ]
        for chunk_id, evidence_ids in relevant_chunk_evidence.items()
    }
    return chunk_to_evidence, evidence_to_answer_points


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
            source_positions.append(
                pending_whitespace_position
                if pending_whitespace_position is not None
                else position
            )
        output.append(character)
        source_positions.append(position)
        in_whitespace = False
        pending_whitespace_position = None

    return "".join(output), source_positions


def locate_chunks_in_source(source: str, chunks: dict[str, dict]) -> dict[str, tuple[int, int]]:
    source_normalized, source_positions = normalized_text_with_source_positions(source)
    located = {}

    for fallback_id, chunk in chunks.items():
        chunk_id = chunk.get("chunk_id") or fallback_id
        chunk_normalized = normalize_whitespace(chunk.get("text", ""))
        if not chunk_id or not chunk_normalized:
            raise ValueError(f"invalid chunk record: {chunk}")

        first = source_normalized.find(chunk_normalized)
        if first < 0:
            raise ValueError(f"chunk text not found in source: {chunk_normalized[:120]!r}")
        second = source_normalized.find(chunk_normalized, first + 1)
        if second >= 0:
            raise ValueError(f"chunk text is ambiguous in source: {chunk_normalized[:120]!r}")

        start = source_positions[first]
        end = source_positions[first + len(chunk_normalized) - 1] + 1
        located[chunk_id] = (start, end)

    return located


def map_question_evidence_to_chunks(
    question: dict,
    chunk_intervals: dict[str, tuple[int, int]],
) -> dict[str, list[str]]:
    chunk_to_evidence = {}
    evidence_spans = question.get("gold_evidence_spans", [])

    for chunk_id, (chunk_start, chunk_end) in chunk_intervals.items():
        matched = [
            evidence["evidence_id"]
            for evidence in evidence_spans
            if max(chunk_start, evidence["char_start"])
            < min(chunk_end, evidence["char_end"])
        ]
        if matched:
            chunk_to_evidence[chunk_id] = matched

    return chunk_to_evidence
