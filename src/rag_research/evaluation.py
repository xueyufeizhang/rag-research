from __future__ import annotations

import re
import math
from collections.abc import Callable, Iterable, Mapping, Sequence

from rag_research.models import InputDocument, QuestionRecord


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


def calc_context_efficiency_metrics(
    source: str,
    retrieved_chunk_ids: list[str],
    chunks: dict[str, dict],
    chunk_intervals: dict[str, tuple[int, int]],
    gold_evidence_spans: list[dict],
    matched_answer_point_count: int,
    token_counter: Callable[[str], int],
) -> dict:
    """Measure evidence yield relative to the retrieved context budget.

    Character counts use the chunks' source-aligned intervals. Overlapping
    retrieved chunks count repeatedly in the context denominator, while covered
    evidence characters are unioned so duplicated evidence is not rewarded.
    Token counts use one fixed evaluation tokenizer over the retrieved chunk
    texts joined in retrieval order.
    """
    retrieved = dedupe_preserving_order(retrieved_chunk_ids)
    retrieved_texts = []
    retrieved_chars = 0
    covered_intervals = []

    for chunk_id in retrieved:
        chunk = chunks.get(chunk_id)
        interval = chunk_intervals.get(chunk_id)
        if chunk is None or interval is None:
            raise ValueError(f"retrieved chunk is missing from the evaluation index: {chunk_id}")

        chunk_start, chunk_end = interval
        if chunk_start < 0 or chunk_end > len(source) or chunk_start >= chunk_end:
            raise ValueError(f"invalid source interval for retrieved chunk {chunk_id}: {interval}")
        retrieved_texts.append(chunk.get("text", ""))
        retrieved_chars += chunk_end - chunk_start

        for evidence in gold_evidence_spans:
            evidence_start = evidence.get("char_start")
            evidence_end = evidence.get("char_end")
            if evidence_start is None or evidence_end is None:
                continue
            overlap_start = max(chunk_start, evidence_start)
            overlap_end = min(chunk_end, evidence_end)
            if overlap_start < overlap_end:
                covered_intervals.append((overlap_start, overlap_end))

    retrieved_text = "\n\n".join(retrieved_texts)
    retrieved_tokens = token_counter(retrieved_text) if retrieved_text else 0
    covered_evidence_chars = _merged_interval_length(covered_intervals)

    return {
        "retrieved_chars": retrieved_chars,
        "retrieved_tokens": retrieved_tokens,
        "covered_evidence_chars": covered_evidence_chars,
        "evidence_density": (
            covered_evidence_chars / retrieved_chars if retrieved_chars else 0.0
        ),
        "answer_points_per_1k_tokens": (
            matched_answer_point_count * 1000.0 / retrieved_tokens
            if retrieved_tokens else 0.0
        ),
    }


def _merged_interval_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if start < end)
    if not ordered:
        return 0

    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


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


def build_multidocument_chunk_index(
    documents: Sequence[InputDocument],
    chunks: Mapping[str, dict],
) -> dict[str, tuple[str, int, int]]:
    """Validate source-aligned chunks and index their document-relative spans.

    Multi-document offsets are meaningful only together with ``document_id``.
    This validation deliberately requires exact source slices so an evaluator
    cannot silently score a chunk store built from different document text.
    """
    documents_by_id = {document.document_id: document for document in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("evaluation documents contain duplicate document_id values")

    indexed: dict[str, tuple[str, int, int]] = {}
    for fallback_id, chunk in chunks.items():
        if not isinstance(chunk, dict):
            raise ValueError(f"invalid chunk record for {fallback_id}: expected an object")

        chunk_id = chunk.get("chunk_id") or fallback_id
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"invalid chunk ID for stored key {fallback_id!r}")
        if chunk_id != fallback_id:
            raise ValueError(
                f"chunk ID does not match its store key: {chunk_id!r} != {fallback_id!r}"
            )

        document_id = chunk.get("document_id")
        if not isinstance(document_id, str) or document_id not in documents_by_id:
            raise ValueError(f"chunk {chunk_id} references unknown document: {document_id!r}")

        char_start = chunk.get("char_start")
        char_end = chunk.get("char_end")
        if (
            not isinstance(char_start, int)
            or isinstance(char_start, bool)
            or not isinstance(char_end, int)
            or isinstance(char_end, bool)
        ):
            raise ValueError(f"chunk {chunk_id} requires integer source offsets")

        source = documents_by_id[document_id].text
        if char_start < 0 or char_start >= char_end or char_end > len(source):
            raise ValueError(
                f"chunk {chunk_id} has invalid source interval: "
                f"({char_start}, {char_end})"
            )
        chunk_text = chunk.get("text")
        if chunk_text != source[char_start:char_end]:
            raise ValueError(f"chunk {chunk_id} text does not match its document source slice")

        indexed[chunk_id] = (document_id, char_start, char_end)

    return indexed


def map_multihop_evidence_to_chunks(
    question: QuestionRecord,
    chunk_index: Mapping[str, tuple[str, int, int]],
) -> dict[str, list[str]]:
    """Map canonical evidence to chunks without crossing document boundaries.

    A chunk is relevant only when it fully contains at least one occurrence of
    the evidence fact. Partial overlap is insufficient for multi-hop evidence
    retrieval because it may omit the part needed for reasoning.
    """
    chunk_to_evidence: dict[str, list[str]] = {}
    for chunk_id, (document_id, chunk_start, chunk_end) in chunk_index.items():
        matched = []
        for evidence in question.evidence:
            if evidence.document_id != document_id:
                continue
            if any(
                chunk_start <= occurrence.char_start
                and occurrence.char_end <= chunk_end
                for occurrence in evidence.occurrences
            ):
                matched.append(evidence.evidence_id)
        if matched:
            chunk_to_evidence[chunk_id] = matched
    return chunk_to_evidence


def normalize_multihop_official_text(value: str) -> str:
    """Apply the exact normalization used by MultiHop-RAG's evaluator."""
    return value.replace(" ", "").replace("\n", "")


def calc_multihop_official_metrics(
    *,
    retrieved_texts: Sequence[str],
    gold_facts: Sequence[str],
) -> dict:
    """Reproduce the official MultiHop-RAG top-10 scoring semantics.

    The official baseline retrieves ten chunks, considers a chunk relevant when
    any normalized gold fact is a substring of its normalized text, and credits
    each gold fact only at the first retrieved chunk that contains it. Its
    ``MAP@10`` formula is intentionally reproduced rather than replaced with a
    conventional average-precision implementation.
    """
    if not gold_facts:
        raise ValueError("official MultiHop-RAG metrics require gold facts")
    if any(not isinstance(text, str) for text in retrieved_texts):
        raise ValueError("official MultiHop-RAG retrieved texts must be strings")
    if any(not isinstance(fact, str) for fact in gold_facts):
        raise ValueError("official MultiHop-RAG gold facts must be strings")

    normalized_gold = [
        normalize_multihop_official_text(fact)
        for fact in gold_facts
    ]
    normalized_retrieved = [
        normalize_multihop_official_text(text)
        for text in retrieved_texts[:10]
    ]

    hits_at_4 = False
    hits_at_10 = False
    average_precision_sum = 0.0
    first_relevant_rank: int | None = None
    found_gold: list[str] = []

    for rank, retrieved_item in enumerate(normalized_retrieved, start=1):
        if not any(gold_item in retrieved_item for gold_item in normalized_gold):
            continue

        hits_at_10 = True
        if rank <= 4:
            hits_at_4 = True
        if first_relevant_rank is None:
            first_relevant_rank = rank

        newly_found_count = 0
        for gold_item in normalized_gold:
            if gold_item in retrieved_item and gold_item not in found_gold:
                newly_found_count += 1
                found_gold.append(gold_item)
        average_precision_sum += newly_found_count / rank

    return {
        "Hits@10": int(hits_at_10),
        "Hits@4": int(hits_at_4),
        "MAP@10": average_precision_sum / min(len(normalized_gold), 10),
        "MRR@10": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "first_relevant_rank": first_relevant_rank,
        "matched_gold_count": len(found_gold),
        "gold_count": len(normalized_gold),
        "retrieved_count": len(normalized_retrieved),
    }


def calc_multihop_retrieval_metrics(
    *,
    retrieved_chunk_ids: Sequence[str],
    requested_k: int | None = None,
    chunks: Mapping[str, dict],
    chunk_to_evidence: Mapping[str, Sequence[str]],
    evidence_to_document: Mapping[str, str],
    evidence_lengths: Mapping[str, int],
    token_counter: Callable[[str], int],
) -> dict:
    """Score one ranked chunk prefix for an answerable multi-hop question."""
    retrieved = dedupe_preserving_order(retrieved_chunk_ids)
    evaluation_k = len(retrieved) if requested_k is None else requested_k
    if evaluation_k <= 0:
        raise ValueError("requested_k must be positive")
    if len(retrieved) > evaluation_k:
        raise ValueError("retrieved chunks exceed requested_k")
    gold_evidence = set(evidence_to_document)
    if not gold_evidence:
        raise ValueError("multi-hop retrieval metrics require gold evidence")

    gold_documents = set(evidence_to_document.values())
    covered_evidence: set[str] = set()
    relevant_chunk_ids: list[str] = []
    retrieved_document_ids: list[str] = []
    first_relevant_rank: int | None = None
    precision_sum = 0.0

    retrieved_texts: list[str] = []
    retrieved_chars = 0
    for rank, chunk_id in enumerate(retrieved, start=1):
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"retrieved chunk is missing from the store: {chunk_id}")
        document_id = chunk.get("document_id")
        text = chunk.get("text")
        if not isinstance(document_id, str) or not isinstance(text, str):
            raise ValueError(f"retrieved chunk has invalid provenance: {chunk_id}")

        retrieved_document_ids.append(document_id)
        retrieved_texts.append(text)
        retrieved_chars += len(text)

        matched = set(chunk_to_evidence.get(chunk_id, ())) & gold_evidence
        if not matched:
            continue
        relevant_chunk_ids.append(chunk_id)
        if first_relevant_rank is None:
            first_relevant_rank = rank
        precision_sum += len(relevant_chunk_ids) / rank
        covered_evidence.update(matched)

    retrieved_document_set = set(retrieved_document_ids)
    matched_documents = retrieved_document_set & gold_documents
    relevant_count = len(relevant_chunk_ids)
    retrieved_count = len(retrieved)
    total_relevant_chunks = len(chunk_to_evidence)
    ideal_relevant_count = min(evaluation_k, total_relevant_chunks)
    discounted_gain = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved, start=1)
        if chunk_id in chunk_to_evidence
    )
    ideal_discounted_gain = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )
    evidence_recall = len(covered_evidence) / len(gold_evidence)
    chunk_precision = relevant_count / evaluation_k
    retrieved_tokens = token_counter("\n\n".join(retrieved_texts)) if retrieved_texts else 0
    covered_evidence_chars = sum(
        evidence_lengths[evidence_id]
        for evidence_id in covered_evidence
    )

    return {
        "requested_k": evaluation_k,
        "retrieved_count": retrieved_count,
        "relevant_retrieved_count": relevant_count,
        "gold_relevant_chunk_count": total_relevant_chunks,
        "chunk_precision": chunk_precision,
        "evidence_recall": evidence_recall,
        "coverage_f1": harmonic_mean(chunk_precision, evidence_recall),
        "joint_evidence_success": covered_evidence == gold_evidence,
        "document_recall": len(matched_documents) / len(gold_documents),
        "joint_document_success": matched_documents == gold_documents,
        "reciprocal_rank": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "average_precision_at_k": (
            precision_sum / min(total_relevant_chunks, evaluation_k)
            if total_relevant_chunks
            else 0.0
        ),
        "ndcg_at_k": (
            discounted_gain / ideal_discounted_gain
            if ideal_discounted_gain
            else 0.0
        ),
        "first_relevant_rank": first_relevant_rank,
        "gold_evidence_count": len(gold_evidence),
        "matched_evidence_count": len(covered_evidence),
        "matched_evidence_ids": sorted(covered_evidence),
        "gold_document_count": len(gold_documents),
        "matched_document_count": len(matched_documents),
        "matched_document_ids": sorted(matched_documents),
        "retrieved_document_count": len(retrieved_document_set),
        "retrieved_document_ids": dedupe_preserving_order(retrieved_document_ids),
        "cross_document_retrieval": len(retrieved_document_set) > 1,
        "relevant_chunk_ids": relevant_chunk_ids,
        "retrieved_chars": retrieved_chars,
        "retrieved_tokens": retrieved_tokens,
        "covered_evidence_chars": covered_evidence_chars,
        "evidence_density": (
            covered_evidence_chars / retrieved_chars if retrieved_chars else 0.0
        ),
    }


def calc_null_retrieval_context_metrics(
    *,
    retrieved_chunk_ids: Sequence[str],
    chunks: Mapping[str, dict],
    token_counter: Callable[[str], int],
) -> dict:
    """Describe retrieval for a null query without inventing relevance labels."""
    retrieved = dedupe_preserving_order(retrieved_chunk_ids)
    texts: list[str] = []
    document_ids: list[str] = []
    for chunk_id in retrieved:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"retrieved chunk is missing from the store: {chunk_id}")
        text = chunk.get("text")
        document_id = chunk.get("document_id")
        if not isinstance(text, str) or not isinstance(document_id, str):
            raise ValueError(f"retrieved chunk has invalid provenance: {chunk_id}")
        texts.append(text)
        document_ids.append(document_id)

    unique_documents = dedupe_preserving_order(document_ids)
    return {
        "k": len(retrieved),
        "retrieved_count": len(retrieved),
        "retrieved_chars": sum(len(text) for text in texts),
        "retrieved_tokens": token_counter("\n\n".join(texts)) if texts else 0,
        "retrieved_document_count": len(unique_documents),
        "retrieved_document_ids": unique_documents,
        "cross_document_retrieval": len(unique_documents) > 1,
        "relevance_metrics_applicable": False,
    }
