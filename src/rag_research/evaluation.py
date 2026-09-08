"""Source-aligned MultiHop-RAG retrieval metrics, independent of model calls."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

from rag_research.models import InputDocument, QuestionRecord


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def harmonic_mean(left: float, right: float) -> float:
    return 2 * left * right / (left + right) if left + right > 0 else 0.0


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


def calc_evidence_reachability(
    question: QuestionRecord,
    chunk_to_evidence: Mapping[str, Sequence[str]],
) -> dict:
    """Describe the single-chunk ceiling without discarding unreachable facts.

    This is an index-wide upper bound, independent of ranking and requested K.
    It is not a claim that a finite retrieved prefix can achieve the bound.
    """
    gold_ids = {evidence.evidence_id for evidence in question.evidence}
    if not gold_ids:
        raise ValueError("evidence reachability requires an answerable question")
    if len(gold_ids) != len(question.evidence):
        raise ValueError("question contains duplicate evidence IDs")
    mapped_ids = {
        evidence_id
        for evidence_ids in chunk_to_evidence.values()
        for evidence_id in evidence_ids
    }
    if not mapped_ids <= gold_ids:
        raise ValueError("chunk mapping references unknown gold evidence")
    missing_ids = gold_ids - mapped_ids
    return {
        "gold_evidence_count": len(gold_ids),
        "reachable_evidence_count": len(mapped_ids),
        "unreachable_evidence_count": len(missing_ids),
        "reachable_evidence_ids": sorted(mapped_ids),
        "unreachable_evidence_ids": sorted(missing_ids),
        "evidence_recall_ceiling": len(mapped_ids) / len(gold_ids),
        "joint_evidence_success_ceiling": not missing_ids,
    }


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _retrieved_context(
    retrieved_chunk_ids: Sequence[str],
    chunks: Mapping[str, dict],
    token_counter: Callable[[str], int],
) -> tuple[dict, dict[str, list[tuple[int, int]]]]:
    """Count raw source context and retain document-scoped source intervals.

    Source equality against the corpus is validated once by
    build_multidocument_chunk_index before a run. This also rejects malformed
    records when the pure metric functions are called directly.
    """
    retrieved = dedupe_preserving_order(retrieved_chunk_ids)
    texts: list[str] = []
    document_ids: list[str] = []
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chunk_id in retrieved:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"retrieved chunk is missing from the store: {chunk_id}")
        text = chunk.get("text")
        document_id = chunk.get("document_id")
        start, end = chunk.get("char_start"), chunk.get("char_end")
        if (
            not isinstance(text, str)
            or not isinstance(document_id, str)
            or not document_id.strip()
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or start >= end
            or end - start != len(text)
        ):
            raise ValueError(f"retrieved chunk has invalid source provenance: {chunk_id}")
        texts.append(text)
        document_ids.append(document_id)
        intervals[document_id].append((start, end))

    unique_documents = dedupe_preserving_order(document_ids)
    return {
        "retrieved_count": len(retrieved),
        "retrieved_chars": sum(len(text) for text in texts),
        "chunk_source_tokens": token_counter("\n\n".join(texts)) if texts else 0,
        "retrieved_document_count": len(unique_documents),
        "retrieved_document_ids": unique_documents,
        "cross_document_retrieval": len(unique_documents) > 1,
    }, {document_id: _merge_intervals(spans) for document_id, spans in intervals.items()}


def _evidence_interval_coverage(
    question: QuestionRecord,
    retrieved_intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[set[str], int]:
    """Return complete same-occurrence union hits and unique evidence characters.

    Density includes partial overlap. Every gold occurrence is a distinct source
    location, so retrieved repetitions at different source locations can both
    contribute characters. Overlap or nested facts at the same source location
    count only once. Complete-fact credit never joins fragments from different
    occurrences, even when they have the same fact text.
    """
    fully_covered: set[str] = set()
    intersections: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for evidence in question.evidence:
        if not evidence.fact or not evidence.occurrences:
            raise ValueError(f"evidence {evidence.evidence_id} requires fact occurrences")
        intervals = retrieved_intervals.get(evidence.document_id, ())
        for occurrence in evidence.occurrences:
            start, end = occurrence.char_start, occurrence.char_end
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end - start != len(evidence.fact)
            ):
                raise ValueError(f"evidence {evidence.evidence_id} has invalid source offsets")
            if any(left <= start and end <= right for left, right in intervals):
                fully_covered.add(evidence.evidence_id)
            for left, right in intervals:
                overlap_start, overlap_end = max(left, start), min(right, end)
                if overlap_start < overlap_end:
                    intersections[evidence.document_id].append((overlap_start, overlap_end))

    covered_chars = sum(
        end - start
        for spans in intersections.values()
        for start, end in _merge_intervals(spans)
    )
    return fully_covered, covered_chars


def calc_multihop_retrieval_metrics(
    *,
    question: QuestionRecord,
    retrieved_chunk_ids: Sequence[str],
    requested_k: int,
    chunks: Mapping[str, dict],
    chunk_to_evidence: Mapping[str, Sequence[str]],
    token_counter: Callable[[str], int],
) -> dict:
    """Score a ranked prefix, preserving all gold facts in recall denominators.

    Primary evidence recall requires one chunk to contain a complete occurrence.
    The separately named cross_chunk_union metrics allow adjacent/overlapping
    chunks in the same document to cover one complete occurrence. P@K always
    divides by requested K; precision_among_returned uses the actual list size.
    """
    retrieved = dedupe_preserving_order(retrieved_chunk_ids)
    if (
        not isinstance(requested_k, int)
        or isinstance(requested_k, bool)
        or requested_k <= 0
    ):
        raise ValueError("requested_k must be a positive integer")
    if len(retrieved) > requested_k:
        raise ValueError("retrieved chunks exceed requested_k")
    reachability = calc_evidence_reachability(question, chunk_to_evidence)
    gold_evidence = {evidence.evidence_id for evidence in question.evidence}
    gold_documents = {evidence.document_id for evidence in question.evidence}
    relevant_mapping = {
        chunk_id: set(evidence_ids)
        for chunk_id, evidence_ids in chunk_to_evidence.items()
        if evidence_ids
    }
    context, retrieved_intervals = _retrieved_context(retrieved, chunks, token_counter)
    union_covered, covered_evidence_chars = _evidence_interval_coverage(
        question, retrieved_intervals,
    )

    covered_evidence: set[str] = set()
    relevant_chunk_ids: list[str] = []
    first_relevant_rank: int | None = None
    precision_sum = 0.0
    discounted_gain = 0.0
    for rank, chunk_id in enumerate(retrieved, start=1):
        matched = relevant_mapping.get(chunk_id, set())
        if not matched:
            continue
        relevant_chunk_ids.append(chunk_id)
        if first_relevant_rank is None:
            first_relevant_rank = rank
        precision_sum += len(relevant_chunk_ids) / rank
        discounted_gain += 1.0 / math.log2(rank + 1)
        covered_evidence.update(matched)

    matched_documents = set(context["retrieved_document_ids"]) & gold_documents
    relevant_count = len(relevant_chunk_ids)
    total_relevant_chunks = len(relevant_mapping)
    ideal_relevant_count = min(requested_k, total_relevant_chunks)
    ideal_discounted_gain = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )
    evidence_recall = len(covered_evidence) / len(gold_evidence)
    chunk_precision = relevant_count / requested_k

    return {
        **context,
        "requested_k": requested_k,
        "relevant_retrieved_count": relevant_count,
        "gold_relevant_chunk_count": total_relevant_chunks,
        "chunk_precision": chunk_precision,
        "precision_among_returned": (
            relevant_count / len(retrieved) if retrieved else 0.0
        ),
        "evidence_recall": evidence_recall,
        "coverage_f1": harmonic_mean(chunk_precision, evidence_recall),
        "joint_evidence_success": covered_evidence == gold_evidence,
        "cross_chunk_union_evidence_recall": len(union_covered) / len(gold_evidence),
        "cross_chunk_union_joint_evidence_success": union_covered == gold_evidence,
        "cross_chunk_union_matched_evidence_count": len(union_covered),
        "cross_chunk_union_matched_evidence_ids": sorted(union_covered),
        "document_recall": len(matched_documents) / len(gold_documents),
        "joint_document_success": matched_documents == gold_documents,
        "reciprocal_rank": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "average_precision_at_k": (
            precision_sum / ideal_relevant_count if ideal_relevant_count else 0.0
        ),
        "ndcg_at_k": (
            discounted_gain / ideal_discounted_gain if ideal_discounted_gain else 0.0
        ),
        "first_relevant_rank": first_relevant_rank,
        "gold_evidence_count": reachability["gold_evidence_count"],
        "matched_evidence_count": len(covered_evidence),
        "matched_evidence_ids": sorted(covered_evidence),
        "gold_document_count": len(gold_documents),
        "matched_document_count": len(matched_documents),
        "matched_document_ids": sorted(matched_documents),
        "relevant_chunk_ids": relevant_chunk_ids,
        "covered_evidence_chars": covered_evidence_chars,
        "evidence_density": (
            covered_evidence_chars / context["retrieved_chars"]
            if context["retrieved_chars"] else 0.0
        ),
    }


def calc_null_retrieval_context_metrics(
    *,
    retrieved_chunk_ids: Sequence[str],
    chunks: Mapping[str, dict],
    token_counter: Callable[[str], int],
) -> dict:
    """Describe a null-query context without inventing relevance labels."""
    context, _ = _retrieved_context(retrieved_chunk_ids, chunks, token_counter)
    return {**context, "relevance_metrics_applicable": False}
