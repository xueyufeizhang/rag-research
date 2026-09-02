from dataclasses import dataclass, field
from typing import Awaitable, Callable

from rag_research.agentic_boundaries import (
    rebalance_document_boundaries,
    validate_boundary_structure,
)
from rag_research.agentic_llm import AgenticLlmGateway
from rag_research.chunking_models import ChunkSpan, SentenceSpan
from rag_research.text_spans import split_sentences


AGENTIC_RECENT_PROPOSITIONS = 3
AGENTIC_CATALOG_MAX_CHUNKS = 20


@dataclass
class AgenticManagedChunk:
    """Mutable semantic state maintained while propositions arrive in order."""

    chunk_id: str
    title: str
    summary: str
    sentence_start: int
    sentence_end: int
    proposition_ranges: list[tuple[int, int]] = field(default_factory=list)
    revision: int = 1

    @property
    def sentence_count(self) -> int:
        return self.sentence_end - self.sentence_start + 1


async def agentic_chunk(
    text: str,
    batch_max_sentences: int,
    batch_max_chars: int,
    min_sentences: int,
    max_sentences: int,
    concurrency: int,
    retries: int,
    llm_func: Callable[..., Awaitable[str]],
    state_events: list[dict[str, object]] | None = None,
) -> list[ChunkSpan]:
    """Build source-aligned chunks with a proposition-aware stateful agent."""
    _validate_runtime_config(
        batch_max_sentences=batch_max_sentences,
        batch_max_chars=batch_max_chars,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        concurrency=concurrency,
        retries=retries,
    )

    sentences = split_sentences(text)
    if not sentences:
        return []

    llm_gateway = AgenticLlmGateway(
        llm_func=llm_func,
        retries=retries,
        concurrency=concurrency,
    )
    proposition_boundaries = await llm_gateway.extract_propositions(
        sentences=sentences,
        batch_max_sentences=batch_max_sentences,
        batch_max_chars=batch_max_chars,
        max_sentences=max_sentences,
        state_events=state_events,
    )
    proposition_count = len(proposition_boundaries)
    print(
        f"[agentic] extracted {proposition_count} propositions from "
        f"{len(sentences)} sentences",
        flush=True,
    )

    managed_chunks = await _route_propositions(
        text=text,
        sentences=sentences,
        proposition_boundaries=proposition_boundaries,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        llm_gateway=llm_gateway,
        state_events=state_events,
    )
    return await _finalize_chunks(
        text=text,
        sentences=sentences,
        managed_chunks=managed_chunks,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        llm_gateway=llm_gateway,
        state_events=state_events,
    )


async def _route_propositions(
    *,
    text: str,
    sentences: list[SentenceSpan],
    proposition_boundaries: list[tuple[int, int]],
    min_sentences: int,
    max_sentences: int,
    llm_gateway: AgenticLlmGateway,
    state_events: list[dict[str, object]] | None,
) -> list[AgenticManagedChunk]:
    managed_chunks: list[AgenticManagedChunk] = []
    proposition_count = len(proposition_boundaries)

    for proposition_index, proposition_range in enumerate(
        proposition_boundaries,
        start=1,
    ):
        start, end = proposition_range
        current = managed_chunks[-1] if managed_chunks else None
        allowed_actions, forced_reason = _allowed_actions(
            current=current,
            proposition_size=end - start + 1,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )
        state_payload = _build_state_payload(
            text=text,
            sentences=sentences,
            managed_chunks=managed_chunks,
            proposition_range=proposition_range,
            proposition_index=proposition_index,
            allowed_actions=allowed_actions,
            forced_reason=forced_reason,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )
        decision = await llm_gateway.decide_transition(
            proposition_index=proposition_index,
            state_payload=state_payload,
            allowed_actions=allowed_actions,
            forced_reason=forced_reason,
            text=text,
            sentences=sentences,
            proposition_range=proposition_range,
            open_chunk_start=(
                current.sentence_start if current is not None else None
            ),
            fallback_title=current.title if current is not None else None,
        )
        target = _apply_transition(
            managed_chunks=managed_chunks,
            proposition_range=proposition_range,
            decision=decision,
        )
        _record_transition(
            state_events=state_events,
            proposition_index=proposition_index,
            proposition_range=proposition_range,
            allowed_actions=allowed_actions,
            forced_reason=forced_reason,
            decision=decision,
            target=target,
        )

        if proposition_index % 10 == 0 or proposition_index == proposition_count:
            print(
                f"[agentic] state {proposition_index}/{proposition_count}, "
                f"{len(managed_chunks)} chunks",
                flush=True,
            )

    return managed_chunks


def _allowed_actions(
    *,
    current: AgenticManagedChunk | None,
    proposition_size: int,
    min_sentences: int,
    max_sentences: int,
) -> tuple[tuple[str, ...], str | None]:
    if current is None:
        return ("new_chunk",), "no chunk exists yet"
    if current.sentence_count + proposition_size > max_sentences:
        return (
            ("new_chunk",),
            "appending would exceed the maximum chunk size",
        )
    if current.sentence_count < min_sentences:
        return ("append",), "the open chunk has not reached the minimum size"
    return ("append", "new_chunk"), None


def _apply_transition(
    *,
    managed_chunks: list[AgenticManagedChunk],
    proposition_range: tuple[int, int],
    decision: dict[str, str],
) -> AgenticManagedChunk:
    start, end = proposition_range
    action = decision["action"]
    if action == "append":
        if not managed_chunks:
            raise RuntimeError("agentic state cannot append without an open chunk")
        target = managed_chunks[-1]
        target.sentence_end = end
        target.proposition_ranges.append(proposition_range)
        target.title = decision["title"]
        target.summary = decision["summary"]
        target.revision += 1
        return target

    target = AgenticManagedChunk(
        chunk_id=f"agentic-chunk-{len(managed_chunks) + 1:04d}",
        title=decision["title"],
        summary=decision["summary"],
        sentence_start=start,
        sentence_end=end,
        proposition_ranges=[proposition_range],
    )
    managed_chunks.append(target)
    return target


def _record_transition(
    *,
    state_events: list[dict[str, object]] | None,
    proposition_index: int,
    proposition_range: tuple[int, int],
    allowed_actions: tuple[str, ...],
    forced_reason: str | None,
    decision: dict[str, str],
    target: AgenticManagedChunk,
) -> None:
    if state_events is None:
        return

    event: dict[str, object] = {
        "event": "transition",
        "proposition_index": proposition_index,
        "proposition_range": list(proposition_range),
        "allowed_actions": list(allowed_actions),
        "forced_reason": forced_reason,
        "action": decision["action"],
        "target_chunk_id": target.chunk_id,
        "target_revision": target.revision,
        "title": target.title,
        "summary": target.summary,
        "reason": decision.get("reason", ""),
        "decision_source": decision.get("decision_source", "llm"),
    }
    for field_name in ("fallback_error", "recovery_error"):
        if decision.get(field_name):
            event[field_name] = decision[field_name]
    state_events.append(event)


async def _finalize_chunks(
    *,
    text: str,
    sentences: list[SentenceSpan],
    managed_chunks: list[AgenticManagedChunk],
    min_sentences: int,
    max_sentences: int,
    llm_gateway: AgenticLlmGateway,
    state_events: list[dict[str, object]] | None,
) -> list[ChunkSpan]:
    original_boundaries = [
        (chunk.sentence_start, chunk.sentence_end)
        for chunk in managed_chunks
    ]
    validate_boundary_structure(
        original_boundaries,
        sentence_count=len(sentences),
    )
    final_boundaries = rebalance_document_boundaries(
        original_boundaries,
        sentence_count=len(sentences),
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    if final_boundaries != original_boundaries and state_events is not None:
        state_events.append({
            "event": "document_rebalance",
            "sentence_count": len(sentences),
            "original_boundaries": [list(item) for item in original_boundaries],
            "final_boundaries": [list(item) for item in final_boundaries],
        })

    metadata_by_boundary = {
        (chunk.sentence_start, chunk.sentence_end): (
            chunk.title,
            chunk.summary,
        )
        for chunk in managed_chunks
    }
    missing_boundaries = [
        boundary
        for boundary in final_boundaries
        if boundary not in metadata_by_boundary
    ]
    if missing_boundaries:
        refreshed = await llm_gateway.describe_chunks(
            text=text,
            sentences=sentences,
            boundaries=missing_boundaries,
        )
        metadata_by_boundary.update(refreshed)

    return [
        _make_chunk_span(
            text=text,
            sentences=sentences,
            boundary=boundary,
            metadata=metadata_by_boundary[boundary],
        )
        for boundary in final_boundaries
    ]


def _make_chunk_span(
    *,
    text: str,
    sentences: list[SentenceSpan],
    boundary: tuple[int, int],
    metadata: tuple[str, str],
) -> ChunkSpan:
    start, end = boundary
    char_start = sentences[start - 1].char_start
    char_end = sentences[end - 1].char_end
    title, summary = metadata
    return ChunkSpan(
        text=text[char_start:char_end],
        char_start=char_start,
        char_end=char_end,
        title=title,
        summary=summary,
    )


def _build_state_payload(
    *,
    text: str,
    sentences: list[SentenceSpan],
    managed_chunks: list[AgenticManagedChunk],
    proposition_range: tuple[int, int],
    proposition_index: int,
    allowed_actions: tuple[str, ...],
    forced_reason: str | None,
    min_sentences: int,
    max_sentences: int,
) -> dict[str, object]:
    start, end = proposition_range
    catalog_chunks = managed_chunks[-AGENTIC_CATALOG_MAX_CHUNKS:]
    catalog = [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "summary": chunk.summary,
            "sentence_count": chunk.sentence_count,
            "revision": chunk.revision,
            "status": "open" if index == len(catalog_chunks) - 1 else "closed",
        }
        for index, chunk in enumerate(catalog_chunks)
    ]
    current = managed_chunks[-1] if managed_chunks else None
    recent_propositions: list[str] = []
    if current is not None:
        for recent_start, recent_end in current.proposition_ranges[
            -AGENTIC_RECENT_PROPOSITIONS:
        ]:
            recent_propositions.append(text[
                sentences[recent_start - 1].char_start:
                sentences[recent_end - 1].char_end
            ].strip())

    return {
        "chunk_catalog": catalog,
        "open_chunk_recent_propositions": recent_propositions,
        "new_proposition": {
            "index": proposition_index,
            "sentence_start": start,
            "sentence_end": end,
            "text": text[
                sentences[start - 1].char_start:
                sentences[end - 1].char_end
            ].strip(),
        },
        "constraints": {
            "minimum_chunk_sentences": min_sentences,
            "maximum_chunk_sentences": max_sentences,
            "allowed_actions": list(allowed_actions),
            "forced_reason": forced_reason,
        },
    }


def _validate_runtime_config(
    *,
    batch_max_sentences: int,
    batch_max_chars: int,
    min_sentences: int,
    max_sentences: int,
    concurrency: int,
    retries: int,
) -> None:
    if batch_max_sentences <= 0:
        raise ValueError("agentic batch max sentences must be positive")
    if batch_max_chars <= 0:
        raise ValueError("agentic batch max chars must be positive")
    if min_sentences <= 0:
        raise ValueError("agentic min sentences must be positive")
    if max_sentences < min_sentences:
        raise ValueError(
            "agentic max sentences must be greater than or equal to min sentences"
        )
    if concurrency <= 0:
        raise ValueError("agentic concurrency must be positive")
    if retries < 0:
        raise ValueError("agentic retries must be non-negative")
