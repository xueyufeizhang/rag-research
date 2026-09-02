import asyncio
import json
import re
from typing import Awaitable, Callable

from json_repair import repair_json

from rag_research.agentic_boundaries import (
    project_boundaries,
    validate_boundary_structure,
)
from rag_research.chunking_models import SentenceSpan
from rag_research.prompts import (
    AGENTIC_METADATA_SYSTEM_PROMPT,
    AGENTIC_PROPOSITION_SYSTEM_PROMPT,
    AGENTIC_STATE_SYSTEM_PROMPT,
    build_agentic_metadata_prompt,
    build_agentic_proposition_prompt,
    build_agentic_state_prompt,
)


AGENTIC_TITLE_MAX_CHARS = 120
AGENTIC_SUMMARY_MAX_CHARS = 600


class AgenticLlmGateway:
    """Own all Agentic Chunking LLM calls and response recovery policy."""

    def __init__(
        self,
        *,
        llm_func: Callable[..., Awaitable[str]],
        retries: int,
        concurrency: int,
    ) -> None:
        self._llm_func = llm_func
        self._retries = retries
        self._concurrency = concurrency

    async def extract_propositions(
        self,
        *,
        sentences: list[SentenceSpan],
        batch_max_sentences: int,
        batch_max_chars: int,
        max_sentences: int,
        state_events: list[dict[str, object]] | None,
    ) -> list[tuple[int, int]]:
        batches = make_sentence_batches(
            sentences,
            max_sentences=batch_max_sentences,
            max_chars=batch_max_chars,
        )
        semaphore = asyncio.Semaphore(self._concurrency)
        batch_offsets: list[int] = []
        offset = 0
        for batch in batches:
            batch_offsets.append(offset)
            offset += len(batch)

        async def process_batch(
            batch_index: int,
            batch: list[SentenceSpan],
            global_offset: int,
        ) -> tuple[list[tuple[int, int]], dict[str, object] | None]:
            numbered_text = "\n".join(
                f"[S{index}] {sentence.text.strip()}"
                for index, sentence in enumerate(batch, start=1)
            )
            prompt = build_agentic_proposition_prompt(
                numbered_text,
                max_sentences=max_sentences,
            )
            last_error: Exception | None = None
            async with semaphore:
                for _ in range(self._retries + 1):
                    try:
                        response = await self._llm_func(
                            system=AGENTIC_PROPOSITION_SYSTEM_PROMPT,
                            prompt=prompt,
                        )
                        proposed = _parse_named_boundaries(
                            response,
                            field_name="propositions",
                        )
                        validate_boundary_structure(proposed, len(batch))
                        projected = project_boundaries(
                            proposed,
                            sentence_count=len(batch),
                            min_sentences=1,
                            max_sentences=max_sentences,
                        )
                        event: dict[str, object] | None = None
                        if proposed != projected:
                            event = {
                                "event": "proposition_projection",
                                "batch_index": batch_index,
                                "sentence_count": len(batch),
                                "original_boundaries": [
                                    list(item) for item in proposed
                                ],
                                "final_boundaries": [
                                    list(item) for item in projected
                                ],
                            }
                        return (
                            [
                                (start + global_offset, end + global_offset)
                                for start, end in projected
                            ],
                            event,
                        )
                    except Exception as exc:
                        last_error = exc
                        prompt += _retry_instruction(exc)
            raise RuntimeError(
                f"agentic proposition batch {batch_index} failed after "
                f"{self._retries + 1} attempts"
            ) from last_error

        results = await asyncio.gather(*(
            process_batch(index, batch, batch_offsets[index - 1])
            for index, batch in enumerate(batches, start=1)
        ))
        boundaries = [
            boundary
            for batch_boundaries, _ in results
            for boundary in batch_boundaries
        ]
        if state_events is not None:
            state_events.extend(
                event for _, event in results if event is not None
            )
        validate_boundary_structure(boundaries, len(sentences))
        return boundaries

    async def decide_transition(
        self,
        *,
        proposition_index: int,
        state_payload: dict[str, object],
        allowed_actions: tuple[str, ...],
        forced_reason: str | None,
        text: str,
        sentences: list[SentenceSpan],
        proposition_range: tuple[int, int],
        open_chunk_start: int | None,
        fallback_title: str | None,
    ) -> dict[str, str]:
        forced_action = allowed_actions[0] if len(allowed_actions) == 1 else None
        prompt = build_agentic_state_prompt(
            state_payload,
            allowed_actions=allowed_actions,
            title_max_chars=AGENTIC_TITLE_MAX_CHARS,
            summary_max_chars=AGENTIC_SUMMARY_MAX_CHARS,
        )
        last_error: Exception | None = None
        recoverable_action = forced_action
        attempt_count = self._retries + 1

        for attempt in range(1, attempt_count + 1):
            try:
                response = await self._llm_func(
                    system=AGENTIC_STATE_SYSTEM_PROMPT,
                    prompt=prompt,
                )
            except Exception as exc:
                last_error = exc
                _log_retry(
                    stage=f"state transition {proposition_index} request",
                    attempt=attempt,
                    attempt_count=attempt_count,
                    error=exc,
                )
                continue

            try:
                return _parse_state_decision(
                    response,
                    allowed_actions=allowed_actions,
                    forced_action=forced_action,
                    forced_reason=forced_reason,
                )
            except Exception as exc:
                last_error = exc
                try:
                    recoverable_action = _parse_state_action(
                        response,
                        allowed_actions=allowed_actions,
                        forced_action=forced_action,
                    )
                except Exception:
                    pass
                _log_retry(
                    stage=(
                        f"state transition {proposition_index} "
                        "output validation"
                    ),
                    attempt=attempt,
                    attempt_count=attempt_count,
                    error=exc,
                )
                prompt += _retry_instruction(exc)

        if recoverable_action is None:
            raise RuntimeError(
                f"agentic state transition {proposition_index} failed after "
                f"{attempt_count} attempts"
            ) from last_error

        metadata_start, metadata_end = proposition_range
        if recoverable_action == "append":
            if open_chunk_start is None:
                raise RuntimeError(
                    "cannot repair append metadata without an open chunk"
                )
            metadata_start = open_chunk_start
        return await self._repair_transition_metadata(
            proposition_index=proposition_index,
            action=recoverable_action,
            forced_reason=forced_reason,
            text=text,
            sentences=sentences,
            boundary=(metadata_start, metadata_end),
            fallback_title=(
                fallback_title if recoverable_action == "append" else None
            ),
            attempt_count=attempt_count,
            last_error=last_error,
        )

    async def describe_chunk(
        self,
        *,
        text: str,
        sentences: list[SentenceSpan],
        boundary: tuple[int, int],
        fallback_title: str | None = None,
    ) -> dict[str, str]:
        start, end = boundary
        chunk_text = text[
            sentences[start - 1].char_start:
            sentences[end - 1].char_end
        ].strip()
        prompt = build_agentic_metadata_prompt(
            chunk_text,
            title_max_chars=AGENTIC_TITLE_MAX_CHARS,
            summary_max_chars=AGENTIC_SUMMARY_MAX_CHARS,
        )
        last_error: Exception | None = None
        attempt_count = self._retries + 1
        for attempt in range(1, attempt_count + 1):
            try:
                response = await self._llm_func(
                    system=AGENTIC_METADATA_SYSTEM_PROMPT,
                    prompt=prompt,
                )
                title, summary = _validate_metadata(
                    _parse_json_object(response)
                )
                return {
                    "title": title,
                    "summary": summary,
                    "decision_source": "llm",
                }
            except Exception as exc:
                last_error = exc
                _log_retry(
                    stage=f"metadata refresh for sentences {start}-{end}",
                    attempt=attempt,
                    attempt_count=attempt_count,
                    error=exc,
                )
                prompt += _retry_instruction(exc)

        title, summary = _fallback_metadata(
            text=text,
            sentences=sentences,
            boundary=boundary,
            preferred_title=fallback_title,
        )
        print(
            f"[agentic] metadata refresh for sentences {start}-{end} using "
            f"source-derived fallback after {attempt_count} failed attempts "
            f"({type(last_error).__name__}: {last_error})",
            flush=True,
        )
        return {
            "title": title,
            "summary": summary,
            "decision_source": "fallback",
            "fallback_error": (
                f"{type(last_error).__name__}: {last_error}"
                if last_error is not None
                else "unknown metadata error"
            ),
        }

    async def describe_chunks(
        self,
        *,
        text: str,
        sentences: list[SentenceSpan],
        boundaries: list[tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[str, str]]:
        print(
            f"[agentic] refreshing metadata for {len(boundaries)} "
            "rebalanced chunks",
            flush=True,
        )
        semaphore = asyncio.Semaphore(self._concurrency)

        async def describe(
            boundary: tuple[int, int],
        ) -> tuple[tuple[int, int], tuple[str, str]]:
            async with semaphore:
                metadata = await self.describe_chunk(
                    text=text,
                    sentences=sentences,
                    boundary=boundary,
                )
            return boundary, (metadata["title"], metadata["summary"])

        return dict(await asyncio.gather(*(
            describe(boundary) for boundary in boundaries
        )))

    async def _repair_transition_metadata(
        self,
        *,
        proposition_index: int,
        action: str,
        forced_reason: str | None,
        text: str,
        sentences: list[SentenceSpan],
        boundary: tuple[int, int],
        fallback_title: str | None,
        attempt_count: int,
        last_error: Exception | None,
    ) -> dict[str, str]:
        error_text = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "unknown state response error"
        )
        print(
            f"[agentic] state transition {proposition_index} repairing metadata "
            f"with a dedicated prompt after {attempt_count} failed state "
            f"responses ({error_text})",
            flush=True,
        )
        metadata = await self.describe_chunk(
            text=text,
            sentences=sentences,
            boundary=boundary,
            fallback_title=fallback_title,
        )
        decision_source = (
            "fallback"
            if metadata.get("decision_source") == "fallback"
            else "metadata_repair"
        )
        recovered = {
            "action": action,
            "title": metadata["title"],
            "summary": metadata["summary"],
            "reason": forced_reason or (
                "source-derived metadata fallback after invalid state output"
                if decision_source == "fallback"
                else "metadata repaired after invalid state output"
            ),
            "decision_source": decision_source,
            "recovery_error": error_text,
        }
        if decision_source == "fallback":
            recovered["fallback_error"] = metadata.get(
                "fallback_error",
                error_text,
            )
        return recovered


def make_sentence_batches(
    sentences: list[SentenceSpan],
    max_sentences: int,
    max_chars: int,
) -> list[list[SentenceSpan]]:
    """Group source sentences for bounded proposition-extraction calls."""
    if max_sentences <= 0:
        raise ValueError("max sentences must be positive")
    if max_chars <= 0:
        raise ValueError("max chars must be positive")

    batches: list[list[SentenceSpan]] = []
    current: list[SentenceSpan] = []
    current_chars = 0
    for sentence_span in sentences:
        sentence_chars = len(sentence_span.text.strip())
        exceeds_sentence_limit = len(current) >= max_sentences
        exceeds_char_limit = (
            bool(current) and current_chars + sentence_chars > max_chars
        )
        if exceeds_sentence_limit or exceeds_char_limit:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(sentence_span)
        current_chars += sentence_chars

    if current:
        batches.append(current)
    return batches


def _parse_state_decision(
    response: str,
    *,
    allowed_actions: tuple[str, ...],
    forced_action: str | None,
    forced_reason: str | None,
) -> dict[str, str]:
    payload = _parse_json_object(response)
    action = _validate_state_action(
        payload,
        allowed_actions=allowed_actions,
        forced_action=forced_action,
    )
    title, summary = _validate_metadata(payload)
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        raise ValueError("agentic state reason must be a string")
    return {
        "action": action,
        "title": title,
        "summary": summary,
        "reason": forced_reason or reason.strip(),
    }


def _parse_state_action(
    response: str,
    *,
    allowed_actions: tuple[str, ...],
    forced_action: str | None,
) -> str:
    payload = {} if forced_action is not None else _parse_json_object(response)
    return _validate_state_action(
        payload,
        allowed_actions=allowed_actions,
        forced_action=forced_action,
    )


def _validate_state_action(
    payload: dict[str, object],
    *,
    allowed_actions: tuple[str, ...],
    forced_action: str | None,
) -> str:
    if forced_action is not None:
        if allowed_actions != (forced_action,):
            raise ValueError("forced action must be the only allowed action")
        return forced_action

    action = payload.get("action")
    if not isinstance(action, str):
        raise ValueError("agentic state response requires a string action")
    action = action.strip().lower()
    if action not in allowed_actions:
        raise ValueError(
            f"action {action!r} is not allowed; expected one of "
            f"{allowed_actions}"
        )
    return action


def _validate_metadata(payload: dict[str, object]) -> tuple[str, str]:
    title = payload.get("title")
    summary = payload.get("summary")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("agentic metadata requires a non-empty title")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("agentic metadata requires a non-empty summary")
    title = title.strip()
    summary = summary.strip()
    if len(title) > AGENTIC_TITLE_MAX_CHARS:
        raise ValueError(
            f"agentic title exceeds {AGENTIC_TITLE_MAX_CHARS} characters"
        )
    if len(summary) > AGENTIC_SUMMARY_MAX_CHARS:
        raise ValueError(
            f"agentic summary exceeds {AGENTIC_SUMMARY_MAX_CHARS} characters"
        )
    return title, summary


def _fallback_metadata(
    *,
    text: str,
    sentences: list[SentenceSpan],
    boundary: tuple[int, int],
    preferred_title: str | None,
) -> tuple[str, str]:
    start, end = boundary
    source_text = text[
        sentences[start - 1].char_start:
        sentences[end - 1].char_end
    ]
    title = preferred_title or sentences[start - 1].text
    return (
        _truncate_metadata(title, AGENTIC_TITLE_MAX_CHARS),
        _truncate_metadata(source_text, AGENTIC_SUMMARY_MAX_CHARS),
    )


def _truncate_metadata(value: str, limit: int) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        return "Source-aligned chunk"
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit - 1].rstrip() + "…"


def _parse_named_boundaries(
    response: str,
    *,
    field_name: str,
) -> list[tuple[int, int]]:
    payload = _parse_json_object(response)
    raw_boundaries = payload.get(field_name)
    if not isinstance(raw_boundaries, list) or not raw_boundaries:
        raise ValueError(
            f"response does not contain a non-empty {field_name} list"
        )

    boundaries: list[tuple[int, int]] = []
    for item in raw_boundaries:
        if not isinstance(item, dict):
            raise ValueError(f"each {field_name} boundary must be an object")
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"each {field_name} boundary requires integer start and end "
                "indexes"
            ) from exc
        boundaries.append((start, end))
    return boundaries


def _parse_json_object(response: str) -> dict[str, object]:
    repaired = repair_json(_extract_json_object(response))
    payload = json.loads(repaired)
    if not isinstance(payload, dict):
        raise ValueError("agentic response must be a JSON object")
    return payload


def _extract_json_object(response: str) -> str:
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    return response[start:end + 1]


def _retry_instruction(error: Exception) -> str:
    return (
        "\n\nYour previous response was invalid. "
        f"Validation error: {error}. Return corrected JSON only."
    )


def _log_retry(
    *,
    stage: str,
    attempt: int,
    attempt_count: int,
    error: Exception,
) -> None:
    if attempt >= attempt_count:
        return
    print(
        f"[agentic] {stage} failed on attempt {attempt}/{attempt_count} "
        f"({type(error).__name__}: {error}); retrying",
        flush=True,
    )
