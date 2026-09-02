from dataclasses import dataclass
import asyncio
from fractions import Fraction
from functools import lru_cache
import json
import re
import pysbd
from typing import Awaitable, Callable, Optional

from json_repair import repair_json

from rag_research.embedding import (
    BatchEmbeddingFunction,
    EmbeddingFunction,
    embed_texts,
)


CHUNKING_PIPELINE_VERSION = 4

AGENTIC_CHUNKING_SYSTEM_PROMPT = """
You are a document segmentation assistant. Your only task is to identify
topic-coherent boundaries in numbered sentences. Return strict JSON containing
inclusive start and end sentence indexes.
""".strip()

_SENTENCE_SEGMENTER = pysbd.Segmenter(
    language="en",
    clean=False,
    char_span=True,
)


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str = "fixed"
    fixed_size: int = 2400
    fixed_overlap: int = 200
    sentence_window_size: int = 8
    sentence_window_overlap: int = 2
    semantic_breakpoint_percentile: float = 90.0
    semantic_min_sentences: int = 8
    semantic_max_sentences: int = 24
    semantic_buffer_size: int = 1
    semantic_embedding_batch_size: int = 32
    semantic_embedding_concurrency: int = 4
    agentic_batch_max_sentences: int = 60
    agentic_batch_max_chars: int = 12000
    agentic_min_sentences: int = 4
    agentic_max_sentences: int = 20
    agentic_concurrency: int = 4
    agentic_retries: int = 2

@dataclass(frozen=True)
class ChunkSpan:
    text: str
    char_start: int
    char_end: int

@dataclass(frozen=True)
class SentenceSpan:
    text: str
    char_start: int
    char_end: int

# def chunk(text: str, config: ChunkConfig) -> list[ChunkSpan]:
#     strategy = config.strategy.lower()
#     if strategy == "fixed":
#         return fixed_size_chunk(text, config.fixed_size, config.fixed_overlap)
#     if strategy == "semantic":
#         raise ValueError("semantic chunking requires embeddings; use chunk_async instead")
#     if strategy == "agentic_ibm":
#         raise ValueError("agentic_ibm chunking requires an LLM; use chunk_async instead")
#     raise ValueError(f"unknown chunking strategy: {config.strategy}")


async def chunk_async(
    text: str,
    config: ChunkConfig,
    embed_func: Optional[EmbeddingFunction] = None,
    embed_many_func: Optional[BatchEmbeddingFunction] = None,
    llm_func: Optional[Callable[..., Awaitable[str]]] = None,
    agentic_projection_events: list[dict[str, object]] | None = None,
) -> list[ChunkSpan]:
    strategy = config.strategy.lower()
    if strategy == "fixed":
        return fixed_size_chunk(text, config.fixed_size, config.fixed_overlap)
    if strategy == "semantic":
        if embed_func is None:
            raise ValueError("semantic chunking requires an embed_func")
        return await semantic_chunk(
            text,
            breakpoint_percentile=config.semantic_breakpoint_percentile,
            min_sentences=config.semantic_min_sentences,
            max_sentences=config.semantic_max_sentences,
            buffer_size=config.semantic_buffer_size,
            embedding_batch_size=config.semantic_embedding_batch_size,
            embedding_concurrency=config.semantic_embedding_concurrency,
            embed_func=embed_func,
            embed_many_func=embed_many_func,
        )
    if strategy == "agentic_ibm":
        if llm_func is None:
            raise ValueError("agentic_ibm chunking requires an llm_func")
        return await agentic_ibm_chunk(
            text=text,
            batch_max_sentences=config.agentic_batch_max_sentences,
            batch_max_chars=config.agentic_batch_max_chars,
            min_sentences=config.agentic_min_sentences,
            max_sentences=config.agentic_max_sentences,
            concurrency=config.agentic_concurrency,
            retries=config.agentic_retries,
            llm_func=llm_func,
            projection_events=agentic_projection_events,
        )
    raise ValueError(f"unknown chunking strategy: {config.strategy}")


def fixed_size_chunk(text: str, size: int, overlap: int) -> list[ChunkSpan]:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than chunk size ({size}), "
                          f"otherwise chunking never advances")

    chunks: list[ChunkSpan] = []
    start = 0
    text_length = len(text)
    step = size - overlap

    while start < text_length:
        end = min(start + size, text_length)
        chunks.append(
            ChunkSpan(
                text=text[start:end],
                char_start=start,
                char_end=end,
            )
        )
        if end == text_length:
            break
        start += step
    return chunks


# def sentence_window_chunk(text: str, window_size: int, window_overlap: int) -> list[ChunkSpan]:
#     if window_size <= 0:
#         raise ValueError(f"sentence window size must be positive, got {window_size}")
#     if window_overlap < 0:
#         raise ValueError(f"sentence window overlap must be non-negative, got {window_overlap}")
#     if window_overlap >= window_size:
#         raise ValueError("sentence window overlap must be smaller than window size")

#     sentences = _split_sentences(text)
#     if not sentences:
#         return []
#     if len(sentences) <= window_size:
#         char_start = sentences[0].char_start
#         char_end = sentences[-1].char_end
#         return [
#             ChunkSpan(
#                 text=text[char_start:char_end],
#                 char_start=char_start,
#                 char_end=char_end,
#             )
#         ]

#     step = window_size - window_overlap
#     starts = list(range(0, len(sentences) - window_size + 1, step))
#     last_start = len(sentences) - window_size
#     if starts[-1] != last_start:
#         starts.append(last_start)

#     chunks: list[ChunkSpan] = []
#     for window_start in starts:
#         window_end = window_start + window_size
#         selected = sentences[window_start:window_end]
#         char_start = selected[0].char_start
#         char_end = selected[-1].char_end
#         chunks.append(
#             ChunkSpan(
#                 text=text[char_start:char_end],
#                 char_start=char_start,
#                 char_end=char_end,
#             )
#         )
#     return chunks


async def semantic_chunk(
    text: str,
    breakpoint_percentile: float,
    min_sentences: int,
    max_sentences: int,
    buffer_size: int,
    embedding_concurrency: int,
    embed_func: EmbeddingFunction,
    embedding_batch_size: int = 32,
    embed_many_func: BatchEmbeddingFunction | None = None,
) -> list[ChunkSpan]:
    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError(f"semantic breakpoint percentile must be between 0 and 100, got {breakpoint_percentile}")
    if min_sentences <= 0:
        raise ValueError(f"semantic min sentences must be positive, got {min_sentences}")
    if max_sentences < min_sentences:
        raise ValueError("semantic max sentences must be greater than or equal to min sentences")
    if buffer_size < 0:
        raise ValueError(f"semantic buffer size must be non-negative, got {buffer_size}")
    if embedding_batch_size <= 0:
        raise ValueError(
            f"semantic embedding batch size must be positive, got {embedding_batch_size}"
        )
    if embedding_concurrency <= 0:
        raise ValueError(f"semantic embedding concurrency must be positive, got {embedding_concurrency}")

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        char_start = sentences[0].char_start
        char_end = sentences[-1].char_end
        return [
            ChunkSpan(
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
            )
        ]

    embedding_inputs = [
        _sentence_context(text, sentences, idx, buffer_size)
        for idx in range(len(sentences))
    ]
    embeddings = await embed_texts(
        embedding_inputs,
        embed_func=embed_func,
        embed_many_func=embed_many_func,
        batch_size=embedding_batch_size,
        concurrency=embedding_concurrency,
    )
    distances = [
        1.0 - _cosine_similarity(embeddings[idx], embeddings[idx + 1])
        for idx in range(len(embeddings) - 1)
    ]
    threshold = _percentile(distances, breakpoint_percentile)

    chunks:list[ChunkSpan] = []
    start = 0
    for idx, distance in enumerate(distances):
        sentence_count = idx - start + 1
        should_split = sentence_count >= min_sentences and distance >= threshold
        must_split = sentence_count >= max_sentences
        if should_split or must_split:
            split_start = sentences[start].char_start
            split_end = sentences[idx].char_end
            chunks.append(
                ChunkSpan(
                    text=text[split_start:split_end],
                    char_start=split_start,
                    char_end=split_end,
                )
            )
            start = idx + 1

    if start < len(sentences):
        tail = sentences[start:]
        if chunks and len(tail) < min_sentences:
            split_start = chunks[-1].char_start
            split_end = tail[-1].char_end
            chunks[-1] = ChunkSpan(
                text=text[split_start:split_end],
                char_start=split_start,
                char_end=split_end,
            )
        else:
            split_start = tail[0].char_start
            split_end = tail[-1].char_end
            chunks.append(
                ChunkSpan(
                    text=text[split_start:split_end],
                    char_start=split_start,
                    char_end=split_end,
                )
            )

    return chunks


async def agentic_ibm_chunk(
    text: str,
    batch_max_sentences: int,
    batch_max_chars: int,
    min_sentences: int,
    max_sentences: int,
    concurrency: int,
    retries: int,
    llm_func: Callable[..., Awaitable[str]],
    projection_events: list[dict[str, object]] | None = None,
) -> list[ChunkSpan]:
    """Use an LLM to choose extractive, contiguous topic boundaries.

    The model only returns sentence indexes. Text is reconstructed from the
    original sentences so chunks remain source-aligned for canonical evidence
    evaluation.
    """
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

    sentences = _split_sentences(text)
    if not sentences:
        return []

    batches = _make_sentence_batches(
        sentences,
        max_sentences=batch_max_sentences,
        max_chars=batch_max_chars,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def process_batch(
        text: str,
        batch_index: int,
        batch: list[SentenceSpan],
    ) -> list[ChunkSpan]:
        prompt = _build_agentic_prompt(
            batch,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
        )
        last_error: Optional[Exception] = None

        async with semaphore:
            for _ in range(retries + 1):
                try:
                    response = await llm_func(
                        system=AGENTIC_CHUNKING_SYSTEM_PROMPT,
                        prompt=prompt,
                    )
                    proposed_boundaries = _parse_agentic_boundaries(response)
                    _validate_agentic_boundary_structure(
                        proposed_boundaries,
                        sentence_count=len(batch),
                    )
                    boundaries = _project_agentic_boundaries(
                        proposed_boundaries,
                        sentence_count=len(batch),
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
                    )
                    _validate_agentic_boundaries(
                        boundaries,
                        sentence_count=len(batch),
                        min_sentences=min_sentences,
                        max_sentences=max_sentences,
                    )
                    if boundaries != proposed_boundaries:
                        event = {
                            "scope": "batch",
                            "batch_index": batch_index,
                            "sentence_count": len(batch),
                            "original_boundaries": [
                                [start, end]
                                for start, end in proposed_boundaries
                            ],
                            "projected_boundaries": [
                                [start, end]
                                for start, end in boundaries
                            ],
                        }
                        if projection_events is not None:
                            projection_events.append(event)
                        print(
                            f"[agentic chunking] batch {batch_index}: "
                            "projected model boundaries onto configured "
                            "sentence limits",
                            flush=True,
                        )
                    return _reconstruct_chunks(text, batch, boundaries)
                except Exception as exc:
                    last_error = exc
                    prompt += (
                        "\n\nYour previous response was invalid. "
                        f"Validation error: {exc}. Return corrected JSON only."
                    )

        raise RuntimeError(
            f"agentic chunking batch {batch_index} failed after "
            f"{retries + 1} attempts"
        ) from last_error

    batch_results = await asyncio.gather(*(
        process_batch(text, index, batch)
        for index, batch in enumerate(batches, start=1)
    ))
    batch_chunks = [
        chunk
        for batch in batch_results
        for chunk in batch
    ]

    document_boundaries = _chunk_spans_to_boundaries(
        sentences,
        batch_chunks,
    )
    allow_short_document_final = not _strict_partition_is_feasible(
        len(sentences),
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    rebalanced_boundaries = _rebalance_agentic_document_boundaries(
        document_boundaries,
        sentence_count=len(sentences),
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    _validate_agentic_boundaries(
        rebalanced_boundaries,
        sentence_count=len(sentences),
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        allow_short_final=allow_short_document_final,
    )
    if rebalanced_boundaries != document_boundaries:
        if projection_events is not None:
            projection_events.append({
                "scope": "document",
                "batch_index": 0,
                "sentence_count": len(sentences),
                "original_boundaries": [
                    [start, end]
                    for start, end in document_boundaries
                ],
                "projected_boundaries": [
                    [start, end]
                    for start, end in rebalanced_boundaries
                ],
            })
        print(
            "[agentic chunking] document: rebalanced short "
            "macro-batch tails onto document-level sentence limits",
            flush=True,
        )

    return _reconstruct_chunks(
        text,
        sentences,
        rebalanced_boundaries,
    )


def _make_sentence_batches(
        sentences: list[SentenceSpan],
        max_sentences: int,
        max_chars: int,
) -> list[list[SentenceSpan]]:
    if max_sentences <= 0:
        raise ValueError("max sentences must be positive")
    if max_chars <= 0:
        raise ValueError("max chars must be positive")

    batches = []
    current = []
    current_chars = 0

    for sentence_span in sentences:
        sentence = sentence_span.text.strip()
        additional_chars = len(sentence)
        exceeds_sentence_limit = len(current) >= max_sentences
        exceeds_char_limit = bool(current) and current_chars + additional_chars > max_chars

        if exceeds_sentence_limit or exceeds_char_limit:
            batches.append(current)
            current = []
            current_chars = 0

        current.append(sentence_span)
        current_chars += additional_chars

    if current:
        batches.append(current)
    return batches


def _build_agentic_prompt(
    sentences: list[SentenceSpan],
    min_sentences: int,
    max_sentences: int,
) -> str:
    numbered_text = "\n".join(
        f"[S{index}] {sentence.text.strip()}"
        for index, sentence in enumerate(sentences, start=1)
    )
    return f"""
    Divide the numbered sentences into contiguous, semantically coherent topic chunks.

    Rules:
    1. Preserve the original sentence order.
    2. Every sentence must appear in exactly one chunk.
    3. Chunks must be contiguous.
    4. Do not omit or duplicate sentences.
    5. Every non-final chunk MUST contain at least {min_sentences} sentences.
    6. No chunk may contain more than {max_sentences} sentences.
    7. Only the final chunk may contain fewer than {min_sentences} sentences.
    8. Return JSON only.
    9. Do not reproduce, summarize, rewrite, or correct the sentences.

    Required format:
    {{
    "chunks": [
        {{"start": 1, "end": 5}},
        {{"start": 6, "end": 12}}
    ]
    }}

    Numbered sentences:
    {numbered_text}
    """.strip()


def _extract_json_object(response: str) -> str:
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    return response[start:end + 1]


def _parse_agentic_boundaries(response: str) -> list[tuple[int, int]]:
    repaired = repair_json(_extract_json_object(response))
    payload = json.loads(repaired)
    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("response does not contain a non-empty chunks list")

    boundaries = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            raise ValueError("each chunk boundary must be an object")
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each chunk requires integer start and end indexes") from exc
        boundaries.append((start, end))
    return boundaries


def _validate_agentic_boundaries(
    boundaries: list[tuple[int, int]],
    sentence_count: int,
    min_sentences: int,
    max_sentences: int,
    allow_short_final: bool = True,
) -> None:
    if min_sentences <= 0:
        raise ValueError("agentic min sentences must be positive")
    if max_sentences < min_sentences:
        raise ValueError(
            "agentic max sentences must be greater than or equal to min sentences"
        )
    _validate_agentic_boundary_structure(boundaries, sentence_count)

    for index, (start, end) in enumerate(boundaries):
        size = end - start + 1
        is_final = index == len(boundaries) - 1
        if size > max_sentences:
            raise ValueError(f"chunk size {size} exceeds maximum {max_sentences}")
        if size < min_sentences and (
            not is_final or not allow_short_final
        ):
            raise ValueError(
                f"chunk size {size} is below minimum {min_sentences}"
            )


def _validate_agentic_boundary_structure(
    boundaries: list[tuple[int, int]],
    sentence_count: int,
) -> None:
    if sentence_count <= 0:
        raise ValueError("sentence count must be positive")
    if not boundaries:
        raise ValueError("no chunk boundaries returned")

    expected_start = 1
    for start, end in boundaries:
        if start != expected_start:
            raise ValueError(f"expected chunk to start at {expected_start}, got {start}")
        if start < 1 or end < start or end > sentence_count:
            raise ValueError(
                f"invalid boundary ({start}, {end}) for {sentence_count} sentences"
            )

        expected_start = end + 1

    if boundaries[-1][1] != sentence_count:
        raise ValueError(
            f"last chunk ends at {boundaries[-1][1]}, expected {sentence_count}"
        )


def _project_agentic_boundaries(
    boundaries: list[tuple[int, int]],
    sentence_count: int,
    min_sentences: int,
    max_sentences: int,
    allow_short_final: bool = True,
) -> list[tuple[int, int]]:
    """Project structurally valid model boundaries onto hard size limits.

    The projection first preserves the model's chunk count whenever that count
    admits a legal partition. It then moves cumulative boundaries by the
    smallest total number of sentence positions. If the original chunk count
    is infeasible, the closest feasible count is used and semantic targets are
    interpolated from the model's cumulative boundaries.
    """
    if min_sentences <= 0:
        raise ValueError("agentic min sentences must be positive")
    if max_sentences < min_sentences:
        raise ValueError(
            "agentic max sentences must be greater than or equal to min sentences"
        )
    _validate_agentic_boundary_structure(boundaries, sentence_count)

    if all(
        (end - start + 1) <= max_sentences
        and (
            index == len(boundaries) - 1
            and allow_short_final
            or (end - start + 1) >= min_sentences
        )
        for index, (start, end) in enumerate(boundaries)
    ):
        return list(boundaries)

    proposed_chunk_count = len(boundaries)
    feasible_chunk_counts = [
        chunk_count
        for chunk_count in range(1, sentence_count + 1)
        if (
            (
                (chunk_count - 1) * min_sentences + 1
                if allow_short_final
                else chunk_count * min_sentences
            ) <= sentence_count
            and sentence_count <= chunk_count * max_sentences
        )
    ]
    if not feasible_chunk_counts:
        raise ValueError(
            "sentence count cannot be partitioned under the configured "
            "agentic minimum and maximum"
        )

    projected_chunk_count = min(
        feasible_chunk_counts,
        key=lambda count: (abs(count - proposed_chunk_count), count),
    )
    proposed_endpoints = [0, *(end for _, end in boundaries)]

    target_endpoints: list[Fraction] = []
    for boundary_index in range(1, projected_chunk_count):
        model_position = Fraction(
            boundary_index * proposed_chunk_count,
            projected_chunk_count,
        )
        left_index = model_position.numerator // model_position.denominator
        fraction = model_position - left_index
        left_endpoint = proposed_endpoints[left_index]
        right_endpoint = proposed_endpoints[left_index + 1]
        target_endpoints.append(
            Fraction(left_endpoint)
            + fraction * (right_endpoint - left_endpoint)
        )

    @lru_cache(maxsize=None)
    def solve(
        chunk_index: int,
        previous_end: int,
    ) -> tuple[Fraction, tuple[int, ...]] | None:
        remaining_chunks = projected_chunk_count - chunk_index
        if remaining_chunks == 1:
            final_size = sentence_count - previous_end
            minimum_final_size = 1 if allow_short_final else min_sentences
            if minimum_final_size <= final_size <= max_sentences:
                return Fraction(0), (sentence_count,)
            return None

        best: tuple[Fraction, tuple[int, ...]] | None = None
        earliest_end = previous_end + min_sentences
        latest_end = min(previous_end + max_sentences, sentence_count - 1)
        for current_end in range(earliest_end, latest_end + 1):
            chunks_after_current = remaining_chunks - 1
            remaining_sentences = sentence_count - current_end
            minimum_remaining = (
                (chunks_after_current - 1) * min_sentences + 1
                if allow_short_final
                else chunks_after_current * min_sentences
            )
            maximum_remaining = chunks_after_current * max_sentences
            if not minimum_remaining <= remaining_sentences <= maximum_remaining:
                continue

            tail = solve(chunk_index + 1, current_end)
            if tail is None:
                continue
            boundary_cost = abs(
                Fraction(current_end) - target_endpoints[chunk_index]
            )
            candidate = (
                boundary_cost + tail[0],
                (current_end, *tail[1]),
            )
            if best is None or candidate < best:
                best = candidate
        return best

    solution = solve(0, 0)
    if solution is None:
        raise ValueError(
            "failed to project agentic boundaries onto configured limits"
        )

    projected: list[tuple[int, int]] = []
    previous_end = 0
    for current_end in solution[1]:
        projected.append((previous_end + 1, current_end))
        previous_end = current_end
    return projected


def _strict_partition_is_feasible(
    sentence_count: int,
    *,
    min_sentences: int,
    max_sentences: int,
) -> bool:
    if sentence_count <= 0:
        return False
    return any(
        chunk_count * min_sentences <= sentence_count
        <= chunk_count * max_sentences
        for chunk_count in range(1, sentence_count + 1)
    )


def _rebalance_agentic_document_boundaries(
    boundaries: list[tuple[int, int]],
    *,
    sentence_count: int,
    min_sentences: int,
    max_sentences: int,
) -> list[tuple[int, int]]:
    """Remove artificial short tails while preserving semantic boundaries.

    Short chunks are first merged with the smallest adjacent chunk whenever
    the merged chunk remains within the maximum. If merging is impossible,
    the minimum number of sentences is borrowed from adjacent chunks. A full
    constrained projection is only used as a last resort.
    """
    _validate_agentic_boundary_structure(boundaries, sentence_count)
    if any(
        end - start + 1 > max_sentences
        for start, end in boundaries
    ):
        raise ValueError(
            "document-level rebalancing received an oversized chunk"
        )
    if not _strict_partition_is_feasible(
        sentence_count,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    ):
        return _project_agentic_boundaries(
            boundaries,
            sentence_count=sentence_count,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            allow_short_final=True,
        )

    endpoints = [end for _, end in boundaries]
    while True:
        previous_end = 0
        sizes: list[int] = []
        for endpoint in endpoints:
            sizes.append(endpoint - previous_end)
            previous_end = endpoint

        short_index = next(
            (
                index
                for index, size in enumerate(sizes)
                if size < min_sentences
            ),
            None,
        )
        if short_index is None:
            break

        merge_candidates: list[tuple[int, int, str]] = []
        if short_index > 0:
            merged_size = sizes[short_index - 1] + sizes[short_index]
            if merged_size <= max_sentences:
                merge_candidates.append((merged_size, 0, "left"))
        if short_index + 1 < len(sizes):
            merged_size = sizes[short_index] + sizes[short_index + 1]
            if merged_size <= max_sentences:
                merge_candidates.append((merged_size, 1, "right"))

        if merge_candidates:
            _, _, merge_side = min(merge_candidates)
            boundary_to_remove = (
                short_index - 1
                if merge_side == "left"
                else short_index
            )
            del endpoints[boundary_to_remove]
            continue

        needed = min_sentences - sizes[short_index]
        left_available = (
            sizes[short_index - 1] - min_sentences
            if short_index > 0
            else 0
        )
        right_available = (
            sizes[short_index + 1] - min_sentences
            if short_index + 1 < len(sizes)
            else 0
        )
        if left_available + right_available < needed:
            return _project_agentic_boundaries(
                boundaries,
                sentence_count=sentence_count,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                allow_short_final=False,
            )

        take_from_left = min(left_available, needed)
        take_from_right = needed - take_from_left
        if take_from_left:
            endpoints[short_index - 1] -= take_from_left
        if take_from_right:
            endpoints[short_index] += take_from_right

    rebalanced: list[tuple[int, int]] = []
    previous_end = 0
    for endpoint in endpoints:
        rebalanced.append((previous_end + 1, endpoint))
        previous_end = endpoint

    _validate_agentic_boundaries(
        rebalanced,
        sentence_count=sentence_count,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        allow_short_final=False,
    )
    return rebalanced


def _chunk_spans_to_boundaries(
    sentences: list[SentenceSpan],
    chunks: list[ChunkSpan],
) -> list[tuple[int, int]]:
    if not sentences or not chunks:
        raise ValueError("sentences and chunks must not be empty")

    sentence_start_indexes = {
        sentence.char_start: index
        for index, sentence in enumerate(sentences, start=1)
    }
    sentence_end_indexes = {
        sentence.char_end: index
        for index, sentence in enumerate(sentences, start=1)
    }
    boundaries: list[tuple[int, int]] = []
    for chunk in chunks:
        start = sentence_start_indexes.get(chunk.char_start)
        end = sentence_end_indexes.get(chunk.char_end)
        if start is None or end is None:
            raise ValueError("agentic chunk is not aligned to sentence boundaries")
        boundaries.append((start, end))

    _validate_agentic_boundary_structure(
        boundaries,
        sentence_count=len(sentences),
    )
    return boundaries


def _reconstruct_chunks(
        text: str,
        sentences: list[SentenceSpan],
        boundaries: list[tuple[int, int]],
) -> list[ChunkSpan]:
    reconstructed_chunks: list[ChunkSpan] = []
    for start, end in boundaries:
        chunk_start = sentences[start - 1].char_start
        chunk_end = sentences[end - 1].char_end
        reconstructed_chunks.append(
            ChunkSpan(
                text=text[chunk_start:chunk_end],
                char_start=chunk_start,
                char_end=chunk_end,
            )
        )
    return reconstructed_chunks


def _split_sentences(text: str) -> list[SentenceSpan]:
    if not text or not text.strip():
        return []

    raw_spans = _SENTENCE_SEGMENTER.segment(text)
    merged_intervals: list[list[int]] = []
    previous_start = -1

    for raw_span in raw_spans:
        start = int(raw_span.start)
        end = int(raw_span.end)

        if not (0 <= start <= end <= len(text)):
            raise ValueError(
                "Sentence segmenter returned an invalid span: "
                f"start={start}, end={end}, "
                f"text_length={len(text)}"
            )
        if start < previous_start:
            raise ValueError(
                "Sentence spans are out of order: "
                f"previous_start={previous_start}, "
                f"start={start}, end={end}"
            )
        previous_start = start

        if start == end or not text[start:end].strip():
            continue

        if merged_intervals and start < merged_intervals[-1][1]:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
        else:
            merged_intervals.append([start, end])

    if not merged_intervals:
        return [SentenceSpan(text=text, char_start=0, char_end=len(text))]

    boundaries = [0]
    boundaries.extend(interval[0] for interval in merged_intervals[1:])
    boundaries.append(len(text))

    sentences = [
        SentenceSpan(
            text=text[start:end],
            char_start=start,
            char_end=end,
        )
        for start, end in zip(boundaries, boundaries[1:])
        if start < end
    ]

    if "".join(sentence.text for sentence in sentences) != text:
        raise ValueError("Sentence spans do not reconstruct the original text")
    for left, right in zip(sentences, sentences[1:]):
        if left.char_end != right.char_start:
            raise ValueError("Sentence spans are not a continuous partition")

    return sentences


def _sentence_context(text: str, sentences: list[SentenceSpan], idx: int, buffer_size: int) -> str:
    start = max(0, idx - buffer_size)
    end = min(len(sentences), idx + buffer_size + 1)
    context_start = sentences[start].char_start
    context_end = sentences[end - 1].char_end
    return text[context_start:context_end].strip()


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
