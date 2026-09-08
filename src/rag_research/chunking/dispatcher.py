"""Public dispatcher for the supported document chunking strategies."""

from typing import Awaitable, Callable

from .agentic_chunking import AgenticProgressCallback, agentic_chunk
from .agentic_boundaries import rebalance_document_boundaries
from .chunking_models import ChunkConfig, ChunkSpan, SentenceSpan
from rag_research.embedding import (
    BatchEmbeddingFunction,
    EmbeddingFunction,
    embed_texts,
)
from rag_research.llm import LLMResponse
from .text_spans import sentence_context, split_sentences


CHUNKING_PIPELINE_VERSION = 6
CHUNKING_STRATEGY_VERSIONS = {"fixed": 6, "semantic": 7, "agentic": 7}
SEMANTIC_BOUNDARY_POLICY = "strict-percentile-rebalance-v2"
SEMANTIC_DISTANCE_ABS_TOLERANCE = 1e-12
UNIFORM_OVERLAP_POLICY = "source-prefix-after-boundaries-v1"


async def chunk_async(
    text: str,
    config: ChunkConfig,
    embed_func: EmbeddingFunction | None = None,
    embed_many_func: BatchEmbeddingFunction | None = None,
    llm_func: Callable[..., Awaitable[str | LLMResponse]] | None = None,
    agentic_state_events: list[dict[str, object]] | None = None,
    agentic_progress_callback: AgenticProgressCallback | None = None,
) -> list[ChunkSpan]:
    """Dispatch a strategy, then apply the one shared overlap postprocess."""
    strategy = config.strategy.lower()
    overlap = config.overlap_size
    if strategy == "fixed":
        spans = fixed_size_chunk(text, config.fixed_size, 0)
        return apply_uniform_overlap(text, spans, overlap, audit=agentic_state_events)
    if strategy == "semantic":
        if embed_func is None:
            raise ValueError("semantic chunking requires an embed_func")
        spans = await semantic_chunk(
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
        return apply_uniform_overlap(text, spans, overlap, audit=agentic_state_events)
    if strategy == "agentic":
        if llm_func is None:
            raise ValueError("agentic chunking requires an llm_func")
        spans = await agentic_chunk(
            text=text,
            batch_max_sentences=config.agentic_batch_max_sentences,
            batch_max_chars=config.agentic_batch_max_chars,
            min_sentences=config.agentic_min_sentences,
            max_sentences=config.agentic_max_sentences,
            concurrency=config.agentic_concurrency,
            retries=config.agentic_retries,
            llm_func=llm_func,
            state_events=agentic_state_events,
            progress_callback=agentic_progress_callback,
        )
        return apply_uniform_overlap(text, spans, overlap, audit=agentic_state_events)
    raise ValueError(f"unknown chunking strategy: {config.strategy}")


def fixed_size_chunk(text: str, size: int, overlap: int) -> list[ChunkSpan]:
    """Split text into fixed windows and apply the common overlap postprocess."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"chunk overlap ({overlap}) must be smaller than chunk size ({size}), "
            "otherwise chunking never advances"
        )

    chunks: list[ChunkSpan] = []
    start = 0
    text_length = len(text)
    # Build a non-overlapping core. Overlap is applied exactly once below.
    step = size
    while start < text_length:
        end = min(start + size, text_length)
        chunks.append(ChunkSpan(text=text[start:end], char_start=start, char_end=end))
        if end == text_length:
            break
        start += step
    # Keep this standalone public helper backwards-compatible. The dispatcher
    # calls it with zero overlap and applies the same postprocess to every
    # strategy.
    return apply_uniform_overlap(text, chunks, overlap)


def apply_uniform_overlap(
    text: str,
    spans: list[ChunkSpan],
    overlap: int,
    *,
    audit: list[dict[str, object]] | None = None,
) -> list[ChunkSpan]:
    """Expand every core span backwards by up to ``overlap`` source characters.

    The operation is strategy-independent and preserves source text exactly.
    At the beginning of a document, or when a requested overlap would make two
    starts identical, the effective overlap is shortened to keep deterministic
    strictly increasing chunk starts. The final chunk end is never changed.
    """
    if not isinstance(text, str):
        raise TypeError("chunking text must be a string")
    if type(overlap) is not int or overlap < 0:
        raise ValueError("chunk overlap must be a non-negative integer")
    if not isinstance(spans, list):
        raise TypeError("chunk spans must be a list")
    if not spans:
        return []

    previous_core_end = 0
    previous_start = -1
    expanded: list[ChunkSpan] = []
    for index, span in enumerate(spans):
        if not isinstance(span, ChunkSpan):
            raise TypeError("chunk spans must contain ChunkSpan values")
        if (
            type(span.char_start) is not int
            or type(span.char_end) is not int
            or not 0 <= span.char_start < span.char_end <= len(text)
            or span.text != text[span.char_start:span.char_end]
        ):
            raise ValueError(f"invalid source-aligned core span at index {index}")
        if index == 0 and span.char_start != 0:
            raise ValueError("core spans must start at the beginning of the source")
        if span.char_start < previous_core_end:
            raise ValueError("core spans must be non-overlapping and ordered")
        start = span.char_start if index == 0 else max(0, span.char_start - overlap)
        if start <= previous_start:
            start = previous_start + 1
        expanded.append(ChunkSpan(
            text=text[start:span.char_end],
            char_start=start,
            char_end=span.char_end,
            title=span.title,
            summary=span.summary,
        ))
        previous_start = start
        previous_core_end = span.char_end

    if expanded[-1].char_end != len(text):
        raise ValueError("core spans must cover the complete source")
    if audit is not None and overlap:
        audit.append({
            "event": "uniform_overlap",
            "policy": UNIFORM_OVERLAP_POLICY,
            "requested_overlap": overlap,
            "core_boundaries": [[span.char_start, span.char_end] for span in spans],
            "expanded_boundaries": [[span.char_start, span.char_end] for span in expanded],
            "effective_overlaps": [
                core.char_start - actual.char_start
                for core, actual in zip(spans, expanded, strict=True)
            ],
        })
    return expanded


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
    """Split at distances strictly above the percentile, ignoring numeric ties.

    A distance must exceed the threshold by more than the absolute tolerance.
    Percentile 100 therefore disables semantic splits; maximum chunk size and
    document-level rebalancing still apply.
    """
    _validate_semantic_config(
        breakpoint_percentile=breakpoint_percentile,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        buffer_size=buffer_size,
        embedding_batch_size=embedding_batch_size,
        embedding_concurrency=embedding_concurrency,
    )
    sentences = split_sentences(text)
    if not sentences:
        return []
    sentence_count = len(sentences)
    fits_one_chunk = sentence_count <= max_sentences
    can_form_two_minimum_chunks = sentence_count >= 2 * min_sentences
    if fits_one_chunk and not can_form_two_minimum_chunks:
        return [_span_from_sentences(text, sentences, 0, len(sentences) - 1)]

    embedding_inputs = [
        sentence_context(text, sentences, index, buffer_size)
        for index in range(len(sentences))
    ]
    embeddings = await embed_texts(
        embedding_inputs,
        embed_func=embed_func,
        embed_many_func=embed_many_func,
        batch_size=embedding_batch_size,
        concurrency=embedding_concurrency,
        purpose="semantic",
    )
    distances = [
        1.0 - _cosine_similarity(embeddings[index], embeddings[index + 1])
        for index in range(len(embeddings) - 1)
    ]
    threshold = _percentile(distances, breakpoint_percentile)

    boundaries: list[tuple[int, int]] = []
    start = 0
    for index, distance in enumerate(distances):
        sentence_count = index - start + 1
        should_split = (
            sentence_count >= min_sentences
            and distance > threshold + SEMANTIC_DISTANCE_ABS_TOLERANCE
        )
        must_split = sentence_count >= max_sentences
        if should_split or must_split:
            boundaries.append((start + 1, index + 1))
            start = index + 1

    if start < len(sentences):
        boundaries.append((start + 1, len(sentences)))

    boundaries = rebalance_document_boundaries(
        boundaries,
        sentence_count=len(sentences),
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    return [
        _span_from_sentences(text, sentences, start - 1, end - 1)
        for start, end in boundaries
    ]


def _validate_semantic_config(
    *,
    breakpoint_percentile: float,
    min_sentences: int,
    max_sentences: int,
    buffer_size: int,
    embedding_batch_size: int,
    embedding_concurrency: int,
) -> None:
    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError(
            "semantic breakpoint percentile must be between 0 and 100, "
            f"got {breakpoint_percentile}"
        )
    if min_sentences <= 0:
        raise ValueError(
            f"semantic min sentences must be positive, got {min_sentences}"
        )
    if max_sentences < min_sentences:
        raise ValueError(
            "semantic max sentences must be greater than or equal to min sentences"
        )
    if buffer_size < 0:
        raise ValueError(
            f"semantic buffer size must be non-negative, got {buffer_size}"
        )
    if embedding_batch_size <= 0:
        raise ValueError(
            "semantic embedding batch size must be positive, "
            f"got {embedding_batch_size}"
        )
    if embedding_concurrency <= 0:
        raise ValueError(
            "semantic embedding concurrency must be positive, "
            f"got {embedding_concurrency}"
        )


def _span_from_sentences(
    text: str,
    sentences: list[SentenceSpan],
    start: int,
    end: int,
) -> ChunkSpan:
    char_start = sentences[start].char_start
    char_end = sentences[end].char_end
    return ChunkSpan(
        text=text[char_start:char_end],
        char_start=char_start,
        char_end=char_end,
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
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
