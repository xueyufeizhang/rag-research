from typing import Awaitable, Callable

from rag_research.agentic_chunking import agentic_chunk
from rag_research.chunking_models import ChunkConfig, ChunkSpan, SentenceSpan
from rag_research.embedding import (
    BatchEmbeddingFunction,
    EmbeddingFunction,
    embed_texts,
)
from rag_research.text_spans import sentence_context, split_sentences


CHUNKING_PIPELINE_VERSION = 4


async def chunk_async(
    text: str,
    config: ChunkConfig,
    embed_func: EmbeddingFunction | None = None,
    embed_many_func: BatchEmbeddingFunction | None = None,
    llm_func: Callable[..., Awaitable[str]] | None = None,
    agentic_state_events: list[dict[str, object]] | None = None,
) -> list[ChunkSpan]:
    """Dispatch a document to the configured chunking strategy."""
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
    if strategy == "agentic":
        if llm_func is None:
            raise ValueError("agentic chunking requires an llm_func")
        return await agentic_chunk(
            text=text,
            batch_max_sentences=config.agentic_batch_max_sentences,
            batch_max_chars=config.agentic_batch_max_chars,
            min_sentences=config.agentic_min_sentences,
            max_sentences=config.agentic_max_sentences,
            concurrency=config.agentic_concurrency,
            retries=config.agentic_retries,
            llm_func=llm_func,
            state_events=agentic_state_events,
        )
    raise ValueError(f"unknown chunking strategy: {config.strategy}")


def fixed_size_chunk(text: str, size: int, overlap: int) -> list[ChunkSpan]:
    """Split text into fixed character windows with optional overlap."""
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
    """Split text at unusually large adjacent-sentence embedding distances."""
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
    if len(sentences) <= max_sentences:
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
    )
    distances = [
        1.0 - _cosine_similarity(embeddings[index], embeddings[index + 1])
        for index in range(len(embeddings) - 1)
    ]
    threshold = _percentile(distances, breakpoint_percentile)

    chunks: list[ChunkSpan] = []
    start = 0
    for index, distance in enumerate(distances):
        sentence_count = index - start + 1
        should_split = (
            sentence_count >= min_sentences and distance >= threshold
        )
        must_split = sentence_count >= max_sentences
        if should_split or must_split:
            chunks.append(_span_from_sentences(text, sentences, start, index))
            start = index + 1

    if start < len(sentences):
        tail_size = len(sentences) - start
        if chunks and tail_size < min_sentences:
            chunks[-1] = ChunkSpan(
                text=text[chunks[-1].char_start:sentences[-1].char_end],
                char_start=chunks[-1].char_start,
                char_end=sentences[-1].char_end,
            )
        else:
            chunks.append(
                _span_from_sentences(
                    text,
                    sentences,
                    start,
                    len(sentences) - 1,
                )
            )
    return chunks


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
