from dataclasses import dataclass
import asyncio
import re
from typing import Awaitable, Callable, Optional


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
    semantic_embedding_concurrency: int = 4


def chunk(text: str, config: ChunkConfig) -> list[str]:
    strategy = config.strategy.lower()
    if strategy == "fixed":
        return fixed_size_chunk(text, config.fixed_size, config.fixed_overlap)
    if strategy == "sentence_window":
        return sentence_window_chunk(text, config.sentence_window_size, config.sentence_window_overlap)
    if strategy == "semantic":
        raise ValueError("semantic chunking requires embeddings; use chunk_async instead")
    raise ValueError(f"unknown chunking strategy: {config.strategy}")


async def chunk_async(
    text: str,
    config: ChunkConfig,
    embed_func: Optional[Callable[[str], Awaitable[list[float]]]] = None,
) -> list[str]:
    strategy = config.strategy.lower()
    if strategy == "semantic":
        if embed_func is None:
            raise ValueError("semantic chunking requires an embed_func")
        return await semantic_chunk(
            text,
            breakpoint_percentile=config.semantic_breakpoint_percentile,
            min_sentences=config.semantic_min_sentences,
            max_sentences=config.semantic_max_sentences,
            buffer_size=config.semantic_buffer_size,
            embedding_concurrency=config.semantic_embedding_concurrency,
            embed_func=embed_func,
        )
    return chunk(text, config)


def fixed_size_chunk(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk overlap must be non-negative, got {overlap}")
    if overlap >= size:
        raise ValueError(f"chunk overlap ({overlap}) must be smaller than chunk size ({size}), "
                          f"otherwise chunking never advances")

    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks


def sentence_window_chunk(text: str, window_size: int, window_overlap: int) -> list[str]:
    if window_size <= 0:
        raise ValueError(f"sentence window size must be positive, got {window_size}")
    if window_overlap < 0:
        raise ValueError(f"sentence window overlap must be non-negative, got {window_overlap}")
    if window_overlap >= window_size:
        raise ValueError("sentence window overlap must be smaller than window size")

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= window_size:
        return [" ".join(sentences)]

    step = window_size - window_overlap
    starts = list(range(0, len(sentences) - window_size + 1, step))

    last_start = len(sentences) - window_size
    if starts[-1] != last_start:
        starts.append(last_start)

    return [
        " ".join(sentences[i:i + window_size])
        for i in starts
    ]


async def semantic_chunk(
    text: str,
    breakpoint_percentile: float,
    min_sentences: int,
    max_sentences: int,
    buffer_size: int,
    embedding_concurrency: int,
    embed_func: Callable[[str], Awaitable[list[float]]],
) -> list[str]:
    if not 0 <= breakpoint_percentile <= 100:
        raise ValueError(f"semantic breakpoint percentile must be between 0 and 100, got {breakpoint_percentile}")
    if min_sentences <= 0:
        raise ValueError(f"semantic min sentences must be positive, got {min_sentences}")
    if max_sentences < min_sentences:
        raise ValueError("semantic max sentences must be greater than or equal to min sentences")
    if buffer_size < 0:
        raise ValueError(f"semantic buffer size must be non-negative, got {buffer_size}")
    if embedding_concurrency <= 0:
        raise ValueError(f"semantic embedding concurrency must be positive, got {embedding_concurrency}")

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        return [" ".join(sentences)]

    embedding_inputs = [
        _sentence_context(sentences, idx, buffer_size)
        for idx in range(len(sentences))
    ]
    embeddings = await _embed_texts(embedding_inputs, embed_func, embedding_concurrency)
    distances = [
        1.0 - _cosine_similarity(embeddings[idx], embeddings[idx + 1])
        for idx in range(len(embeddings) - 1)
    ]
    threshold = _percentile(distances, breakpoint_percentile)

    chunks = []
    start = 0
    for idx, distance in enumerate(distances):
        sentence_count = idx - start + 1
        should_split = sentence_count >= min_sentences and distance >= threshold
        must_split = sentence_count >= max_sentences
        if should_split or must_split:
            chunks.append(" ".join(sentences[start:idx + 1]))
            start = idx + 1

    if start < len(sentences):
        tail = sentences[start:]
        if chunks and len(tail) < min_sentences:
            chunks[-1] = chunks[-1] + " " + " ".join(tail)
        else:
            chunks.append(" ".join(tail))

    return chunks


def _split_sentences(text: str) -> list[str]:
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if s.strip()
    ]


def _sentence_context(sentences: list[str], idx: int, buffer_size: int) -> str:
    start = max(0, idx - buffer_size)
    end = min(len(sentences), idx + buffer_size + 1)
    return " ".join(sentences[start:end])


async def _embed_texts(
    texts: list[str],
    embed_func: Callable[[str], Awaitable[list[float]]],
    concurrency: int,
) -> list[list[float]]:
    sem = asyncio.Semaphore(concurrency)

    async def embed_one(text: str) -> list[float]:
        async with sem:
            return await embed_func(text)

    return await asyncio.gather(*(embed_one(text) for text in texts))


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
