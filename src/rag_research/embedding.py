import asyncio
import math
from collections.abc import Awaitable, Callable, Sequence


EmbeddingFunction = Callable[[str], Awaitable[list[float]]]
BatchEmbeddingFunction = Callable[
    [Sequence[str]],
    Awaitable[list[list[float]]],
]
EmbeddingProgressFunction = Callable[[int, int], None]


def _validate_vectors(
    vectors: object,
    *,
    expected_count: int,
) -> list[list[float]]:
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes)):
        raise ValueError("embedding backend must return a sequence of vectors")
    if len(vectors) != expected_count:
        raise ValueError(
            "embedding backend returned "
            f"{len(vectors)} vectors for {expected_count} texts"
        )

    validated: list[list[float]] = []
    expected_dimension: int | None = None

    for vector_index, vector in enumerate(vectors):
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
            raise ValueError(
                f"embedding vector {vector_index} must be a sequence"
            )
        if not vector:
            raise ValueError(f"embedding vector {vector_index} must not be empty")

        try:
            numeric_vector = [float(value) for value in vector]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"embedding vector {vector_index} must contain numeric values"
            ) from error

        if any(not math.isfinite(value) for value in numeric_vector):
            raise ValueError(
                f"embedding vector {vector_index} contains NaN or infinity"
            )
        if not any(value != 0.0 for value in numeric_vector):
            raise ValueError(
                f"embedding vector {vector_index} must not be a zero vector"
            )

        dimension = len(numeric_vector)
        if expected_dimension is None:
            expected_dimension = dimension
        elif dimension != expected_dimension:
            raise ValueError(
                f"embedding vector {vector_index} has dimension {dimension}, "
                f"expected {expected_dimension}"
            )

        validated.append(numeric_vector)

    return validated


async def embed_texts(
    texts: Sequence[str],
    *,
    embed_func: EmbeddingFunction,
    embed_many_func: BatchEmbeddingFunction | None,
    batch_size: int,
    concurrency: int,
    on_progress: EmbeddingProgressFunction | None = None,
) -> list[list[float]]:
    if batch_size <= 0:
        raise ValueError("embedding batch size must be positive")
    if concurrency <= 0:
        raise ValueError("embedding concurrency must be positive")
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        raise TypeError("embedding texts must be a sequence of strings")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("embedding texts must contain only strings")
    if not texts:
        return []
    if not callable(embed_func):
        raise TypeError("embed_func must be callable")
    if embed_many_func is not None and not callable(embed_many_func):
        raise TypeError("embed_many_func must be callable")

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0

    def report_progress(increment: int) -> None:
        nonlocal completed
        completed += increment
        if on_progress is not None:
            on_progress(completed, len(texts))

    if embed_many_func is None:
        async def embed_one(text: str) -> list[float]:
            async with semaphore:
                vector = await embed_func(text)
            report_progress(1)
            return vector

        raw_vectors = await asyncio.gather(*(
            embed_one(text)
            for text in texts
        ))
    else:
        batches = [
            texts[start:start + batch_size]
            for start in range(0, len(texts), batch_size)
        ]

        async def embed_batch(batch: Sequence[str]) -> list[list[float]]:
            async with semaphore:
                vectors = await embed_many_func(batch)
            validated_batch = _validate_vectors(
                vectors,
                expected_count=len(batch),
            )
            report_progress(len(batch))
            return validated_batch

        batch_vectors = await asyncio.gather(*(
            embed_batch(batch)
            for batch in batches
        ))
        raw_vectors = [
            vector
            for vectors in batch_vectors
            for vector in vectors
        ]

    return _validate_vectors(
        raw_vectors,
        expected_count=len(texts),
    )
