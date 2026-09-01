import unittest

from rag_research.chunking import (
    ChunkConfig,
    ChunkSpan,
    chunk_async,
    semantic_chunk,
)


SENTENCES = [
    "Alpha is here.",
    "Bravo is here.",
    "Charlie is here.",
    "Delta is here.",
    "Echo is here.",
]
SOURCE = " ".join(SENTENCES)


def source_span(source: str, value: str) -> ChunkSpan:
    start = source.index(value)
    return ChunkSpan(value, start, start + len(value))


class SemanticChunkingTests(unittest.IsolatedAsyncioTestCase):
    def assert_source_aligned(
        self,
        source: str,
        chunks: list[ChunkSpan],
    ) -> None:
        previous_start = -1

        for chunk in chunks:
            self.assertIsInstance(chunk, ChunkSpan)
            self.assertGreaterEqual(chunk.char_start, 0)
            self.assertGreater(chunk.char_end, chunk.char_start)
            self.assertLessEqual(chunk.char_end, len(source))
            self.assertGreater(chunk.char_start, previous_start)
            self.assertEqual(
                source[chunk.char_start:chunk.char_end],
                chunk.text,
            )
            previous_start = chunk.char_start

    async def test_empty_text_returns_no_chunks_without_embedding(self):
        calls = 0

        async def embed_func(text: str) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0, 0.0]

        chunks = await semantic_chunk(
            text="",
            breakpoint_percentile=90,
            min_sentences=1,
            max_sentences=4,
            buffer_size=1,
            embedding_concurrency=1,
            embed_func=embed_func,
        )

        self.assertEqual(chunks, [])
        self.assertEqual(calls, 0)

    async def test_short_document_returns_one_verbatim_chunk(self):
        text = "Alpha is here.\n\nBravo is here."
        calls = 0

        async def embed_func(value: str) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0, 0.0]

        chunks = await semantic_chunk(
            text=text,
            breakpoint_percentile=90,
            min_sentences=1,
            max_sentences=4,
            buffer_size=1,
            embedding_concurrency=1,
            embed_func=embed_func,
        )

        self.assertEqual(chunks, [ChunkSpan(text, 0, len(text))])
        self.assertEqual(calls, 0)

    async def test_semantic_boundary_returns_source_aligned_chunks(self):
        async def embed_func(value: str) -> list[float]:
            if value.startswith(("Alpha", "Bravo")):
                return [1.0, 0.0]
            return [0.0, 1.0]

        chunks = await semantic_chunk(
            text=SOURCE,
            breakpoint_percentile=90,
            min_sentences=2,
            max_sentences=4,
            buffer_size=0,
            embedding_concurrency=2,
            embed_func=embed_func,
        )

        self.assertEqual(
            chunks,
            [
                source_span(SOURCE, " ".join(SENTENCES[:2]) + " "),
                source_span(SOURCE, " ".join(SENTENCES[2:])),
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), SOURCE)
        self.assert_source_aligned(SOURCE, chunks)

    async def test_chunk_async_dispatches_semantic_strategy(self):
        async def embed_func(value: str) -> list[float]:
            return [1.0, float(len(value))]

        config = ChunkConfig(
            strategy="semantic",
            semantic_min_sentences=1,
            semantic_max_sentences=10,
            semantic_buffer_size=0,
            semantic_embedding_concurrency=1,
        )

        chunks = await chunk_async(
            SOURCE,
            config,
            embed_func=embed_func,
        )

        self.assertEqual(chunks, [ChunkSpan(SOURCE, 0, len(SOURCE))])

    async def test_chunk_async_uses_batch_embeddings_for_semantic_strategy(self):
        calls: list[list[str]] = []

        async def embed_func(_: str) -> list[float]:
            raise AssertionError("single embedding fallback must not be used")

        async def embed_many_func(texts) -> list[list[float]]:
            calls.append(list(texts))
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

        config = ChunkConfig(
            strategy="semantic",
            semantic_min_sentences=1,
            semantic_max_sentences=4,
            semantic_buffer_size=0,
            semantic_embedding_batch_size=2,
            semantic_embedding_concurrency=1,
        )

        chunks = await chunk_async(
            SOURCE,
            config,
            embed_func=embed_func,
            embed_many_func=embed_many_func,
        )

        self.assertEqual([len(batch) for batch in calls], [2, 2, 1])
        self.assert_source_aligned(SOURCE, chunks)

    async def test_chunk_async_requires_embeddings(self):
        config = ChunkConfig(strategy="semantic")

        with self.assertRaisesRegex(ValueError, "requires an embed_func"):
            await chunk_async(SOURCE, config)

    async def test_invalid_configuration_is_rejected(self):
        async def embed_func(value: str) -> list[float]:
            return [1.0, 0.0]

        cases = [
            ({"breakpoint_percentile": -1}, "between 0 and 100"),
            ({"breakpoint_percentile": 101}, "between 0 and 100"),
            ({"min_sentences": 0}, "min sentences must be positive"),
            (
                {"min_sentences": 3, "max_sentences": 2},
                "max sentences must be greater than or equal",
            ),
            ({"buffer_size": -1}, "buffer size must be non-negative"),
            ({"embedding_batch_size": 0}, "batch size must be positive"),
            ({"embedding_concurrency": 0}, "concurrency must be positive"),
        ]
        defaults = {
            "breakpoint_percentile": 90,
            "min_sentences": 1,
            "max_sentences": 4,
            "buffer_size": 1,
            "embedding_batch_size": 2,
            "embedding_concurrency": 1,
        }

        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                arguments = {**defaults, **overrides}
                with self.assertRaisesRegex(ValueError, message):
                    await semantic_chunk(
                        text=SOURCE,
                        embed_func=embed_func,
                        **arguments,
                    )


if __name__ == "__main__":
    unittest.main()
