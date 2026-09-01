import asyncio
import unittest

from rag_research.embedding import embed_texts


class BatchEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_batches_preserve_input_order(self):
        calls: list[list[str]] = []

        async def embed_one(_: str) -> list[float]:
            raise AssertionError("single embedding fallback must not be used")

        async def embed_many(texts) -> list[list[float]]:
            calls.append(list(texts))
            return [
                [float(text.removeprefix("text-")) + 1.0, 1.0]
                for text in texts
            ]

        vectors = await embed_texts(
            [f"text-{index}" for index in range(5)],
            embed_func=embed_one,
            embed_many_func=embed_many,
            batch_size=2,
            concurrency=1,
        )

        self.assertEqual(
            calls,
            [["text-0", "text-1"], ["text-2", "text-3"], ["text-4"]],
        )
        self.assertEqual(
            vectors,
            [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0]],
        )

    async def test_batch_concurrency_is_bounded(self):
        active = 0
        maximum_active = 0

        async def embed_one(_: str) -> list[float]:
            raise AssertionError("single embedding fallback must not be used")

        async def embed_many(texts) -> list[list[float]]:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

        await embed_texts(
            [f"text-{index}" for index in range(10)],
            embed_func=embed_one,
            embed_many_func=embed_many,
            batch_size=2,
            concurrency=2,
        )

        self.assertEqual(maximum_active, 2)

    async def test_single_embedding_fallback_is_bounded(self):
        active = 0
        maximum_active = 0

        async def embed_one(_: str) -> list[float]:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [1.0, 1.0]

        vectors = await embed_texts(
            [f"text-{index}" for index in range(8)],
            embed_func=embed_one,
            embed_many_func=None,
            batch_size=4,
            concurrency=3,
        )

        self.assertEqual(len(vectors), 8)
        self.assertEqual(maximum_active, 3)

    async def test_backend_vector_count_must_match_batch(self):
        async def embed_one(_: str) -> list[float]:
            return [1.0, 1.0]

        async def embed_many(_texts) -> list[list[float]]:
            return [[1.0, 1.0]]

        with self.assertRaisesRegex(ValueError, "1 vectors for 2 texts"):
            await embed_texts(
                ["first", "second"],
                embed_func=embed_one,
                embed_many_func=embed_many,
                batch_size=2,
                concurrency=1,
            )

    async def test_invalid_vectors_are_rejected_before_indexing(self):
        cases = [
            ([[1.0, 2.0], [1.0]], "dimension"),
            ([[1.0, 2.0], [float("nan"), 1.0]], "NaN or infinity"),
            ([[1.0, 2.0], [0.0, 0.0]], "zero vector"),
            ([[1.0, 2.0], []], "must not be empty"),
            ([[1.0, 2.0], ["bad", 1.0]], "numeric values"),
        ]

        async def embed_one(_: str) -> list[float]:
            return [1.0, 1.0]

        for returned_vectors, message in cases:
            with self.subTest(message=message):
                async def embed_many(_texts, result=returned_vectors):
                    return result

                with self.assertRaisesRegex(ValueError, message):
                    await embed_texts(
                        ["first", "second"],
                        embed_func=embed_one,
                        embed_many_func=embed_many,
                        batch_size=2,
                        concurrency=1,
                    )

    async def test_empty_input_does_not_call_backend(self):
        calls = 0

        async def embed_one(_: str) -> list[float]:
            nonlocal calls
            calls += 1
            return [1.0, 1.0]

        vectors = await embed_texts(
            [],
            embed_func=embed_one,
            embed_many_func=None,
            batch_size=2,
            concurrency=1,
        )

        self.assertEqual(vectors, [])
        self.assertEqual(calls, 0)

    async def test_invalid_execution_configuration_is_rejected(self):
        async def embed_one(_: str) -> list[float]:
            return [1.0, 1.0]

        cases = [
            ({"batch_size": 0, "concurrency": 1}, "batch size"),
            ({"batch_size": 1, "concurrency": 0}, "concurrency"),
        ]

        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, message):
                    await embed_texts(
                        ["text"],
                        embed_func=embed_one,
                        embed_many_func=None,
                        **arguments,
                    )


if __name__ == "__main__":
    unittest.main()
