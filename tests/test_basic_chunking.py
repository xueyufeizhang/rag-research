import unittest

from rag_research.chunking import (
    ChunkSpan,
    fixed_size_chunk,
)


class ChunkSpanAssertions:
    def assert_source_aligned(
        self,
        source: str,
        chunks: list[ChunkSpan],
    ) -> None:
        previous_start = -1

        for chunk in chunks:
            self.assertGreaterEqual(chunk.char_start, 0)
            self.assertGreater(chunk.char_end, chunk.char_start)
            self.assertLessEqual(chunk.char_end, len(source))
            self.assertGreater(chunk.char_start, previous_start)
            self.assertEqual(
                source[chunk.char_start:chunk.char_end],
                chunk.text,
            )
            previous_start = chunk.char_start


class FixedSizeChunkTests(ChunkSpanAssertions, unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(fixed_size_chunk("", size=4, overlap=1), [])

    def test_text_shorter_than_window_returns_one_source_aligned_chunk(self):
        text = "abc"

        chunks = fixed_size_chunk(text, size=10, overlap=2)

        self.assertEqual(chunks, [ChunkSpan("abc", 0, 3)])
        self.assert_source_aligned(text, chunks)

    def test_exact_window_does_not_create_overlap_only_tail(self):
        text = "abcdefghij"

        chunks = fixed_size_chunk(text, size=10, overlap=2)

        self.assertEqual(chunks, [ChunkSpan(text, 0, len(text))])

    def test_fixed_windows_have_the_requested_overlap(self):
        text = "abcdefghij"

        chunks = fixed_size_chunk(text, size=4, overlap=1)

        self.assertEqual(
            chunks,
            [
                ChunkSpan("abcd", 0, 4),
                ChunkSpan("defg", 3, 7),
                ChunkSpan("ghij", 6, 10),
            ],
        )
        self.assert_source_aligned(text, chunks)

        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.char_end - current.char_start, 1)

    def test_zero_overlap_covers_every_character_once(self):
        text = "abcdefghij"

        chunks = fixed_size_chunk(text, size=4, overlap=0)

        self.assertEqual(
            chunks,
            [
                ChunkSpan("abcd", 0, 4),
                ChunkSpan("efgh", 4, 8),
                ChunkSpan("ij", 8, 10),
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assert_source_aligned(text, chunks)

    def test_invalid_configuration_is_rejected(self):
        cases = [
            (0, 0, "chunk size must be positive"),
            (-1, 0, "chunk size must be positive"),
            (4, -1, "chunk overlap must be non-negative"),
            (4, 4, "must be smaller than chunk size"),
            (4, 5, "must be smaller than chunk size"),
        ]

        for size, overlap, message in cases:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaisesRegex(ValueError, message):
                    fixed_size_chunk("text", size=size, overlap=overlap)
if __name__ == "__main__":
    unittest.main()
