import unittest

from rag_research.chunking import (
    ChunkSpan,
    ChunkConfig,
    SentenceSpan,
    _make_sentence_batches,
    _project_agentic_boundaries,
    _rebalance_agentic_document_boundaries,
    _split_sentences,
    _validate_agentic_boundaries,
    agentic_ibm_chunk,
    chunk_async,
)


FOUR_SENTENCES = "One is here. Two is here. Three is here. Four is here."


def source_span(source: str, value: str) -> ChunkSpan:
    start = source.index(value)
    return ChunkSpan(value, start, start + len(value))


def assert_source_aligned(
    test_case: unittest.TestCase,
    source: str,
    chunks: list[ChunkSpan],
) -> None:
    for chunk in chunks:
        test_case.assertGreaterEqual(chunk.char_start, 0)
        test_case.assertGreater(chunk.char_end, chunk.char_start)
        test_case.assertLessEqual(chunk.char_end, len(source))
        test_case.assertEqual(
            source[chunk.char_start:chunk.char_end],
            chunk.text,
        )


class AgenticIBMChunkingTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconstructs_verbatim_chunks_from_valid_boundaries(self):
        async def fake_llm(system: str, prompt: str) -> str:
            self.assertIn("[S4] Four is here.", prompt)
            return """```json
            {
              "chunks": [
                {"start": 1, "end": 2},
                {"start": 3, "end": 4}
              ]
            }
            ```"""

        chunks = await agentic_ibm_chunk(
            text=FOUR_SENTENCES,
            batch_max_sentences=10,
            batch_max_chars=1000,
            min_sentences=2,
            max_sentences=2,
            concurrency=1,
            retries=0,
            llm_func=fake_llm,
        )

        self.assertEqual(
            chunks,
            [
                source_span(FOUR_SENTENCES, "One is here. Two is here. "),
                source_span(FOUR_SENTENCES, "Three is here. Four is here."),
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), FOUR_SENTENCES)
        reconstructed_texts = [
            sentence.text
            for chunk in chunks
            for sentence in _split_sentences(chunk.text)
        ]
        self.assertEqual(
            reconstructed_texts,
            [sentence.text for sentence in _split_sentences(FOUR_SENTENCES)],
        )
        assert_source_aligned(self, FOUR_SENTENCES, chunks)

    async def test_retries_after_invalid_boundaries(self):
        calls = 0

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                return '{"chunks": [{"start": 1, "end": 2}, {"start": 4, "end": 4}]}'
            self.assertIn("previous response was invalid", prompt)
            return '{"chunks": [{"start": 1, "end": 2}, {"start": 3, "end": 4}]}'

        chunks = await agentic_ibm_chunk(
            text=FOUR_SENTENCES,
            batch_max_sentences=10,
            batch_max_chars=1000,
            min_sentences=2,
            max_sentences=2,
            concurrency=1,
            retries=1,
            llm_func=fake_llm,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(len(chunks), 2)
        assert_source_aligned(self, FOUR_SENTENCES, chunks)

    async def test_raises_after_retries_instead_of_mixing_in_fallback_chunks(self):
        calls = 0

        async def invalid_llm(system: str, prompt: str) -> str:
            nonlocal calls
            calls += 1
            return "not JSON"

        text = FOUR_SENTENCES + " Five is here."
        with self.assertRaisesRegex(
            RuntimeError,
            "agentic chunking batch 1 failed after 2 attempts",
        ):
            await agentic_ibm_chunk(
                text=text,
                batch_max_sentences=10,
                batch_max_chars=1000,
                min_sentences=1,
                max_sentences=2,
                concurrency=1,
                retries=1,
                llm_func=invalid_llm,
            )

        self.assertEqual(calls, 2)

    async def test_chunk_async_dispatches_agentic_strategy(self):
        async def fake_llm(system: str, prompt: str) -> str:
            return '{"chunks": [{"start": 1, "end": 4}]}'

        config = ChunkConfig(
            strategy="agentic_ibm",
            agentic_batch_max_sentences=10,
            agentic_batch_max_chars=1000,
            agentic_min_sentences=1,
            agentic_max_sentences=4,
            agentic_concurrency=1,
            agentic_retries=0,
        )
        chunks = await chunk_async(FOUR_SENTENCES, config, llm_func=fake_llm)
        self.assertEqual(
            chunks,
            [ChunkSpan(FOUR_SENTENCES, 0, len(FOUR_SENTENCES))],
        )

    async def test_chunk_async_requires_llm(self):
        config = ChunkConfig(strategy="agentic_ibm")
        with self.assertRaisesRegex(ValueError, "requires an llm_func"):
            await chunk_async(FOUR_SENTENCES, config)

    def test_batches_obey_sentence_and_character_limits(self):
        sentences = [
            SentenceSpan("aaaa", 0, 4),
            SentenceSpan("bbbb", 5, 9),
            SentenceSpan("cccc", 10, 14),
            SentenceSpan("dddd", 15, 19),
        ]
        batches = _make_sentence_batches(
            sentences,
            max_sentences=3,
            max_chars=9,
        )
        self.assertEqual(
            batches,
            [sentences[:2], sentences[2:]],
        )

    async def test_multiple_batches_keep_document_level_offsets(self):
        async def fake_llm(system: str, prompt: str) -> str:
            return '{"chunks": [{"start": 1, "end": 2}]}'

        chunks = await agentic_ibm_chunk(
            text=FOUR_SENTENCES,
            batch_max_sentences=2,
            batch_max_chars=1000,
            min_sentences=1,
            max_sentences=2,
            concurrency=2,
            retries=0,
            llm_func=fake_llm,
        )

        self.assertEqual(
            chunks,
            [
                source_span(FOUR_SENTENCES, "One is here. Two is here. "),
                source_span(FOUR_SENTENCES, "Three is here. Four is here."),
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), FOUR_SENTENCES)
        assert_source_aligned(self, FOUR_SENTENCES, chunks)

    def test_sentence_spans_are_a_lossless_partition(self):
        text = "  Alpha is here.\n\nBravo is here.  "

        sentences = _split_sentences(text)

        self.assertEqual("".join(sentence.text for sentence in sentences), text)
        self.assertEqual(sentences[0].char_start, 0)
        self.assertEqual(sentences[-1].char_end, len(text))
        for left, right in zip(sentences, sentences[1:]):
            self.assertEqual(left.char_end, right.char_start)
        for sentence in sentences:
            self.assertEqual(
                sentence.text,
                text[sentence.char_start:sentence.char_end],
            )

    def test_overlapping_ellipsis_spans_are_merged_without_text_loss(self):
        text = "Loading . . .\n\nStudents are here."

        sentences = _split_sentences(text)

        self.assertEqual(
            sentences,
            [
                SentenceSpan("Loading . . .\n\n", 0, 15),
                SentenceSpan("Students are here.", 15, len(text)),
            ],
        )
        self.assertEqual("".join(sentence.text for sentence in sentences), text)

    def test_validator_rejects_gaps(self):
        with self.assertRaisesRegex(ValueError, "expected chunk to start at 3"):
            _validate_agentic_boundaries(
                [(1, 2), (4, 4)],
                sentence_count=4,
                min_sentences=1,
                max_sentences=3,
            )

    def test_projection_keeps_already_valid_boundaries(self):
        boundaries = [(1, 10), (11, 24), (25, 29)]

        projected = _project_agentic_boundaries(
            boundaries,
            sentence_count=29,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(projected, boundaries)

    def test_projection_splits_an_oversized_single_chunk(self):
        projected = _project_agentic_boundaries(
            [(1, 26)],
            sentence_count=26,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(projected, [(1, 13), (14, 26)])

    def test_projection_moves_an_undersized_non_final_boundary(self):
        projected = _project_agentic_boundaries(
            [(1, 7), (8, 27)],
            sentence_count=27,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(projected, [(1, 10), (11, 27)])

    def test_projection_selects_the_nearest_feasible_chunk_count(self):
        projected = _project_agentic_boundaries(
            [(1, 60)],
            sentence_count=60,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(projected, [(1, 20), (21, 40), (41, 60)])

    def test_document_rebalancing_borrows_only_when_merge_is_impossible(self):
        rebalanced = _rebalance_agentic_document_boundaries(
            [(1, 24), (25, 29)],
            sentence_count=29,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(rebalanced, [(1, 19), (20, 29)])

    def test_projection_produces_legal_boundaries_for_every_batch_size(self):
        for sentence_count in range(1, 61):
            proposals = [
                [(1, sentence_count)],
                [
                    (sentence_index, sentence_index)
                    for sentence_index in range(1, sentence_count + 1)
                ],
            ]
            for proposed in proposals:
                with self.subTest(
                    sentence_count=sentence_count,
                    proposed_chunk_count=len(proposed),
                ):
                    projected = _project_agentic_boundaries(
                        proposed,
                        sentence_count=sentence_count,
                        min_sentences=10,
                        max_sentences=24,
                    )
                    _validate_agentic_boundaries(
                        projected,
                        sentence_count=sentence_count,
                        min_sentences=10,
                        max_sentences=24,
                    )

    async def test_size_violation_is_projected_without_retrying(self):
        text = " ".join(f"Sentence {index}." for index in range(1, 28))
        calls = 0
        projection_events: list[dict[str, object]] = []

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal calls
            calls += 1
            return (
                '{"chunks": ['
                '{"start": 1, "end": 7}, '
                '{"start": 8, "end": 27}'
                ']}'
            )

        chunks = await agentic_ibm_chunk(
            text=text,
            batch_max_sentences=60,
            batch_max_chars=12000,
            min_sentences=10,
            max_sentences=24,
            concurrency=1,
            retries=2,
            llm_func=fake_llm,
            projection_events=projection_events,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(
            [len(_split_sentences(chunk.text)) for chunk in chunks],
            [10, 17],
        )
        self.assertEqual(
            projection_events,
            [
                {
                    "scope": "batch",
                    "batch_index": 1,
                    "sentence_count": 27,
                    "original_boundaries": [[1, 7], [8, 27]],
                    "projected_boundaries": [[1, 10], [11, 27]],
                }
            ],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        assert_source_aligned(self, text, chunks)

    async def test_document_projection_rebalances_short_macro_batch_tails(self):
        text = " ".join(
            f"Sentence {index}."
            for index in range(1, 31)
        )
        projection_events: list[dict[str, object]] = []

        async def fake_llm(system: str, prompt: str) -> str:
            return (
                '{"chunks": ['
                '{"start": 1, "end": 10}, '
                '{"start": 11, "end": 15}'
                ']}'
            )

        chunks = await agentic_ibm_chunk(
            text=text,
            batch_max_sentences=15,
            batch_max_chars=12000,
            min_sentences=10,
            max_sentences=24,
            concurrency=2,
            retries=0,
            llm_func=fake_llm,
            projection_events=projection_events,
        )

        self.assertEqual(
            [len(_split_sentences(chunk.text)) for chunk in chunks],
            [15, 15],
        )
        self.assertEqual(len(projection_events), 1)
        self.assertEqual(projection_events[0]["scope"], "document")
        self.assertEqual(projection_events[0]["batch_index"], 0)
        self.assertEqual(
            projection_events[0]["original_boundaries"],
            [[1, 10], [11, 15], [16, 25], [26, 30]],
        )
        self.assertEqual(
            projection_events[0]["projected_boundaries"],
            [[1, 15], [16, 30]],
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        assert_source_aligned(self, text, chunks)


if __name__ == "__main__":
    unittest.main()
