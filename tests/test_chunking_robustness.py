import asyncio
import itertools
import json
import unittest
from fractions import Fraction
from unittest.mock import AsyncMock, patch

from rag_research.chunking.agentic_boundaries import (
    project_boundaries,
    validate_boundaries,
)
from rag_research.chunking.agentic_chunking import agentic_chunk
from rag_research.chunking.agentic_llm import (
    AgenticLlmGateway,
    _parse_named_boundaries,
    make_sentence_batches,
)
from rag_research.chunking.chunking_models import ChunkConfig, ChunkSpan, SentenceSpan
from rag_research.chunking.dispatcher import (
    apply_uniform_overlap,
    chunk_async,
    semantic_chunk,
)
from rag_research.chunking.text_spans import split_sentences
from rag_research.llm import LLMResponse
from rag_research.prompts import (
    AGENTIC_METADATA_SYSTEM_PROMPT,
    AGENTIC_PROPOSITION_SYSTEM_PROMPT,
    AGENTIC_STATE_SYSTEM_PROMPT,
)


def all_partitions(sentence_count):
    for cut_count in range(sentence_count):
        for cuts in itertools.combinations(range(1, sentence_count), cut_count):
            endpoints = (0, *cuts, sentence_count)
            yield list(zip(
                (value + 1 for value in endpoints[:-1]), endpoints[1:]
            ))


class ProjectionRobustnessTests(unittest.TestCase):
    def test_projection_matches_exhaustive_objective_for_small_inputs(self):
        """Enumerate partitions independently of the dynamic program."""
        for sentence_count in range(1, 8):
            partitions = list(all_partitions(sentence_count))
            for minimum in range(1, 4):
                for maximum in range(minimum, 4):
                    for allow_short in (False, True):
                        legal = [
                            partition for partition in partitions
                            if all(
                                (1 if allow_short and index == len(partition) - 1
                                 else minimum) <= end - start + 1 <= maximum
                                for index, (start, end) in enumerate(partition)
                            )
                        ]
                        for proposed in partitions:
                            with self.subTest(
                                n=sentence_count, proposed=proposed,
                                minimum=minimum, maximum=maximum,
                                allow_short=allow_short,
                            ):
                                if not legal:
                                    with self.assertRaises(ValueError):
                                        project_boundaries(
                                            proposed, sentence_count, minimum,
                                            maximum, allow_short,
                                        )
                                    continue
                                chunk_count = min(
                                    {len(partition) for partition in legal},
                                    key=lambda count: (abs(count - len(proposed)), count),
                                )
                                model_ends = [0, *(end for _, end in proposed)]
                                targets = []
                                for index in range(1, chunk_count):
                                    position = Fraction(index * len(proposed), chunk_count)
                                    left = position.numerator // position.denominator
                                    targets.append(
                                        model_ends[left]
                                        + (position - left)
                                        * (model_ends[left + 1] - model_ends[left])
                                    )

                                def objective(partition):
                                    ends = tuple(end for _, end in partition)
                                    return (
                                        sum(abs(end - target) for end, target
                                            in zip(ends[:-1], targets)),
                                        ends,
                                    )

                                expected = min(
                                    (partition for partition in legal
                                     if len(partition) == chunk_count),
                                    key=objective,
                                )
                                self.assertEqual(
                                    project_boundaries(
                                        proposed, sentence_count, minimum,
                                        maximum, allow_short,
                                    ),
                                    expected,
                                )

    def test_projection_handles_two_thousand_sentences_without_recursion(self):
        projected = project_boundaries([(1, 2000)], 2000, 1, 2)
        self.assertEqual(projected, [(start, start + 1) for start in range(1, 2000, 2)])
        validate_boundaries(projected, 2000, 1, 2)


class UniformOverlapTests(unittest.IsolatedAsyncioTestCase):
    def test_common_overlap_is_a_single_canonical_configuration_value(self):
        self.assertEqual(ChunkConfig().overlap_size, 200)
        self.assertEqual(ChunkConfig(overlap_size=7).overlap_size, 7)
        self.assertEqual(
            ChunkConfig(overlap_size=7).fingerprint_dict()["overlap_size"],
            7,
        )

    def test_postprocess_expands_core_spans_without_changing_ends_or_metadata(self):
        source = "0123456789ABCDEFGHIJ"
        core = [
            ChunkSpan(source[:5], 0, 5),
            ChunkSpan(source[5:12], 5, 12, title="Second", summary="Topic"),
            ChunkSpan(source[12:], 12, len(source)),
        ]

        expanded = apply_uniform_overlap(source, core, 3)

        self.assertEqual(
            [(chunk.char_start, chunk.char_end, chunk.text) for chunk in expanded],
            [(0, 5, source[:5]), (2, 12, source[2:12]), (9, 20, source[9:])],
        )
        self.assertEqual((expanded[1].title, expanded[1].summary), ("Second", "Topic"))
        self.assertEqual("0123456789ABCDEFGHIJ", source)

    async def test_dispatcher_applies_one_common_postprocess_to_all_strategies(self):
        source = "0123456789ABCDEFGHIJ"
        core = [ChunkSpan(source[:5], 0, 5), ChunkSpan(source[5:], 5, len(source))]

        fixed = await chunk_async(
            source,
            ChunkConfig(strategy="fixed", fixed_size=5, overlap_size=3),
        )
        self.assertEqual([chunk.char_start for chunk in fixed], [0, 2, 7, 12])

        with patch(
            "rag_research.chunking.dispatcher.semantic_chunk",
            new=AsyncMock(return_value=core),
        ):
            semantic = await chunk_async(
                source,
                ChunkConfig(strategy="semantic", overlap_size=3),
                embed_func=AsyncMock(),
            )
        self.assertEqual(
            [(chunk.char_start, chunk.char_end) for chunk in semantic],
            [(0, 5), (2, len(source))],
        )

        with patch(
            "rag_research.chunking.dispatcher.agentic_chunk",
            new=AsyncMock(return_value=core),
        ):
            agentic = await chunk_async(
                source,
                ChunkConfig(strategy="agentic", overlap_size=3),
                llm_func=AsyncMock(),
            )
        self.assertEqual(
            [(chunk.char_start, chunk.char_end) for chunk in agentic],
            [(0, 5), (2, len(source))],
        )


class SemanticTiePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def chunk_vectors(self, vectors, *, percentile=90, minimum=8, maximum=24):
        sentences = [f"Sentence {index} is present." for index in range(len(vectors))]
        text = " ".join(sentences)
        by_sentence = dict(zip(sentences, vectors))

        async def embed(value):
            return by_sentence[value.removeprefix("clustering: ").strip()]

        chunks = await semantic_chunk(
            text, percentile, minimum, maximum, 0, 1, embed,
        )
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        for chunk in chunks:
            self.assertEqual(chunk.text, text[chunk.char_start:chunk.char_end])
        return [len(split_sentences(chunk.text)) for chunk in chunks]

    async def test_equal_distances_have_no_soft_boundaries(self):
        self.assertEqual(await self.chunk_vectors([[1.0, 0.0]] * 24), [24])
        self.assertEqual(
            await self.chunk_vectors([[1.0, 0.0], [0.0, 1.0]] * 12), [24],
        )
        self.assertEqual(
            await self.chunk_vectors([[1.0, 0.0]] * 24, minimum=4, maximum=10),
            [10, 10, 4],
        )

    async def test_mostly_zero_distances_keep_only_the_real_topic_change(self):
        vectors = [[1.0, 0.0]] * 12 + [[0.0, 1.0]] * 12
        self.assertEqual(await self.chunk_vectors(vectors), [12, 12])

    async def test_percentile_one_hundred_uses_only_hard_size_boundaries(self):
        vectors = [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 7
        self.assertEqual(
            await self.chunk_vectors(vectors, percentile=100, minimum=2, maximum=6),
            [6, 4],
        )

    async def test_roundoff_above_threshold_does_not_create_a_boundary(self):
        with patch(
            "rag_research.chunking.dispatcher._cosine_similarity",
            side_effect=[1.0, 1.0 - 5e-13, 1.0, 1.0],
        ):
            self.assertEqual(
                await self.chunk_vectors(
                    [[1.0, 0.0]] * 5, percentile=50, minimum=1, maximum=5,
                ),
                [5],
            )


class AgenticInputContractTests(unittest.TestCase):
    def test_json_boundaries_require_actual_integers(self):
        for field in ("start", "end"):
            for value in (1.0, 1.9, True, False, "1", None):
                with self.subTest(field=field, value=value):
                    boundary = {"start": 1, "end": 1, field: value}
                    with self.assertRaisesRegex(ValueError, "integer start and end"):
                        _parse_named_boundaries(
                            json.dumps({"propositions": [boundary]}),
                            field_name="propositions",
                        )
        self.assertEqual(
            _parse_named_boundaries(
                '{"propositions":[{"start":1,"end":2}]}',
                field_name="propositions",
            ),
            [(1, 2)],
        )

    def test_oversized_sentence_reports_budget_and_original_span(self):
        sentence = SentenceSpan("  abcdef  ", 12, 22)
        with self.assertRaisesRegex(
            ValueError, r"span=\[12, 22\).*stripped_source_chars=6, budget=5"
        ):
            make_sentence_batches([sentence], max_sentences=10, max_chars=5)
        self.assertEqual(sentence.text, "  abcdef  ")
        self.assertEqual(
            make_sentence_batches([sentence], max_sentences=10, max_chars=6),
            [[sentence]],
        )


class AgenticMetadataAuditTests(unittest.IsolatedAsyncioTestCase):
    async def test_oversized_source_fails_before_any_model_request(self):
        calls = []

        async def fake_llm(**kwargs):
            calls.append(kwargs)
            self.fail("oversized source must be rejected before model calls")

        with self.assertRaisesRegex(ValueError, "budget=5"):
            await agentic_chunk("A long sentence.", 10, 5, 1, 3, 1, 0, fake_llm)
        self.assertEqual(calls, [])

    async def test_final_rebalanced_metadata_fallback_is_recorded(self):
        text = "First sentence. Second sentence. Third sentence."

        async def fake_llm(system, prompt):
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return LLMResponse(
                    '{"propositions":[{"start":1,"end":2},{"start":3,"end":3}]}',
                    truncated=False,
                )
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                return LLMResponse(
                    '{"action":"new_chunk","title":"Topic","summary":"Summary."}',
                    truncated=False,
                )
            self.assertEqual(system, AGENTIC_METADATA_SYSTEM_PROMPT)
            return '{"title":"","summary":"Invalid title."}'

        events = []
        chunks = await agentic_chunk(text, 10, 1000, 2, 3, 1, 0, fake_llm, events)
        refreshes = [event for event in events if event["event"] == "metadata_refresh"]
        self.assertEqual(len(refreshes), 1)
        self.assertEqual(refreshes[0]["final_boundary"], [1, 3])
        self.assertEqual(refreshes[0]["decision_source"], "fallback")
        self.assertIn("non-empty title", refreshes[0]["fallback_error"])
        self.assertEqual(refreshes[0]["title"], chunks[0].title)
        self.assertEqual(refreshes[0]["summary"], chunks[0].summary)
        self.assertEqual(chunks[0].text, text)

    async def test_refresh_trace_order_follows_boundaries_not_completion(self):
        text = "Alpha. Beta. Gamma."
        sentences = split_sentences(text)
        gates = [asyncio.Event() for _ in sentences]
        gates[-1].set()
        completion_order = []

        async def fake_llm(system, prompt):
            self.assertEqual(system, AGENTIC_METADATA_SYSTEM_PROMPT)
            index = next(
                index for index, sentence in enumerate(sentences)
                if json.dumps(sentence.text.strip()) in prompt
            )
            await gates[index].wait()
            completion_order.append(index + 1)
            if index:
                gates[index - 1].set()
            return json.dumps({"title": f"Title {index}", "summary": "Complete."})

        events = []
        gateway = AgenticLlmGateway(llm_func=fake_llm, retries=0, concurrency=3)
        await gateway.describe_chunks(
            text=text, sentences=sentences,
            boundaries=[(1, 1), (2, 2), (3, 3)], state_events=events,
        )
        self.assertEqual(completion_order, [3, 2, 1])
        self.assertEqual([event["final_boundary"] for event in events], [[1, 1], [2, 2], [3, 3]])
        self.assertTrue(all(event["decision_source"] == "llm" for event in events))


class AgenticTruncatedResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_truncated_proposition_json_is_retried_even_when_parseable(self):
        calls = 0

        async def fake_llm(**kwargs):
            nonlocal calls
            calls += 1
            return LLMResponse(
                '{"propositions":[{"start":1,"end":1}]}',
                finish_reason="length" if calls == 1 else "stop",
                truncated=calls == 1,
            )

        gateway = AgenticLlmGateway(llm_func=fake_llm, retries=1, concurrency=1)
        boundaries = await gateway.extract_propositions(
            sentences=split_sentences("First sentence."), batch_max_sentences=10,
            batch_max_chars=1000, max_sentences=5, state_events=[],
        )
        self.assertEqual(boundaries, [(1, 1)])
        self.assertEqual(calls, 2)

    async def test_truncated_state_cannot_supply_a_routing_action(self):
        calls = 0

        async def fake_llm(**kwargs):
            nonlocal calls
            calls += 1
            return LLMResponse(
                '{"action":"append","title":"Topic","summary":"Valid JSON."}',
                finish_reason="length", truncated=True,
            )

        text = "First sentence. Second sentence."
        gateway = AgenticLlmGateway(llm_func=fake_llm, retries=1, concurrency=1)
        with self.assertRaises(RuntimeError) as raised:
            await gateway.decide_transition(
                proposition_index=2, state_payload={},
                allowed_actions=("append", "new_chunk"), forced_reason=None,
                text=text, sentences=split_sentences(text), proposition_range=(2, 2),
                open_chunk_start=1, fallback_title="Open topic",
            )
        self.assertEqual(calls, 2)
        self.assertIn("truncated", str(raised.exception.__cause__))

    async def test_truncated_final_metadata_uses_audited_source_fallback(self):
        async def fake_llm(**kwargs):
            return LLMResponse(
                '{"title":"Unusable title","summary":"Valid but truncated JSON."}',
                finish_reason="length", truncated=True,
            )

        text = "First source sentence."
        gateway = AgenticLlmGateway(llm_func=fake_llm, retries=0, concurrency=1)
        events = []
        metadata = await gateway.describe_chunks(
            text=text, sentences=split_sentences(text), boundaries=[(1, 1)],
            state_events=events,
        )
        self.assertEqual(metadata[(1, 1)], (text, text))
        self.assertEqual(events[0]["decision_source"], "fallback")
        self.assertIn("truncated", events[0]["fallback_error"])


if __name__ == "__main__":
    unittest.main()
