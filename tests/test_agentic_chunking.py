import json
import unittest

from rag_research.agentic_boundaries import (
    project_boundaries,
    rebalance_document_boundaries,
    validate_boundaries,
)
from rag_research.agentic_chunking import agentic_chunk
from rag_research.agentic_llm import make_sentence_batches
from rag_research.chunking import (
    ChunkConfig,
    ChunkSpan,
    SentenceSpan,
    chunk_async,
)
from rag_research.prompts import (
    AGENTIC_METADATA_SYSTEM_PROMPT,
    AGENTIC_PROPOSITION_SYSTEM_PROMPT,
    AGENTIC_STATE_SYSTEM_PROMPT,
)
from rag_research.text_spans import split_sentences


FOUR_SENTENCES = "One is here. Two is here. Three is here. Four is here."


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


class AgenticBoundaryConstraintTests(unittest.TestCase):
    def test_sentence_spans_are_a_lossless_partition(self):
        text = "  Alpha is here.\n\nBravo is here.  "

        sentences = split_sentences(text)

        self.assertEqual("".join(sentence.text for sentence in sentences), text)
        self.assertEqual(sentences[0].char_start, 0)
        self.assertEqual(sentences[-1].char_end, len(text))
        for left, right in zip(sentences, sentences[1:]):
            self.assertEqual(left.char_end, right.char_start)

    def test_overlapping_ellipsis_spans_are_merged_without_text_loss(self):
        text = "Loading . . .\n\nStudents are here."

        sentences = split_sentences(text)

        self.assertEqual(
            sentences,
            [
                SentenceSpan("Loading . . .\n\n", 0, 15),
                SentenceSpan("Students are here.", 15, len(text)),
            ],
        )
        self.assertEqual("".join(sentence.text for sentence in sentences), text)

    def test_batches_obey_sentence_and_character_limits(self):
        sentences = [
            SentenceSpan("aaaa", 0, 4),
            SentenceSpan("bbbb", 5, 9),
            SentenceSpan("cccc", 10, 14),
            SentenceSpan("dddd", 15, 19),
        ]

        batches = make_sentence_batches(
            sentences,
            max_sentences=3,
            max_chars=9,
        )

        self.assertEqual(batches, [sentences[:2], sentences[2:]])

    def test_validator_rejects_gaps(self):
        with self.assertRaisesRegex(ValueError, "expected chunk to start at 3"):
            validate_boundaries(
                [(1, 2), (4, 4)],
                sentence_count=4,
                min_sentences=1,
                max_sentences=3,
            )

    def test_projection_keeps_valid_boundaries(self):
        boundaries = [(1, 10), (11, 24), (25, 29)]

        projected = project_boundaries(
            boundaries,
            sentence_count=29,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(projected, boundaries)

    def test_projection_splits_an_oversized_unit(self):
        projected = project_boundaries(
            [(1, 26)],
            sentence_count=26,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(projected, [(1, 13), (14, 26)])

    def test_projection_moves_an_undersized_boundary(self):
        projected = project_boundaries(
            [(1, 7), (8, 27)],
            sentence_count=27,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(projected, [(1, 10), (11, 27)])

    def test_document_rebalancing_borrows_when_merge_is_impossible(self):
        rebalanced = rebalance_document_boundaries(
            [(1, 24), (25, 29)],
            sentence_count=29,
            min_sentences=10,
            max_sentences=24,
        )

        self.assertEqual(rebalanced, [(1, 19), (20, 29)])

    def test_projection_is_legal_for_every_supported_batch_size(self):
        for sentence_count in range(1, 61):
            proposals = [
                [(1, sentence_count)],
                [
                    (index, index)
                    for index in range(1, sentence_count + 1)
                ],
            ]
            for proposed in proposals:
                with self.subTest(
                    sentence_count=sentence_count,
                    proposed_chunk_count=len(proposed),
                ):
                    projected = project_boundaries(
                        proposed,
                        sentence_count=sentence_count,
                        min_sentences=10,
                        max_sentences=24,
                    )
                    validate_boundaries(
                        projected,
                        sentence_count=sentence_count,
                        min_sentences=10,
                        max_sentences=24,
                    )


class StatefulAgenticChunkingTests(unittest.IsolatedAsyncioTestCase):
    async def test_tracks_propositions_and_updates_chunk_state(self):
        state_call = 0
        state_prompts: list[str] = []

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal state_call
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return (
                    '{"propositions": ['
                    '{"start": 1, "end": 1}, '
                    '{"start": 2, "end": 2}, '
                    '{"start": 3, "end": 3}, '
                    '{"start": 4, "end": 4}'
                    ']}'
                )
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                state_prompts.append(prompt)
                decisions = [
                    ("new_chunk", "First topic", "The first topic begins."),
                    ("append", "First topic", "The first two statements form one topic."),
                    ("new_chunk", "Second topic", "A different topic begins."),
                    ("append", "Second topic", "The final two statements form another topic."),
                ]
                action, title, summary = decisions[state_call]
                state_call += 1
                return json.dumps({
                    "action": action,
                    "title": title,
                    "summary": summary,
                })
            self.fail(f"unexpected system prompt: {system}")

        events: list[dict[str, object]] = []
        chunks = await agentic_chunk(
            text=FOUR_SENTENCES,
            batch_max_sentences=10,
            batch_max_chars=1000,
            min_sentences=1,
            max_sentences=4,
            concurrency=2,
            retries=0,
            llm_func=fake_llm,
            state_events=events,
        )

        self.assertEqual(
            [chunk.text for chunk in chunks],
            [
                "One is here. Two is here. ",
                "Three is here. Four is here.",
            ],
        )
        self.assertEqual(
            [(chunk.title, chunk.summary) for chunk in chunks],
            [
                ("First topic", "The first two statements form one topic."),
                ("Second topic", "The final two statements form another topic."),
            ],
        )
        self.assertEqual(
            [event["action"] for event in events if event["event"] == "transition"],
            ["new_chunk", "append", "new_chunk", "append"],
        )
        self.assertIn('"status": "closed"', state_prompts[3])
        self.assertIn('"status": "open"', state_prompts[3])
        self.assertEqual("".join(chunk.text for chunk in chunks), FOUR_SENTENCES)
        assert_source_aligned(self, FOUR_SENTENCES, chunks)

    async def test_forces_single_allowed_action_without_retry(self):
        state_calls = 0

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal state_calls
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return '{"propositions": [{"start": 1, "end": 4}]}'
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                state_calls += 1
                return json.dumps({
                    "action": "append",
                    "title": "Forced initial chunk",
                    "summary": "The program forces the only legal action.",
                })
            self.fail(f"unexpected system prompt: {system}")

        chunks = await agentic_chunk(
            text=FOUR_SENTENCES,
            batch_max_sentences=10,
            batch_max_chars=1000,
            min_sentences=1,
            max_sentences=4,
            concurrency=1,
            retries=0,
            llm_func=fake_llm,
        )

        self.assertEqual(state_calls, 1)
        self.assertEqual(chunks[0].title, "Forced initial chunk")

    async def test_retries_an_invalid_choice_when_two_actions_are_allowed(self):
        text = "One. Two."
        state_calls = 0

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal state_calls
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return (
                    '{"propositions": ['
                    '{"start": 1, "end": 1}, '
                    '{"start": 2, "end": 2}'
                    ']}'
                )
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                state_calls += 1
                if state_calls == 1:
                    return json.dumps({
                        "action": "append",
                        "title": "First",
                        "summary": "The forced initial state.",
                    })
                if state_calls == 2:
                    return json.dumps({
                        "action": "merge",
                        "title": "Invalid",
                        "summary": "An unsupported routing action.",
                    })
                self.assertIn("previous response was invalid", prompt)
                return json.dumps({
                    "action": "append",
                    "title": "Combined",
                    "summary": "Both statements remain together.",
                })
            self.fail(f"unexpected system prompt: {system}")

        chunks = await agentic_chunk(
            text=text,
            batch_max_sentences=10,
            batch_max_chars=1000,
            min_sentences=1,
            max_sentences=4,
            concurrency=1,
            retries=1,
            llm_func=fake_llm,
        )

        self.assertEqual(state_calls, 3)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].title, "Combined")

    async def test_invalid_routing_action_never_uses_metadata_fallback(self):
        text = "One. Two."
        state_calls = 0

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal state_calls
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return (
                    '{"propositions": ['
                    '{"start": 1, "end": 1}, '
                    '{"start": 2, "end": 2}'
                    ']}'
                )
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                state_calls += 1
                if state_calls == 1:
                    return json.dumps({
                        "title": "First",
                        "summary": "The initial forced state.",
                    })
                return json.dumps({
                    "action": "merge",
                    "title": "Invalid routing",
                    "summary": "The action is outside the protocol.",
                })
            if system == AGENTIC_METADATA_SYSTEM_PROMPT:
                self.fail("metadata repair must not invent a routing action")
            self.fail(f"unexpected system prompt: {system}")

        with self.assertRaisesRegex(
            RuntimeError,
            "state transition 2 failed after 2 attempts",
        ):
            await agentic_chunk(
                text=text,
                batch_max_sentences=10,
                batch_max_chars=1000,
                min_sentences=1,
                max_sentences=4,
                concurrency=1,
                retries=1,
                llm_func=fake_llm,
            )

    async def test_invalid_state_metadata_uses_dedicated_metadata_repair(self):
        text = "First topic sentence. Second supporting sentence."
        state_calls = 0
        metadata_calls = 0

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal state_calls, metadata_calls
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return (
                    '{"propositions": ['
                    '{"start": 1, "end": 1}, '
                    '{"start": 2, "end": 2}'
                    ']}'
                )
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                state_calls += 1
                if state_calls == 1:
                    return json.dumps({
                        "title": "First topic",
                        "summary": "The initial state.",
                    })
                return json.dumps({
                    "action": "new_chunk",
                    "title": "",
                    "summary": "Missing title on every attempt.",
                })
            if system == AGENTIC_METADATA_SYSTEM_PROMPT:
                metadata_calls += 1
                return json.dumps({
                    "title": "Repaired topic",
                    "summary": "Metadata regenerated from the resulting chunk.",
                })
            self.fail(f"unexpected system prompt: {system}")

        events: list[dict[str, object]] = []
        chunks = await agentic_chunk(
            text=text,
            batch_max_sentences=10,
            batch_max_chars=1000,
            min_sentences=2,
            max_sentences=4,
            concurrency=1,
            retries=2,
            llm_func=fake_llm,
            state_events=events,
        )

        transitions = [
            event for event in events if event["event"] == "transition"
        ]
        self.assertEqual(state_calls, 4)
        self.assertEqual(metadata_calls, 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(transitions[-1]["action"], "append")
        self.assertEqual(
            transitions[-1]["decision_source"],
            "metadata_repair",
        )
        self.assertIn("non-empty title", transitions[-1]["recovery_error"])
        self.assertNotIn("fallback_error", transitions[-1])
        self.assertEqual(chunks[0].title, "Repaired topic")

    async def test_metadata_repair_failure_uses_audited_source_fallback(self):
        text = "First topic sentence. Second supporting sentence."

        async def fake_llm(system: str, prompt: str) -> str:
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return (
                    '{"propositions": ['
                    '{"start": 1, "end": 1}, '
                    '{"start": 2, "end": 2}'
                    ']}'
                )
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                if '"index": 1' in prompt:
                    return json.dumps({
                        "title": "First topic",
                        "summary": "The initial state.",
                    })
                return json.dumps({
                    "title": "",
                    "summary": "Missing title.",
                })
            if system == AGENTIC_METADATA_SYSTEM_PROMPT:
                return json.dumps({"title": "", "summary": "Still invalid."})
            self.fail(f"unexpected system prompt: {system}")

        events: list[dict[str, object]] = []
        chunks = await agentic_chunk(
            text=text,
            batch_max_sentences=10,
            batch_max_chars=1000,
            min_sentences=2,
            max_sentences=4,
            concurrency=1,
            retries=1,
            llm_func=fake_llm,
            state_events=events,
        )

        transitions = [
            event for event in events if event["event"] == "transition"
        ]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(transitions[-1]["action"], "append")
        self.assertEqual(transitions[-1]["decision_source"], "fallback")
        self.assertIn("non-empty title", transitions[-1]["fallback_error"])
        self.assertEqual(chunks[0].title, "First topic")
        self.assertIn("Second supporting sentence", chunks[0].summary)

    async def test_hard_size_constraints_limit_allowed_actions(self):
        text = "One. Two. Three. Four."
        state_call = 0
        observed_prompts: list[str] = []

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal state_call
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return (
                    '{"propositions": ['
                    '{"start": 1, "end": 1}, '
                    '{"start": 2, "end": 2}, '
                    '{"start": 3, "end": 3}, '
                    '{"start": 4, "end": 4}'
                    ']}'
                )
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                observed_prompts.append(prompt)
                actions = ["append", "new_chunk", "append", "new_chunk"]
                action = actions[state_call]
                state_call += 1
                return json.dumps({
                    "action": action,
                    "title": f"Topic {1 if state_call <= 2 else 2}",
                    "summary": "Size-constrained state.",
                })
            if system == AGENTIC_METADATA_SYSTEM_PROMPT:
                return json.dumps({
                    "title": "Final",
                    "summary": "Final metadata.",
                })
            self.fail(f"unexpected system prompt: {system}")

        chunks = await agentic_chunk(
            text=text,
            batch_max_sentences=10,
            batch_max_chars=1000,
            min_sentences=2,
            max_sentences=2,
            concurrency=1,
            retries=0,
            llm_func=fake_llm,
        )

        self.assertEqual(
            [len(split_sentences(chunk.text)) for chunk in chunks],
            [2, 2],
        )
        self.assertIn('"allowed_actions": [\n      "append"', observed_prompts[1])
        self.assertIn('"allowed_actions": [\n      "new_chunk"', observed_prompts[2])

    async def test_proposition_projection_is_audited(self):
        text = " ".join(f"Sentence {index}." for index in range(1, 27))
        state_call = 0

        async def fake_llm(system: str, prompt: str) -> str:
            nonlocal state_call
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return '{"propositions": [{"start": 1, "end": 26}]}'
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                state_call += 1
                return json.dumps({
                    "action": "new_chunk",
                    "title": f"Part {state_call}",
                    "summary": "A projected proposition group.",
                })
            self.fail(f"unexpected system prompt: {system}")

        events: list[dict[str, object]] = []
        chunks = await agentic_chunk(
            text=text,
            batch_max_sentences=30,
            batch_max_chars=10000,
            min_sentences=10,
            max_sentences=24,
            concurrency=1,
            retries=0,
            llm_func=fake_llm,
            state_events=events,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(events[0]["event"], "proposition_projection")
        self.assertEqual(events[0]["original_boundaries"], [[1, 26]])
        assert_source_aligned(self, text, chunks)

    async def test_chunk_async_dispatches_agentic_strategy(self):
        async def fake_llm(system: str, prompt: str) -> str:
            if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                return '{"propositions": [{"start": 1, "end": 4}]}'
            if system == AGENTIC_STATE_SYSTEM_PROMPT:
                return json.dumps({
                    "action": "new_chunk",
                    "title": "All",
                    "summary": "All four statements.",
                })
            self.fail(f"unexpected system prompt: {system}")

        config = ChunkConfig(
            strategy="agentic",
            agentic_batch_max_sentences=10,
            agentic_batch_max_chars=1000,
            agentic_min_sentences=1,
            agentic_max_sentences=4,
            agentic_concurrency=1,
            agentic_retries=0,
        )

        chunks = await chunk_async(FOUR_SENTENCES, config, llm_func=fake_llm)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, FOUR_SENTENCES)
        self.assertEqual(chunks[0].title, "All")

    async def test_chunk_async_requires_llm(self):
        with self.assertRaisesRegex(ValueError, "requires an llm_func"):
            await chunk_async(FOUR_SENTENCES, ChunkConfig(strategy="agentic"))


if __name__ == "__main__":
    unittest.main()
