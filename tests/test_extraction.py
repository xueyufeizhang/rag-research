import asyncio
import json
import itertools
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rag_research.extraction import Entity, Relation, _parse_response, extract
from rag_research.llm import LLMResponse
from rag_research.models import ChunkRecord
from rag_research.prompts import PROMPTS


def _chunk(chunk_id: str, model_text: str = "model input") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="doc-1",
        text="source text",
        model_text=model_text,
        chunk_index=0,
        char_start=0,
        char_end=len("source text"),
    )


def _response(
    entities: list[dict] | None = None,
    relationships: list[dict] | None = None,
) -> str:
    normalized_entities = []
    for entity in entities or []:
        normalized_entities.append({
            "type": "Other",
            "description": f"{entity.get('name', 'Entity')} description",
            **entity,
        })
    normalized_relationships = []
    for relationship in relationships or []:
        normalized_relationships.append({
            "keywords": ["related"],
            "description": "The entities are directly related.",
            **relationship,
        })
    return json.dumps({
        "entities": normalized_entities,
        "relationships": normalized_relationships,
    })


class ParseResponseTests(unittest.TestCase):
    def test_every_few_shot_output_passes_the_real_parser_on_its_input(self):
        for index, example in enumerate(PROMPTS["entity_extraction_examples"]):
            with self.subTest(example=index):
                input_text = re.search(r"---Input Text---\s*```\s*(.*?)\s*```", example, re.DOTALL).group(1)
                output = example.split("---Output---", 1)[1]
                entities, relations = _parse_response(output, f"example-{index}", input_text)
                self.assertTrue(entities)
                self.assertTrue(relations)

    def test_grounding_accepts_only_orthographic_name_variations(self):
        for spelling in ("peat-soil", "peat\u2011soil", "Ｐｅａｔ　Ｓｏｉｌ", "peat\n  soil"):
            with self.subTest(spelling=spelling):
                entities, _ = _parse_response(
                    _response(entities=[{"name": "Peat Soil"}]),
                    "grounding", f"The {spelling} was damaged.",
                )
                self.assertEqual(entities[0].name, "Peat Soil")
        for text in ("Mineral soil was damaged.", "Peat soils were damaged.", "Soil containing peat was damaged."):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "example leakage"):
                    _parse_response(_response(entities=[{"name": "Peat-Soil"}]), "grounding", text)

    def test_parses_and_cleans_valid_records(self):
        response = _response(
            entities=[
                {
                    "name": "  Alice  ",
                    "type": "  Person ",
                    "description": "  Researcher  ",
                },
                {
                    "name": " Bob ",
                    "type": " person ",
                    "description": " Engineer ",
                },
            ],
            relationships=[
                {
                    "source": " Alice ",
                    "target": " Bob ",
                    "keywords": [" works with ", ""],
                    "description": " colleague ",
                },
            ],
        )

        entities, relations = _parse_response(
            response,
            "doc-1:chunk:0",
            "Alice works with Bob.",
        )

        self.assertEqual(
            entities,
            [
                Entity("Alice", "Person", "Researcher", ["doc-1:chunk:0"]),
                Entity("Bob", "Person", "Engineer", ["doc-1:chunk:0"]),
            ],
        )
        self.assertEqual(
            relations,
            [
                Relation(
                    "Alice",
                    "Bob",
                    ["works with"],
                    "colleague",
                    ["doc-1:chunk:0"],
                )
            ],
        )

    def test_accepts_explicitly_empty_extraction(self):
        self.assertEqual(
            _parse_response(_response(), "doc-1:chunk:0", "model input"),
            ([], []),
        )

    def test_rejects_missing_or_unknown_entity_fields(self):
        invalid_entities = (
            {"name": "Alpha", "type": "Person"},
            {"name": "Alpha", "description": "A person."},
            {"name": "Alpha", "type": "Service", "description": "A service."},
            {"name": "Alpha", "type": "Person", "description": ""},
        )
        for entity in invalid_entities:
            with self.subTest(entity=entity):
                response = json.dumps({
                    "entities": [entity],
                    "relationships": [],
                })
                with self.assertRaises(ValueError):
                    _parse_response(
                        response,
                        "doc-1:chunk:0",
                        "Alpha is present.",
                    )

    def test_rejects_incomplete_and_self_relationships(self):
        entities = [
            {
                "name": "Alpha",
                "type": "Concept",
                "description": "Alpha is a concept.",
            },
            {
                "name": "Beta",
                "type": "Concept",
                "description": "Beta is a concept.",
            },
        ]
        invalid_relationships = (
            {
                "source": "Alpha",
                "target": "Beta",
                "description": "Alpha relates to Beta.",
            },
            {
                "source": "Alpha",
                "target": "Beta",
                "keywords": ["related"],
            },
            {
                "source": "Alpha",
                "target": "alpha",
                "keywords": ["identity"],
                "description": "Invalid self relationship.",
            },
        )
        for relationship in invalid_relationships:
            with self.subTest(relationship=relationship):
                response = json.dumps({
                    "entities": entities,
                    "relationships": [relationship],
                })
                with self.assertRaises(ValueError):
                    _parse_response(
                        response,
                        "doc-1:chunk:0",
                        "Alpha relates to Beta.",
                    )

    def test_rejects_empty_or_missing_json(self):
        for response in ("", "   ", "no JSON here"):
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    _parse_response(response, "doc-1:chunk:0", "model input")

    def test_requires_both_top_level_arrays(self):
        invalid_responses = (
            "{}",
            '{"entities": []}',
            '{"relationships": []}',
            '{"entities": null, "relationships": []}',
            '{"entities": [], "relationships": {}}',
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    _parse_response(response, "doc-1:chunk:0", "model input")

    def test_rejects_relationship_with_missing_response_entity(self):
        response = _response(
            entities=[{"name": "Acme"}],
            relationships=[{"source": "Acme", "target": "Missing"}],
        )

        with self.assertRaisesRegex(ValueError, "endpoints must appear"):
            _parse_response(response, "doc-1:chunk:0", "Acme bought supplies.")

    def test_enforces_raw_entity_and_total_record_limits(self):
        too_many_entities = _response(
            entities=[{"name": f"Entity {index}"} for index in range(21)],
        )
        with self.assertRaisesRegex(
            ValueError,
            "entity count 21 exceeds maximum 20",
        ):
            _parse_response(
                too_many_entities,
                "doc-1:chunk:0",
                "model input",
            )

        too_many_total_records = _response(
            entities=[{"name": "Alpha"}, {"name": "Beta"}],
            relationships=[
                {"source": "Alpha", "target": "Beta"}
                for _ in range(49)
            ],
        )
        with self.assertRaisesRegex(
            ValueError,
            "total record count 51 exceeds maximum 50",
        ):
            _parse_response(
                too_many_total_records,
                "doc-1:chunk:0",
                "Alpha and Beta are related.",
            )

    def test_deduplicates_entities_and_undirected_relationships(self):
        response = _response(
            entities=[
                {
                    "name": "Alpha",
                    "type": "Organization",
                    "description": "First description",
                },
                {
                    "name": " alpha ",
                    "type": "Organization",
                    "description": "Second description",
                },
                {
                    "name": "Beta",
                    "type": "Person",
                    "description": "Beta description",
                },
            ],
            relationships=[
                {
                    "source": "Alpha",
                    "target": "Beta",
                    "keywords": ["works with", "Works With"],
                    "description": "First relation",
                },
                {
                    "source": "beta",
                    "target": "ALPHA",
                    "keywords": "collaboration, works with",
                    "description": "Second relation",
                },
            ],
        )

        entities, relations = _parse_response(
            response,
            "doc-1:chunk:0",
            "Alpha works with Beta.",
        )

        self.assertEqual(
            entities,
            [
                Entity(
                    "Alpha",
                    "Organization",
                    "First description | Second description",
                    ["doc-1:chunk:0"],
                ),
                Entity(
                    "Beta",
                    "Person",
                    "Beta description",
                    ["doc-1:chunk:0"],
                ),
            ],
        )
        self.assertEqual(
            relations,
            [
                Relation(
                    "Alpha",
                    "Beta",
                    ["works with", "collaboration"],
                    "First relation | Second relation",
                    ["doc-1:chunk:0"],
                )
            ],
        )

    def test_rejects_ungrounded_prompt_example_entity(self):
        response = _response(
            entities=[{"name": "Dr. Elena Vasquez", "type": "Person"}],
        )

        with self.assertRaisesRegex(ValueError, "prompt example leakage"):
            _parse_response(
                response,
                "doc-1:chunk:0",
                "These headphones are discounted today.",
            )

    def test_allows_prompt_example_name_when_grounded_in_input(self):
        response = _response(
            entities=[{"name": "Dr. Elena Vasquez", "type": "Person"}],
        )

        entities, relations = _parse_response(
            response,
            "doc-1:chunk:0",
            "Dr. Elena Vasquez led the expedition.",
        )

        self.assertEqual(
            entities,
            [
                Entity(
                    "Dr. Elena Vasquez",
                    "Person",
                    "Dr. Elena Vasquez description",
                    ["doc-1:chunk:0"],
                )
            ],
        )
        self.assertEqual(relations, [])


class ExtractTests(unittest.IsolatedAsyncioTestCase):
    async def test_cap_statistics_use_raw_counts_and_explicit_chunk_denominators(self):
        names = [f"Worker {index}" for index in range(20)]
        pairs = list(itertools.combinations(names, 2))[:30]
        capped = _response(
            entities=[{"name": name} for name in names],
            relationships=[{"source": source, "target": target} for source, target in pairs],
        )
        result = await extract(
            [_chunk("capped"), _chunk("small")],
            AsyncMock(side_effect=[LLMResponse(capped, "stop", truncated=False), _response()]),
            con_num=1,
        )
        summary = result.statistics["run_summary"]
        self.assertEqual(summary["attempt_count"], 2)
        for limit_name in ("entity_limit", "total_limit"):
            metric = summary[limit_name]
            self.assertEqual(metric["known_count_chunk_count"], 2)
            self.assertEqual(metric["exactly_at_limit_attempt_count"], 1)
            self.assertEqual(metric["at_or_above_limit_rate_per_known_chunk"], 0.5)
        self.assertEqual(summary["truncation_known_response_attempt_count"], 1)
        self.assertEqual(summary["truncation_unknown_response_attempt_count"], 1)
        self.assertEqual(summary["output_truncation_rate_per_known_response"], 0.0)

        duplicates = _response(entities=[{"name": "Repeated"} for _ in range(20)])
        duplicate_result = await extract([_chunk("duplicates")], AsyncMock(return_value=duplicates), con_num=1)
        self.assertEqual(len(duplicate_result.entities), 1)
        self.assertEqual(duplicate_result.statistics["chunks"][0]["attempts"][0]["raw_entity_count"], 20)
        self.assertEqual(duplicate_result.statistics["run_summary"]["entity_limit"]["exactly_at_limit_attempt_count"], 1)
        self.assertEqual(duplicate_result.statistics["run_summary"]["accepted_entity_limit"]["exactly_at_limit_attempt_count"], 0)
        self.assertEqual(duplicate_result.statistics["chunks"][0]["attempts"][0]["accepted_entity_count"], 1)

    async def test_repaired_json_has_unknown_raw_counts_but_observable_accepted_counts(self):
        malformed = "{'entities': [{'name': 'Alpha', 'type': 'Person', 'description': 'A person.'}], 'relationships': []}"
        result = await extract([_chunk("repaired")], AsyncMock(return_value=malformed), 1)
        self.assertEqual(result.failed_chunk_ids, [])
        attempt = result.statistics["chunks"][0]["attempts"][0]
        self.assertEqual(attempt["raw_count_source"], "unavailable")
        self.assertIsNone(attempt["raw_entity_count"])
        self.assertEqual(attempt["accepted_entity_count"], 1)
        summary = result.statistics["run_summary"]
        self.assertEqual(summary["entity_limit"]["known_count_attempt_count"], 0)
        self.assertEqual(summary["accepted_entity_limit"]["known_count_attempt_count"], 1)

    async def test_attempt_start_is_durable_before_the_backend_is_called(self):
        with tempfile.TemporaryDirectory() as directory:
            async def llm(*, system: str, prompt: str) -> str:
                payload = json.loads(next(Path(directory, "f", "attempts").glob("*.json")).read_text())
                attempt = payload["attempts"][0]
                self.assertEqual(attempt["outcome"], "pending")
                self.assertIsNone(attempt["duration_seconds"])
                self.assertFalse(attempt["response_received"])
                self.assertIsNone(attempt["raw_entity_count"])
                return _response()

            await extract(
                [_chunk("durable")], llm, 1,
                cache_directory=directory, extraction_fingerprint="f", cache_scope="scope",
            )
            payload = json.loads(next(Path(directory, "f", "attempts").glob("*.json")).read_text())
            self.assertEqual(len(payload["attempts"]), 1)
            self.assertEqual(payload["attempts"][0]["outcome"], "success")

    async def test_resume_preserves_unresolved_historical_attempts_as_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            captured = {}

            async def interrupted_llm(*, system: str, prompt: str) -> str:
                path = next(Path(directory, "f", "attempts").glob("*.json"))
                captured["path"] = path
                captured["pending"] = path.read_text()
                raise asyncio.CancelledError()

            arguments = dict(cache_directory=directory, extraction_fingerprint="f", cache_scope="scope")
            with self.assertRaises(asyncio.CancelledError):
                await extract([_chunk("interrupted")], interrupted_llm, 1, **arguments)
            # Reproduce the exact ledger state a hard kill leaves before the
            # running call can report any outcome or completion metadata.
            captured["path"].write_text(captured["pending"])
            resumed = await extract([_chunk("interrupted")], AsyncMock(return_value=_response()), 1, **arguments)
            history = resumed.statistics["historical_summary"]
            self.assertEqual(history["attempt_count"], 1)
            self.assertEqual(history["unresolved_attempt_count"], 1)
            self.assertEqual(history["response_attempt_count"], 0)
            self.assertIsNone(history["output_truncation_rate_per_known_response"])
            self.assertEqual(resumed.statistics["run_summary"]["attempt_count"], 1)
            attempts = resumed.statistics["chunks"][0]["attempts"]
            self.assertEqual([attempt["outcome"] for attempt in attempts], ["pending", "success"])

    async def test_five_limit_errors_only_create_four_actual_retries(self):
        response = _response(entities=[{"name": f"Entity {index}"} for index in range(21)])
        with patch("rag_research.extraction.asyncio.sleep", new=AsyncMock()):
            result = await extract(
                [_chunk("over-limit")],
                AsyncMock(return_value=LLMResponse(response, "stop", truncated=False)),
                con_num=1,
            )
        summary = result.statistics["run_summary"]
        self.assertEqual(summary["attempt_count"], 5)
        self.assertEqual(summary["over_limit_attempt_count"], 5)
        self.assertEqual(summary["over_limit_error_attempt_count"], 5)
        self.assertEqual(summary["over_limit_retry_count"], 4)
        self.assertEqual(summary["over_limit_retry_rate_per_violation"], 0.8)
        self.assertEqual(summary["retry_rate_per_attempted_chunk"], 1.0)
        attempts = result.statistics["chunks"][0]["attempts"]
        self.assertIsNone(attempts[0]["retry_of"])
        self.assertEqual(attempts[-1]["retry_of"], attempts[-2]["attempt_id"])
        self.assertTrue(all(attempt["raw_entity_count"] == 21 for attempt in attempts))

    async def test_explicit_truncation_retries_even_when_json_is_valid(self):
        llm = AsyncMock(side_effect=[
            LLMResponse(_response(), "length", {"completion_tokens": 4096}, True),
            LLMResponse(_response(), "stop", {"completion_tokens": 20}, False),
        ])
        with tempfile.TemporaryDirectory() as directory:
            with patch("rag_research.extraction.asyncio.sleep", new=AsyncMock()):
                result = await extract(
                    [_chunk("truncated")], llm, 1,
                    cache_directory=directory, extraction_fingerprint="f", cache_scope="scope",
                )
            summary = result.statistics["run_summary"]
            self.assertEqual(summary["output_truncation_rate_per_known_response"], 0.5)
            self.assertEqual(summary["retry_count"], 1)
            record = json.loads(next(Path(directory, "f", "records").glob("*.json")).read_text())
            self.assertEqual([attempt["outcome"] for attempt in record["attempts"]], ["truncated", "success"])
            self.assertEqual(record["attempts"][0]["usage"]["completion_tokens"], 4096)

    async def test_invalid_json_does_not_invent_a_truncation_signal(self):
        with patch("rag_research.extraction.asyncio.sleep", new=AsyncMock()):
            result = await extract([_chunk("unknown")], AsyncMock(return_value="broken JSON"), 1)
        summary = result.statistics["run_summary"]
        self.assertEqual(summary["truncation_unknown_response_attempt_count"], 5)
        self.assertEqual(summary["truncated_response_attempt_count"], 0)
        self.assertIsNone(summary["output_truncation_rate_per_known_response"])
        self.assertIsNone(summary["entity_limit"]["at_or_above_limit_rate_per_known_attempt"])
        self.assertTrue(all(attempt["raw_entity_count"] is None for attempt in result.statistics["chunks"][0]["attempts"]))

    async def test_failed_attempt_history_survives_resume_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = dict(cache_directory=directory, extraction_fingerprint="f", cache_scope="scope")
            chunk = _chunk("resumed")
            with patch("rag_research.extraction.asyncio.sleep", new=AsyncMock()):
                failed = await extract([chunk], AsyncMock(side_effect=OSError("offline")), 1, **arguments)
            failed_report = json.loads(Path(failed.statistics["report_path"]).read_text())
            self.assertEqual(failed_report["status"], "incomplete")
            self.assertEqual(failed_report["run_summary"]["backend_error_attempt_count"], 5)
            self.assertEqual(failed_report["run_summary"]["response_attempt_count"], 0)
            resumed = await extract([chunk], AsyncMock(return_value=_response()), 1, **arguments)
            self.assertEqual(resumed.statistics["run_summary"]["attempt_count"], 1)
            self.assertEqual(resumed.statistics["historical_summary"]["attempt_count"], 5)
            self.assertEqual(resumed.statistics["cumulative_summary"]["attempt_count"], 6)
            self.assertEqual(resumed.statistics["run_summary"]["retry_count"], 0)
            final_llm = AsyncMock(side_effect=AssertionError("cache must be reused"))
            cached = await extract([chunk], final_llm, 1, **arguments)
            final_llm.assert_not_awaited()
            self.assertEqual(cached.statistics["cached_chunk_count"], 1)
            self.assertEqual(cached.statistics["run_summary"]["attempt_count"], 0)
            self.assertIsNone(cached.statistics["run_summary"]["output_truncation_rate_per_known_response"])
            self.assertEqual(cached.statistics["historical_summary"]["attempt_count"], 6)
            self.assertEqual(len(list(Path(directory, "f", "runs", "scope").glob("*.json"))), 3)

    async def test_cancellation_during_backoff_does_not_count_an_unmade_retry(self):
        response = _response(entities=[{"name": f"Entity {index}"} for index in range(21)])
        with tempfile.TemporaryDirectory() as directory:
            with patch("rag_research.extraction.asyncio.sleep", new=AsyncMock(side_effect=asyncio.CancelledError)):
                with self.assertRaises(asyncio.CancelledError):
                    await extract(
                        [_chunk("cancelled")], AsyncMock(return_value=response), 1,
                        cache_directory=directory, extraction_fingerprint="f", cache_scope="scope",
                    )
            report = json.loads(next(Path(directory, "f", "runs", "scope").glob("*.json")).read_text())
            self.assertEqual(report["status"], "interrupted")
            self.assertEqual(report["run_summary"]["over_limit_attempt_count"], 1)
            self.assertEqual(report["run_summary"]["over_limit_retry_count"], 0)

    async def test_statistics_versions_and_missing_attempt_fields_are_rejected(self):
        for corruption in ("record-version", "state-version", "ledger-version", "missing-count"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                arguments = dict(cache_directory=directory, extraction_fingerprint="f", cache_scope="scope")
                chunk = _chunk("corrupted")
                await extract([chunk], AsyncMock(return_value=_response()), 1, **arguments)
                if corruption == "state-version":
                    path = Path(directory, "f", "states", "scope.json")
                else:
                    folder = "attempts" if corruption == "ledger-version" else "records"
                    path = next(Path(directory, "f", folder).glob("*.json"))
                payload = json.loads(path.read_text())
                if corruption == "record-version":
                    payload["statistics_schema_version"] = 999
                elif corruption == "state-version":
                    payload["statistics"]["schema_version"] = 999
                elif corruption == "ledger-version":
                    payload["schema_version"] = 999
                else:
                    del payload["attempts"][0]["raw_entity_count"]
                path.write_text(json.dumps(payload))
                llm = AsyncMock(side_effect=AssertionError("invalid statistics must not run models"))
                with self.assertRaisesRegex(ValueError, "statistics"):
                    await extract([chunk], llm, 1, **arguments)
                llm.assert_not_awaited()

    async def test_cache_rejects_accepted_statistics_that_disagree_with_records(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = dict(cache_directory=directory, extraction_fingerprint="f", cache_scope="scope")
            chunk = _chunk("inconsistent-accepted-count")
            await extract([chunk], AsyncMock(return_value=_response()), 1, **arguments)
            path = next(Path(directory, "f", "records").glob("*.json"))
            payload = json.loads(path.read_text())
            payload["attempts"][-1]["accepted_entity_count"] = 1
            payload["attempts"][-1]["accepted_total_count"] = 1
            path.write_text(json.dumps(payload))
            llm = AsyncMock(side_effect=AssertionError("corrupt cache must not invoke models"))
            with self.assertRaisesRegex(ValueError, "statistics disagree with accepted records"):
                await extract([chunk], llm, 1, **arguments)
            llm.assert_not_awaited()

    async def test_validation_retry_includes_contract_feedback(self):
        prompts: list[str] = []

        async def llm_func(*, system: str, prompt: str) -> str:
            prompts.append(prompt)
            if len(prompts) == 1:
                return json.dumps({
                    "entities": [{
                        "name": "Alpha",
                        "type": "Service",
                        "description": "Alpha is a service.",
                    }],
                    "relationships": [],
                })
            return _response(entities=[{
                "name": "Alpha",
                "type": "Other",
                "description": "Alpha is an entity.",
            }])

        with patch(
            "rag_research.extraction.asyncio.sleep",
            new=AsyncMock(),
        ):
            result = await extract(
                [_chunk("doc-1:chunk:0", "Alpha is present.")],
                llm_func,
                con_num=1,
            )

        self.assertEqual(len(prompts), 2)
        self.assertIn("Correction Required", prompts[1])
        self.assertIn("unknown entity type", prompts[1])
        self.assertEqual(result.entities[0].type, "Other")
        self.assertEqual(result.failed_chunk_ids, [])

    async def test_uses_model_text_and_preserves_chunk_id(self):
        prompts: list[str] = []

        async def llm_func(*, system: str, prompt: str) -> str:
            self.assertTrue(system)
            prompts.append(prompt)
            return _response(
                entities=[{"name": "Unique Model Text", "type": "Content"}],
            )

        result = await extract(
            [_chunk("doc-1:chunk:0", model_text="UNIQUE MODEL TEXT")],
            llm_func,
            con_num=1,
        )

        self.assertIn("UNIQUE MODEL TEXT", prompts[0])
        self.assertNotIn("source text", prompts[0])
        self.assertEqual(result.entities[0].source_id, ["doc-1:chunk:0"])
        self.assertEqual(result.failed_chunk_ids, [])

    async def test_invalid_schema_is_retried_then_reported(self):
        llm_func = AsyncMock(return_value="{}")

        with patch(
                "rag_research.extraction.asyncio.sleep",
                new=AsyncMock(),
        ) as sleep_mock:
            result = await extract(
                [_chunk("doc-1:chunk:0")],
                llm_func,
                con_num=1,
            )

        self.assertEqual(llm_func.await_count, 5)
        self.assertEqual(sleep_mock.await_count, 4)
        self.assertEqual(result.entities, [])
        self.assertEqual(result.relations, [])
        self.assertEqual(result.failed_chunk_ids, ["doc-1:chunk:0"])

    async def test_llm_failure_is_retried_then_reported(self):
        llm_func = AsyncMock(side_effect=OSError("provider unavailable"))

        with patch(
                "rag_research.extraction.asyncio.sleep",
                new=AsyncMock(),
        ):
            result = await extract(
                [_chunk("doc-1:chunk:0")],
                llm_func,
                con_num=1,
            )

        self.assertEqual(llm_func.await_count, 5)
        self.assertEqual(result.failed_chunk_ids, ["doc-1:chunk:0"])

    async def test_retry_backoff_does_not_hold_concurrency_slot(self):
        healthy_called = asyncio.Event()
        retry_calls = 0

        async def llm_func(*, system: str, prompt: str) -> str:
            nonlocal retry_calls
            if "RETRY CHUNK" in prompt:
                retry_calls += 1
                if retry_calls == 1:
                    raise OSError("temporary failure")
            else:
                healthy_called.set()
            return _response()

        async def wait_for_healthy_chunk(_: float) -> None:
            await healthy_called.wait()

        with patch(
                "rag_research.extraction.asyncio.sleep",
                side_effect=wait_for_healthy_chunk,
        ):
            result = await asyncio.wait_for(
                extract(
                    [
                        _chunk("doc-1:chunk:0", "RETRY CHUNK"),
                        _chunk("doc-1:chunk:1", "HEALTHY CHUNK"),
                    ],
                    llm_func,
                    con_num=1,
                ),
                timeout=1,
            )

        self.assertEqual(retry_calls, 2)
        self.assertEqual(result.failed_chunk_ids, [])

    async def test_never_exceeds_configured_concurrency(self):
        active_calls = 0
        peak_calls = 0
        two_calls_started = asyncio.Event()

        async def llm_func(*, system: str, prompt: str) -> str:
            nonlocal active_calls, peak_calls
            active_calls += 1
            peak_calls = max(peak_calls, active_calls)
            if active_calls == 2:
                two_calls_started.set()
            await two_calls_started.wait()
            await asyncio.sleep(0)
            active_calls -= 1
            return _response()

        result = await extract(
            [_chunk(f"doc-1:chunk:{i}") for i in range(5)],
            llm_func,
            con_num=2,
        )

        self.assertEqual(peak_calls, 2)
        self.assertEqual(result.failed_chunk_ids, [])

    async def test_unexpected_parser_error_is_not_silenced(self):
        llm_func = AsyncMock(return_value=_response())

        with patch(
                "rag_research.extraction._parse_response",
                side_effect=RuntimeError("programming error"),
        ):
            with self.assertRaisesRegex(RuntimeError, "programming error"):
                await extract(
                    [_chunk("doc-1:chunk:0")],
                    llm_func,
                    con_num=1,
                )

        self.assertEqual(llm_func.await_count, 1)

    async def test_empty_input_returns_empty_result(self):
        llm_func = AsyncMock(return_value=_response())

        result = await extract([], llm_func, con_num=1)

        self.assertEqual(result.entities, [])
        self.assertEqual(result.relations, [])
        self.assertEqual(result.failed_chunk_ids, [])
        llm_func.assert_not_awaited()

    async def test_rejects_non_positive_concurrency(self):
        llm_func = AsyncMock(return_value=_response())

        for con_num in (0, -1):
            with self.subTest(con_num=con_num):
                with self.assertRaises(ValueError):
                    await extract([], llm_func, con_num=con_num)

        llm_func.assert_not_awaited()

    async def test_successful_chunks_are_cached_and_only_failures_are_retried(self):
        chunks = [
            _chunk("doc-1:chunk:0", "FIRST CHUNK"),
            _chunk("doc-1:chunk:1", "SECOND CHUNK"),
        ]

        with tempfile.TemporaryDirectory() as directory:
            async def first_llm(*, system: str, prompt: str) -> str:
                if "FIRST CHUNK" in prompt:
                    return _response(entities=[{"name": "First"}])
                raise OSError("temporary failure")

            with patch(
                "rag_research.extraction.asyncio.sleep",
                new=AsyncMock(),
            ):
                first_result = await extract(
                    chunks,
                    first_llm,
                    con_num=2,
                    cache_directory=directory,
                    extraction_fingerprint="extraction-a",
                    cache_scope="build-a",
                )

            self.assertEqual(
                first_result.failed_chunk_ids,
                ["doc-1:chunk:1"],
            )
            state_path = Path(
                directory,
                "extraction-a",
                "states",
                "build-a.json",
            )
            first_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(first_state["status"], "incomplete")
            self.assertEqual(first_state["completed_chunk_count"], 1)
            self.assertEqual(
                len(list(Path(directory, "extraction-a", "records").glob("*.json"))),
                1,
            )

            second_prompts: list[str] = []

            async def second_llm(*, system: str, prompt: str) -> str:
                second_prompts.append(prompt)
                return _response(entities=[{"name": "Second"}])

            second_result = await extract(
                chunks,
                second_llm,
                con_num=2,
                cache_directory=directory,
                extraction_fingerprint="extraction-a",
                cache_scope="build-a",
            )

            self.assertEqual(len(second_prompts), 1)
            self.assertIn("SECOND CHUNK", second_prompts[0])
            self.assertNotIn("FIRST CHUNK", second_prompts[0])
            self.assertEqual(
                [entity.name for entity in second_result.entities],
                ["First", "Second"],
            )
            self.assertEqual(second_result.failed_chunk_ids, [])
            second_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(second_state["status"], "complete")
            self.assertEqual(second_state["completed_chunk_count"], 2)

    async def test_cache_is_isolated_by_extraction_fingerprint(self):
        chunk = _chunk("doc-1:chunk:0")

        with tempfile.TemporaryDirectory() as directory:
            first_llm = AsyncMock(return_value=_response())
            await extract(
                [chunk],
                first_llm,
                con_num=1,
                cache_directory=directory,
                extraction_fingerprint="extraction-a",
                cache_scope="build-a",
            )

            second_llm = AsyncMock(return_value=_response())
            await extract(
                [chunk],
                second_llm,
                con_num=1,
                cache_directory=directory,
                extraction_fingerprint="extraction-b",
                cache_scope="build-b",
            )

            first_llm.assert_awaited_once()
            second_llm.assert_awaited_once()
            self.assertTrue(Path(directory, "extraction-a").is_dir())
            self.assertTrue(Path(directory, "extraction-b").is_dir())

    async def test_cache_identity_includes_model_input(self):
        first_chunk = _chunk("stable-chunk-id", "FIRST MODEL INPUT")
        second_chunk = _chunk("stable-chunk-id", "SECOND MODEL INPUT")

        with tempfile.TemporaryDirectory() as directory:
            first_llm = AsyncMock(
                return_value=_response(entities=[{"name": "First"}])
            )
            await extract(
                [first_chunk],
                first_llm,
                con_num=1,
                cache_directory=directory,
                extraction_fingerprint="extraction-a",
                cache_scope="build-a",
            )

            second_llm = AsyncMock(
                return_value=_response(entities=[{"name": "Second"}])
            )
            second_result = await extract(
                [second_chunk],
                second_llm,
                con_num=1,
                cache_directory=directory,
                extraction_fingerprint="extraction-a",
                cache_scope="build-b",
            )

            first_llm.assert_awaited_once()
            second_llm.assert_awaited_once()
            self.assertEqual(
                [entity.name for entity in second_result.entities],
                ["Second"],
            )
            self.assertEqual(
                len(list(
                    Path(directory, "extraction-a", "records").glob("*.json")
                )),
                2,
            )

    async def test_corrupted_cache_record_is_rejected(self):
        chunk = _chunk("doc-1:chunk:0")

        with tempfile.TemporaryDirectory() as directory:
            await extract(
                [chunk],
                AsyncMock(return_value=_response()),
                con_num=1,
                cache_directory=directory,
                extraction_fingerprint="extraction-a",
                cache_scope="build-a",
            )
            record_path = next(
                Path(directory, "extraction-a", "records").glob("*.json")
            )
            record_path.write_text("{}", encoding="utf-8")
            llm_func = AsyncMock(return_value=_response())

            with self.assertRaisesRegex(ValueError, "cache schema"):
                await extract(
                    [chunk],
                    llm_func,
                    con_num=1,
                    cache_directory=directory,
                    extraction_fingerprint="extraction-a",
                    cache_scope="build-a",
                )

            llm_func.assert_not_awaited()

    async def test_cache_record_must_satisfy_cross_record_contract(self):
        chunk = _chunk(
            "doc-1:chunk:0",
            "Alpha collaborates with Beta.",
        )
        response = _response(
            entities=[{"name": "Alpha"}, {"name": "Beta"}],
            relationships=[{"source": "Alpha", "target": "Beta"}],
        )

        with tempfile.TemporaryDirectory() as directory:
            await extract(
                [chunk],
                AsyncMock(return_value=response),
                con_num=1,
                cache_directory=directory,
                extraction_fingerprint="extraction-a",
                cache_scope="build-a",
            )
            record_path = next(
                Path(directory, "extraction-a", "records").glob("*.json")
            )
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            payload["relations"][0]["target"] = "Missing"
            record_path.write_text(json.dumps(payload), encoding="utf-8")
            llm_func = AsyncMock(return_value=_response())

            with self.assertRaisesRegex(ValueError, "endpoints must appear"):
                await extract(
                    [chunk],
                    llm_func,
                    con_num=1,
                    cache_directory=directory,
                    extraction_fingerprint="extraction-a",
                    cache_scope="build-a",
                )

            llm_func.assert_not_awaited()

    async def test_cache_arguments_must_be_provided_together(self):
        llm_func = AsyncMock(return_value=_response())
        cases = [
            {"cache_directory": "cache"},
            {"extraction_fingerprint": "extraction"},
            {"cache_scope": "build"},
            {
                "cache_directory": "cache",
                "extraction_fingerprint": "extraction",
            },
        ]

        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "provided together"):
                    await extract(
                        [],
                        llm_func,
                        con_num=1,
                        **arguments,
                    )

        llm_func.assert_not_awaited()
