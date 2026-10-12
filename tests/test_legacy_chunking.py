"""Replay static pre-Jev behavior and cache artifacts without model services.

Expected values live in fixtures/legacy_behavior; there is no update mode.
"""

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import AsyncMock

from rag_research.chunking import ChunkConfig
from rag_research.chunking.text_spans import split_sentences
from rag_research.core import LightRAG, LightRAGConfig
from rag_research.models import InputDocument
from rag_research.prompts import (
    AGENTIC_METADATA_SYSTEM_PROMPT,
    AGENTIC_PROPOSITION_SYSTEM_PROMPT,
    AGENTIC_STATE_SYSTEM_PROMPT,
)


FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures/legacy_behavior"
STAGES = {
    AGENTIC_PROPOSITION_SYSTEM_PROMPT: "proposition",
    AGENTIC_STATE_SYSTEM_PROMPT: "state",
    AGENTIC_METADATA_SYSTEM_PROMPT: "metadata",
}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def make_config(case):
    values = copy.deepcopy(case["light_rag_config"])
    return LightRAGConfig(
        chunk_config=ChunkConfig(**values.pop("chunk_config")), **values,
    )


class ScriptedModels:
    """Only return checked-in responses; also detect swallowed provider errors."""

    def __init__(self, case):
        self.script = copy.deepcopy(case["llm_responses"])
        self.vectors = case["embedding_vectors"]
        self.model_calls = []
        self.embedding_batches = []
        self.errors = []

    def fail(self, message):
        self.errors.append(message)
        raise AssertionError(message)

    async def llm(self, *, system, prompt):
        stage = STAGES.get(system)
        self.model_calls.append({
            "stage": stage,
            "system_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        })
        if not self.script:
            self.fail(f"Unexpected model call: {stage}")
        response = self.script.pop(0)
        if response["stage"] != stage:
            self.fail(f"Expected {response['stage']}, got {stage}")
        return response["text"]

    async def embed(self, text):
        self.fail("Unexpected single embedding call; fixture uses fixed batches")

    async def embed_many(self, texts):
        self.embedding_batches.append(list(texts))
        if any(text not in self.vectors for text in texts):
            self.fail("Embedding input differs from the historical source context")
        return [list(self.vectors[text]) for text in texts]


def make_rag(case, directory, *, llm, embed, embed_many):
    return LightRAG(
        working_dir=str(Path(directory) / "working"),
        cache_directory=str(Path(directory) / "cache"),
        config=make_config(case),
        con_num=1,
        llm_func=llm,
        embed_func=embed,
        embed_many_func=embed_many,
    )


async def observe_case(case, directory):
    """Collect actual outputs only. The caller compares them to static goldens."""
    models = ScriptedModels(case)
    rag = make_rag(
        case, directory,
        llm=models.llm, embed=models.embed, embed_many=models.embed_many,
    )
    document = InputDocument(**copy.deepcopy(case["document"]))
    fingerprint = rag._make_chunking_fingerprint()
    chunks, loaded = await rag._chunk_document(document, fingerprint)
    if loaded or models.errors or models.script:
        raise AssertionError({
            "unexpected_cache_hit": loaded,
            "provider_errors": models.errors,
            "unused_responses": models.script,
        })
    cache_path = Path(rag._chunk_cache_path(fingerprint, document))
    observed = {
        "sentences": [asdict(span) for span in split_sentences(document.text)],
        "chunks": [asdict(chunk) for chunk in chunks],
        "chunk_embedding_texts": [rag._make_chunk_embedding_text(chunk) for chunk in chunks],
        "model_calls": models.model_calls,
        "embedding_batches": models.embedding_batches,
        "cache_relative_path": str(cache_path.relative_to(rag.cache_directory)),
    }
    return observed, json.loads(cache_path.read_text(encoding="utf-8"))


class LegacyChunkingTests(unittest.IsolatedAsyncioTestCase):
    maxDiff = None

    def setUp(self):
        fixture_bytes = {
            path: path.read_bytes() for path in FIXTURE_DIRECTORY.rglob("*.json")
        }
        self.cases = json.loads(
            fixture_bytes[FIXTURE_DIRECTORY / "behavior.json"]
        )["cases"]
        self.addCleanup(self.assert_fixtures_unchanged, fixture_bytes)

    def assert_fixtures_unchanged(self, fixture_bytes):
        for path, original in fixture_bytes.items():
            self.assertEqual(path.read_bytes(), original, str(path))

    def cache_bytes(self, name):
        return (FIXTURE_DIRECTORY / "caches" / f"{name}.json").read_bytes()

    def assert_json_equal(self, actual, expected):
        # Preserve serialized type distinctions such as 1 versus True.
        self.assertEqual(canonical_json(actual), canonical_json(expected))

    def test_fixture_files_match_capture_manifest(self):
        manifest = json.loads(
            (FIXTURE_DIRECTORY / "capture_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            set(manifest["fixture_sha256"]),
            {"fingerprints.json", "behavior.json"}
            | {f"caches/{name}.json" for name in self.cases},
        )
        for relative_path, expected in manifest["fixture_sha256"].items():
            with self.subTest(fixture=relative_path):
                actual = hashlib.sha256(
                    (FIXTURE_DIRECTORY / relative_path).read_bytes()
                ).hexdigest()
                self.assertEqual(actual, expected)

    async def assert_case_matches_golden(self, name):
        case = self.cases[name]
        with tempfile.TemporaryDirectory() as directory:
            actual, cache_payload = await observe_case(case, directory)
        self.assertEqual(set(actual), set(case["expected"]))
        for field, expected in case["expected"].items():
            with self.subTest(case=name, output=field):
                self.assert_json_equal(actual[field], expected)
        with self.subTest(case=name, output="complete_cache_payload"):
            self.assert_json_equal(cache_payload, json.loads(self.cache_bytes(name)))

    async def test_fixed_overlap_matches_legacy_output(self):
        await self.assert_case_matches_golden("fixed_overlap")

    async def test_semantic_topic_shift_matches_legacy_output(self):
        await self.assert_case_matches_golden("semantic_topic_shift")

    async def test_semantic_uniform_distances_match_legacy_output(self):
        await self.assert_case_matches_golden("semantic_uniform")

    async def test_agentic_routing_and_metadata_repair_match_legacy_output(self):
        await self.assert_case_matches_golden("agentic_routing_repair")

    async def test_agentic_projection_and_fallback_match_legacy_output(self):
        await self.assert_case_matches_golden("agentic_projection_fallback")

    def cache_only_rag(self, case, directory):
        callbacks = [
            AsyncMock(side_effect=AssertionError("Cache replay must not call a model"))
            for _ in range(3)
        ]
        # A failed provider call can be caught by agentic fallback; check awaits too.
        for callback in callbacks:
            self.addCleanup(callback.assert_not_awaited)
        return make_rag(
            case, directory,
            llm=callbacks[0], embed=callbacks[1], embed_many=callbacks[2],
        )

    def install_cache(self, rag, case, payload_bytes):
        # Use the captured path, so a future layout change cannot hide a miss.
        path = Path(rag.cache_directory) / case["expected"]["cache_relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload_bytes)
        return path

    async def test_historical_cache_files_are_reused_without_model_calls(self):
        for name, case in self.cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                rag = self.cache_only_rag(case, directory)
                original = self.cache_bytes(name)
                cache_path = self.install_cache(rag, case, original)
                chunks, loaded = await rag._chunk_document(
                    InputDocument(**copy.deepcopy(case["document"])),
                    rag._make_chunking_fingerprint(),
                )
                self.assertTrue(loaded)
                self.assert_json_equal([asdict(chunk) for chunk in chunks], case["expected"]["chunks"])
                self.assertEqual(cache_path.read_bytes(), original)

    async def assert_corrupt_cache_rejected(self, name, mutate, message):
        case = self.cases[name]
        payload = json.loads(self.cache_bytes(name))
        mutate(payload["document"])
        with tempfile.TemporaryDirectory() as directory:
            rag = self.cache_only_rag(case, directory)
            original = canonical_json(payload).encode("utf-8")
            path = self.install_cache(rag, case, original)
            with self.assertRaisesRegex(ValueError, message):
                await rag._chunk_document(
                    InputDocument(**copy.deepcopy(case["document"])),
                    rag._make_chunking_fingerprint(),
                )
            self.assertEqual(path.read_bytes(), original)

    async def test_cache_identity_mismatches_are_rejected(self):
        changes = (
            ("schema_version", -1, "unsupported chunking cache schema"),
            ("chunking_fingerprint", "wrong", "fingerprint mismatch"),
            ("document_id", "wrong", "document ID mismatch"),
            ("text_sha256", "wrong", "document text mismatch"),
            ("strategy", "unknown", "strategy mismatch"),
        )
        for name in self.cases:
            for field, value, message in changes:
                with self.subTest(case=name, field=field):
                    await self.assert_corrupt_cache_rejected(
                        name, lambda payload: payload.update({field: value}), message,
                    )

    async def test_cache_source_text_and_offsets_are_validated(self):
        changes = (
            (lambda p: p["spans"][0].update(text="changed source"), "text mismatch"),
            (lambda p: p["spans"][0].update(char_start=False), "invalid chunking cache offsets"),
            (lambda p: p["spans"][0].update(char_start=1), "does not start at zero"),
            (lambda p: p["spans"].pop(), "does not cover the document"),
            (lambda p: p["spans"][1].update(char_start=p["spans"][0]["char_end"] + 1), "coverage gap"),
        )
        for name in self.cases:
            for mutate, message in changes:
                with self.subTest(case=name, rejection=message):
                    await self.assert_corrupt_cache_rejected(name, mutate, message)

    async def test_cache_metadata_rules_are_preserved(self):
        for name, case in self.cases.items():
            agentic = case["light_rag_config"]["chunk_config"]["strategy"] == "agentic"
            with self.subTest(case=name):
                await self.assert_corrupt_cache_rejected(
                    name,
                    lambda p: p["spans"][0].update(title="partial", summary=None),
                    "incomplete chunking metadata",
                )
                await self.assert_corrupt_cache_rejected(
                    name,
                    lambda p: p["spans"][0].update(
                        title=None if agentic else "unexpected title",
                        summary=None if agentic else "unexpected summary",
                    ),
                    "lacks metadata" if agentic else "contains agentic metadata",
                )

    async def test_cache_trace_rules_are_preserved(self):
        for name, case in self.cases.items():
            agentic = case["light_rag_config"]["chunk_config"]["strategy"] == "agentic"
            with self.subTest(case=name):
                await self.assert_corrupt_cache_rejected(
                    name, lambda p: p.update(agentic_state_events=["malformed"]),
                    "invalid agentic state events",
                )
                await self.assert_corrupt_cache_rejected(
                    name,
                    lambda p: p.update(agentic_state_events=[] if agentic else [{"event": "state_transition"}]),
                    "has no trace" if agentic else "contains non-overlap events",
                )


if __name__ == "__main__":
    unittest.main()
