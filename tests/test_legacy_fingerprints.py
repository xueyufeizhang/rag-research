"""Freeze pre-Jev configuration, provenance, and stage fingerprints.

The checked-in fixture is the oracle. This module has no fixture update mode.
"""

import copy
import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_research.chunking import ChunkConfig
from rag_research.core import LightRAG, LightRAGConfig
from rag_research.models import InputDocument


FIXTURE_PATH = Path(__file__).parent / "fixtures/legacy_behavior/fingerprints.json"
FIXTURE_SHA256 = "01c6fa06120bec6d1925e8075ccd8aee7ec1a2818fbe22db2361efbeedcfe986"
LEGACY_STRATEGIES = ("fixed", "semantic", "agentic")

# Keep the complete historical field sets, including fields unused by a strategy.
# New strategy-specific fields may be added without entering these old payloads.
LEGACY_CHUNK_FIELDS = (
    "strategy", "fixed_size", "overlap_size",
    "semantic_breakpoint_percentile", "semantic_min_sentences",
    "semantic_max_sentences", "semantic_buffer_size",
    "semantic_embedding_batch_size", "semantic_embedding_concurrency",
    "agentic_batch_max_sentences", "agentic_batch_max_chars",
    "agentic_min_sentences", "agentic_max_sentences", "agentic_concurrency",
    "agentic_retries",
)
LEGACY_RAG_FIELDS = (
    "chunk_config", "chunk_top_k", "chunk_candidate_top_k", "entity_top_k",
    "relation_top_k", "relation_candidate_top_k", "embedding_batch_size",
    "embedding_concurrency", "llm_backend", "llm_model", "embedding_backend",
    "embedding_model",
)


def canonical_json(value):
    """Compare serialized types as well as values (e.g. 3 differs from 3.0)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class LegacyFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.fixture_bytes = FIXTURE_PATH.read_bytes()
        self.fixture = json.loads(self.fixture_bytes)

    def tearDown(self):
        # Read after the test as well; the fixture is never written or regenerated.
        self.assertEqual(FIXTURE_PATH.read_bytes(), self.fixture_bytes)

    def make_rag(self, case):
        values = copy.deepcopy(case["light_rag_config"])
        chunk_values = values.pop("chunk_config")
        config = LightRAGConfig(chunk_config=ChunkConfig(**chunk_values), **values)
        # These pure fingerprint methods use only config. Avoid store/model setup.
        rag = object.__new__(LightRAG)
        rag.config = config
        return rag

    def assert_case_matches_golden(self, strategy):
        original_fixture = canonical_json(self.fixture)
        case = self.fixture["cases"][strategy]
        rag = self.make_rag(case)
        documents = tuple(
            InputDocument(**copy.deepcopy(value))
            for value in self.fixture["documents"]
        )
        actual = {
            "chunk_config_fingerprint_dict": rag.config.chunk_config.fingerprint_dict(),
            "build_provenance": rag._make_build_provenance(),
            "chunking_fingerprint": rag._make_chunking_fingerprint(),
            "extraction_fingerprint": rag._make_extraction_fingerprint(),
            "build_fingerprint": rag._make_build_fingerprint(documents),
        }
        self.assertEqual(set(actual), set(case["expected"]))
        for name, expected in case["expected"].items():
            with self.subTest(strategy=strategy, output=name):
                self.assertEqual(canonical_json(actual[name]), canonical_json(expected))
        self.assertEqual(canonical_json(self.fixture), original_fixture)

    def test_fixed_matches_static_legacy_fingerprints(self):
        self.assert_case_matches_golden("fixed")

    def test_semantic_matches_static_legacy_fingerprints(self):
        self.assert_case_matches_golden("semantic")

    def test_agentic_matches_static_legacy_fingerprints(self):
        self.assert_case_matches_golden("agentic")

    def test_fixture_contains_complete_explicit_legacy_inputs(self):
        self.assertEqual(self.fixture["schema_version"], 1)
        self.assertEqual(tuple(self.fixture["cases"]), LEGACY_STRATEGIES)
        self.assertTrue(self.fixture["documents"])
        for strategy, case in self.fixture["cases"].items():
            with self.subTest(strategy=strategy):
                values = case["light_rag_config"]
                self.assertEqual(set(values), set(LEGACY_RAG_FIELDS))
                self.assertEqual(set(values["chunk_config"]), set(LEGACY_CHUNK_FIELDS))
                self.assertEqual(values["chunk_config"]["strategy"], strategy)
                self.assertEqual(
                    canonical_json(case["expected"]["chunk_config_fingerprint_dict"]),
                    canonical_json(values["chunk_config"]),
                )
                config = self.make_rag(case).config
                for name in LEGACY_RAG_FIELDS:
                    actual = (
                        config.chunk_config.fingerprint_dict()
                        if name == "chunk_config" else getattr(config, name)
                    )
                    self.assertEqual(canonical_json(actual), canonical_json(values[name]))

    def test_explicit_inputs_ignore_environment_overrides(self):
        conflicting_environment = {
            "CHUNKING_STRATEGY": "environment-only-strategy",
            "FIXED_WINDOW_SIZE": "8000", "CHUNK_OVERLAP": "300",
            "SEMANTIC_BREAKPOINT_PERCENTILE": "97.5",
            "SEMANTIC_MIN_SENTENCES": "8", "SEMANTIC_MAX_SENTENCES": "24",
            "SEMANTIC_BUFFER_SIZE": "4", "SEMANTIC_EMBEDDING_BATCH_SIZE": "31",
            "SEMANTIC_EMBEDDING_CONCURRENCY": "6",
            "AGENTIC_BATCH_MAX_SENTENCES": "40", "AGENTIC_BATCH_MAX_CHARS": "9000",
            "AGENTIC_MIN_SENTENCES": "5", "AGENTIC_MAX_SENTENCES": "18",
            "AGENTIC_CONCURRENCY": "8", "AGENTIC_RETRIES": "4",
            "CHUNK_TOP_K": "11", "CHUNK_CANDIDATE_TOP_K": "41",
            "ENTITY_TOP_K": "12", "RELATION_TOP_K": "13",
            "RELATION_CANDIDATE_TOP_K": "43",
            "EMBEDDING_BATCH_SIZE": "37", "EMBEDDING_CONCURRENCY": "9",
            "LLM_BACKEND": "ollama", "LLM_MODEL": "environment-llm",
            "API_MODEL": "environment-api", "EMBED_MODEL": "environment-embedding",
        }
        for label, environment in (("empty", {}), ("conflicting", conflicting_environment)):
            with patch.dict(os.environ, environment, clear=True):
                for strategy in LEGACY_STRATEGIES:
                    with self.subTest(environment=label, strategy=strategy):
                        self.assert_case_matches_golden(strategy)

    def test_static_fixture_bytes_match_captured_digest(self):
        self.assertEqual(hashlib.sha256(self.fixture_bytes).hexdigest(), FIXTURE_SHA256)


if __name__ == "__main__":
    unittest.main()
