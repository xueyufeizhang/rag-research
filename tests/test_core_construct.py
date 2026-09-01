import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rag_research.chunking import AGENTIC_CHUNKING_SYSTEM_PROMPT, ChunkConfig
from rag_research.core import LightRAG, LightRAGConfig
from rag_research.models import InputDocument
from rag_research.prompts import PROMPTS
from rag_research.storage import KVStore


def _config() -> LightRAGConfig:
    return LightRAGConfig(
        chunk_config=ChunkConfig(
            strategy="fixed",
            fixed_size=100,
            fixed_overlap=0,
        ),
        embedding_batch_size=2,
        embedding_concurrency=2,
        llm_backend="test",
        llm_model="test-llm",
        embedding_backend="test",
        embedding_model="test-embedding",
    )


def _documents() -> list[InputDocument]:
    return [
        InputDocument(
            document_id="doc-1",
            text="Alpha document.",
            metadata={"title": "Alpha"},
        ),
        InputDocument(
            document_id="doc-2",
            text="Beta document.",
            metadata={"title": "Beta"},
        ),
    ]


def _empty_extraction_response() -> str:
    return json.dumps({"entities": [], "relationships": []})


class ConstructManifestTests(unittest.IsolatedAsyncioTestCase):
    async def test_construct_audits_agentic_boundary_projection(self):
        text = " ".join(f"Sentence {index}." for index in range(1, 28))
        document = InputDocument(document_id="projected-doc", text=text)
        config = replace(
            _config(),
            chunk_config=ChunkConfig(
                strategy="agentic_ibm",
                agentic_batch_max_sentences=60,
                agentic_batch_max_chars=12000,
                agentic_min_sentences=10,
                agentic_max_sentences=24,
                agentic_concurrency=1,
                agentic_retries=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            agentic_calls = 0

            async def llm(*, system: str, prompt: str) -> str:
                nonlocal agentic_calls
                if system == AGENTIC_CHUNKING_SYSTEM_PROMPT:
                    agentic_calls += 1
                    return (
                        '{"chunks": ['
                        '{"start": 1, "end": 7}, '
                        '{"start": 8, "end": 27}'
                        ']}'
                    )
                return _empty_extraction_response()

            rag = LightRAG(
                working_dir=directory,
                llm_func=llm,
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=config,
            )

            result = await rag.construct((document,))

            self.assertEqual(agentic_calls, 1)
            self.assertEqual(result.chunk_count, 2)
            self.assertEqual(result.chunking_projection_count, 1)

            fingerprint = rag._make_build_fingerprint((document,))
            cache_path = Path(rag._chunk_cache_path(
                fingerprint,
                document.document_id,
            ))
            cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertEqual(
                cache_payload["document"]["boundary_projection_events"],
                [
                    {
                        "batch_index": 1,
                        "sentence_count": 27,
                        "original_boundaries": [[1, 7], [8, 27]],
                        "projected_boundaries": [[1, 10], [11, 27]],
                    }
                ],
            )

            manifest = json.loads(
                Path(directory, "build_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["build"]["chunking_projection_count"],
                1,
            )

    async def test_agentic_chunking_resumes_from_completed_document_cache(self):
        config = replace(
            _config(),
            chunk_config=ChunkConfig(
                strategy="agentic_ibm",
                agentic_batch_max_sentences=10,
                agentic_batch_max_chars=1000,
                agentic_min_sentences=1,
                agentic_max_sentences=5,
                agentic_concurrency=1,
                agentic_retries=0,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            first_agentic_prompts: list[str] = []

            async def first_llm(*, system: str, prompt: str) -> str:
                if system == AGENTIC_CHUNKING_SYSTEM_PROMPT:
                    first_agentic_prompts.append(prompt)
                    if "Alpha document." in prompt:
                        return '{"chunks": [{"start": 1, "end": 1}]}'
                    raise OSError("temporary chunking failure")
                return _empty_extraction_response()

            first_rag = LightRAG(
                working_dir=directory,
                llm_func=first_llm,
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=config,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "agentic chunking batch 1 failed",
            ):
                await first_rag.construct(_documents())

            self.assertEqual(len(first_agentic_prompts), 2)
            cached_documents = list(
                Path(directory, "chunking_cache").rglob("*.json")
            )
            self.assertEqual(len(cached_documents), 1)
            self.assertFalse(Path(directory, "build_manifest.json").exists())

            second_agentic_prompts: list[str] = []

            async def second_llm(*, system: str, prompt: str) -> str:
                if system == AGENTIC_CHUNKING_SYSTEM_PROMPT:
                    second_agentic_prompts.append(prompt)
                    return '{"chunks": [{"start": 1, "end": 1}]}'
                return _empty_extraction_response()

            second_rag = LightRAG(
                working_dir=directory,
                llm_func=second_llm,
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=config,
            )
            result = await second_rag.construct(_documents())

            self.assertEqual(result.failed_chunk_ids, [])
            self.assertEqual(len(second_agentic_prompts), 1)
            self.assertIn("Beta document.", second_agentic_prompts[0])
            self.assertNotIn("Alpha document.", second_agentic_prompts[0])
            self.assertEqual(
                len(list(Path(directory, "chunking_cache").rglob("*.json"))),
                2,
            )

    async def test_semantic_chunking_reuses_document_cache(self):
        document = InputDocument(
            document_id="semantic-doc",
            text="One. Two. Three. Four. Five.",
        )
        config = replace(
            _config(),
            chunk_config=ChunkConfig(
                strategy="semantic",
                semantic_breakpoint_percentile=90,
                semantic_min_sentences=1,
                semantic_max_sentences=2,
                semantic_buffer_size=0,
                semantic_embedding_batch_size=2,
                semantic_embedding_concurrency=1,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            embedding_calls = 0

            async def embed(value: str) -> list[float]:
                nonlocal embedding_calls
                embedding_calls += 1
                return [1.0, float(len(value))]

            rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=embed,
                config=config,
            )
            fingerprint = rag._make_build_fingerprint([document])

            first_chunks, first_cached, first_projections = await rag._chunk_document(
                document,
                fingerprint,
            )
            first_call_count = embedding_calls
            second_chunks, second_cached, second_projections = await rag._chunk_document(
                document,
                fingerprint,
            )

            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertGreater(first_call_count, 0)
            self.assertEqual(embedding_calls, first_call_count)
            self.assertEqual(second_chunks, first_chunks)
            self.assertEqual(first_projections, [])
            self.assertEqual(second_projections, [])

    async def test_corrupted_chunking_cache_is_rejected(self):
        document = _documents()[0]
        with tempfile.TemporaryDirectory() as directory:
            rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=_config(),
            )
            fingerprint = rag._make_build_fingerprint([document])
            await rag._chunk_document(document, fingerprint)

            cache_path = Path(rag._chunk_cache_path(
                fingerprint,
                document.document_id,
            ))
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            payload["document"]["spans"][0]["text"] = "corrupted"
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "text mismatch"):
                await rag._chunk_document(document, fingerprint)

    async def test_matching_build_is_reused_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            first_llm = AsyncMock(return_value=_empty_extraction_response())
            first_embed = AsyncMock(return_value=[1.0, 0.0])
            first_rag = LightRAG(
                working_dir=directory,
                llm_func=first_llm,
                con_num=2,
                embed_func=first_embed,
                config=_config(),
            )

            first_result = await first_rag.construct(tuple(_documents()))

            self.assertEqual(first_result.document_count, 2)
            self.assertEqual(first_result.chunk_count, 2)
            self.assertEqual(len(first_rag.chunk_kv.all()), 2)
            self.assertTrue(Path(directory, "build_manifest.json").exists())
            self.assertEqual(
                first_result.build_provenance["embedding"]["model"],
                "test-embedding",
            )

            stored_chunks = list(first_rag.chunk_kv.all().values())
            self.assertEqual(
                {chunk["document_id"] for chunk in stored_chunks},
                {"doc-1", "doc-2"},
            )
            self.assertTrue(all(chunk["chunk_id"] for chunk in stored_chunks))
            self.assertTrue(all("Title:" in chunk["model_text"] for chunk in stored_chunks))

            cached_llm = AsyncMock(return_value=_empty_extraction_response())
            cached_embed = AsyncMock(return_value=[1.0, 0.0])
            cached_rag = LightRAG(
                working_dir=directory,
                llm_func=cached_llm,
                con_num=2,
                embed_func=cached_embed,
                config=_config(),
            )

            cached_result = await cached_rag.construct(list(reversed(_documents())))

            self.assertEqual(cached_result, first_result)
            cached_llm.assert_not_awaited()
            cached_embed.assert_not_awaited()

    async def test_different_fingerprint_is_rejected_before_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=_config(),
            )
            await rag.construct(_documents())

            llm = AsyncMock(return_value=_empty_extraction_response())
            embed = AsyncMock(return_value=[1.0, 0.0])
            reloaded = LightRAG(
                working_dir=directory,
                llm_func=llm,
                con_num=1,
                embed_func=embed,
                config=_config(),
            )
            changed_documents = _documents()
            changed_documents[0] = InputDocument(
                document_id="doc-1",
                text="Changed alpha document.",
                metadata={"title": "Alpha"},
            )

            with self.assertRaisesRegex(RuntimeError, "different documents"):
                await reloaded.construct(changed_documents)

            llm.assert_not_awaited()
            embed.assert_not_awaited()

    async def test_store_without_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy_chunks = KVStore(str(Path(directory, "chunks.json")))
            legacy_chunks.set("legacy-chunk", {"text": "legacy"})
            legacy_chunks.save()

            llm = AsyncMock(return_value=_empty_extraction_response())
            embed = AsyncMock(return_value=[1.0, 0.0])
            rag = LightRAG(
                working_dir=directory,
                llm_func=llm,
                con_num=1,
                embed_func=embed,
                config=_config(),
            )

            with self.assertRaisesRegex(RuntimeError, "no build manifest"):
                await rag.construct(_documents())

            llm.assert_not_awaited()
            embed.assert_not_awaited()

    async def test_manifest_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=_config(),
            )
            await rag.construct(_documents())

            manifest_path = Path(directory, "build_manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["build"]["chunk_count"] += 1
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            reloaded = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=_config(),
            )

            with self.assertRaisesRegex(RuntimeError, "counts do not match"):
                await reloaded.construct(_documents())

    async def test_construct_uses_batch_embedding_for_all_index_types(self):
        with tempfile.TemporaryDirectory() as directory:
            extraction_response = json.dumps({
                "entities": [
                    {
                        "name": "Alpha",
                        "type": "Concept",
                        "description": "Alpha description",
                    },
                    {
                        "name": "Beta",
                        "type": "Concept",
                        "description": "Beta description",
                    },
                ],
                "relationships": [
                    {
                        "source": "Alpha",
                        "target": "Beta",
                        "keywords": ["linked"],
                        "description": "Relation description",
                    }
                ],
            })
            single_embed = AsyncMock(return_value=[1.0, 1.0])
            batches: list[list[str]] = []

            async def embed_many(texts):
                batches.append(list(texts))
                return [[float(index + 1), 1.0] for index, _ in enumerate(texts)]

            rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=extraction_response),
                con_num=2,
                embed_func=single_embed,
                embed_many_func=embed_many,
                config=_config(),
            )

            result = await rag.construct(tuple(_documents()))

            self.assertEqual(result.entity_count, 2)
            self.assertEqual(result.relation_count, 1)
            self.assertEqual(result.chunk_count, 2)
            self.assertEqual(len(batches), 3)
            self.assertEqual(
                batches[0],
                [
                    "Alpha Alpha description | Alpha description",
                    "Beta Beta description | Beta description",
                ],
            )
            self.assertEqual(
                batches[1],
                ["linked Relation description | Relation description"],
            )
            self.assertEqual(len(batches[2]), 2)
            self.assertTrue(all("Content:" in text for text in batches[2]))
            single_embed.assert_not_awaited()
            self.assertEqual(len(rag.entity_vidx._ids), 2)
            self.assertEqual(len(rag.relation_vidx._ids), 1)
            self.assertEqual(len(rag.chunk_vidx._ids), 2)

    async def test_embedding_model_change_invalidates_cached_build(self):
        with tempfile.TemporaryDirectory() as directory:
            first_rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 1.0]),
                config=_config(),
            )
            await first_rag.construct(_documents())

            changed_config = replace(
                _config(),
                embedding_model="different-embedding-model",
            )
            llm = AsyncMock(return_value=_empty_extraction_response())
            embed = AsyncMock(return_value=[1.0, 1.0])
            reloaded = LightRAG(
                working_dir=directory,
                llm_func=llm,
                con_num=1,
                embed_func=embed,
                config=changed_config,
            )

            with self.assertRaisesRegex(RuntimeError, "different documents, models"):
                await reloaded.construct(_documents())

            llm.assert_not_awaited()
            embed.assert_not_awaited()

    async def test_string_is_not_accepted_as_document_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 1.0]),
                config=_config(),
            )

            with self.assertRaisesRegex(TypeError, "sequence of InputDocument"):
                await rag.construct("not documents")

    def test_extraction_prompt_change_changes_build_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 1.0]),
                config=_config(),
            )
            original_fingerprint = rag._make_build_fingerprint(_documents())
            changed_prompt = (
                PROMPTS["entity_extraction_user_prompt"]
                + "\nAdditional extraction instruction."
            )

            with patch.dict(
                PROMPTS,
                {"entity_extraction_user_prompt": changed_prompt},
            ):
                changed_fingerprint = rag._make_build_fingerprint(_documents())

            self.assertNotEqual(original_fingerprint, changed_fingerprint)

    async def test_incomplete_cached_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first_rag = LightRAG(
                working_dir=directory,
                llm_func=AsyncMock(return_value=_empty_extraction_response()),
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 1.0]),
                config=_config(),
            )
            await first_rag.construct(_documents())

            manifest_path = Path(directory, "build_manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["build"]["failed_chunk_ids"] = ["failed-chunk"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            llm = AsyncMock(return_value=_empty_extraction_response())
            embed = AsyncMock(return_value=[1.0, 1.0])
            reloaded = LightRAG(
                working_dir=directory,
                llm_func=llm,
                con_num=1,
                embed_func=embed,
                config=_config(),
            )

            with self.assertRaisesRegex(RuntimeError, "cached build is incomplete"):
                await reloaded.construct(_documents())

            llm.assert_not_awaited()
            embed.assert_not_awaited()

    async def test_failed_extraction_is_not_indexed_and_resumes_from_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            async def first_llm(*, system: str, prompt: str) -> str:
                if "Alpha document." in prompt:
                    return _empty_extraction_response()
                raise OSError("temporary failure")

            first_rag = LightRAG(
                working_dir=directory,
                llm_func=first_llm,
                con_num=2,
                embed_func=AsyncMock(return_value=[1.0, 1.0]),
                config=_config(),
            )

            with patch(
                "rag_research.extraction.asyncio.sleep",
                new=AsyncMock(),
            ):
                with self.assertRaisesRegex(RuntimeError, "construction aborted"):
                    await first_rag.construct(_documents())

            self.assertFalse(Path(directory, "build_manifest.json").exists())
            self.assertFalse(Path(directory, "chunks.json").exists())
            self.assertEqual(first_rag.chunk_vidx._ids, [])
            self.assertTrue(Path(directory, "extraction_cache").is_dir())

            second_prompts: list[str] = []

            async def second_llm(*, system: str, prompt: str) -> str:
                second_prompts.append(prompt)
                return _empty_extraction_response()

            second_rag = LightRAG(
                working_dir=directory,
                llm_func=second_llm,
                con_num=2,
                embed_func=AsyncMock(return_value=[1.0, 1.0]),
                config=_config(),
            )
            result = await second_rag.construct(_documents())

            self.assertEqual(result.failed_chunk_ids, [])
            self.assertEqual(len(second_prompts), 1)
            self.assertIn("Beta document.", second_prompts[0])
            self.assertNotIn("Alpha document.", second_prompts[0])
            self.assertTrue(Path(directory, "build_manifest.json").exists())
