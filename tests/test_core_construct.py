import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rag_research.chunking import ChunkConfig
from rag_research.prompts import (
    AGENTIC_METADATA_SYSTEM_PROMPT,
    AGENTIC_PROPOSITION_SYSTEM_PROMPT,
    AGENTIC_STATE_SYSTEM_PROMPT,
)
from rag_research.core import (
    ENTITY_DESCRIPTION_MAX_CHARS,
    ENTITY_DESCRIPTION_MAX_VARIANTS,
    LightRAG,
    LightRAGConfig,
    _aggregate_descriptions,
    _merge_extraction_records,
)
from rag_research.extraction import Entity, Relation
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
    async def test_stateful_agentic_metadata_and_trace_are_persisted(self):
        text = "One is here. Two is here. Three is here. Four is here."
        document = InputDocument(document_id="stateful-doc", text=text)
        config = replace(
            _config(),
            chunk_config=ChunkConfig(
                strategy="agentic",
                agentic_batch_max_sentences=10,
                agentic_batch_max_chars=1000,
                agentic_min_sentences=1,
                agentic_max_sentences=4,
                agentic_concurrency=1,
                agentic_retries=0,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            state_call = 0

            async def llm(*, system: str, prompt: str) -> str:
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
                    decisions = [
                        ("new_chunk", "First", "First topic starts."),
                        ("append", "First", "First topic continues."),
                        ("new_chunk", "Second", "Second topic starts."),
                        ("append", "Second", "Second topic continues."),
                    ]
                    action, title, summary = decisions[state_call]
                    state_call += 1
                    return json.dumps({
                        "action": action,
                        "title": title,
                        "summary": summary,
                    })
                return _empty_extraction_response()

            embed_func = AsyncMock(return_value=[1.0, 0.0])
            rag = LightRAG(
                working_dir=directory,
                llm_func=llm,
                con_num=1,
                embed_func=embed_func,
                config=config,
            )

            result = await rag.construct((document,))

            self.assertEqual(result.chunk_count, 2)
            stored_chunks = list(rag.chunk_kv.all().values())
            self.assertEqual(
                [chunk["metadata"]["chunk_title"] for chunk in stored_chunks],
                ["First", "Second"],
            )
            self.assertTrue(all(
                "Semantic chunk state:" not in chunk["model_text"]
                for chunk in stored_chunks
            ))
            embedded_texts = [
                call.args[0]
                for call in embed_func.await_args_list
            ]
            self.assertTrue(any(
                "Semantic chunk state:" in embedded_text
                for embedded_text in embedded_texts
            ))

            cache_path = Path(rag._chunk_cache_path(
                rag._make_chunking_fingerprint(),
                document,
            ))
            payload = json.loads(cache_path.read_text(encoding="utf-8"))["document"]
            self.assertEqual(len(payload["agentic_state_events"]), 4)
            self.assertEqual(
                [span["title"] for span in payload["spans"]],
                ["First", "Second"],
            )

            async def fail_if_called(*, system: str, prompt: str) -> str:
                raise AssertionError("completed agentic cache should be reused")

            resumed_rag = LightRAG(
                working_dir=str(Path(directory, "resumed")),
                cache_directory=rag.cache_directory,
                llm_func=fail_if_called,
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=config,
            )
            resumed_chunks, loaded = await resumed_rag._chunk_document(
                document,
                resumed_rag._make_chunking_fingerprint(),
            )
            self.assertTrue(loaded)
            self.assertEqual(
                [chunk.metadata["chunk_title"] for chunk in resumed_chunks],
                ["First", "Second"],
            )

    async def test_construct_audits_agentic_boundary_projection(self):
        text = " ".join(f"Sentence {index}." for index in range(1, 28))
        document = InputDocument(document_id="projected-doc", text=text)
        config = replace(
            _config(),
            chunk_config=ChunkConfig(
                strategy="agentic",
                agentic_batch_max_sentences=60,
                agentic_batch_max_chars=12000,
                agentic_min_sentences=10,
                agentic_max_sentences=24,
                agentic_concurrency=1,
                agentic_retries=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            proposition_calls = 0
            state_calls = 0

            async def llm(*, system: str, prompt: str) -> str:
                nonlocal proposition_calls, state_calls
                if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                    proposition_calls += 1
                    return '{"propositions": [{"start": 1, "end": 27}]}'
                if system == AGENTIC_STATE_SYSTEM_PROMPT:
                    state_calls += 1
                    return json.dumps({
                        "action": "new_chunk",
                        "title": f"Projected {state_calls}",
                        "summary": "A size-constrained proposition group.",
                    })
                return _empty_extraction_response()

            rag = LightRAG(
                working_dir=directory,
                llm_func=llm,
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=config,
            )

            result = await rag.construct((document,))

            self.assertEqual(proposition_calls, 1)
            self.assertEqual(state_calls, 2)
            self.assertEqual(result.chunk_count, 2)

            fingerprint = rag._make_chunking_fingerprint()
            cache_path = Path(rag._chunk_cache_path(
                fingerprint,
                document,
            ))
            cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))[
                "document"
            ]
            projection = cache_payload["agentic_state_events"][0]
            self.assertEqual(projection["event"], "proposition_projection")
            self.assertEqual(projection["original_boundaries"], [[1, 27]])
            self.assertEqual(projection["final_boundaries"][-1][-1], 27)

            manifest = json.loads(
                Path(directory, "build_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["build"]["build_provenance"]["chunking"]["strategy"],
                "agentic",
            )

    async def test_construct_audits_document_tail_rebalancing(self):
        text = " ".join(
            f"Sentence {index}."
            for index in range(1, 31)
        )
        document = InputDocument(document_id="rebalanced-doc", text=text)
        config = replace(
            _config(),
            chunk_config=ChunkConfig(
                strategy="agentic",
                agentic_batch_max_sentences=15,
                agentic_batch_max_chars=12000,
                agentic_min_sentences=10,
                agentic_max_sentences=24,
                agentic_concurrency=2,
                agentic_retries=0,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            state_call = 0

            async def llm(*, system: str, prompt: str) -> str:
                nonlocal state_call
                if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                    return (
                        '{"propositions": ['
                        '{"start": 1, "end": 10}, '
                        '{"start": 11, "end": 15}'
                        ']}'
                    )
                if system == AGENTIC_STATE_SYSTEM_PROMPT:
                    actions = ["new_chunk", "new_chunk", "append", "new_chunk"]
                    action = actions[state_call]
                    state_call += 1
                    return json.dumps({
                        "action": action,
                        "title": f"State {state_call}",
                        "summary": "A managed document segment.",
                    })
                if system == AGENTIC_METADATA_SYSTEM_PROMPT:
                    return json.dumps({
                        "title": "Rebalanced state",
                        "summary": "A finalized rebalanced document segment.",
                    })
                return _empty_extraction_response()

            rag = LightRAG(
                working_dir=directory,
                llm_func=llm,
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=config,
            )
            result = await rag.construct((document,))

            self.assertEqual(result.chunk_count, 2)
            cache_path = Path(rag._chunk_cache_path(
                rag._make_chunking_fingerprint(),
                document,
            ))
            events = json.loads(
                cache_path.read_text(encoding="utf-8")
            )["document"]["agentic_state_events"]
            rebalance = events[-1]
            self.assertEqual(rebalance["event"], "document_rebalance")
            self.assertEqual(
                rebalance["final_boundaries"],
                [[1, 10], [11, 30]],
            )

    async def test_agentic_chunking_resumes_from_completed_document_cache(self):
        config = replace(
            _config(),
            chunk_config=ChunkConfig(
                strategy="agentic",
                agentic_batch_max_sentences=10,
                agentic_batch_max_chars=1000,
                agentic_min_sentences=1,
                agentic_max_sentences=5,
                agentic_concurrency=1,
                agentic_retries=0,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            first_proposition_prompts: list[str] = []
            first_state_prompts: list[str] = []

            async def first_llm(*, system: str, prompt: str) -> str:
                if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                    first_proposition_prompts.append(prompt)
                    if "Alpha document." in prompt:
                        return '{"propositions": [{"start": 1, "end": 1}]}'
                    raise OSError("temporary chunking failure")
                if system == AGENTIC_STATE_SYSTEM_PROMPT:
                    first_state_prompts.append(prompt)
                    return json.dumps({
                        "action": "new_chunk",
                        "title": "Document",
                        "summary": "A short source document.",
                    })
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
                "agentic proposition batch 1 failed",
            ):
                await first_rag.construct(_documents())

            self.assertEqual(len(first_proposition_prompts), 2)
            self.assertEqual(len(first_state_prompts), 1)
            cached_documents = list(
                Path(directory, "pipeline_cache", "chunking").rglob("*.json")
            )
            self.assertEqual(len(cached_documents), 1)
            self.assertFalse(Path(directory, "build_manifest.json").exists())

            second_proposition_prompts: list[str] = []
            second_state_prompts: list[str] = []

            async def second_llm(*, system: str, prompt: str) -> str:
                if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                    second_proposition_prompts.append(prompt)
                    return '{"propositions": [{"start": 1, "end": 1}]}'
                if system == AGENTIC_STATE_SYSTEM_PROMPT:
                    second_state_prompts.append(prompt)
                    return json.dumps({
                        "action": "new_chunk",
                        "title": "Document",
                        "summary": "A short source document.",
                    })
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
            self.assertEqual(len(second_proposition_prompts), 1)
            self.assertEqual(len(second_state_prompts), 1)
            self.assertIn("Beta document.", second_proposition_prompts[0])
            self.assertNotIn("Alpha document.", second_proposition_prompts[0])
            self.assertEqual(
                len(list(
                    Path(directory, "pipeline_cache", "chunking").rglob("*.json")
                )),
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
            fingerprint = rag._make_chunking_fingerprint()

            first_chunks, first_cached = await rag._chunk_document(
                document,
                fingerprint,
            )
            first_call_count = embedding_calls
            second_chunks, second_cached = await rag._chunk_document(
                document,
                fingerprint,
            )

            self.assertFalse(first_cached)
            self.assertTrue(second_cached)
            self.assertGreater(first_call_count, 0)
            self.assertEqual(embedding_calls, first_call_count)
            self.assertEqual(second_chunks, first_chunks)

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
            fingerprint = rag._make_chunking_fingerprint()
            await rag._chunk_document(document, fingerprint)

            cache_path = Path(rag._chunk_cache_path(
                fingerprint,
                document,
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
                    "Alpha Alpha description",
                    "Beta Beta description",
                ],
            )
            self.assertEqual(
                batches[1],
                ["linked Relation description"],
            )
            self.assertEqual(len(batches[2]), 2)
            self.assertTrue(all("Content:" in text for text in batches[2]))
            single_embed.assert_not_awaited()
            self.assertEqual(len(rag.entity_vidx._ids), 2)
            self.assertEqual(len(rag.relation_vidx._ids), 1)
            self.assertEqual(len(rag.chunk_vidx._ids), 2)

    def test_global_merge_normalizes_names_and_bounds_descriptions(self):
        descriptions = [
            f"Description {index} " + ("x" * 500)
            for index in range(30)
        ]
        entities = [
            Entity(
                "TalkSport" if index % 2 == 0 else "talkSPORT",
                "Organization",
                description,
                [f"chunk-{index}"],
            )
            for index, description in enumerate(descriptions)
        ] + [
            Entity(
                "Club",
                "Organization",
                "Club description",
                ["chunk-club"],
            )
        ]
        relations = [
            Relation(
                "TalkSport",
                "Club",
                ["Coverage"],
                "TalkSport covers the club.",
                ["chunk-0"],
            ),
            Relation(
                "talkSPORT",
                "Club",
                ["coverage", "Reporting"],
                "talkSPORT reports on the club.",
                ["chunk-1"],
            ),
        ]

        clean_entities, clean_relations = _merge_extraction_records(
            entities,
            relations,
        )

        self.assertEqual(set(clean_entities), {"TalkSport", "Club"})
        merged_entity = clean_entities["TalkSport"]
        self.assertEqual(len(merged_entity.source_id), 30)
        self.assertLessEqual(
            len(merged_entity.description),
            ENTITY_DESCRIPTION_MAX_CHARS,
        )
        self.assertLessEqual(
            len(merged_entity.description.split(" | ")),
            ENTITY_DESCRIPTION_MAX_VARIANTS,
        )
        self.assertEqual(set(clean_relations), {"Club||TalkSport"})
        merged_relation = clean_relations["Club||TalkSport"]
        self.assertEqual(merged_relation.source, "Club")
        self.assertEqual(merged_relation.target, "TalkSport")
        self.assertEqual(merged_relation.keywords, ["Coverage", "Reporting"])
        self.assertEqual(
            merged_relation.source_id,
            ["chunk-0", "chunk-1"],
        )

    def test_description_aggregation_deduplicates_case_insensitively(self):
        self.assertEqual(
            _aggregate_descriptions(
                ["Alpha description", " alpha DESCRIPTION ", "Beta"],
                max_variants=12,
                max_chars=4000,
            ),
            "Alpha description | Beta",
        )

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
            original_build_fingerprint = rag._make_build_fingerprint(
                _documents()
            )
            original_chunking_fingerprint = (
                rag._make_chunking_fingerprint()
            )
            original_extraction_fingerprint = (
                rag._make_extraction_fingerprint()
            )
            changed_prompt = (
                PROMPTS["entity_extraction_user_prompt"]
                + "\nAdditional extraction instruction."
            )

            with patch.dict(
                PROMPTS,
                {"entity_extraction_user_prompt": changed_prompt},
            ):
                changed_build_fingerprint = rag._make_build_fingerprint(
                    _documents()
                )
                changed_chunking_fingerprint = (
                    rag._make_chunking_fingerprint()
                )
                changed_extraction_fingerprint = (
                    rag._make_extraction_fingerprint()
                )

            self.assertNotEqual(
                original_build_fingerprint,
                changed_build_fingerprint,
            )
            self.assertEqual(
                original_chunking_fingerprint,
                changed_chunking_fingerprint,
            )
            self.assertNotEqual(
                original_extraction_fingerprint,
                changed_extraction_fingerprint,
            )

    def test_stage_fingerprints_include_only_stage_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            base_config = _config()
            changed_llm_config = replace(
                base_config,
                llm_model="different-llm",
            )
            fixed_rag = LightRAG(
                working_dir=Path(directory, "fixed-a"),
                llm_func=AsyncMock(),
                con_num=1,
                embed_func=AsyncMock(),
                config=base_config,
            )
            changed_fixed_rag = LightRAG(
                working_dir=Path(directory, "fixed-b"),
                llm_func=AsyncMock(),
                con_num=1,
                embed_func=AsyncMock(),
                config=changed_llm_config,
            )

            self.assertEqual(
                fixed_rag._make_chunking_fingerprint(),
                changed_fixed_rag._make_chunking_fingerprint(),
            )
            self.assertNotEqual(
                fixed_rag._make_extraction_fingerprint(),
                changed_fixed_rag._make_extraction_fingerprint(),
            )

            agentic_config = replace(
                base_config,
                chunk_config=replace(
                    base_config.chunk_config,
                    strategy="agentic",
                ),
            )
            changed_agentic_config = replace(
                agentic_config,
                llm_model="different-llm",
            )
            agentic_rag = LightRAG(
                working_dir=Path(directory, "agentic-a"),
                llm_func=AsyncMock(),
                con_num=1,
                embed_func=AsyncMock(),
                config=agentic_config,
            )
            changed_agentic_rag = LightRAG(
                working_dir=Path(directory, "agentic-b"),
                llm_func=AsyncMock(),
                con_num=1,
                embed_func=AsyncMock(),
                config=changed_agentic_config,
            )

            self.assertNotEqual(
                agentic_rag._make_chunking_fingerprint(),
                changed_agentic_rag._make_chunking_fingerprint(),
            )

            semantic_config = replace(
                base_config,
                chunk_config=replace(
                    base_config.chunk_config,
                    strategy="semantic",
                ),
            )
            changed_semantic_config = replace(
                semantic_config,
                embedding_model="different-embedding",
            )
            semantic_rag = LightRAG(
                working_dir=Path(directory, "semantic-a"),
                llm_func=AsyncMock(),
                con_num=1,
                embed_func=AsyncMock(),
                config=semantic_config,
            )
            changed_semantic_rag = LightRAG(
                working_dir=Path(directory, "semantic-b"),
                llm_func=AsyncMock(),
                con_num=1,
                embed_func=AsyncMock(),
                config=changed_semantic_config,
            )

            self.assertNotEqual(
                semantic_rag._make_chunking_fingerprint(),
                changed_semantic_rag._make_chunking_fingerprint(),
            )
            self.assertEqual(
                semantic_rag._make_extraction_fingerprint(),
                changed_semantic_rag._make_extraction_fingerprint(),
            )

    async def test_extraction_change_reuses_shared_chunking_cache_only(self):
        config = replace(
            _config(),
            chunk_config=ChunkConfig(
                strategy="agentic",
                agentic_batch_max_sentences=10,
                agentic_batch_max_chars=1000,
                agentic_min_sentences=1,
                agentic_max_sentences=5,
                agentic_concurrency=1,
                agentic_retries=0,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            cache_directory = Path(directory, "shared-cache")
            first_agentic_calls = 0
            first_extraction_calls = 0

            async def first_llm(*, system: str, prompt: str) -> str:
                nonlocal first_agentic_calls, first_extraction_calls
                if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                    first_agentic_calls += 1
                    return '{"propositions": [{"start": 1, "end": 1}]}'
                if system == AGENTIC_STATE_SYSTEM_PROMPT:
                    first_agentic_calls += 1
                    return json.dumps({
                        "action": "new_chunk",
                        "title": "Document",
                        "summary": "A short source document.",
                    })
                first_extraction_calls += 1
                return _empty_extraction_response()

            first_rag = LightRAG(
                working_dir=Path(directory, "build-a"),
                cache_directory=cache_directory,
                llm_func=first_llm,
                con_num=1,
                embed_func=AsyncMock(return_value=[1.0, 0.0]),
                config=config,
            )
            first_result = await first_rag.construct(_documents())

            second_agentic_calls = 0
            second_extraction_calls = 0

            async def second_llm(*, system: str, prompt: str) -> str:
                nonlocal second_agentic_calls, second_extraction_calls
                if system == AGENTIC_PROPOSITION_SYSTEM_PROMPT:
                    second_agentic_calls += 1
                    return '{"propositions": [{"start": 1, "end": 1}]}'
                if system == AGENTIC_STATE_SYSTEM_PROMPT:
                    second_agentic_calls += 1
                    return json.dumps({
                        "action": "new_chunk",
                        "title": "Document",
                        "summary": "A short source document.",
                    })
                second_extraction_calls += 1
                return _empty_extraction_response()

            changed_prompt = (
                PROMPTS["entity_extraction_user_prompt"]
                + "\nAdditional extraction instruction."
            )
            with patch.dict(
                PROMPTS,
                {"entity_extraction_user_prompt": changed_prompt},
            ):
                second_rag = LightRAG(
                    working_dir=Path(directory, "build-b"),
                    cache_directory=cache_directory,
                    llm_func=second_llm,
                    con_num=1,
                    embed_func=AsyncMock(return_value=[1.0, 0.0]),
                    config=config,
                )
                second_result = await second_rag.construct(_documents())

            self.assertEqual(first_agentic_calls, 4)
            self.assertEqual(first_extraction_calls, 2)
            self.assertEqual(second_agentic_calls, 0)
            self.assertEqual(second_extraction_calls, 2)
            self.assertEqual(
                first_result.chunking_fingerprint,
                second_result.chunking_fingerprint,
            )
            self.assertNotEqual(
                first_result.extraction_fingerprint,
                second_result.extraction_fingerprint,
            )
            self.assertNotEqual(
                first_result.build_fingerprint,
                second_result.build_fingerprint,
            )
            self.assertEqual(
                len(list(Path(cache_directory, "chunking").iterdir())),
                1,
            )
            self.assertEqual(
                len(list(Path(cache_directory, "extraction").iterdir())),
                2,
            )

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
            self.assertTrue(
                Path(directory, "pipeline_cache", "extraction").is_dir()
            )

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
