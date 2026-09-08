import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts import evaluate_multihop_retrieval as runner
from rag_research.core import LightRAG
from rag_research.datasets.multihop_rag import load_multihop_rag
from rag_research.evaluation import (
    build_multidocument_chunk_index, calc_evidence_reachability,
    map_multihop_evidence_to_chunks,
)

from scripts.evaluate_multihop_retrieval import (
    build_official_export,
    build_summaries,
    compact_trace,
    parse_k_values,
)


class MultiHopEvaluatorRunnerTests(unittest.TestCase):
    def test_k_values_are_positive_sorted_and_unique(self):
        self.assertEqual(parse_k_values("20, 1,5,5"), (1, 5, 20))
        with self.assertRaisesRegex(ValueError, "positive integers"):
            parse_k_values("1,0")

    def test_compact_trace_keeps_ranking_provenance_without_large_text_fields(self):
        trace = compact_trace({
            "query": "question",
            "mode": "local",
            "requested_top_k": 5,
            "entity_ids": ["Entity"],
            "relation_ids": ["A||B"],
            "entities": [{
                "name": "Entity",
                "type": "Concept",
                "description": "large description",
                "source_id": ["c1"],
                "dense_score": 0.8,
            }],
            "relations": [{
                "source": "A",
                "target": "B",
                "description": "large description",
                "source_id": ["c1"],
                "rerank_score": 0.7,
            }],
            "chunk_ids": ["c1"],
            "chunks": [{
                "chunk_id": "c1",
                "document_id": "doc-1",
                "text": "large chunk text",
                "dense_score": 0.6,
                "retrieval_sources": ["local"],
                "introduced_by": [{"kind": "entity", "id": "Entity"}],
            }],
        })

        self.assertNotIn("description", trace["entities"][0])
        self.assertNotIn("source_id", trace["relations"][0])
        self.assertNotIn("text", trace["chunks"][0])
        self.assertEqual(trace["chunks"][0]["rank"], 1)
        self.assertEqual(trace["chunks"][0]["document_id"], "doc-1")
        self.assertEqual(trace["chunks"][0]["introduced_by"][0]["id"], "Entity")

    def test_summaries_keep_null_queries_out_of_relevance_averages(self):
        answerable_metrics = {
            "requested_k": 2,
            "retrieved_count": 2,
            "relevant_retrieved_count": 1,
            "gold_evidence_count": 2,
            "matched_evidence_count": 1,
            "gold_document_count": 2,
            "matched_document_count": 1,
            "retrieved_chars": 100,
            "covered_evidence_chars": 20,
            "chunk_precision": 0.5,
            "precision_among_returned": 0.5,
            "evidence_recall": 0.5,
            "coverage_f1": 0.5,
            "joint_evidence_success": False,
            "cross_chunk_union_evidence_recall": 0.5,
            "cross_chunk_union_joint_evidence_success": False,
            "cross_chunk_union_matched_evidence_count": 1,
            "document_recall": 0.5,
            "joint_document_success": False,
            "reciprocal_rank": 1.0,
            "average_precision_at_k": 0.5,
            "ndcg_at_k": 0.7,
            "chunk_source_tokens": 25,
            "generation_context_tokens": 40,
            "retrieved_document_count": 2,
            "cross_document_retrieval": True,
        }
        null_metrics = {
            "retrieved_count": 2,
            "retrieved_chars": 120,
            "chunk_source_tokens": 30,
            "generation_context_tokens": 45,
            "retrieved_document_count": 2,
            "cross_document_retrieval": True,
            "relevance_metrics_applicable": False,
        }
        results = [
            {
                "mode": "naive",
                "question_type": "inference_query",
                "question_id": "q-answerable",
                "evidence_count": 2,
                "gold_document_count": 2,
                "index_coverage": {
                    "gold_evidence_count": 2, "reachable_evidence_count": 1,
                    "unreachable_evidence_count": 1, "evidence_recall_ceiling": 0.5,
                    "joint_evidence_success_ceiling": False,
                },
                "thesis_extended": {"metrics_by_k": {"2": answerable_metrics}},
                "official": {
                    "applicable": True,
                    "metrics": {
                        "Hits@4": 1,
                        "Hits@10": 1,
                        "MAP@10": 0.5,
                        "MRR@10": 1.0,
                    },
                },
            },
            {
                "mode": "naive",
                "question_type": "null_query",
                "question_id": "q-null", "evidence_count": 0,
                "gold_document_count": 0, "index_coverage": {"applicable": False},
                "thesis_extended": {"metrics_by_k": {"2": null_metrics}},
                "official": {"applicable": False},
            },
        ]

        summaries = build_summaries(results, ("naive",), (2,))
        overall = next(
            row
            for row in summaries["thesis_extended"]["answerable"]
            if row["group"] == "all_answerable"
        )

        self.assertEqual(overall["question_count"], 1)
        self.assertEqual(overall["macro_evidence_recall"], 0.5)
        self.assertEqual(
            summaries["thesis_extended"]["null_queries"][0]["question_count"],
            1,
        )
        self.assertFalse(
            summaries["thesis_extended"]["null_queries"][0][
                "relevance_metrics_applicable"
            ]
        )
        self.assertEqual(summaries["official"][0]["question_count"], 1)
        self.assertEqual(summaries["official"][0]["Hits@10"], 1.0)
        self.assertEqual(summaries["index_coverage"]["question_count"], 1)
        self.assertEqual(overall["avg_chunk_source_tokens"], 25)
        self.assertEqual(overall["avg_generation_context_tokens"], 40)
        self.assertEqual(summaries["thesis_extended"]["null_queries"][0][
            "avg_generation_context_tokens"], 45)
        self.assertNotIn("avg_retrieved_tokens", overall)
        self.assertEqual({row["group"] for row in summaries["thesis_extended"]["answerable"]}, {
            "all_answerable", "question_type", "evidence_count", "gold_document_count",
        })

    def test_official_export_matches_the_upstream_json_shape(self):
        rows = [{
            "mode": "naive",
            "question": "question",
            "answer": "answer",
            "question_type": "inference_query",
            "gold_evidence": [{
                "fact": "gold fact",
                "metadata": {"title": "Gold title"},
            }],
            "trace": {
                "chunks": [{"chunk_id": "c1", "dense_score": 0.75}],
            },
        }]
        chunks = {"c1": {"model_text": "Title: Gold title\n\ngold fact"}}

        exported = build_official_export(rows, mode="naive", chunks=chunks)

        self.assertEqual(exported[0]["query"], "question")
        self.assertEqual(exported[0]["gold_list"], [{
            "title": "Gold title",
            "fact": "gold fact",
        }])
        self.assertEqual(exported[0]["retrieval_list"], [{
            "text": "Title: Gold title\n\ngold fact",
            "score": 0.75,
        }])


class MultiHopRunnerIntegrationTests(unittest.TestCase):
    def test_unreachable_evidence_completes_exports_resumes_and_rejects_old_schema(self):
        """Exercise the real runner and dataset loader with no model or network calls."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_directory = root / "dataset"
            working_directory = root / "store"
            dataset_directory.mkdir()
            working_directory.mkdir()
            raw_documents = [
                {
                    "url": f"https://example.test/{index}", "title": f"Document {index}",
                    "source": "Example", "author": "Author", "published_at": "2024-01-01",
                    "category": "News", "body": body,
                }
                for index, body in enumerate(("alpha fact omega", "second evidence"))
            ]
            raw_evidence = [
                {**{key: value for key, value in document.items() if key != "body"}, "fact": fact}
                for document, fact in zip(raw_documents, ("alpha fact", "second evidence"))
            ]
            (dataset_directory / "corpus.json").write_text(json.dumps(raw_documents))
            (dataset_directory / "MultiHopRAG.json").write_text(json.dumps([
                {"query": "Combine the facts", "answer": "Both facts", "question_type": "inference_query",
                 "evidence_list": raw_evidence},
                {"query": "Unknown information", "answer": "Insufficient information.",
                 "question_type": "null_query", "evidence_list": []},
            ]))
            # Presence is checked by the runner; model/store construction is mocked below.
            (working_directory / "build_manifest.json").write_text('{"build": {}}')
            chunks = {}
            for chunk_id, document_index, start, end in (
                ("left", 0, 0, 5), ("right", 0, 5, len(raw_documents[0]["body"])),
                ("other", 1, 0, len(raw_documents[1]["body"])),
            ):
                document = raw_documents[document_index]
                text = document["body"][start:end]
                chunks[chunk_id] = {
                    "chunk_id": chunk_id, "document_id": document["url"],
                    "text": text, "model_text": f"Title: {document['title']}\n\n{text}",
                    "char_start": start, "char_end": end,
                }

            async def retrieve_trace(*, query, mode, top_k):
                return {
                    "query": query, "mode": mode, "requested_top_k": top_k,
                    "chunk_ids": list(chunks), "chunks": list(chunks.values()),
                    "entities": [{"name": "Graph entity", "type": "Concept",
                                  "description": "Extra graph context for generation"}]
                    if mode == "hybrid" else [],
                    "relations": [], "entity_ids": [], "relation_ids": [],
                }

            rag = SimpleNamespace(
                construct=AsyncMock(return_value=SimpleNamespace(
                    build_fingerprint="build-fixture", chunking_fingerprint="chunking-fixture",
                    extraction_fingerprint="extraction-fixture", chunk_count=len(chunks),
                )),
                retrieve_trace=AsyncMock(side_effect=retrieve_trace),
                chunk_kv=SimpleNamespace(all=lambda: chunks),
                config=SimpleNamespace(
                    chunk_candidate_top_k=20, entity_top_k=10, relation_top_k=5,
                    relation_candidate_top_k=20, chunk_config=SimpleNamespace(strategy="fixed"),
                ),
                build_context=lambda entities, relations, prefix: LightRAG.build_context(
                    None, entities, relations, prefix,
                ),
            )
            tokenizer = SimpleNamespace(tokenize=lambda text: text.split())
            environment = {
                "MULTIHOP_DATASET_DIR": str(dataset_directory),
                "WORKING_DIR": str(working_directory),
                "CACHE_DIR": str(root / "cache"),
                "EVAL_OUTPUT_DIR": str(root / "evaluations"),
                "EVAL_RUN_NAME": "offline-fixture", "EVAL_MODES": "naive,hybrid",
                "EVAL_K_VALUES": "1,3,5", "EVAL_INCLUDE_NULL": "true", "EVAL_CONCURRENCY": "2",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(runner, "LightRAG", return_value=rag),
                patch.object(runner, "create_reranker", return_value=None),
                patch.object(runner.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                patch.object(runner, "llm_func", AsyncMock(side_effect=AssertionError("LLM called"))) as llm,
                patch.object(runner, "embed_func", AsyncMock(side_effect=AssertionError("embedding called"))) as embed,
                patch.object(runner, "embed_many_func", AsyncMock(side_effect=AssertionError("batch embedding called"))) as embed_many,
            ):
                run_directory, manifest = asyncio.run(runner.main())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["schema_version"], 3)
                self.assertEqual(manifest["result_count"], 4)
                self.assertEqual(manifest["index_coverage"]["unreachable_evidence_count"], 1)
                self.assertEqual(manifest["index_coverage"]["joint_evidence_success_rate_ceiling"], 0.0)
                result_bytes = (run_directory / "results.json").read_bytes()
                results = json.loads(result_bytes)
                answerable = [row for row in results if row["question_type"] != "null_query"]
                for row in answerable:
                    self.assertEqual(row["index_coverage"]["reachable_evidence_count"], 1)
                    metrics = row["thesis_extended"]["metrics_by_k"]["3"]
                    self.assertEqual(metrics["evidence_recall"], 0.5)
                    self.assertEqual(metrics["cross_chunk_union_evidence_recall"], 1.0)
                    self.assertNotIn("retrieved_tokens", metrics)
                    self.assertIn("generation_context_tokens", metrics)
                    self.assertNotIn("hop_count", row)
                    self.assertEqual(row["evidence_count"], 2)
                null = next(row for row in results if row["question_type"] == "null_query")
                self.assertNotIn("evidence_recall", null["thesis_extended"]["metrics_by_k"]["3"])
                self.assertIn("generation_context_tokens", null["thesis_extended"]["metrics_by_k"]["3"])
                summaries = json.loads((run_directory / "summaries.json").read_text())
                k5 = next(row for row in summaries["thesis_extended"]["answerable"]
                          if row["group"] == "all_answerable" and row["mode"] == "naive"
                          and row["requested_k"] == 5)
                self.assertEqual(k5["macro_chunk_precision"], 1 / 5)
                self.assertEqual(k5["micro_chunk_precision"], 1 / 5)
                self.assertEqual(k5["micro_precision_among_returned"], 1 / 3)
                self.assertEqual(summaries["index_coverage"]["question_count"], 1)
                self.assertTrue(all(row["question_count"] == 1 for row in summaries["official"]))
                official_export = json.loads((run_directory / "official/naive.json").read_text())
                self.assertEqual(len(official_export), 2)  # Upstream filters null rows itself.
                self.assertEqual(official_export[0]["retrieval_list"][0]["text"], chunks["left"]["model_text"])
                # The unchanged official matcher does not union two retrieved chunks.
                self.assertEqual(answerable[0]["official"]["metrics"]["MAP@10"], (1 / 3) / 2)
                self.assertEqual(len(list((run_directory / "checkpoints").rglob("*.json"))), 4)
                self.assertEqual(rag.retrieve_trace.await_count, 4)

                rag.retrieve_trace.side_effect = AssertionError("resume must not retrieve")
                second_directory, second_manifest = asyncio.run(runner.main())
                self.assertEqual(second_directory, run_directory)
                self.assertEqual(second_manifest["evaluation_fingerprint"], manifest["evaluation_fingerprint"])
                self.assertEqual((run_directory / "results.json").read_bytes(), result_bytes)
                self.assertEqual(rag.retrieve_trace.await_count, 4)

                checkpoint = next((run_directory / "checkpoints").rglob("*.json"))
                original_checkpoint = checkpoint.read_text()
                incompatible = json.loads(original_checkpoint)
                incompatible["schema_version"] = 2
                checkpoint.write_text(json.dumps(incompatible))
                with self.assertRaisesRegex(RuntimeError, "checkpoint schema"):
                    asyncio.run(runner.main())
                checkpoint.write_text(original_checkpoint)
                manifest_path = run_directory / "run_manifest.json"
                incompatible_manifest = json.loads(manifest_path.read_text())
                incompatible_manifest["schema_version"] = 2
                manifest_path.write_text(json.dumps(incompatible_manifest))
                with self.assertRaisesRegex(RuntimeError, "different experiment"):
                    asyncio.run(runner.main())
                llm.assert_not_awaited()
                embed.assert_not_awaited()
                embed_many.assert_not_awaited()


class MultiHopRealDatasetMappingSmokeTests(unittest.TestCase):
    def test_all_real_questions_are_mapped_without_dropping_boundary_failures(self):
        """Read the optional local benchmark; never touch stored indexes or call models."""
        dataset_directory = Path(__file__).resolve().parents[1] / "data/raw/MultiHopRAG"
        if not (dataset_directory / "MultiHopRAG.json").is_file():
            self.skipTest("local MultiHopRAG benchmark is not installed")
        dataset = load_multihop_rag(dataset_directory)
        self.assertEqual(len(dataset.documents), 609)
        self.assertEqual(len(dataset.questions), 2556)
        # A deterministic 1800-character, zero-overlap baseline reproduces the
        # real boundary regression without depending on a mutable stored index.
        chunks = {}
        for document_index, document in enumerate(dataset.documents):
            for start in range(0, len(document.text), 1800):
                end = min(start + 1800, len(document.text))
                chunk_id = f"d{document_index}-{start}"
                chunks[chunk_id] = {
                    "chunk_id": chunk_id, "document_id": document.document_id,
                    "text": document.text[start:end], "char_start": start, "char_end": end,
                }
        index = build_multidocument_chunk_index(dataset.documents, chunks)
        coverages = [
            calc_evidence_reachability(question, map_multihop_evidence_to_chunks(question, index))
            for question in dataset.questions if question.question_type != "null_query"
        ]
        summary = runner.summarize_index_coverage(coverages)
        self.assertEqual(summary["question_count"], 2255)
        self.assertEqual(summary["gold_evidence_count"], 6084)
        self.assertEqual(summary["unreachable_evidence_count"], 313)
        self.assertEqual(summary["questions_with_unreachable_evidence"], 303)
        self.assertAlmostEqual(summary["joint_evidence_success_rate_ceiling"], (2255 - 303) / 2255)


if __name__ == "__main__":
    unittest.main()
