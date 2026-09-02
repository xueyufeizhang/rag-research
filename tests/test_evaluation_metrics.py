import unittest

from rag_research.evaluation import (
    build_multidocument_chunk_index,
    build_entity_normalizer,
    calc_context_efficiency_metrics,
    calc_evidence_metrics,
    calc_multihop_official_metrics,
    calc_multihop_retrieval_metrics,
    calc_null_retrieval_context_metrics,
    calc_set_metrics,
    map_multihop_evidence_to_chunks,
    normalize_relation_key,
)
from rag_research.models import (
    EvidenceOccurrence,
    EvidenceRecord,
    InputDocument,
    QuestionRecord,
)


class EvidenceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.chunk_to_evidence = {
            "c1": ["e1"],
            "c2": ["e1"],
            "c3": ["e2"],
            "c4": [],
        }
        self.evidence_to_answer_points = {"e1": [1], "e2": [2, 3]}

    def test_recall_counts_evidence_not_overlapping_chunks(self):
        metrics = calc_evidence_metrics(
            ["c1", "c2"],
            self.chunk_to_evidence,
            self.evidence_to_answer_points,
            answer_point_count=3,
        )

        self.assertEqual(metrics["chunk_precision"], 1.0)
        self.assertEqual(metrics["evidence_recall"], 0.5)
        self.assertEqual(metrics["answer_point_recall"], 1 / 3)
        self.assertEqual(metrics["redundancy_rate"], 0.5)
        self.assertEqual(metrics["ndcg_at_k"], 1.0)
        self.assertNotIn("precision", metrics)
        self.assertNotIn("recall", metrics)
        self.assertNotIn("f1", metrics)

    def test_rank_and_full_coverage(self):
        metrics = calc_evidence_metrics(
            ["c4", "c1", "c3"],
            self.chunk_to_evidence,
            self.evidence_to_answer_points,
            answer_point_count=3,
        )

        self.assertEqual(metrics["chunk_precision"], 2 / 3)
        self.assertEqual(metrics["evidence_recall"], 1.0)
        self.assertEqual(metrics["answer_point_recall"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["first_relevant_rank"], 2)
        self.assertLess(metrics["ndcg_at_k"], 1.0)

    def test_context_efficiency_counts_unique_evidence_against_full_context(self):
        source = "alpha beta gamma delta epsilon"
        chunks = {
            "c1": {"text": "alpha beta gamma"},
            "c2": {"text": "gamma delta epsilon"},
        }
        chunk_intervals = {
            "c1": (0, len("alpha beta gamma")),
            "c2": (source.index("gamma"), len(source)),
        }
        evidence_spans = [
            {
                "evidence_id": "e1",
                "char_start": source.index("beta"),
                "char_end": source.index("gamma") + len("gamma"),
            },
            {
                "evidence_id": "e2",
                "char_start": source.index("delta"),
                "char_end": source.index("delta") + len("delta"),
            },
        ]

        metrics = calc_context_efficiency_metrics(
            source=source,
            retrieved_chunk_ids=["c1", "c2"],
            chunks=chunks,
            chunk_intervals=chunk_intervals,
            gold_evidence_spans=evidence_spans,
            matched_answer_point_count=2,
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(metrics["retrieved_chars"], 35)
        self.assertEqual(metrics["retrieved_tokens"], 6)
        self.assertEqual(metrics["covered_evidence_chars"], 15)
        self.assertAlmostEqual(metrics["evidence_density"], 15 / 35)
        self.assertAlmostEqual(metrics["answer_points_per_1k_tokens"], 2000 / 6)


class EntityAndRelationNormalizationTests(unittest.TestCase):
    def test_aliases_are_applied_to_entities_and_relation_endpoints(self):
        normalize_entity = build_entity_normalizer(
            ["Jacob Marley", "Scrooge"],
            {"Marley": "Jacob Marley", "Ebenezer Scrooge": "Scrooge"},
        )

        metrics = calc_set_metrics(
            ["Marley", "Ebenezer Scrooge"],
            ["Jacob Marley", "Scrooge"],
            normalize_entity,
        )
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(
            normalize_relation_key("Ebenezer Scrooge||Marley", normalize_entity),
            "Jacob Marley||Scrooge",
        )


class MultiDocumentEvidenceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.documents = (
            InputDocument(document_id="doc-a", text="alpha evidence one omega"),
            InputDocument(document_id="doc-b", text="alpha evidence two omega"),
            InputDocument(document_id="doc-c", text="irrelevant context"),
        )
        self.chunks = {
            "a-full": {
                "chunk_id": "a-full",
                "document_id": "doc-a",
                "text": "alpha evidence one omega",
                "char_start": 0,
                "char_end": len("alpha evidence one omega"),
            },
            "a-duplicate": {
                "chunk_id": "a-duplicate",
                "document_id": "doc-a",
                "text": "evidence one",
                "char_start": len("alpha "),
                "char_end": len("alpha evidence one"),
            },
            "b-full": {
                "chunk_id": "b-full",
                "document_id": "doc-b",
                "text": "alpha evidence two omega",
                "char_start": 0,
                "char_end": len("alpha evidence two omega"),
            },
            "c-irrelevant": {
                "chunk_id": "c-irrelevant",
                "document_id": "doc-c",
                "text": "irrelevant context",
                "char_start": 0,
                "char_end": len("irrelevant context"),
            },
        }
        self.question = QuestionRecord(
            question_id="q1",
            dataset_index=0,
            query="Combine both facts",
            answer="answer",
            question_type="inference_query",
            evidence=(
                EvidenceRecord(
                    evidence_id="e1",
                    document_id="doc-a",
                    fact="evidence one",
                    occurrences=(EvidenceOccurrence(6, 18),),
                ),
                EvidenceRecord(
                    evidence_id="e2",
                    document_id="doc-b",
                    fact="evidence two",
                    occurrences=(EvidenceOccurrence(6, 18),),
                ),
            ),
        )

    def test_evidence_mapping_is_document_scoped_and_requires_full_containment(self):
        partial_chunks = {
            **self.chunks,
            "a-partial": {
                "chunk_id": "a-partial",
                "document_id": "doc-a",
                "text": "evidence",
                "char_start": 6,
                "char_end": 14,
            },
        }
        index = build_multidocument_chunk_index(self.documents, partial_chunks)

        mapping = map_multihop_evidence_to_chunks(self.question, index)

        self.assertEqual(mapping["a-full"], ["e1"])
        self.assertEqual(mapping["a-duplicate"], ["e1"])
        self.assertEqual(mapping["b-full"], ["e2"])
        self.assertNotIn("a-partial", mapping)
        self.assertNotIn("c-irrelevant", mapping)

    def test_chunk_index_rejects_source_mismatch(self):
        invalid = {
            "bad": {
                "chunk_id": "bad",
                "document_id": "doc-a",
                "text": "different",
                "char_start": 0,
                "char_end": 5,
            }
        }

        with self.assertRaisesRegex(ValueError, "source slice"):
            build_multidocument_chunk_index(self.documents, invalid)

    def test_joint_evidence_and_document_metrics_use_the_ranked_prefix(self):
        index = build_multidocument_chunk_index(self.documents, self.chunks)
        mapping = map_multihop_evidence_to_chunks(self.question, index)

        metrics = calc_multihop_retrieval_metrics(
            retrieved_chunk_ids=[
                "c-irrelevant",
                "a-full",
                "a-duplicate",
                "b-full",
            ],
            chunks=self.chunks,
            chunk_to_evidence=mapping,
            evidence_to_document={"e1": "doc-a", "e2": "doc-b"},
            evidence_lengths={"e1": 12, "e2": 12},
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(metrics["evidence_recall"], 1.0)
        self.assertTrue(metrics["joint_evidence_success"])
        self.assertEqual(metrics["document_recall"], 1.0)
        self.assertTrue(metrics["joint_document_success"])
        self.assertEqual(metrics["chunk_precision"], 3 / 4)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["retrieved_document_count"], 3)
        self.assertTrue(metrics["cross_document_retrieval"])
        self.assertEqual(metrics["covered_evidence_chars"], 24)

    def test_null_query_reports_context_without_relevance_scores(self):
        metrics = calc_null_retrieval_context_metrics(
            retrieved_chunk_ids=["a-full", "b-full"],
            chunks=self.chunks,
            token_counter=lambda text: len(text.split()),
        )

        self.assertFalse(metrics["relevance_metrics_applicable"])
        self.assertEqual(metrics["retrieved_document_count"], 2)
        self.assertTrue(metrics["cross_document_retrieval"])
        self.assertNotIn("evidence_recall", metrics)

    def test_precision_at_k_penalizes_a_short_result_list(self):
        index = build_multidocument_chunk_index(self.documents, self.chunks)
        mapping = map_multihop_evidence_to_chunks(self.question, index)

        metrics = calc_multihop_retrieval_metrics(
            retrieved_chunk_ids=["a-full"],
            requested_k=5,
            chunks=self.chunks,
            chunk_to_evidence=mapping,
            evidence_to_document={"e1": "doc-a", "e2": "doc-b"},
            evidence_lengths={"e1": 12, "e2": 12},
            token_counter=lambda text: len(text.split()),
        )

        self.assertEqual(metrics["retrieved_count"], 1)
        self.assertEqual(metrics["requested_k"], 5)
        self.assertEqual(metrics["chunk_precision"], 0.2)


class MultiHopOfficialMetricsTests(unittest.TestCase):
    def test_reproduces_official_normalization_and_map_formula(self):
        metrics = calc_multihop_official_metrics(
            retrieved_texts=[
                "irrelevant",
                "prefix fact\n one suffix",
                "fact one appears again",
                "prefix fact two suffix",
            ],
            gold_facts=["fact one", "fact two"],
        )

        self.assertEqual(metrics["Hits@4"], 1)
        self.assertEqual(metrics["Hits@10"], 1)
        self.assertEqual(metrics["MAP@10"], (1 / 2 + 1 / 4) / 2)
        self.assertEqual(metrics["MRR@10"], 1 / 2)
        self.assertEqual(metrics["matched_gold_count"], 2)

    def test_hit_after_rank_four_only_counts_for_hits_at_ten(self):
        metrics = calc_multihop_official_metrics(
            retrieved_texts=["no"] * 4 + ["the gold fact is here"],
            gold_facts=["gold fact"],
        )

        self.assertEqual(metrics["Hits@4"], 0)
        self.assertEqual(metrics["Hits@10"], 1)
        self.assertEqual(metrics["MAP@10"], 1 / 5)
        self.assertEqual(metrics["MRR@10"], 1 / 5)

    def test_never_scores_beyond_the_official_top_ten(self):
        metrics = calc_multihop_official_metrics(
            retrieved_texts=["no"] * 10 + ["gold"],
            gold_facts=["gold"],
        )

        self.assertEqual(metrics["Hits@10"], 0)
        self.assertEqual(metrics["MAP@10"], 0.0)
        self.assertEqual(metrics["MRR@10"], 0.0)


if __name__ == "__main__":
    unittest.main()
