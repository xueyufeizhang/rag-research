import unittest

from rag_research.evaluation import (
    build_entity_normalizer,
    calc_context_efficiency_metrics,
    calc_evidence_metrics,
    calc_set_metrics,
    normalize_relation_key,
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


if __name__ == "__main__":
    unittest.main()
