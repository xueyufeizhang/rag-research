import unittest

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
            "retrieved_count": 2,
            "relevant_retrieved_count": 1,
            "gold_evidence_count": 2,
            "matched_evidence_count": 1,
            "gold_document_count": 2,
            "matched_document_count": 1,
            "retrieved_chars": 100,
            "covered_evidence_chars": 20,
            "chunk_precision": 0.5,
            "evidence_recall": 0.5,
            "coverage_f1": 0.5,
            "joint_evidence_success": False,
            "document_recall": 0.5,
            "joint_document_success": False,
            "reciprocal_rank": 1.0,
            "average_precision_at_k": 0.5,
            "ndcg_at_k": 0.7,
            "retrieved_tokens": 25,
            "retrieved_document_count": 2,
            "cross_document_retrieval": True,
        }
        null_metrics = {
            "retrieved_count": 2,
            "retrieved_chars": 120,
            "retrieved_tokens": 30,
            "retrieved_document_count": 2,
            "cross_document_retrieval": True,
            "relevance_metrics_applicable": False,
        }
        results = [
            {
                "mode": "naive",
                "question_type": "inference_query",
                "hop_count": 2,
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
                "hop_count": 0,
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


if __name__ == "__main__":
    unittest.main()
