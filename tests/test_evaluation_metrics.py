import unittest

from rag_research.evaluation import (
    build_multidocument_chunk_index,
    calc_evidence_reachability,
    calc_multihop_official_metrics,
    calc_multihop_retrieval_metrics,
    calc_null_retrieval_context_metrics,
    map_multihop_evidence_to_chunks,
)
from rag_research.models import EvidenceOccurrence, EvidenceRecord, InputDocument, QuestionRecord


def question_with(*evidence):
    return QuestionRecord(
        question_id="q1", dataset_index=0, query="Combine facts", answer="answer",
        question_type="inference_query", evidence=tuple(evidence),
    )


def fact(evidence_id, document_id, text, *intervals):
    return EvidenceRecord(
        evidence_id=evidence_id, document_id=document_id, fact=text,
        occurrences=tuple(EvidenceOccurrence(*interval) for interval in intervals),
    )


def chunk(chunk_id, document_id, source, start, end):
    return {
        "chunk_id": chunk_id, "document_id": document_id, "text": source[start:end],
        "char_start": start, "char_end": end,
    }


def score(question, documents, chunks, retrieved_ids, requested_k=None):
    index = build_multidocument_chunk_index(documents, chunks)
    mapping = map_multihop_evidence_to_chunks(question, index)
    return calc_multihop_retrieval_metrics(
        question=question,
        retrieved_chunk_ids=retrieved_ids,
        requested_k=requested_k if requested_k is not None else max(len(retrieved_ids), 1),
        chunks=chunks,
        chunk_to_evidence=mapping,
        token_counter=lambda text: len(text.split()),
    )


class MultiDocumentEvidenceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.documents = (
            InputDocument(document_id="doc-a", text="alpha evidence one omega"),
            InputDocument(document_id="doc-b", text="alpha evidence two omega"),
            InputDocument(document_id="doc-c", text="irrelevant context"),
        )
        a, b, c = (document.text for document in self.documents)
        self.chunks = {
            "a-full": chunk("a-full", "doc-a", a, 0, len(a)),
            "a-duplicate": chunk("a-duplicate", "doc-a", a, 6, 18),
            "b-full": chunk("b-full", "doc-b", b, 0, len(b)),
            "c-irrelevant": chunk("c-irrelevant", "doc-c", c, 0, len(c)),
        }
        self.question = question_with(
            fact("e1", "doc-a", "evidence one", (6, 18)),
            fact("e2", "doc-b", "evidence two", (6, 18)),
        )

    def test_mapping_requires_same_document_and_full_containment(self):
        chunks = {**self.chunks, "a-partial": chunk(
            "a-partial", "doc-a", self.documents[0].text, 6, 14,
        )}
        index = build_multidocument_chunk_index(self.documents, chunks)
        mapping = map_multihop_evidence_to_chunks(self.question, index)
        self.assertEqual(mapping["a-full"], ["e1"])
        self.assertEqual(mapping["a-duplicate"], ["e1"])
        self.assertEqual(mapping["b-full"], ["e2"])
        self.assertNotIn("a-partial", mapping)
        self.assertNotIn("c-irrelevant", mapping)

    def test_corrupt_source_offsets_and_text_remain_errors(self):
        for changes in (
            {"text": "different"}, {"char_start": -1}, {"char_start": True},
            {"char_end": 500}, {"document_id": "unknown"}, {"chunk_id": "other"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_multidocument_chunk_index(self.documents, {
                    "a-full": {**self.chunks["a-full"], **changes},
                })

    def test_ranking_document_recall_and_source_density(self):
        metrics = score(self.question, self.documents, self.chunks, [
            "c-irrelevant", "a-full", "a-duplicate", "b-full",
        ])
        self.assertEqual(metrics["evidence_recall"], 1.0)
        self.assertTrue(metrics["joint_evidence_success"])
        self.assertEqual(metrics["document_recall"], 1.0)
        self.assertTrue(metrics["joint_document_success"])
        self.assertEqual(metrics["chunk_precision"], 3 / 4)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)
        self.assertEqual(metrics["retrieved_document_count"], 3)
        self.assertEqual(metrics["covered_evidence_chars"], 24)
        self.assertEqual(metrics["chunk_source_tokens"], 12)
        self.assertNotIn("retrieved_tokens", metrics)

    def test_document_hit_does_not_imply_evidence_hit(self):
        chunks = {"a-prefix": chunk("a-prefix", "doc-a", self.documents[0].text, 0, 5)}
        metrics = score(self.question, self.documents, chunks, ["a-prefix"])
        self.assertEqual(metrics["document_recall"], 0.5)
        self.assertEqual(metrics["evidence_recall"], 0.0)

    def test_unreachable_facts_remain_in_denominators(self):
        chunks = {"a-full": self.chunks["a-full"]}
        index = build_multidocument_chunk_index(self.documents, chunks)
        mapping = map_multihop_evidence_to_chunks(self.question, index)
        coverage = calc_evidence_reachability(self.question, mapping)
        metrics = score(self.question, self.documents, chunks, ["a-full"])
        self.assertEqual(coverage["reachable_evidence_ids"], ["e1"])
        self.assertEqual(coverage["unreachable_evidence_ids"], ["e2"])
        self.assertEqual(coverage["evidence_recall_ceiling"], 0.5)
        self.assertFalse(coverage["joint_evidence_success_ceiling"])
        self.assertEqual(metrics["gold_evidence_count"], 2)
        self.assertEqual(metrics["evidence_recall"], 0.5)
        self.assertFalse(metrics["joint_evidence_success"])

    def test_empty_ranking_and_no_reachable_gold_are_valid_misses(self):
        metrics = score(self.question, self.documents, {}, [], requested_k=5)
        self.assertEqual(metrics["gold_evidence_count"], 2)
        self.assertEqual(metrics["evidence_recall"], 0.0)
        self.assertEqual(metrics["ndcg_at_k"], 0.0)
        self.assertEqual(metrics["average_precision_at_k"], 0.0)
        self.assertEqual(metrics["evidence_density"], 0.0)
        self.assertFalse(metrics["cross_chunk_union_joint_evidence_success"])

    def test_precision_at_k_penalizes_short_result_list(self):
        metrics = score(self.question, self.documents, self.chunks, ["a-full"], requested_k=5)
        self.assertEqual(metrics["chunk_precision"], 0.2)
        self.assertEqual(metrics["precision_among_returned"], 1.0)
        self.assertEqual(metrics["requested_k"], 5)

    def test_null_query_has_source_context_without_relevance_scores(self):
        metrics = calc_null_retrieval_context_metrics(
            retrieved_chunk_ids=["a-full", "b-full"], chunks=self.chunks,
            token_counter=lambda text: len(text.split()),
        )
        self.assertFalse(metrics["relevance_metrics_applicable"])
        self.assertEqual(metrics["retrieved_document_count"], 2)
        self.assertEqual(metrics["chunk_source_tokens"], 8)
        self.assertNotIn("evidence_recall", metrics)
        self.assertNotIn("retrieved_tokens", metrics)


class SourceIntervalCoverageTests(unittest.TestCase):
    def test_adjacent_chunks_cover_one_occurrence_only_in_union_protocol(self):
        source = "abcdefghij"
        documents = [InputDocument(document_id="a", text=source)]
        question = question_with(fact("e", "a", source, (0, 10)))
        chunks = {
            "left": chunk("left", "a", source, 0, 5),
            "right": chunk("right", "a", source, 5, 10),
        }
        metrics = score(question, documents, chunks, ["left", "right"])
        self.assertEqual(metrics["evidence_recall"], 0.0)
        self.assertEqual(metrics["cross_chunk_union_evidence_recall"], 1.0)
        self.assertTrue(metrics["cross_chunk_union_joint_evidence_success"])
        self.assertEqual(metrics["covered_evidence_chars"], 10)
        self.assertEqual(metrics["evidence_density"], 1.0)

    def test_gap_or_cross_document_fragments_do_not_complete_a_fact(self):
        source = "abcdefghij"
        documents = [InputDocument(document_id=doc, text=source) for doc in ("a", "b")]
        question = question_with(fact("e", "a", source, (0, 10)))
        for left_document, right_start, expected_chars in (("a", 6, 9), ("b", 5, 5)):
            chunks = {
                "left": chunk("left", left_document, source, 0, 5),
                "right": chunk("right", "a", source, right_start, 10),
            }
            with self.subTest(left_document=left_document, right_start=right_start):
                metrics = score(question, documents, chunks, ["left", "right"])
                self.assertEqual(metrics["cross_chunk_union_evidence_recall"], 0.0)
                self.assertEqual(metrics["covered_evidence_chars"], expected_chars)

    def test_fragments_from_different_occurrences_cannot_be_joined(self):
        source = "abcdef--abcdef"
        documents = [InputDocument(document_id="a", text=source)]
        question = question_with(fact("e", "a", "abcdef", (0, 6), (8, 14)))
        chunks = {
            "first-half": chunk("first-half", "a", source, 0, 3),
            "second-half": chunk("second-half", "a", source, 11, 14),
        }
        metrics = score(question, documents, chunks, list(chunks))
        self.assertEqual(metrics["evidence_recall"], 0.0)
        self.assertEqual(metrics["cross_chunk_union_evidence_recall"], 0.0)
        self.assertEqual(metrics["covered_evidence_chars"], 6)
        # High character density is not a claim of complete fact retrieval.
        self.assertEqual(metrics["evidence_density"], 1.0)

    def test_density_unions_nested_facts_and_duplicate_source_coverage(self):
        source = "abcdefghij"
        documents = [InputDocument(document_id="a", text=source)]
        question = question_with(
            fact("outer", "a", source, (0, 10)), fact("inner", "a", "bcdefgh", (1, 8)),
        )
        chunks = {
            "first": chunk("first", "a", source, 0, 10),
            "duplicate": chunk("duplicate", "a", source, 0, 10),
        }
        metrics = score(question, documents, chunks, list(chunks))
        self.assertEqual(metrics["covered_evidence_chars"], 10)
        self.assertEqual(metrics["retrieved_chars"], 20)
        self.assertEqual(metrics["evidence_density"], 0.5)
        self.assertEqual(metrics["matched_evidence_count"], 2)

    def test_density_counts_distinct_source_occurrences_but_fact_recall_once(self):
        source = "abcdef--abcdef"
        documents = [InputDocument(document_id="a", text=source)]
        question = question_with(fact("e", "a", "abcdef", (0, 6), (8, 14)))
        chunks = {
            "first": chunk("first", "a", source, 0, 6),
            "second": chunk("second", "a", source, 8, 14),
        }
        metrics = score(question, documents, chunks, list(chunks))
        self.assertEqual(metrics["covered_evidence_chars"], 12)
        self.assertEqual(metrics["evidence_density"], 1.0)
        self.assertEqual(metrics["matched_evidence_count"], 1)


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
