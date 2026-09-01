import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rag_research.datasets.multihop_rag import (
    NULL_QUERY_ANSWER,
    load_multihop_documents,
    load_multihop_questions,
    load_multihop_rag,
)


def make_document(
    url: str,
    body: str,
    *,
    author: str | None = "Author",
) -> dict[str, object]:
    return {
        "title": f"Title for {url}",
        "author": author,
        "source": "Test Source",
        "published_at": "2024-01-01",
        "category": "test",
        "url": url,
        "body": body,
    }


def make_evidence(document: dict[str, object], fact: str) -> dict[str, object]:
    return {
        "title": document["title"],
        "author": document["author"],
        "source": document["source"],
        "published_at": document["published_at"],
        "category": document["category"],
        "url": document["url"],
        "fact": fact,
    }


def make_question(
    query: str,
    evidences: list[dict[str, object]],
    *,
    answer: str = "An answer",
    question_type: str = "inference_query",
) -> dict[str, object]:
    return {
        "query": query,
        "answer": answer,
        "question_type": question_type,
        "evidence_list": evidences,
    }


class MultiHopRAGLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary_directory.name)
        self.corpus_path = self.directory / "corpus.json"
        self.questions_path = self.directory / "MultiHopRAG.json"

        self.raw_documents = [
            make_document("https://example.test/a", "ababa and a first fact"),
            make_document(
                "https://example.test/b",
                "A second fact appears here.",
                author=None,
            ),
        ]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.write_text(
            json.dumps(value, ensure_ascii=False),
            encoding="utf-8",
        )

    def _write_corpus(self, documents: object | None = None) -> None:
        self._write_json(
            self.corpus_path,
            self.raw_documents if documents is None else documents,
        )

    def _load_documents(self):
        self._write_corpus()
        return load_multihop_documents(self.corpus_path)

    def _valid_non_null_question(self) -> dict[str, object]:
        return make_question(
            "What follows from the two facts?",
            [
                make_evidence(self.raw_documents[0], "first fact"),
                make_evidence(self.raw_documents[1], "second fact"),
            ],
        )

    def test_documents_are_source_ordered_immutable_records(self):
        self.raw_documents.append(
            make_document(
                "https://example.test/c",
                "Third document.",
                author="",
            )
        )

        documents = self._load_documents()

        self.assertIsInstance(documents, tuple)
        self.assertEqual(
            [document.document_id for document in documents],
            [document["url"] for document in self.raw_documents],
        )
        self.assertEqual(documents[0].text, self.raw_documents[0]["body"])
        self.assertIsNone(documents[1].metadata["author"])
        self.assertEqual(documents[2].metadata["author"], "")
        self.assertNotIn("body", documents[0].metadata)

    def test_duplicate_document_urls_are_rejected(self):
        duplicate = dict(self.raw_documents[0])
        self._write_corpus([self.raw_documents[0], duplicate])

        with self.assertRaisesRegex(ValueError, "duplicate corpus URL"):
            load_multihop_documents(self.corpus_path)

    def test_required_document_strings_are_validated(self):
        required_fields = ("url", "body", "title", "source", "published_at", "category")

        for field in required_fields:
            with self.subTest(field=field):
                invalid_document = dict(self.raw_documents[0])
                invalid_document[field] = "   "
                self._write_corpus([invalid_document])

                with self.assertRaisesRegex(ValueError, field):
                    load_multihop_documents(self.corpus_path)

    def test_author_must_be_a_string_or_null(self):
        invalid_document = dict(self.raw_documents[0])
        invalid_document["author"] = 42
        self._write_corpus([invalid_document])

        with self.assertRaisesRegex(ValueError, "author must be a string or null"):
            load_multihop_documents(self.corpus_path)

    def test_questions_preserve_order_and_locate_every_fact_occurrence(self):
        documents = self._load_documents()
        repeated_fact_question = make_question(
            "Where does the repeated fact occur?",
            [
                make_evidence(self.raw_documents[0], "aba"),
                make_evidence(self.raw_documents[1], "second fact"),
            ],
        )
        second_question = self._valid_non_null_question()
        self._write_json(self.questions_path, [repeated_fact_question, second_question])

        questions = load_multihop_questions(self.questions_path, documents)

        self.assertIsInstance(questions, tuple)
        self.assertEqual([question.dataset_index for question in questions], [0, 1])
        self.assertEqual(
            [question.query for question in questions],
            [repeated_fact_question["query"], second_question["query"]],
        )
        self.assertIsInstance(questions[0].evidence, tuple)
        self.assertEqual(
            [(item.char_start, item.char_end) for item in questions[0].evidence[0].occurrences],
            [(0, 3), (2, 5)],
        )

        source = documents[0].text
        for occurrence in questions[0].evidence[0].occurrences:
            self.assertEqual(
                source[occurrence.char_start:occurrence.char_end],
                questions[0].evidence[0].fact,
            )

    def test_generated_ids_are_stable_and_unique(self):
        documents = self._load_documents()
        raw_questions = [
            self._valid_non_null_question(),
            make_question(
                "A different question",
                [
                    make_evidence(self.raw_documents[0], "first fact"),
                    make_evidence(self.raw_documents[1], "second fact"),
                ],
                question_type="comparison_query",
            ),
        ]
        self._write_json(self.questions_path, raw_questions)

        first_load = load_multihop_questions(self.questions_path, documents)
        second_load = load_multihop_questions(self.questions_path, documents)

        first_question_ids = [question.question_id for question in first_load]
        second_question_ids = [question.question_id for question in second_load]
        first_evidence_ids = [
            evidence.evidence_id
            for question in first_load
            for evidence in question.evidence
        ]
        second_evidence_ids = [
            evidence.evidence_id
            for question in second_load
            for evidence in question.evidence
        ]

        self.assertEqual(first_question_ids, second_question_ids)
        self.assertEqual(first_evidence_ids, second_evidence_ids)
        self.assertEqual(len(first_question_ids), len(set(first_question_ids)))
        self.assertEqual(len(first_evidence_ids), len(set(first_evidence_ids)))
        self.assertTrue(all(item.startswith("mhr-q-") for item in first_question_ids))
        self.assertTrue(all(item.startswith("mhr-e-") for item in first_evidence_ids))

    def test_valid_null_query_has_no_evidence(self):
        documents = self._load_documents()
        raw_question = make_question(
            "A question the corpus cannot answer",
            [],
            answer=NULL_QUERY_ANSWER,
            question_type="null_query",
        )
        self._write_json(self.questions_path, [raw_question])

        questions = load_multihop_questions(self.questions_path, documents)

        self.assertEqual(questions[0].answer, NULL_QUERY_ANSWER)
        self.assertEqual(questions[0].evidence, ())

    def test_required_question_strings_are_validated(self):
        documents = self._load_documents()

        for field in ("query", "answer", "question_type"):
            with self.subTest(field=field):
                raw_question = self._valid_non_null_question()
                raw_question[field] = "   "
                self._write_json(self.questions_path, [raw_question])

                with self.assertRaisesRegex(ValueError, field):
                    load_multihop_questions(self.questions_path, documents)

    def test_evidence_list_must_be_an_array_of_objects(self):
        documents = self._load_documents()
        cases = [
            (None, "evidence_list must be a list"),
            ({"not": "a list"}, "evidence_list must be a list"),
            (["not an object", "not an object"], "must be an object"),
        ]

        for evidence_list, message in cases:
            with self.subTest(evidence_list=evidence_list):
                raw_question = self._valid_non_null_question()
                raw_question["evidence_list"] = evidence_list
                self._write_json(self.questions_path, [raw_question])

                with self.assertRaisesRegex(ValueError, message):
                    load_multihop_questions(self.questions_path, documents)

    def test_required_evidence_strings_are_validated(self):
        documents = self._load_documents()

        for field in ("url", "fact"):
            with self.subTest(field=field):
                raw_question = self._valid_non_null_question()
                first_evidence = raw_question["evidence_list"][0]
                first_evidence[field] = "   "
                self._write_json(self.questions_path, [raw_question])

                with self.assertRaisesRegex(ValueError, field):
                    load_multihop_questions(self.questions_path, documents)

    def test_question_type_must_be_known(self):
        documents = self._load_documents()
        raw_question = self._valid_non_null_question()
        raw_question["question_type"] = "unsupported_query"
        self._write_json(self.questions_path, [raw_question])

        with self.assertRaisesRegex(ValueError, "unknown question type"):
            load_multihop_questions(self.questions_path, documents)

    def test_null_query_invariants_are_enforced(self):
        documents = self._load_documents()
        cases = [
            (
                make_question(
                    "Null query with evidence",
                    [make_evidence(self.raw_documents[0], "first fact")],
                    answer=NULL_QUERY_ANSWER,
                    question_type="null_query",
                ),
                "must not have evidence",
            ),
            (
                make_question(
                    "Null query with a regular answer",
                    [],
                    answer="A regular answer",
                    question_type="null_query",
                ),
                "must use",
            ),
        ]

        for raw_question, message in cases:
            with self.subTest(message=message):
                self._write_json(self.questions_path, [raw_question])
                with self.assertRaisesRegex(ValueError, message):
                    load_multihop_questions(self.questions_path, documents)

    def test_non_null_query_invariants_are_enforced(self):
        documents = self._load_documents()
        evidence = make_evidence(self.raw_documents[0], "first fact")
        cases = [
            (
                make_question(
                    "Non-null query with null answer",
                    [evidence, make_evidence(self.raw_documents[1], "second fact")],
                    answer=NULL_QUERY_ANSWER,
                ),
                "non-null query has null answer",
            ),
            (make_question("Too little evidence", [evidence]), "2–4 evidence"),
            (
                make_question("Too much evidence", [evidence] * 5),
                "2–4 evidence",
            ),
        ]

        for raw_question, message in cases:
            with self.subTest(message=message):
                self._write_json(self.questions_path, [raw_question])
                with self.assertRaisesRegex(ValueError, message):
                    load_multihop_questions(self.questions_path, documents)

    def test_evidence_must_reference_a_known_document(self):
        documents = self._load_documents()
        unknown_document = make_document(
            "https://example.test/unknown",
            "Unknown fact",
        )
        raw_question = make_question(
            "Question with unknown document",
            [
                make_evidence(unknown_document, "Unknown fact"),
                make_evidence(self.raw_documents[1], "second fact"),
            ],
        )
        self._write_json(self.questions_path, [raw_question])

        with self.assertRaisesRegex(ValueError, "unknown document"):
            load_multihop_questions(self.questions_path, documents)

    def test_evidence_fact_must_appear_verbatim_in_document(self):
        documents = self._load_documents()
        raw_question = make_question(
            "Question with an absent fact",
            [
                make_evidence(self.raw_documents[0], "not in the document"),
                make_evidence(self.raw_documents[1], "second fact"),
            ],
        )
        self._write_json(self.questions_path, [raw_question])

        with self.assertRaisesRegex(ValueError, "evidence fact not found"):
            load_multihop_questions(self.questions_path, documents)

    def test_evidence_metadata_must_match_corpus_metadata(self):
        documents = self._load_documents()
        mismatched_evidence = make_evidence(self.raw_documents[0], "first fact")
        mismatched_evidence["title"] = "A conflicting title"
        raw_question = make_question(
            "Question with conflicting metadata",
            [
                mismatched_evidence,
                make_evidence(self.raw_documents[1], "second fact"),
            ],
        )
        self._write_json(self.questions_path, [raw_question])

        with self.assertRaisesRegex(ValueError, "metadata mismatch.*title"):
            load_multihop_questions(self.questions_path, documents)

    def test_duplicate_queries_are_rejected(self):
        documents = self._load_documents()
        raw_question = self._valid_non_null_question()
        self._write_json(self.questions_path, [raw_question, raw_question])

        with self.assertRaisesRegex(RuntimeError, "question ID collision"):
            load_multihop_questions(self.questions_path, documents)

    def test_duplicate_evidence_within_question_is_rejected(self):
        documents = self._load_documents()
        evidence = make_evidence(self.raw_documents[0], "first fact")
        raw_question = make_question("Duplicate evidence", [evidence, evidence])
        self._write_json(self.questions_path, [raw_question])

        with self.assertRaisesRegex(RuntimeError, "Evidence ID collision"):
            load_multihop_questions(self.questions_path, documents)

    def test_json_root_must_be_an_array_of_objects(self):
        cases = [
            ({"not": "an array"}, "JSON array"),
            (["not an object"], "JSON objects"),
        ]

        for value, message in cases:
            with self.subTest(value=value):
                self._write_json(self.corpus_path, value)
                with self.assertRaisesRegex(ValueError, message):
                    load_multihop_documents(self.corpus_path)

    def test_malformed_json_is_reported_as_value_error(self):
        self.corpus_path.write_text("[{", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "invalid JSON file"):
            load_multihop_documents(self.corpus_path)

    def test_combined_loader_returns_content_hashes(self):
        raw_questions = [self._valid_non_null_question()]
        self._write_corpus()
        self._write_json(self.questions_path, raw_questions)

        dataset = load_multihop_rag(self.directory)

        self.assertEqual(len(dataset.documents), 2)
        self.assertEqual(len(dataset.questions), 1)
        self.assertEqual(
            dataset.corpus_sha256,
            hashlib.sha256(self.corpus_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            dataset.questions_sha256,
            hashlib.sha256(self.questions_path.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
