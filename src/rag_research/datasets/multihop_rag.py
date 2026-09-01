import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from rag_research.models import (
    InputDocument,
    EvidenceOccurrence,
    EvidenceRecord,
    QuestionRecord,
)

NULL_QUERY_ANSWER = "Insufficient information."

ALLOWED_QUESTION_TYPES = {
    "comparison_query",
    "inference_query",
    "temporal_query",
    "null_query",
}

METADATA_FIELDS = (
    "title",
    "author",
    "source",
    "published_at",
    "category",
    "url",
)

@dataclass(frozen=True)
class MultiHopRAGDataset:
    documents: tuple[InputDocument, ...]
    questions: tuple[QuestionRecord, ...]
    corpus_sha256: str
    questions_sha256: str


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON file: {path}") from error

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    if any(not isinstance(item, dict) for item in data):
        raise ValueError(f"{path} must contain JSON objects")

    return data

def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value

def _make_question_id(query: str) -> str:
    payload = f"multihop-rag-question\0{query}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"mhr-q-{digest[:24]}"

def _make_evidence_id(
    question_id: str,
    document_id: str,
    fact: str,
) -> str:
    payload = f"{question_id}\0{document_id}\0{fact}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"mhr-e-{digest[:24]}"

def _file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

def _find_all_occurrences(text: str, fact: str) -> tuple[EvidenceOccurrence, ...]:
    occurrences = []
    search_start = 0

    while True:
        char_start = text.find(fact, search_start)
        if char_start == -1:
            break
        char_end = char_start + len(fact)
        occurrences.append(
            EvidenceOccurrence(
                char_start=char_start,
                char_end=char_end,
            )
        )
        search_start = char_start + 1

    if not occurrences:
        raise ValueError("evidence fact not found")

    return tuple(occurrences)



def load_multihop_documents(file_path: str | Path) -> tuple[InputDocument, ...]:
    raw_corpus = _load_json_list(Path(file_path))

    documents: list[InputDocument] = []
    seen_document_ids: set[str] = set()

    for idx, doc in enumerate(raw_corpus):
        document_id = _require_non_empty_string(doc.get("url"), f"corpus[{idx}].url")
        text = _require_non_empty_string(doc.get("body"), f"corpus[{idx}].body")
        if document_id in seen_document_ids:
            raise ValueError(f"duplicate corpus URL: {document_id}")
        seen_document_ids.add(document_id)
        metadata = {
            field: doc.get(field)
            for field in METADATA_FIELDS
        }

        for field in (
            "title",
            "source",
            "published_at",
            "category",
        ):
            _require_non_empty_string(
                doc.get(field),
                f"corpus[{idx}].{field}",
            )

        author = doc.get("author")
        if author is not None and not isinstance(author, str):
            raise ValueError(f"corpus[{idx}].author must be a string or null")

        documents.append(
            InputDocument(
                document_id=document_id,
                text=text,
                metadata=metadata,
            )
        )

    return tuple(documents)



def load_multihop_questions(
    file_path: str | Path,
    documents: tuple[InputDocument, ...],
) -> tuple[QuestionRecord, ...]:
    raw_questions = _load_json_list(Path(file_path))

    documents_by_id = {
        document.document_id: document
        for document in documents
    }

    questions: list[QuestionRecord] = []
    seen_question_ids: set[str] = set()
    seen_evidence_ids: set[str] = set()

    for question_idx, question in enumerate(raw_questions):
        query = _require_non_empty_string(
            question.get("query"),
            f"question[{question_idx}].query",
        )
        question_id = _make_question_id(query)
        if question_id in seen_question_ids:
            raise RuntimeError(f"question ID collision: {question_id}")
        seen_question_ids.add(question_id)

        answer = _require_non_empty_string(
            question.get("answer"),
            f"question[{question_idx}].answer",
        )

        question_type = _require_non_empty_string(
            question.get("question_type"),
            f"question[{question_idx}].question_type",
        )
        raw_evidences = question.get("evidence_list")
        if not isinstance(raw_evidences, list):
            raise ValueError(f"questions[{question_idx}].evidence_list must be a list")
        evidences: list[EvidenceRecord] = []

        if question_type not in ALLOWED_QUESTION_TYPES:
            raise ValueError(
                f"{question_id}: unknown question type "
                f"{question_type!r}"
            )

        if question_type == "null_query":
            if raw_evidences:
                raise ValueError(f"{question_id}: null query must not have evidence")
            if answer != NULL_QUERY_ANSWER:
                raise ValueError(
                    f"{question_id}: null query must use "
                    f"{NULL_QUERY_ANSWER!r} as answer"
                )

        else:
            if answer == NULL_QUERY_ANSWER:
                raise ValueError(f"{question_id}: non-null query has null answer")
            if not 2 <= len(raw_evidences) <= 4:
                raise ValueError(
                    f"{question_id}: non-null query must have 2–4 evidence records"
                )


        for evidence_idx, evidence in enumerate(raw_evidences):
            if not isinstance(evidence, dict):
                raise ValueError(
                    f"{question_id}: evidence[{evidence_idx}] must be an object"
                )

            document_id = _require_non_empty_string(
                evidence.get("url"),
                f"questions[{question_idx}].evidence[{evidence_idx}].url",
            )
            fact = _require_non_empty_string(
                evidence.get("fact"),
                f"questions[{question_idx}].evidence[{evidence_idx}].fact",
            )

            document = documents_by_id.get(document_id)
            if document is None:
                raise ValueError(
                    f"question {question_id} references "
                    f"unknown document: {document_id}"
                )
            occurrences = _find_all_occurrences(document.text, fact)

            evidence_id = _make_evidence_id(question_id, document_id, fact)
            if evidence_id in seen_evidence_ids:
                raise RuntimeError(f"Evidence ID collision: {evidence_id}")
            seen_evidence_ids.add(evidence_id)

            metadata = {
                field: evidence.get(field)
                for field in METADATA_FIELDS
            }

            for field in METADATA_FIELDS:
                evidence_value = evidence.get(field)
                corpus_value = document.metadata.get(field)

                if evidence_value != corpus_value:
                    raise ValueError(
                        f"{question_id}: evidence metadata mismatch "
                        f"for {document_id}, field {field}: "
                        f"{evidence_value!r} != {corpus_value!r}"
                    )

            evidences.append(
                EvidenceRecord(
                    evidence_id=evidence_id,
                    document_id=document_id,
                    fact=fact,
                    occurrences=occurrences,
                    metadata=metadata,
                )
            )

        questions.append(
            QuestionRecord(
                question_id=question_id,
                dataset_index=question_idx,
                query=query,
                answer=answer,
                question_type=question_type,
                evidence=tuple(evidences)
            )
        )

    return tuple(questions)


def load_multihop_rag(dataset_directory: str | Path) -> MultiHopRAGDataset:
    directory = Path(dataset_directory)
    corpus_path = directory / "corpus.json"
    questions_path = directory / "MultiHopRAG.json"

    documents = load_multihop_documents(corpus_path)
    questions = load_multihop_questions(questions_path, documents)

    return MultiHopRAGDataset(
        documents=documents,
        questions=questions,
        corpus_sha256=_file_sha256(corpus_path),
        questions_sha256=_file_sha256(questions_path),
    )
