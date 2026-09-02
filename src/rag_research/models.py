from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InputDocument:
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    text: str
    model_text: str
    chunk_index: int
    char_start: int | None
    char_end: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    document_count: int
    chunk_count: int
    entity_count: int
    relation_count: int
    failed_chunk_ids: list[str]
    build_fingerprint: str
    chunking_fingerprint: str
    extraction_fingerprint: str
    build_provenance: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EvidenceOccurrence:
    char_start: int
    char_end: int


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    document_id: str
    fact: str
    occurrences: tuple[EvidenceOccurrence, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuestionRecord:
    question_id: str
    dataset_index: int
    query: str
    answer: str
    question_type: str
    evidence: tuple[EvidenceRecord, ...]
