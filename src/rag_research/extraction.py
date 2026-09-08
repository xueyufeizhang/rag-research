import asyncio
import hashlib
import json
import math
import os
import re
import tempfile
import time
import unicodedata
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json_repair

from rag_research.models import ChunkRecord
from rag_research.llm import LLMResponse, normalize_llm_response
from rag_research.prompts import PROMPTS


EXTRACTION_PIPELINE_VERSION = 6
EXTRACTION_FEW_SHOT_POLICY = "production-schema-only-test-fixtures-offline-v1"
EXTRACTION_CACHE_SCHEMA_VERSION = 4
EXTRACTION_STATISTICS_SCHEMA_VERSION = 1
MAX_EXTRACTION_ATTEMPTS = 5
MAX_ENTITY_RECORDS = 20
MAX_TOTAL_RECORDS = 50
ALLOWED_ENTITY_TYPES = (
    "Person",
    "Creature",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Content",
    "Data",
    "Artifact",
    "NaturalObject",
    "Other",
)
_ENTITY_TYPES_BY_CASEFOLD = {
    entity_type.casefold(): entity_type
    for entity_type in ALLOWED_ENTITY_TYPES
}


def _normalize_grounding_text(value: str) -> str:
    # Only orthographic equivalence: no stemming, synonyms, edit distance, or
    # semantic similarity. Keep every word and its order.
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.translate(str.maketrans({
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
        "\u2014": "-", "\u2212": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
    }))
    return re.sub(r"[\s-]+", " ", normalized).strip()


def _collect_example_entity_names() -> dict[str, str]:
    """Collect offline-fixture names for the defensive leakage guard.

    The fixtures are never sent to the model. Keeping their names here lets
    the parser reject a known fixture entity if a provider or prompt assembly
    regression causes it to appear without grounding in the current chunk.
    """
    examples = PROMPTS.get("entity_extraction_test_examples")
    if not isinstance(examples, list) or any(
        not isinstance(example, str)
        for example in examples
    ):
        raise TypeError("entity extraction examples must be a list of strings")

    names: dict[str, str] = {}
    for example in examples:
        for match in re.finditer(r'"name"\s*:\s*"([^"\\]+)"', example):
            name = match.group(1).strip()
            if name:
                names.setdefault(_normalize_grounding_text(name), name)
    return names


_EXAMPLE_ENTITY_NAMES = _collect_example_entity_names()

LLMFunction = Callable[..., Awaitable[str | LLMResponse]]


@dataclass
class Entity:
    name: str
    type: str
    description: str
    source_id: list[str]

@dataclass
class Relation:
    source: str
    target: str
    keywords: list[str]
    description: str
    source_id: list[str]

@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    failed_chunk_ids: list[str]
    statistics: dict[str, Any] = field(default_factory=dict)

@dataclass
class ChunkExtractionResult:
    chunk_id: str
    entities: list[Entity]
    relations: list[Relation]
    error: str | None
    attempts: list[dict[str, Any]] = field(default_factory=list)


class ExtractionLimitError(ValueError):
    """An observable model output exceeds the per-response record limits."""


def _safe_retry_feedback(error: Exception) -> str:
    """Return a category-level correction hint without echoing model text.

    Validation errors can contain model-generated names, descriptions, or
    provider details. Echoing those values into the next prompt would create a
    second leakage channel, especially after a known few-shot entity is
    rejected. Retry feedback therefore contains only stable contract
    categories; it never includes ``str(error)``.
    """
    message = str(error).casefold()
    if isinstance(error, ExtractionLimitError):
        return (
            "The previous response exceeded the configured record limit. "
            "Select fewer records and return a complete JSON object."
        )
    if "prompt example leakage" in message:
        return (
            "The previous response contained an ungrounded entity. "
            "possible prompt example leakage was detected; use only names "
            "explicitly present in the current input text."
        )
    if "unknown entity type" in message:
        return (
            "The previous response used an unknown entity type. Use exactly "
            "one of the allowed entity type labels."
        )
    if "json" in message or "object" in message or "array" in message:
        return (
            "The previous response violated the JSON shape contract. Return "
            "one object with `entities` and `relationships` arrays only."
        )
    if "relationship" in message:
        return (
            "The previous response violated the relationship contract. Check "
            "required fields, distinct endpoints, and response-local entities."
        )
    return (
        "The previous response violated the extraction contract. Check all "
        "required fields, grounding, and output limits, then return corrected "
        "JSON only."
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _limit_exceeded(attempt: dict) -> bool | None:
    entity_count = attempt["raw_entity_count"]
    total_count = attempt["raw_total_count"]
    if (
        entity_count is not None and entity_count > MAX_ENTITY_RECORDS
        or total_count is not None and total_count > MAX_TOTAL_RECORDS
    ):
        return True
    return False if entity_count is not None and total_count is not None else None


def _summarize_attempts(records: list[dict]) -> dict[str, Any]:
    """All rates expose their observable denominator; absent evidence is null."""
    by_chunk = {record["chunk_id"]: record["attempts"] for record in records}
    attempts = [attempt for values in by_chunk.values() for attempt in values]
    by_id = {attempt["attempt_id"]: attempt for attempt in attempts}
    attempted_chunks = sum(bool(values) for values in by_chunk.values())
    retried_chunks = sum(any(a["retry_of"] is not None for a in values) for values in by_chunk.values())
    retries = [a for a in attempts if a["retry_of"] is not None]
    limit_known = [a for a in attempts if _limit_exceeded(a) is not None]
    over_limit = [a for a in attempts if _limit_exceeded(a) is True]
    over_limit_retries = sum(
        _limit_exceeded(by_id[a["retry_of"]]) is True
        for a in retries
        if a["retry_of"] in by_id
    )
    limit_errors = sum(a["outcome"] == "limit_error" for a in attempts)
    limit_error_retries = sum(
        by_id[a["retry_of"]]["outcome"] == "limit_error"
        for a in retries if a["retry_of"] in by_id
    )
    limit_known_chunks = sum(any(_limit_exceeded(a) is not None for a in values) for values in by_chunk.values())
    over_limit_chunks = sum(any(_limit_exceeded(a) is True for a in values) for values in by_chunk.values())
    over_limit_retried_chunks = sum(
        any(a["retry_of"] in by_id and _limit_exceeded(by_id[a["retry_of"]]) is True for a in values)
        for values in by_chunk.values()
    )
    responses = [a for a in attempts if a["response_received"]]
    known_truncation = [a for a in responses if a["truncated"] is not None]
    truncated = sum(a["truncated"] is True for a in known_truncation)
    summary = {
        "attempt_count": len(attempts),
        "attempted_chunk_count": attempted_chunks,
        "response_attempt_count": len(responses),
        "backend_error_attempt_count": sum(a["outcome"] == "backend_error" for a in attempts),
        "unresolved_attempt_count": sum(a["outcome"] == "pending" for a in attempts),
        "successful_attempt_count": sum(a["outcome"] == "success" for a in attempts),
        "retry_count": len(retries),
        "retry_rate_per_attempt": _ratio(len(retries), len(attempts)),
        "retried_chunk_count": retried_chunks,
        "retry_rate_per_attempted_chunk": _ratio(retried_chunks, attempted_chunks),
        "known_limit_attempt_count": len(limit_known),
        "over_limit_attempt_count": len(over_limit),
        "over_limit_rate_per_known_attempt": _ratio(len(over_limit), len(limit_known)),
        "over_limit_error_attempt_count": limit_errors,
        "limit_error_retry_count": limit_error_retries,
        "limit_error_retry_rate_per_error": _ratio(limit_error_retries, limit_errors),
        "over_limit_retry_count": over_limit_retries,
        "over_limit_retry_rate_per_violation": _ratio(over_limit_retries, len(over_limit)),
        "known_limit_chunk_count": limit_known_chunks,
        "over_limit_chunk_count": over_limit_chunks,
        "over_limit_rate_per_known_chunk": _ratio(over_limit_chunks, limit_known_chunks),
        "over_limit_retried_chunk_count": over_limit_retried_chunks,
        "over_limit_retry_rate_per_violating_chunk": _ratio(over_limit_retried_chunks, over_limit_chunks),
        "truncation_known_response_attempt_count": len(known_truncation),
        "truncation_unknown_response_attempt_count": len(responses) - len(known_truncation),
        "truncated_response_attempt_count": truncated,
        "output_truncation_rate_per_known_response": _ratio(truncated, len(known_truncation)),
        "truncation_unknown_rate_per_response": _ratio(len(responses) - len(known_truncation), len(responses)),
    }
    for label, field_name, limit in (
        ("entity_limit", "raw_entity_count", MAX_ENTITY_RECORDS),
        ("total_limit", "raw_total_count", MAX_TOTAL_RECORDS),
        ("accepted_entity_limit", "accepted_entity_count", MAX_ENTITY_RECORDS),
        ("accepted_total_limit", "accepted_total_count", MAX_TOTAL_RECORDS),
    ):
        known = [a for a in attempts if a[field_name] is not None]
        observed_chunks = [values for values in by_chunk.values() if any(a[field_name] is not None for a in values)]
        reached_attempts = sum(a[field_name] >= limit for a in known)
        reached_chunks = sum(any(a[field_name] is not None and a[field_name] >= limit for a in values) for values in observed_chunks)
        summary[label] = {
            "limit": limit,
            "known_count_attempt_count": len(known),
            "unknown_count_attempt_count": len(attempts) - len(known),
            "exactly_at_limit_attempt_count": sum(a[field_name] == limit for a in known),
            "at_or_above_limit_attempt_count": reached_attempts,
            "at_or_above_limit_rate_per_known_attempt": _ratio(reached_attempts, len(known)),
            "known_count_chunk_count": len(observed_chunks),
            "unknown_count_chunk_count": attempted_chunks - len(observed_chunks),
            "at_or_above_limit_chunk_count": reached_chunks,
            "at_or_above_limit_rate_per_known_chunk": _ratio(reached_chunks, len(observed_chunks)),
        }
    return summary


def _validate_attempts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("invalid extraction statistics: attempts must be a list")
    required = {
        "attempt_id", "run_id", "attempt_number", "retry_of", "started_at",
        "duration_seconds", "response_received", "finish_reason", "truncated",
        "usage", "response_sha256", "raw_entity_count", "raw_relation_count",
        "raw_total_count", "raw_count_source", "accepted_entity_count",
        "accepted_relation_count", "accepted_total_count", "outcome", "error",
    }
    previous_by_run: dict[str, dict] = {}
    ids: set[str] = set()
    for attempt in value:
        if not isinstance(attempt, dict) or not required <= set(attempt):
            raise ValueError("invalid extraction statistics: missing attempt fields")
        for key in ("attempt_id", "run_id", "started_at"):
            if not isinstance(attempt[key], str) or not attempt[key]:
                raise ValueError(f"invalid extraction statistics: {key}")
        if attempt["attempt_id"] in ids:
            raise ValueError("invalid extraction statistics: duplicate attempt ID")
        ids.add(attempt["attempt_id"])
        previous = previous_by_run.get(attempt["run_id"])
        expected_number = previous["attempt_number"] + 1 if previous else 1
        expected_retry = previous["attempt_id"] if previous else None
        if type(attempt["attempt_number"]) is not int or attempt["attempt_number"] != expected_number or not 1 <= attempt["attempt_number"] <= MAX_EXTRACTION_ATTEMPTS:
            raise ValueError("invalid extraction statistics: attempt sequence")
        if attempt["retry_of"] != expected_retry or previous and previous["outcome"] == "success":
            raise ValueError("invalid extraction statistics: retry reference")
        previous_by_run[attempt["run_id"]] = attempt
        for key in (
            "raw_entity_count", "raw_relation_count", "raw_total_count",
            "accepted_entity_count", "accepted_relation_count", "accepted_total_count",
        ):
            if attempt[key] is not None and (type(attempt[key]) is not int or attempt[key] < 0):
                raise ValueError(f"invalid extraction statistics: {key}")
        entity_count, relation_count = attempt["raw_entity_count"], attempt["raw_relation_count"]
        total = entity_count + relation_count if entity_count is not None and relation_count is not None else None
        if attempt["raw_total_count"] != total:
            raise ValueError("invalid extraction statistics: raw total count")
        accepted_entity_count = attempt["accepted_entity_count"]
        accepted_relation_count = attempt["accepted_relation_count"]
        accepted_total = (
            accepted_entity_count + accepted_relation_count
            if accepted_entity_count is not None and accepted_relation_count is not None
            else None
        )
        if attempt["accepted_total_count"] != accepted_total:
            raise ValueError("invalid extraction statistics: accepted total count")
        if attempt["raw_count_source"] not in ("strict_json", "unavailable"):
            raise ValueError("invalid extraction statistics: raw count source")
        if attempt["raw_count_source"] == "unavailable" and (entity_count is not None or relation_count is not None):
            raise ValueError("invalid extraction statistics: unavailable raw counts")
        duration = attempt["duration_seconds"]
        if duration is None and attempt["outcome"] != "pending":
            raise ValueError("invalid extraction statistics: missing completed duration")
        if duration is not None and (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
            raise ValueError("invalid extraction statistics: duration")
        if type(attempt["response_received"]) is not bool:
            raise ValueError("invalid extraction statistics: response_received")
        if attempt["truncated"] is not None and type(attempt["truncated"]) is not bool:
            raise ValueError("invalid extraction statistics: truncated")
        for key in ("finish_reason", "response_sha256", "error"):
            if attempt[key] is not None and not isinstance(attempt[key], str):
                raise ValueError(f"invalid extraction statistics: {key}")
        if attempt["usage"] is not None and not isinstance(attempt["usage"], dict):
            raise ValueError("invalid extraction statistics: usage")
        if attempt["outcome"] not in {"pending", "success", "validation_error", "limit_error", "truncated", "backend_error", "cancelled", "internal_error"}:
            raise ValueError("invalid extraction statistics: outcome")
        if attempt["outcome"] == "success" and (
            attempt["truncated"] is True or attempt["error"] is not None
            or not attempt["response_received"] or _limit_exceeded(attempt) is True
            or accepted_entity_count is None or accepted_total is None
            or accepted_entity_count > MAX_ENTITY_RECORDS or accepted_total > MAX_TOTAL_RECORDS
        ):
            raise ValueError("invalid extraction statistics: successful attempt state")
        if attempt["outcome"] != "success" and any(attempt[key] is not None for key in (
            "accepted_entity_count", "accepted_relation_count", "accepted_total_count",
        )):
            raise ValueError("invalid extraction statistics: accepted counts without success")
        if attempt["outcome"] == "pending" and (
            attempt["response_received"] or attempt["error"] is not None or duration is not None
        ):
            raise ValueError("invalid extraction statistics: unresolved attempt state")
        if not attempt["response_received"] and any(attempt[key] is not None for key in (
            "finish_reason", "truncated", "usage", "response_sha256",
            "raw_entity_count", "raw_relation_count", "raw_total_count",
        )):
            raise ValueError("invalid extraction statistics: metadata without a response")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid extraction statistics: not JSON serializable") from error
    return value


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _model_text_sha256(chunk: ChunkRecord) -> str:
    return hashlib.sha256(chunk.model_text.encode("utf-8")).hexdigest()


def _validate_cached_entity(raw: object, chunk_id: str) -> Entity:
    if not isinstance(raw, dict):
        raise ValueError(f"cached entity for {chunk_id} must be an object")

    name = raw.get("name")
    entity_type = raw.get("type")
    description = raw.get("description")
    source_id = raw.get("source_id")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"cached entity for {chunk_id} has an invalid name")
    if entity_type not in ALLOWED_ENTITY_TYPES:
        raise ValueError(f"cached entity for {chunk_id} has an invalid type")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"cached entity for {chunk_id} has an invalid description")
    if source_id != [chunk_id]:
        raise ValueError(f"cached entity for {chunk_id} has invalid source IDs")

    return Entity(
        name=name,
        type=entity_type,
        description=description,
        source_id=list(source_id),
    )


def _validate_cached_relation(raw: object, chunk_id: str) -> Relation:
    if not isinstance(raw, dict):
        raise ValueError(f"cached relation for {chunk_id} must be an object")

    source = raw.get("source")
    target = raw.get("target")
    keywords = raw.get("keywords")
    description = raw.get("description")
    source_id = raw.get("source_id")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"cached relation for {chunk_id} has an invalid source")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"cached relation for {chunk_id} has an invalid target")
    if not isinstance(keywords, list) or not keywords or any(
        not isinstance(keyword, str) or not keyword.strip()
        for keyword in keywords
    ):
        raise ValueError(f"cached relation for {chunk_id} has invalid keywords")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"cached relation for {chunk_id} has an invalid description")
    if source_id != [chunk_id]:
        raise ValueError(f"cached relation for {chunk_id} has invalid source IDs")

    return Relation(
        source=source,
        target=target,
        keywords=list(keywords),
        description=description,
        source_id=list(source_id),
    )


class ExtractionCache:
    def __init__(
        self,
        cache_directory: str | Path,
        extraction_fingerprint: str,
        cache_scope: str,
    ) -> None:
        if (
            not isinstance(extraction_fingerprint, str)
            or not extraction_fingerprint.strip()
        ):
            raise ValueError("extraction fingerprint must be a non-empty string")
        if not isinstance(cache_scope, str) or not cache_scope.strip():
            raise ValueError("extraction cache scope must be a non-empty string")

        self.extraction_fingerprint = extraction_fingerprint
        self.cache_scope = cache_scope
        self.directory = Path(cache_directory) / extraction_fingerprint
        self.records_directory = self.directory / "records"
        self.attempts_directory = self.directory / "attempts"
        self.state_path = self.directory / "states" / f"{cache_scope}.json"

    def _record_path(self, chunk: ChunkRecord) -> Path:
        identity = f"{chunk.chunk_id}\0{_model_text_sha256(chunk)}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.records_directory / f"{digest}.json"

    def report_path(self, run_id: str) -> Path:
        return self.directory / "runs" / self.cache_scope / f"{run_id}.json"

    def load_attempts(self, chunk: ChunkRecord) -> list[dict[str, Any]]:
        path = self.attempts_directory / self._record_path(chunk).name
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != EXTRACTION_STATISTICS_SCHEMA_VERSION:
            raise ValueError(f"unsupported extraction attempt statistics schema: {path}")
        expected = {
            "extraction_fingerprint": self.extraction_fingerprint,
            "chunk_id": chunk.chunk_id,
            "model_text_sha256": _model_text_sha256(chunk),
        }
        if any(payload.get(key) != item for key, item in expected.items()):
            raise ValueError(f"extraction attempt statistics identity mismatch: {path}")
        return _validate_attempts(payload.get("attempts"))

    def save_attempts(self, chunk: ChunkRecord, attempts: list[dict[str, Any]]) -> None:
        _atomic_write_json(
            self.attempts_directory / self._record_path(chunk).name,
            {
                "schema_version": EXTRACTION_STATISTICS_SCHEMA_VERSION,
                "extraction_fingerprint": self.extraction_fingerprint,
                "chunk_id": chunk.chunk_id,
                "model_text_sha256": _model_text_sha256(chunk),
                "attempts": _validate_attempts(attempts),
            },
        )

    def validate_state(self) -> None:
        if not self.state_path.exists():
            return
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != EXTRACTION_CACHE_SCHEMA_VERSION:
            raise ValueError("unsupported extraction state schema")
        if payload.get("extraction_fingerprint") != self.extraction_fingerprint or payload.get("cache_scope") != self.cache_scope:
            raise ValueError("extraction state identity mismatch")
        statistics = payload.get("statistics")
        if not isinstance(statistics, dict) or statistics.get("schema_version") != EXTRACTION_STATISTICS_SCHEMA_VERSION:
            raise ValueError("unsupported extraction state statistics schema")
        if not isinstance(statistics.get("chunks"), list):
            raise ValueError("invalid extraction state statistics chunks")
        run_id = statistics.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("invalid extraction state statistics run ID")
        chunk_ids = set()
        for chunk in statistics["chunks"]:
            if (
                not isinstance(chunk, dict)
                or not isinstance(chunk.get("chunk_id"), str)
                or not chunk["chunk_id"]
                or chunk["chunk_id"] in chunk_ids
                or not isinstance(chunk.get("model_text_sha256"), str)
                or type(chunk.get("cache_hit")) is not bool
                or chunk.get("status") not in {"pending", "running", "success", "failed", "interrupted"}
            ):
                raise ValueError("invalid extraction state statistics chunk")
            chunk_ids.add(chunk["chunk_id"])
            _validate_attempts(chunk.get("attempts"))
        for key, predicate in (
            ("run_summary", lambda a: a["run_id"] == run_id),
            ("historical_summary", lambda a: a["run_id"] != run_id),
            ("cumulative_summary", lambda a: True),
        ):
            records = [
                dict(chunk, attempts=[a for a in chunk["attempts"] if predicate(a)])
                for chunk in statistics["chunks"]
            ]
            if statistics.get(key) != _summarize_attempts(records):
                raise ValueError(f"invalid extraction state statistics {key}")

    def load(self, chunk: ChunkRecord) -> ChunkExtractionResult | None:
        path = self._record_path(chunk)
        if not path.exists():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid extraction cache record: {path}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError(f"invalid extraction cache record: {path}")
        if payload.get("schema_version") != EXTRACTION_CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported extraction cache schema: {path}")
        if (
            payload.get("extraction_fingerprint")
            != self.extraction_fingerprint
        ):
            raise ValueError(f"extraction cache fingerprint mismatch: {path}")
        if payload.get("chunk_id") != chunk.chunk_id:
            raise ValueError(f"extraction cache chunk ID mismatch: {path}")
        if payload.get("model_text_sha256") != _model_text_sha256(chunk):
            raise ValueError(f"extraction cache text mismatch: {path}")
        if payload.get("statistics_schema_version") != EXTRACTION_STATISTICS_SCHEMA_VERSION:
            raise ValueError(f"unsupported extraction cache statistics schema: {path}")
        attempts = _validate_attempts(payload.get("attempts"))
        if not attempts or attempts[-1]["outcome"] != "success":
            raise ValueError(f"extraction cache lacks a successful attempt: {path}")

        raw_entities = payload.get("entities")
        raw_relations = payload.get("relations")
        if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
            raise ValueError(f"invalid extraction cache payload: {path}")

        entities = [
            _validate_cached_entity(raw, chunk.chunk_id)
            for raw in raw_entities
        ]
        relations = [
            _validate_cached_relation(raw, chunk.chunk_id)
            for raw in raw_relations
        ]
        _validate_extraction_contract(
            entities,
            relations,
            source_id=chunk.chunk_id,
            input_text=chunk.model_text,
        )
        if (
            attempts[-1]["accepted_entity_count"] != len(entities)
            or attempts[-1]["accepted_relation_count"] != len(relations)
            or attempts[-1]["accepted_total_count"] != len(entities) + len(relations)
        ):
            raise ValueError(f"extraction cache statistics disagree with accepted records: {path}")

        return ChunkExtractionResult(
            chunk_id=chunk.chunk_id,
            entities=entities,
            relations=relations,
            error=None,
            attempts=attempts,
        )

    def save(
        self,
        chunk: ChunkRecord,
        result: ChunkExtractionResult,
    ) -> None:
        if result.chunk_id != chunk.chunk_id:
            raise ValueError("cannot cache extraction under a different chunk ID")
        if result.error is not None:
            raise ValueError("failed extraction results must not be cached")
        attempts = _validate_attempts(result.attempts)
        if not attempts or attempts[-1]["outcome"] != "success":
            raise ValueError("cannot cache extraction without a successful attempt")
        if (
            attempts[-1]["accepted_entity_count"] != len(result.entities)
            or attempts[-1]["accepted_relation_count"] != len(result.relations)
            or attempts[-1]["accepted_total_count"] != len(result.entities) + len(result.relations)
        ):
            raise ValueError("extraction statistics disagree with accepted records")

        _validate_extraction_contract(
            result.entities,
            result.relations,
            source_id=chunk.chunk_id,
            input_text=chunk.model_text,
        )

        _atomic_write_json(
            self._record_path(chunk),
            {
                "schema_version": EXTRACTION_CACHE_SCHEMA_VERSION,
                "extraction_fingerprint": self.extraction_fingerprint,
                "chunk_id": chunk.chunk_id,
                "model_text_sha256": _model_text_sha256(chunk),
                "entities": [asdict(entity) for entity in result.entities],
                "relations": [asdict(relation) for relation in result.relations],
                "statistics_schema_version": EXTRACTION_STATISTICS_SCHEMA_VERSION,
                "attempts": attempts,
            },
        )

    def save_state(
        self,
        *,
        chunk_count: int,
        completed_chunk_count: int,
        failed_chunk_ids: list[str],
        statistics: dict[str, Any],
        status: str | None = None,
    ) -> None:
        _atomic_write_json(
            self.state_path,
            {
                "schema_version": EXTRACTION_CACHE_SCHEMA_VERSION,
                "extraction_fingerprint": self.extraction_fingerprint,
                "cache_scope": self.cache_scope,
                "status": status or ("incomplete" if failed_chunk_ids else "complete"),
                "chunk_count": chunk_count,
                "completed_chunk_count": completed_chunk_count,
                "failed_chunk_ids": failed_chunk_ids,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "statistics": statistics,
            },
        )


def _required_text(
    value: object,
    *,
    field_name: str,
    source_id: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"chunk {source_id}: {field_name} must be a non-empty string"
        )
    return value.strip()


def _normalize_entity_type(value: object, source_id: str) -> str:
    raw_type = _required_text(
        value,
        field_name="entity type",
        source_id=source_id,
    )
    entity_type = _ENTITY_TYPES_BY_CASEFOLD.get(raw_type.casefold())
    if entity_type is None:
        raise ValueError(
            f"chunk {source_id}: unknown entity type {raw_type!r}; "
            f"expected one of {ALLOWED_ENTITY_TYPES}"
        )
    return entity_type


def _normalize_keywords(value: object, source_id: str) -> list[str]:
    if isinstance(value, list):
        if any(not isinstance(keyword, str) for keyword in value):
            raise ValueError(
                f"chunk {source_id}: relationship keywords must contain strings"
            )
        keywords = [keyword.strip() for keyword in value if keyword.strip()]
    elif isinstance(value, str):
        keywords = [keyword.strip() for keyword in value.split(",") if keyword.strip()]
    else:
        raise ValueError(
            f"chunk {source_id}: relationship keywords must be a string or list"
        )

    deduplicated: list[str] = []
    _merge_keywords(deduplicated, keywords)
    if not deduplicated:
        raise ValueError(
            f"chunk {source_id}: relationship keywords must not be empty"
        )
    return deduplicated


def _merge_text(existing: str, new: str) -> str:
    if not new or new.casefold() == existing.casefold():
        return existing
    if not existing:
        return new
    return f"{existing} | {new}"


def _merge_keywords(existing: list[str], new: list[str]) -> None:
    seen = {keyword.casefold() for keyword in existing}
    for keyword in new:
        key = keyword.casefold()
        if key not in seen:
            existing.append(keyword)
            seen.add(key)


def _contains_entity_name(input_text: str, entity_name: str) -> bool:
    escaped_name = re.escape(_normalize_grounding_text(entity_name))
    pattern = rf"(?<!\w){escaped_name}(?!\w)"
    return re.search(pattern, _normalize_grounding_text(input_text)) is not None


def _validate_extraction_contract(
    entities: list[Entity],
    relations: list[Relation],
    *,
    source_id: str,
    input_text: str,
) -> None:
    if len(entities) > MAX_ENTITY_RECORDS:
        raise ExtractionLimitError(
            f"chunk {source_id}: entity count {len(entities)} exceeds maximum "
            f"{MAX_ENTITY_RECORDS}"
        )
    total_records = len(entities) + len(relations)
    if total_records > MAX_TOTAL_RECORDS:
        raise ExtractionLimitError(
            f"chunk {source_id}: total record count {total_records} exceeds "
            f"maximum {MAX_TOTAL_RECORDS}"
        )

    entity_names: dict[str, str] = {}
    for entity in entities:
        if not entity.name.strip():
            raise ValueError(f"chunk {source_id}: entity name must not be empty")
        if entity.type not in ALLOWED_ENTITY_TYPES:
            raise ValueError(
                f"chunk {source_id}: unknown entity type {entity.type!r}"
            )
        if not entity.description.strip():
            raise ValueError(
                f"chunk {source_id}: entity description must not be empty"
            )
        if entity.source_id != [source_id]:
            raise ValueError(
                f"chunk {source_id}: entity has invalid source IDs"
            )
        key = entity.name.casefold()
        if key in entity_names:
            raise ValueError(
                f"chunk {source_id}: duplicate entity name {entity.name!r}"
            )
        entity_names[key] = entity.name

        example_name = _EXAMPLE_ENTITY_NAMES.get(_normalize_grounding_text(entity.name))
        if example_name is not None and not _contains_entity_name(
            input_text,
            entity.name,
        ):
            raise ValueError(
                f"chunk {source_id}: possible prompt example leakage: entity "
                f"{example_name!r} is absent from the input text"
            )

    relation_pairs: set[tuple[str, str]] = set()
    for relation in relations:
        if not relation.source.strip() or not relation.target.strip():
            raise ValueError(
                f"chunk {source_id}: relationship endpoints must not be empty"
            )
        if not relation.keywords or any(
            not keyword.strip()
            for keyword in relation.keywords
        ):
            raise ValueError(
                f"chunk {source_id}: relationship keywords must not be empty"
            )
        if not relation.description.strip():
            raise ValueError(
                f"chunk {source_id}: relationship description must not be empty"
            )
        if relation.source_id != [source_id]:
            raise ValueError(
                f"chunk {source_id}: relationship has invalid source IDs"
            )
        source_key = relation.source.casefold()
        target_key = relation.target.casefold()
        if source_key == target_key:
            raise ValueError(
                f"chunk {source_id}: self-relationships are not allowed: "
                f"{relation.source!r}"
            )
        if source_key not in entity_names or target_key not in entity_names:
            raise ValueError(
                f"chunk {source_id}: relationship endpoints must appear in "
                f"the same response's entities list: {relation.source!r} -> "
                f"{relation.target!r}"
            )
        pair = tuple(sorted((source_key, target_key)))
        if pair in relation_pairs:
            raise ValueError(
                f"chunk {source_id}: duplicate relationship pair "
                f"{relation.source!r} -- {relation.target!r}"
            )
        relation_pairs.add(pair)


def _decode_response(response: str, source_id: str) -> dict:
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"chunk {source_id}: empty response")

    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        raise ValueError(f"chunk {source_id}: no JSON object found")
    json_str = match.group()

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        data = json_repair.loads(json_str)
    if not isinstance(data, dict):
        raise ValueError(f"chunk {source_id}: parsed JSON is not an object")
    return data


def _raw_response_counts(response: str) -> tuple[int | None, int | None, str]:
    # JSON repair may synthesize or complete arrays. Those inferred lengths
    # are unsuitable as observations of the provider's raw output.
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        data = json.loads(match.group()) if match is not None else None
    except json.JSONDecodeError:
        return None, None, "unavailable"
    if not isinstance(data, dict):
        return None, None, "unavailable"
    entities, relations = data.get("entities"), data.get("relationships")
    return (
        len(entities) if isinstance(entities, list) else None,
        len(relations) if isinstance(relations, list) else None,
        "strict_json",
    )


def _parse_response(
    response: str,
    source_id: str,
    input_text: str,
) -> tuple[list[Entity], list[Relation]]:
    data = _decode_response(response, source_id)

    raw_entities = data.get("entities")
    raw_relations = data.get("relationships")
    if not isinstance(raw_entities, list):
        raise ValueError(f"chunk {source_id}: entities must be a list")
    if not isinstance(raw_relations, list):
        raise ValueError(f"chunk {source_id}: relationships must be a list")
    if len(raw_entities) > MAX_ENTITY_RECORDS:
        raise ExtractionLimitError(
            f"chunk {source_id}: entity count {len(raw_entities)} exceeds "
            f"maximum {MAX_ENTITY_RECORDS}"
        )
    raw_total_records = len(raw_entities) + len(raw_relations)
    if raw_total_records > MAX_TOTAL_RECORDS:
        raise ExtractionLimitError(
            f"chunk {source_id}: total record count {raw_total_records} "
            f"exceeds maximum {MAX_TOTAL_RECORDS}"
        )

    entities_by_name: dict[str, Entity] = {}

    for entity_index, e in enumerate(raw_entities):
        if not isinstance(e, dict):
            raise ValueError(
                f"chunk {source_id}: entity[{entity_index}] must be an object"
            )

        clean_name = _required_text(
            e.get("name"),
            field_name=f"entity[{entity_index}].name",
            source_id=source_id,
        )
        key = clean_name.casefold()
        entity_type = _normalize_entity_type(e.get("type"), source_id)
        description = _required_text(
            e.get("description"),
            field_name=f"entity[{entity_index}].description",
            source_id=source_id,
        )
        existing = entities_by_name.get(key)
        if existing is not None:
            if existing.type != entity_type:
                raise ValueError(
                    f"chunk {source_id}: duplicate entity {clean_name!r} "
                    f"has conflicting types {existing.type!r} and "
                    f"{entity_type!r}"
                )
            existing.description = _merge_text(
                existing.description,
                description,
            )
            continue

        entities_by_name[key] = Entity(
            name=clean_name,
            type=entity_type,
            description=description,
            source_id=[source_id],
        )

    relations_by_pair: dict[tuple[str, str], Relation] = {}
    for relation_index, r in enumerate(raw_relations):
        if not isinstance(r, dict):
            raise ValueError(
                f"chunk {source_id}: relationship[{relation_index}] must be an object"
            )

        clean_source = _required_text(
            r.get("source"),
            field_name=f"relationship[{relation_index}].source",
            source_id=source_id,
        )
        clean_target = _required_text(
            r.get("target"),
            field_name=f"relationship[{relation_index}].target",
            source_id=source_id,
        )
        source_key = clean_source.casefold()
        target_key = clean_target.casefold()
        if source_key == target_key:
            raise ValueError(
                f"chunk {source_id}: self-relationships are not allowed: "
                f"{clean_source!r}"
            )
        source_entity = entities_by_name.get(source_key)
        target_entity = entities_by_name.get(target_key)
        if source_entity is None or target_entity is None:
            raise ValueError(
                f"chunk {source_id}: relationship endpoints must appear in "
                f"the same response's entities list: {clean_source!r} -> "
                f"{clean_target!r}"
            )

        deduplicated_keywords = _normalize_keywords(
            r.get("keywords"),
            source_id,
        )
        description = _required_text(
            r.get("description"),
            field_name=f"relationship[{relation_index}].description",
            source_id=source_id,
        )

        pair = tuple(sorted((source_key, target_key)))
        existing = relations_by_pair.get(pair)
        if existing is not None:
            _merge_keywords(existing.keywords, deduplicated_keywords)
            existing.description = _merge_text(
                existing.description,
                description,
            )
            continue

        relations_by_pair[pair] = Relation(
            source=source_entity.name,
            target=target_entity.name,
            keywords=deduplicated_keywords,
            description=description,
            source_id=[source_id],
        )

    entities = list(entities_by_name.values())
    relations = list(relations_by_pair.values())
    _validate_extraction_contract(
        entities,
        relations,
        source_id=source_id,
        input_text=input_text,
    )
    return entities, relations


async def extract(
    chunks: list[ChunkRecord],
    llm_func: LLMFunction,
    con_num: int,
    *,
    cache_directory: str | Path | None = None,
    extraction_fingerprint: str | None = None,
    cache_scope: str | None = None,
) -> ExtractionResult:
    if con_num <= 0:
        raise ValueError("extraction concurrency must be positive")
    if not isinstance(chunks, list) or any(
        not isinstance(chunk, ChunkRecord)
        for chunk in chunks
    ):
        raise TypeError("chunks must be a list of ChunkRecord")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk IDs must be unique")
    cache_arguments = (
        cache_directory,
        extraction_fingerprint,
        cache_scope,
    )
    if any(value is None for value in cache_arguments) and not all(
        value is None
        for value in cache_arguments
    ):
        raise ValueError(
            "cache_directory, extraction_fingerprint, and cache_scope "
            "must be provided together"
        )
    if not callable(llm_func):
        raise TypeError("llm_func must be callable")

    chunks_num = len(chunks)
    sem = asyncio.Semaphore(con_num)
    done_count = 0
    start = time.monotonic()
    run_id = uuid.uuid4().hex
    cache = (
        ExtractionCache(cache_directory, extraction_fingerprint, cache_scope)
        if cache_directory is not None and extraction_fingerprint is not None
        and cache_scope is not None else None
    )
    if cache is not None:
        cache.validate_state()

    results_by_chunk_id: dict[str, ChunkExtractionResult] = {}
    statistics_by_chunk: dict[str, dict] = {}
    pending: list[tuple[int, ChunkRecord]] = []
    for idx, chunk in enumerate(chunks, start=1):
        cached_result = cache.load(chunk) if cache is not None else None
        history = cache.load_attempts(chunk) if cache is not None else []
        if cached_result is not None:
            if history and history != cached_result.attempts:
                raise ValueError("extraction cache and attempt history disagree")
            history = cached_result.attempts
            results_by_chunk_id[chunk.chunk_id] = cached_result
        else:
            pending.append((idx, chunk))
        statistics_by_chunk[chunk.chunk_id] = {
            "chunk_id": chunk.chunk_id,
            "model_text_sha256": _model_text_sha256(chunk),
            "cache_hit": cached_result is not None,
            "status": "success" if cached_result is not None else "pending",
            "attempts": history,
        }

    cached_count = len(results_by_chunk_id)
    work_count = len(pending)
    if cached_count:
        print(f"[extract] loaded {cached_count}/{chunks_num} chunks from cache; "
              f"processing {work_count}", flush=True)

    def make_statistics(status: str) -> dict[str, Any]:
        records = list(statistics_by_chunk.values())
        current = [dict(record, attempts=[a for a in record["attempts"] if a["run_id"] == run_id]) for record in records]
        historical = [dict(record, attempts=[a for a in record["attempts"] if a["run_id"] != run_id]) for record in records]
        return {
            "schema_version": EXTRACTION_STATISTICS_SCHEMA_VERSION,
            "run_id": run_id,
            "status": status,
            "limits": {
                "entity_records": MAX_ENTITY_RECORDS,
                "total_records": MAX_TOTAL_RECORDS,
                "max_attempts": MAX_EXTRACTION_ATTEMPTS,
            },
            "metric_definitions": {
                "record_counts": "Strict-JSON array lengths before deduplication; repaired or unavailable raw counts remain null.",
                "accepted_record_counts": "Validated successful output lengths after deduplication; unsuccessful attempts have null accepted counts.",
                "limit_reached": "At or above the per-response cap in any observable attempt; exact cap counts are reported separately.",
                "chunk_limit_rates": "Chunks with any qualifying attempt divided by chunks with at least one observable count.",
                "retry": "A subsequent call initiated within the same run and chunk; pending calls remain explicitly unresolved and resumed first calls are separate from retries.",
                "output_truncation": "Provider-explicit truncated responses divided by responses with known truncation status; unknowns and no-response calls are separate.",
            },
            "chunk_count": chunks_num,
            "cached_chunk_count": cached_count,
            "completed_chunk_count": sum(record["status"] == "success" for record in records),
            "failed_chunk_count": sum(record["status"] == "failed" for record in records),
            "run_summary": _summarize_attempts(current),
            "historical_summary": _summarize_attempts(historical),
            "cumulative_summary": _summarize_attempts(records),
            "chunks": records,
            "report_path": str(cache.report_path(run_id).resolve()) if cache is not None else None,
        }

    def persist_report(status: str) -> dict[str, Any]:
        statistics = make_statistics(status)
        if cache is not None:
            _atomic_write_json(cache.report_path(run_id), statistics)
            cache.save_state(
                chunk_count=chunks_num,
                completed_chunk_count=statistics["completed_chunk_count"],
                failed_chunk_ids=[key for key, record in statistics_by_chunk.items() if record["status"] == "failed"],
                statistics=statistics,
                status=status,
            )
        return statistics

    persist_report("running")
    system_prompt = PROMPTS["entity_extraction_system_prompt"].format(
        entity_types_guidance=PROMPTS["default_entity_types_guidance"],
        max_total_records=MAX_TOTAL_RECORDS,
        max_entity_records=MAX_ENTITY_RECORDS,
    )
    # Retries receive a separate contract reminder but still no semantic
    # examples. This prevents a failed response from being repeatedly exposed
    # to realistic names and facts that can be copied into the next attempt.
    retry_system_prompt = (
        system_prompt
        + "\n---Retry Safety---\n"
        + "This is a correction attempt. Re-read only the current input text; "
        + "do not reproduce any content from the instructions."
    )

    async def process_one(idx: int, chunk: ChunkRecord) -> ChunkExtractionResult:
        nonlocal done_count
        last_err: Exception | None = None
        record = statistics_by_chunk[chunk.chunk_id]
        history = record["attempts"]
        base_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            entity_types_guidance=PROMPTS["default_entity_types_guidance"],
            input_text=chunk.model_text,
            max_total_records=MAX_TOTAL_RECORDS,
            max_entity_records=MAX_ENTITY_RECORDS,
        )
        retry_prompt = base_prompt
        previous_attempt_id = None
        for attempt_number in range(1, MAX_EXTRACTION_ATTEMPTS + 1):
            attempt: dict[str, Any] | None = None
            stage = "backend"
            try:
                async with sem:
                    # A retry exists only once its next model call begins.
                    # Cancellation during backoff therefore creates no retry.
                    t0 = time.monotonic()
                    attempt = {
                        "attempt_id": uuid.uuid4().hex,
                        "run_id": run_id,
                        "attempt_number": attempt_number,
                        "retry_of": previous_attempt_id,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": None,
                        "response_received": False,
                        "finish_reason": None,
                        "truncated": None,
                        "usage": None,
                        "response_sha256": None,
                        "raw_entity_count": None,
                        "raw_relation_count": None,
                        "raw_total_count": None,
                        "raw_count_source": "unavailable",
                        "accepted_entity_count": None,
                        "accepted_relation_count": None,
                        "accepted_total_count": None,
                        "outcome": "pending",
                        "error": None,
                    }
                    record["status"] = "running"
                    history.append(attempt)
                    if cache is not None:
                        try:
                            cache.save_attempts(chunk, history)
                        except BaseException:
                            # Persistence failed before the model callable
                            # began; this is not a backend call or a retry.
                            history.pop()
                            attempt = None
                            raise
                    raw_response = await llm_func(
                        system=system_prompt if attempt_number == 1 else retry_system_prompt,
                        prompt=retry_prompt,
                    )
                    attempt["response_received"] = True
                stage = "normalize"
                response = normalize_llm_response(raw_response)
                attempt.update({
                    "finish_reason": response.finish_reason,
                    "truncated": response.truncated,
                    "usage": response.usage,
                    "response_sha256": hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
                })
                stage = "parse"
                entity_count, relation_count, count_source = _raw_response_counts(response.text)
                attempt.update({
                    "raw_entity_count": entity_count,
                    "raw_relation_count": relation_count,
                    "raw_total_count": entity_count + relation_count if entity_count is not None and relation_count is not None else None,
                    "raw_count_source": count_source,
                })
                if response.truncated is True:
                    attempt["outcome"] = "truncated"
                    raise ValueError("provider explicitly reported truncated output; return a complete JSON object within the output budget")
                entities, relations = _parse_response(response.text, chunk.chunk_id, chunk.model_text)
                attempt["outcome"] = "success"
                attempt.update({
                    "accepted_entity_count": len(entities),
                    "accepted_relation_count": len(relations),
                    "accepted_total_count": len(entities) + len(relations),
                })
            except asyncio.CancelledError:
                record["status"] = "interrupted"
                if attempt is not None:
                    attempt["outcome"] = "cancelled"
                    attempt["error"] = "CancelledError: extraction call cancelled"
                raise
            except Exception as error:
                last_err = error
                if attempt is None:
                    raise
                attempt["error"] = f"{type(error).__name__}: {error}"
                if stage == "backend":
                    attempt["outcome"] = "backend_error"
                elif isinstance(error, ExtractionLimitError):
                    attempt["outcome"] = "limit_error"
                elif isinstance(error, ValueError) or stage == "normalize" and isinstance(error, TypeError):
                    if attempt["outcome"] != "truncated":
                        attempt["outcome"] = "validation_error"
                    retry_prompt = (
                        base_prompt + "\n\n---Correction Required---\n"
                        + _safe_retry_feedback(error)
                        + " Return a corrected JSON object only."
                    )
                else:
                    attempt["outcome"] = "internal_error"
                    record["status"] = "failed"
                    raise
                if isinstance(error, ExtractionLimitError):
                    retry_prompt = (
                        base_prompt + "\n\n---Correction Required---\n"
                        + _safe_retry_feedback(error)
                    )
            finally:
                if attempt is not None:
                    if attempt["outcome"] != "pending":
                        attempt["duration_seconds"] = time.monotonic() - t0
                    previous_attempt_id = attempt["attempt_id"]
                    if cache is not None:
                        cache.save_attempts(chunk, history)

            if attempt["outcome"] == "success":
                result = ChunkExtractionResult(chunk.chunk_id, entities, relations, None, history)
                if cache is not None:
                    cache.save(chunk, result)
                record["status"] = "success"
                done_count += 1
                print(f"[extract] {done_count}/{work_count} (chunk {idx}) "
                      f"+{len(entities)}ent +{len(relations)}rel", flush=True)
                return result
            if attempt_number < MAX_EXTRACTION_ATTEMPTS:
                delay = 2 ** (attempt_number - 1) * 5
                print(f"[extract] chunk {idx} failed ({last_err}); "
                      f"retry after {delay}s, attempt {attempt_number}/{MAX_EXTRACTION_ATTEMPTS}", flush=True)
                await asyncio.sleep(delay)

        record["status"] = "failed"
        done_count += 1
        return ChunkExtractionResult(
            chunk_id=chunk.chunk_id,
            entities=[], relations=[],
            error=f"{type(last_err).__name__}: {last_err}",
            attempts=history,
        )

    tasks = [asyncio.create_task(process_one(idx, chunk)) for idx, chunk in pending]
    try:
        pending_results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        persist_report("interrupted")
        raise
    results_by_chunk_id.update({result.chunk_id: result for result in pending_results})

    all_entities: list[Entity] = []
    all_relations: list[Relation] = []
    failed_chunk_ids: list[str] = []
    for chunk in chunks:
        result = results_by_chunk_id[chunk.chunk_id]
        if result.error is not None:
            failed_chunk_ids.append(chunk.chunk_id)
        else:
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)

    statistics = persist_report("incomplete" if failed_chunk_ids else "complete")
    print(f"[extract] All done: {len(all_entities)} entities, {len(all_relations)} relations, "
          f"{len(failed_chunk_ids)} failed chunks, {time.monotonic()-start:.0f}s", flush=True)
    return ExtractionResult(
        entities=all_entities,
        relations=all_relations,
        failed_chunk_ids=failed_chunk_ids,
        statistics=statistics,
    )
