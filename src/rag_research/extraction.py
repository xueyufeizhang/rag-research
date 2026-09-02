import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json_repair

from rag_research.models import ChunkRecord
from rag_research.prompts import PROMPTS


EXTRACTION_PIPELINE_VERSION = 4
EXTRACTION_CACHE_SCHEMA_VERSION = 3
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


def _collect_example_entity_names() -> dict[str, str]:
    examples = PROMPTS.get("entity_extraction_examples")
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
                names.setdefault(name.casefold(), name)
    return names


_EXAMPLE_ENTITY_NAMES = _collect_example_entity_names()

LLMFunction = Callable[..., Awaitable[str]]


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

@dataclass
class ChunkExtractionResult:
    chunk_id: str
    entities: list[Entity]
    relations: list[Relation]
    error: str | None


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
        self.state_path = self.directory / "states" / f"{cache_scope}.json"

    def _record_path(self, chunk: ChunkRecord) -> Path:
        identity = f"{chunk.chunk_id}\0{_model_text_sha256(chunk)}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.records_directory / f"{digest}.json"

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

        return ChunkExtractionResult(
            chunk_id=chunk.chunk_id,
            entities=entities,
            relations=relations,
            error=None,
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
            },
        )

    def save_state(
        self,
        *,
        chunk_count: int,
        completed_chunk_count: int,
        failed_chunk_ids: list[str],
    ) -> None:
        _atomic_write_json(
            self.state_path,
            {
                "schema_version": EXTRACTION_CACHE_SCHEMA_VERSION,
                "extraction_fingerprint": self.extraction_fingerprint,
                "cache_scope": self.cache_scope,
                "status": "incomplete" if failed_chunk_ids else "complete",
                "chunk_count": chunk_count,
                "completed_chunk_count": completed_chunk_count,
                "failed_chunk_ids": failed_chunk_ids,
                "updated_at": datetime.now(timezone.utc).isoformat(),
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
    escaped_name = re.escape(entity_name).replace(r"\ ", r"\s+")
    pattern = rf"(?<!\w){escaped_name}(?!\w)"
    return re.search(pattern, input_text, re.IGNORECASE) is not None


def _validate_extraction_contract(
    entities: list[Entity],
    relations: list[Relation],
    *,
    source_id: str,
    input_text: str,
) -> None:
    if len(entities) > MAX_ENTITY_RECORDS:
        raise ValueError(
            f"chunk {source_id}: entity count {len(entities)} exceeds maximum "
            f"{MAX_ENTITY_RECORDS}"
        )
    total_records = len(entities) + len(relations)
    if total_records > MAX_TOTAL_RECORDS:
        raise ValueError(
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

        example_name = _EXAMPLE_ENTITY_NAMES.get(key)
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


def _parse_response(
    response: str,
    source_id: str,
    input_text: str,
) -> tuple[list[Entity], list[Relation]]:
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

    raw_entities = data.get("entities")
    raw_relations = data.get("relationships")
    if not isinstance(raw_entities, list):
        raise ValueError(f"chunk {source_id}: entities must be a list")
    if not isinstance(raw_relations, list):
        raise ValueError(f"chunk {source_id}: relationships must be a list")
    if len(raw_entities) > MAX_ENTITY_RECORDS:
        raise ValueError(
            f"chunk {source_id}: entity count {len(raw_entities)} exceeds "
            f"maximum {MAX_ENTITY_RECORDS}"
        )
    raw_total_records = len(raw_entities) + len(raw_relations)
    if raw_total_records > MAX_TOTAL_RECORDS:
        raise ValueError(
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
    start = time.time()
    cache = (
        ExtractionCache(
            cache_directory,
            extraction_fingerprint,
            cache_scope,
        )
        if cache_directory is not None
        and extraction_fingerprint is not None
        and cache_scope is not None
        else None
    )

    results_by_chunk_id: dict[str, ChunkExtractionResult] = {}
    pending: list[tuple[int, ChunkRecord]] = []
    for idx, chunk in enumerate(chunks, start=1):
        cached_result = cache.load(chunk) if cache is not None else None
        if cached_result is None:
            pending.append((idx, chunk))
        else:
            results_by_chunk_id[chunk.chunk_id] = cached_result

    cached_count = len(results_by_chunk_id)
    if cached_count:
        print(
            f"[extract] loaded {cached_count}/{chunks_num} chunks from cache; "
            f"processing {len(pending)}",
            flush=True,
        )
    work_count = len(pending)

    async def process_one(
        idx: int,
        chunk: ChunkRecord,
    ) -> ChunkExtractionResult:
        nonlocal done_count
        last_err: Exception | None = None
        base_prompt = PROMPTS["entity_extraction_user_prompt"].format(
            entity_types_guidance=PROMPTS["default_entity_types_guidance"],
            input_text=chunk.model_text,
            max_total_records=MAX_TOTAL_RECORDS,
            max_entity_records=MAX_ENTITY_RECORDS,
        )
        retry_prompt = base_prompt
        for attempt in range(5):
            t0 = time.time()
            try:
                async with sem:
                    response = await llm_func(
                        system=PROMPTS["entity_extraction_system_prompt"].format(
                            entity_types_guidance=PROMPTS["default_entity_types_guidance"],
                            examples="\n\n".join(
                                PROMPTS["entity_extraction_examples"]
                            ),
                            max_total_records=MAX_TOTAL_RECORDS,
                            max_entity_records=MAX_ENTITY_RECORDS,
                        ),
                        prompt=retry_prompt,
                    )
            except Exception as e:
                last_err = e
            else:
                try:
                    entities, relations = _parse_response(
                        response,
                        chunk.chunk_id,
                        chunk.model_text,
                    )
                except ValueError as e:
                    last_err = e
                    retry_prompt = (
                        base_prompt
                        + "\n\n---Correction Required---\n"
                        + "Your previous response violated the output contract: "
                        + str(e)
                        + ". Return a corrected JSON object only."
                    )
                else:
                    result = ChunkExtractionResult(
                        chunk_id=chunk.chunk_id,
                        entities=entities,
                        relations=relations,
                        error=None,
                    )
                    if cache is not None:
                        cache.save(chunk, result)

                    done_count += 1
                    elapsed = time.time() - start
                    eta = elapsed / done_count * (work_count - done_count)
                    print(f"[extract] {done_count}/{work_count} (chunk {idx}) "
                          f"+{len(entities)}ent +{len(relations)}rel "
                          f"chunk {time.time()-t0:.1f}s elapsed {elapsed:.0f}s eta {eta:.0f}s",
                          flush=True)
                    return result

            if attempt < 4:
                wait = 2 ** attempt * 5
                print(f"[extract] chunk {idx} failed "
                      f"({type(last_err).__name__}: {last_err}) "
                      f"Retry after {wait}s {attempt+1}/5", flush=True)
                await asyncio.sleep(wait)
            else:
                print(f"[extract] chunk {idx} failed "
                      f"({type(last_err).__name__}: {last_err}) "
                      f"No retries left", flush=True)

        done_count += 1
        elapsed = time.time() - start
        eta = elapsed / done_count * (work_count - done_count)
        print(f"[extract] {done_count}/{work_count} (chunk {idx}) failed after 5 retries. "
              f"elapsed {elapsed:.0f}s eta {eta:.0f}s last error: {last_err}", flush=True)
        return ChunkExtractionResult(
            chunk_id=chunk.chunk_id,
            entities=[],
            relations=[],
            error=f"{type(last_err).__name__}: {last_err}",
        )

    pending_results = await asyncio.gather(
        *(
            process_one(idx, chunk)
            for idx, chunk in pending
        )
    )
    results_by_chunk_id.update({
        result.chunk_id: result
        for result in pending_results
    })

    all_entities: list[Entity] = []
    all_relations: list[Relation] = []
    failed_chunk_ids: list[str] = []

    for chunk in chunks:
        result = results_by_chunk_id[chunk.chunk_id]
        if result.error is not None:
            failed_chunk_ids.append(chunk.chunk_id)
            continue
        all_entities.extend(result.entities)
        all_relations.extend(result.relations)

    if cache is not None:
        cache.save_state(
            chunk_count=chunks_num,
            completed_chunk_count=chunks_num - len(failed_chunk_ids),
            failed_chunk_ids=failed_chunk_ids,
        )

    if failed_chunk_ids:
        print(
            f"[extract] {len(failed_chunk_ids)}/{chunks_num} "
            "chunk(s) failed and were skipped",
            flush=True,
        )
    print(f"[extract] All done: {len(all_entities)} entities, {len(all_relations)} relations, "
          f"Total time: {time.time()-start:.0f}s", flush=True)

    return ExtractionResult(
        entities=all_entities,
        relations=all_relations,
        failed_chunk_ids=failed_chunk_ids,
    )
