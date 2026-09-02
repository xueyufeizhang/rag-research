from rag_research.chunking import (
    AGENTIC_CHUNKING_SYSTEM_PROMPT,
    CHUNKING_PIPELINE_VERSION,
    ChunkConfig,
    ChunkSpan,
    _strict_partition_is_feasible,
    _validate_agentic_boundaries,
    chunk_async,
)
from rag_research.embedding import BatchEmbeddingFunction, embed_texts
from rag_research.extraction import (
    EXTRACTION_PIPELINE_VERSION,
    Entity,
    ExtractionResult,
    Relation,
    extract,
)
from rag_research.models import InputDocument, ChunkRecord, BuildResult
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
from rag_research.storage import KVStore, GraphStore, VectorIndex
from typing import Any
from collections.abc import Sequence
import json
from rag_research.prompts import PROMPTS
import os
import hashlib
import re
import unicodedata

load_dotenv()

CHUNKING_CACHE_SCHEMA_VERSION = 4
BUILD_PIPELINE_VERSION = 4
ENTITY_DESCRIPTION_MAX_VARIANTS = 12
ENTITY_DESCRIPTION_MAX_CHARS = 4000
RELATION_DESCRIPTION_MAX_VARIANTS = 12
RELATION_DESCRIPTION_MAX_CHARS = 4000


def _normalize_entity_name_key(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _select_canonical_entity_name(entities: list[Entity]) -> str:
    counts: dict[str, int] = {}
    first_positions: dict[str, int] = {}
    for index, entity in enumerate(entities):
        counts[entity.name] = counts.get(entity.name, 0) + 1
        first_positions.setdefault(entity.name, index)
    return max(
        counts,
        key=lambda name: (counts[name], -first_positions[name]),
    )


def _select_canonical_entity_type(entities: list[Entity]) -> str:
    counts: dict[str, int] = {}
    first_positions: dict[str, int] = {}
    for index, entity in enumerate(entities):
        counts[entity.type] = counts.get(entity.type, 0) + 1
        first_positions.setdefault(entity.type, index)
    return max(
        counts,
        key=lambda entity_type: (
            counts[entity_type],
            entity_type != "Other",
            -first_positions[entity_type],
        ),
    )


def _unique_strings(values: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean_value = value.strip()
        key = clean_value.casefold()
        if clean_value and key not in seen:
            unique.append(clean_value)
            seen.add(key)
    return unique


def _evenly_spaced(values: list[str], limit: int) -> list[str]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    indexes = [
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [values[index] for index in indexes]


def _aggregate_descriptions(
    descriptions: Sequence[str],
    *,
    max_variants: int,
    max_chars: int,
) -> str:
    if max_variants <= 0 or max_chars <= 0:
        raise ValueError("description aggregation limits must be positive")

    candidates = _evenly_spaced(
        _unique_strings(descriptions),
        max_variants,
    )
    selected: list[str] = []
    current_length = 0
    separator_length = len(" | ")
    for description in candidates:
        extra_length = len(description) + (
            separator_length if selected else 0
        )
        if current_length + extra_length <= max_chars:
            selected.append(description)
            current_length += extra_length
            continue
        if not selected:
            selected.append(
                description[:max_chars]
                if len(description) <= max_chars
                else description[:max_chars - 1].rstrip() + "…"
            )
        break
    return " | ".join(selected)


def _merge_extraction_records(
    entities: Sequence[Entity],
    relations: Sequence[Relation],
) -> tuple[dict[str, Entity], dict[str, Relation]]:
    entity_groups: dict[str, list[Entity]] = {}
    for entity in entities:
        key = _normalize_entity_name_key(entity.name)
        entity_groups.setdefault(key, []).append(entity)

    clean_entities: dict[str, Entity] = {}
    canonical_names: dict[str, str] = {}
    for normalized_name, group in entity_groups.items():
        canonical_name = _select_canonical_entity_name(group)
        canonical_names[normalized_name] = canonical_name
        clean_entities[canonical_name] = Entity(
            name=canonical_name,
            type=_select_canonical_entity_type(group),
            description=_aggregate_descriptions(
                [entity.description for entity in group],
                max_variants=ENTITY_DESCRIPTION_MAX_VARIANTS,
                max_chars=ENTITY_DESCRIPTION_MAX_CHARS,
            ),
            source_id=_unique_strings([
                source_id
                for entity in group
                for source_id in entity.source_id
            ]),
        )

    relation_groups: dict[str, list[Relation]] = {}
    relation_endpoints: dict[str, tuple[str, str]] = {}
    for relation in relations:
        source_key = _normalize_entity_name_key(relation.source)
        target_key = _normalize_entity_name_key(relation.target)
        source = canonical_names.get(source_key)
        target = canonical_names.get(target_key)
        if source is None or target is None:
            raise RuntimeError(
                "relation endpoint is absent after global entity resolution: "
                f"{relation.source!r} -- {relation.target!r}"
            )
        if source_key == target_key:
            raise RuntimeError(
                "self-relation survived extraction validation: "
                f"{relation.source!r} -- {relation.target!r}"
            )

        left, right = sorted((source, target))
        relation_key = f"{left}||{right}"
        relation_groups.setdefault(relation_key, []).append(relation)
        relation_endpoints[relation_key] = (left, right)

    clean_relations: dict[str, Relation] = {}
    for relation_key, group in relation_groups.items():
        source, target = relation_endpoints[relation_key]
        clean_relations[relation_key] = Relation(
            source=source,
            target=target,
            keywords=_unique_strings([
                keyword
                for relation in group
                for keyword in relation.keywords
            ]),
            description=_aggregate_descriptions(
                [relation.description for relation in group],
                max_variants=RELATION_DESCRIPTION_MAX_VARIANTS,
                max_chars=RELATION_DESCRIPTION_MAX_CHARS,
            ),
            source_id=_unique_strings([
                source_id
                for relation in group
                for source_id in relation.source_id
            ]),
        )

    return clean_entities, clean_relations

@dataclass
class LightRAGConfig:
    chunk_config: ChunkConfig = field(default_factory=lambda: ChunkConfig(
        strategy=os.getenv("CHUNKING_STRATEGY", "fixed"),
        fixed_size=int(os.getenv("FIXED_WINDOW_SIZE", 2400)),
        fixed_overlap=int(os.getenv("FIXED_WINDOW_OVERLAP", 200)),
        sentence_window_size=int(os.getenv("SENTENCE_WINDOW_SIZE", 8)),
        sentence_window_overlap=int(os.getenv("SENTENCE_WINDOW_OVERLAP", 2)),
        semantic_breakpoint_percentile=float(os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", 90)),
        semantic_min_sentences=int(os.getenv("SEMANTIC_MIN_SENTENCES", 8)),
        semantic_max_sentences=int(os.getenv("SEMANTIC_MAX_SENTENCES", 24)),
        semantic_buffer_size=int(os.getenv("SEMANTIC_BUFFER_SIZE", 1)),
        semantic_embedding_batch_size=int(os.getenv("SEMANTIC_EMBEDDING_BATCH_SIZE", 32)),
        semantic_embedding_concurrency=int(os.getenv("SEMANTIC_EMBEDDING_CONCURRENCY", 4)),
        agentic_batch_max_sentences=int(os.getenv("AGENTIC_BATCH_MAX_SENTENCES", 60)),
        agentic_batch_max_chars=int(os.getenv("AGENTIC_BATCH_MAX_CHARS", 12000)),
        agentic_min_sentences=int(os.getenv("AGENTIC_MIN_SENTENCES", 4)),
        agentic_max_sentences=int(os.getenv("AGENTIC_MAX_SENTENCES", 20)),
        agentic_concurrency=int(os.getenv("AGENTIC_CONCURRENCY", 4)),
        agentic_retries=int(os.getenv("AGENTIC_RETRIES", 2)),
    ))
    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K", 5))
    chunk_candidate_top_k: int = int(os.getenv("CHUNK_CANDIDATE_TOP_K", 20))
    entity_top_k: int = int(os.getenv("ENTITY_TOP_K", 5))
    relation_top_k: int = int(os.getenv("RELATION_TOP_K", 5))
    relation_candidate_top_k: int = int(os.getenv("RELATION_CANDIDATE_TOP_K", 20))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))
    embedding_concurrency: int = int(os.getenv("EMBEDDING_CONCURRENCY", 2))
    llm_backend: str = os.getenv("LLM_BACKEND", "ollama")
    llm_model: str = (
        os.getenv("API_MODEL", "")
        if os.getenv("LLM_BACKEND", "ollama") == "api"
        else os.getenv("LLM_MODEL", "")
    )
    embedding_backend: str = "ollama"
    embedding_model: str = os.getenv("EMBED_MODEL", "")


class LightRAG:

    def __init__(
        self,
        working_dir,
        llm_func,
        con_num,
        embed_func,
        config=None,
        reranker=None,
        embed_many_func: BatchEmbeddingFunction | None = None,
        cache_directory: str | os.PathLike[str] | None = None,
    ):
        self.working_dir = working_dir
        self.cache_directory = os.fspath(
            cache_directory
            if cache_directory is not None
            else os.path.join(working_dir, "pipeline_cache")
        )
        self.llm_func = llm_func
        self.con_num = int(con_num)
        self.embed_func = embed_func
        self.embed_many_func = embed_many_func
        self.config = config or LightRAGConfig()
        if self.config.embedding_batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        if self.config.embedding_concurrency <= 0:
            raise ValueError("embedding concurrency must be positive")
        os.makedirs(working_dir, exist_ok=True)
        os.makedirs(self.cache_directory, exist_ok=True)

        self.reranker = reranker

        self.entity_kv = KVStore(os.path.join(working_dir, "entities.json"))
        self.relation_kv = KVStore(os.path.join(working_dir, "relations.json"))
        self.chunk_kv = KVStore(os.path.join(working_dir, "chunks.json"))
        self.entity_vidx = VectorIndex(os.path.join(working_dir, "entity_vectors"))
        self.relation_vidx = VectorIndex(os.path.join(working_dir, "relation_vectors"))
        self.chunk_vidx = VectorIndex(os.path.join(working_dir, "chunk_vectors"))
        self.graph = GraphStore(os.path.join(working_dir, "graph.json"))
        self.build_manifest_kv = KVStore(os.path.join(working_dir, "build_manifest.json"))

        self._load_all()


    def _load_all(self):
        for store in (self.entity_kv, self.relation_kv, self.chunk_kv,
                      self.entity_vidx, self.relation_vidx, self.chunk_vidx, self.graph,
                      self.build_manifest_kv):
            store.load()

    def _save_all(self):
        for store in (self.entity_kv, self.relation_kv, self.chunk_kv,
                      self.entity_vidx, self.relation_vidx, self.chunk_vidx, self.graph):
            store.save()

    def _store_files_exist(self) -> bool:
        paths = (
            self.entity_kv.file_path,
            self.relation_kv.file_path,
            self.chunk_kv.file_path,
            self.graph.file_path,
            self.entity_vidx.id_path,
            self.entity_vidx.vector_path,
            self.relation_vidx.id_path,
            self.relation_vidx.vector_path,
            self.chunk_vidx.id_path,
            self.chunk_vidx.vector_path,
        )
        return any(os.path.exists(path) for path in paths)

    def _load_cached_build(
            self,
            build_fingerprint: str,
    ) -> BuildResult | None:
        cached_data = self.build_manifest_kv.get("build")
        if cached_data is None:
            if self._store_files_exist():
                raise RuntimeError(
                    "working directory contains store files but no build manifest; "
                    "it may contain a legacy or incomplete build"
                )
            return None

        if not isinstance(cached_data, dict):
            raise RuntimeError("invalid build manifest: build must be an object")

        try:
            cached_build = BuildResult(**cached_data)
        except TypeError as error:
            raise RuntimeError("invalid build manifest fields") from error

        if cached_build.build_fingerprint != build_fingerprint:
            raise RuntimeError(
                "working directory already contains an index built from different "
                "documents, models, prompts, or chunking configuration; "
                "use a different working directory"
            )

        if cached_build.failed_chunk_ids:
            raise RuntimeError(
                "cached build is incomplete: "
                f"{len(cached_build.failed_chunk_ids)} chunks failed; "
                "it cannot be reused"
            )

        stored_counts = {
            "chunk_count": len(self.chunk_kv.all()),
            "entity_count": len(self.entity_kv.all()),
            "relation_count": len(self.relation_kv.all()),
        }
        expected_counts = {
            "chunk_count": cached_build.chunk_count,
            "entity_count": cached_build.entity_count,
            "relation_count": cached_build.relation_count,
        }
        if stored_counts != expected_counts:
            raise RuntimeError(
                "stored record counts do not match build manifest: "
                f"expected {expected_counts}, found {stored_counts}"
            )

        return cached_build

    def _save_build_manifest(self, build_result: BuildResult) -> None:
        manifest = KVStore(self.build_manifest_kv.file_path)
        manifest.set("build", build_result)
        manifest.save()
        self.build_manifest_kv = manifest

    def _build_context(self, entities: list[dict], relations: list[dict], chunks: list[dict]) -> str:
        parts = []
        if entities:
            entity_lines = "\n".join(json.dumps(
                {
                    "name": e.get("name", ""),
                    "type": e.get("type", ""),
                    "description": e.get("description", ""),
                },
                ensure_ascii=False,
            ) for e in entities)
            parts.append("-----Entities-----\n" + entity_lines)
        if relations:
            relation_lines = "\n".join(json.dumps(
                {
                    "source": r.get("source", ""),
                    "target": r.get("target", ""),
                    "keywords": r.get("keywords", []),
                    "description": r.get("description", ""),
                },
                ensure_ascii=False,
            ) for r in relations)
            parts.append("-----Relations-----\n" + relation_lines)
        if chunks:
            chunk_lines = "\n".join(json.dumps(
                {"content": c.get("text", "")},
                ensure_ascii=False,
            ) for c in chunks if c)
            parts.append("-----Chunks-----\n" + chunk_lines)
        return "\n\n".join(parts)

    def _get_relation_candidates_from_entities(self, entities: list[dict]) -> dict[str, dict]:
        candidate_relations = {}
        seen = set()

        for entity in entities:
            name = entity.get("name")
            if not name or self.graph.get_node(name) is None:
                continue

            for nb in self.graph.get_neighbors(name):
                relation_key = "||".join(sorted([name, nb]))
                if relation_key in seen:
                    continue
                relation = self.relation_kv.get(relation_key)
                if relation is None:
                    continue
                candidate_relations[relation_key] = relation
                seen.add(relation_key)
        return candidate_relations

    def _rerank_relations(self, query: str, shortlist: list[dict]) -> list[dict]:
        if not shortlist:
            return []
        if self.reranker is None:
            return shortlist[:self.config.relation_top_k]

        def relation_to_text(relation: dict) -> str:
            source = relation.get("source", "")
            target = relation.get("target", "")
            keywords = ", ".join(relation.get("keywords", []))
            description = relation.get("description", "")
            return (
                f"source: {source}; "
                f"target: {target}; "
                f"keywords: {keywords}; "
                f"description: {description}"
            )

        pairs = [(query, relation_to_text(relation)) for relation in shortlist]
        rerank_scores = self.reranker.predict(pairs)
        ranked_relations = sorted(
            zip(shortlist, rerank_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            relation
            for relation, _ in ranked_relations[:self.config.relation_top_k]
        ]


    def _get_entities_from_relations(self, relations: list[dict]) -> list[dict]:
        entities = []
        seen = set()

        for relation in relations:
            for node in (relation.get("source"), relation.get("target")):
                if not node or node in seen:
                    continue
                entity = self.entity_kv.get(node)
                if entity:
                    entities.append(entity)
                    seen.add(node)
        return entities

    def _get_chunks_by_source_ids(self, source_ids: list[str]) -> list[dict]:
        chunks = []
        seen = set()
        for sid in source_ids:
            if not sid or sid in seen:
                continue
            chunk = self.chunk_kv.get(sid)
            if chunk:
                chunks.append({
                    **chunk,
                    "chunk_id": sid,
                })
                seen.add(sid)
        return chunks

    def _dense_filter_chunks(
            self,
            emb: list[float],
            candidate_chunks: list[dict],
    ) -> list[dict]:
        candidate_chunks_by_id = {
            chunk["chunk_id"]: chunk
            for chunk in candidate_chunks
            if chunk.get("chunk_id") and chunk.get("text")
        }
        if not candidate_chunks_by_id:
            return []

        candidate_vidx = VectorIndex(os.path.join(self.working_dir, "temporary_candidate_chunk_vectors"))
        for chunk_id in candidate_chunks_by_id:
            vector = self.chunk_vidx.get_vector(chunk_id)
            if vector is None:
                continue
            candidate_vidx.add(chunk_id, vector)

        dense_hits = candidate_vidx.query(emb, self.config.chunk_candidate_top_k)
        return [
            candidate_chunks_by_id[chunk_id]
            for chunk_id, _ in dense_hits
            if chunk_id in candidate_chunks_by_id
        ]

    def _rerank_chunks(self, query: str, shortlist: list[dict]) -> list[dict]:
        if not shortlist:
            return []
        if self.reranker is None:
            return shortlist[:self.config.chunk_top_k]

        pairs = [(query, chunk["text"]) for chunk in shortlist]
        rerank_scores = self.reranker.predict(pairs)
        ranked_chunks = sorted(
            zip(shortlist, rerank_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            chunk
            for chunk, _ in ranked_chunks[:self.config.chunk_top_k]
        ]
    
    async def _naive_retrieve(self, query: str) -> list[dict]:
        emb = await self.embed_func(query)
        dense_hits = self.chunk_vidx.query(
            emb,
            self.config.chunk_candidate_top_k,
        )
        shortlist = []
        for chunk_id, dense_score in dense_hits:
            chunk = self.chunk_kv.get(chunk_id)
            if chunk:
                shortlist.append({
                    **chunk,
                    "chunk_id": chunk_id,
                    "dense_score": dense_score,
                })
        return self._rerank_chunks(query, shortlist)


    def _dense_filter_local_relations(
            self,
            emb: list[float],
            candidate_relations: dict[str, dict],
    ) -> list[dict]:
        candidate_vidx = VectorIndex(os.path.join(self.working_dir, "temporary_local_relation_vectors"))
        for key in candidate_relations.keys():
            vector = self.relation_vidx.get_vector(key)
            if vector is None:
                continue
            candidate_vidx.add(key, vector)

        dense_hits = candidate_vidx.query(emb, self.config.relation_candidate_top_k)
        shortlist = [
            candidate_relations[relation_key]
            for relation_key, _ in dense_hits
            if relation_key in candidate_relations
        ]
        return shortlist

    async def _local_retrieve(
            self,
            query: str,
            init_emb: list[float] | None = None,
            retrieve_chunks: bool = True,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        emb = init_emb if init_emb is not None else await self.embed_func(query)

        dense_hits = self.entity_vidx.query(emb, self.config.entity_top_k)
        entities = [self.entity_kv.get(k) for k, _ in dense_hits if self.entity_kv.get(k)]
        candidate_relations = self._get_relation_candidates_from_entities(entities)
        relation_shortlist = self._dense_filter_local_relations(emb, candidate_relations)
        relations = self._rerank_relations(query, relation_shortlist)
        if not retrieve_chunks:
            return entities, relations, []

        source_ids = []
        for e in entities:
            source_ids.extend(e.get("source_id", []))
        for r in relations:
            source_ids.extend(r.get("source_id", []))
        candidate_chunks = self._get_chunks_by_source_ids(source_ids)
        chunk_shortlist = self._dense_filter_chunks(emb, candidate_chunks)
        chunks = self._rerank_chunks(query, chunk_shortlist)
        return entities, relations, chunks

    async def _global_retrieve(
            self,
            query: str,
            init_emb: list[float] | None = None,
            retrieve_chunks: bool = True,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        emb = init_emb if init_emb is not None else await self.embed_func(query)

        dense_hits = self.relation_vidx.query(emb, self.config.relation_candidate_top_k)
        relation_shortlist = [self.relation_kv.get(k) for k, _ in dense_hits if self.relation_kv.get(k)]
        relations = self._rerank_relations(query, relation_shortlist)
        entities = self._get_entities_from_relations(relations)
        if not retrieve_chunks:
            return entities, relations, []

        source_ids = []
        for r in relations:
            source_ids.extend(r.get("source_id", []))
        for e in entities:
            source_ids.extend(e.get("source_id", []))
        candidate_chunks = self._get_chunks_by_source_ids(source_ids)
        chunk_shortlist = self._dense_filter_chunks(emb, candidate_chunks)
        chunks = self._rerank_chunks(query, chunk_shortlist)
        return entities, relations, chunks

    def _dedupe_entities(self, entities: list[dict]) -> list[dict]:
        output_entities = []
        seen = set()

        for entity in entities:
            key = entity.get("name")
            if not key or key in seen:
                continue
            output_entities.append(entity)
            seen.add(key)
        return output_entities

    def _dedupe_relations(self, relations: list[dict]) -> list[dict]:
        output_relations = []
        seen = set()

        for relation in relations:
            source = relation.get("source")
            target = relation.get("target")
            if not source or not target:
                continue
            key = "||".join(sorted([source, target]))
            if key in seen:
                continue
            output_relations.append(relation)
            seen.add(key)
        return output_relations


    async def _hybrid_retrieve(self, query: str) -> tuple[list[dict], list[dict], list[dict]]:
        emb = await self.embed_func(query)

        local_entities, local_relations, _ = await self._local_retrieve(query, emb, retrieve_chunks=False)
        global_entities, global_relations, _ = await self._global_retrieve(query, emb, retrieve_chunks=False)
        entities = self._dedupe_entities(local_entities + global_entities)
        relations = self._dedupe_relations(local_relations + global_relations)

        source_ids = []
        for e in entities:
            source_ids.extend(e.get("source_id", []))
        for r in relations:
            source_ids.extend(r.get("source_id", []))
        candidate_chunks = self._get_chunks_by_source_ids(source_ids)
        shortlist = self._dense_filter_chunks(emb, candidate_chunks)
        chunks = self._rerank_chunks(query, shortlist)
        return entities, relations, chunks


    def _make_build_provenance(self) -> dict[str, Any]:
        chunking_strategy = self.config.chunk_config.strategy.lower()
        chunking_provenance: dict[str, Any] = {
            "strategy": chunking_strategy,
            "pipeline_version": CHUNKING_PIPELINE_VERSION,
        }
        if chunking_strategy == "semantic":
            chunking_provenance.update({
                "embedding_backend": self.config.embedding_backend,
                "embedding_model": self.config.embedding_model,
            })
        elif chunking_strategy == "agentic_ibm":
            chunking_provenance.update({
                "llm_backend": self.config.llm_backend,
                "llm_model": self.config.llm_model,
                "prompt_sha256": hashlib.sha256(
                    AGENTIC_CHUNKING_SYSTEM_PROMPT.encode("utf-8")
                ).hexdigest(),
            })

        extraction_prompt = {
            key: PROMPTS[key]
            for key in (
                "default_entity_types_guidance",
                "entity_extraction_system_prompt",
                "entity_extraction_user_prompt",
                "entity_extraction_examples",
            )
        }
        prompt_json = json.dumps(
            extraction_prompt,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        return {
            "chunking": chunking_provenance,
            "extraction": {
                "backend": self.config.llm_backend,
                "model": self.config.llm_model,
                "pipeline_version": EXTRACTION_PIPELINE_VERSION,
                "prompt_sha256": hashlib.sha256(
                    prompt_json.encode("utf-8")
                ).hexdigest(),
            },
            "embedding": {
                "backend": self.config.embedding_backend,
                "model": self.config.embedding_model,
            },
            "assembly": {
                "pipeline_version": BUILD_PIPELINE_VERSION,
                "entity_name_normalization": "nfkc-whitespace-casefold",
                "entity_description_max_variants": (
                    ENTITY_DESCRIPTION_MAX_VARIANTS
                ),
                "entity_description_max_chars": (
                    ENTITY_DESCRIPTION_MAX_CHARS
                ),
                "relation_description_max_variants": (
                    RELATION_DESCRIPTION_MAX_VARIANTS
                ),
                "relation_description_max_chars": (
                    RELATION_DESCRIPTION_MAX_CHARS
                ),
            },
        }


    @staticmethod
    def _fingerprint_payload(payload: dict[str, Any]) -> str:
        canonical_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


    def _make_chunking_fingerprint(
        self,
        build_provenance: dict[str, Any] | None = None,
    ) -> str:
        provenance = build_provenance or self._make_build_provenance()
        return self._fingerprint_payload({
            "schema_version": 1,
            "chunk_config": asdict(self.config.chunk_config),
            "chunking_provenance": provenance["chunking"],
        })


    def _make_extraction_fingerprint(
        self,
        build_provenance: dict[str, Any] | None = None,
    ) -> str:
        provenance = build_provenance or self._make_build_provenance()
        return self._fingerprint_payload({
            "schema_version": 1,
            "extraction_provenance": provenance["extraction"],
        })


    def _make_build_fingerprint(
        self,
        documents: Sequence[InputDocument],
        build_provenance: dict[str, Any] | None = None,
    ) -> str:
        document_records = [
            {
                "document_id": document.document_id,
                "text_sha256": hashlib.sha256(
                    document.text.encode("utf-8")
                ).hexdigest(),
                "metadata": document.metadata,
            } for document in documents
        ]

        document_records.sort(
            key=lambda record: record["document_id"]
        )

        payload = {
            "schema_version": 4,
            "documents": document_records,
            "chunk_config": asdict(
                self.config.chunk_config
            ),
            "build_provenance": build_provenance or self._make_build_provenance(),
        }

        return self._fingerprint_payload(payload)


    async def _embed_into_index(
        self,
        records: Sequence[tuple[str, str]],
        vector_index: VectorIndex,
        label: str,
    ) -> None:
        if not records:
            return
        if not callable(self.embed_func):
            raise TypeError("construct requires a callable embed_func")

        total = len(records)
        completed = 0
        window_size = (
            self.config.embedding_batch_size
            * self.config.embedding_concurrency
        )
        print(
            f"[embed] {label}: 0/{total} "
            f"(batch={self.config.embedding_batch_size}, "
            f"concurrency={self.config.embedding_concurrency})",
            flush=True,
        )

        for start in range(0, total, window_size):
            window = records[start:start + window_size]
            vectors = await embed_texts(
                [text for _, text in window],
                embed_func=self.embed_func,
                embed_many_func=self.embed_many_func,
                batch_size=self.config.embedding_batch_size,
                concurrency=self.config.embedding_concurrency,
            )
            for (record_id, _), vector in zip(window, vectors, strict=True):
                vector_index.add(record_id, vector)

            completed += len(window)
            print(f"[embed] {label}: {completed}/{total}", flush=True)


    def _chunk_cache_path(
        self,
        chunking_fingerprint: str,
        document: InputDocument,
    ) -> str:
        text_sha256 = hashlib.sha256(
            document.text.encode("utf-8")
        ).hexdigest()
        document_digest = hashlib.sha256(
            f"{document.document_id}\0{text_sha256}".encode("utf-8")
        ).hexdigest()
        return os.path.join(
            self.cache_directory,
            "chunking",
            chunking_fingerprint,
            "documents",
            f"{document_digest}.json",
        )


    def _validate_chunk_spans(
        self,
        document: InputDocument,
        raw_spans: object,
        *,
        cache_path: str,
    ) -> list[ChunkSpan]:
        if not isinstance(raw_spans, list) or not raw_spans:
            raise ValueError(
                f"invalid chunking cache spans: {cache_path}"
            )

        spans: list[ChunkSpan] = []
        previous_start = -1
        covered_end = 0
        for index, raw_span in enumerate(raw_spans):
            if not isinstance(raw_span, dict):
                raise ValueError(
                    f"invalid chunking cache span {index}: {cache_path}"
                )

            span_text = raw_span.get("text")
            char_start = raw_span.get("char_start")
            char_end = raw_span.get("char_end")
            if not isinstance(span_text, str):
                raise ValueError(
                    f"invalid chunking cache text at span {index}: {cache_path}"
                )
            if type(char_start) is not int or type(char_end) is not int:
                raise ValueError(
                    f"invalid chunking cache offsets at span {index}: {cache_path}"
                )
            if not 0 <= char_start < char_end <= len(document.text):
                raise ValueError(
                    f"out-of-range chunking cache span {index}: {cache_path}"
                )
            if char_start <= previous_start:
                raise ValueError(
                    f"unordered chunking cache span {index}: {cache_path}"
                )
            if index == 0 and char_start != 0:
                raise ValueError(
                    f"chunking cache does not start at zero: {cache_path}"
                )
            if char_start > covered_end:
                raise ValueError(
                    f"chunking cache has a coverage gap at span {index}: {cache_path}"
                )
            if document.text[char_start:char_end] != span_text:
                raise ValueError(
                    f"chunking cache text mismatch at span {index}: {cache_path}"
                )

            spans.append(ChunkSpan(span_text, char_start, char_end))
            previous_start = char_start
            covered_end = max(covered_end, char_end)

        if covered_end != len(document.text):
            raise ValueError(
                f"chunking cache does not cover the document: {cache_path}"
            )
        return spans


    def _load_cached_chunk_spans(
        self,
        document: InputDocument,
        chunking_fingerprint: str,
    ) -> tuple[list[ChunkSpan], list[dict[str, object]]] | None:
        cache_path = self._chunk_cache_path(
            chunking_fingerprint,
            document,
        )
        if not os.path.exists(cache_path):
            return None

        cache_store = KVStore(cache_path)
        cache_store.load()
        payload = cache_store.get("document")
        if not isinstance(payload, dict):
            raise ValueError(f"invalid chunking cache record: {cache_path}")
        if payload.get("schema_version") != CHUNKING_CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported chunking cache schema: {cache_path}")
        if payload.get("chunking_fingerprint") != chunking_fingerprint:
            raise ValueError(f"chunking cache fingerprint mismatch: {cache_path}")
        if payload.get("document_id") != document.document_id:
            raise ValueError(f"chunking cache document ID mismatch: {cache_path}")
        expected_text_sha256 = hashlib.sha256(
            document.text.encode("utf-8")
        ).hexdigest()
        if payload.get("text_sha256") != expected_text_sha256:
            raise ValueError(f"chunking cache document text mismatch: {cache_path}")
        if payload.get("strategy") != self.config.chunk_config.strategy.lower():
            raise ValueError(f"chunking cache strategy mismatch: {cache_path}")

        spans = self._validate_chunk_spans(
            document,
            payload.get("spans"),
            cache_path=cache_path,
        )
        projection_events = self._validate_chunking_projection_events(
            payload.get("boundary_projection_events", []),
            cache_path=cache_path,
        )
        if (
            self.config.chunk_config.strategy.lower() != "agentic_ibm"
            and projection_events
        ):
            raise ValueError(
                f"non-agentic chunking cache contains projections: {cache_path}"
            )
        return spans, projection_events


    def _validate_chunking_projection_events(
        self,
        raw_events: object,
        *,
        cache_path: str,
    ) -> list[dict[str, object]]:
        if not isinstance(raw_events, list):
            raise ValueError(
                f"invalid chunking projection events: {cache_path}"
            )

        events: list[dict[str, object]] = []
        seen_event_keys: set[tuple[str, int]] = set()
        for event_index, raw_event in enumerate(raw_events):
            if not isinstance(raw_event, dict):
                raise ValueError(
                    f"invalid chunking projection event {event_index}: {cache_path}"
                )
            scope = raw_event.get("scope")
            batch_index = raw_event.get("batch_index")
            sentence_count = raw_event.get("sentence_count")
            if scope not in {"batch", "document"}:
                raise ValueError(
                    f"invalid projection scope: {cache_path}"
                )
            if type(batch_index) is not int or (
                scope == "batch" and batch_index <= 0
            ) or (
                scope == "document" and batch_index != 0
            ):
                raise ValueError(
                    f"invalid projected batch index: {cache_path}"
                )
            event_key = (scope, batch_index)
            if event_key in seen_event_keys:
                raise ValueError(
                    f"duplicate projection event: {cache_path}"
                )
            if type(sentence_count) is not int or sentence_count <= 0:
                raise ValueError(
                    f"invalid projected sentence count: {cache_path}"
                )

            normalized_event: dict[str, object] = {
                "scope": scope,
                "batch_index": batch_index,
                "sentence_count": sentence_count,
            }
            for field_name in (
                "original_boundaries",
                "projected_boundaries",
            ):
                raw_boundaries = raw_event.get(field_name)
                if not isinstance(raw_boundaries, list) or not raw_boundaries:
                    raise ValueError(
                        f"invalid {field_name}: {cache_path}"
                    )
                boundaries: list[list[int]] = []
                expected_start = 1
                for raw_boundary in raw_boundaries:
                    if (
                        not isinstance(raw_boundary, list)
                        or len(raw_boundary) != 2
                        or any(type(value) is not int for value in raw_boundary)
                    ):
                        raise ValueError(
                            f"invalid {field_name}: {cache_path}"
                        )
                    start, end = raw_boundary
                    if start != expected_start or end < start or end > sentence_count:
                        raise ValueError(
                            f"invalid {field_name}: {cache_path}"
                        )
                    boundaries.append([start, end])
                    expected_start = end + 1
                if boundaries[-1][1] != sentence_count:
                    raise ValueError(
                        f"incomplete {field_name}: {cache_path}"
                    )
                normalized_event[field_name] = boundaries

            if (
                normalized_event["original_boundaries"]
                == normalized_event["projected_boundaries"]
            ):
                raise ValueError(
                    f"chunking projection event contains no adjustment: {cache_path}"
                )

            projected_boundaries = [
                (start, end)
                for start, end in normalized_event["projected_boundaries"]
            ]
            try:
                _validate_agentic_boundaries(
                    projected_boundaries,
                    sentence_count=sentence_count,
                    min_sentences=self.config.chunk_config.agentic_min_sentences,
                    max_sentences=self.config.chunk_config.agentic_max_sentences,
                    allow_short_final=(
                        scope == "batch"
                        or not _strict_partition_is_feasible(
                            sentence_count,
                            min_sentences=(
                                self.config.chunk_config.agentic_min_sentences
                            ),
                            max_sentences=(
                                self.config.chunk_config.agentic_max_sentences
                            ),
                        )
                    ),
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid projected boundaries: {cache_path}"
                ) from exc
            events.append(normalized_event)
            seen_event_keys.add(event_key)

        events.sort(key=lambda event: (
            event["scope"] == "document",
            int(event["batch_index"]),
        ))
        return events


    def _save_chunk_spans(
        self,
        document: InputDocument,
        chunking_fingerprint: str,
        spans: list[ChunkSpan],
        projection_events: list[dict[str, object]],
    ) -> None:
        cache_path = self._chunk_cache_path(
            chunking_fingerprint,
            document,
        )
        validated_spans = self._validate_chunk_spans(
            document,
            [asdict(span) for span in spans],
            cache_path=cache_path,
        )
        validated_projection_events = self._validate_chunking_projection_events(
            projection_events,
            cache_path=cache_path,
        )
        if (
            self.config.chunk_config.strategy.lower() != "agentic_ibm"
            and validated_projection_events
        ):
            raise ValueError(
                "non-agentic chunking cannot record boundary projections"
            )
        cache_store = KVStore(cache_path)
        cache_store.set(
            "document",
            {
                "schema_version": CHUNKING_CACHE_SCHEMA_VERSION,
                "chunking_fingerprint": chunking_fingerprint,
                "document_id": document.document_id,
                "text_sha256": hashlib.sha256(
                    document.text.encode("utf-8")
                ).hexdigest(),
                "strategy": self.config.chunk_config.strategy.lower(),
                "spans": [asdict(span) for span in validated_spans],
                "boundary_projection_events": validated_projection_events,
            },
        )
        cache_store.save()


    async def _chunk_document(
        self,
        document: InputDocument,
        chunking_fingerprint: str,
    ) -> tuple[list[ChunkRecord], bool, list[dict[str, object]]]:
        cached_result = self._load_cached_chunk_spans(
            document,
            chunking_fingerprint,
        )
        loaded_from_cache = cached_result is not None
        if cached_result is None:
            projection_events: list[dict[str, object]] = []
            spans = await chunk_async(
                document.text,
                self.config.chunk_config,
                embed_func=self.embed_func,
                embed_many_func=self.embed_many_func,
                llm_func=self.llm_func,
                agentic_projection_events=projection_events,
            )
            self._save_chunk_spans(
                document,
                chunking_fingerprint,
                spans,
                projection_events,
            )
        else:
            spans, projection_events = cached_result

        chunks: list[ChunkRecord] = []
        for idx, span in enumerate(spans):
            chunk_id = self._make_chunk_id(document.document_id, idx, span)
            chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                document_id=document.document_id,
                text=span.text,
                model_text=self._make_model_text(document, span),
                chunk_index=idx,
                char_start=span.char_start,
                char_end=span.char_end,
                metadata=dict(document.metadata)
            ))
        return chunks, loaded_from_cache, projection_events


    def _make_chunk_id(self, document_id: str, chunk_index: int, span: ChunkSpan) -> str:
        payload = (
            f"{document_id}\0"
            f"{chunk_index}\0"
            f"{span.char_start}\0"
            f"{span.char_end}\0"
            f"{span.text}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"chunk-{digest[:24]}"


    def _make_model_text(self, document: InputDocument, span: ChunkSpan) -> str:
        metadata = document.metadata
        header_fields = [
            ("Title", metadata.get("title")),
            ("Source", metadata.get("source")),
            ("Author", metadata.get("author")),
            ("Published at", metadata.get("published_at")),
            ("Category", metadata.get("category")),
            ("URL", metadata.get("url")),
        ]

        header = "\n".join(
            f"{name}: {value}"
            for name, value in header_fields
            if value is not None and str(value).strip()
        )
        if not header:
            return span.text

        return f"Header:\n{header}\n\nContent:\n{span.text}"


    async def construct(self, documents: Sequence[InputDocument]) -> BuildResult:
        """
        1. split documents into chunks 👌
        2. R(·)：extract entities and relations + P(·)：create Key-Value Pair 👌
        3. D(·)：remove duplications and merge 👌
        4. storage（KV store + vector index + graph）👌
        """
        if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
            raise TypeError("documents must be a sequence of InputDocument")
        if not documents:
            raise ValueError("documents must not be empty")
        if not callable(self.llm_func):
            raise TypeError("construct requires a callable llm_func")
        if not callable(self.embed_func):
            raise TypeError("construct requires a callable embed_func")
        if self.embed_many_func is not None and not callable(self.embed_many_func):
            raise TypeError("embed_many_func must be callable")

        for idx, document in enumerate(documents):
            if not isinstance(document, InputDocument):
                raise TypeError(f"documents[{idx}] must be an InputDocument")
            if not isinstance(document.document_id, str):
                raise TypeError(f"documents[{idx}].document_id must be a string")
            if not document.document_id.strip():
                raise ValueError(f"documents[{idx}].document_id must not be empty")
            if not isinstance(document.text, str):
                raise TypeError(f"document {document.document_id}: text must be a string")
            if not document.text.strip():
                raise ValueError(f"document {document.document_id}: text must not be empty")

        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document_id must be unique")

        build_provenance = self._make_build_provenance()
        build_fingerprint = self._make_build_fingerprint(
            documents,
            build_provenance,
        )
        chunking_fingerprint = self._make_chunking_fingerprint(
            build_provenance,
        )
        extraction_fingerprint = self._make_extraction_fingerprint(
            build_provenance,
        )
        cached_build = self._load_cached_build(build_fingerprint)
        if cached_build is not None:
            print("[construct] matching index already built, reuse")
            return cached_build

        chunks: list[ChunkRecord] = []
        cached_document_count = 0
        chunking_projection_count = 0
        chunking_rebalance_count = 0
        document_count = len(documents)
        for index, document in enumerate(documents, start=1):
            (
                document_chunks,
                loaded_from_cache,
                projection_events,
            ) = await self._chunk_document(
                document,
                chunking_fingerprint,
            )
            chunks.extend(document_chunks)
            cached_document_count += int(loaded_from_cache)
            chunking_projection_count += sum(
                event.get("scope") == "batch"
                for event in projection_events
            )
            chunking_rebalance_count += sum(
                event.get("scope") == "document"
                for event in projection_events
            )
            if index % 10 == 0 or index == document_count:
                print(
                    f"[chunk] {index}/{document_count} documents, "
                    f"{len(chunks)} chunks, "
                    f"{cached_document_count} document cache hits, "
                    f"{chunking_projection_count} projected batches, "
                    f"{chunking_rebalance_count} rebalanced documents",
                    flush=True,
                )

        extraction_results: ExtractionResult = await extract(
            chunks,
            self.llm_func,
            self.con_num,
            cache_directory=os.path.join(
                self.cache_directory,
                "extraction",
            ),
            extraction_fingerprint=extraction_fingerprint,
            cache_scope=build_fingerprint,
        )

        if extraction_results.failed_chunk_ids:
            raise RuntimeError(
                "construction aborted because extraction failed for "
                f"{len(extraction_results.failed_chunk_ids)} chunks"
            )

        clean_entities, clean_relations = _merge_extraction_records(
            extraction_results.entities,
            extraction_results.relations,
        )
        
        entity_embedding_records = [
            (entity_key, f"{entity_key} {entity.description}".strip())
            for entity_key, entity in clean_entities.items()
        ]
        relation_embedding_records = [
            (
                relation_key,
                (
                    " ".join(relation.keywords)
                    + " "
                    + relation.description
                ).strip()
                or f"{relation.source} {relation.target}",
            )
            for relation_key, relation in clean_relations.items()
        ]
        chunk_embedding_records = [
            (chunk.chunk_id, chunk.model_text)
            for chunk in chunks
        ]

        await self._embed_into_index(
            entity_embedding_records,
            self.entity_vidx,
            "entities",
        )
        await self._embed_into_index(
            relation_embedding_records,
            self.relation_vidx,
            "relations",
        )
        await self._embed_into_index(
            chunk_embedding_records,
            self.chunk_vidx,
            "chunks",
        )

        for entity_key, entity in clean_entities.items():
            self.entity_kv.set(entity_key, entity)
            self.graph.add_node(entity)

        for relation_key, relation in clean_relations.items():
            self.relation_kv.set(relation_key, relation)
            self.graph.add_edge(relation)

        for chunk in chunks:
            self.chunk_kv.set(chunk.chunk_id, chunk)

        build_result = BuildResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            entity_count=len(clean_entities),
            relation_count=len(clean_relations),
            failed_chunk_ids=extraction_results.failed_chunk_ids,
            build_fingerprint=build_fingerprint,
            chunking_fingerprint=chunking_fingerprint,
            extraction_fingerprint=extraction_fingerprint,
            build_provenance=build_provenance,
            chunking_projection_count=chunking_projection_count,
            chunking_rebalance_count=chunking_rebalance_count,
        )

        self._save_all()
        self._save_build_manifest(build_result)

        return build_result

        
    async def retrieve(self, query: str, mode: str = 'hybrid') -> str:
        """
        1. Naive 👌
        2. Local 👌
        3. Global 👌
        4. Hybrid 👌
        """
        if mode == "naive":
            chunks = await self._naive_retrieve(query)
            context = self._build_context([], [], chunks)
            print(f"\n[retrieved {len(chunks)} chunks]\n{context[:500]}\n---\n")
            system_prompt = PROMPTS["naive_rag_response"].format(response_type="Multiple Paragraphs", context_data=context)
        elif mode == "local":
            entities, relations, chunks = await self._local_retrieve(query)
            context = self._build_context(entities, relations, chunks)
            system_prompt = PROMPTS["rag_response"].format(response_type="Multiple Paragraphs", context_data=context)
        elif mode == "global":
            entities, relations, chunks = await self._global_retrieve(query)
            context = self._build_context(entities, relations, chunks)
            system_prompt = PROMPTS["rag_response"].format(response_type="Multiple Paragraphs", context_data=context)
        elif mode == "hybrid":
            entities, relations, chunks = await self._hybrid_retrieve(query)
            context = self._build_context(entities, relations, chunks)
            system_prompt = PROMPTS["rag_response"].format(response_type="Multiple Paragraphs", context_data=context)
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        if mode in ["local", "global", "hybrid"]:
            print(f"\n[retrieved {len(entities)} entities, {len(relations)} relations, and {len(chunks)} chunks]\n{context[:500]}\n-----\n")
        return await self.llm_func(system=system_prompt, prompt=query)


    def _relation_key(self, relation: dict) -> str:
        source = relation.get("source")
        target = relation.get("target")
        if not source or not target:
            return ""
        return "||".join(sorted([source, target]))

    async def retrieve_trace(self, query: str, mode: str = "hybrid") -> dict:
        if mode == "naive":
            chunks = await self._naive_retrieve(query)
            entities, relations = [], []
        elif mode == "local":
            entities, relations, chunks = await self._local_retrieve(query)
        elif mode == "global":
            entities, relations, chunks = await self._global_retrieve(query)
        elif mode == "hybrid":
            entities, relations, chunks = await self._hybrid_retrieve(query)
        else:
            raise ValueError(f"unknown retrieval mode: {mode}")

        entity_ids = list(dict.fromkeys(
            e.get("name") for e in entities if e.get("name")
        ))
        relation_ids = list(dict.fromkeys(
            self._relation_key(r) for r in relations if r.get("source") and r.get("target")
        ))
        chunk_ids = list(dict.fromkeys(
            c.get("chunk_id") for c in chunks if c.get("chunk_id")
        ))
        return {
            "query": query,
            "mode": mode,
            "entity_ids": entity_ids,
            "relation_ids": relation_ids,
            "chunk_ids": chunk_ids,
            "entities": entities,
            "relations": relations,
            "chunks": chunks,
        }
