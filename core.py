from chunk import chunk
from extract import extract
from dataclasses import dataclass
from dotenv import load_dotenv
from storage import KVStore, GraphStore, VectorIndex
from typing import Any
import json
from prompt import PROMPTS
import os

load_dotenv()

@dataclass
class LightRAGConfig:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 2400))
    chunk_overlap_size: int = int(os.getenv("CHUNK_OVERLAP_SIZE", 200))
    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K", 5))
    entity_top_k: int = int(os.getenv("ENTITY_TOP_K", 5))
    relation_top_k: int = int(os.getenv("RELATION_TOP_K", 5))


class LightRAG:

    def __init__(self, working_dir, llm_func, con_num, embed_func, config=None):
        self.working_dir = working_dir
        self.llm_func = llm_func
        self.con_num = int(con_num)
        self.embed_func = embed_func
        self.config = config or LightRAGConfig()
        os.makedirs(working_dir, exist_ok=True)

        self.entity_kv = KVStore(os.path.join(working_dir, "entities.json"))
        self.relation_kv = KVStore(os.path.join(working_dir, "relations.json"))
        self.chunk_kv = KVStore(os.path.join(working_dir, "chunks.json"))
        self.entity_vidx = VectorIndex(os.path.join(working_dir, "entity_vectors"))
        self.relation_vidx = VectorIndex(os.path.join(working_dir, "relation_vectors"))
        self.chunk_vidx = VectorIndex(os.path.join(working_dir, "chunk_vectors"))
        self.graph = GraphStore(os.path.join(working_dir, "graph.json"))

        self._load_all()


    def _load_all(self):
        for store in (self.entity_kv, self.relation_kv, self.chunk_kv,
                      self.entity_vidx, self.relation_vidx, self.chunk_vidx, self.graph):
            store.load()

    def _save_all(self):
        for store in (self.entity_kv, self.relation_kv, self.chunk_kv,
                      self.entity_vidx, self.relation_vidx, self.chunk_vidx, self.graph):
            store.save()

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

    def _get_relations_from_entities(self, entities: list[dict]) -> list[dict]:
        relations = []
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
                if relation:
                    relations.append(relation)
                    seen.add(relation_key)
        return relations

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
            if not chunk:
                continue
            chunks.append(chunk)
            seen.add(sid)
        return chunks
    
    async def _naive_retrieve(self, query: str) -> list[dict]:
        emb = await self.embed_func(query)
        hits = self.chunk_vidx.query(emb, self.config.chunk_top_k)   # [(chunk_key, score)]
        return [self.chunk_kv.get(k) for k, _ in hits if self.chunk_kv.get(k)]

    async def _local_retrieve(self, query: str) -> tuple[list[dict], list[dict], list[dict]]:
        emb = await self.embed_func(query)
        hits = self.entity_vidx.query(emb, self.config.entity_top_k)
        entities = [self.entity_kv.get(k) for k, _ in hits if self.entity_kv.get(k)]
        relations = self._get_relations_from_entities(entities)

        source_ids = []
        for e in entities:
            source_ids.extend(e.get("source_id", []))
        for r in relations:
            source_ids.extend(r.get("source_id", []))
        chunks = self._get_chunks_by_source_ids(source_ids)
        return entities, relations, chunks

    async def _global_retrieve(self, query: str) -> tuple[list[dict], list[dict], list[dict]]:
        emb = await self.embed_func(query)
        hits = self.relation_vidx.query(emb, self.config.relation_top_k)
        relations = [self.relation_kv.get(k) for k, _ in hits if self.relation_kv.get(k)]
        entities = self._get_entities_from_relations(relations)

        source_ids = []
        # for e in entities:
        #     source_ids.extend(e.get("source_id", []))
        for r in relations:
            source_ids.extend(r.get("source_id", []))
        chunks = self._get_chunks_by_source_ids(source_ids)
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

    def _dedupe_chunks(self, chunks: list[dict]) -> list[dict]:
        output_chunks = []
        seen = set()

        for chunk in chunks:
            key = chunk.get("text")
            if not key or key in seen:
                continue
            output_chunks.append(chunk)
            seen.add(key)
        return output_chunks

    async def _hybrid_retrieve(self, query: str) -> tuple[list[dict], list[dict], list[dict]]:
        local_entities, local_relations, local_chunks = await self._local_retrieve(query)
        global_entities, global_relations, global_chunks = await self._global_retrieve(query)
        entities = self._dedupe_entities(local_entities + global_entities)
        relations = self._dedupe_relations(local_relations + global_relations)
        chunks = self._dedupe_chunks(local_chunks + global_chunks)
        return entities, relations, chunks


    async def construct(self, documents: str, file_id: str = "") -> None:
        """
        1. split documents into chunks 👌
        2. R(·)：extract entities and relations + P(·)：create Key-Value Pair 👌
        3. D(·)：remove duplications and merge 👌
        4. storage（KV store + vector index + graph）👌
        """
        if self.chunk_kv.all():
            print("[construct] store already built, skip")
            return
        chunks = chunk(documents, self.config.chunk_size, self.config.chunk_overlap_size)
        # chunks = chunks[:5]
        all_entities, all_relations = await extract(chunks, self.llm_func, file_id, self.con_num)

        clean_entities = {}
        clean_relations = {}

        for entity in all_entities:
            exist_name = clean_entities.get(entity.name, None)
            if exist_name:
                if entity.description:
                    exist_name.description += (" | " if exist_name.description else "") + entity.description
                exist_name.source_id.extend(entity.source_id)
            else:
                clean_entities[entity.name] = entity
          
        for relation in all_relations:
            relation_pair = "||".join(sorted([relation.source, relation.target]))
            exist_pair = clean_relations.get(relation_pair, None)
            if exist_pair:
                exist_pair.keywords.extend(relation.keywords)
                if relation.description:
                    exist_pair.description += (" | " if exist_pair.description else "") + relation.description
                exist_pair.source_id.extend(relation.source_id)
            else:
                clean_relations[relation_pair] = relation

        for ev in clean_entities.values():
            ev.source_id = list(dict.fromkeys(ev.source_id))

        for rv in clean_relations.values():
            rv.source_id = list(dict.fromkeys(rv.source_id))
            rv.keywords = list(dict.fromkeys(rv.keywords))
        
        for ek in clean_entities.keys():
            entity = clean_entities[ek]
            self.entity_kv.set(ek, entity)
            self.entity_vidx.add(ek, await self.embed_func(ek+" "+entity.description))
            self.graph.add_node(entity)

        for rk in clean_relations.keys():
            relation = clean_relations[rk]
            self.relation_kv.set(rk, relation)
            self.relation_vidx.add(rk, await self.embed_func(" ".join(relation.keywords)+" "+relation.description))
            self.graph.add_edge(relation)

        for i in range(len(chunks)):
            key = f"{file_id}_chunk_{i+1}"
            self.chunk_kv.set(key, {"text": chunks[i], "file_id": file_id})
            self.chunk_vidx.add(key, await self.embed_func(chunks[i]))

        self._save_all()

        
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