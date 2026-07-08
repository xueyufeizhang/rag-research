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
    chunk_size: int = int(os.getenv("CHUNK_SIZE"))
    chunk_overlap_size: int = int(os.getenv("CHUNK_OVERLAP_SIZE"))
    chunk_top_k: int = int(os.getenv("CHUNK_TOP_K"))


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
            pass
        if relations:
            pass
        if chunks:
            lines = "\n".join(json.dumps(
                {"content": c["text"]}, ensure_ascii=False) for c in chunks)
            parts.append("-----Chunks-----\n" + lines)
        return "\n\n".join(parts)
    
    async def _naive_retrieve(self, query: str) -> list[dict]:
        emb = await self.embed_func(query)
        hits = self.chunk_vidx.query(emb, self.config.chunk_top_k)   # [(chunk_key, score)]
        return [self.chunk_kv.get(k) for k, _ in hits if self.chunk_kv.get(k)]


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
        2. Local
        3. Global
        4. Hybrid
        """
        if mode == "naive":
            chunks = await self._naive_retrieve(query)
            context = self._build_context([], [], chunks)
            print(f"\n[retrieved {len(chunks)} chunks]\n{context[:500]}\n---\n")
        else:
            NotImplementedError(mode)
        system_prompt = PROMPTS["naive_rag_response"].format(content_data=context, response_type="Multiple Paragraphs")
        return await self.llm_func(system=system_prompt, prompt=query)