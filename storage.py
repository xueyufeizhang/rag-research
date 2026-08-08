import dataclasses
from typing import Any
import numpy as np
import os
import json
import networkx as nx

class KVStore:
    def __init__(self, file_path: str):
        self._store = {}
        self.file_path = file_path
    
    def get(self, key: str):
        return self._store.get(key, None)

    def set(self, key: str, value: Any):
        self._store[key] = dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
    
    def all(self):
        return self._store

    def save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.file_path):
          with open(self.file_path, "r", encoding="utf-8") as f:
              self._store = json.load(f)


class GraphStore:
    def __init__(self, file_path: str):
        self._graph = nx.Graph()
        self.file_path = file_path

    def add_node(self, entity: Any):
        entity = dataclasses.asdict(entity) if dataclasses.is_dataclass(entity) else entity
        self._graph.add_node(entity["name"], **entity)

    def add_edge(self, relation: Any):
        relation = dataclasses.asdict(relation) if dataclasses.is_dataclass(relation) else relation
        self._graph.add_edge(relation["source"], relation["target"], **relation)

    def get_node(self, name: str):
        if name in self._graph:
          return self._graph.nodes[name]
        return None

    def get_edge(self, source: str, target: str):
        return self._graph[source][target]

    def get_neighbors(self, name: str):
        return list(self._graph.neighbors(name))

    def save(self):
        data = nx.node_link_data(self._graph)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.file_path):
          with open(self.file_path, "r", encoding="utf-8") as f:
              data = json.load(f)
          self._graph = nx.node_link_graph(data)


class VectorIndex:
    def __init__(self, file_path: str):
        self._ids = []
        self._vectors = None
        self._pending = []
        self.vector_path = file_path+".npy"
        self.id_path = file_path+".json"

    def add(self, key: str, vector: list[float]):
        self._ids.append(key)
        self._pending.append(vector)

    def _build(self):
        if self._pending:
            new = np.array(self._pending, dtype=np.float32)
            self._vectors = new if self._vectors is None else np.vstack([self._vectors, new])
            self._pending = []

    def query(self, query: list[float], top_k: int):
        self._build()
        if self._vectors is not None:
            np_query = np.array(query, dtype=np.float32)
            np_query = np_query / np.linalg.norm(np_query)
            matrices = self._vectors / np.linalg.norm(self._vectors, axis=1, keepdims=True)
            scores = matrices @ np_query
            top_idx = np.argsort(scores)[::-1][:top_k]
            return [(self._ids[i], float(scores[i])) for i in top_idx]
        return []
    
    def save(self):
        self._build()
        with open(self.id_path, "w", encoding="utf-8") as f:
            json.dump(self._ids, f, ensure_ascii=False, indent=2)
        np.save(self.vector_path, self._vectors)

    def load(self):
        if os.path.exists(self.id_path):
            with open(self.id_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._ids = data
        if os.path.exists(self.vector_path):
            self._vectors = np.load(self.vector_path)
        
