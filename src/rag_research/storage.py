import dataclasses
import json
import os
import tempfile
from typing import Any

import networkx as nx
import numpy as np


def _atomic_write_json(file_path: str, data: Any) -> None:
    directory = os.path.dirname(file_path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(file_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, file_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _atomic_save_numpy(file_path: str, array: np.ndarray) -> None:
    directory = os.path.dirname(file_path) or "."
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory,
        prefix=f".{os.path.basename(file_path)}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as file:
            np.save(file, array, allow_pickle=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, file_path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


class KVStore:
    def __init__(self, file_path: str):
        self._store = {}
        self.file_path = file_path

    def get(self, key: str):
        return self._store.get(key, None)

    def set(self, key: str, value: Any):
        if not isinstance(key, str) or not key.strip():
            raise ValueError("KV store key must be a non-empty string")
        self._store[key] = (
            dataclasses.asdict(value)
            if dataclasses.is_dataclass(value)
            else value
        )

    def all(self):
        return self._store

    def save(self):
        _atomic_write_json(self.file_path, self._store)

    def load(self):
        self._store = {}
        if not os.path.exists(self.file_path):
            return

        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"invalid KV store file: {self.file_path}")
        self._store = data


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
        _atomic_write_json(self.file_path, data)

    def load(self):
        self._graph = nx.Graph()
        if not os.path.exists(self.file_path):
            return

        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        self._graph = nx.node_link_graph(data)


class VectorIndex:
    def __init__(self, file_path: str):
        self._ids: list[str] = []
        self._id_to_index: dict[str, int] = {}
        self._vectors: np.ndarray | None = None
        self._pending: list[np.ndarray] = []
        self.vector_path = file_path + ".npy"
        self.id_path = file_path + ".json"

    def add(self, key: str, vector: list[float]):
        if not isinstance(key, str) or not key.strip():
            raise ValueError("vector key must be a non-empty string")
        if key in self._id_to_index:
            raise ValueError(f"duplicate vector ID: {key}")

        try:
            array = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError(f"vector {key} must contain numeric values") from error

        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"vector {key} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"vector {key} contains NaN or infinity")
        vector_norm = np.linalg.norm(array.astype(np.float64))
        if not np.isfinite(vector_norm) or vector_norm == 0:
            raise ValueError(f"vector {key} must not be a zero vector")

        expected_dimension: int | None = None
        if self._vectors is not None:
            expected_dimension = self._vectors.shape[1]
        elif self._pending:
            expected_dimension = self._pending[0].shape[0]
        if expected_dimension is not None and array.shape[0] != expected_dimension:
            raise ValueError(
                f"vector {key} has dimension {array.shape[0]}, "
                f"expected {expected_dimension}"
            )

        self._id_to_index[key] = len(self._ids)
        self._ids.append(key)
        self._pending.append(array)

    def _build(self):
        if self._pending:
            new = np.stack(self._pending).astype(np.float32, copy=False)
            self._vectors = new if self._vectors is None else np.vstack([self._vectors, new])
            self._pending = []

    def get_vector(self, key: str):
        self._build()
        index = self._id_to_index.get(key)
        if index is None or self._vectors is None:
            return None
        return self._vectors[index].copy()

    def query(self, query: list[float], top_k: int):
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        self._build()
        if self._vectors is None:
            return []

        try:
            np_query = np.asarray(query, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("query vector must contain numeric values") from error

        if np_query.ndim != 1 or np_query.size == 0:
            raise ValueError("query vector must be a non-empty one-dimensional array")
        if np_query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"query dimension {np_query.shape[0]} does not match "
                f"index dimension {self._vectors.shape[1]}"
            )
        if not np.all(np.isfinite(np_query)):
            raise ValueError("query vector contains NaN or infinity")

        query_norm = np.linalg.norm(np_query.astype(np.float64))
        if not np.isfinite(query_norm) or query_norm == 0:
            raise ValueError("query vector must not be zero")

        matrix_norms = np.linalg.norm(
            self._vectors.astype(np.float64),
            axis=1,
            keepdims=True,
        )
        if np.any(~np.isfinite(matrix_norms)) or np.any(matrix_norms == 0):
            raise RuntimeError("vector index contains zero vectors")

        normalized_query = np_query / query_norm
        normalized_vectors = self._vectors / matrix_norms
        scores = normalized_vectors @ normalized_query
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._ids[i], float(scores[i])) for i in top_idx]

    def save(self):
        self._build()

        if self._vectors is None:
            if self._ids:
                raise RuntimeError("vector index is inconsistent: IDs exist but vectors are missing")
            _atomic_write_json(self.id_path, [])
            if os.path.exists(self.vector_path):
                os.remove(self.vector_path)
            return

        if len(self._ids) != len(self._vectors):
            raise RuntimeError(
                "vector index is inconsistent: "
                f"{len(self._ids)} IDs but {len(self._vectors)} vectors"
            )

        _atomic_save_numpy(self.vector_path, self._vectors)
        _atomic_write_json(self.id_path, self._ids)

    def load(self):
        self._ids = []
        self._id_to_index = {}
        self._vectors = None
        self._pending = []

        id_exists = os.path.exists(self.id_path)
        vector_exists = os.path.exists(self.vector_path)

        if id_exists:
            with open(self.id_path, "r", encoding="utf-8") as file:
                self._ids = json.load(file)

            if not isinstance(self._ids, list):
                raise ValueError(f"invalid vector index ID file: {self.id_path}")
            if any(not isinstance(key, str) or not key.strip() for key in self._ids):
                raise ValueError(f"vector index contains invalid IDs: {self.id_path}")
            if len(self._ids) != len(set(self._ids)):
                raise ValueError(f"vector index contains duplicate IDs: {self.id_path}")

        if vector_exists:
            self._vectors = np.load(
                self.vector_path,
                allow_pickle=False,
            )

        if self._ids and self._vectors is None:
            raise RuntimeError(
                "vector index is incomplete: "
                f"{len(self._ids)} IDs exist but vector file is missing"
            )

        if not self._ids and self._vectors is not None:
            raise RuntimeError(
                "vector index is incomplete: "
                "vector file exists but IDs are missing"
            )

        if (
            self._vectors is not None
            and len(self._ids) != len(self._vectors)
        ):
            raise RuntimeError(
                "vector index is inconsistent: "
                f"{len(self._ids)} IDs but "
                f"{len(self._vectors)} vectors"
            )

        if self._vectors is not None:
            if self._vectors.ndim != 2 or self._vectors.shape[1] == 0:
                raise ValueError("vector matrix must be a non-empty two-dimensional matrix")
            if not np.issubdtype(self._vectors.dtype, np.number):
                raise ValueError("vector matrix must contain numeric values")
            self._vectors = self._vectors.astype(np.float32, copy=False)
            if not np.all(np.isfinite(self._vectors)):
                raise ValueError("vector matrix contains NaN or infinity")
            vector_norms = np.linalg.norm(
                self._vectors.astype(np.float64),
                axis=1,
            )
            if np.any(~np.isfinite(vector_norms)) or np.any(vector_norms == 0):
                raise ValueError("vector matrix contains zero vectors")

        self._id_to_index = {
            key: index
            for index, key in enumerate(self._ids)
        }
