import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import networkx as nx

from rag_research.chunking import ChunkConfig
from rag_research.core import LightRAG, LightRAGConfig
from rag_research.models import InputDocument


def _config() -> LightRAGConfig:
    return LightRAGConfig(
        chunk_config=ChunkConfig(
            strategy="fixed",
            fixed_size=100,
            overlap_size=0,
        ),
        llm_backend="test",
        llm_model="test-llm",
        embedding_backend="test",
        embedding_model="test-embedding",
    )


def _documents() -> list[InputDocument]:
    return [InputDocument(
        document_id="integrity-document",
        text="Alpha works with Beta. Beta works with Gamma.",
    )]


def _extraction_response(*, empty: bool = False) -> str:
    return json.dumps({
        "entities": [] if empty else [
            {"name": name, "type": "Person", "description": f"{name} is a worker."}
            for name in ("Alpha", "Beta", "Gamma")
        ],
        "relationships": [] if empty else [
            {
                "source": source,
                "target": target,
                "keywords": ["collaboration"],
                "description": f"{source} works with {target}.",
            }
            for source, target in (("Alpha", "Beta"), ("Beta", "Gamma"))
        ],
    })


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_graph(directory: Path):
    return nx.node_link_graph(_read_json(directory / "graph.json"))


def _write_graph(directory: Path, graph) -> None:
    _write_json(directory / "graph.json", nx.node_link_data(graph))


class IndexIntegrityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.complete = self.directory / "complete"
        self.original_result = await self._build(self.complete)
        self.assertEqual(self.original_result.chunk_count, 1)
        self.assertEqual(self.original_result.entity_count, 3)
        self.assertEqual(self.original_result.relation_count, 2)

    async def _build(self, directory: Path, *, empty: bool = False):
        rag = LightRAG(
            working_dir=str(directory),
            llm_func=AsyncMock(return_value=_extraction_response(empty=empty)),
            con_num=1,
            embed_func=AsyncMock(return_value=[1.0, 0.5]),
            config=_config(),
        )
        return await rag.construct(_documents())

    def _copy(self, name: str, *, source: Path | None = None) -> Path:
        directory = self.directory / name
        shutil.copytree(source or self.complete, directory)
        return directory

    def _reader(self, directory: Path):
        llm = AsyncMock(side_effect=AssertionError("validation must not call the LLM"))
        embed = AsyncMock(side_effect=AssertionError("validation must not embed"))
        return llm, embed, {
            "working_dir": str(directory),
            "llm_func": llm,
            "con_num": 1,
            "embed_func": embed,
            "config": _config(),
        }

    async def _assert_rejected_without_writes(self, directory: Path):
        before = _snapshot(directory)
        llm, embed, arguments = self._reader(directory)
        with self.assertRaises(RuntimeError):
            rag = LightRAG(**arguments)
            await rag.construct(_documents())
        llm.assert_not_awaited()
        embed.assert_not_awaited()
        self.assertEqual(_snapshot(directory), before)

    async def test_complete_nonempty_index_reuses_without_models_or_writes(self):
        before = _snapshot(self.complete)
        llm, embed, arguments = self._reader(self.complete)
        rag = LightRAG(**arguments)

        result = await rag.construct(_documents())

        self.assertEqual(result, self.original_result)
        llm.assert_not_awaited()
        embed.assert_not_awaited()
        self.assertEqual(_snapshot(self.complete), before)

    async def test_empty_entity_and_relation_indexes_need_no_vector_matrix(self):
        directory = self.directory / "empty"
        first_result = await self._build(directory, empty=True)
        for prefix in ("entity", "relation"):
            self.assertEqual(_read_json(directory / f"{prefix}_vectors.json"), [])
            self.assertFalse((directory / f"{prefix}_vectors.npy").exists())
        before = _snapshot(directory)
        llm, embed, arguments = self._reader(directory)

        result = await LightRAG(**arguments).construct(_documents())

        self.assertEqual(result, first_result)
        llm.assert_not_awaited()
        embed.assert_not_awaited()
        self.assertEqual(_snapshot(directory), before)

    async def test_missing_required_nonempty_store_files_are_rejected(self):
        filenames = (
            "graph.json", "entities.json", "relations.json", "chunks.json",
            "entity_vectors.json", "relation_vectors.json", "chunk_vectors.json",
            "entity_vectors.npy", "relation_vectors.npy", "chunk_vectors.npy",
        )
        for filename in filenames:
            with self.subTest(filename=filename):
                directory = self._copy(f"missing-{filename}")
                (directory / filename).unlink()
                await self._assert_rejected_without_writes(directory)

    async def test_missing_vector_id_and_matrix_pair_is_rejected(self):
        for prefix in ("entity", "relation", "chunk"):
            with self.subTest(index=prefix):
                directory = self._copy(f"missing-pair-{prefix}")
                (directory / f"{prefix}_vectors.json").unlink()
                (directory / f"{prefix}_vectors.npy").unlink()
                await self._assert_rejected_without_writes(directory)

    async def test_empty_indexes_still_require_explicit_store_and_id_files(self):
        empty = self.directory / "empty"
        await self._build(empty, empty=True)
        for filename in (
            "entities.json", "relations.json", "graph.json",
            "entity_vectors.json", "relation_vectors.json",
        ):
            with self.subTest(filename=filename):
                directory = self._copy(f"empty-missing-{filename}", source=empty)
                (directory / filename).unlink()
                await self._assert_rejected_without_writes(directory)

    async def test_vector_id_substitution_with_unchanged_count_is_rejected(self):
        for prefix in ("entity", "relation", "chunk"):
            with self.subTest(index=prefix):
                directory = self._copy(f"substituted-id-{prefix}")
                path = directory / f"{prefix}_vectors.json"
                ids = _read_json(path)
                ids[0] = "nonexistent-record"
                _write_json(path, ids)
                await self._assert_rejected_without_writes(directory)

    async def test_missing_extra_or_substituted_graph_nodes_are_rejected(self):
        for mutation in ("missing", "extra", "substituted"):
            with self.subTest(mutation=mutation):
                directory = self._copy(f"graph-node-{mutation}")
                graph = _read_graph(directory)
                if mutation == "missing":
                    graph.remove_node("Alpha")
                elif mutation == "extra":
                    graph.add_node("Unknown", name="Unknown")
                else:
                    graph = nx.relabel_nodes(graph, {"Alpha": "Unknown"})
                _write_graph(directory, graph)
                await self._assert_rejected_without_writes(directory)

    async def test_stale_graph_attributes_with_unchanged_ids_are_rejected(self):
        for record_type in ("entity", "relation"):
            with self.subTest(record_type=record_type):
                directory = self._copy(f"stale-attributes-{record_type}")
                graph = _read_graph(directory)
                if record_type == "entity":
                    graph.nodes["Alpha"]["description"] = "Outdated entity description."
                else:
                    graph.edges["Alpha", "Beta"]["description"] = "Outdated relation description."
                _write_graph(directory, graph)
                await self._assert_rejected_without_writes(directory)

    async def test_missing_extra_or_substituted_graph_edges_are_rejected(self):
        for mutation in ("missing", "extra", "substituted"):
            with self.subTest(mutation=mutation):
                directory = self._copy(f"graph-edge-{mutation}")
                graph = _read_graph(directory)
                if mutation in ("missing", "substituted"):
                    graph.remove_edge("Alpha", "Beta")
                if mutation in ("extra", "substituted"):
                    graph.add_edge("Alpha", "Gamma")
                _write_graph(directory, graph)
                await self._assert_rejected_without_writes(directory)

    async def test_dangling_entity_and_relation_source_ids_are_rejected(self):
        for filename in ("entities.json", "relations.json"):
            with self.subTest(store=filename):
                directory = self._copy(f"dangling-source-{filename}")
                path = directory / filename
                records = _read_json(path)
                record = next(iter(records.values()))
                record["source_id"] = ["nonexistent-chunk"]
                _write_json(path, records)
                # Keep duplicated graph attributes in sync so rejection must
                # check the source reference against actual chunk records.
                graph = _read_graph(directory)
                if filename == "entities.json":
                    graph.nodes[record["name"]]["source_id"] = record["source_id"]
                else:
                    graph.edges[record["source"], record["target"]]["source_id"] = record["source_id"]
                _write_graph(directory, graph)
                await self._assert_rejected_without_writes(directory)

    async def test_construct_rechecks_files_removed_after_initialization(self):
        directory = self._copy("deleted-after-load")
        llm, embed, arguments = self._reader(directory)
        rag = LightRAG(**arguments)
        (directory / "graph.json").unlink()
        before = _snapshot(directory)

        with self.assertRaises(RuntimeError):
            await rag.construct(_documents())

        llm.assert_not_awaited()
        embed.assert_not_awaited()
        self.assertEqual(_snapshot(directory), before)

    async def test_construct_rechecks_mutated_in_memory_source_references(self):
        directory = self._copy("mutated-after-load")
        llm, embed, arguments = self._reader(directory)
        rag = LightRAG(**arguments)
        rag.entity_kv.get("Alpha")["source_id"] = ["nonexistent-chunk"]
        before = _snapshot(directory)

        with self.assertRaises(RuntimeError):
            await rag.construct(_documents())

        llm.assert_not_awaited()
        embed.assert_not_awaited()
        self.assertEqual(_snapshot(directory), before)


if __name__ == "__main__":
    unittest.main()
