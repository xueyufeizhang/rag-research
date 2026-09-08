import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from scripts import build_multihop_index as runner
from rag_research.chunking import ChunkConfig
from rag_research.models import BuildResult, InputDocument


def fake_rag():
    return SimpleNamespace(
        construct=AsyncMock(return_value=BuildResult(
            document_count=1, chunk_count=1, entity_count=0, relation_count=0,
            failed_chunk_ids=[], build_fingerprint="build",
            chunking_fingerprint="chunking", extraction_fingerprint="extraction",
        )),
        config=SimpleNamespace(
            chunk_config=ChunkConfig(), embedding_batch_size=32,
            embedding_concurrency=2,
        ),
        last_build_reused=False,
        last_extraction_statistics=None,
    )


class MultiHopBuilderTests(unittest.IsolatedAsyncioTestCase):
    def test_subset_requires_explicit_output_to_avoid_using_full_index_directory(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            runner.parse_args(["--corpus", "sample.json"])
        self.assertEqual(raised.exception.code, 2)

    async def test_subset_build_preserves_source_without_needing_question_file(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus_path = Path(directory, "sample.json")
            output_path = Path(directory, "subset-index")
            record = {
                "url": "https://example.test/doc", "body": "  Original text.\n",
                "title": "Title", "author": "Author", "source": "Source",
                "published_at": "2024-01-01", "category": "Test",
            }
            corpus_path.write_text(json.dumps([record]), encoding="utf-8")
            rag = fake_rag()
            output = io.StringIO()
            with (
                patch.object(runner, "LightRAG", return_value=rag) as constructor,
                patch.object(runner, "load_multihop_rag", side_effect=AssertionError(
                    "subset builds must not load full-dataset questions"
                )),
                redirect_stdout(output),
            ):
                await runner.main([
                    "--corpus", str(corpus_path), "--working-dir", str(output_path),
                ])

            documents = rag.construct.await_args.args[0]
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].text, record["body"])
            self.assertEqual(documents[0].document_id, record["url"])
            self.assertEqual(documents[0].metadata["author"], record["author"])
            self.assertEqual(constructor.call_args.kwargs["working_dir"], output_path)
            report = json.loads(output.getvalue().split("\n", 1)[1])
            self.assertIsNone(report["dataset"]["question_count"])
            self.assertIsNone(report["dataset"]["questions_sha256"])
            self.assertEqual(report["dataset"]["corpus_sha256"], hashlib.sha256(
                corpus_path.read_bytes()
            ).hexdigest())

    async def test_default_build_still_loads_full_dataset_and_reports_questions(self):
        documents = (InputDocument(document_id="doc", text="Source text."),)
        dataset = SimpleNamespace(
            documents=documents, questions=(object(), object()),
            corpus_sha256="corpus-hash", questions_sha256="questions-hash",
        )
        rag = fake_rag()
        output = io.StringIO()
        with (
            patch.object(runner, "load_multihop_rag", return_value=dataset),
            patch.object(runner, "LightRAG", return_value=rag) as constructor,
            redirect_stdout(output),
        ):
            await runner.main([])

        rag.construct.assert_awaited_once_with(documents)
        self.assertEqual(constructor.call_args.kwargs["working_dir"], runner.working_directory)
        report = json.loads(output.getvalue().split("\n", 1)[1])
        self.assertEqual(report["dataset"]["question_count"], 2)
        self.assertEqual(report["dataset"]["questions_sha256"], "questions-hash")


if __name__ == "__main__":
    unittest.main()
