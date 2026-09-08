import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from rag_research.chunking import ChunkConfig
from rag_research.core import LightRAG, LightRAGConfig
from rag_research.llm import LLMResponse
from rag_research.models import InputDocument


EMPTY_EXTRACTION = '{"entities": [], "relationships": []}'


class PipelineReportingTests(unittest.IsolatedAsyncioTestCase):
    def make_rag(self, directory, llm):
        return LightRAG(
            working_dir=directory,
            llm_func=llm,
            con_num=1,
            embed_func=AsyncMock(return_value=[1.0, 0.5]),
            config=LightRAGConfig(
                chunk_config=ChunkConfig(strategy="fixed", fixed_size=500, overlap_size=0),
                llm_backend="test", llm_model="test", embedding_model="test",
            ),
        )

    async def test_build_persists_full_report_and_compact_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = self.make_rag(directory, AsyncMock(return_value=LLMResponse(
                text=EMPTY_EXTRACTION, finish_reason="stop", truncated=False,
            )))
            documents = [InputDocument(document_id="doc", text="A short document.")]

            result = await rag.construct(documents)

            report = json.loads(Path(directory, "extraction_report.json").read_text())
            self.assertEqual(report["build_fingerprint"], result.build_fingerprint)
            self.assertEqual(report["extraction_fingerprint"], result.extraction_fingerprint)
            self.assertEqual(report["statistics"], rag.last_extraction_statistics)
            self.assertEqual(len(report["statistics"]["chunks"]), 1)
            self.assertNotIn("chunks", result.extraction_statistics)
            manifest = json.loads(Path(directory, "build_manifest.json").read_text())
            self.assertEqual(manifest["build"]["extraction_statistics"], result.extraction_statistics)
            self.assertFalse(rag.last_build_reused)

            llm = AsyncMock(side_effect=AssertionError("a complete index must be reused"))
            resumed = self.make_rag(directory, llm)
            original_report = Path(directory, "extraction_report.json").read_bytes()
            cached = await resumed.construct(documents)

            self.assertEqual(cached, result)
            self.assertTrue(resumed.last_build_reused)
            self.assertIsNone(resumed.last_extraction_statistics)
            self.assertEqual(Path(directory, "extraction_report.json").read_bytes(), original_report)
            llm.assert_not_awaited()
            resumed.embed_func.assert_not_awaited()

    async def test_failed_extraction_keeps_diagnostics_without_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = self.make_rag(directory, AsyncMock(return_value=LLMResponse(
                text=EMPTY_EXTRACTION, finish_reason="length", truncated=True,
            )))
            with patch("rag_research.extraction.asyncio.sleep", new=AsyncMock()):
                with self.assertRaisesRegex(RuntimeError, "extraction failed"):
                    await rag.construct([InputDocument(document_id="doc", text="Some text.")])

            report = json.loads(Path(directory, "extraction_report.json").read_text())
            attempts = report["statistics"]["chunks"][0]["attempts"]
            self.assertTrue(attempts)
            self.assertTrue(all(attempt["truncated"] is True for attempt in attempts))
            self.assertFalse(Path(directory, "build_manifest.json").exists())
            rag.embed_func.assert_not_awaited()

    async def test_generation_and_context_measurement_share_public_serializer(self):
        with tempfile.TemporaryDirectory() as directory:
            llm = AsyncMock(return_value=LLMResponse(text="The answer.", truncated=False))
            rag = self.make_rag(directory, llm)
            chunks = [{"chunk_id": "c1", "text": "Original evidence."}]
            rag._naive_retrieve = AsyncMock(return_value=chunks)

            answer = await rag.retrieve("Question?", mode="naive")

            self.assertEqual(answer, "The answer.")
            self.assertIn(rag.build_context([], [], chunks), llm.await_args.kwargs["system"])
            self.assertEqual(rag._build_context([], [], chunks), rag.build_context([], [], chunks))

    async def test_generation_rejects_explicitly_truncated_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = self.make_rag(directory, AsyncMock(return_value=LLMResponse(
                text="Incomplete answer", finish_reason="length", truncated=True,
            )))
            rag._naive_retrieve = AsyncMock(return_value=[])

            with self.assertRaisesRegex(RuntimeError, "answer generation was truncated"):
                await rag.retrieve("Question?", mode="naive")

    async def test_legacy_string_model_responses_remain_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            rag = self.make_rag(directory, AsyncMock(return_value="Legacy answer"))
            rag._naive_retrieve = AsyncMock(return_value=[])
            self.assertEqual(await rag.retrieve("Question?", mode="naive"), "Legacy answer")


if __name__ == "__main__":
    unittest.main()
