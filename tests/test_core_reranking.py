import asyncio
import tempfile
import unittest

from rag_research.core import LightRAG, LightRAGConfig


class FakeReranker:
    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.seen_pairs = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen_pairs = pairs
        return [self.scores_by_text[text] for _, text in pairs]


class RerankingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.reranker = FakeReranker({
            "dense first": 0.1,
            "cross encoder first": 0.9,
            "outside dense shortlist": 1.0,
            "source: A; target: B; keywords: first; description: dense first": 0.1,
            "source: C; target: D; keywords: second; description: cross encoder first": 0.9,
        })
        config = LightRAGConfig(
            chunk_top_k=2,
            chunk_candidate_top_k=2,
            relation_top_k=2,
        )
        self.rag = LightRAG(
            working_dir=self.temp_dir.name,
            llm_func=None,
            con_num=1,
            embed_func=None,
            config=config,
            reranker=self.reranker,
        )
        self.rag.chunk_vidx.add("c1", [1.0, 0.0])
        self.rag.chunk_vidx.add("c2", [0.8, 0.6])
        self.rag.chunk_vidx.add("c3", [0.0, 1.0])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_dense_shortlist_is_cross_encoder_reranked(self):
        chunks = [
            {"chunk_id": "c1", "text": "dense first"},
            {"chunk_id": "c2", "text": "cross encoder first"},
            {"chunk_id": "c3", "text": "outside dense shortlist"},
        ]

        shortlist = self.rag._dense_filter_chunks([1.0, 0.0], chunks)
        ranked = self.rag._rerank_chunks("query", shortlist)

        self.assertEqual([chunk["chunk_id"] for chunk in ranked], ["c2", "c1"])
        self.assertEqual(
            self.reranker.seen_pairs,
            [
                ("query", "dense first"),
                ("query", "cross encoder first"),
            ],
        )

    def test_empty_candidates_do_not_call_reranker(self):
        ranked = self.rag._rerank_chunks("query", [])

        self.assertEqual(ranked, [])
        self.assertEqual(self.reranker.seen_pairs, [])

    def test_chunks_keep_dense_order_without_reranker(self):
        self.rag.reranker = None
        chunks = [
            {"chunk_id": "c1", "text": "dense first"},
            {"chunk_id": "c2", "text": "cross encoder first"},
            {"chunk_id": "c3", "text": "outside dense shortlist"},
        ]

        shortlist = self.rag._dense_filter_chunks([1.0, 0.0], chunks)
        ranked = self.rag._rerank_chunks("query", shortlist)

        self.assertEqual([chunk["chunk_id"] for chunk in ranked], ["c1", "c2"])
        self.assertEqual(self.reranker.seen_pairs, [])

    def test_relations_are_cross_encoder_reranked(self):
        relations = [
            {
                "source": "A",
                "target": "B",
                "keywords": ["first"],
                "description": "dense first",
            },
            {
                "source": "C",
                "target": "D",
                "keywords": ["second"],
                "description": "cross encoder first",
            },
        ]

        ranked = self.rag._rerank_relations("query", relations)

        self.assertEqual([relation["source"] for relation in ranked], ["C", "A"])

    def test_relations_keep_dense_order_without_reranker(self):
        self.rag.reranker = None
        relations = [
            {"source": "A", "target": "B"},
            {"source": "C", "target": "D"},
            {"source": "E", "target": "F"},
        ]

        ranked = self.rag._rerank_relations("query", relations)

        self.assertEqual([relation["source"] for relation in ranked], ["A", "C"])

    def test_reranker_is_optional_at_construction(self):
        rag = LightRAG(
            working_dir=self.temp_dir.name,
            llm_func=None,
            con_num=1,
            embed_func=None,
            config=LightRAGConfig(),
            reranker=None,
        )

        self.assertIsNone(rag.reranker)

    def test_naive_retrieval_uses_cross_encoder_after_dense_retrieval(self):
        async def embed_func(_: str) -> list[float]:
            return [1.0, 0.0]

        self.rag.embed_func = embed_func
        self.rag.chunk_kv.set("c1", {"text": "dense first"})
        self.rag.chunk_kv.set("c2", {"text": "cross encoder first"})
        self.rag.chunk_kv.set("c3", {"text": "outside dense shortlist"})

        ranked = asyncio.run(self.rag._naive_retrieve("query"))

        self.assertEqual([chunk["chunk_id"] for chunk in ranked], ["c2", "c1"])
        self.assertIn("dense_score", ranked[0])

    def test_naive_retrieval_keeps_dense_order_without_reranker(self):
        async def embed_func(_: str) -> list[float]:
            return [1.0, 0.0]

        self.rag.embed_func = embed_func
        self.rag.reranker = None
        self.rag.chunk_kv.set("c1", {"text": "dense first"})
        self.rag.chunk_kv.set("c2", {"text": "cross encoder first"})
        self.rag.chunk_kv.set("c3", {"text": "outside dense shortlist"})

        ranked = asyncio.run(self.rag._naive_retrieve("query"))

        self.assertEqual([chunk["chunk_id"] for chunk in ranked], ["c1", "c2"])

    def test_trace_top_k_expands_candidate_pool_and_preserves_scores(self):
        async def embed_func(_: str) -> list[float]:
            return [1.0, 0.0]

        self.rag.embed_func = embed_func
        self.rag.chunk_kv.set("c1", {"text": "dense first"})
        self.rag.chunk_kv.set("c2", {"text": "cross encoder first"})
        self.rag.chunk_kv.set("c3", {"text": "outside dense shortlist"})

        trace = asyncio.run(
            self.rag.retrieve_trace("query", mode="naive", top_k=3)
        )

        self.assertEqual(trace["requested_top_k"], 3)
        self.assertEqual(trace["chunk_ids"], ["c3", "c2", "c1"])
        self.assertTrue(all("dense_score" in chunk for chunk in trace["chunks"]))
        self.assertTrue(all("rerank_score" in chunk for chunk in trace["chunks"]))
        self.assertTrue(all(
            chunk["retrieval_sources"] == ["naive"]
            for chunk in trace["chunks"]
        ))

    def test_trace_rejects_non_positive_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            asyncio.run(self.rag.retrieve_trace("query", mode="naive", top_k=0))


if __name__ == "__main__":
    unittest.main()
