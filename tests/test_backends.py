import unittest
from unittest.mock import patch

import httpx

from rag_research import backends
from rag_research.embedding import EmbeddingInputTooLongError


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {
            "embeddings": [[1.0, 0.0]],
        }
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://test/api/embed")
            raise httpx.HTTPStatusError(
                "request failed",
                request=request,
                response=httpx.Response(
                    self.status_code,
                    request=request,
                    text=self.text,
                ),
            )

    def json(self):
        return self._payload


class _FakeAsyncClient:
    response = _FakeResponse()
    posts: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def post(self, _url, *, json, timeout):
        self.posts.append({"json": json, "timeout": timeout})
        return self.response


class OllamaEmbeddingBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_request_disables_server_truncation(self):
        client = _FakeAsyncClient()
        client.posts = []
        client.response = _FakeResponse(
            payload={"embeddings": [[1.0, 0.0], [0.0, 1.0]]},
        )
        with patch.object(backends.httpx, "AsyncClient", return_value=client):
            vectors = await backends.embed_many_func(["first", "second"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(client.posts[0]["json"]["truncate"], False)
        self.assertEqual(client.posts[0]["json"]["input"], ["first", "second"])

    async def test_context_rejection_is_reported_as_overflow(self):
        client = _FakeAsyncClient()
        client.posts = []
        client.response = _FakeResponse(
            status_code=400,
            text="input length exceeds maximum context length",
        )
        with patch.object(backends.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(EmbeddingInputTooLongError):
                await backends.embed_many_func(["too long"])


if __name__ == "__main__":
    unittest.main()
