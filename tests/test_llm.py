import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rag_research.llm import (
    LLMResponse,
    normalize_llm_response,
    truncation_from_finish_reason,
)


class LLMResponseTests(unittest.TestCase):
    def test_strings_preserve_unknown_status_and_structured_responses_keep_metadata(self):
        plain = normalize_llm_response('{"entities": [], "relationships": []}')
        self.assertIsNone(plain.truncated)
        self.assertIsNone(plain.finish_reason)
        self.assertIsNone(plain.usage)
        structured = LLMResponse("partial", "length", {"completion_tokens": 4096}, True)
        self.assertIs(normalize_llm_response(structured), structured)

    def test_only_explicit_stop_and_length_signals_decide_truncation(self):
        self.assertTrue(truncation_from_finish_reason("length"))
        self.assertFalse(truncation_from_finish_reason("stop"))
        for reason in (None, "", "content_filter", "tool_calls", "unexpected"):
            with self.subTest(reason=reason):
                self.assertIsNone(truncation_from_finish_reason(reason))

    def test_invalid_response_types_fail_explicitly(self):
        for value in (None, {}, 42):
            with self.subTest(value=value), self.assertRaises(TypeError):
                normalize_llm_response(value)
        with self.assertRaises(TypeError):
            LLMResponse("text", truncated=0)


class LLMBackendMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_api_preserves_finish_reason_and_usage_even_for_empty_content(self):
        from rag_research import backends

        for reason, expected in (("length", True), ("stop", False), ("content_filter", None)):
            with self.subTest(reason=reason):
                completion = SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content=None),
                        finish_reason=reason,
                    )],
                    usage=Mock(model_dump=Mock(return_value={"completion_tokens": 4096})),
                )
                create = AsyncMock(return_value=completion)
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                with patch.object(backends, "api_client", client):
                    result = await backends.api_llm("system", "prompt")
                self.assertEqual(result.text, "")
                self.assertEqual(result.finish_reason, reason)
                self.assertIs(result.truncated, expected)
                self.assertEqual(result.usage, {"completion_tokens": 4096})
                create.assert_awaited_once()

    async def test_ollama_preserves_done_reason_and_token_counts(self):
        from rag_research import backends

        for reason, expected in (("length", True), ("stop", False), (None, None)):
            with self.subTest(reason=reason):
                response = Mock()
                response.json.return_value = {
                    "response": "model text", "done_reason": reason,
                    "prompt_eval_count": 123, "eval_count": 45,
                }
                client = AsyncMock()
                client.post.return_value = response
                context = AsyncMock()
                context.__aenter__.return_value = client
                with patch.object(backends.httpx, "AsyncClient", return_value=context):
                    result = await backends.ollama_llm("system", "prompt")
                self.assertEqual(result.text, "model text")
                self.assertEqual(result.finish_reason, reason)
                self.assertIs(result.truncated, expected)
                self.assertEqual(result.usage, {"prompt_eval_count": 123, "eval_count": 45})
                response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
