import os
from collections.abc import Sequence

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from rag_research.embedding import EmbeddingInputTooLongError
from rag_research.llm import LLMResponse, truncation_from_finish_reason

load_dotenv()
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")    # "ollama" | "api"
CON_NUM = os.getenv("CON_NUM", 4)    # Concurrency number

API_BASE_URL = os.getenv("API_BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")
API_MODEL = os.getenv("API_MODEL", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
OLLAMA_MODEL = os.getenv("LLM_MODEL", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "")
EXTRACTION_TIMEOUT = int(os.getenv("EXTRACTION_TIMEOUT", 600))

RERANK_MODEL = os.getenv("RERANK_MODEL", "mixedbread-ai/mxbai-rerank-base-v1")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "true").strip().lower() == "true"

api_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY) if LLM_BACKEND == "api" else None


def create_reranker():
    if not ENABLE_RERANKER:
        return None

    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, cache_folder="./models")


async def ollama_llm(system: str, prompt: str) -> LLMResponse:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "system": system,
                "prompt": prompt,
                "stream": False,
            },
            timeout=EXTRACTION_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        reason = payload.get("done_reason")
        usage = {
            key: payload[key]
            for key in (
                "prompt_eval_count", "eval_count", "total_duration",
                "load_duration", "prompt_eval_duration", "eval_duration",
            )
            if key in payload
        }
        return LLMResponse(
            text=payload["response"],
            finish_reason=reason,
            usage=usage or None,
            truncated=truncation_from_finish_reason(reason),
        )
     
async def api_llm(system: str, prompt: str) -> LLMResponse:
    extra_body = {"thinking": {"type": "disabled"}}
    if "openrouter.ai" in API_BASE_URL:
        extra_body["reasoning"] = {"enabled": False}

    resp = await api_client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        timeout=EXTRACTION_TIMEOUT,
        extra_body=extra_body,
    )

    choice = resp.choices[0]
    content = choice.message.content
    usage = resp.usage.model_dump(mode="json") if resp.usage is not None else None
    return LLMResponse(
        text=content if content is not None else "",
        finish_reason=choice.finish_reason,
        usage=usage,
        truncated=truncation_from_finish_reason(choice.finish_reason),
    )

llm_func = api_llm if LLM_BACKEND == "api" else ollama_llm


async def embed_many_func(texts: Sequence[str]) -> list[list[float]]:
    if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
        raise TypeError("embedding input must be a sequence of strings")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("embedding input must contain only strings")
    if not texts:
        return []

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": list(texts),
                # Never allow the server's default silent truncation. The
                # stored chunk, extraction input, and vector must represent
                # exactly the same text.
                "truncate": False,
            },
            timeout=EXTRACTION_TIMEOUT
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = response.text.lower()
            if response.status_code == 400 and any(
                marker in detail
                for marker in (
                    "context length",
                    "input length",
                    "too long",
                    "truncate",
                )
            ):
                raise EmbeddingInputTooLongError(
                    "Ollama rejected an embedding input; truncation is disabled "
                    "and the input exceeds the model context or request limits"
                ) from error
            raise
        payload = response.json()

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list):
        raise ValueError("embedding response does not contain an embeddings list")
    if len(embeddings) != len(texts):
        raise ValueError(
            "embedding response returned "
            f"{len(embeddings)} vectors for {len(texts)} texts"
        )
    return embeddings


async def embed_func(text: str) -> list[float]:
    return (await embed_many_func([text]))[0]
