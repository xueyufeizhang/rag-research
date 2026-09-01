import os
from collections.abc import Sequence

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

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


async def ollama_llm(system: str, prompt: str) -> str:
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
        return response.json()["response"]
     
async def api_llm(system: str, prompt: str) -> str:
    resp = await api_client.chat.completions.create(
        model=API_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        timeout=EXTRACTION_TIMEOUT,
        # extra_body={"thinking": {"type": "disabled"}},
        extra_body={"thinking": {"type": "disabled"}, "reasoning": {"enabled": False}},
    )

    choice = resp.choices[0]
    content = choice.message.content

    if not content:
        print("[api_llm] empty content", flush=True)
        print(f"[api_llm] finish_reason: {choice.finish_reason}", flush=True)
        print(f"[api_llm] usage: {resp.usage}", flush=True)
        print(f"[api_llm] message: {choice.message}", flush=True)

    return content

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
            json={"model": EMBED_MODEL, "input": list(texts)},
            timeout=EXTRACTION_TIMEOUT
        )
        response.raise_for_status()
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
