import os
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

api_client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY) if LLM_BACKEND == "api" else None


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
    )
    return resp.choices[0].message.content

llm_func = api_llm if LLM_BACKEND == "api" else ollama_llm

async def embed_func(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=EXTRACTION_TIMEOUT
        )
        return response.json()["embeddings"][0]