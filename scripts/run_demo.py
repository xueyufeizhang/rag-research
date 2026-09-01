import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from rag_research.core import LightRAG
from rag_research.backends import create_reranker, embed_func, embed_many_func, llm_func
from rag_research.models import InputDocument

load_dotenv()
CON_NUM = int(os.getenv("CON_NUM", 4))
reranker = create_reranker()

async def main():
    lightrag = LightRAG(
        working_dir=os.getenv("WORKING_DIR", "./artifacts/stores/dickens_fixed"),
        cache_directory=os.getenv(
            "CACHE_DIR",
            "./artifacts/cache/rag_research",
        ),
        llm_func=llm_func,
        con_num=CON_NUM,
        embed_func=embed_func,
        embed_many_func=embed_many_func,
        reranker=reranker,
    )
    with open("./data/raw/a_christmas_carol.txt", "r", encoding="utf-8") as file:
        source = file.read()
    await lightrag.construct((
        InputDocument(
            document_id="carol",
            text=source,
            metadata={"title": "A Christmas Carol"},
        ),
    ))
    answer =  await lightrag.retrieve("Who is Scrooge?", mode="hybrid")
    print("-----Answer-----")
    print(answer)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
