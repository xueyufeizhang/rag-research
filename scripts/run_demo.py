import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
from rag_research.core import LightRAG
from rag_research.backends import llm_func, embed_func, reranker

load_dotenv()
CON_NUM = os.getenv("CON_NUM", 4)

async def main():
    lightrag = LightRAG(
        working_dir=os.getenv("WORKING_DIR", "./artifacts/stores/dickens_fixed_size"),
        llm_func=llm_func,
        con_num=CON_NUM,
        embed_func=embed_func,
        reranker=reranker,
    )
    with open("./data/raw/a_christmas_carol.txt", "r", encoding="utf-8")as f:
        await lightrag.construct(f.read(), "carol")
    answer =  await lightrag.retrieve("Who is Scrooge?", mode="hybrid")
    print("-----Answer-----")
    print(answer)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
