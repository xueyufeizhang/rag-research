import os
from dotenv import load_dotenv
from core import LightRAG
from backend import llm_func, embed_func

load_dotenv()
CON_NUM = os.getenv("CON_NUM", 4)

async def main():
    lightrag = LightRAG(os.getenv("WORKING_DIR", "./dickens"), llm_func, CON_NUM, embed_func)
    with open("./carol.txt", "r", encoding="utf-8")as f: 
        await lightrag.construct(f.read(), "carol")
    answer =  await lightrag.retrieve("Who is Scrooge?", mode="naive")
    print("-----Answer-----")
    print(answer)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())