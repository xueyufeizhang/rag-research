import os
from pathlib import Path
import time
import json
from dataclasses import asdict
from dotenv import load_dotenv
from rag_research.core import LightRAG
from rag_research.datasets.multihop_rag import load_multihop_rag
from rag_research.backends import embed_func, embed_many_func, llm_func

load_dotenv()
CON_NUM = int(os.getenv("CON_NUM", 4))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
dataset_directory = Path(
    os.getenv("MULTIHOP_DATASET_DIR", PROJECT_ROOT / "data/raw/MultiHopRAG")
)
working_directory = Path(
    os.getenv("WORKING_DIR", PROJECT_ROOT / "artifacts/stores/multihop_rag_fixed")
)

async def main() -> None:
    started_at = time.perf_counter()

    dataset = load_multihop_rag(dataset_directory)
    rag = LightRAG(
        working_dir=working_directory,
        llm_func=llm_func,
        con_num=CON_NUM,
        embed_func=embed_func,
        embed_many_func=embed_many_func,
        reranker=None,
    )
    result = await rag.construct(dataset.documents)

    report = {
        "status": "incomplete" if result.failed_chunk_ids else "complete",
        "dataset": {
            "document_count": len(dataset.documents),
            "question_count": len(dataset.questions),
            "corpus_sha256": dataset.corpus_sha256,
            "questions_sha256": dataset.questions_sha256,
        },
        "build": asdict(result),
        "chunk_config": asdict(rag.config.chunk_config),
        "embedding_batch_size": rag.config.embedding_batch_size,
        "embedding_concurrency": rag.config.embedding_concurrency,
        "elapsed_seconds": time.perf_counter() - started_at,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if result.failed_chunk_ids:
        raise RuntimeError(
            f"incomplete build: {len(result.failed_chunk_ids)} chunks failed"
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
