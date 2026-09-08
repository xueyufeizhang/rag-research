import argparse
import hashlib
import os
from pathlib import Path
import time
import json
from dataclasses import asdict
from dotenv import load_dotenv
from rag_research.core import LightRAG
from rag_research.datasets.multihop_rag import load_multihop_documents, load_multihop_rag

load_dotenv()
CON_NUM = int(os.getenv("CON_NUM", 4))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
dataset_directory = Path(
    os.getenv("MULTIHOP_DATASET_DIR", PROJECT_ROOT / "data/raw/MultiHopRAG")
)
working_directory = Path(
    os.getenv("WORKING_DIR", PROJECT_ROOT / "artifacts/stores/multihop_rag_fixed")
)
cache_directory = Path(
    os.getenv("CACHE_DIR", PROJECT_ROOT / "artifacts/cache/multihop_rag")
)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a MultiHop-RAG corpus index.")
    parser.add_argument(
        "--corpus", type=Path,
        help="Build only this corpus JSON, without loading the question set.",
    )
    parser.add_argument(
        "--working-dir", type=Path,
        help="Index output directory (overrides WORKING_DIR).",
    )
    args = parser.parse_args(argv)
    if args.corpus is not None and args.working_dir is None:
        parser.error("--corpus requires --working-dir to keep subset indexes separate")
    return args


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started_at = time.perf_counter()

    if args.corpus is not None:
        corpus_path = args.corpus.resolve()
        documents = load_multihop_documents(corpus_path)
        dataset_report = {
            "mode": "corpus_only",
            "corpus_path": str(corpus_path),
            "document_count": len(documents),
            "question_count": None,
            "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "questions_sha256": None,
        }
    else:
        dataset = load_multihop_rag(dataset_directory)
        documents = dataset.documents
        dataset_report = {
            "mode": "full_dataset",
            "corpus_path": str((dataset_directory / "corpus.json").resolve()),
            "document_count": len(documents),
            "question_count": len(dataset.questions),
            "corpus_sha256": dataset.corpus_sha256,
            "questions_sha256": dataset.questions_sha256,
        }
    output_directory = args.working_dir or working_directory
    print(f"[build] {len(documents)} documents | output {output_directory}", flush=True)

    from rag_research.backends import embed_func, embed_many_func, llm_func

    rag = LightRAG(
        working_dir=output_directory,
        llm_func=llm_func,
        con_num=CON_NUM,
        embed_func=embed_func,
        embed_many_func=embed_many_func,
        reranker=None,
        cache_directory=cache_directory,
    )
    result = await rag.construct(documents)

    report = {
        "status": "incomplete" if result.failed_chunk_ids else "complete",
        "dataset": dataset_report,
        "working_directory": str(output_directory),
        "build": asdict(result),
        "reused_completed_index": rag.last_build_reused,
        "extraction_this_invocation": (
            rag.last_extraction_statistics.get("run_summary")
            if rag.last_extraction_statistics is not None
            else None
        ),
        "chunk_config": rag.config.chunk_config.fingerprint_dict(),
        "embedding_batch_size": rag.config.embedding_batch_size,
        "embedding_concurrency": rag.config.embedding_concurrency,
        "cache_directory": str(cache_directory),
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
