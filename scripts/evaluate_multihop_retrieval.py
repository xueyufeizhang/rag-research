import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer

from rag_research.backends import (
    EMBED_MODEL,
    RERANK_MODEL,
    create_reranker,
    embed_func,
    embed_many_func,
    llm_func,
)
from rag_research.core import LightRAG
from rag_research.datasets.multihop_rag import load_multihop_rag
from rag_research.evaluation import (
    build_multidocument_chunk_index,
    calc_multihop_official_metrics,
    calc_multihop_retrieval_metrics,
    calc_null_retrieval_context_metrics,
    harmonic_mean,
    map_multihop_evidence_to_chunks,
)
from rag_research.models import QuestionRecord


load_dotenv()

EVALUATION_SCHEMA_VERSION = 2
DEFAULT_MODES = ("naive", "local", "global", "hybrid")
DEFAULT_K_VALUES = (1, 3, 5, 10, 20)
OFFICIAL_TOP_K = 10
OFFICIAL_EVALUATOR_URL = (
    "https://github.com/yixuantt/MultiHop-RAG/blob/main/retrieval_evaluate.py"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv(value: str, *, label: str) -> tuple[str, ...]:
    parsed = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not parsed:
        raise ValueError(f"{label} must contain at least one value")
    return parsed


def parse_k_values(value: str) -> tuple[int, ...]:
    raw_values = parse_csv(value, label="EVAL_K_VALUES")
    try:
        values = tuple(sorted({int(item) for item in raw_values}))
    except ValueError as error:
        raise ValueError("EVAL_K_VALUES must contain positive integers") from error
    if any(value <= 0 for value in values):
        raise ValueError("EVAL_K_VALUES must contain positive integers")
    return values


def parse_bool(value: str, *, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{label} must be true or false")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def stable_fingerprint(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compact_trace(trace: dict) -> dict:
    entities = [
        {
            key: entity[key]
            for key in ("name", "type", "dense_score", "rerank_score")
            if key in entity
        }
        for entity in trace.get("entities", [])
    ]
    relations = [
        {
            **{
                key: relation[key]
                for key in (
                    "source",
                    "target",
                    "dense_score",
                    "rerank_score",
                )
                if key in relation
            },
            "relation_id": "||".join(sorted([
                relation.get("source", ""),
                relation.get("target", ""),
            ])),
        }
        for relation in trace.get("relations", [])
    ]
    chunks = [
        {
            "rank": rank,
            **{
                key: chunk[key]
                for key in (
                    "chunk_id",
                    "document_id",
                    "chunk_index",
                    "char_start",
                    "char_end",
                    "dense_score",
                    "rerank_score",
                    "retrieval_sources",
                    "introduced_by",
                )
                if key in chunk
            },
        }
        for rank, chunk in enumerate(trace.get("chunks", []), start=1)
    ]
    return {
        "query": trace.get("query"),
        "mode": trace.get("mode"),
        "requested_top_k": trace.get("requested_top_k"),
        "entity_ids": trace.get("entity_ids", []),
        "relation_ids": trace.get("relation_ids", []),
        "chunk_ids": trace.get("chunk_ids", []),
        "entities": entities,
        "relations": relations,
        "chunks": chunks,
    }


def build_official_retrieval_list(
    trace: dict,
    chunks: dict[str, dict],
) -> list[dict]:
    """Build the top-10 text/score records consumed by the official evaluator."""
    retrieval_list = []
    for retrieved in trace.get("chunks", [])[:OFFICIAL_TOP_K]:
        chunk_id = retrieved.get("chunk_id")
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"official export references an unknown chunk: {chunk_id}")
        model_text = chunk.get("model_text")
        if not isinstance(model_text, str):
            raise ValueError(
                f"official export requires model_text for retrieved chunk: {chunk_id}"
            )
        score = retrieved.get("rerank_score")
        if score is None:
            score = retrieved.get("dense_score", 0.0)
        if score is None:
            score = 0.0
        retrieval_list.append({
            "text": model_text,
            "score": float(score),
        })
    return retrieval_list


def build_official_export(
    results: list[dict],
    *,
    mode: str,
    chunks: dict[str, dict],
) -> list[dict]:
    """Render one mode in the JSON shape accepted by retrieval_evaluate.py."""
    exported = []
    for row in results:
        if row["mode"] != mode:
            continue
        gold_list = []
        for evidence in row["gold_evidence"]:
            metadata = evidence.get("metadata", {})
            gold_list.append({
                **(metadata if isinstance(metadata, dict) else {}),
                "fact": evidence["fact"],
            })
        exported.append({
            "query": row["question"],
            "answer": row["answer"],
            "question_type": row["question_type"],
            "retrieval_list": build_official_retrieval_list(row["trace"], chunks),
            "gold_list": gold_list,
        })
    return exported


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0


def summarize_answerable_group(
    *,
    mode: str,
    group_name: str,
    group_value: str | int,
    rows: list[dict],
    k_values: tuple[int, ...],
) -> list[dict]:
    summaries = []
    for requested_k in k_values:
        metrics = [
            row["thesis_extended"]["metrics_by_k"][str(requested_k)]
            for row in rows
        ]
        retrieved_total = sum(item["retrieved_count"] for item in metrics)
        relevant_total = sum(item["relevant_retrieved_count"] for item in metrics)
        evidence_total = sum(item["gold_evidence_count"] for item in metrics)
        evidence_matched = sum(item["matched_evidence_count"] for item in metrics)
        document_total = sum(item["gold_document_count"] for item in metrics)
        document_matched = sum(item["matched_document_count"] for item in metrics)
        retrieved_chars = sum(item["retrieved_chars"] for item in metrics)
        covered_chars = sum(item["covered_evidence_chars"] for item in metrics)
        micro_precision = relevant_total / retrieved_total if retrieved_total else 0.0
        micro_evidence_recall = evidence_matched / evidence_total if evidence_total else 0.0

        summaries.append({
            "mode": mode,
            "group": group_name,
            "group_value": group_value,
            "question_count": len(rows),
            "requested_k": requested_k,
            "avg_returned_chunks": mean(metrics, "retrieved_count"),
            "macro_chunk_precision": mean(metrics, "chunk_precision"),
            "macro_evidence_recall": mean(metrics, "evidence_recall"),
            "macro_coverage_f1": mean(metrics, "coverage_f1"),
            "joint_evidence_success_rate": mean(metrics, "joint_evidence_success"),
            "macro_document_recall": mean(metrics, "document_recall"),
            "joint_document_success_rate": mean(metrics, "joint_document_success"),
            "mean_reciprocal_rank": mean(metrics, "reciprocal_rank"),
            "mean_average_precision_at_k": mean(metrics, "average_precision_at_k"),
            "mean_ndcg_at_k": mean(metrics, "ndcg_at_k"),
            "micro_chunk_precision": micro_precision,
            "micro_evidence_recall": micro_evidence_recall,
            "micro_coverage_f1": harmonic_mean(micro_precision, micro_evidence_recall),
            "micro_document_recall": (
                document_matched / document_total if document_total else 0.0
            ),
            "avg_retrieved_tokens": mean(metrics, "retrieved_tokens"),
            "avg_retrieved_chars": mean(metrics, "retrieved_chars"),
            "avg_retrieved_document_count": mean(metrics, "retrieved_document_count"),
            "cross_document_retrieval_rate": mean(metrics, "cross_document_retrieval"),
            "evidence_density": (
                covered_chars / retrieved_chars if retrieved_chars else 0.0
            ),
        })
    return summaries


def build_summaries(
    results: list[dict],
    modes: tuple[str, ...],
    k_values: tuple[int, ...],
) -> dict:
    answerable_summaries: list[dict] = []
    null_summaries: list[dict] = []
    official_summaries: list[dict] = []

    for mode in modes:
        mode_rows = [row for row in results if row["mode"] == mode]
        answerable = [row for row in mode_rows if row["question_type"] != "null_query"]
        if answerable:
            answerable_summaries.extend(summarize_answerable_group(
                mode=mode,
                group_name="all_answerable",
                group_value="all",
                rows=answerable,
                k_values=k_values,
            ))
            official_metrics = [row["official"]["metrics"] for row in answerable]
            official_summaries.append({
                "mode": mode,
                "question_count": len(answerable),
                "Hits@10": mean(official_metrics, "Hits@10"),
                "Hits@4": mean(official_metrics, "Hits@4"),
                "MAP@10": mean(official_metrics, "MAP@10"),
                "MRR@10": mean(official_metrics, "MRR@10"),
            })

        for question_type in sorted({row["question_type"] for row in answerable}):
            grouped = [row for row in answerable if row["question_type"] == question_type]
            answerable_summaries.extend(summarize_answerable_group(
                mode=mode,
                group_name="question_type",
                group_value=question_type,
                rows=grouped,
                k_values=k_values,
            ))

        for hop_count in sorted({row["hop_count"] for row in answerable}):
            grouped = [row for row in answerable if row["hop_count"] == hop_count]
            answerable_summaries.extend(summarize_answerable_group(
                mode=mode,
                group_name="hop_count",
                group_value=hop_count,
                rows=grouped,
                k_values=k_values,
            ))

        null_rows = [row for row in mode_rows if row["question_type"] == "null_query"]
        if not null_rows:
            continue
        for requested_k in k_values:
            metrics = [
                row["thesis_extended"]["metrics_by_k"][str(requested_k)]
                for row in null_rows
            ]
            null_summaries.append({
                "mode": mode,
                "question_count": len(null_rows),
                "requested_k": requested_k,
                "avg_returned_chunks": mean(metrics, "retrieved_count"),
                "avg_retrieved_tokens": mean(metrics, "retrieved_tokens"),
                "avg_retrieved_chars": mean(metrics, "retrieved_chars"),
                "avg_retrieved_document_count": mean(metrics, "retrieved_document_count"),
                "cross_document_retrieval_rate": mean(metrics, "cross_document_retrieval"),
                "relevance_metrics_applicable": False,
            })

    return {
        "protocols": {
            "thesis_extended": {
                "evidence_unit": "canonical MultiHopRAG evidence fact",
                "evidence_match": "same document and full occurrence containment",
                "joint_evidence_success": "all gold evidence facts retrieved",
                "joint_document_success": "all gold evidence documents retrieved",
                "null_queries": (
                    "context statistics only; refusal is evaluated during generation"
                ),
            },
            "official": {
                "source": OFFICIAL_EVALUATOR_URL,
                "retrieved_k": OFFICIAL_TOP_K,
                "text_normalization": (
                    "remove literal spaces and newline characters exactly as official code"
                ),
                "relevance": "normalized gold fact is a substring of retrieved text",
                "metrics": ["Hits@10", "Hits@4", "MAP@10", "MRR@10"],
                "null_queries": "excluded from official aggregation",
            },
        },
        "thesis_extended": {
            "answerable": answerable_summaries,
            "null_queries": null_summaries,
        },
        "official": official_summaries,
    }


def select_questions(
    questions: tuple[QuestionRecord, ...],
    *,
    included_types: tuple[str, ...],
    max_questions: int | None,
) -> tuple[QuestionRecord, ...]:
    selected = tuple(
        question
        for question in questions
        if question.question_type in included_types
    )
    if max_questions is not None:
        selected = selected[:max_questions]
    if not selected:
        raise ValueError("the evaluation question selection is empty")
    return selected


async def main() -> tuple[Path, dict]:
    started_at = time.perf_counter()
    dataset_directory = Path(os.getenv(
        "MULTIHOP_DATASET_DIR",
        PROJECT_ROOT / "data/raw/MultiHopRAG",
    )).resolve()
    working_directory = Path(os.getenv(
        "WORKING_DIR",
        PROJECT_ROOT / "artifacts/stores/multihop_rag_fixed",
    )).resolve()
    cache_directory = Path(os.getenv(
        "CACHE_DIR",
        PROJECT_ROOT / "artifacts/cache/multihop_rag",
    )).resolve()
    output_root = Path(os.getenv(
        "EVAL_OUTPUT_DIR",
        PROJECT_ROOT / "artifacts/evaluations/multihop_rag",
    )).resolve()
    build_manifest_path = working_directory / "build_manifest.json"
    if not build_manifest_path.is_file():
        raise RuntimeError(
            "WORKING_DIR does not contain a completed build manifest; "
            "build the MultiHopRAG index before running retrieval evaluation"
        )

    modes = parse_csv(
        os.getenv("EVAL_MODES", ",".join(DEFAULT_MODES)),
        label="EVAL_MODES",
    )
    unknown_modes = sorted(set(modes) - set(DEFAULT_MODES))
    if unknown_modes:
        raise ValueError(f"unknown EVAL_MODES: {unknown_modes}")
    k_values = parse_k_values(os.getenv(
        "EVAL_K_VALUES",
        ",".join(str(value) for value in DEFAULT_K_VALUES),
    ))
    concurrency = int(os.getenv("EVAL_CONCURRENCY", "4"))
    if concurrency <= 0:
        raise ValueError("EVAL_CONCURRENCY must be positive")
    max_questions_value = int(os.getenv("EVAL_MAX_QUESTIONS", "0"))
    if max_questions_value < 0:
        raise ValueError("EVAL_MAX_QUESTIONS must be zero or positive")
    max_questions = max_questions_value or None
    include_null = parse_bool(
        os.getenv("EVAL_INCLUDE_NULL", "true"),
        label="EVAL_INCLUDE_NULL",
    )
    default_types = (
        "comparison_query",
        "inference_query",
        "temporal_query",
        *(("null_query",) if include_null else ()),
    )
    included_types = parse_csv(
        os.getenv("EVAL_QUESTION_TYPES", ",".join(default_types)),
        label="EVAL_QUESTION_TYPES",
    )
    unknown_types = sorted(set(included_types) - set(default_types))
    if unknown_types:
        raise ValueError(
            "EVAL_QUESTION_TYPES contains disabled or unknown types: "
            f"{unknown_types}"
        )

    dataset = load_multihop_rag(dataset_directory)
    questions = select_questions(
        dataset.questions,
        included_types=included_types,
        max_questions=max_questions,
    )
    reranker = create_reranker()
    tokenizer_model = os.getenv("EVAL_TOKENIZER_MODEL", RERANK_MODEL)
    tokenizer = (
        reranker.tokenizer
        if reranker is not None and tokenizer_model == RERANK_MODEL
        else AutoTokenizer.from_pretrained(
            tokenizer_model,
            cache_dir=PROJECT_ROOT / "models",
        )
    )
    token_counter = lambda text: len(tokenizer.tokenize(text))

    rag = LightRAG(
        working_dir=working_directory,
        cache_directory=cache_directory,
        llm_func=llm_func,
        con_num=int(os.getenv("CON_NUM", "4")),
        embed_func=embed_func,
        embed_many_func=embed_many_func,
        reranker=reranker,
    )
    build = await rag.construct(dataset.documents)
    chunks = rag.chunk_kv.all()
    chunk_index = build_multidocument_chunk_index(dataset.documents, chunks)

    evidence_maps: dict[str, dict[str, list[str]]] = {}
    for question in questions:
        if question.question_type == "null_query":
            continue
        evidence_map = map_multihop_evidence_to_chunks(question, chunk_index)
        mapped_evidence = {
            evidence_id
            for evidence_ids in evidence_map.values()
            for evidence_id in evidence_ids
        }
        expected_evidence = {evidence.evidence_id for evidence in question.evidence}
        if mapped_evidence != expected_evidence:
            missing = sorted(expected_evidence - mapped_evidence)
            raise RuntimeError(
                f"question {question.question_id} has evidence absent from all chunks: {missing}"
            )
        evidence_maps[question.question_id] = evidence_map

    retrieval_config = {
        "modes": modes,
        "k_values": k_values,
        "question_types": included_types,
        "max_questions": max_questions,
        "chunk_candidate_top_k": rag.config.chunk_candidate_top_k,
        "entity_top_k": rag.config.entity_top_k,
        "relation_top_k": rag.config.relation_top_k,
        "relation_candidate_top_k": rag.config.relation_candidate_top_k,
        "reranker_enabled": reranker is not None,
        "reranker_model": RERANK_MODEL if reranker is not None else None,
        "embedding_model": EMBED_MODEL,
        "evaluation_tokenizer": tokenizer_model,
        "official": {
            "enabled": True,
            "top_k": OFFICIAL_TOP_K,
            "evaluator": OFFICIAL_EVALUATOR_URL,
        },
    }
    fingerprint_material = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "build_fingerprint": build.build_fingerprint,
        "questions_sha256": dataset.questions_sha256,
        "retrieval_config": retrieval_config,
        "selected_question_ids": [question.question_id for question in questions],
        "code_sha256": {
            "core": hashlib.sha256(
                (PROJECT_ROOT / "src/rag_research/core.py").read_bytes()
            ).hexdigest(),
            "evaluation": hashlib.sha256(
                (PROJECT_ROOT / "src/rag_research/evaluation.py").read_bytes()
            ).hexdigest(),
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    evaluation_fingerprint = stable_fingerprint(fingerprint_material)
    rerank_label = "rerank" if reranker is not None else "dense"
    default_run_name = (
        f"{rag.config.chunk_config.strategy}_{rerank_label}_"
        f"{evaluation_fingerprint[:12]}"
    )
    run_name = os.getenv("EVAL_RUN_NAME", default_run_name).strip()
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("EVAL_RUN_NAME must be a single non-empty path component")
    run_directory = output_root / run_name
    checkpoint_directory = run_directory / "checkpoints"
    manifest_path = run_directory / "run_manifest.json"

    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "running",
        "evaluation_fingerprint": evaluation_fingerprint,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "dataset": {
            "directory": str(dataset_directory),
            "corpus_sha256": dataset.corpus_sha256,
            "questions_sha256": dataset.questions_sha256,
            "document_count": len(dataset.documents),
            "dataset_question_count": len(dataset.questions),
            "selected_question_count": len(questions),
        },
        "index": {
            "working_directory": str(working_directory),
            "build_fingerprint": build.build_fingerprint,
            "chunking_fingerprint": build.chunking_fingerprint,
            "extraction_fingerprint": build.extraction_fingerprint,
            "chunk_count": build.chunk_count,
        },
        "retrieval_config": retrieval_config,
        "execution": {
            "concurrency": concurrency,
            "cache_directory": str(cache_directory),
        },
        "code_sha256": fingerprint_material["code_sha256"],
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("evaluation_fingerprint") != evaluation_fingerprint:
            raise RuntimeError(
                f"evaluation output {run_directory} belongs to a different experiment"
            )
        manifest["created_at"] = existing_manifest.get("created_at", manifest["created_at"])
    atomic_write_json(manifest_path, manifest)

    jobs: list[tuple[QuestionRecord, str, Path]] = []
    completed_rows: list[dict] = []
    for question in questions:
        for mode in modes:
            checkpoint_path = checkpoint_directory / mode / f"{question.question_id}.json"
            if checkpoint_path.exists():
                row = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if row.get("evaluation_fingerprint") != evaluation_fingerprint:
                    raise RuntimeError(f"incompatible checkpoint: {checkpoint_path}")
                if (
                    row.get("question_id") != question.question_id
                    or row.get("dataset_index") != question.dataset_index
                    or row.get("mode") != mode
                ):
                    raise RuntimeError(f"checkpoint identity mismatch: {checkpoint_path}")
                completed_rows.append(row)
            else:
                jobs.append((question, mode, checkpoint_path))

    semaphore = asyncio.Semaphore(concurrency)

    async def evaluate_one(
        question: QuestionRecord,
        mode: str,
        checkpoint_path: Path,
    ) -> dict:
        async with semaphore:
            trace = await rag.retrieve_trace(
                query=question.query,
                mode=mode,
                top_k=max(max(k_values), OFFICIAL_TOP_K),
            )

        metrics_by_k: dict[str, dict] = {}
        for requested_k in k_values:
            retrieved_ids = trace.get("chunk_ids", [])[:requested_k]
            if question.question_type == "null_query":
                metrics = calc_null_retrieval_context_metrics(
                    retrieved_chunk_ids=retrieved_ids,
                    chunks=chunks,
                    token_counter=token_counter,
                )
            else:
                metrics = calc_multihop_retrieval_metrics(
                    retrieved_chunk_ids=retrieved_ids,
                    requested_k=requested_k,
                    chunks=chunks,
                    chunk_to_evidence=evidence_maps[question.question_id],
                    evidence_to_document={
                        evidence.evidence_id: evidence.document_id
                        for evidence in question.evidence
                    },
                    evidence_lengths={
                        evidence.evidence_id: len(evidence.fact)
                        for evidence in question.evidence
                    },
                    token_counter=token_counter,
                )
            metrics["requested_k"] = requested_k
            metrics_by_k[str(requested_k)] = metrics

        if question.question_type == "null_query":
            official = {
                "applicable": False,
                "reason": "official evaluator excludes null_query rows",
            }
        else:
            official_retrieval = build_official_retrieval_list(trace, chunks)
            official = {
                "applicable": True,
                "metrics": calc_multihop_official_metrics(
                    retrieved_texts=[item["text"] for item in official_retrieval],
                    gold_facts=[evidence.fact for evidence in question.evidence],
                ),
            }

        row = {
            "evaluation_fingerprint": evaluation_fingerprint,
            "question_id": question.question_id,
            "dataset_index": question.dataset_index,
            "question": question.query,
            "answer": question.answer,
            "question_type": question.question_type,
            "hop_count": len(question.evidence),
            "mode": mode,
            "gold_evidence": [asdict(evidence) for evidence in question.evidence],
            "gold_document_ids": list(dict.fromkeys(
                evidence.document_id for evidence in question.evidence
            )),
            "thesis_extended": {
                "metrics_by_k": metrics_by_k,
            },
            "official": official,
            "trace": compact_trace(trace),
        }
        atomic_write_json(checkpoint_path, row)
        return row

    tasks = [
        asyncio.create_task(evaluate_one(question, mode, checkpoint_path))
        for question, mode, checkpoint_path in jobs
    ]
    progress = tqdm(total=len(questions) * len(modes), initial=len(completed_rows))
    try:
        for task in asyncio.as_completed(tasks):
            completed_rows.append(await task)
            progress.update(1)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    finally:
        progress.close()

    mode_order = {mode: index for index, mode in enumerate(modes)}
    completed_rows.sort(key=lambda row: (
        row["dataset_index"],
        mode_order[row["mode"]],
    ))
    expected_result_count = len(questions) * len(modes)
    if len(completed_rows) != expected_result_count:
        raise RuntimeError(
            f"incomplete evaluation: {len(completed_rows)} of {expected_result_count} results"
        )

    summaries = build_summaries(completed_rows, modes, k_values)
    results_path = run_directory / "results.json"
    summaries_path = run_directory / "summaries.json"
    atomic_write_json(results_path, completed_rows)
    atomic_write_json(summaries_path, summaries)

    official_directory = run_directory / "official"
    official_files = {}
    for mode in modes:
        official_path = official_directory / f"{mode}.json"
        atomic_write_json(
            official_path,
            build_official_export(completed_rows, mode=mode, chunks=chunks),
        )
        official_files[mode] = str(official_path.relative_to(run_directory))

    manifest.update({
        "status": "complete",
        "updated_at": utc_now(),
        "completed_at": utc_now(),
        "result_count": len(completed_rows),
        "checkpoint_count": len(completed_rows),
        "elapsed_seconds_this_run": time.perf_counter() - started_at,
        "results_file": results_path.name,
        "summaries_file": summaries_path.name,
        "official_files": official_files,
    })
    atomic_write_json(manifest_path, manifest)
    return run_directory, manifest


if __name__ == "__main__":
    output_directory, final_manifest = asyncio.run(main())
    print(json.dumps({
        "status": final_manifest["status"],
        "evaluation_fingerprint": final_manifest["evaluation_fingerprint"],
        "result_count": final_manifest["result_count"],
        "output_directory": str(output_directory),
    }, ensure_ascii=False, indent=2))
