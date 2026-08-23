import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import asyncio
import hashlib
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

from rag_research.backends import (
    API_MODEL,
    EMBED_MODEL,
    LLM_BACKEND,
    OLLAMA_MODEL,
    RERANK_MODEL,
    embed_func,
    llm_func,
    reranker,
)
from rag_research.core import LightRAG
from rag_research.evaluation import (
    build_chunk_evidence_map,
    build_entity_normalizer,
    calc_evidence_metrics,
    calc_set_metrics,
    harmonic_mean,
    locate_chunks_in_source,
    map_question_evidence_to_chunks,
    normalize_relation_key,
)


load_dotenv()
CON_NUM = int(os.getenv("CON_NUM", 5))
DEFAULT_MODES = ["naive", "local", "global", "hybrid"]


def average(rows: list[dict], metric_group: str, metric: str) -> float:
    return sum(row[metric_group].get(metric, 0.0) for row in rows) / len(rows) if rows else 0.0


def summarize_mode(mode: str, mode_results: list[dict]) -> dict:
    total = len(mode_results)
    summary = {
        "mode": mode,
        "count": total,
        "metric_protocol": "canonical evidence coverage",
        "avg_entity_recall": average(mode_results, "entity_metrics", "recall"),
        "avg_entity_precision": average(mode_results, "entity_metrics", "precision"),
        "avg_entity_f1": average(mode_results, "entity_metrics", "f1"),
        "avg_relation_recall": average(mode_results, "relation_metrics", "recall"),
        "avg_relation_precision": average(mode_results, "relation_metrics", "precision"),
        "avg_relation_f1": average(mode_results, "relation_metrics", "f1"),
        "avg_chunk_precision": average(mode_results, "chunk_metrics", "chunk_precision"),
        "avg_evidence_recall": average(mode_results, "chunk_metrics", "evidence_recall"),
        "avg_answer_point_recall": average(mode_results, "chunk_metrics", "answer_point_recall"),
        "avg_coverage_f1": average(mode_results, "chunk_metrics", "coverage_f1"),
        "mean_reciprocal_rank": average(mode_results, "chunk_metrics", "reciprocal_rank"),
        "avg_ndcg_at_k": average(mode_results, "chunk_metrics", "ndcg_at_k"),
        "avg_chunk_redundancy_rate": average(mode_results, "chunk_metrics", "redundancy_rate"),
        "entity_hit_rate": average(mode_results, "entity_metrics", "hit"),
        "relation_hit_rate": average(mode_results, "relation_metrics", "hit"),
        "chunk_hit_rate": average(mode_results, "chunk_metrics", "chunk_hit"),
    }

    retrieved_total = sum(row["chunk_metrics"]["retrieved_count"] for row in mode_results)
    relevant_total = sum(row["chunk_metrics"]["relevant_retrieved_count"] for row in mode_results)
    evidence_total = sum(row["chunk_metrics"]["gold_evidence_count"] for row in mode_results)
    evidence_matched = sum(row["chunk_metrics"]["matched_evidence_count"] for row in mode_results)
    answer_point_total = sum(row["chunk_metrics"]["gold_answer_point_count"] for row in mode_results)
    answer_point_matched = sum(row["chunk_metrics"]["matched_answer_point_count"] for row in mode_results)

    micro_precision = relevant_total / retrieved_total if retrieved_total else 0.0
    micro_recall = evidence_matched / evidence_total if evidence_total else 0.0
    summary.update({
        "micro_chunk_precision": micro_precision,
        "micro_evidence_recall": micro_recall,
        "micro_coverage_f1": harmonic_mean(micro_precision, micro_recall),
        "micro_answer_point_recall": (
            answer_point_matched / answer_point_total if answer_point_total else 0.0
        ),
    })
    return summary


def print_summary(summary: dict) -> None:
    print(f"\n[{summary['mode']}] {summary['count']} questions")
    print(f"  entity recall:            {summary['avg_entity_recall']:.3f}")
    print(f"  entity precision:         {summary['avg_entity_precision']:.3f}")
    print(f"  entity f1:                {summary['avg_entity_f1']:.3f}\n")
    print(f"  relation recall:          {summary['avg_relation_recall']:.3f}")
    print(f"  relation precision:       {summary['avg_relation_precision']:.3f}")
    print(f"  relation f1:              {summary['avg_relation_f1']:.3f}\n")

    print(f"  chunk precision (macro):  {summary['avg_chunk_precision']:.3f}")
    print(f"  evidence recall (macro):  {summary['avg_evidence_recall']:.3f}")
    print(f"  coverage f1 (macro):      {summary['avg_coverage_f1']:.3f}")
    print(f"  answer-point recall:      {summary['avg_answer_point_recall']:.3f}")
    print(f"  MRR:                      {summary['mean_reciprocal_rank']:.3f}")
    print(f"  nDCG@K:                   {summary['avg_ndcg_at_k']:.3f}")
    print(f"  chunk redundancy rate:    {summary['avg_chunk_redundancy_rate']:.3f}")
    print(f"  chunk precision (micro):  {summary['micro_chunk_precision']:.3f}")
    print(f"  evidence recall (micro):  {summary['micro_evidence_recall']:.3f}")
    print(f"  coverage f1 (micro):      {summary['micro_coverage_f1']:.3f}\n")

    print(f"  entity hit rate:          {summary['entity_hit_rate']:.3f}")
    print(f"  relation hit rate:        {summary['relation_hit_rate']:.3f}")
    print(f"  chunk hit rate:           {summary['chunk_hit_rate']:.3f}")


async def eval_retrieval() -> tuple[list[dict], list[dict]]:
    source_path = PROJECT_ROOT / "data/raw/a_christmas_carol.txt"
    source = source_path.read_text(encoding="utf-8")
    working_dir = os.getenv("WORKING_DIR", "./artifacts/stores/dickens_fixed_size")
    lightrag = LightRAG(
        working_dir=working_dir,
        llm_func=llm_func,
        con_num=CON_NUM,
        embed_func=embed_func,
        reranker=reranker,
    )
    await lightrag.construct(source, "carol")
    chunking_strategy = lightrag.config.chunk_config.strategy
    llm_model = API_MODEL if LLM_BACKEND == "api" else OLLAMA_MODEL
    reranking_enabled = lightrag.reranker is not None
    rerank_model = RERANK_MODEL if reranking_enabled else None

    eval_set_path = Path(os.getenv("EVAL_SET", "./data/evaluation/carol_canonical.json"))
    eval_data = json.loads(eval_set_path.read_text(encoding="utf-8"))
    eval_questions = eval_data.get("questions", [])
    if eval_data.get("annotation", {}).get("chunk_independent") is not True:
        raise ValueError("EVAL_SET must be a chunk-independent canonical evidence file")
    missing_evidence = [
        item.get("id")
        for item in eval_questions
        if not item.get("gold_evidence_spans")
    ]
    if missing_evidence:
        raise ValueError(f"questions without canonical evidence: {missing_evidence}")
    aliases = eval_data.get("entity_aliases", {})
    canonical_names = {
        entity
        for item in eval_questions
        for entity in item.get("gold_entities", [])
    }
    normalize_entity = build_entity_normalizer(canonical_names, aliases)
    normalize_relation = lambda key: normalize_relation_key(key, normalize_entity)

    expected_source_hash = eval_data.get("source", {}).get("sha256")
    if expected_source_hash:
        actual_source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if actual_source_hash != expected_source_hash:
            raise ValueError(
                "canonical evidence offsets do not match the source document: "
                f"expected {expected_source_hash}, got {actual_source_hash}"
            )
    chunk_intervals = locate_chunks_in_source(source, lightrag.chunk_kv.all())

    requested_modes = [
        mode.strip()
        for mode in os.getenv("EVAL_MODES", ",".join(DEFAULT_MODES)).split(",")
        if mode.strip()
    ]
    unknown_modes = sorted(set(requested_modes) - set(DEFAULT_MODES))
    if unknown_modes:
        raise ValueError(f"unknown EVAL_MODES: {unknown_modes}")

    results = []
    summaries = []
    for mode in requested_modes:
        mode_results = []
        print(f"----- {mode} retrieval -----\n")
        for item in tqdm(eval_questions):
            question = item.get("question")
            if not question:
                continue

            trace = await lightrag.retrieve_trace(query=question, mode=mode)
            retrieved_entities = trace.get("entity_ids", [])
            retrieved_relations = trace.get("relation_ids", [])
            retrieved_chunks = trace.get("chunk_ids", [])
            gold_entities = item.get("gold_entities", [])
            gold_relations = item.get("gold_relations", [])

            entity_metrics = calc_set_metrics(
                retrieved_entities,
                gold_entities,
                normalize_entity,
            )
            relation_metrics = calc_set_metrics(
                retrieved_relations,
                gold_relations,
                normalize_relation,
            )

            relevant_chunk_evidence = item.get("gold_chunk_evidence") or (
                map_question_evidence_to_chunks(item, chunk_intervals)
            )
            chunk_to_evidence, evidence_to_answer_points = build_chunk_evidence_map(
                item,
                relevant_chunk_evidence,
            )
            chunk_metrics = calc_evidence_metrics(
                retrieved_chunks,
                chunk_to_evidence,
                evidence_to_answer_points,
                len(item.get("gold_answer_points", [])),
            )

            row = {
                "id": item.get("id"),
                "mode": mode,
                "question": question,
                "chunking_strategy": chunking_strategy,
                "llm_backend": LLM_BACKEND,
                "llm_model": llm_model,
                "embedding_model": EMBED_MODEL,
                "reranking_enabled": reranking_enabled,
                "rerank_model": rerank_model,
                "metric_protocol": "canonical evidence coverage",
                "entity_metrics": entity_metrics,
                "relation_metrics": relation_metrics,
                "chunk_metrics": chunk_metrics,
                "retrieved_entities": retrieved_entities,
                "retrieved_relations": retrieved_relations,
                "retrieved_chunks": retrieved_chunks,
                "gold_entities": gold_entities,
                "gold_relations": gold_relations,
                "gold_evidence_spans": item.get("gold_evidence_spans", []),
                "gold_chunk_evidence": relevant_chunk_evidence,
            }
            results.append(row)
            mode_results.append(row)

        summary = summarize_mode(mode, mode_results)
        summary.update({
            "chunking_strategy": chunking_strategy,
            "llm_backend": LLM_BACKEND,
            "llm_model": llm_model,
            "embedding_model": EMBED_MODEL,
            "reranking_enabled": reranking_enabled,
            "rerank_model": rerank_model,
        })
        summaries.append(summary)
        print_summary(summary)

    return results, summaries


if __name__ == "__main__":
    results, summaries = asyncio.run(eval_retrieval())

    output_dir = PROJECT_ROOT / "artifacts/evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = summaries[0] if summaries else {}
    chunking_strategy = run_summary.get(
        "chunking_strategy",
        os.getenv("CHUNKING_STRATEGY", "unknown"),
    )
    rerank_label = (
        "rerank"
        if run_summary.get("reranking_enabled", reranker is not None)
        else "dense_only"
    )
    experiment_name = f"{chunking_strategy}_{rerank_label}"
    results_path = output_dir / f"retrieval_eval_{experiment_name}_results.json"
    summaries_path = output_dir / f"retrieval_eval_{experiment_name}_summaries.json"

    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summaries_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved results to {results_path}")
    print(f"Saved summaries to {summaries_path}")
