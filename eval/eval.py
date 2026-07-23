import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json, os
from tqdm import tqdm
from dotenv import load_dotenv
from core import LightRAG
from backend import llm_func, embed_func

load_dotenv()
CON_NUM = os.getenv("CON_NUM", 4)

def normalize_relation_key(key: str) -> str:
    parts = key.split("||")
    if len(parts) != 2:
        return key.strip()
    return "||".join(sorted([parts[0].strip(), parts[1].strip()]))

def calc_recall(retrieved: list[str], gold: list[str]) -> dict:
    retrieved_set = set(retrieved)
    gold_set = set(gold)
    matched = retrieved_set & gold_set

    return {
        "matched": sorted(matched),
        "matched_count": len(matched),
        "gold_count": len(gold_set),
        "retrieved_count": len(retrieved_set),
        "recall": len(matched) / len(gold_set) if gold_set else 0.0,
        "hit": len(matched) > 0,
    }

def summarize_mode(mode: str, mode_results: list[dict]) -> dict:
    total = len(mode_results)
    if total == 0:
        return {
            "mode": mode,
            "count": 0,
            "avg_entity_recall": 0.0,
            "avg_relation_recall": 0.0,
            "avg_chunk_recall": 0.0,
            "entity_hit_rate": 0.0,
            "relation_hit_rate": 0.0,
            "chunk_hit_rate": 0.0,
        }
    return {
        "mode": mode,
        "count": total,
        "avg_entity_recall": sum(r["entity_metrics"]["recall"] for r in mode_results) / total,
        "avg_relation_recall": sum(r["relation_metrics"]["recall"] for r in mode_results) / total,
        "avg_chunk_recall": sum(r["chunk_metrics"]["recall"] for r in mode_results) / total,
        "entity_hit_rate": sum(r["entity_metrics"]["hit"] for r in mode_results) / total,
        "relation_hit_rate": sum(r["relation_metrics"]["hit"] for r in mode_results) / total,
        "chunk_hit_rate": sum(r["chunk_metrics"]["hit"] for r in mode_results) / total,
    }

async def evaluate() -> tuple[list[dict], list[dict]]:
    lightrag = LightRAG(os.getenv("WORKING_DIR", "./dickens"), llm_func, CON_NUM, embed_func)
    with open("./carol.txt", "r", encoding="utf-8")as f: 
        await lightrag.construct(f.read(), "carol")
    with open("./eval/carol_eval_set.json", "r", encoding="utf-8") as f:
        eval_questions = json.load(f).get("questions", [])

    results = []
    summaries = []
    for mode in ["naive", "local", "global", "hybrid"]:
        mode_results = []
        print(f"----- {mode} retrieval -----\n")
        for item in tqdm(eval_questions):
            question = item.get("question")
            if not question:
                continue
            trace = await lightrag.retrieve_trace(query=question, mode=mode)
            retrieved_entities = trace.get("entity_ids", [])
            retrieved_relations = [normalize_relation_key(r) for r in trace.get("relation_ids", [])]
            retrieved_chunks = trace.get("chunk_ids", [])
            gold_entities = item.get("gold_entities", [])
            gold_relations = [normalize_relation_key(gr) for gr in item.get("gold_relations", [])]
            gold_chunks = item.get("gold_chunks", [])

            entity_metrics = calc_recall(retrieved_entities, gold_entities)
            relation_metrics = calc_recall(retrieved_relations, gold_relations)
            chunk_metrics = calc_recall(retrieved_chunks, gold_chunks)

            row = {
                "id": item.get("id"),
                "mode": mode,
                "question": question,
                "entity_metrics": entity_metrics,
                "relation_metrics": relation_metrics,
                "chunk_metrics": chunk_metrics,
                "retrieved_entities": retrieved_entities,
                "retrieved_relations": retrieved_relations,
                "retrieved_chunks": retrieved_chunks,
                "gold_entities": gold_entities,
                "gold_relations": gold_relations,
                "gold_chunks": gold_chunks
            }
            results.append(row)
            mode_results.append(row)

        summary = summarize_mode(mode, mode_results)
        summaries.append(summary)
        print(f"\n[{summary['mode']}] {summary['count']} questions")
        print(f"  entity recall:   {summary['avg_entity_recall']:.3f}")
        print(f"  relation recall: {summary['avg_relation_recall']:.3f}")
        print(f"  chunk recall:    {summary['avg_chunk_recall']:.3f}")
        print(f"  entity hit rate:   {summary['entity_hit_rate']:.3f}")
        print(f"  relation hit rate: {summary['relation_hit_rate']:.3f}")
        print(f"  chunk hit rate:    {summary['chunk_hit_rate']:.3f}")
    return results, summaries


if __name__ == "__main__":
    import asyncio
    results, summaries = asyncio.run(evaluate())

    os.makedirs("./eval/runs", exist_ok=True)
    with open("./eval/runs/retrieval_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open("./eval/runs/retrieval_eval_results_summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

