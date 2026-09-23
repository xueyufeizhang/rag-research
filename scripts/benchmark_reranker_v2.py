"""Offline, same-candidate v1/v2 pilot using a completed naive evaluation.

Models must already be downloaded. This does not rebuild an index, call an
embedding/LLM service, or change the main evaluation's configuration/results.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import sys
import threading
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import psutil
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

from rag_research.datasets.multihop_rag import load_multihop_rag
from rag_research.evaluation import (
    build_multidocument_chunk_index,
    calc_multihop_official_metrics,
    calc_multihop_retrieval_metrics,
    map_multihop_evidence_to_chunks,
)

V1_REVISION = "800f24c113213a187e65bde9db00c15a2bb12738"
V2_REVISION = "3ea9d4dffa7d12a4f366be8e275c349de9fc9865"


def read_json(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def snapshot(name, revision):
    return ROOT / "models" / f"models--mixedbread-ai--mxbai-rerank-base-{name}" / "snapshots" / revision


class MemoryProbe:
    def __init__(self, device):
        self.device = device
        self.stop = threading.Event()
        self.peak_rss = 0
        self.peak_mps_driver = 0
        self.process = psutil.Process()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            if self.device == "mps":
                self.peak_mps_driver = max(self.peak_mps_driver, torch.mps.driver_allocated_memory())
            self.stop.wait(0.1)

    def finish(self):
        self.stop.set()
        self.thread.join()
        return {"peak_process_rss_gib": self.peak_rss / 2**30,
                "peak_mps_driver_gib": self.peak_mps_driver / 2**30,
                "note": "Sampled every 100 ms; RSS and GPU allocations can overlap on unified memory; do not sum."}


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def aggregate(rows, name):
    metrics = [row[name]["official"] for row in rows]
    out = {key: statistics.mean(x[key] for x in metrics)
           for key in ("Hits@10", "Hits@4", "MAP@10", "MRR@10")}
    for k in (1, 5, 10, 20):
        for key in ("evidence_recall", "joint_evidence_success", "chunk_precision"):
            out[f"{key}@{k}"] = statistics.mean(row[name]["extended"][str(k)][key] for row in rows)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-type", type=int, default=8)
    parser.add_argument("--stress-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--reuse-v1-from", type=Path,
                        help="Reuse a verified same-config v1 pilot's scores/timings; rerun v2 only.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this execution context; run with Apple GPU access.")
    if args.per_type < 1 or args.stress_count < 0 or args.batch_size < 1:
        raise ValueError("invalid sample or batch size")
    args.output.mkdir(parents=True, exist_ok=False)
    run = ROOT / "artifacts/evaluations/multihop_rag/baseline_v1_fixed"
    store = ROOT / "artifacts/stores/retrieval_baseline_v1_fixed"
    old_manifest = read_json(run / "run_manifest.json")
    build = read_json(store / "build_manifest.json")["build"]
    assert old_manifest["status"] == "complete"
    assert old_manifest["index"]["build_fingerprint"] == build["build_fingerprint"]
    assert old_manifest["retrieval_config"]["chunk_candidate_top_k"] == 20
    dataset = load_multihop_rag(ROOT / "data/raw/MultiHopRAG")
    assert dataset.corpus_sha256 == old_manifest["dataset"]["corpus_sha256"]
    assert dataset.questions_sha256 == old_manifest["dataset"]["questions_sha256"]
    chunks = read_json(store / "chunks.json")
    chunk_index = build_multidocument_chunk_index(dataset.documents, chunks)
    questions = {q.question_id: q for q in dataset.questions if q.question_type != "null_query"}
    tokenizer = AutoTokenizer.from_pretrained(snapshot("v1", V1_REVISION), local_files_only=True)
    count = lambda text: len(tokenizer(text, add_special_tokens=False, truncation=False, verbose=False)["input_ids"])
    chunk_tokens = {cid: count(c["text"]) for cid, c in chunks.items()}
    records = {}
    for qid, q in questions.items():
        row = read_json(run / "checkpoints/naive" / f"{qid}.json")
        assert row["evaluation_fingerprint"] == old_manifest["evaluation_fingerprint"]
        assert row["trace"]["requested_top_k"] == 20 and len(row["trace"]["chunk_ids"]) == 20
        assert len(set(row["trace"]["chunk_ids"])) == 20
        lengths = [count(q.query) + chunk_tokens[cid] + 3 for cid in row["trace"]["chunk_ids"]]
        records[qid] = {"saved": row, "v1_lengths": lengths}
    rng = random.Random(args.seed)
    chosen = []
    for kind in ("comparison_query", "inference_query", "temporal_query"):
        ids = sorted(qid for qid, q in questions.items() if q.question_type == kind)
        chosen += [(qid, "stratified_random") for qid in rng.sample(ids, args.per_type)]
    excluded = {qid for qid, _ in chosen}
    stress = sorted((qid for qid in questions if qid not in excluded),
                    key=lambda qid: (-max(records[qid]["v1_lengths"]), qid))
    chosen += [(qid, "long_input_stress") for qid in stress[:args.stress_count]]
    results = []
    evidence_maps = {}

    def score(qid, ranked):
        q = questions[qid]
        return {
            "official": calc_multihop_official_metrics(
                retrieved_texts=[chunks[cid]["model_text"] for cid in ranked[:10]],
                gold_facts=[e.fact for e in q.evidence]),
            "extended": {str(k): calc_multihop_retrieval_metrics(
                question=q, retrieved_chunk_ids=ranked[:k], requested_k=k,
                chunks=chunks, chunk_to_evidence=evidence_maps[qid], token_counter=count)
                for k in (1, 5, 10, 20)},
        }

    for qid, cohort in chosen:
        rec = records[qid]
        evidence_maps[qid] = map_multihop_evidence_to_chunks(questions[qid], chunk_index)
        stored = rec["saved"]
        # Recover the original dense ordering for a stable, model-independent tie break.
        dense = sorted(stored["trace"]["chunks"], key=lambda c: (-c["dense_score"], c["chunk_id"]))
        row = {"question_id": qid, "question": questions[qid].query,
               "question_type": questions[qid].question_type, "cohort": cohort,
               "candidate_ids": [c["chunk_id"] for c in dense],
               "saved_v1_ids": stored["trace"]["chunk_ids"],
               "v1_pair_lengths": rec["v1_lengths"]}
        row["saved_v1"] = score(qid, row["saved_v1_ids"])
        assert row["saved_v1"]["official"] == stored["official"]["metrics"]
        for k in (1, 5, 10, 20):
            for key in ("evidence_recall", "joint_evidence_success", "chunk_precision"):
                assert row["saved_v1"]["extended"][str(k)][key] == stored["thesis_extended"]["metrics_by_k"][str(k)][key]
        results.append(row)
    del records
    gc.collect()
    manifest = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Naive reranker-only paired pilot; both models see all 20 original dense candidates. Not a full graph-mode evaluation.",
        "selection": {"seed": args.seed, "random_per_answerable_type": args.per_type,
                      "stress_count": args.stress_count, "stress_selection": "Largest v1 pair lengths, excluding random cohort; metrics reported separately."},
        "baseline_fingerprint": old_manifest["evaluation_fingerprint"],
        "build_fingerprint": build["build_fingerprint"],
        "model_revisions": {"v1": V1_REVISION, "v2": V2_REVISION},
        "versions": {p: version(p) for p in ("torch", "transformers", "sentence-transformers")},
        "device": args.device, "dtype": "float16", "batch_size": args.batch_size,
        "score_activation": {"v1": "sigmoid (existing baseline)", "v2": "identity (raw 1-versus-0 logit difference; avoids sigmoid saturation)"},
        "v2_max_length": 8192, "evaluation_tokenizer": "mixedbread-ai/mxbai-rerank-base-v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hardware_total_gib": psutil.virtual_memory().total / 2**30,
        "selected": [{"question_id": qid, "cohort": c} for qid, c in chosen],
    }
    save(args.output / "run_manifest.json", manifest)
    save(args.output / "results.json", results)
    print(json.dumps({"phase": "selected", "questions": len(results), "pairs": len(results)*20}), flush=True)
    performance = {}
    if args.reuse_v1_from:
        previous_manifest = read_json(args.reuse_v1_from / "run_manifest.json")
        assert previous_manifest["status"] == "complete"
        for key in ("baseline_fingerprint", "build_fingerprint", "model_revisions", "versions",
                    "device", "dtype", "batch_size", "selection", "selected"):
            assert previous_manifest[key] == manifest[key], f"cannot reuse incompatible v1: {key}"
        previous = {row["question_id"]: row for row in read_json(args.reuse_v1_from / "results.json")}
        for row in results:
            old = previous[row["question_id"]]
            assert old["candidate_ids"] == row["candidate_ids"]
            assert old["saved_v1"] == row["saved_v1"]
            row["v1"] = old["v1"]
        performance["v1"] = read_json(args.reuse_v1_from / "performance.json")["v1"]
        manifest["v1_scores_and_timing_reused_from"] = str(args.reuse_v1_from.resolve())
        save(args.output / "run_manifest.json", manifest)
        save(args.output / "results.json", results)
        save(args.output / "performance.json", performance)
    for name, revision, limit in (("v1", V1_REVISION, 512), ("v2", V2_REVISION, 8192)):
        if name == "v1" and args.reuse_v1_from:
            continue
        memory = MemoryProbe(args.device)
        memory.thread.start()
        started = time.perf_counter()
        model = CrossEncoder(str(snapshot(name, revision)), local_files_only=True,
                             device=args.device, max_length=limit,
                             model_kwargs={"torch_dtype": torch.float16},
                             activation_fn=torch.nn.Identity() if name == "v2" else None)
        sync(args.device)
        load_seconds = time.perf_counter() - started
        # Warm up outside all timings; official v2 chat template and LogitScore are loaded.
        warm = [(results[0]["question"], chunks[cid]["text"]) for cid in results[0]["candidate_ids"][:2]]
        model.predict(warm, batch_size=args.batch_size, show_progress_bar=False)
        sync(args.device)
        elapsed = []
        for i, row in enumerate(results, 1):
            pairs = [(row["question"], chunks[cid]["text"]) for cid in row["candidate_ids"]]
            lengths = []
            used = []
            # Inspect exactly the model-ready inputs, including the v2 chat template.
            for offset in range(0, len(pairs), args.batch_size):
                part = pairs[offset:offset + args.batch_size]
                full = model.preprocess(part, processing_kwargs={"text": {"truncation": False}})
                lengths += full["attention_mask"].sum(dim=1).tolist()
                if name == "v2" and max(lengths) > limit:
                    raise ValueError("v2 overflow: refusing to benchmark truncated inputs")
                actual = model.preprocess(part)
                used += actual["attention_mask"].sum(dim=1).tolist()
                del full, actual
            if name == "v2":
                assert lengths == used, "v2 silently truncated or changed input length"
            sync(args.device)
            started = time.perf_counter()
            scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            sync(args.device)
            seconds = time.perf_counter() - started
            scores = [float(s) for s in scores]
            assert len(scores) == len(pairs) and all(math.isfinite(s) for s in scores)
            order = sorted(range(len(scores)), key=lambda j: -scores[j])
            ranked = [row["candidate_ids"][j] for j in order]
            row[name] = {**score(row["question_id"], ranked), "ranked_ids": ranked,
                         "scores_in_candidate_order": scores, "seconds": seconds,
                         "full_input_tokens": lengths, "model_input_tokens": used,
                         "truncated_pairs": sum(a != b for a, b in zip(lengths, used))}
            # At K=20 a reranker cannot change evidence recall on this identical pool.
            assert row[name]["extended"]["20"]["evidence_recall"] == row["saved_v1"]["extended"]["20"]["evidence_recall"]
            elapsed.append(seconds)
            save(args.output / "results.json", results)
            print(json.dumps({"model": name, "done": i, "total": len(results),
                              "seconds": round(seconds, 3), "max_input": max(lengths),
                              "truncated": row[name]["truncated_pairs"],
                              "recall10": row[name]["extended"]["10"]["evidence_recall"]}), flush=True)
        performance[name] = {"load_seconds": load_seconds, "total_rerank_seconds": sum(elapsed),
                             "median_seconds_per_20_candidates": statistics.median(elapsed),
                             "p95_seconds_per_20_candidates": sorted(elapsed)[math.ceil(len(elapsed)*0.95)-1],
                             "pairs_per_second": len(results)*20/sum(elapsed),
                             "memory": memory.finish(),
                             "truncated_pairs": sum(row[name]["truncated_pairs"] for row in results),
                             "max_full_input_tokens": max(max(row[name]["full_input_tokens"]) for row in results)}
        if name == "v2":
            # Extra compatibility/length smoke test for real merged graph relations.
            relations = read_json(store / "relations.json")
            picked = sorted(relations.items(), key=lambda kv: len(kv[1].get("description", "")), reverse=True)[:3]
            probes = []
            for rid, relation in picked:
                query = f"What is the relationship between {relation['source']} and {relation['target']}?"
                text = (f"source: {relation['source']}; target: {relation['target']}; "
                        f"keywords: {', '.join(relation.get('keywords', []))}; description: {relation.get('description', '')}")
                pair = [(query, text)]
                full = model.preprocess(pair, processing_kwargs={"text": {"truncation": False}})
                n = int(full["attention_mask"].sum())
                assert n <= limit
                actual = model.preprocess(pair)
                assert int(actual["attention_mask"].sum()) == n
                prediction = float(model.predict(pair, batch_size=1, show_progress_bar=False)[0])
                assert math.isfinite(prediction)
                probes.append({"relation_id": rid, "full_input_tokens": n, "score": prediction, "truncated": False})
                del full, actual
            save(args.output / "relation_smoke.json", {"scope": "Synthetic endpoint questions with real relation texts; checks inference only, not graph retrieval quality.", "probes": probes})
        del model
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
        save(args.output / "performance.json", performance)
    summaries = {}
    for cohort in ("stratified_random", "long_input_stress"):
        selected = [row for row in results if row["cohort"] == cohort]
        if not selected:
            continue
        summaries[cohort] = {"question_count": len(selected),
                             **{name: aggregate(selected, name) for name in ("saved_v1", "v1", "v2")}}
        delta = [row["v2"]["extended"]["10"]["evidence_recall"] - row["saved_v1"]["extended"]["10"]["evidence_recall"] for row in selected]
        summaries[cohort]["recall10_v2_vs_saved_v1"] = {"improved": sum(d > 0 for d in delta), "same": sum(d == 0 for d in delta), "worse": sum(d < 0 for d in delta)}
        summaries[cohort]["fresh_v1_top10_order_changed"] = sum(row["v1"]["ranked_ids"][:10] != row["saved_v1_ids"][:10] for row in selected)
    save(args.output / "summaries.json", summaries)
    manifest.update(status="complete", completed_at=datetime.now(timezone.utc).isoformat(), result_count=len(results))
    save(args.output / "run_manifest.json", manifest)
    print(json.dumps({"complete": True, "output": str(args.output), "summaries": summaries, "performance": performance}), flush=True)


if __name__ == "__main__":
    main()
