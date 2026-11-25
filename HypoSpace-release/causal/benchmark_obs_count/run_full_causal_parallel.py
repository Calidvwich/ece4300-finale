import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "causal" / "datasets" / "n4_obs3_all.json"
PARTIAL = ROOT / "causal" / "benchmark_obs_count" / "checkpoints" / "full_api_causal_n4_obs3" / "checkpoint_causal_enhanced_OpenAI_deepseek-flash__20261125_004320.json"
RUN_DIR = ROOT / "causal" / "benchmark_obs_count" / "full_api_causal_n4_obs3_parallel"
FINAL = ROOT / "causal" / "benchmark_obs_count" / "full_repo_causal_n4_obs3_deepseek.json"
PYTHON = Path(sys.executable)
WORKERS = 8


def stats_block(values):
    values = np.asarray(values, dtype=float)
    p = float(stats.ttest_1samp(values, 0).pvalue) if len(values) > 1 else None
    return {"mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std()) if len(values) else 0.0,
            "var": float(values.var()) if len(values) else 0.0,
            "min": float(values.min()) if len(values) else 0.0,
            "max": float(values.max()) if len(values) else 0.0,
            "p_value": p}


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    partial = json.loads(PARTIAL.read_text(encoding="utf-8"))
    done = {r["observation_set_id"] for r in partial["results"]}
    tasks = data["datasets"]
    remaining = [t for t in tasks if t["observation_set_id"] not in done]
    if len(done) != 10:
        raise RuntimeError(f"Expected exactly 10 preserved tasks; found {len(done)}")
    shard_size = math.ceil(len(remaining) / WORKERS)
    shards = [remaining[i:i + shard_size] for i in range(0, len(remaining), shard_size)]
    processes = []
    for i, shard in enumerate(shards):
        shard_dir = RUN_DIR / f"shard_{i:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        ds = dict(data)
        ds["datasets"] = shard
        ds["metadata"] = dict(data.get("metadata", {}), n_datasets=len(shard))
        shard_file = shard_dir / "dataset.json"
        out_file = shard_dir / "result.json"
        log_file = (shard_dir / "run.log").open("w", encoding="utf-8")
        shard_file.write_text(json.dumps(ds), encoding="utf-8")
        cmd = [str(PYTHON), "-u", str(ROOT / "causal" / "run_causal_benchmark.py"),
               "--dataset", str(shard_file), "--config", str(ROOT / "causal" / "config" / "config_deepseek.yaml"),
               "--n-samples", str(len(shard)), "--query-multiplier", "1", "--independent-samples",
               "--max-retries", "3", "--seed", str(33550336 + i),
               "--checkpoint-dir", str(shard_dir / "checkpoints"), "--output", str(out_file), "--quiet"]
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((i, proc, log_file, len(shard), shard_dir, out_file))
        print(f"Started shard {i + 1}/{len(shards)}: {len(shard)} tasks, PID={proc.pid}", flush=True)

    last_report = time.monotonic()
    while any(proc.poll() is None for _, proc, *_ in processes):
        time.sleep(20)
        if time.monotonic() - last_report >= 60:
            completed_shards = sum(proc.poll() == 0 for _, proc, *_ in processes)
            ckpt_tasks = 0
            for _, _, _, _, shard_dir, _ in processes:
                cps = list((shard_dir / "checkpoints").glob("*.json"))
                if cps:
                    try:
                        ckpt_tasks += len(json.loads(cps[0].read_text(encoding="utf-8")).get("results", []))
                    except Exception:
                        pass
            print(f"Progress: {len(done) + ckpt_tasks}/392 tasks checkpointed; {completed_shards}/{len(shards)} shards finished", flush=True)
            last_report = time.monotonic()

    failures = [(i, proc.returncode, shard_dir) for i, proc, _, _, shard_dir, _ in processes if proc.returncode != 0]
    for _, _, log, *_ in processes:
        log.close()
    if failures:
        raise RuntimeError(f"Shard failures: {failures}; outputs/checkpoints retained in {RUN_DIR}")

    results = list(partial["results"])
    worker_results = []
    for _, _, _, _, _, out_file in processes:
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        worker_results.append(payload)
        results.extend(payload["per_sample_results"])
    if len(results) != 392 or len({r["observation_set_id"] for r in results}) != 392:
        raise RuntimeError(f"Incomplete or duplicate task set: {len(results)} results")

    results.sort(key=lambda r: r["observation_set_id"])
    queries = sum(r["n_queries"] for r in results)
    errors = [e for r in results for e in r.get("errors", [])]
    prompt_tokens = sum(r.get("token_usage", {}).get("prompt_tokens", 0) for r in results)
    completion_tokens = sum(r.get("token_usage", {}).get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
    cost = sum(r.get("cost", 0.0) for r in results)
    metrics = {k: stats_block([r[k] for r in results]) for k in ("parse_success_rate", "valid_rate", "novelty_rate", "recovery_rate")}
    error_types = {}
    for e in errors:
        typ = e.get("error_type", "unknown")
        error_types[typ] = error_types.get(typ, 0) + 1
    output = {
        "run_id": "full_causal_n4_obs3_independent_20261125",
        "llm_name": partial["llm_name"], "n_samples": len(results),
        "n_queries_per_sample": None, "query_multiplier": 1.0,
        "query_mode": "adaptive_1.0x_ground_truth_count", "independent_samples": True,
        "seed": 33550336, "timestamp": datetime.now().isoformat(),
        "max_edges_constraint": 6, "statistics": metrics,
        "token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens, "avg_tokens_per_sample": total_tokens / len(results),
                        "avg_tokens_per_query": total_tokens / queries if queries else 0},
        "cost": {"total_cost": cost, "avg_cost_per_sample": cost / len(results),
                 "avg_cost_per_query": cost / queries if queries else 0},
        "error_summary": {"total_errors": len(errors), "error_types": error_types,
                          "error_rate": len(errors) / queries if queries else 0},
        "sampling_protocol": {"dataset": str(DATASET.relative_to(ROOT)), "tasks": 392,
                              "planned_queries": 2172, "completed_queries": queries,
                              "paper_budget": "N = |H_O| per task", "parallel_workers": WORKERS,
                              "preserved_serial_tasks": len(done)},
        "per_sample_results": results,
    }
    FINAL.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"COMPLETE: {len(results)} tasks, {queries} queries, cost=${cost:.4f}; saved {FINAL}", flush=True)


if __name__ == "__main__":
    main()
