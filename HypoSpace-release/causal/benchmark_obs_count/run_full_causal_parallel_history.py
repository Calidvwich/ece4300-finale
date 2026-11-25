import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "causal" / "datasets" / "n4_obs3_all.json"
RUN_DIR = ROOT / "causal" / "benchmark_obs_count" / "full_api_causal_n4_obs3_history"
FINAL = ROOT / "causal" / "benchmark_obs_count" / "full_repo_causal_n4_obs3_deepseek_history.json"
PYTHON = Path(sys.executable)
WORKERS = 8


def stats_block(values):
    values = np.asarray(values, dtype=float)
    p = float(stats.ttest_1samp(values, 0).pvalue) if len(values) > 1 else None
    return {"mean": float(values.mean()), "std": float(values.std()), "var": float(values.var()),
            "min": float(values.min()), "max": float(values.max()), "p_value_vs_zero": p}


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    tasks = data["datasets"]
    shard_size = math.ceil(len(tasks) / WORKERS)
    shards = [tasks[i:i + shard_size] for i in range(0, len(tasks), shard_size)]
    processes = []
    for i, shard in enumerate(shards):
        shard_dir = RUN_DIR / f"shard_{i:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        ds = dict(data)
        ds["datasets"] = shard
        ds["metadata"] = dict(data.get("metadata", {}), n_datasets=len(shard))
        shard_file, out_file = shard_dir / "dataset.json", shard_dir / "result.json"
        shard_file.write_text(json.dumps(ds), encoding="utf-8")
        log = (shard_dir / "run.log").open("w", encoding="utf-8")
        cmd = [str(PYTHON), "-u", str(ROOT / "causal" / "run_causal_benchmark.py"),
               "--dataset", str(shard_file), "--config", str(ROOT / "causal" / "config" / "config_deepseek.yaml"),
               "--n-samples", str(len(shard)), "--query-multiplier", "1", "--max-retries", "3",
               "--seed", str(33550400 + i), "--checkpoint-dir", str(shard_dir / "checkpoints"),
               "--output", str(out_file), "--quiet"]
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        processes.append((i, proc, log, shard_dir, out_file))
        print(f"Started history shard {i + 1}/{len(shards)} ({len(shard)} tasks), PID={proc.pid}", flush=True)

    last = time.monotonic()
    while any(proc.poll() is None for _, proc, *_ in processes):
        time.sleep(20)
        if time.monotonic() - last >= 60:
            done_shards = sum(proc.poll() == 0 for _, proc, *_ in processes)
            count = 0
            for _, _, _, shard_dir, out_file in processes:
                if out_file.exists():
                    try:
                        count += json.loads(out_file.read_text(encoding="utf-8"))["n_samples"]
                    except Exception:
                        pass
                else:
                    for cp in (shard_dir / "checkpoints").glob("*.json"):
                        try:
                            count += len(json.loads(cp.read_text(encoding="utf-8")).get("results", []))
                        except Exception:
                            pass
            print(f"History-mode progress: {count}/392 tasks; {done_shards}/8 shards finished", flush=True)
            last = time.monotonic()

    for _, _, log, *_ in processes:
        log.close()
    failed = [(i, proc.returncode, str(shard_dir)) for i, proc, _, shard_dir, _ in processes if proc.returncode != 0]
    if failed:
        raise RuntimeError(f"History-mode shard failures: {failed}; retained logs/checkpoints in {RUN_DIR}")

    payloads = [json.loads(out_file.read_text(encoding="utf-8")) for _, _, _, _, out_file in processes]
    results = [r for p in payloads for r in p["per_sample_results"]]
    results.sort(key=lambda r: r["observation_set_id"])
    ids = {r["observation_set_id"] for r in results}
    expected_ids = {t["observation_set_id"] for t in tasks}
    if len(results) != 392 or ids != expected_ids:
        raise RuntimeError(f"Task mismatch: rows={len(results)}, unique={len(ids)}, expected={len(expected_ids)}")
    expected_q = sum(len(t["ground_truth_graphs"]) for t in tasks)
    queries = sum(r["n_queries"] for r in results)
    if queries != expected_q:
        raise RuntimeError(f"Query budget mismatch: {queries} != {expected_q}")

    errors = [e for r in results for e in r.get("errors", [])]
    error_types = {}
    for e in errors:
        typ = e.get("error_type", "unknown")
        error_types[typ] = error_types.get(typ, 0) + 1
    prompt_tokens = sum(r.get("token_usage", {}).get("prompt_tokens", 0) for r in results)
    completion_tokens = sum(r.get("token_usage", {}).get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in results)
    cost = sum(r.get("cost", 0.0) for r in results)
    output = {
        "run_id": "full_causal_n4_obs3_history_conditioned_20261125",
        "llm_name": payloads[0]["llm_name"], "n_samples": len(results),
        "n_queries_per_sample": None, "query_multiplier": 1.0,
        "query_mode": "adaptive_1.0x_ground_truth_count", "independent_samples": False,
        "seed": 33550400, "timestamp": datetime.now().isoformat(), "max_edges_constraint": 6,
        "statistics": {k: stats_block([r[k] for r in results]) for k in
                       ("parse_success_rate", "valid_rate", "novelty_rate", "recovery_rate")},
        "token_usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens, "avg_tokens_per_sample": total_tokens / len(results),
                        "avg_tokens_per_query": total_tokens / queries},
        "cost": {"total_cost": cost, "avg_cost_per_sample": cost / len(results),
                 "avg_cost_per_query": cost / queries},
        "error_summary": {"total_errors": len(errors), "error_types": error_types,
                          "error_rate": len(errors) / queries},
        "sampling_protocol": {"dataset": str(DATASET.relative_to(ROOT)), "tasks": 392,
                              "planned_queries": expected_q, "completed_queries": queries,
                              "paper_budget": "N = |H_O| per task", "parallel_workers": WORKERS,
                              "prompt_history": "prior predictions included (clone default)",
                              "history_scope": "within each observation-set run"},
        "per_sample_results": results,
    }
    FINAL.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"COMPLETE history mode: {len(results)} tasks, {queries} queries, cost=${cost:.4f}; saved {FINAL}", flush=True)


if __name__ == "__main__":
    main()
