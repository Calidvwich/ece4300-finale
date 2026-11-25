"""Paired DeepSeek baseline vs learned graph-edit sampler on one HypoSpace task."""
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
CAUSAL = HERE.parents[1]
sys.path.insert(0, str(CAUSAL))
from evaluate_cached_ranker import features, sigmoid
from run_causal_benchmark import CausalBenchmarkEnhanced, setup_llm

DATA = CAUSAL / "benchmark_obs_count" / "n5_obs1to5_80each.json"
OUT = HERE / "graph_edit_api_ab_n1_006.json"
TARGET_ID = "n1_006"
BUDGET = 100
NODES = ["A", "B", "C", "D", "E"]
PAIRS = [(a, b) for a in NODES for b in NODES if a != b]


def graph_hash(graph):
    return hashlib.sha256(json.dumps(sorted(map(tuple, graph["edges"])), separators=(",", ":")).encode()).hexdigest()


def is_dag(edges):
    adj = {n: [] for n in NODES}
    for a, b in edges:
        if a == b or a not in adj or b not in adj:
            return False
        adj[a].append(b)
    state = {n: 0 for n in NODES}

    def visit(node):
        if state[node] == 1:
            return False
        if state[node] == 2:
            return True
        state[node] = 1
        if not all(visit(child) for child in adj[node]):
            return False
        state[node] = 2
        return True

    return all(visit(n) for n in NODES)


def validate(graph, observations):
    edges = set(map(tuple, graph.get("edges", [])))
    if not is_dag(edges):
        return False
    adj = {n: [] for n in NODES}
    for a, b in edges:
        adj[a].append(b)
    for observation in observations:
        reached, stack = set(), list(adj[observation["perturbed_node"]])
        while stack:
            node = stack.pop()
            if node in reached:
                continue
            reached.add(node)
            stack.extend(adj[node])
        if any(int(n in reached) != int(observation["effects"].get(n, 0)) for n in NODES):
            return False
    return True


def fit_edit_prior(dataset):
    rng = random.Random(83102)
    train = [item for n, group in dataset["datasets_by_n_observations"].items() if int(n) >= 2 for item in group]
    xs, ys = [], []
    for item in rng.sample(train, min(40, len(train))):
        positives = item["ground_truth_graphs"]
        positive_hashes = {graph_hash(g) for g in positives}
        for graph in rng.sample(positives, min(40, len(positives))):
            xs.append(features(graph, item["observations"]))
            ys.append(1.0)
            edges = set(map(tuple, graph["edges"]))
            candidates = PAIRS[:]
            rng.shuffle(candidates)
            made = 0
            for a, b in candidates:
                edited = set(edges)
                if (a, b) in edited:
                    edited.remove((a, b))
                else:
                    edited.add((a, b))
                if is_dag(edited):
                    neighbor = {"nodes": NODES, "edges": sorted(edited)}
                    xs.append(features(neighbor, item["observations"]))
                    ys.append(float(graph_hash(neighbor) in positive_hashes))
                    made += 1
                if made == 2:
                    break
    x, y = np.stack(xs), np.asarray(ys, dtype=float)
    mean, scale = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / scale
    weights, bias = np.zeros(z.shape[1]), 0.0
    for epoch in range(700):
        p = sigmoid(z @ weights + bias)
        lr = 0.08 / (1 + epoch / 250)
        weights -= lr * ((z.T @ (p - y)) / len(y) + 0.02 * weights)
        bias -= lr * float(np.mean(p - y))

    def score(graph, observations):
        return float(sigmoid(((features(graph, observations) - mean) / scale) @ weights + bias))

    return score, len(y)


def neighbors(graph, rng, seen, limit=12):
    edges = set(map(tuple, graph["edges"]))
    choices = PAIRS[:]
    rng.shuffle(choices)
    found = []
    for a, b in choices:
        edited = set(edges)
        if (a, b) in edited:
            edited.remove((a, b))
        else:
            edited.add((a, b))
        if not is_dag(edited):
            continue
        candidate = {"nodes": NODES, "edges": sorted(edited)}
        key = graph_hash(candidate)
        if key not in seen:
            seen.add(key)
            found.append((candidate, key))
        if len(found) >= limit:
            break
    return found


def summarize(records, gt_hashes, query_budget):
    parsed = [r for r in records if r.get("graph")]
    valid = [r for r in parsed if r["valid"]]
    unique_all = {graph_hash(r["graph"]) for r in parsed}
    unique_valid = {graph_hash(r["graph"]) for r in valid}
    recovered = unique_valid & gt_hashes
    return {
        "n_queries": query_budget,
        "parse_success": len(parsed),
        "n_valid": len(valid),
        "n_unique_all": len(unique_all),
        "n_unique_valid": len(unique_valid),
        "n_recovered_gt": len(recovered),
        "validity": len(valid) / query_budget,
        "uniqueness": len(unique_all) / query_budget,
        "recovery": len(recovered) / len(gt_hashes),
        "prompt_tokens": sum(r.get("usage", {}).get("prompt_tokens", 0) for r in records),
        "completion_tokens": sum(r.get("usage", {}).get("completion_tokens", 0) for r in records),
        "cost": sum(r.get("cost", 0.0) for r in records),
    }


def save(result):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temp.replace(OUT)


def main():
    api_key = os.environ.get("HYPOSPACE_DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("HYPOSPACE_DEEPSEEK_API_KEY is not available to this process")
    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    item = next(it for it in dataset["datasets_by_n_observations"]["1"] if it["observation_set_id"] == TARGET_ID)
    observations = item["observations"]
    gt_hashes = {graph_hash(g) for g in item["ground_truth_graphs"]}
    score, train_rows = fit_edit_prior(dataset)
    bench = CausalBenchmarkEnhanced(str(DATA))
    llm = setup_llm("deepseek", model="deepseek-flash", api_key=api_key, temperature=0.7)
    result = {
        "experiment": "DeepSeek baseline vs learned compatibility-guided DAG edit sampler",
        "target_id": TARGET_ID,
        "n_nodes": 5,
        "n_observations": len(observations),
        "n_ground_truths": len(gt_hashes),
        "budget_per_arm": BUDGET,
        "ml_training": "40 training observation sets with n_observations>=2; logistic regression on graph/observation features; test set excluded",
        "learned_train_rows": train_rows,
        "model": "deepseek-flash",
        "temperature": 0.7,
        "arms": {},
    }
    if OUT.exists():
        old = json.loads(OUT.read_text(encoding="utf-8"))
        if old.get("target_id") == TARGET_ID and old.get("budget_per_arm") == BUDGET:
            result = old

    # Baseline: standard benchmark prompt and iterative history, no ML edit module.
    records = result["arms"].setdefault("deepseek_baseline", [])
    hypotheses = [r["graph"] for r in records if r.get("graph")]
    for q in range(len(records), BUDGET):
        prompt = bench.create_prompt(observations, hypotheses)
        data = llm.query_with_usage(prompt)
        text = data["response"]
        graph = bench.parse_llm_response(text)
        rec = {"query": q + 1, "response": text, "graph": graph.to_dict() if graph else None,
               "valid": bool(graph and bench.validate_hypothesis(graph, observations)),
               "usage": data.get("usage", {}), "cost": data.get("cost", 0.0)}
        records.append(rec)
        if graph:
            hypotheses.append(graph)
        result["arm_metrics"] = {"deepseek_baseline": summarize(records, gt_hashes, len(records))}
        save(result)
        print(f"baseline {q+1}/{BUDGET}: valid={rec['valid']} cost=${rec['cost']:.4f}", flush=True)
        time.sleep(0.15)

    # ML arm: one fresh DeepSeek proposal followed by 99 learned, unique local DAG edits.
    records = result["arms"].setdefault("learned_graph_edit", [])
    if not records:
        prompt = bench.create_prompt(observations, [])
        data = llm.query_with_usage(prompt)
        text = data["response"]
        graph = bench.parse_llm_response(text)
        rec = {"query": 1, "kind": "deepseek_seed", "response": text,
               "graph": graph.to_dict() if graph else None,
               "valid": bool(graph and bench.validate_hypothesis(graph, observations)),
               "usage": data.get("usage", {}), "cost": data.get("cost", 0.0)}
        records.append(rec)
        result["arm_metrics"] = {"deepseek_baseline": summarize(result["arms"]["deepseek_baseline"], gt_hashes, BUDGET)}
        save(result)
        time.sleep(0.15)
    seed_graph = next((r["graph"] for r in records if r.get("graph")), None)
    if seed_graph is None:
        raise RuntimeError("DeepSeek seed could not be parsed; result saved for inspection")
    rng = random.Random(83102)
    seen = {graph_hash(seed_graph)}
    frontier = [seed_graph] if validate(seed_graph, observations) else []
    while len(records) < BUDGET:
        candidates = []
        for parent in frontier:
            for candidate, key in neighbors(parent, rng, seen):
                candidates.append((score(candidate, observations), candidate, key))
        if not candidates:
            break
        candidates.sort(key=lambda x: (-x[0], x[2]))
        new_frontier = []
        for confidence, candidate, key in candidates:
            if len(records) >= BUDGET:
                break
            valid = validate(candidate, observations)
            rec = {"query": len(records) + 1, "kind": "learned_edit", "graph": candidate,
                   "valid": valid, "learned_compatibility_score": confidence,
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, "cost": 0.0}
            records.append(rec)
            if valid:
                new_frontier.append(candidate)
            result["arm_metrics"] = {
                "deepseek_baseline": summarize(result["arms"]["deepseek_baseline"], gt_hashes, BUDGET),
                "learned_graph_edit": summarize(records, gt_hashes, len(records)),
            }
            save(result)
        frontier = new_frontier
        print(f"learned_edit {len(records)}/{BUDGET}: valid={sum(r['valid'] for r in records)}", flush=True)
    result["arm_metrics"] = {
        "deepseek_baseline": summarize(result["arms"]["deepseek_baseline"], gt_hashes, BUDGET),
        "learned_graph_edit": summarize(records, gt_hashes, len(records)),
    }
    result["comparison"] = {
        metric: result["arm_metrics"]["learned_graph_edit"][metric] - result["arm_metrics"]["deepseek_baseline"][metric]
        for metric in ("validity", "uniqueness", "recovery")
    }
    save(result)
    print(json.dumps({"arm_metrics": result["arm_metrics"], "comparison": result["comparison"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
