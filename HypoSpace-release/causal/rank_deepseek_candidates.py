"""Learned post-hoc candidate selector for cached DeepSeek graph proposals.

Expected input JSON: {"train": [{"features": [...], "label": 0|1}],
"test": [{"observation_set_id": str, "candidates": [...]}]}.
Labels are produced from training split only; test labels/ground truth are never read.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25, 25)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=33550336)
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    train = data["train"]
    x = np.asarray([r["features"] for r in train], float)
    y = np.asarray([r["label"] for r in train], float)
    if len(set(y.tolist())) < 2:
        raise ValueError("Training labels need both positive and negative examples")
    mean, scale = x.mean(0), x.std(0) + 1e-6
    x = (x - mean) / scale
    rng = np.random.default_rng(args.seed)
    w = np.zeros(x.shape[1]); b = 0.0
    for epoch in range(1800):
        p = sigmoid(x @ w + b)
        lr = .12 / (1 + epoch / 500)
        w -= lr * (x.T @ (p-y) / len(y) + .01*w)
        b -= lr * float(np.mean(p-y))
    selected = []
    for row in data["test"]:
        cand = row["candidates"]
        scores = [float(sigmoid(np.dot((np.asarray(c["features"], float)-mean)/scale, w)+b)) for c in cand]
        order = sorted(range(len(cand)), key=lambda i: (-scores[i], i))
        selected.append({"observation_set_id": row["observation_set_id"], "baseline_order": cand[:args.top_k], "ranked_order": [cand[i] for i in order[:args.top_k]], "scores": [scores[i] for i in order[:args.top_k]]})
    result = {"method": "deepseek_plus_learned_logistic_reranker", "train_candidates": len(train), "top_k": args.top_k, "test": selected}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved reranked held-out pools: {len(selected)}; trained on {len(train)} candidates")


if __name__ == "__main__":
    main()
