"""
因果图数据集加载、Fixed-Prompt 构造、无泄露数据切分与 DeepSeek 基线统计工具。

核心原则：
1. Prompt 格式与官方 `HypoSpace-release/causal/run_causal_benchmark.py` 的
   `independent_samples=True`（Fixed Prompt）完全一致，确保实验协议严格对齐。
2. 训练 / 验证 / 测试集严格按 `observation_set_id` 隔离，零重叠、零泄露。
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from config import DataConfig


def load_causal_dataset(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    加载 HypoSpace Causal 数据集，兼容三种官方 JSON 结构。

    Args:
        path: 数据集 JSON 文件路径。

    Returns:
        (metadata, tasks) 元组，每个 task 均已补齐 `nodes`、`max_edges` 与 `n_compatible_graphs`。
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = raw.get("metadata", {})
    default_nodes: List[str] = list(metadata.get("nodes", []))
    default_max_edges: Optional[int] = metadata.get("max_edges", None)

    tasks: List[Dict[str, Any]] = []
    if "datasets_by_n_observations" in raw:
        for group in raw["datasets_by_n_observations"].values():
            tasks.extend(group)
    elif "datasets" in raw:
        tasks = list(raw["datasets"])
    elif "sampled_datasets" in raw:
        tasks = list(raw["sampled_datasets"])
    else:
        raise ValueError(f"Unsupported causal dataset schema in {path}")

    # 若 metadata 未显式提供 max_edges，按官方逻辑从 ground truth 推断
    inferred_max_edges = default_max_edges
    if inferred_max_edges is None and tasks:
        max_gt_edges = 0
        for item in tasks:
            for gt in item.get("ground_truth_graphs", []):
                max_gt_edges = max(max_gt_edges, len(gt.get("edges", [])))
        inferred_max_edges = max_gt_edges

    normalized: List[Dict[str, Any]] = []
    for item in tasks:
        rec = dict(item)
        if not rec.get("nodes"):
            rec["nodes"] = list(default_nodes)
        if rec.get("max_edges") is None:
            rec["max_edges"] = inferred_max_edges
        if "n_compatible_graphs" not in rec:
            rec["n_compatible_graphs"] = len(rec.get("ground_truth_graphs", []))
        normalized.append(rec)

    return metadata, normalized


def build_fixed_prompt(
    nodes: Sequence[str],
    observations: Sequence[Dict[str, Any]],
    max_edges: Optional[int] = None,
) -> str:
    """
    构造与官方 `CausalBenchmarkEnhanced.create_prompt(..., prior_hypotheses=[])`
    完全一致的 Fixed-Prompt。
    """
    nodes_str = ", ".join(nodes)
    obs_block = "\n".join(obs["string"] for obs in observations)
    constraint_line = (
        f"\nConstraint: The graph should have at most {max_edges} edges."
        if max_edges is not None
        else ""
    )

    sections = [
        "You are given observations from perturbation experiments on a causal system.",
        "",
        "Semantics:",
        "- When a node is perturbed, the perturbed node is 0.",
        "- A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.",
        "- All other nodes are 0.",
        "",
        f"Nodes: {nodes_str}{constraint_line}",
        "",
        "Observations:",
        obs_block,
        "",
        "Prior predictions (do not repeat if avoidable):",
        "None",
        "",
        "Task:",
        "Output a single directed acyclic graph (DAG) over the nodes above that explains all observations.",
        "",
        "Diversity rule:",
        '- A "diverse" graph is any valid graph whose edge set is NOT identical to any prior prediction.',
        "- Generate diverse graphs when possible to explore the solution space.",
        "",
        "Formatting rules:",
        "1) Use only the listed nodes. No self-loops. No cycles.",
        "2) Respond with exactly one line:",
        "- If there are edges: Graph: A->B, B->C",
        "- If there are no edges: Graph: No edges",
    ]
    return "\n".join(sections)


def format_graph_target(
    edges: Sequence[Sequence[str]],
    shuffle_edges: bool = False,
    rng: Optional[random.Random] = None,
) -> str:
    """将边列表格式化为官方要求的单行输出格式 `Graph: ...`。"""
    if not edges:
        return "Graph: No edges"
    edge_list = [(str(u), str(v)) for u, v in edges]
    if shuffle_edges:
        generator = rng if rng is not None else random
        generator.shuffle(edge_list)
    else:
        edge_list.sort()
    return "Graph: " + ", ".join(f"{u}->{v}" for u, v in edge_list)


def get_gt_bucket(n_gt: int) -> str:
    """将假设空间大小 |H_O| 划分为 4 个分层桶，便于分层抽样与分桶评测。"""
    if n_gt <= 1:
        return "1"
    if n_gt <= 4:
        return "2-4"
    if n_gt <= 8:
        return "5-8"
    return "9+"


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """
    使用模型自带 chat template 构造输入文本；若无 template 则退回纯文本提示。
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt + "\n"


def create_leakage_free_splits(
    n3_path: Path,
    n4_path: Path,
    output_path: Path,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 33550336,
) -> Dict[str, Any]:
    """
    构建严格无泄露的 Train / Val / Test 切分并保存到 JSON。

    切分策略：
    - `n4_obs3_all.json`（392 题）按 |H_O| 桶（1, 2-4, 5-8, 9+）分层切分为 Train / Val / Test；
    - `n3_all_observations.json`（19 题，3 节点）全部放入 Train，提供小图热身；
    - 严格校验 `n4` 内部 `train / val / test` 的 `observation_set_id` 两两交集为空。
    """
    rng = random.Random(seed)
    _, n3_tasks = load_causal_dataset(n3_path)
    _, n4_tasks = load_causal_dataset(n4_path)

    buckets: Dict[str, List[Dict[str, Any]]] = {"1": [], "2-4": [], "5-8": [], "9+": []}
    for task in n4_tasks:
        t = dict(task)
        t["source_dataset"] = "n4_obs3_all"
        buckets[get_gt_bucket(int(t["n_compatible_graphs"]))].append(t)

    n4_train: List[Dict[str, Any]] = []
    n4_val: List[Dict[str, Any]] = []
    n4_test: List[Dict[str, Any]] = []

    for bucket_name in ("1", "2-4", "5-8", "9+"):
        items = buckets[bucket_name]
        rng.shuffle(items)
        total = len(items)
        n_test = max(1, round(total * test_ratio))
        n_val = max(1, round(total * val_ratio))
        if n_test + n_val >= total:
            raise ValueError(f"Bucket {bucket_name} too small ({total}) for split ratios")

        n4_test.extend(items[:n_test])
        n4_val.extend(items[n_test : n_test + n_val])
        n4_train.extend(items[n_test + n_val :])

    train_ids_n4 = {t["observation_set_id"] for t in n4_train}
    val_ids_n4 = {t["observation_set_id"] for t in n4_val}
    test_ids_n4 = {t["observation_set_id"] for t in n4_test}

    # 强制断言零泄露
    assert not (train_ids_n4 & val_ids_n4), "Train/Val ID overlap detected!"
    assert not (train_ids_n4 & test_ids_n4), "Train/Test ID overlap detected!"
    assert not (val_ids_n4 & test_ids_n4), "Val/Test ID overlap detected!"

    n3_train: List[Dict[str, Any]] = []
    for task in n3_tasks:
        t = dict(task)
        t["source_dataset"] = "n3_all_observations"
        n3_train.append(t)

    # 按 ID 排序保证可复现与可读性
    n3_train.sort(key=lambda x: x["observation_set_id"])
    n4_train.sort(key=lambda x: x["observation_set_id"])
    n4_val.sort(key=lambda x: x["observation_set_id"])
    n4_test.sort(key=lambda x: x["observation_set_id"])

    train_tasks = n3_train + n4_train

    def _bucket_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"1": 0, "2-4": 0, "5-8": 0, "9+": 0}
        for it in items:
            counts[get_gt_bucket(int(it["n_compatible_graphs"]))] += 1
        return counts

    payload: Dict[str, Any] = {
        "metadata": {
            "seed": seed,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n3_source": str(n3_path.name),
            "n4_source": str(n4_path.name),
            "counts": {
                "train_total": len(train_tasks),
                "train_n3": len(n3_train),
                "train_n4": len(n4_train),
                "val_n4": len(n4_val),
                "test_n4": len(n4_test),
            },
            "bucket_distribution_n4": {
                "train": _bucket_counts(n4_train),
                "val": _bucket_counts(n4_val),
                "test": _bucket_counts(n4_test),
            },
        },
        "split_ids": {
            "train_n3": [t["observation_set_id"] for t in n3_train],
            "train_n4": [t["observation_set_id"] for t in n4_train],
            "val": [t["observation_set_id"] for t in n4_val],
            "test": [t["observation_set_id"] for t in n4_test],
        },
        "splits": {
            "train": train_tasks,
            "val": n4_val,
            "test": n4_test,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_baseline_on_split(
    baseline_json_path: Path,
    target_ids: Iterable[str],
) -> Dict[str, Any]:
    """
    从队友保存的 DeepSeek-Flash Fixed-Prompt 全量评测 JSON 中，
    精确抽取指定测试集 ID（`target_ids`）上的配对基线指标。
    """
    id_set: Set[str] = set(target_ids)
    raw = json.loads(Path(baseline_json_path).read_text(encoding="utf-8"))
    matched = [
        r for r in raw.get("per_sample_results", []) if r.get("observation_set_id") in id_set
    ]
    if not matched:
        raise ValueError(f"No matching observation_set_id found in {baseline_json_path}")

    n = len(matched)
    macro_parse = sum(r.get("parse_success_rate", 0.0) for r in matched) / n
    macro_valid = sum(r.get("valid_rate", 0.0) for r in matched) / n
    macro_unique = sum(r.get("novelty_rate", 0.0) for r in matched) / n
    macro_recovery = sum(r.get("recovery_rate", 0.0) for r in matched) / n

    total_queries = sum(r.get("n_queries", 0) for r in matched)
    total_valid = sum(r.get("n_valid", 0) for r in matched)
    total_unique_valid = sum(r.get("n_unique_valid", 0) for r in matched)
    total_gt = sum(r.get("n_ground_truths", 0) for r in matched)
    total_recovered = sum(r.get("n_recovered_gts", 0) for r in matched)

    by_bucket: Dict[str, Dict[str, float]] = {}
    for bucket_name in ("1", "2-4", "5-8", "9+"):
        sub = [
            r
            for r in matched
            if get_gt_bucket(int(r.get("n_ground_truths", 0))) == bucket_name
        ]
        if sub:
            by_bucket[bucket_name] = {
                "n_tasks": len(sub),
                "valid_rate": sum(r.get("valid_rate", 0.0) for r in sub) / len(sub),
                "uniqueness_rate": sum(r.get("novelty_rate", 0.0) for r in sub) / len(sub),
                "recovery_rate": sum(r.get("recovery_rate", 0.0) for r in sub) / len(sub),
            }

    return {
        "llm_name": raw.get("llm_name", "baseline"),
        "matched_tasks": n,
        "total_queries": total_queries,
        "total_ground_truths": total_gt,
        "macro": {
            "parse_success_rate": macro_parse,
            "valid_rate": macro_valid,
            "uniqueness_rate": macro_unique,
            "recovery_rate": macro_recovery,
        },
        "pooled": {
            "valid_rate": total_valid / max(1, total_queries),
            "uniqueness_rate": total_unique_valid / max(1, total_queries),
            "recovery_rate": total_recovered / max(1, total_gt),
        },
        "by_gt_bucket": by_bucket,
    }


def main() -> None:
    """命令行入口：生成无泄露切分文件并打印配对 DeepSeek 基线指标。"""
    cfg = DataConfig()
    parser = argparse.ArgumentParser(description="Create leakage-free causal splits")
    parser.add_argument("--n3", type=Path, default=cfg.n3_dataset_path)
    parser.add_argument("--n4", type=Path, default=cfg.n4_dataset_path)
    parser.add_argument("--output", type=Path, default=cfg.splits_output_path)
    parser.add_argument("--val-ratio", type=float, default=cfg.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=cfg.test_ratio)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--baseline", type=Path, default=cfg.deepseek_baseline_path)
    args = parser.parse_args()

    payload = create_leakage_free_splits(
        n3_path=args.n3,
        n4_path=args.n4,
        output_path=args.output,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Saved splits to: {args.output}")
    print(json.dumps(payload["metadata"], indent=2))

    if args.baseline.exists():
        test_ids = payload["split_ids"]["test"]
        baseline_stats = summarize_baseline_on_split(args.baseline, test_ids)
        print("\nPaired DeepSeek-Flash (Fixed-Prompt) on ID Test split:")
        print(json.dumps(baseline_stats, indent=2))


if __name__ == "__main__":
    main()
