"""
因果图数据集加载、Fixed-Prompt 构造、零同构泄露切分与 DeepSeek 基线统计工具（v2.0）。

核心升级（解决节点置换同构泄露）：
- 4 节点 `n4_obs3_all.json` 的 392 题本质上仅由 22 个因果观测同构类（在 A/B/C/D 节点重命名置换下等价）
  构成；若仅按 `observation_set_id` 随机切分，全部 22 个结构类都会落入训练集。
- 本模块通过 `canonical_isomorphism_key` 计算每道题在全节点置换群 S_n 下的字典序最小规范签名，
  以**同构结构类**为最小不可分割单元按 |H_O| 分桶整组切分，确保 Train / Val / Test 之间
  既无 `observation_set_id` 重合，也无节点换名等价结构重合（零同构泄露）。
"""

import argparse
import itertools
import json
import random
from collections import defaultdict
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


def canonical_isomorphism_key(task: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    计算任务在节点名称全置换群（如 A/B/C/D 的 4!=24 种重命名）下的规范同构签名。

    包含 `(节点数, max_edges, 规范化观测元组)`：
    两道题若仅差节点字母重命名，其返回的签名严格相同。
    """
    nodes = list(task["nodes"])
    observations = task["observations"]
    if len(nodes) > 6:
        raise ValueError(f"Permutation canonicalization supports <=6 nodes, got {len(nodes)}")

    best_obs: Optional[Tuple[Any, ...]] = None
    for perm in itertools.permutations(nodes):
        mapping = dict(zip(nodes, perm))
        obs_tuples = tuple(
            sorted(
                (
                    mapping[obs["perturbed_node"]],
                    tuple(sorted((mapping[n], int(obs["effects"].get(n, 0))) for n in nodes)),
                )
                for obs in observations
            )
        )
        if best_obs is None or obs_tuples < best_obs:
            best_obs = obs_tuples

    return (len(nodes), task.get("max_edges"), best_obs)


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
    seed: int = 33550336,
) -> Dict[str, Any]:
    """
    构建**零同构泄露**的 Train / Val / Test 分层整组切分并保存到 JSON。

    切分策略（以 22 个 4 节点同构结构类为最小分配单元）：
    - 先用 `canonical_isomorphism_key` 把 `n4_obs3_all.json` 的 392 题聚类为 22 个同构结构组；
    - 按 |H_O| 桶（`1`, `2-4`, `5-8`, `9+`）分层分配整组结构：
      - 每个桶分配 1 个完整结构组进入 `Val`（共 4 组，覆盖全难度桶）；
      - 每个桶分配 1 个完整结构组进入 `Test`（另在组数最多的 `2-4` 桶多分配 1 组，共 5 组，覆盖全难度桶）；
      - 其余 13 个完整结构组全部进入 `Train`；
    - `n3_all_observations.json`（19 题，3 节点）全部加入 `Train`；
    - 严格断言 `n4` 的 `Train / Val / Test` 之间在 **题目 ID** 和 **同构结构键** 两个层级上交集均为 0。
    """
    rng = random.Random(seed)
    _, n3_tasks = load_causal_dataset(n3_path)
    _, n4_tasks = load_causal_dataset(n4_path)

    iso_groups_map: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for task in n4_tasks:
        t = dict(task)
        t["source_dataset"] = "n4_obs3_all"
        iso_groups_map[canonical_isomorphism_key(t)].append(t)

    # 按组内代表 ID 稳定排序后，根据 |H_O| 分桶
    bucket_groups: Dict[str, List[Tuple[Tuple[Any, ...], List[Dict[str, Any]]]]] = {
        "1": [],
        "2-4": [],
        "5-8": [],
        "9+": [],
    }
    for key, grp_tasks in iso_groups_map.items():
        grp_tasks_sorted = sorted(grp_tasks, key=lambda x: x["observation_set_id"])
        bucket = get_gt_bucket(int(grp_tasks_sorted[0]["n_compatible_graphs"]))
        bucket_groups[bucket].append((key, grp_tasks_sorted))

    # 每个桶内：保证 Test / Val 覆盖全桶，同时 Train 保留多数结构组学习规律
    test_group_quota = {"1": 1, "2-4": 2, "5-8": 1, "9+": 1}
    val_group_quota = {"1": 1, "2-4": 1, "5-8": 1, "9+": 1}

    n4_train: List[Dict[str, Any]] = []
    n4_val: List[Dict[str, Any]] = []
    n4_test: List[Dict[str, Any]] = []
    group_counts_by_split: Dict[str, Dict[str, int]] = {
        "train": {"1": 0, "2-4": 0, "5-8": 0, "9+": 0},
        "val": {"1": 0, "2-4": 0, "5-8": 0, "9+": 0},
        "test": {"1": 0, "2-4": 0, "5-8": 0, "9+": 0},
    }

    for bucket_name in ("1", "2-4", "5-8", "9+"):
        groups = sorted(
            bucket_groups[bucket_name],
            key=lambda item: item[1][0]["observation_set_id"],
        )
        rng.shuffle(groups)
        n_test_g = test_group_quota[bucket_name]
        n_val_g = val_group_quota[bucket_name]
        if n_test_g + n_val_g >= len(groups):
            raise ValueError(
                f"Bucket {bucket_name} only has {len(groups)} isomorphism groups."
            )

        for _, grp in groups[:n_test_g]:
            n4_test.extend(grp)
            group_counts_by_split["test"][bucket_name] += 1
        for _, grp in groups[n_test_g : n_test_g + n_val_g]:
            n4_val.extend(grp)
            group_counts_by_split["val"][bucket_name] += 1
        for _, grp in groups[n_test_g + n_val_g :]:
            n4_train.extend(grp)
            group_counts_by_split["train"][bucket_name] += 1

    # 双重零泄露断言：题目 ID 不重叠 + 同构结构键不重叠
    train_ids_n4 = {t["observation_set_id"] for t in n4_train}
    val_ids_n4 = {t["observation_set_id"] for t in n4_val}
    test_ids_n4 = {t["observation_set_id"] for t in n4_test}
    assert not (train_ids_n4 & val_ids_n4), "Train/Val ID overlap detected!"
    assert not (train_ids_n4 & test_ids_n4), "Train/Test ID overlap detected!"
    assert not (val_ids_n4 & test_ids_n4), "Val/Test ID overlap detected!"

    train_iso_n4 = {canonical_isomorphism_key(t) for t in n4_train}
    val_iso_n4 = {canonical_isomorphism_key(t) for t in n4_val}
    test_iso_n4 = {canonical_isomorphism_key(t) for t in n4_test}
    assert not (train_iso_n4 & val_iso_n4), "Train/Val isomorphism leakage detected!"
    assert not (train_iso_n4 & test_iso_n4), "Train/Test isomorphism leakage detected!"
    assert not (val_iso_n4 & test_iso_n4), "Val/Test isomorphism leakage detected!"

    n3_train: List[Dict[str, Any]] = []
    for task in n3_tasks:
        t = dict(task)
        t["source_dataset"] = "n3_all_observations"
        n3_train.append(t)

    n3_train.sort(key=lambda x: x["observation_set_id"])
    n4_train.sort(key=lambda x: x["observation_set_id"])
    n4_val.sort(key=lambda x: x["observation_set_id"])
    n4_test.sort(key=lambda x: x["observation_set_id"])
    train_tasks = n3_train + n4_train

    def _bucket_task_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"1": 0, "2-4": 0, "5-8": 0, "9+": 0}
        for it in items:
            counts[get_gt_bucket(int(it["n_compatible_graphs"]))] += 1
        return counts

    payload: Dict[str, Any] = {
        "metadata": {
            "split_strategy": "isomorphism_group_stratified",
            "seed": seed,
            "n3_source": str(n3_path.name),
            "n4_source": str(n4_path.name),
            "total_n4_isomorphism_groups": len(iso_groups_map),
            "counts": {
                "train_total": len(train_tasks),
                "train_n3": len(n3_train),
                "train_n4": len(n4_train),
                "val_n4": len(n4_val),
                "test_n4": len(n4_test),
                "train_n4_groups": len(train_iso_n4),
                "val_n4_groups": len(val_iso_n4),
                "test_n4_groups": len(test_iso_n4),
            },
            "group_distribution_n4": group_counts_by_split,
            "bucket_distribution_n4": {
                "train": _bucket_task_counts(n4_train),
                "val": _bucket_task_counts(n4_val),
                "test": _bucket_task_counts(n4_test),
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
    """命令行入口：生成零同构泄露切分文件并打印配对 DeepSeek 基线指标。"""
    cfg = DataConfig()
    parser = argparse.ArgumentParser(description="Create zero-isomorphism-leakage causal splits")
    parser.add_argument("--n3", type=Path, default=cfg.n3_dataset_path)
    parser.add_argument("--n4", type=Path, default=cfg.n4_dataset_path)
    parser.add_argument("--output", type=Path, default=cfg.splits_output_path)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--baseline", type=Path, default=cfg.deepseek_baseline_path)
    args = parser.parse_args()

    payload = create_leakage_free_splits(
        n3_path=args.n3,
        n4_path=args.n4,
        output_path=args.output,
        seed=args.seed,
    )
    print(f"Saved splits to: {args.output}")
    print(json.dumps(payload["metadata"], indent=2))

    if args.baseline.exists():
        test_ids = payload["split_ids"]["test"]
        baseline_stats = summarize_baseline_on_split(args.baseline, test_ids)
        print("\nPaired DeepSeek-Flash (Fixed-Prompt) on Unseen-Structure Test split:")
        print(json.dumps(baseline_stats, indent=2))


if __name__ == "__main__":
    main()
