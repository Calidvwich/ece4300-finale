#!/usr/bin/env python3
"""Structure-group (D4 symmetry) and cluster-bootstrap statistical analysis for 3D HypoSpace."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRPO_3D_DIR = PROJECT_ROOT / "HypoSpace-release" / "3d" / "grpo_lora"
if str(GRPO_3D_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_3D_DIR))

from data_utils import (  # noqa: E402
    canonical_isomorphism_key,
    get_gt_bucket,
    load_3d_dataset,
)

BUCKET_ORDER: Tuple[str, ...] = ("1", "2-4", "5-8", "9+")
TASK_METRICS: Tuple[str, ...] = (
    "recovery_rate",
    "recovery_norm",
    "valid_rate",
    "uniqueness_rate",
    "parse_success_rate",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    default_splits = GRPO_3D_DIR / "splits" / "3d_splits.json"
    parser = argparse.ArgumentParser(
        description="Analyze 3D HypoSpace evaluation outputs with D4-group aggregation and cluster bootstrap CIs."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=SPEC",
        help="Model spec NAME=PATH[,PATH2...] supporting glob wildcards.",
    )
    parser.add_argument(
        "--splits_path",
        type=Path,
        default=default_splits,
        help=f"Path to 3d_splits.json (default: {default_splits}).",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="Primary split in splits_path ('train', 'val', 'test', or 'all'; default: test).",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Optional 3D dataset JSON (e.g., 3d_g3_h3_b4_ood.json).",
    )
    parser.add_argument(
        "--deepseek_baseline",
        type=Path,
        default=None,
        help="Optional DeepSeek 3D runner JSON (maps novelty_rate -> uniqueness_rate).",
    )
    parser.add_argument(
        "--pairs",
        action="append",
        default=None,
        metavar="A:B",
        help="Pairwise comparison A:B (repeatable). Defaults to standard 3D ablation pairs when present.",
    )
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=33550336)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def theoretical_recovery_ceiling(n_gt: int) -> float:
    """Expected recovery 1 - (1 - 1/K)^K under i.i.d. uniform sampling of K draws."""
    if n_gt <= 0:
        return 0.0
    return 1.0 - (1.0 - 1.0 / float(n_gt)) ** float(n_gt)


def exact_binomial_sign_test(n_pos: int, n_neg: int) -> float:
    """Exact two-sided binomial sign test p-value via math.comb only."""
    n_eff = n_pos + n_neg
    if n_eff == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n_eff, i) for i in range(k + 1))
    return min(1.0, 2.0 * float(tail) / float(2**n_eff))


def resolve_spec_paths(spec: str) -> List[Path]:
    """Expand comma-separated paths and glob patterns."""
    resolved: List[Path] = []
    seen = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        matches = sorted(glob.glob(part))
        candidates = [Path(m) for m in matches] if matches else [Path(part)]
        for path in candidates:
            if not path.exists():
                raise FileNotFoundError(f"Evaluation file not found: {path} (from '{spec}')")
            norm = path.resolve()
            if norm not in seen:
                seen.add(norm)
                resolved.append(path)
    if not resolved:
        raise ValueError(f"No evaluation files resolved from spec: '{spec}'")
    return resolved


def parse_model_specs(raw_specs: Sequence[str]) -> List[Tuple[str, List[Path]]]:
    """Parse repeatable --model NAME=SPEC arguments."""
    models: List[Tuple[str, List[Path]]] = []
    seen_names = set()
    for item in raw_specs:
        if "=" not in item:
            raise ValueError(f"Invalid --model '{item}'; expected NAME=PATH[,PATH2...].")
        name, spec = item.split("=", 1)
        name = name.strip()
        if not name or name in seen_names:
            raise ValueError(f"Invalid or duplicate model name '{name}'.")
        seen_names.add(name)
        models.append((name, resolve_spec_paths(spec)))
    return models


def make_group_uid(key: Tuple[Any, ...]) -> str:
    """Deterministic short hash from D4 canonical_isomorphism_key."""
    return hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:8]


def build_task_metadata(
    splits_path: Path,
    split_name: str,
    dataset_path: Optional[Path],
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Load 3D task metadata and group by D4 canonical_isomorphism_key."""
    tasks_by_id: Dict[str, Dict[str, Any]] = {}
    if dataset_path is not None:
        _, tasks = load_3d_dataset(dataset_path)
        for task in tasks:
            obs_id = str(task["observation_set_id"])
            tasks_by_id[obs_id] = task
        source_desc = f"dataset:{dataset_path}"
    else:
        splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
        splits = splits_data.get("splits", {})
        ordered_splits = (
            list(splits.keys())
            if split_name == "all"
            else [split_name] + [k for k in splits.keys() if k != split_name]
        )
        for s_name in ordered_splits:
            for task in splits.get(s_name, []):
                obs_id = str(task.get("observation_set_id", task.get("observation_id")))
                if obs_id not in tasks_by_id:
                    copied = dict(task)
                    copied["_split"] = s_name
                    tasks_by_id[obs_id] = copied
        source_desc = f"splits:{splits_path} (primary={split_name})"

    key_to_tasks: Dict[Tuple[Any, ...], List[str]] = defaultdict(list)
    raw_keys: Dict[str, Tuple[Any, ...]] = {}
    for obs_id, task in tasks_by_id.items():
        canon_key = canonical_isomorphism_key(task)
        raw_keys[obs_id] = canon_key
        key_to_tasks[canon_key].append(obs_id)

    sorted_keys = sorted(key_to_tasks.keys(), key=repr)
    key_to_label = {
        canon_key: f"D4_{idx:02d}_{make_group_uid(canon_key)}"
        for idx, canon_key in enumerate(sorted_keys, start=1)
    }

    metadata: Dict[str, Dict[str, Any]] = {}
    for obs_id, task in tasks_by_id.items():
        n_gt = int(
            task.get("n_ground_truths")
            or task.get("n_compatible_structures")
            or len(task.get("ground_truth_structures", []))
        )
        canon_key = raw_keys[obs_id]
        metadata[obs_id] = {
            "observation_set_id": obs_id,
            "n_blocks": int(task.get("n_blocks", 0)),
            "n_ground_truths": n_gt,
            "bucket": get_gt_bucket(n_gt),
            "ceiling": theoretical_recovery_ceiling(n_gt),
            "group_id": key_to_label[canon_key],
            "group_hash": make_group_uid(canon_key),
            "split": task.get("_split", "ood"),
        }
    return metadata, source_desc


def extract_task_record(
    entry: Dict[str, Any],
    task_meta: Dict[str, Any],
    is_deepseek: bool = False,
) -> Dict[str, Optional[float]]:
    """Extract normalized scalar metrics for one 3D observation set."""
    n_gt = int(entry.get("n_ground_truths", task_meta["n_ground_truths"]))
    ceiling = theoretical_recovery_ceiling(n_gt)
    recovery = float(entry["recovery_rate"])
    recovery_norm = (recovery / ceiling) if ceiling > 0.0 else 0.0

    if is_deepseek:
        uniq_raw = entry.get("novelty_rate", entry.get("uniqueness_rate"))
    else:
        uniq_raw = entry.get("uniqueness_rate", entry.get("novelty_rate"))
    uniqueness = float(uniq_raw) if uniq_raw is not None else None

    if "parse_success_rate" in entry and entry["parse_success_rate"] is not None:
        parse_rate: Optional[float] = float(entry["parse_success_rate"])
    elif "n_parsed" in entry and "n_queries" in entry and entry["n_queries"]:
        parse_rate = float(entry["n_parsed"]) / float(entry["n_queries"])
    else:
        parse_rate = None

    return {
        "recovery_rate": recovery,
        "recovery_norm": recovery_norm,
        "valid_rate": float(entry["valid_rate"]),
        "uniqueness_rate": uniqueness,
        "parse_success_rate": parse_rate,
        "ceiling": ceiling,
    }


def load_model_task_metrics(
    paths: Sequence[Path],
    metadata: Dict[str, Dict[str, Any]],
    is_deepseek: bool = False,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], int]:
    """Load one or more seed JSON files for a model and average per task across seeds."""
    per_task: Dict[str, List[Dict[str, Optional[float]]]] = defaultdict(list)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("per_sample_results", [])
        if not rows:
            raise ValueError(f"No per_sample_results found in {path}")
        for row in rows:
            obs_id = str(row.get("observation_set_id", row.get("observation_id")))
            if obs_id not in metadata:
                continue
            per_task[obs_id].append(
                extract_task_record(row, metadata[obs_id], is_deepseek=is_deepseek)
            )

    averaged: Dict[str, Dict[str, Optional[float]]] = {}
    for obs_id, records in per_task.items():
        merged: Dict[str, Optional[float]] = {}
        for key in (*TASK_METRICS, "ceiling"):
            vals = [r[key] for r in records if r[key] is not None]
            merged[key] = float(np.mean(vals)) if vals else None
        merged["n_seeds"] = float(len(records))
        averaged[obs_id] = merged
    return averaged, len(paths)


def mean_std(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Compute sample mean and sample std (ddof=1 if len > 1 else 0)."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=np.float64)
    return float(np.mean(arr)), (float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)


def summarize_model(
    model_name: str,
    task_ids: Sequence[str],
    task_metrics: Dict[str, Dict[str, Optional[float]]],
    metadata: Dict[str, Dict[str, Any]],
    n_files: int,
) -> Dict[str, Any]:
    """Compute task-level, bucket-level, and D4-group-level summaries."""
    groups: Dict[str, List[str]] = defaultdict(list)
    buckets: Dict[str, List[str]] = defaultdict(list)
    for obs_id in task_ids:
        meta = metadata[obs_id]
        groups[meta["group_id"]].append(obs_id)
        buckets[meta["bucket"]].append(obs_id)

    task_means: Dict[str, Optional[float]] = {}
    for metric in (*TASK_METRICS, "ceiling"):
        task_means[metric], _ = mean_std(task_metrics[obs_id][metric] for obs_id in task_ids)

    group_rows: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(groups.keys()):
        g_tasks = groups[group_id]
        g_summary: Dict[str, Any] = {
            "group_id": group_id,
            "n_tasks": len(g_tasks),
            "buckets": sorted({metadata[t]["bucket"] for t in g_tasks}, key=BUCKET_ORDER.index),
        }
        for metric in (*TASK_METRICS, "ceiling"):
            g_summary[metric], _ = mean_std(task_metrics[t][metric] for t in g_tasks)
        group_rows[group_id] = g_summary

    group_level_stats: Dict[str, Dict[str, Optional[float]]] = {}
    for metric in (*TASK_METRICS, "ceiling"):
        g_mean, g_std = mean_std(group_rows[g][metric] for g in group_rows)
        group_level_stats[metric] = {"mean": g_mean, "std": g_std}

    bucket_stats: Dict[str, Dict[str, Any]] = {}
    for bucket in BUCKET_ORDER:
        b_tasks = buckets.get(bucket, [])
        if not b_tasks:
            continue
        b_entry: Dict[str, Any] = {
            "n_tasks": len(b_tasks),
            "n_groups": len({metadata[t]["group_id"] for t in b_tasks}),
        }
        for metric in (*TASK_METRICS, "ceiling"):
            b_entry[metric], _ = mean_std(task_metrics[t][metric] for t in b_tasks)
        bucket_stats[bucket] = b_entry

    return {
        "model_name": model_name,
        "n_files": n_files,
        "n_tasks": len(task_ids),
        "n_groups": len(group_rows),
        "task_macro": task_means,
        "group_macro": group_level_stats,
        "by_bucket": bucket_stats,
        "per_group": group_rows,
    }


def compute_pairwise_comparison(
    name_a: str,
    name_b: str,
    task_ids: Sequence[str],
    metrics_a: Dict[str, Dict[str, Optional[float]]],
    metrics_b: Dict[str, Dict[str, Optional[float]]],
    metadata: Dict[str, Dict[str, Any]],
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    """Compare model A vs model B (delta = A - B) using cluster bootstrap over D4 groups."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for obs_id in task_ids:
        groups[metadata[obs_id]["group_id"]].append(obs_id)
    sorted_group_ids = sorted(groups.keys())
    n_groups = len(sorted_group_ids)

    rng = np.random.default_rng(bootstrap_seed)
    boot_indices = rng.integers(0, n_groups, size=(bootstrap_iters, n_groups)) if n_groups else None

    tol = 1e-12
    comparisons: Dict[str, Any] = {}
    for metric in ("recovery_rate", "recovery_norm", "valid_rate", "uniqueness_rate"):
        task_deltas = []
        for obs_id in task_ids:
            va = metrics_a[obs_id].get(metric)
            vb = metrics_b[obs_id].get(metric)
            if va is not None and vb is not None:
                task_deltas.append(float(va) - float(vb))
        if not task_deltas:
            continue

        t_arr = np.asarray(task_deltas, dtype=np.float64)
        t_wins = int(np.sum(t_arr > tol))
        t_losses = int(np.sum(t_arr < -tol))
        t_ties = int(len(t_arr) - t_wins - t_losses)

        group_deltas = []
        for g_id in sorted_group_ids:
            g_tasks = groups[g_id]
            vals_a = [metrics_a[t][metric] for t in g_tasks if metrics_a[t][metric] is not None]
            vals_b = [metrics_b[t][metric] for t in g_tasks if metrics_b[t][metric] is not None]
            if vals_a and vals_b:
                group_deltas.append(float(np.mean(vals_a)) - float(np.mean(vals_b)))

        g_arr = np.asarray(group_deltas, dtype=np.float64)
        g_wins = int(np.sum(g_arr > tol))
        g_losses = int(np.sum(g_arr < -tol))
        g_ties = int(len(g_arr) - g_wins - g_losses)
        sign_p = exact_binomial_sign_test(g_wins, g_losses)

        if len(g_arr) == n_groups and boot_indices is not None and n_groups > 0:
            boot_means = np.mean(g_arr[boot_indices], axis=1)
        elif len(g_arr) > 0:
            local_rng = np.random.default_rng(bootstrap_seed)
            local_idx = local_rng.integers(0, len(g_arr), size=(bootstrap_iters, len(g_arr)))
            boot_means = np.mean(g_arr[local_idx], axis=1)
        else:
            boot_means = np.asarray([0.0], dtype=np.float64)

        comparisons[metric] = {
            "task_level_descriptive": {
                "n_tasks": int(len(t_arr)),
                "mean_delta": float(np.mean(t_arr)),
                "wins": t_wins,
                "ties": t_ties,
                "losses": t_losses,
            },
            "group_level": {
                "n_groups": int(len(g_arr)),
                "mean_delta": float(np.mean(g_arr)),
                "std_delta": float(np.std(g_arr, ddof=1)) if len(g_arr) > 1 else 0.0,
                "wins": g_wins,
                "ties": g_ties,
                "losses": g_losses,
                "sign_test_p": float(sign_p),
                "bootstrap_ci_95": [float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))],
            },
        }

    return {
        "pair": f"{name_a} - {name_b}",
        "model_a": name_a,
        "model_b": name_b,
        "n_common_tasks": len(task_ids),
        "n_common_groups": n_groups,
        "bootstrap_iters": bootstrap_iters,
        "bootstrap_seed": bootstrap_seed,
        "metrics": comparisons,
    }


def fmt_pct(val: Optional[float], signed: bool = False, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    pct = val * 100.0
    return f"{pct:+.{decimals}f}%" if signed else f"{pct:.{decimals}f}%"


def theoretical_recovery_ceiling_mean(task_ids: Sequence[str], metadata: Dict[str, Dict[str, Any]]) -> float:
    return float(np.mean([metadata[t]["ceiling"] for t in task_ids])) if task_ids else 0.0


def resolve_pairs(raw_pairs: Optional[Sequence[str]], model_names: Sequence[str]) -> List[Tuple[str, str]]:
    """Resolve explicit --pairs or return standard 3D ablation pairs available in model_names."""
    known = set(model_names)
    if not raw_pairs:
        preferred = [
            ("diversity", "sft"),
            ("diversity", "validity_final"),
            ("diversity", "deepseek"),
            ("sft", "deepseek"),
            ("diversity", "base"),
            ("sft", "base"),
        ]
        selected = [(a, b) for a, b in preferred if a in known and b in known]
        if selected:
            return selected
        if len(model_names) >= 2:
            return [(model_names[0], other) for other in model_names[1:]]
        return []

    pairs: List[Tuple[str, str]] = []
    for spec in raw_pairs:
        if ":" not in spec:
            raise ValueError(f"Invalid --pairs '{spec}'; expected MODEL_A:MODEL_B.")
        a, b = (p.strip() for p in spec.split(":", 1))
        if a not in known or b not in known:
            raise ValueError(f"Unknown model in --pairs '{spec}'; known: {sorted(known)}")
        pairs.append((a, b))
    return pairs


def render_markdown_report(
    source_desc: str,
    common_task_ids: Sequence[str],
    summaries: Sequence[Dict[str, Any]],
    pairwise_results: Sequence[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
) -> str:
    """Render Markdown report matching `id_v2_stats.md` structure."""
    n_groups = len({metadata[t]["group_id"] for t in common_task_ids})
    mean_ceiling = theoretical_recovery_ceiling_mean(common_task_ids, metadata)

    lines: List[str] = [
        "# HypoSpace 3D Voxel Reconstruction 结构分组与标准化统计报告",
        "",
        f"- **任务元数据来源**: `{source_desc}`",
        f"- **对齐评测题目数**: `{len(common_task_ids)}`",
        f"- **D4 同构等价类数**: `{n_groups}`",
        f"- **i.i.d. 均匀采样期望上界 Macro Ceiling**: `{fmt_pct(mean_ceiling)}`",
        "",
        "## 1. 模型总览（Task-level 与 Structure-Group-level）",
        "",
        "| 模型 | 文件数 | Task Macro Recovery | Task Norm Recovery | Group Recovery (mean ± std) | Group Norm Recovery (mean ± std) | Task Valid | Task Uniqueness | Task Parse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        tm = s["task_macro"]
        gm = s["group_macro"]
        g_rec = gm["recovery_rate"]
        g_norm = gm["recovery_norm"]
        lines.append(
            f"| {s['model_name']} | {s['n_files']} | "
            f"{fmt_pct(tm['recovery_rate'])} | {fmt_pct(tm['recovery_norm'])} | "
            f"{fmt_pct(g_rec['mean'])} ± {fmt_pct(g_rec['std'])} | "
            f"{fmt_pct(g_norm['mean'])} ± {fmt_pct(g_norm['std'])} | "
            f"{fmt_pct(tm['valid_rate'])} | {fmt_pct(tm['uniqueness_rate'])} | "
            f"{fmt_pct(tm['parse_success_rate'])} |"
        )

    lines.extend([
        "",
        "## 2. 按合法解数量分桶统计（含有限采样参考上界）",
        "",
        "| 分桶 | 题数 | D4 组数 | 参考上界 Ceiling | "
        + " | ".join(f"{s['model_name']} Rec / Norm" for s in summaries)
        + " |",
        "|---|---:|---:|---:|" + "---:|" * len(summaries),
    ])
    ref_buckets = summaries[0]["by_bucket"]
    for bucket in BUCKET_ORDER:
        if bucket not in ref_buckets:
            continue
        b_info = ref_buckets[bucket]
        cells = [
            f"{fmt_pct(s['by_bucket'].get(bucket, {}).get('recovery_rate'))} / "
            f"{fmt_pct(s['by_bucket'].get(bucket, {}).get('recovery_norm'))}"
            for s in summaries
        ]
        lines.append(
            f"| `{bucket}` | {b_info['n_tasks']} | {b_info['n_groups']} | "
            f"{fmt_pct(b_info['ceiling'])} | " + " | ".join(cells) + " |"
        )

    if pairwise_results:
        lines.extend([
            "",
            "## 3. 成对模型对比（D4 等价类聚类显著性检验）",
            "",
            "> 注：Task W/T/L 仅作描述性参考；显著性推断以 D4 同构等价类（`n = n_groups`）为单位，结合精确双侧二项符号检验与 10,000 次聚类自助法 95% CI 判读。",
            "",
        ])
        for comp in pairwise_results:
            n_g = comp["n_common_groups"]
            lines.extend([
                f"### `{comp['pair']}`（对齐 {comp['n_common_tasks']} 题 / {n_g} 个 D4 组，Bootstrap {comp['bootstrap_iters']} 次，seed={comp['bootstrap_seed']}）",
                "",
                f"| 指标 | Task Delta | Task W/T/L (描述性) | Group Delta (mean ± std) | Group W/T/L | Exact Sign Test p (n={n_g}) | Cluster Bootstrap 95% CI |",
                "|---|---:|:---:|---:|:---:|---:|:---:|",
            ])
            for metric in ("recovery_rate", "recovery_norm", "valid_rate", "uniqueness_rate"):
                if metric not in comp["metrics"]:
                    continue
                m = comp["metrics"][metric]
                t_info = m["task_level_descriptive"]
                g_info = m["group_level"]
                ci_low, ci_high = g_info["bootstrap_ci_95"]
                lines.append(
                    f"| `{metric}` | {fmt_pct(t_info['mean_delta'], signed=True)} | "
                    f"{t_info['wins']}/{t_info['ties']}/{t_info['losses']} | "
                    f"{fmt_pct(g_info['mean_delta'], signed=True)} ± {fmt_pct(g_info['std_delta'])} | "
                    f"{g_info['wins']}/{g_info['ties']}/{g_info['losses']} | "
                    f"{g_info['sign_test_p']:.4f} | "
                    f"[{fmt_pct(ci_low, signed=True)}, {fmt_pct(ci_high, signed=True)}] |"
                )
            lines.append("")

    lines.extend([
        "## 4. 各 D4 等价类明细（Group-level Breakdown）",
        "",
        "| D4 等价类 ID | 题数 | 分桶 | 参考上界 | "
        + " | ".join(f"{s['model_name']} Rec (Norm)" for s in summaries)
        + " |",
        "|---|---:|---|---:|" + "---:|" * len(summaries),
    ])
    for group_id in sorted(summaries[0]["per_group"].keys()):
        ref_g = summaries[0]["per_group"][group_id]
        bucket_str = ",".join(ref_g["buckets"])
        cells = [
            f"{fmt_pct(s['per_group'][group_id]['recovery_rate'])} ({fmt_pct(s['per_group'][group_id]['recovery_norm'])})"
            for s in summaries
        ]
        lines.append(
            f"| `{group_id}` | {ref_g['n_tasks']} | `{bucket_str}` | "
            f"{fmt_pct(ref_g['ceiling'])} | " + " | ".join(cells) + " |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    model_specs = parse_model_specs(args.model)
    metadata, source_desc = build_task_metadata(
        splits_path=args.splits_path,
        split_name=args.split_name,
        dataset_path=args.dataset_path,
    )

    loaded_models: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    file_counts: Dict[str, int] = {}
    ordered_names: List[str] = []

    for name, paths in model_specs:
        task_map, n_files = load_model_task_metrics(paths, metadata, is_deepseek=False)
        loaded_models[name] = task_map
        file_counts[name] = n_files
        ordered_names.append(name)

    if args.deepseek_baseline is not None:
        if not args.deepseek_baseline.exists():
            raise FileNotFoundError(f"DeepSeek baseline not found: {args.deepseek_baseline}")
        ds_name = "deepseek" if "deepseek" not in loaded_models else "deepseek_baseline"
        ds_map, n_files = load_model_task_metrics([args.deepseek_baseline], metadata, is_deepseek=True)
        loaded_models[ds_name] = ds_map
        file_counts[ds_name] = n_files
        ordered_names.append(ds_name)

    common_sets = [set(loaded_models[name].keys()) for name in ordered_names]
    common_task_ids = sorted(set.intersection(*common_sets)) if common_sets else []
    if not common_task_ids:
        raise RuntimeError("No overlapping observation_set_id found across loaded 3D models.")

    summaries = [
        summarize_model(
            model_name=name,
            task_ids=common_task_ids,
            task_metrics=loaded_models[name],
            metadata=metadata,
            n_files=file_counts[name],
        )
        for name in ordered_names
    ]

    pairs = resolve_pairs(args.pairs, ordered_names)
    pairwise_results = [
        compute_pairwise_comparison(
            name_a=a,
            name_b=b,
            task_ids=common_task_ids,
            metrics_a=loaded_models[a],
            metrics_b=loaded_models[b],
            metadata=metadata,
            bootstrap_iters=args.bootstrap_iters,
            bootstrap_seed=args.bootstrap_seed,
        )
        for a, b in pairs
    ]

    md_report = render_markdown_report(
        source_desc=source_desc,
        common_task_ids=common_task_ids,
        summaries=summaries,
        pairwise_results=pairwise_results,
        metadata=metadata,
    )
    print(md_report, end="")

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(md_report, encoding="utf-8")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        json_payload = {
            "source": source_desc,
            "n_common_tasks": len(common_task_ids),
            "n_common_groups": len({metadata[t]["group_id"] for t in common_task_ids}),
            "theoretical_ceiling_macro": theoretical_recovery_ceiling_mean(common_task_ids, metadata),
            "models": summaries,
            "pairwise": pairwise_results,
        }
        args.output_json.write_text(
            json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
