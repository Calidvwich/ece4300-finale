"""Boolean task/group statistics, adapted from the Causal group_stats.py report."""

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
GRPO_LORA_DIR = PROJECT_ROOT / "HypoSpace-release" / "boolean" / "grpo_lora"
if str(GRPO_LORA_DIR) not in sys.path:
    sys.path.insert(0, str(GRPO_LORA_DIR))

from data_utils import (  # noqa: E402
    canonical_isomorphism_key,
    get_gt_bucket,
    load_boolean_dataset,
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
    """Parse command-line arguments."""
    default_splits = GRPO_LORA_DIR / "splits" / "boolean_splits.json"
    parser = argparse.ArgumentParser(
        description=(
            "Analyze HypoSpace Boolean evaluation outputs with isomorphism-group "
            "aggregation, theoretical ceiling normalization, and cluster bootstrap CIs."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=SPEC",
        help=(
            "Model specification NAME=PATH[,PATH2...] supporting glob wildcards "
            "(repeatable, e.g., --model grpo=results/eval_grpo_seed*.json)."
        ),
    )
    parser.add_argument(
        "--splits_path",
        type=Path,
        default=default_splits,
        help=f"Path to boolean_splits.json (default: {default_splits}).",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="test",
        help="Primary split key in splits_path ('train', 'val', 'test', or 'all'; default: test).",
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help=(
            "Optional raw dataset JSON (e.g., boolean_full_d2.json for OOD). "
            "When provided, task metadata and isomorphism keys are loaded from here instead of splits_path."
        ),
    )
    parser.add_argument(
        "--deepseek_baseline",
        type=Path,
        default=None,
        help="Optional DeepSeek baseline JSON matched by observation_set_id (novelty_rate -> uniqueness_rate).",
    )
    parser.add_argument(
        "--pairs",
        action="append",
        default=None,
        metavar="A:B",
        help=(
            "Pairwise comparison A:B (computes A - B). Repeatable. "
            "Defaults to the supported Boolean comparisons."
        ),
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=10000,
        help="Number of cluster bootstrap iterations (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=33550336,
        help="Random seed for cluster bootstrap (default: 33550336).",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=None,
        help="Optional output path for the Markdown report.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional output path for structured JSON statistics.",
    )
    args = parser.parse_args()
    if args.bootstrap_iters < 1:
        parser.error("--bootstrap_iters must be positive")
    return args


def theoretical_recovery_ceiling(n_gt: int) -> float:
    """Compute finite-sample theoretical recovery ceiling 1 - (1 - 1/K)^K.

    Under uniform independent sampling of K = n_gt draws with replacement from K
    equally likely valid graphs, the expected fraction of distinct GTs recovered
    is 1 - (1 - 1/K)^K.
    """
    if n_gt <= 0:
        return 0.0
    return 1.0 - (1.0 - 1.0 / float(n_gt)) ** float(n_gt)


def exact_binomial_sign_test(n_pos: int, n_neg: int) -> float:
    """Compute exact two-sided binomial sign test p-value using math.comb only.

    Under H0: P(diff > 0) = P(diff < 0) = 0.5 conditional on non-zero differences
    (n_eff = n_pos + n_neg). Returns 1.0 if n_eff == 0.
    """
    n_eff = n_pos + n_neg
    if n_eff == 0:
        return 1.0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(n_eff, i) for i in range(k + 1))
    return min(1.0, 2.0 * float(tail) / float(2**n_eff))


def resolve_spec_paths(spec: str) -> List[Path]:
    """Expand comma-separated paths and glob patterns for a model specification."""
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
                raise FileNotFoundError(f"Evaluation file not found: {path} (from spec '{spec}')")
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
            raise ValueError(f"Invalid --model format '{item}'; expected NAME=PATH[,PATH2...].")
        name, spec = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty model name in '{item}'.")
        if name in seen_names:
            raise ValueError(f"Duplicate model name '{name}'.")
        seen_names.add(name)
        models.append((name, resolve_spec_paths(spec)))
    return models


def make_group_uid(key: Tuple[Any, ...]) -> str:
    """Build a short, deterministic structural identifier from canonical_isomorphism_key."""
    digest = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()[:8]
    return digest


def build_task_metadata(
    splits_path: Path,
    split_name: str,
    dataset_path: Optional[Path],
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """Load task definitions and compute canonical_isomorphism_key for every task.

    Returns a mapping task_uid -> metadata dict and a source description.
    """
    if dataset_path is not None:
        _, tasks = load_boolean_dataset(dataset_path)
        task_rows = [(task, "ood") for task in tasks]
        source_desc = f"dataset:{dataset_path}"
    else:
        splits = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]
        requested = splits if split_name == "all" else {split_name: splits[split_name]}
        task_rows = [(task, name) for name, tasks in requested.items() for task in tasks]
        source_desc = f"splits:{splits_path} (split={split_name})"

    keys = sorted({canonical_isomorphism_key(task) for task, _ in task_rows})
    labels = {key: f"G{i:02d}_{make_group_uid(key)}" for i, key in enumerate(keys, start=1)}
    metadata = {}
    for task, split in task_rows:
        key = canonical_isomorphism_key(task)
        n_gt = task["n_compatible_expressions"]
        task_uid = task["task_uid"]
        metadata[task_uid] = {
            "task_uid": task_uid,
            "observation_set_id": str(task["observation_set_id"]),
            "n_observations": task["n_observations"],
            "n_ground_truths": n_gt,
            "bucket": get_gt_bucket(n_gt),
            "ceiling": theoretical_recovery_ceiling(n_gt),
            "group_id": labels[key],
            "group_hash": make_group_uid(key),
            "split": split,
        }
    return metadata, source_desc


def extract_task_record(
    entry: Dict[str, Any],
    task_meta: Dict[str, Any],
    is_deepseek: bool = False,
) -> Dict[str, Optional[float]]:
    """Extract normalized scalar metrics for one task observation record."""
    n_gt = int(entry.get("n_ground_truths", task_meta["n_ground_truths"]))
    ceiling = theoretical_recovery_ceiling(n_gt)
    recovery = float(entry["recovery_rate"])
    recovery_norm = (recovery / ceiling) if ceiling > 0.0 else 0.0

    if is_deepseek:
        uniqueness_raw = entry.get("novelty_rate", entry.get("uniqueness_rate"))
    else:
        uniqueness_raw = entry.get("uniqueness_rate", entry.get("novelty_rate"))
    uniqueness = float(uniqueness_raw) if uniqueness_raw is not None else None

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
    """Load one or more seed JSON files for a model and average per-task metrics across seeds."""
    by_observation = defaultdict(list)
    for task_uid, meta in metadata.items():
        by_observation[meta["observation_set_id"]].append(task_uid)
    per_task_seed_records: Dict[str, List[Dict[str, Optional[float]]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("per_sample_results", [])
        if not rows:
            raise ValueError(f"No per_sample_results found in {path}")
        if is_deepseek and payload.get("sampling_protocol") != "fixed_prompt":
            raise ValueError(f"Expected a Boolean --independent-samples DeepSeek result in {path}")
        seen = set()
        for row in rows:
            if is_deepseek:
                ids = by_observation.get(str(row["observation_set_id"]), [])
                if len(ids) > 1:
                    raise ValueError(
                        f"Ambiguous DeepSeek observation ID {row['observation_set_id']} in {path}; "
                        "select test or a single dataset."
                    )
                task_uid = ids[0] if ids else None
            else:
                task_uid = row["task_uid"]
            if task_uid not in metadata:
                continue
            expected_gt = metadata[task_uid]["n_ground_truths"]
            if row.get("n_ground_truths") != expected_gt or row.get("n_queries") != expected_gt:
                raise ValueError(f"Expected N = |H_O| = {expected_gt} for {task_uid} in {path}")
            if task_uid in seen:
                raise ValueError(f"Duplicate result for {task_uid} in {path}")
            seen.add(task_uid)
            per_task_seed_records[task_uid].append(
                extract_task_record(row, metadata[task_uid], is_deepseek=is_deepseek)
            )
        if not seen:
            raise ValueError(f"No matching task IDs in {path}")

    averaged: Dict[str, Dict[str, Optional[float]]] = {}
    for obs_id, records in per_task_seed_records.items():
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
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_val, std_val


def summarize_model(
    model_name: str,
    task_ids: Sequence[str],
    task_metrics: Dict[str, Dict[str, Optional[float]]],
    metadata: Dict[str, Dict[str, Any]],
    n_files: int,
) -> Dict[str, Any]:
    """Compute task-level, bucket-level, and structure-group-level summaries for one model."""
    groups: Dict[str, List[str]] = defaultdict(list)
    buckets: Dict[str, List[str]] = defaultdict(list)
    for obs_id in task_ids:
        meta = metadata[obs_id]
        groups[meta["group_id"]].append(obs_id)
        buckets[meta["bucket"]].append(obs_id)

    task_means: Dict[str, Optional[float]] = {}
    for metric in (*TASK_METRICS, "ceiling"):
        mean_val, _ = mean_std(task_metrics[obs_id][metric] for obs_id in task_ids)
        task_means[metric] = mean_val

    group_rows: Dict[str, Dict[str, Any]] = {}
    for group_id in sorted(groups.keys()):
        g_tasks = groups[group_id]
        g_summary: Dict[str, Any] = {
            "group_id": group_id,
            "n_tasks": len(g_tasks),
            "task_uids": g_tasks,
            "buckets": sorted({metadata[t]["bucket"] for t in g_tasks}, key=BUCKET_ORDER.index),
        }
        for metric in (*TASK_METRICS, "ceiling"):
            g_mean, _ = mean_std(task_metrics[t][metric] for t in g_tasks)
            g_summary[metric] = g_mean
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
            b_mean, _ = mean_std(task_metrics[t][metric] for t in b_tasks)
            b_entry[metric] = b_mean
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
    """Compare model A vs model B (delta = A - B) at task and isomorphism-group levels."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for obs_id in task_ids:
        groups[metadata[obs_id]["group_id"]].append(obs_id)
    sorted_group_ids = sorted(groups.keys())
    n_groups = len(sorted_group_ids)

    rng = np.random.default_rng(bootstrap_seed)
    boot_indices = (
        rng.integers(0, n_groups, size=(bootstrap_iters, n_groups)) if n_groups > 0 else None
    )

    tol = 1e-12
    comparisons: Dict[str, Any] = {}
    for metric in ("recovery_rate", "recovery_norm", "valid_rate", "uniqueness_rate"):
        task_deltas = {}
        for obs_id in task_ids:
            va = metrics_a[obs_id].get(metric)
            vb = metrics_b[obs_id].get(metric)
            if va is not None and vb is not None:
                task_deltas[obs_id] = float(va) - float(vb)

        if not task_deltas:
            continue

        t_arr = np.asarray(list(task_deltas.values()), dtype=np.float64)
        t_wins = int(np.sum(t_arr > tol))
        t_losses = int(np.sum(t_arr < -tol))
        t_ties = int(len(t_arr) - t_wins - t_losses)

        group_deltas = []
        for g_id in sorted_group_ids:
            g_tasks = groups[g_id]
            paired = [task_deltas[t] for t in g_tasks if t in task_deltas]
            if paired:
                group_deltas.append(float(np.mean(paired)))

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

        ci_low = float(np.percentile(boot_means, 2.5))
        ci_high = float(np.percentile(boot_means, 97.5))

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
                "bootstrap_ci_95": [ci_low, ci_high],
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
    """Format a proportion in [0, 1] as a percentage string."""
    if val is None:
        return "N/A"
    pct = val * 100.0
    if signed:
        return f"{pct:+.{decimals}f}%"
    return f"{pct:.{decimals}f}%"


def render_markdown_report(
    source_desc: str,
    common_task_ids: Sequence[str],
    summaries: Sequence[Dict[str, Any]],
    pairwise_results: Sequence[Dict[str, Any]],
    metadata: Dict[str, Dict[str, Any]],
) -> str:
    """Render all computed statistics as readable GitHub-flavored Markdown."""
    n_groups = len({metadata[t]["group_id"] for t in common_task_ids})
    mean_ceiling = theoretical_recovery_ceiling_mean(common_task_ids, metadata)

    lines: List[str] = [
        "# HypoSpace Boolean 结构分组与标准化统计报告",
        "",
        f"- **任务元数据来源**: `{source_desc}`",
        f"- **对齐评测题目数**: `{len(common_task_ids)}`",
        f"- **同构结构组数**: `{n_groups}`",
        f"- **有限采样理论上限 Macro Ceiling**: `{fmt_pct(mean_ceiling)}`",
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
        g_rec_str = f"{fmt_pct(g_rec['mean'])} ± {fmt_pct(g_rec['std'])}"
        g_norm_str = f"{fmt_pct(g_norm['mean'])} ± {fmt_pct(g_norm['std'])}"
        lines.append(
            f"| {s['model_name']} | {s['n_files']} | "
            f"{fmt_pct(tm['recovery_rate'])} | {fmt_pct(tm['recovery_norm'])} | "
            f"{g_rec_str} | {g_norm_str} | "
            f"{fmt_pct(tm['valid_rate'])} | {fmt_pct(tm['uniqueness_rate'])} | "
            f"{fmt_pct(tm['parse_success_rate'])} |"
        )

    lines.extend([
        "",
        "## 2. 按合法解数量分桶统计（含理论恢复上限）",
        "",
        "| 分桶 | 题数 | 结构组数 | 理论上限 Ceiling | "
        + " | ".join(f"{s['model_name']} Rec / Norm" for s in summaries)
        + " |",
        "|---|---:|---:|---:|" + "---:|" * len(summaries),
    ])
    ref_buckets = summaries[0]["by_bucket"]
    for bucket in BUCKET_ORDER:
        if bucket not in ref_buckets:
            continue
        b_info = ref_buckets[bucket]
        model_cells = []
        for s in summaries:
            sb = s["by_bucket"].get(bucket, {})
            model_cells.append(f"{fmt_pct(sb.get('recovery_rate'))} / {fmt_pct(sb.get('recovery_norm'))}")
        lines.append(
            f"| `{bucket}` | {b_info['n_tasks']} | {b_info['n_groups']} | "
            f"{fmt_pct(b_info['ceiling'])} | " + " | ".join(model_cells) + " |"
        )

    if pairwise_results:
        lines.extend([
            "",
            "## 3. 成对模型对比（结构组聚类显著性检验）",
            "",
            "> 注：Task W/T/L 仅作描述性参考（同构组内题目非独立）；显著性推断以结构组（`n = n_groups`）为单位，当组数较少时二项符号检验统计功效较低，需结合 10,000 次聚类自助法 95% CI 判读。",
            "",
        ])
        for comp in pairwise_results:
            n_g = comp["n_common_groups"]
            lines.extend([
                f"### `{comp['pair']}`（对齐 {comp['n_common_tasks']} 题 / {n_g} 个结构组，Bootstrap {comp['bootstrap_iters']} 次，seed={comp['bootstrap_seed']}）",
                "",
                "| 指标 | Task Delta | Task W/T/L (描述性) | Group Delta (mean ± std) | Group W/T/L | Exact Sign Test p (n={n_g}, low power) | Cluster Bootstrap 95% CI |".replace(
                    "{n_g}", str(n_g)
                ),
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
        "## 4. 各同构结构组明细（Group-level Breakdown）",
        "",
        "| 结构组 ID | 题数 | 分桶 | 理论上限 | "
        + " | ".join(f"{s['model_name']} Rec (Norm)" for s in summaries)
        + " |",
        "|---|---:|---|---:|" + "---:|" * len(summaries),
    ])
    for group_id in sorted(summaries[0]["per_group"].keys()):
        ref_g = summaries[0]["per_group"][group_id]
        bucket_str = ",".join(ref_g["buckets"])
        cells = []
        for s in summaries:
            sg = s["per_group"][group_id]
            cells.append(f"{fmt_pct(sg['recovery_rate'])} ({fmt_pct(sg['recovery_norm'])})")
        lines.append(
            f"| `{group_id}` | {ref_g['n_tasks']} | `{bucket_str}` | "
            f"{fmt_pct(ref_g['ceiling'])} | " + " | ".join(cells) + " |"
        )

    return "\n".join(lines) + "\n"


def theoretical_recovery_ceiling_mean(
    task_ids: Sequence[str],
    metadata: Dict[str, Dict[str, Any]],
) -> float:
    """Compute macro average theoretical ceiling across tasks."""
    if not task_ids:
        return 0.0
    return float(np.mean([metadata[t]["ceiling"] for t in task_ids]))


def resolve_pairs(
    raw_pairs: Optional[Sequence[str]],
    model_names: Sequence[str],
) -> List[Tuple[str, str]]:
    """Resolve pairwise comparisons A:B (defaults to first model vs. all others)."""
    if not raw_pairs:
        required = (
            ("diversity", "sft"),
            ("diversity", "validity_final"),
            ("diversity", "deepseek"),
            ("sft", "deepseek"),
            ("diversity", "base"),
            ("sft", "base"),
        )
        return [(a, b) for a, b in required if a in model_names and b in model_names]

    pairs: List[Tuple[str, str]] = []
    known = set(model_names)
    for spec in raw_pairs:
        if ":" not in spec:
            raise ValueError(f"Invalid --pairs format '{spec}'; expected MODEL_A:MODEL_B.")
        a, b = (part.strip() for part in spec.split(":", 1))
        if a not in known or b not in known:
            raise ValueError(f"Unknown model in --pairs '{spec}'; known models: {sorted(known)}")
        pairs.append((a, b))
    return pairs


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
        ds_map, n_files = load_model_task_metrics(
            [args.deepseek_baseline], metadata, is_deepseek=True
        )
        loaded_models[ds_name] = ds_map
        file_counts[ds_name] = n_files
        ordered_names.append(ds_name)

    common_sets = [set(loaded_models[name].keys()) for name in ordered_names]
    common_task_ids = sorted(set.intersection(*common_sets)) if common_sets else []
    if not common_task_ids:
        raise RuntimeError(
            "No overlapping task_uid found across loaded models and task metadata."
        )

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
            "bootstrap": {"iterations": args.bootstrap_iters, "seed": args.bootstrap_seed},
            "models": summaries,
            "pairwise": pairwise_results,
        }
        args.output_json.write_text(
            json.dumps(json_payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
