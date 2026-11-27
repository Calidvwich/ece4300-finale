"""
从 RTX 5090 3D 统计报告（stats_3d.py JSON）与训练日志生成 4 张高清学术图表（300 DPI）。

输出目录：`results_analysis/figures_3d/`
1. fig1_id_ablation.png        - Unseen-D4 ID Test 多方消融对比
2. fig2_difficulty_scaling.png - 按方块数/假设空间桶的分桶 Recovery 与参考上界对比
3. fig3_ood_transfer.png       - 4-Block OOD 跨空间复杂度零样本迁移对比
4. fig4_training_curves.png    - Stage 1 SFT Loss、Stage 2 Diversity vs Validity-Only 验证曲线与 KL 稳定性
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.94,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RES_DIR = ROOT / "HypoSpace-release" / "3d" / "grpo_lora" / "results"
DEFAULT_OUT_DIR = ROOT / "HypoSpace-release" / "3d" / "grpo_lora" / "outputs"
DEFAULT_FIG_DIR = Path(__file__).resolve().parent / "figures_3d"

COLORS = {
    "base": "#94A3B8",
    "deepseek": "#F59E0B",
    "sft": "#3B82F6",
    "validity_final": "#EF4444",
    "diversity": "#10B981",
    "ceiling": "#1E293B",
}

DISPLAY_NAMES = {
    "base": "Qwen2.5-7B Base",
    "deepseek": "DeepSeek-Flash (API)",
    "sft": "SFT-LoRA (Stage 1)",
    "validity_final": "Validity-GRPO (Final)",
    "diversity": "Diversity-GRPO (Ours)",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rolling_mean(values: Sequence[float], window: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1 or window <= 1:
        return arr
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(arr[start : i + 1]))
    return out


def plot_fig1_id_ablation(id_stats: Dict[str, Any], fig_dir: Path) -> None:
    """Plot 3D Unseen-D4 ID ablation across available models."""
    models = {m["model_name"]: m for m in id_stats["models"]}
    order = [k for k in ("base", "deepseek", "sft", "validity_final", "diversity") if k in models]

    metric_specs = [
        ("Macro Validity", lambda m: (m["task_macro"].get("valid_rate") or 0.0) * 100.0),
        ("Macro Recovery\n(= Uniqueness)", lambda m: (m["task_macro"].get("recovery_rate") or 0.0) * 100.0),
        ("Group Recovery\n(D4 Classes)", lambda m: (m["group_macro"]["recovery_rate"]["mean"] or 0.0) * 100.0),
        ("Norm Recovery\n(vs Ceiling)", lambda m: (m["task_macro"].get("recovery_norm") or 0.0) * 100.0),
    ]

    x = np.arange(len(metric_specs))
    n_m = len(order)
    width = 0.78 / max(1, n_m)
    offsets = np.linspace(-0.5 * (n_m - 1) * width, 0.5 * (n_m - 1) * width, n_m)

    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    for idx, key in enumerate(order):
        vals = [fn(models[key]) for _, fn in metric_specs]
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.90,
            label=DISPLAY_NAMES.get(key, key),
            color=COLORS.get(key, "#64748B"),
            edgecolor="#1E293B",
            linewidth=0.6,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.2,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.8 if n_m >= 5 else 8.4,
                fontweight="bold" if key == "diversity" else "normal",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metric_specs])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 112)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8.4)
    fig.suptitle(
        f"3D Voxel Unseen-D4 ID Test ({id_stats['n_common_tasks']} Tasks / {id_stats['n_common_groups']} D4 Classes)",
        fontsize=12.6,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(fig_dir / "fig1_id_ablation.png", dpi=300)
    plt.close(fig)


def plot_fig2_difficulty_scaling(id_stats: Dict[str, Any], fig_dir: Path) -> None:
    """Plot Recovery across |H_O| buckets present in 3D ID split."""
    models = {m["model_name"]: m for m in id_stats["models"]}
    order = [k for k in ("base", "deepseek", "sft", "validity_final", "diversity") if k in models]
    ref_b = models[order[-1]]["by_bucket"]
    buckets = [b for b in ("1", "2-4", "5-8", "9+") if b in ref_b]
    x = np.arange(len(buckets))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ceilings = [ref_b[b]["ceiling"] * 100.0 for b in buckets]
    ax.plot(
        x,
        ceilings,
        label="i.i.d. Reference Ceiling",
        color=COLORS["ceiling"],
        linestyle=":",
        marker="*",
        markersize=9.5,
        linewidth=2.0,
    )
    for key in order:
        ys = [models[key]["by_bucket"][b]["recovery_rate"] * 100.0 for b in buckets]
        is_ours = key == "diversity"
        ax.plot(
            x,
            ys,
            label=DISPLAY_NAMES.get(key, key),
            color=COLORS.get(key, "#64748B"),
            marker="D" if is_ours else "o",
            linewidth=2.7 if is_ours else 2.0,
            markersize=7.5 if is_ours else 6.0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"|H_O|={b}\n(n={ref_b[b]['n_tasks']}, g={ref_b[b]['n_groups']})" for b in buckets])
    ax.set_ylabel("Macro Recovery Rate (%)")
    ax.set_ylim(-3, 112)
    ax.set_title("3D ID Recovery by Hypothesis Space Size |H_O|")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_difficulty_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_ood_transfer(ood_stats: Dict[str, Any], fig_dir: Path) -> None:
    """Plot 4-Block (K=81) OOD transfer bar chart."""
    models = {m["model_name"]: m for m in ood_stats["models"]}
    order = [k for k in ("base", "deepseek", "sft", "validity_final", "diversity") if k in models]
    metric_specs = [
        ("Macro Validity", lambda m: (m["task_macro"].get("valid_rate") or 0.0) * 100.0),
        ("Macro Recovery", lambda m: (m["task_macro"].get("recovery_rate") or 0.0) * 100.0),
        ("Norm Recovery", lambda m: (m["task_macro"].get("recovery_norm") or 0.0) * 100.0),
    ]
    x = np.arange(len(metric_specs))
    n_m = len(order)
    width = 0.76 / max(1, n_m)
    offsets = np.linspace(-0.5 * (n_m - 1) * width, 0.5 * (n_m - 1) * width, n_m)

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    for idx, key in enumerate(order):
        vals = [fn(models[key]) for _, fn in metric_specs]
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.90,
            label=DISPLAY_NAMES.get(key, key),
            color=COLORS.get(key, "#64748B"),
            edgecolor="#1E293B",
            linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.2,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold" if key == "diversity" else "normal",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metric_specs])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 108)
    ax.set_title(
        f"3D Zero-Shot OOD Transfer to 4-Block Structures ({ood_stats['n_common_tasks']} D4 Classes, K=81)"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_ood_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_training_curves(
    sft_metrics: Optional[Dict[str, Any]],
    div_rows: List[Dict[str, Any]],
    val_only_rows: Optional[List[Dict[str, Any]]],
    fig_dir: Path,
) -> None:
    """Plot Stage 1 SFT Loss, Stage 2 Val Group Recovery ablation, and Val Validity + KL."""
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    if sft_metrics and sft_metrics.get("history"):
        steps = [r["step"] for r in sft_metrics["history"]]
        losses = [r["loss"] for r in sft_metrics["history"]]
        axes[0].plot(steps, losses, color="#93C5FD", alpha=0.55, linewidth=1.2, label="Step Loss")
        axes[0].plot(steps, rolling_mean(losses, 10), color=COLORS["sft"], linewidth=2.2, label="MA-10")
    axes[0].set_title("3D Stage 1: SFT-LoRA Loss")
    axes[0].set_xlabel("Optimizer Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(loc="upper right", fontsize=8.5)

    div_val = [r for r in div_rows if "val_recovery_group_mean" in r]
    div_steps = [r["step"] for r in div_val]
    div_rec = [r["val_recovery_group_mean"] * 100.0 for r in div_val]
    if div_rec:
        axes[1].axhline(div_rec[0], color=COLORS["sft"], linestyle=":", linewidth=1.5, label=f"Step 0 ({div_rec[0]:.2f}%)")
        axes[1].plot(div_steps, div_rec, color=COLORS["diversity"], marker="D", linewidth=2.3, label="Diversity-GRPO")
    if val_only_rows:
        vo_val = [r for r in val_only_rows if "val_recovery_group_mean" in r]
        axes[1].plot(
            [r["step"] for r in vo_val],
            [r["val_recovery_group_mean"] * 100.0 for r in vo_val],
            color=COLORS["validity_final"],
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label="Validity-Only GRPO",
        )
    axes[1].set_title("3D Stage 2: Val Group Recovery")
    axes[1].set_xlabel("GRPO Step")
    axes[1].set_ylabel("Group-Mean Val Recovery (%)")
    axes[1].legend(loc="best", fontsize=8.2)

    ax_c = axes[2]
    ax_c_r = ax_c.twinx()
    ax_c_r.spines["top"].set_visible(False)
    l_all = []
    if div_val:
        l_all += ax_c.plot(div_steps, [r["val_valid_rate"] * 100.0 for r in div_val], color=COLORS["diversity"], marker="D", label="Diversity Valid %")
    if val_only_rows:
        vo_val = [r for r in val_only_rows if "val_valid_rate" in r]
        l_all += ax_c.plot([r["step"] for r in vo_val], [r["val_valid_rate"] * 100.0 for r in vo_val], color=COLORS["validity_final"], linestyle="--", marker="s", label="Validity-Only Valid %")
    tr_rows = [r for r in div_rows if "kl" in r]
    if tr_rows:
        l_all += ax_c_r.plot([r["step"] for r in tr_rows], rolling_mean([r["kl"] for r in tr_rows], 10), color="#F97316", label="Diversity KL (MA-10)")
    ax_c.set_title("3D Stage 2: Val Validity & KL")
    ax_c.set_xlabel("GRPO Step")
    ax_c.set_ylabel("Val Valid Rate (%)")
    ax_c_r.set_ylabel("KL(policy || ref)", color="#EA580C")
    if l_all:
        ax_c.legend(l_all, [l.get_label() for l in l_all], loc="lower right", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_training_curves.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D GRPO-LoRA figures.")
    parser.add_argument("--id_stats", type=Path, default=Path(__file__).resolve().parent / "id_3d_stats.json")
    parser.add_argument("--ood_stats", type=Path, default=Path(__file__).resolve().parent / "ood_3d_b4_stats.json")
    parser.add_argument("--run_tag", type=str, default="qwen7b_3d")
    parser.add_argument("--outputs_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig_dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    if args.id_stats.exists():
        id_stats = load_json(args.id_stats)
        plot_fig1_id_ablation(id_stats, args.fig_dir)
        plot_fig2_difficulty_scaling(id_stats, args.fig_dir)
    if args.ood_stats.exists():
        ood_stats = load_json(args.ood_stats)
        plot_fig3_ood_transfer(ood_stats, args.fig_dir)

    sft_path = args.outputs_dir / f"{args.run_tag}_sft" / "sft_metrics.json"
    div_path = args.outputs_dir / f"{args.run_tag}_grpo_diversity" / "grpo_metrics.jsonl"
    val_path = args.outputs_dir / f"{args.run_tag}_grpo_validity" / "grpo_metrics.jsonl"
    if div_path.exists():
        plot_fig4_training_curves(
            sft_metrics=load_json(sft_path) if sft_path.exists() else None,
            div_rows=load_jsonl(div_path),
            val_only_rows=load_jsonl(val_path) if val_path.exists() else None,
            fig_dir=args.fig_dir,
        )
    for png in sorted(args.fig_dir.glob("*.png")):
        print(f"Generated: {png} ({png.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
