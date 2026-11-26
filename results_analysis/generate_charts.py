"""
从 RTX 5090 v2.1 零同构泄露全量实验结果与结构分组统计报告生成 4 张高清学术图表（300 DPI）。

输入数据：
- results_analysis/id_v2_stats.json 与 ood_n5_v2_stats.json（与 group_stats.py 保持单源一致）
- qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_*.json
- qwen7b_v2_results_and_logs/causal/grpo_lora/outputs/qwen7b_v2_sft/sft_metrics.json
- qwen7b_v2_results_and_logs/causal/grpo_lora/outputs/qwen7b_v2_grpo_{validity,diversity}/grpo_metrics.jsonl

输出图表（results_analysis/figures/）：
1. fig1_id_ablation.png        - 4 节点未见结构 ID 测试集（76 题 / 5 结构组，3 种子均值）四方消融对比
2. fig2_difficulty_scaling.png - 按假设空间大小 |H_O| 分桶的 Recovery 曲线与理论恢复上限对照
3. fig3_ood_transfer.png       - 5 节点 OOD 跨规模零样本泛化对比（80 题 / 22 结构组）
4. fig4_training_curves.png    - Stage 1 SFT Loss、Stage 2 Diversity vs Validity-Only 消融曲线与 KL 稳定性
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
STATS_DIR = Path(__file__).resolve().parent
FIG_DIR = STATS_DIR / "figures"
V2_ROOT = ROOT / "qwen7b_v2_results_and_logs" / "causal" / "grpo_lora"
RES_DIR = V2_ROOT / "results"
OUT_DIR = V2_ROOT / "outputs"

COLORS = {
    "base": "#94A3B8",      # Slate gray
    "deepseek": "#F59E0B",  # Amber
    "sft": "#3B82F6",       # Royal blue
    "validity": "#EF4444",  # Coral red (ablation collapse)
    "grpo": "#10B981",      # Emerald green (Ours)
    "ceiling": "#1E293B",   # Dark slate
}


def load_json(path: Path) -> Dict[str, Any]:
    """Read a UTF-8 JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a UTF-8 JSONL file."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rolling_mean(values: Sequence[float], window: int = 20) -> np.ndarray:
    """Compute trailing rolling mean with variable initial window."""
    arr = np.asarray(values, dtype=float)
    if len(arr) <= 1 or window <= 1:
        return arr
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(arr[start : i + 1]))
    return out


def mean_pooled_recovery(pattern: str) -> float:
    """Average summary.pooled.recovery_rate across matching seed JSON files."""
    files = sorted(RES_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No result files match {RES_DIR / pattern}")
    vals = [load_json(p)["summary"]["pooled"]["recovery_rate"] for p in files]
    return float(np.mean(vals))


def index_models(stats_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index stats_payload['models'] by model_name."""
    return {m["model_name"]: m for m in stats_payload["models"]}


def index_pairs(stats_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index stats_payload['pairwise'] by pair string."""
    return {p["pair"]: p for p in stats_payload["pairwise"]}


def plot_fig1_id_ablation(id_stats: Dict[str, Any]) -> None:
    """Plot 4-node unseen-structure ID 4-way ablation bar chart (3-seed mean)."""
    models = index_models(id_stats)
    pairs = index_pairs(id_stats)

    seed_ref = load_json(RES_DIR / "qwen7b_v2_unseen_grpo_diversity_seed33550336.json")
    pooled_map = {
        "base": mean_pooled_recovery("qwen7b_v2_unseen_base_seed*.json"),
        "deepseek": seed_ref["summary"]["paired_baseline"]["pooled"]["recovery_rate"],
        "sft": mean_pooled_recovery("qwen7b_v2_unseen_sft_seed*.json"),
        "diversity": mean_pooled_recovery("qwen7b_v2_unseen_grpo_diversity_seed*.json"),
    }

    method_specs = [
        ("base", "Qwen2.5-7B Base (3-seed mean)", COLORS["base"]),
        ("deepseek", "DeepSeek-Flash (API Baseline)", COLORS["deepseek"]),
        ("sft", "SFT-LoRA / Validity-GRPO* (Stage 1)", COLORS["sft"]),
        ("diversity", "Diversity-GRPO-LoRA (Ours, 3-seed mean)", COLORS["grpo"]),
    ]
    metric_specs = [
        ("Macro Validity", lambda m, name: m["task_macro"]["valid_rate"] * 100.0),
        ("Macro Recovery\n(= Uniqueness)", lambda m, name: m["task_macro"]["recovery_rate"] * 100.0),
        ("Group Recovery\n(5 Struct Groups)", lambda m, name: m["group_macro"]["recovery_rate"]["mean"] * 100.0),
        ("Pooled Recovery", lambda m, name: pooled_map[name] * 100.0),
    ]

    x = np.arange(len(metric_specs))
    width = 0.195
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(method_specs))

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    for idx, (key, label, color) in enumerate(method_specs):
        m_data = models[key]
        vals = [fn(m_data, key) for _, fn in metric_specs]
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.90,
            label=label,
            color=color,
            edgecolor="#1E293B",
            linewidth=0.6,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.4,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.4,
                fontweight="bold" if key == "diversity" else "normal",
                color="#0F172A",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metric_specs], fontsize=10.5)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 126)
    ax.set_title(
        "Unseen-Structure 4-Node ID Test (76 Tasks / 5 Groups, Zero Isomorphism Leakage)"
    )
    ax.legend(loc="upper right", fontsize=8.8)

    ds_pair = pairs["diversity - deepseek"]["metrics"]["recovery_rate"]
    sft_rec = pairs["diversity - sft"]["metrics"]["recovery_rate"]
    sft_val = pairs["diversity - sft"]["metrics"]["valid_rate"]
    ds_ci = ds_pair["group_level"]["bootstrap_ci_95"]
    sft_vci = sft_val["group_level"]["bootstrap_ci_95"]
    callout = (
        f"vs. DeepSeek-Flash: +{ds_pair['task_level_descriptive']['mean_delta']*100:.1f}% Task "
        f"/ +{ds_pair['group_level']['mean_delta']*100:.1f}% Group Rec "
        f"(95% CI [{ds_ci[0]*100:+.1f}%, {ds_ci[1]*100:+.1f}%])\n"
        f"vs. SFT-LoRA: +{sft_val['task_level_descriptive']['mean_delta']*100:.1f}% Valid "
        f"(Group CI [{sft_vci[0]*100:+.1f}%, {sft_vci[1]*100:+.1f}%]), "
        f"+{sft_rec['task_level_descriptive']['mean_delta']*100:.1f}% Task Rec"
    )
    ax.text(
        0.018,
        0.965,
        callout,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        bbox={"boxstyle": "round,pad=0.38", "facecolor": "#ECFDF5", "edgecolor": "#10B981"},
    )
    fig.text(
        0.012,
        0.01,
        "*Note: Validity-Only GRPO collapsed on val recovery and reverted to best_step=0 (SFT init), yielding identical test scores.",
        fontsize=8.4,
        color="#475569",
    )

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(FIG_DIR / "fig1_id_ablation.png", dpi=300)
    plt.close(fig)


def plot_fig2_difficulty_scaling(id_stats: Dict[str, Any]) -> None:
    """Plot |H_O| difficulty bucket scaling alongside finite-sample theoretical ceilings."""
    models = index_models(id_stats)
    buckets = ["1", "2-4", "5-8", "9+"]
    ref_buckets = models["diversity"]["by_bucket"]
    bucket_labels = [
        f"|H_O| = {b}\n(n={ref_buckets[b]['n_tasks']}, g={ref_buckets[b]['n_groups']})"
        for b in buckets
    ]

    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=(9.4, 5.4))

    # Plot theoretical ceiling first so it sits cleanly behind highlighted series
    ceilings = [ref_buckets[b]["ceiling"] * 100.0 for b in buckets]
    ax.plot(
        x,
        ceilings,
        label="Finite-Sample Ceiling 1-(1-1/K)^K",
        color=COLORS["ceiling"],
        linestyle=":",
        marker="*",
        markersize=9.5,
        linewidth=2.0,
        zorder=2,
    )
    for xi, yi in zip(x, ceilings):
        if xi > 0:
            ax.annotate(
                f"{yi:.1f}%",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.3,
                color=COLORS["ceiling"],
            )

    series = [
        ("base", "Qwen2.5-7B Base", COLORS["base"], "o", "--"),
        ("deepseek", "DeepSeek-Flash (API)", COLORS["deepseek"], "s", "-."),
        ("sft", "SFT-LoRA (Stage 1)", COLORS["sft"], "^", "-"),
        ("diversity", "Diversity-GRPO-LoRA (Ours)", COLORS["grpo"], "D", "-"),
    ]

    offset_table = {
        "base": [(0, -14), (0, 7), (0, -14), (0, -14)],
        "deepseek": [(18, 4), (0, -14), (0, -14), (0, 8)],
        "sft": [(-24, 2), (0, -14), (0, 8), (-22, -14)],
        "diversity": [(22, -12), (0, 8), (0, -14), (20, 8)],
    }

    for key, label, color, marker, ls in series:
        ys = [models[key]["by_bucket"][b]["recovery_rate"] * 100.0 for b in buckets]
        is_ours = key == "diversity"
        ax.plot(
            x,
            ys,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=2.8 if is_ours else 2.0,
            markersize=8.0 if is_ours else 6.5,
            zorder=4 if is_ours else 3,
        )
        for xi, yi in zip(x, ys):
            dx, dy = offset_table[key][xi]
            ax.annotate(
                f"{yi:.1f}%",
                (xi, yi),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="center",
                fontsize=8.4,
                fontweight="bold" if is_ours else "normal",
                color=color if key != "base" else "#475569",
            )

    grpo_hard = models["diversity"]["by_bucket"]["9+"]["recovery_rate"] * 100.0
    grpo_hard_norm = models["diversity"]["by_bucket"]["9+"]["recovery_norm"] * 100.0
    ds_hard = models["deepseek"]["by_bucket"]["9+"]["recovery_rate"] * 100.0
    ratio = grpo_hard / max(1e-6, ds_hard)
    ax.annotate(
        f"{ratio:.1f}x DeepSeek on |H_O|>=9\n({grpo_hard_norm:.1f}% of ceiling)",
        xy=(3, grpo_hard),
        xytext=(1.95, 79),
        arrowprops={"arrowstyle": "->", "color": COLORS["grpo"], "lw": 1.6},
        fontsize=9.2,
        fontweight="bold",
        color="#065F46",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#ECFDF5", "edgecolor": COLORS["grpo"]},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Macro Recovery Rate (%)")
    ax.set_ylim(-4, 114)
    ax.set_title("Recovery Breakdown by |H_O| vs. Theoretical Finite-Sample Ceiling (4-Node Test)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.14), fontsize=8.6)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_difficulty_scaling.png", dpi=300)
    plt.close(fig)


def plot_fig3_ood_transfer(ood_stats: Dict[str, Any]) -> None:
    """Plot 5-node OOD zero-shot transfer comparison (80 tasks, 22 structure groups)."""
    models = index_models(ood_stats)
    pairs = index_pairs(ood_stats)

    pooled_map = {
        "base": load_json(RES_DIR / "qwen7b_v2_ood_n5_base_seed33550336.json")["summary"]["pooled"]["recovery_rate"],
        "sft": load_json(RES_DIR / "qwen7b_v2_ood_n5_sft_seed33550336.json")["summary"]["pooled"]["recovery_rate"],
        "diversity": load_json(RES_DIR / "qwen7b_v2_ood_n5_grpo_diversity_seed33550336.json")["summary"]["pooled"]["recovery_rate"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    method_specs = [
        ("base", "Qwen2.5-7B Base", COLORS["base"]),
        ("sft", "SFT-LoRA / Validity-GRPO*", COLORS["sft"]),
        ("diversity", "Diversity-GRPO (Ours)", COLORS["grpo"]),
    ]
    metric_specs = [
        ("Macro\nValidity", lambda m, name: m["task_macro"]["valid_rate"] * 100.0),
        ("Macro\nRecovery", lambda m, name: m["task_macro"]["recovery_rate"] * 100.0),
        ("Group\nRecovery", lambda m, name: m["group_macro"]["recovery_rate"]["mean"] * 100.0),
        ("Pooled\nRecovery", lambda m, name: pooled_map[name] * 100.0),
    ]
    x1 = np.arange(len(metric_specs))
    w = 0.255
    offs = [-w, 0.0, w]

    for idx, (key, label, color) in enumerate(method_specs):
        vals = [fn(models[key], key) for _, fn in metric_specs]
        bars = ax1.bar(
            x1 + offs[idx],
            vals,
            width=w * 0.90,
            label=label,
            color=color,
            edgecolor="#1E293B",
            linewidth=0.6,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.4,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold" if key == "diversity" else "normal",
            )

    ood_pair = pairs["diversity - sft"]["metrics"]["recovery_rate"]["group_level"]
    ood_ci = ood_pair["bootstrap_ci_95"]
    ax1.text(
        0.025,
        0.96,
        f"vs. SFT Group Rec: {ood_pair['mean_delta']*100:+.1f}%\n"
        f"(12W/8T/2L, Sign p={ood_pair['sign_test_p']:.4f},\n"
        f"95% CI [{ood_ci[0]*100:+.1f}%, {ood_ci[1]*100:+.1f}%])",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ECFDF5", "edgecolor": "#10B981"},
    )
    ax1.set_xticks(x1)
    ax1.set_xticklabels([m[0] for m in metric_specs])
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 96)
    ax1.set_title("OOD 5-Node Overall (80 Tasks, 22 Groups)")
    ax1.legend(loc="upper right", fontsize=8.4)

    buckets = ["1", "2-4", "5-8", "9+"]
    ref_b = models["diversity"]["by_bucket"]
    x2 = np.arange(len(buckets))
    for idx, (key, label, color) in enumerate(method_specs):
        vals = [models[key]["by_bucket"][b]["recovery_rate"] * 100.0 for b in buckets]
        bars = ax2.bar(
            x2 + offs[idx],
            vals,
            width=w * 0.90,
            label=label,
            color=color,
            edgecolor="#1E293B",
            linewidth=0.6,
            zorder=3,
        )
        for bar_idx, (bar, v) in enumerate(zip(bars, vals)):
            x_shift = -0.025 if (bar_idx == 0 and key == "sft") else (0.025 if (bar_idx == 0 and key == "diversity") else 0.0)
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0 + x_shift,
                v + 1.4,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold" if key == "diversity" else "normal",
            )

    for b_idx, b in enumerate(buckets):
        c_val = ref_b[b]["ceiling"] * 100.0
        lbl = "Theoretical Ceiling" if b_idx == 0 else None
        ax2.hlines(
            c_val,
            b_idx - 1.45 * w,
            b_idx + 1.45 * w,
            colors=COLORS["ceiling"],
            linestyles="--",
            linewidth=1.5,
            label=lbl,
            zorder=4,
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [f"|H_O|={b}\n(n={ref_b[b]['n_tasks']}, g={ref_b[b]['n_groups']})" for b in buckets]
    )
    ax2.set_ylabel("Macro Recovery (%)")
    ax2.set_ylim(0, 112)
    ax2.set_title("OOD 5-Node Recovery by |H_O| Bucket")
    ax2.legend(loc="upper right", fontsize=8.3)

    fig.suptitle(
        "Zero-Shot Transfer from 3-4 Nodes to Unseen 5-Node Causal Graphs (v2.1)",
        fontsize=13.2,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(FIG_DIR / "fig3_ood_transfer.png", dpi=300)
    plt.close(fig)


def plot_fig4_training_curves(
    sft_metrics: Dict[str, Any],
    div_rows: List[Dict[str, Any]],
    val_only_rows: List[Dict[str, Any]],
) -> None:
    """Plot Stage 1 SFT loss, Stage 2 Diversity vs Validity-Only val ablation, and KL stability."""
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7))

    # Panel A: Stage 1 SFT Loss
    sft_steps = [r["step"] for r in sft_metrics["history"]]
    sft_losses = [r["loss"] for r in sft_metrics["history"]]
    axes[0].plot(sft_steps, sft_losses, color="#93C5FD", alpha=0.55, linewidth=1.2, label="Step Loss")
    axes[0].plot(
        sft_steps,
        rolling_mean(sft_losses, window=10),
        color=COLORS["sft"],
        linewidth=2.3,
        label="MA-10 Loss",
    )
    cov_pct = sft_metrics["gt_coverage_summary"]["mean_coverage_ratio"] * 100.0
    axes[0].set_title(f"Stage 1: Debiased SFT Loss ({cov_pct:.0f}% GT Cov)")
    axes[0].set_xlabel("Optimizer Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(loc="upper right", fontsize=8.8)

    # Panel B: Stage 2 Val Group Recovery (Diversity vs Validity-Only Ablation)
    div_val_rows = [r for r in div_rows if "val_recovery_group_mean" in r]
    vonly_val_rows = [r for r in val_only_rows if "val_recovery_group_mean" in r]

    div_steps = [r["step"] for r in div_val_rows]
    div_rec = [r["val_recovery_group_mean"] * 100.0 for r in div_val_rows]
    vonly_steps = [r["step"] for r in vonly_val_rows]
    vonly_rec = [r["val_recovery_group_mean"] * 100.0 for r in vonly_val_rows]
    step0_rec = div_rec[0]

    axes[1].axhline(
        step0_rec,
        color=COLORS["sft"],
        linestyle=":",
        linewidth=1.5,
        label=f"Step 0 SFT Init ({step0_rec:.2f}%)",
    )
    axes[1].plot(
        div_steps,
        div_rec,
        color=COLORS["grpo"],
        marker="D",
        linewidth=2.4,
        markersize=6.0,
        label="Diversity-GRPO (Ours)",
    )
    axes[1].plot(
        vonly_steps,
        vonly_rec,
        color=COLORS["validity"],
        marker="s",
        linestyle="--",
        linewidth=2.1,
        markersize=5.5,
        label="Validity-Only GRPO (Ablation)",
    )
    best_idx = int(np.argmax(div_rec))
    axes[1].scatter(
        [div_steps[best_idx]],
        [div_rec[best_idx]],
        color="#047857",
        s=130,
        marker="*",
        zorder=5,
        label=f"Best Policy @ step {div_steps[best_idx]} ({div_rec[best_idx]:.2f}%)",
    )
    axes[1].set_title("Stage 2 Val Group Recovery (Mode Collapse Test)")
    axes[1].set_xlabel("GRPO Optimizer Step")
    axes[1].set_ylabel("Group-Mean Val Recovery (%)")
    axes[1].set_ylim(50.5, 59.8)
    axes[1].legend(loc="lower left", fontsize=8.2)

    # Panel C: Stage 2 Val Validity vs KL Stability
    ax_c = axes[2]
    ax_c_r = ax_c.twinx()
    ax_c_r.spines["top"].set_visible(False)

    div_val_valid = [r["val_valid_rate"] * 100.0 for r in div_val_rows]
    vonly_val_valid = [r["val_valid_rate"] * 100.0 for r in vonly_val_rows]
    div_train_rows = [r for r in div_rows if "kl" in r]
    train_steps = [r["step"] for r in div_train_rows]
    kls = [r["kl"] for r in div_train_rows]

    lc1 = ax_c.plot(
        vonly_steps,
        vonly_val_valid,
        color=COLORS["validity"],
        linestyle="--",
        marker="s",
        markersize=4.8,
        linewidth=1.9,
        label="Validity-Only Val Valid %",
    )
    lc2 = ax_c.plot(
        div_steps,
        div_val_valid,
        color=COLORS["grpo"],
        marker="D",
        markersize=5.0,
        linewidth=2.2,
        label="Diversity-GRPO Val Valid %",
    )
    lc3 = ax_c_r.plot(
        train_steps,
        rolling_mean(kls, window=10),
        color="#F97316",
        linewidth=1.8,
        alpha=0.9,
        label="Diversity KL (MA-10)",
    )
    ax_c.set_title("Stage 2 Val Validity & KL Stability")
    ax_c.set_xlabel("GRPO Optimizer Step")
    ax_c.set_ylabel("Val Valid Rate (%)", color="#0F172A")
    ax_c.set_ylim(74, 101)
    ax_c_r.set_ylabel("KL(policy || ref)", color="#EA580C")
    lines_c = lc1 + lc2 + lc3
    ax_c.legend(lines_c, [l.get_label() for l in lines_c], loc="lower right", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_training_curves.png", dpi=300)
    plt.close(fig)


def main() -> None:
    """Regenerate all 4 figures from verified v2.1 outputs."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    id_stats = load_json(STATS_DIR / "id_v2_stats.json")
    ood_stats = load_json(STATS_DIR / "ood_n5_v2_stats.json")
    sft_metrics = load_json(OUT_DIR / "qwen7b_v2_sft" / "sft_metrics.json")
    div_rows = load_jsonl(OUT_DIR / "qwen7b_v2_grpo_diversity" / "grpo_metrics.jsonl")
    val_only_rows = load_jsonl(OUT_DIR / "qwen7b_v2_grpo_validity" / "grpo_metrics.jsonl")

    plot_fig1_id_ablation(id_stats)
    plot_fig2_difficulty_scaling(id_stats)
    plot_fig3_ood_transfer(ood_stats)
    plot_fig4_training_curves(sft_metrics, div_rows, val_only_rows)

    for png in sorted(FIG_DIR.glob("*.png")):
        print(f"Generated: {png} ({png.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
