"""
从 RTX 5090 实验输出 JSON 与训练日志生成 4 张高清学术对比图表（300 DPI）。

输入数据：
- qwen7b_results_and_logs/causal/grpo_lora/results/qwen7b_*.json
- qwen7b_results_and_logs/causal/grpo_lora/outputs/qwen7b_sft/sft_metrics.json
- qwen7b_results_and_logs/causal/grpo_lora/outputs/qwen7b_grpo/grpo_metrics.jsonl

输出图表（results_analysis/figures/）：
1. fig1_id_ablation.png        - ID 79 题四方消融对比柱状图
2. fig2_difficulty_scaling.png - 不同 |H_O| 假设空间难度下的 Recovery 衰减曲线
3. fig3_ood_transfer.png       - 5 节点 OOD 跨规模泛化对比图（整体 + 分桶）
4. fig4_training_curves.png    - Stage 1 SFT Loss 与 Stage 2 GRPO Reward / Valid / Val / KL 曲线
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".mpl_cache"),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# 全局样式设置（清爽学术风格）
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "qwen7b_results_and_logs" / "causal" / "grpo_lora" / "results"
OUT_DIR = ROOT / "qwen7b_results_and_logs" / "causal" / "grpo_lora" / "outputs"
DS_BASELINE_PATH = (
    ROOT
    / "HypoSpace-release"
    / "causal"
    / "benchmark_obs_count"
    / "full_repo_causal_n4_obs3_deepseek.json"
)
FIG_DIR = Path(__file__).resolve().parent / "figures"

COLORS = {
    "base": "#94A3B8",      # Slate gray
    "deepseek": "#F59E0B",  # Amber / Orange
    "sft": "#3B82F6",       # Royal blue
    "grpo": "#10B981",      # Emerald green
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rolling_mean(values: List[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(arr[start : i + 1]))
    return out


def plot_fig1_id_ablation(
    id_base: Dict[str, Any],
    id_sft: Dict[str, Any],
    id_grpo: Dict[str, Any],
    ds_raw: Dict[str, Any],
) -> None:
    """绘制 ID 79 题四方消融对比柱状图。"""
    ds_summary = id_grpo["summary"]["paired_baseline"]
    grpo_by_id = {r["observation_set_id"]: r for r in id_grpo["per_sample_results"]}
    ds_by_id = {r["observation_set_id"]: r for r in ds_raw["per_sample_results"]}
    ids = sorted(grpo_by_id.keys())
    _, p_val = stats.ttest_rel(
        [grpo_by_id[i]["recovery_rate"] for i in ids],
        [ds_by_id[i]["recovery_rate"] for i in ids],
    )

    methods = [
        ("Qwen2.5-7B\nBase", id_base["summary"], COLORS["base"]),
        ("DeepSeek-Flash\n(API Baseline)", ds_summary, COLORS["deepseek"]),
        ("Qwen2.5-7B +\nSFT-LoRA (Stage 1)", id_sft["summary"], COLORS["sft"]),
        ("Qwen2.5-7B +\nDiversity-GRPO (Ours)", id_grpo["summary"], COLORS["grpo"]),
    ]

    metrics = [
        ("Macro Validity", lambda s: s["macro"]["valid_rate"] * 100.0),
        ("Macro Recovery\n(= Uniqueness)", lambda s: s["macro"]["recovery_rate"] * 100.0),
        ("Pooled Recovery", lambda s: s["pooled"]["recovery_rate"] * 100.0),
    ]

    x = np.arange(len(metrics))
    width = 0.19
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for idx, (label, summary, color) in enumerate(methods):
        vals = [fn(summary) for _, fn in metrics]
        bars = ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.92,
            label=label.replace("\n", " "),
            color=color,
            edgecolor="#1E293B",
            linewidth=0.6,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.5,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold" if idx == 3 else "normal",
                color="#0F172A",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 124)
    ax.set_title("Causal ID Test (79 Held-out Tasks, Fixed-Prompt Protocol)")
    ax.legend(loc="upper right", fontsize=9.2)

    delta_rec = (
        id_grpo["summary"]["macro"]["recovery_rate"] - ds_summary["macro"]["recovery_rate"]
    ) * 100.0
    ax.text(
        0.02,
        0.96,
        f"GRPO vs. DeepSeek-Flash:\n{delta_rec:+.1f}% Macro Recovery (p = {p_val:.2e})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ECFDF5", "edgecolor": "#10B981"},
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_id_ablation.png", dpi=300)
    plt.close(fig)


def plot_fig2_difficulty_scaling(
    id_base: Dict[str, Any],
    id_sft: Dict[str, Any],
    id_grpo: Dict[str, Any],
) -> None:
    """绘制不同 |H_O| 假设空间大小桶下的 Recovery 衰减曲线。"""
    ds_summary = id_grpo["summary"]["paired_baseline"]
    buckets = ["1", "2-4", "5-8", "9+"]
    bucket_labels = [
        f"|H_O| = {b}\n(n={id_grpo['summary']['by_gt_bucket'][b]['n_tasks']})"
        for b in buckets
    ]

    series = [
        ("Qwen2.5-7B Base", id_base["summary"]["by_gt_bucket"], COLORS["base"], "o", "--"),
        ("DeepSeek-Flash (API)", ds_summary["by_gt_bucket"], COLORS["deepseek"], "s", "-."),
        ("SFT-LoRA (Stage 1)", id_sft["summary"]["by_gt_bucket"], COLORS["sft"], "^", "-"),
        ("Diversity-GRPO-LoRA (Ours)", id_grpo["summary"]["by_gt_bucket"], COLORS["grpo"], "D", "-"),
    ]

    x = np.arange(len(buckets))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for label, bdict, color, marker, ls in series:
        ys = [bdict[b]["recovery_rate"] * 100.0 for b in buckets]
        lw = 2.8 if "Ours" in label else 2.0
        ms = 8 if "Ours" in label else 6.5
        ax.plot(
            x,
            ys,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=lw,
            markersize=ms,
            zorder=4 if "Ours" in label else 3,
        )
        for xi, yi in zip(x, ys):
            offset_y = 6
            if "SFT" in label and xi == 2:
                offset_y = 7
            elif "SFT" in label and xi == 1:
                offset_y = -13
            elif "Ours" in label and xi in (2, 3):
                offset_y = -14
            elif "Ours" in label and xi == 0:
                offset_y = -13
            ax.annotate(
                f"{yi:.1f}%",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center",
                fontsize=8.8,
                fontweight="bold" if "Ours" in label else "normal",
                color=color if "Base" not in label else "#475569",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylabel("Macro Recovery Rate (%)")
    ax.set_ylim(-3, 112)
    ax.set_title("Sublinear Coverage Breakdown Across Hypothesis Space Sizes |H_O|")
    ax.legend(loc="upper right", fontsize=9.5)

    # 强调 |H_O| >= 9 桶的 3x 提升
    grpo_hard = id_grpo["summary"]["by_gt_bucket"]["9+"]["recovery_rate"] * 100.0
    ds_hard = ds_summary["by_gt_bucket"]["9+"]["recovery_rate"] * 100.0
    ratio = grpo_hard / max(1e-6, ds_hard)
    ax.annotate(
        f"{ratio:.1f}x DeepSeek\non |H_O|>=9",
        xy=(3, grpo_hard),
        xytext=(2.18, 63),
        arrowprops={"arrowstyle": "->", "color": COLORS["grpo"], "lw": 1.5},
        fontsize=9.5,
        fontweight="bold",
        color="#065F46",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ECFDF5", "edgecolor": COLORS["grpo"]},
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_difficulty_scaling.png", dpi=300)
    plt.close(fig)


def plot_fig3_ood_transfer(
    ood_base: Dict[str, Any],
    ood_sft: Dict[str, Any],
    ood_grpo: Dict[str, Any],
) -> None:
    """绘制 5 节点 OOD 跨规模泛化对比图（左：整体指标；右：分桶 Recovery）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0))

    # 左图：整体 Macro Validity / Macro Recovery / Pooled Recovery
    methods = [
        ("Qwen2.5-7B Base", ood_base["summary"], COLORS["base"]),
        ("SFT-LoRA (Stage 1)", ood_sft["summary"], COLORS["sft"]),
        ("Diversity-GRPO (Ours)", ood_grpo["summary"], COLORS["grpo"]),
    ]
    metric_specs = [
        ("Macro Validity", lambda s: s["macro"]["valid_rate"] * 100.0),
        ("Macro Recovery", lambda s: s["macro"]["recovery_rate"] * 100.0),
        ("Pooled Recovery", lambda s: s["pooled"]["recovery_rate"] * 100.0),
    ]
    x1 = np.arange(len(metric_specs))
    w = 0.25
    offs = [-w, 0.0, w]

    for idx, (label, s, color) in enumerate(methods):
        vals = [fn(s) for _, fn in metric_specs]
        bars = ax1.bar(
            x1 + offs[idx],
            vals,
            width=w * 0.92,
            label=label,
            color=color,
            edgecolor="#1E293B",
            linewidth=0.6,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.5,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold" if idx == 2 else "normal",
            )

    ax1.set_xticks(x1)
    ax1.set_xticklabels([m[0] for m in metric_specs])
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 92)
    ax1.set_title("OOD 5-Node Overall (80 Tasks)")
    ax1.legend(loc="upper right", fontsize=9)

    # 右图：各难度分桶 Recovery
    buckets = ["1", "2-4", "5-8", "9+"]
    x2 = np.arange(len(buckets))
    for idx, (label, s, color) in enumerate(methods):
        vals = [s["by_gt_bucket"][b]["recovery_rate"] * 100.0 for b in buckets]
        bars = ax2.bar(
            x2 + offs[idx],
            vals,
            width=w * 0.92,
            label=label,
            color=color,
            edgecolor="#1E293B",
            linewidth=0.6,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.5,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [f"|H_O|={b}\n(n={ood_grpo['summary']['by_gt_bucket'][b]['n_tasks']})" for b in buckets]
    )
    ax2.set_ylabel("Macro Recovery (%)")
    ax2.set_ylim(0, 102)
    ax2.set_title("OOD 5-Node Recovery by |H_O| Bucket")

    fig.suptitle(
        "Zero-Shot Generalization from 3-4 Nodes to Unseen 5-Node Causal Graphs",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_ood_transfer.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_training_curves(sft_metrics: Dict[str, Any], grpo_rows: List[Dict[str, Any]]) -> None:
    """绘制 Stage 1 SFT 与 Stage 2 Diversity-GRPO 训练动态曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # Panel A: Stage 1 SFT Loss
    sft_steps = [r["step"] for r in sft_metrics["history"]]
    sft_losses = [r["loss"] for r in sft_metrics["history"]]
    axes[0].plot(sft_steps, sft_losses, color="#93C5FD", alpha=0.55, linewidth=1.2, label="Step Loss")
    axes[0].plot(
        sft_steps,
        rolling_mean(sft_losses, window=10),
        color=COLORS["sft"],
        linewidth=2.2,
        label="MA-10 Loss",
    )
    axes[0].set_title("Stage 1: SFT-LoRA Loss")
    axes[0].set_xlabel("Optimizer Step")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(loc="upper right", fontsize=9)

    # Panel B: Stage 2 GRPO Reward & Valid Rate
    steps = [r["step"] for r in grpo_rows]
    rewards = [r["mean_reward"] for r in grpo_rows]
    valids = [r["valid_rate"] * 100.0 for r in grpo_rows]
    ax_b = axes[1]
    ax_b_r = ax_b.twinx()
    ax_b_r.spines["top"].set_visible(False)

    l1 = ax_b.plot(
        steps,
        rolling_mean(rewards, window=40),
        color=COLORS["grpo"],
        linewidth=2.2,
        label="Mean Reward (MA-40)",
    )
    l2 = ax_b_r.plot(
        steps,
        rolling_mean(valids, window=40),
        color=COLORS["sft"],
        linewidth=2.0,
        linestyle="--",
        label="Rollout Valid % (MA-40)",
    )
    ax_b.set_title("Stage 2: GRPO Reward & Validity")
    ax_b.set_xlabel("GRPO Step")
    ax_b.set_ylabel("Mean Group Reward", color=COLORS["grpo"])
    ax_b_r.set_ylabel("Valid Rate (%)", color=COLORS["sft"])
    ax_b_r.set_ylim(75, 102)
    lines = l1 + l2
    ax_b.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=8.5)

    # Panel C: Validation Recovery & KL Divergence
    val_rows = [r for r in grpo_rows if "val_recovery_rate" in r]
    kls = [r["kl"] for r in grpo_rows]
    ax_c = axes[2]
    ax_c_r = ax_c.twinx()
    ax_c_r.spines["top"].set_visible(False)

    lc1 = ax_c.plot(
        [r["step"] for r in val_rows],
        [r["val_recovery_rate"] * 100.0 for r in val_rows],
        color="#8B5CF6",
        marker="o",
        linewidth=2.2,
        markersize=5.5,
        label="Val Recovery (%)",
    )
    lc2 = ax_c_r.plot(
        steps,
        rolling_mean(kls, window=20),
        color="#F97316",
        linewidth=1.8,
        alpha=0.9,
        label="KL(policy || ref) (MA-20)",
    )
    ax_c.set_title("Stage 2: Val Recovery & KL Stability")
    ax_c.set_xlabel("GRPO Step")
    ax_c.set_ylabel("Val Recovery (%)", color="#8B5CF6")
    ax_c_r.set_ylabel("KL Divergence", color="#F97316")
    lines_c = lc1 + lc2
    ax_c.legend(lines_c, [l.get_label() for l in lines_c], loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_training_curves.png", dpi=300)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    id_base = load_json(RES_DIR / "qwen7b_id_base.json")
    id_sft = load_json(RES_DIR / "qwen7b_id_sft.json")
    id_grpo = load_json(RES_DIR / "qwen7b_id_grpo.json")
    ood_base = load_json(RES_DIR / "qwen7b_ood_n5_base.json")
    ood_sft = load_json(RES_DIR / "qwen7b_ood_n5_sft.json")
    ood_grpo = load_json(RES_DIR / "qwen7b_ood_n5.json")
    ds_raw = load_json(DS_BASELINE_PATH)

    sft_metrics = load_json(OUT_DIR / "qwen7b_sft" / "sft_metrics.json")
    grpo_rows = [
        json.loads(line)
        for line in (OUT_DIR / "qwen7b_grpo" / "grpo_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    plot_fig1_id_ablation(id_base, id_sft, id_grpo, ds_raw)
    plot_fig2_difficulty_scaling(id_base, id_sft, id_grpo)
    plot_fig3_ood_transfer(ood_base, ood_sft, ood_grpo)
    plot_fig4_training_curves(sft_metrics, grpo_rows)

    for png in sorted(FIG_DIR.glob("*.png")):
        print(f"Generated: {png} ({png.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
