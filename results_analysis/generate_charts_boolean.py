"""Render the Boolean ablation, difficulty, OOD, and training figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np


STYLES = {
    "base": ("Base", "#94A3B8"),
    "deepseek": ("DeepSeek", "#D58A08"),
    "sft": ("SFT", "#3B82F6"),
    "validity_best": ("Validity-only (best)", "#FCA5A5"),
    "validity_final": ("Validity-only (final)", "#EF4444"),
    "diversity": ("Diversity (best)", "#059669"),
    "diversity_final": ("Diversity (final)", "#6EE7B7"),
}
BUCKETS = ("1", "2-4", "5-8", "9+")
REFERENCE_COLOR = "#1E293B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.facecolor": "white",
})


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def index_models(path: Path) -> dict[str, dict]:
    records = {record["model_name"]: record for record in read_json(path)["models"]}
    return {name: records[name] for name in dict.fromkeys((*STYLES, *records)) if name in records}


def style(name: str) -> tuple[str, str]:
    return STYLES.get(name, (name, REFERENCE_COLOR))


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_ablation(models: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.6), sharey=True, layout="constrained")
    y = np.arange(len(models))
    for ax, (metric, title) in zip(axes, (
        ("valid_rate", "Validity"),
        ("uniqueness_rate", "Uniqueness"),
        ("recovery_rate", "Recovery"),
    )):
        values = [record["group_macro"][metric]["mean"] for record in models.values()]
        bars = ax.barh(y, values, color=[style(name)[1] for name in models], height=0.65)
        ax.bar_label(bars, [f"{v:.1%}" for v in values], padding=3, fontsize=8)
        ax.set(title=title, xlim=(0, 1.15), yticks=y, yticklabels=[style(name)[0] for name in models])
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    fig.suptitle("Boolean ID evaluation · mean across observation classes", fontweight="bold")
    save(fig, path)


def plot_difficulty(models: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    first = next(iter(models.values()))["by_bucket"]
    buckets = [bucket for bucket in BUCKETS if bucket in first]
    x = np.arange(len(buckets))
    for ax, (metric, title) in zip(axes, (
        ("recovery_rate", "Recovery"), ("recovery_norm", "Recovery / uniform reference"),
    )):
        for name, record in models.items():
            label, color = style(name)
            ax.plot(x, [record["by_bucket"][bucket][metric] for bucket in buckets],
                    color=color, label=label, marker="o")
        if metric == "recovery_rate":
            ax.plot(x, [first[bucket]["ceiling"] for bucket in buckets],
                    color=REFERENCE_COLOR, linestyle="--", marker="D", markersize=4,
                    label="Uniform reference")
        else:
            ax.axhline(1, color=REFERENCE_COLOR, linestyle="--", label="Uniform reference")
        ax.set(title=title, xticks=x, xticklabels=buckets, xlabel="Compatible mechanisms per task", ylim=(0, None))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(fontsize=8)
    save(fig, path)


def plot_transfer(id_models: dict, ood_models: dict, path: Path) -> None:
    common = [name for name in id_models if name in ood_models]
    if not common:
        raise ValueError("ID and OOD stats must contain at least one shared model")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for ax, (metric, title) in zip(axes, (
        ("recovery_rate", "Class-weighted Recovery"),
        ("recovery_norm", "Class-weighted normalized Recovery"),
    )):
        width = 0.8 / len(common)
        for index, name in enumerate(common):
            label, color = style(name)
            positions = np.arange(2) + (index - (len(common) - 1) / 2) * width
            values = [models[name]["group_macro"][metric]["mean"] for models in (id_models, ood_models)]
            ax.bar(positions, values, width=width, color=color, label=label)
        ax.set(title=title, xticks=[0, 1], xticklabels=["ID (extended)", "OOD (full + NOR)"])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(fontsize=8)
    save(fig, path)


def plot_training(sft: dict, grpo: dict[str, list[dict]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    history = sft["history"]
    axes[0].plot([r["step"] for r in history], [r["loss"] for r in history], color=style("sft")[1])
    axes[0].set(title="Stage 1", ylabel="SFT cross-entropy loss")

    for model, rows in grpo.items():
        label = "Diversity" if model == "diversity" else "Validity-only"
        color = style(model)[1]
        validation = [r for r in rows if "val_recovery_group_mean" in r]
        kl_rows = [r for r in rows if "kl" in r]
        linestyle = "-" if model == "diversity" else "--"
        axes[1].plot([r["step"] for r in validation],
                     [r["val_recovery_group_mean"] for r in validation],
                     color=color, linestyle=linestyle, marker="o", markersize=3, label=label)
        axes[2].plot([r["step"] for r in kl_rows], [r["kl"] for r in kl_rows], color=color, label=label)
    axes[1].set(title="Stage 2 validation", ylabel="Class-weighted Recovery")
    axes[1].set_ylim(bottom=-0.002, top=max(0.05, axes[1].get_ylim()[1]))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(fontsize=8)
    axes[2].set(title="Stage 2 training", ylabel="KL vs. SFT reference")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Optimizer step")
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Boolean stats and training logs")
    parser.add_argument("--id-stats", type=Path, required=True)
    parser.add_argument("--ood-stats", type=Path, required=True)
    parser.add_argument("--sft-metrics", type=Path, required=True)
    parser.add_argument("--diversity-log", type=Path, required=True)
    parser.add_argument("--validity-log", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "figures_boolean")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    id_models = index_models(args.id_stats)
    ood_models = index_models(args.ood_stats)
    training = {"diversity": read_jsonl(args.diversity_log)}
    if args.validity_log:
        training["validity_final"] = read_jsonl(args.validity_log)
    plot_ablation(id_models, args.output_dir / "fig1_id_ablation.png")
    plot_difficulty(id_models, args.output_dir / "fig2_difficulty_scaling.png")
    plot_transfer(id_models, ood_models, args.output_dir / "fig3_ood_transfer.png")
    plot_training(read_json(args.sft_metrics), training, args.output_dir / "fig4_training_curves.png")


if __name__ == "__main__":
    main()
