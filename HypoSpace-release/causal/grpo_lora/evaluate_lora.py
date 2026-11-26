"""
统一 Fixed-Prompt 评测脚本（支持 Base 模型 / SFT-LoRA / GRPO-LoRA）。

功能特性：
1. 严格遵循 HypoSpace 官方 Fixed-Prompt 协议（无历史提示，`N = |H_O| * query_multiplier`）；
2. 支持评测留出 ID 测试集（`splits/causal_splits.json` 中的 `test`）或任意外部 OOD 数据集（如 5 节点难题）；
3. 当评测 ID 测试集并提供队友保存的 `full_repo_causal_n4_obs3_deepseek.json` 时，
   自动在完全相同的 `observation_set_id` 上输出配对对比表（Macro、Pooled 及各 |H_O| 分桶）。
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import EvalConfig  # noqa: E402
from data_utils import (  # noqa: E402
    build_fixed_prompt,
    format_chat_prompt,
    get_gt_bucket,
    load_causal_dataset,
    summarize_baseline_on_split,
)
from verifier_reward import evaluate_observation_match, parse_causal_graph  # noqa: E402


def set_seed(seed: int) -> None:
    """固定随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(args: argparse.Namespace) -> Dict[str, Any]:
    """加载模型/适配器并在 Fixed-Prompt 协议下执行完整评测。"""
    set_seed(args.seed)

    if args.dataset_path is not None:
        _, tasks = load_causal_dataset(args.dataset_path)
        eval_source = str(args.dataset_path)
        default_source_name = args.dataset_path.stem
    else:
        splits_payload = json.loads(Path(args.splits_path).read_text(encoding="utf-8"))
        tasks = splits_payload["splits"][args.split_name]
        eval_source = f"{args.splits_path}:{args.split_name}"
        default_source_name = args.split_name

    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    if args.adapter_dir is not None:
        adapter_path = Path(args.adapter_dir)
        # 兼容 GRPO 保存时嵌套在 `policy/` 子目录或直接放在 `adapter_dir` 的情况
        if (adapter_path / "policy" / "adapter_config.json").exists():
            adapter_path = adapter_path / "policy"
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"Loaded LoRA adapter from: {adapter_path}")
    else:
        print("Evaluating Base Model without LoRA adapter.")

    model.to(device)
    model.eval()

    per_task_results: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        gt_graphs = task.get("ground_truth_graphs", [])
        n_gt = max(1, len(gt_graphs))
        n_queries = max(1, int(round(n_gt * args.query_multiplier)))
        gt_edge_sets = {
            frozenset(tuple(e) for e in g.get("edges", [])) for g in gt_graphs
        }

        prompt_text = format_chat_prompt(
            tokenizer,
            build_fixed_prompt(
                nodes=task["nodes"],
                observations=task["observations"],
                max_edges=task.get("max_edges"),
            ),
        )
        prompt_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        p_len = prompt_ids.shape[1]

        responses: List[str] = []
        remaining = n_queries
        while remaining > 0:
            bsz = min(args.gen_batch_size, remaining)
            batch_prompt = prompt_ids.repeat(bsz, 1)
            batch_mask = torch.ones_like(batch_prompt)
            with torch.no_grad():
                generated = model.generate(
                    input_ids=batch_prompt,
                    attention_mask=batch_mask,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            for row in range(bsz):
                text = tokenizer.decode(
                    generated[row, p_len:], skip_special_tokens=True
                ).strip()
                responses.append(text)
            remaining -= bsz

        parse_ok = 0
        valid_count = 0
        unique_all = set()
        unique_valid = set()

        for resp in responses:
            g = parse_causal_graph(
                resp, nodes=task["nodes"], max_edges=task.get("max_edges")
            )
            if g is None:
                continue
            parse_ok += 1
            edge_key = frozenset(g.edges)
            unique_all.add(edge_key)
            is_valid, _ = evaluate_observation_match(g, task["observations"])
            if is_valid:
                valid_count += 1
                unique_valid.add(edge_key)

        recovered_gt = len(unique_valid & gt_edge_sets)
        source_dataset = str(task.get("source_dataset", default_source_name))
        task_uid = f"{source_dataset}:{task['observation_set_id']}"
        per_task_results.append(
            {
                "observation_set_id": task["observation_set_id"],
                "task_uid": task_uid,
                "n_observations": int(task.get("n_observations", len(task["observations"]))),
                "n_ground_truths": n_gt,
                "n_queries": n_queries,
                "n_parsed": parse_ok,
                "n_valid": valid_count,
                "n_unique_all": len(unique_all),
                "n_unique_valid": len(unique_valid),
                "n_recovered_gts": recovered_gt,
                "parse_success_rate": parse_ok / n_queries,
                "valid_rate": valid_count / n_queries,
                "uniqueness_rate": len(unique_valid) / n_queries,
                "recovery_rate": recovered_gt / n_gt,
                "raw_responses": responses,
            }
        )

        if idx % 20 == 0 or idx == len(tasks):
            print(
                f"[Eval] {idx}/{len(tasks)} "
                f"macro_valid={np.mean([r['valid_rate'] for r in per_task_results]):.4f} "
                f"macro_uniq={np.mean([r['uniqueness_rate'] for r in per_task_results]):.4f} "
                f"macro_rec={np.mean([r['recovery_rate'] for r in per_task_results]):.4f}"
            )

    n_tasks = len(per_task_results)
    total_q = sum(r["n_queries"] for r in per_task_results)
    total_valid = sum(r["n_valid"] for r in per_task_results)
    total_uniq = sum(r["n_unique_valid"] for r in per_task_results)
    total_gt = sum(r["n_ground_truths"] for r in per_task_results)
    total_rec = sum(r["n_recovered_gts"] for r in per_task_results)

    by_bucket: Dict[str, Dict[str, float]] = {}
    for bucket in ("1", "2-4", "5-8", "9+"):
        sub = [
            r for r in per_task_results if get_gt_bucket(r["n_ground_truths"]) == bucket
        ]
        if sub:
            by_bucket[bucket] = {
                "n_tasks": len(sub),
                "valid_rate": float(np.mean([r["valid_rate"] for r in sub])),
                "uniqueness_rate": float(np.mean([r["uniqueness_rate"] for r in sub])),
                "recovery_rate": float(np.mean([r["recovery_rate"] for r in sub])),
            }

    summary: Dict[str, Any] = {
        "model_name": args.model_name,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "eval_source": eval_source,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "n_tasks": n_tasks,
        "total_queries": total_q,
        "total_ground_truths": total_gt,
        "macro": {
            "parse_success_rate": float(np.mean([r["parse_success_rate"] for r in per_task_results])),
            "valid_rate": float(np.mean([r["valid_rate"] for r in per_task_results])),
            "uniqueness_rate": float(np.mean([r["uniqueness_rate"] for r in per_task_results])),
            "recovery_rate": float(np.mean([r["recovery_rate"] for r in per_task_results])),
        },
        "pooled": {
            "valid_rate": total_valid / max(1, total_q),
            "uniqueness_rate": total_uniq / max(1, total_q),
            "recovery_rate": total_rec / max(1, total_gt),
        },
        "by_gt_bucket": by_bucket,
    }

    if (
        args.dataset_path is None
        and args.compare_baseline_path is not None
        and Path(args.compare_baseline_path).exists()
    ):
        task_ids = [r["observation_set_id"] for r in per_task_results]
        baseline_summary = summarize_baseline_on_split(args.compare_baseline_path, task_ids)
        summary["paired_baseline"] = baseline_summary

        print("\n=== Paired Comparison (Same Observation Sets, Fixed-Prompt) ===")
        print(
            f"DeepSeek-Flash | Valid={baseline_summary['macro']['valid_rate']:.2%} | "
            f"Uniq={baseline_summary['macro']['uniqueness_rate']:.2%} | "
            f"Recovery={baseline_summary['macro']['recovery_rate']:.2%}"
        )
        print(
            f"Evaluated Model| Valid={summary['macro']['valid_rate']:.2%} | "
            f"Uniq={summary['macro']['uniqueness_rate']:.2%} | "
            f"Recovery={summary['macro']['recovery_rate']:.2%}"
        )

    output_payload = {"summary": summary, "per_sample_results": per_task_results}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Saved evaluation result to: {args.output_path}")
    return output_payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    cfg = EvalConfig()
    parser = argparse.ArgumentParser(description="Evaluate Base/SFT/GRPO LoRA under Fixed-Prompt")
    parser.add_argument("--model_name", type=str, default=cfg.model_name)
    parser.add_argument("--adapter_dir", type=Path, default=cfg.adapter_dir)
    parser.add_argument("--splits_path", type=Path, default=cfg.splits_path)
    parser.add_argument("--split_name", type=str, default=cfg.split_name, choices=["train", "val", "test"])
    parser.add_argument("--dataset_path", type=Path, default=cfg.dataset_path)
    parser.add_argument("--compare_baseline_path", type=Path, default=cfg.compare_baseline_path)
    parser.add_argument("--output_path", type=Path, default=cfg.output_path)
    parser.add_argument("--query_multiplier", type=float, default=cfg.query_multiplier)
    parser.add_argument("--gen_batch_size", type=int, default=cfg.gen_batch_size)
    parser.add_argument("--temperature", type=float, default=cfg.temperature)
    parser.add_argument("--top_p", type=float, default=cfg.top_p)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.max_new_tokens)
    parser.add_argument("--max_tasks", type=int, default=cfg.max_tasks)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--bf16", action="store_true", default=cfg.bf16)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate_model(parse_args())
