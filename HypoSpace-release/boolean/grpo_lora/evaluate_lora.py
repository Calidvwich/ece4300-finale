"""Independent Boolean fixed-prompt evaluation for Base, SFT and GRPO checkpoints."""

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
    load_boolean_dataset,
    write_json,
)
from verifier_reward import score_task_responses  # noqa: E402


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
        _, tasks = load_boolean_dataset(args.dataset_path)
        eval_source = str(args.dataset_path)
    else:
        splits_payload = json.loads(Path(args.splits_path).read_text(encoding="utf-8"))
        tasks = splits_payload["splits"][args.split_name]
        eval_source = f"{args.splits_path}:{args.split_name}"

    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]
    if not tasks or args.gen_batch_size < 1:
        raise ValueError("Evaluation requires at least one task and positive gen_batch_size")

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
        n_queries = task["n_compatible_expressions"]
        prompt_text = format_chat_prompt(tokenizer, build_fixed_prompt(task))
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

        per_task_results.append(score_task_responses(responses, task))

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
    total_uniq = sum(r["n_unique_all"] for r in per_task_results)
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

    output_payload = {"summary": summary, "per_sample_results": per_task_results}
    write_json(args.output_path, output_payload)
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
    parser.add_argument("--output_path", type=Path, default=cfg.output_path)
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
