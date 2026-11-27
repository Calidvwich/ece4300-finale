"""
3D Voxel Reconstruction 统一 Fixed-Prompt 评测脚本（ID Test 与 4-Block OOD 通用）。

严格对齐：
- 独立采样（prior_structures=None，每题 N = |H_O| * query_multiplier）；
- 采样默认与 `SamplingConfig(T=0.9, top_p=0.95, max_new_tokens=128)` 完全一致；
- `Validity` 使用官方 `validate_structure_matches_observations`，`Uniqueness / Recovery`
  基于规范去重哈希 `Structure3D.get_hash()` 计算，保证与 DeepSeek 基线完全可比；
- 输出 JSON schema 满足任务书 0.4 节要求，可直接送入 `results_analysis/stats_3d.py`。
"""

from __future__ import annotations

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
    load_3d_dataset,
)
from verifier_reward import (  # noqa: E402
    decode_compact_layers,
    parse_llm_response,
    validate_structure_matches_observations,
)


def set_seed(seed: int) -> None:
    """固定评测随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_n_samples(
    model: Any,
    tokenizer: Any,
    raw_prompt: str,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    gen_batch_size: int,
    device: torch.device,
) -> List[str]:
    """使用 Fixed-Prompt 独立批采样生成 `n_samples` 条候选项。"""
    chat_prompt = format_chat_prompt(tokenizer, raw_prompt)
    prompt_ids = tokenizer.encode(
        chat_prompt, add_special_tokens=False, return_tensors="pt"
    ).to(device)
    prompt_len = prompt_ids.shape[1]

    responses: List[str] = []
    remaining = n_samples
    while remaining > 0:
        bsz = min(max(1, gen_batch_size), remaining)
        batch_input = prompt_ids.repeat(bsz, 1)
        batch_mask = torch.ones_like(batch_input)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input,
                attention_mask=batch_mask,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row in range(bsz):
            responses.append(
                tokenizer.decode(outputs[row, prompt_len:], skip_special_tokens=True).strip()
            )
        remaining -= bsz

    return responses


def evaluate_model(args: argparse.Namespace) -> None:
    """在指定 3D 数据集或 split 上运行统一 Fixed-Prompt 评测并保存结果 JSON。"""
    set_seed(args.seed)

    if args.dataset_path is not None:
        _, tasks = load_3d_dataset(args.dataset_path)
        eval_source = f"dataset:{args.dataset_path}"
        default_source_name = args.dataset_path.stem
    else:
        splits_payload = json.loads(Path(args.splits_path).read_text(encoding="utf-8"))
        tasks = splits_payload["splits"][args.split_name]
        eval_source = f"splits:{args.splits_path}:{args.split_name}"
        default_source_name = f"3d_splits_{args.split_name}"

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
        if (adapter_path / "policy" / "adapter_config.json").exists():
            adapter_path = adapter_path / "policy"
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"Loaded LoRA adapter from: {adapter_path}")
    model.to(device)
    model.eval()
    model.config.use_cache = True

    per_task_results: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        obs = task["observation"]
        grid_size = int(task.get("grid_size", 3))
        max_height = int(task.get("max_height", 3))
        raw_prompt = build_fixed_prompt(
            observations=obs,
            grid_size=grid_size,
            max_height=max_height,
        )
        gt_hashes = {
            decode_compact_layers(gt.get("layers", [])).get_hash()
            for gt in task.get("ground_truth_structures", [])
        }
        n_gt = max(1, len(gt_hashes))
        n_queries = max(1, int(round(n_gt * args.query_multiplier)))

        responses = generate_n_samples(
            model=model,
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            n_samples=n_queries,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            gen_batch_size=args.gen_batch_size,
            device=device,
        )

        parse_ok = 0
        valid_count = 0
        unique_all = set()
        unique_valid = set()

        for resp in responses:
            raw_struct = parse_llm_response(resp)
            if raw_struct is None:
                continue
            parse_ok += 1
            norm = raw_struct.normalize()
            s_hash = norm.get_hash()
            unique_all.add(s_hash)
            if validate_structure_matches_observations(norm, obs):
                valid_count += 1
                unique_valid.add(s_hash)

        recovered_gt = len(unique_valid & gt_hashes)
        obs_id = str(task.get("observation_set_id", task.get("observation_id", "unknown")))
        source_dataset = str(task.get("source_dataset", default_source_name))
        task_uid = f"{source_dataset}:{obs_id}"
        n_obs = 1 if isinstance(obs, str) else len(obs)

        per_task_results.append(
            {
                "observation_set_id": obs_id,
                "task_uid": task_uid,
                "n_observations": n_obs,
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

        if idx % 10 == 0 or idx == len(tasks):
            print(
                f"[3D-Eval] {idx}/{len(tasks)} "
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
        "seed": args.seed,
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
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
            "valid_rate": (total_valid / total_q) if total_q else 0.0,
            "uniqueness_rate": (total_uniq / total_q) if total_q else 0.0,
            "recovery_rate": (total_rec / total_gt) if total_gt else 0.0,
        },
        "by_gt_bucket": by_bucket,
    }

    if args.compare_baseline_path is not None and Path(args.compare_baseline_path).exists():
        ds_raw = json.loads(Path(args.compare_baseline_path).read_text(encoding="utf-8"))
        ds_by_id = {
            str(r["observation_set_id"]): r
            for r in ds_raw.get("per_sample_results", [])
        }
        common = [r for r in per_task_results if r["observation_set_id"] in ds_by_id]
        if common:
            ds_matched = [ds_by_id[r["observation_set_id"]] for r in common]
            summary["paired_baseline"] = {
                "n_tasks": len(common),
                "macro_valid": float(np.mean([r["valid_rate"] for r in ds_matched])),
                "macro_uniq": float(
                    np.mean([r.get("novelty_rate", r.get("uniqueness_rate", 0.0)) for r in ds_matched])
                ),
                "macro_recovery": float(np.mean([r["recovery_rate"] for r in ds_matched])),
            }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"summary": summary, "per_sample_results": per_task_results},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== 3D Evaluation Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation result to: {out_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    cfg = EvalConfig()
    parser = argparse.ArgumentParser(description="Fixed-Prompt evaluation for 3D GRPO-LoRA")
    parser.add_argument("--model_name", type=str, default=cfg.model_name)
    parser.add_argument("--adapter_dir", type=Path, default=cfg.adapter_dir)
    parser.add_argument("--splits_path", type=Path, default=cfg.splits_path)
    parser.add_argument("--split_name", type=str, default=cfg.split_name)
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
