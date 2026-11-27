"""Stage 1 Boolean SFT-LoRA, adapted from the Causal SFT script."""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set

import numpy as np
import torch
from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import LoRAConfig, SFTConfig  # noqa: E402
from data_utils import build_fixed_prompt, format_chat_prompt, format_expression_target  # noqa: E402


def set_seed(seed: int) -> None:
    """固定随机种子以确保训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_task_gt_orders(
    train_tasks: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, List[int]]:
    """
    为每道训练题生成固定的随机 Ground-Truth 索引排列，用于跨 Epoch 无重复轮换切片。
    """
    rng = random.Random(seed)
    orders: Dict[str, List[int]] = {}
    for task in train_tasks:
        uid = task["task_uid"]
        n_gts = len(task.get("ground_truth_expressions", []))
        indices = list(range(n_gts))
        rng.shuffle(indices)
        orders[uid] = indices
    return orders


class SFTExample(NamedTuple):
    prompt: str
    target: str
    task_uid: str
    gt_idx: int


def build_sft_epoch_examples(
    train_tasks: List[Dict[str, Any]],
    max_samples_per_task: int,
    epoch_idx: int,
    task_gt_orders: Dict[str, List[int]],
    shuffle_operands: bool,
    seed: int,
) -> List[SFTExample]:
    """
    构造指定轮次的监督样本，并保留任务和答案索引。

    每道题抽取样本数 m = min(K, max_samples_per_task)，从 `task_gt_orders[uid]` 中
    按 `[(epoch_idx * m + j) % K for j in range(m)]` 循环轮换选取，
    在保持各任务每 epoch 权重不变的同时覆盖全部标准答案。
    """
    rng = random.Random(seed + epoch_idx * 1009)
    examples: List[SFTExample] = []

    for task in train_tasks:
        gts = list(task.get("ground_truth_expressions", []))
        n_gts = len(gts)
        if n_gts == 0:
            continue

        uid = task["task_uid"]
        order = task_gt_orders[uid]
        m = min(n_gts, max_samples_per_task)
        prompt = build_fixed_prompt(task)

        selected_indices = [(epoch_idx * m + j) % n_gts for j in range(m)]
        for pos in selected_indices:
            gt_idx = order[pos]
            target = format_expression_target(
                gts[gt_idx]["formula"], task,
                shuffle_operands=shuffle_operands,
                rng=rng,
            )
            examples.append(SFTExample(prompt, target, uid, gt_idx))

    rng.shuffle(examples)
    return examples


class BooleanSFTDataset(Dataset):
    """带 Prompt Loss Masking 的布尔表达式监督微调数据集。"""

    def __init__(
        self,
        examples: List[SFTExample],
        tokenizer: Any,
        max_seq_len: int,
    ) -> None:
        self.samples: List[Dict[str, Any]] = []
        eos_id = tokenizer.eos_token_id

        for raw_prompt, target, task_uid, gt_idx in examples:
            prompt_text = format_chat_prompt(tokenizer, raw_prompt)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if eos_id is not None:
                target_ids = target_ids + [eos_id]

            input_ids = prompt_ids + target_ids
            # 仅对模型需要生成的 target_ids 计算交叉熵损失，prompt 部分置为 -100
            labels = [-100] * len(prompt_ids) + target_ids

            if len(input_ids) > max_seq_len:
                raise ValueError(
                    f"{task_uid} requires {len(input_ids)} tokens; increase --max_seq_len"
                )

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
                "task_uid": task_uid,
                "gt_idx": gt_idx,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def collate_sft_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
) -> tuple[Dict[str, torch.Tensor], list[tuple[str, int]]]:
    """动态对齐 batch 内最大长度，padding 位置的 label 设为 -100。"""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids_list: List[List[int]] = []
    attn_mask_list: List[List[int]] = []
    labels_list: List[List[int]] = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids_list.append(item["input_ids"] + [pad_token_id] * pad_len)
        attn_mask_list.append(item["attention_mask"] + [0] * pad_len)
        labels_list.append(item["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
    }, [(item["task_uid"], item["gt_idx"]) for item in batch]


def train_sft(args: argparse.Namespace) -> None:
    """执行 Stage 1 SFT-LoRA 轮换训练并保存适配器与 GT 覆盖统计。"""
    set_seed(args.seed)
    splits_payload = json.loads(Path(args.splits_path).read_text(encoding="utf-8"))
    train_tasks = splits_payload["splits"]["train"]

    task_gt_orders = init_task_gt_orders(train_tasks, seed=args.seed)
    epoch0_examples = build_sft_epoch_examples(
        train_tasks=train_tasks,
        max_samples_per_task=args.max_samples_per_task,
        epoch_idx=0,
        task_gt_orders=task_gt_orders,
        shuffle_operands=not args.no_shuffle_operands,
        seed=args.seed,
    )
    print(f"Loaded {len(train_tasks)} train tasks -> {len(epoch0_examples)} SFT examples/epoch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    print(f"Device: {device}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()

    lora_defaults = LoRAConfig()
    peft_cfg = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=lora_defaults.lora_dropout,
        target_modules=list(lora_defaults.target_modules),
        bias="none",
    )
    model = get_peft_model(base_model, peft_cfg)
    model.to(device)
    model.print_trainable_parameters()

    num_batches_per_epoch = math.ceil(len(epoch0_examples) / args.batch_size)
    steps_per_epoch = max(1, math.ceil(num_batches_per_epoch / args.grad_accum))
    total_steps = args.max_steps if args.max_steps is not None else steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.05)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    optimizer.zero_grad()
    global_step = 0
    accum_loss = 0.0
    history: List[Dict[str, float]] = []
    cumulative_covered: Dict[str, Set[int]] = {
        t["task_uid"]: set() for t in train_tasks
    }

    for epoch in range(args.epochs):
        epoch_examples = build_sft_epoch_examples(
            train_tasks=train_tasks,
            max_samples_per_task=args.max_samples_per_task,
            epoch_idx=epoch,
            task_gt_orders=task_gt_orders,
            shuffle_operands=not args.no_shuffle_operands,
            seed=args.seed,
        )
        dataset = BooleanSFTDataset(epoch_examples, tokenizer, max_seq_len=args.max_seq_len)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_sft_batch(b, tokenizer.pad_token_id),
        )

        for step_idx, (batch, sources) in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            accum_start = (step_idx // args.grad_accum) * args.grad_accum
            accum_length = min(args.grad_accum, len(dataloader) - accum_start)
            loss = outputs.loss / accum_length
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Non-finite SFT loss at step {global_step + 1}')
            loss.backward()
            for task_uid, gt_idx in sources:
                cumulative_covered[task_uid].add(gt_idx)
            accum_loss += float(loss.item())

            is_update = ((step_idx + 1) % args.grad_accum == 0) or (
                step_idx + 1 == len(dataloader)
            )
            if not is_update:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            history.append({"step": global_step, "epoch": epoch + 1, "loss": accum_loss})
            if global_step % 10 == 0 or global_step == 1 or global_step == total_steps:
                print(
                    f"[SFT] step={global_step}/{total_steps} epoch={epoch + 1} loss={accum_loss:.4f}"
                )
            accum_loss = 0.0

            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    per_task_coverage: List[Dict[str, Any]] = []
    for task in train_tasks:
        uid = task["task_uid"]
        n_gts = len(task.get("ground_truth_expressions", []))
        n_cov = len(cumulative_covered.get(uid, set()))
        per_task_coverage.append(
            {
                "task_uid": uid,
                "observation_set_id": task["observation_set_id"],
                "n_ground_truths": n_gts,
                "covered_unique_gts": n_cov,
                "coverage_ratio": (n_cov / n_gts) if n_gts > 0 else 1.0,
            }
        )

    gt_coverage_summary = {
        "n_tasks": len(per_task_coverage),
        "full_coverage_tasks": sum(1 for r in per_task_coverage if r["covered_unique_gts"] >= r["n_ground_truths"]),
        "mean_coverage_ratio": float(np.mean([r["coverage_ratio"] for r in per_task_coverage])),
        "total_ground_truths": sum(r["n_ground_truths"] for r in per_task_coverage),
        "total_covered_unique_gts": sum(r["covered_unique_gts"] for r in per_task_coverage),
        "large_space_k9plus_mean_coverage": float(
            np.mean([r["coverage_ratio"] for r in per_task_coverage if r["n_ground_truths"] >= 9])
        )
        if any(r["n_ground_truths"] >= 9 for r in per_task_coverage)
        else 1.0,
    }

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics_path = output_dir / "sft_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "train_examples_per_epoch": len(epoch0_examples),
                "epochs": args.epochs,
                "total_steps": global_step,
                "gt_coverage_summary": gt_coverage_summary,
                "per_task_gt_coverage": per_task_coverage,
                "history": history,
            },
            indent=2, allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"Stage 1 SFT-LoRA saved to: {output_dir} (GT coverage={gt_coverage_summary['mean_coverage_ratio']:.2%})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    cfg = SFTConfig()
    lora_cfg = LoRAConfig()
    parser = argparse.ArgumentParser(description="Stage 1: SFT-LoRA warm-up for Boolean HypoSpace")
    parser.add_argument("--model_name", type=str, default=cfg.model_name)
    parser.add_argument("--splits_path", type=Path, default=cfg.splits_path)
    parser.add_argument("--output_dir", type=Path, default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--max_steps", type=int, default=cfg.max_steps)
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--grad_accum", type=int, default=cfg.grad_accum)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--max_seq_len", type=int, default=cfg.max_seq_len)
    parser.add_argument("--max_samples_per_task", type=int, default=cfg.max_samples_per_task)
    parser.add_argument("--no_shuffle_operands", action="store_true")
    parser.add_argument("--lora_r", type=int, default=lora_cfg.r)
    parser.add_argument("--lora_alpha", type=int, default=lora_cfg.lora_alpha)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--bf16", action="store_true", default=cfg.bf16)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train_sft(parse_args())
