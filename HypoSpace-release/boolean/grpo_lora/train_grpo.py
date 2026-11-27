"""Stage 2 Boolean Diversity/Validity-Only GRPO, adapted from the Causal GRPO script."""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig as PeftLoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import GRPOConfig, LoRAConfig, RewardConfig  # noqa: E402
from data_utils import (  # noqa: E402
    build_fixed_prompt,
    canonical_isomorphism_key,
    format_chat_prompt,
)
from verifier_reward import (  # noqa: E402
    compute_grpo_advantages,
    compute_group_diversity_rewards,
    score_task_responses,
)


def set_seed(seed: int) -> None:
    """固定随机种子以确保训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def disable_all_dropout(model: torch.nn.Module) -> int:
    """将模型内所有 Dropout 层概率置为 0，确保 PPO 比率与参考 KL 零随机噪声。"""
    count = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
            count += 1
    return count


def theoretical_uniform_ceiling(n_ground_truths: int) -> float:
    """计算独立均匀采样 $K$ 次时单个样本恢复率的理论期望上限 $1 - (1 - 1/K)^K$。"""
    k = max(1, int(n_ground_truths))
    return 1.0 - (1.0 - 1.0 / float(k)) ** k


def select_stratified_val_tasks(
    val_tasks: List[Dict[str, Any]],
    tasks_per_group: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    在训练开始前对验证集按同构结构组做固定种子分层抽样（仅抽样一次）。

    - 使用 `data_utils.canonical_isomorphism_key` 聚类结构组并赋予确定性 `group_id`；
    - 当 `tasks_per_group >= 0` 时，每组抽 `min(tasks_per_group, len(group))` 题；
    - 当 `tasks_per_group < 0` 时，保留全部验证题。
    """
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for task in val_tasks:
        grouped[canonical_isomorphism_key(task)].append(dict(task))

    sorted_groups = sorted(
        grouped.values(),
        key=lambda grp: (
            int(grp[0].get("n_compatible_expressions", 1)),
            min(t["observation_set_id"] for t in grp),
        ),
    )

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for g_idx, grp in enumerate(sorted_groups):
        grp_sorted = sorted(grp, key=lambda t: t["observation_set_id"])
        k_val = int(grp_sorted[0].get("n_compatible_expressions", 1))
        group_id = f"G{g_idx + 1}_K{k_val}"

        if tasks_per_group >= 0 and len(grp_sorted) > tasks_per_group:
            chosen = rng.sample(grp_sorted, tasks_per_group)
            chosen.sort(key=lambda t: t["observation_set_id"])
        else:
            chosen = grp_sorted

        for t in chosen:
            item = dict(t)
            item["_val_group_id"] = group_id
            selected.append(item)

    return selected


def compute_group_weighted_val_metrics(
    task_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    计算验证集的多维度指标：先在各同构结构组内求平均，再对所有组做等权平均。

    返回字段包含：
    - `val_recovery_group_mean`：主选优指标（各结构组等权平均 Recovery）
    - `val_recovery_norm_group_mean`：各组归一化 Recovery（除以 $1 - (1 - 1/K)^K$）的等权平均
    - `val_per_group`：每个结构组的详细统计
    - `val_tasks`、`val_valid_rate`、`val_uniqueness_rate`、`val_recovery_rate`：按题平均的兼容指标
    """
    if not task_records:
        return {
            "val_tasks": 0.0,
            "val_valid_rate": 0.0,
            "val_uniqueness_rate": 0.0,
            "val_recovery_rate": 0.0,
            "val_recovery_group_mean": 0.0,
            "val_recovery_norm_group_mean": 0.0,
            "val_per_group": {},
        }

    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in task_records:
        gid = str(rec.get("group_id", "default"))
        by_group[gid].append(rec)

    val_per_group: Dict[str, Dict[str, Any]] = {}
    group_recs: List[float] = []
    group_norm_recs: List[float] = []

    for gid in sorted(by_group.keys()):
        items = by_group[gid]
        g_valid = float(np.mean([it["valid_rate"] for it in items]))
        g_uniq = float(np.mean([it["uniqueness_rate"] for it in items]))
        g_rec = float(np.mean([it["recovery_rate"] for it in items]))
        g_norm_rec = float(
            np.mean(
                [
                    it["recovery_rate"]
                    / max(1e-8, theoretical_uniform_ceiling(int(it["n_ground_truths"])))
                    for it in items
                ]
            )
        )
        val_per_group[gid] = {
            "n_tasks": len(items),
            "n_ground_truths": int(items[0]["n_ground_truths"]),
            "valid_rate": g_valid,
            "uniqueness_rate": g_uniq,
            "recovery_rate": g_rec,
            "recovery_norm_rate": g_norm_rec,
        }
        group_recs.append(g_rec)
        group_norm_recs.append(g_norm_rec)

    return {
        "val_tasks": float(len(task_records)),
        "val_valid_rate": float(np.mean([r["valid_rate"] for r in task_records])),
        "val_uniqueness_rate": float(np.mean([r["uniqueness_rate"] for r in task_records])),
        "val_recovery_rate": float(np.mean([r["recovery_rate"] for r in task_records])),
        "val_recovery_group_mean": float(np.mean(group_recs)),
        "val_recovery_norm_group_mean": float(np.mean(group_norm_recs)),
        "val_per_group": val_per_group,
    }


def compute_completion_logprobs(
    model: PeftModel,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """计算批量样本在补全部分每个 token 上的对数概率 `log pi(y_t | x, y_<t)`。"""
    outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    # Normalize only sampled completion tokens in float32. Keep a single
    # connected zero when every response terminated before sampling a token.
    mask = completion_mask.bool()
    shift_logits = outputs.logits[:, :-1, :][mask].float()
    shift_labels = full_ids[:, 1:][mask]
    token_logprobs = -F.cross_entropy(shift_logits, shift_labels, reduction="none")
    return torch.zeros_like(completion_mask).masked_scatter(mask, token_logprobs)


def compute_completion_logprobs_chunked(
    model: PeftModel,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    micro_batch_size: int,
) -> torch.Tensor:
    """按 `micro_batch_size` 分块计算补全对数概率，防止整组同时展开大词表 logits 导致 OOM。"""
    chunks: List[torch.Tensor] = []
    total = full_ids.shape[0]
    step = max(1, micro_batch_size)
    for start in range(0, total, step):
        end = min(total, start + step)
        chunks.append(
            compute_completion_logprobs(
                model,
                full_ids[start:end],
                attention_mask[start:end],
                completion_mask[start:end],
            )
        )
    return torch.cat(chunks, dim=0)


def sample_group_rollouts(
    model: PeftModel,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """使用当前 `"policy"` 适配器针对同一个 prompt 批量采样 `group_size` 条响应。"""
    model.eval()
    model.config.use_cache = True
    prompt_len = prompt_ids.shape[1]
    batch_prompt = prompt_ids.repeat(group_size, 1).to(device)
    batch_mask = torch.ones_like(batch_prompt, dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=batch_prompt,
            attention_mask=batch_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    seq_len = generated.shape[1]
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    attn_mask = torch.ones((group_size, seq_len), dtype=torch.long, device=device)
    comp_mask = torch.zeros((group_size, seq_len - 1), dtype=torch.float32, device=device)
    decoded_texts: List[str] = []

    for row in range(group_size):
        comp_tokens = generated[row, prompt_len:]
        valid_len = len(comp_tokens)
        for pos, tok in enumerate(comp_tokens.tolist()):
            if (eos_id is not None and tok == eos_id) or (tok == pad_id):
                valid_len = pos + (1 if (eos_id is not None and tok == eos_id) else 0)
                break

        end_idx = prompt_len + valid_len
        if end_idx < seq_len:
            attn_mask[row, end_idx:] = 0

        start_shift = max(0, prompt_len - 1)
        end_shift = max(start_shift, end_idx - 1)
        comp_mask[row, start_shift:end_shift] = 1.0

        text = tokenizer.decode(
            generated[row, prompt_len:end_idx],
            skip_special_tokens=True,
        ).strip()
        decoded_texts.append(text)

    return generated, attn_mask, comp_mask, decoded_texts


def evaluate_val_split(
    model: PeftModel,
    tokenizer: Any,
    val_tasks: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    val_seed: int,
    gen_batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    在固定验证集子集上使用与 rollout 完全一致的采样参数执行确定性、分批评估，
    并按结构组等权汇总 `val_recovery_group_mean`。
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        set_seed(val_seed)
        model.eval()
        model.config.use_cache = True
        model.set_adapter("policy")

        task_records: List[Dict[str, Any]] = []
        for task in val_tasks:
            prompt_text = format_chat_prompt(
                tokenizer,
                build_fixed_prompt(task),
            )
            prompt_ids = tokenizer.encode(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            p_len = prompt_ids.shape[1]
            n_gt = max(1, int(task.get("n_compatible_expressions", 1)))

            texts: List[str] = []
            remaining = n_gt
            while remaining > 0:
                bsz = min(max(1, gen_batch_size), remaining)
                batch_prompt = prompt_ids.repeat(bsz, 1)
                batch_mask = torch.ones_like(batch_prompt)
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=batch_prompt,
                        attention_mask=batch_mask,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                for row in range(bsz):
                    texts.append(
                        tokenizer.decode(generated[row, p_len:], skip_special_tokens=True).strip()
                    )
                remaining -= bsz

            summary = score_task_responses(texts, task)
            summary["group_id"] = task["_val_group_id"]
            task_records.append(summary)

        return compute_group_weighted_val_metrics(task_records)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def load_policy_and_ref_model(
    model_name: str,
    sft_adapter_dir: Optional[Path],
    dtype: torch.dtype,
    device: torch.device,
    gradient_checkpointing: bool = True,
) -> Tuple[PeftModel, Optional[str]]:
    """加载基础模型与双 LoRA 适配器（并关闭所有 Dropout 以保证比率与 KL 一致性）。"""
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = True
    if gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()

    if sft_adapter_dir is not None:
        if not (Path(sft_adapter_dir) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"SFT adapter missing from {sft_adapter_dir}")
        model = PeftModel.from_pretrained(
            base_model,
            str(sft_adapter_dir),
            adapter_name="policy",
            is_trainable=True,
        )
        model.load_adapter(
            str(sft_adapter_dir),
            adapter_name="ref",
            is_trainable=False,
        )
        ref_adapter_name: Optional[str] = "ref"
    else:
        lora_defaults = LoRAConfig()
        peft_cfg = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_defaults.r,
            lora_alpha=lora_defaults.lora_alpha,
            lora_dropout=0.0,
            target_modules=list(lora_defaults.target_modules),
            bias="none",
        )
        model = get_peft_model(base_model, peft_cfg, adapter_name="policy")
        ref_adapter_name = None

    disabled_dropouts = disable_all_dropout(model)
    print(f"Disabled {disabled_dropouts} dropout modules for deterministic GRPO log-probs.")
    model.set_adapter("policy")
    model.to(device)
    return model, ref_adapter_name


def train_grpo(args: argparse.Namespace) -> None:
    """执行 Stage 2 Diversity-Aware GRPO-LoRA 主训练流程。"""
    set_seed(args.seed)
    splits_payload = json.loads(Path(args.splits_path).read_text(encoding="utf-8"))

    all_train = splits_payload["splits"]["train"]
    train_tasks = [
        t for t in all_train if int(t.get("n_compatible_expressions", 0)) >= args.min_gt_for_grpo
    ]
    raw_val_tasks = splits_payload["splits"]["val"]
    if not train_tasks:
        raise RuntimeError("No training tasks satisfy min_gt_for_grpo filter.")

    # 验证集在训练开始前只抽样一次，之后所有验证步均使用完全相同的题目子集
    selected_val_tasks = select_stratified_val_tasks(
        val_tasks=raw_val_tasks,
        tasks_per_group=args.val_tasks_per_group,
        seed=args.val_seed,
    )
    if args.val_max_tasks is not None:
        selected_val_tasks = selected_val_tasks[: args.val_max_tasks]

    if not selected_val_tasks:
        raise ValueError("GRPO requires at least one validation task")
    tasks_per_step = max(1, args.tasks_per_step)
    steps_per_epoch = math.ceil(len(train_tasks) / tasks_per_step)
    total_opt_steps = (
        args.max_steps if args.max_steps is not None else steps_per_epoch * args.epochs
    )
    warmup_steps = int(total_opt_steps * 0.10)

    print(
        f"GRPO train tasks: {len(train_tasks)}/{len(all_train)} (min_gt={args.min_gt_for_grpo}), "
        f"val tasks: {len(selected_val_tasks)}/{len(raw_val_tasks)} "
        f"(val_tasks_per_group={args.val_tasks_per_group}), tasks_per_step={tasks_per_step}, "
        f"total_opt_steps={total_opt_steps}, reward_mode={args.reward_mode}, "
        f"sampling=(T={args.temperature}, top_p={args.top_p}, max_new_tokens={args.max_new_tokens})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, ref_adapter_name = load_policy_and_ref_model(
        model_name=args.model_name,
        sft_adapter_dir=args.sft_adapter_dir,
        dtype=dtype,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainable_params = [
        p for n, p in model.named_parameters() if p.requires_grad and "policy" in n
    ]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    reward_cfg = RewardConfig(
        mode=args.reward_mode,
        valid_floor_reward=args.valid_floor_reward,
        diversity_weight=args.diversity_weight,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best_policy"
    metrics_file = output_dir / "grpo_metrics.jsonl"
    metrics_file.write_text("", encoding="utf-8")

    # Step 0：训练前先评估 SFT 起点作为保底基准，初始化 best_policy
    step0_val_stats = evaluate_val_split(
        model=model,
        tokenizer=tokenizer,
        val_tasks=selected_val_tasks,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        val_seed=args.val_seed,
        gen_batch_size=args.val_gen_batch_size,
        device=device,
    )
    step0_row: Dict[str, Any] = {
        "step": 0,
        "epoch": 0,
        "reward_mode": args.reward_mode,
        **step0_val_stats,
    }
    with metrics_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(step0_row, allow_nan=False) + "\n")

    best_val_score: float = float(step0_val_stats["val_recovery_group_mean"])
    best_step: int = 0
    model.set_adapter("policy")
    model.save_pretrained(best_dir, selected_adapters=["policy"])
    tokenizer.save_pretrained(best_dir)
    print(
        f"[GRPO:{args.reward_mode}] step=0 (SFT init) | "
        f"val_valid={step0_val_stats['val_valid_rate']:.3f} "
        f"val_group_rec={best_val_score:.3f} "
        f"val_norm_group_rec={step0_val_stats['val_recovery_norm_group_mean']:.3f}"
    )

    rng = random.Random(args.seed)
    opt_step = 0
    mb_step = max(1, args.micro_batch_size)

    for epoch in range(args.epochs):
        epoch_tasks = list(train_tasks)
        rng.shuffle(epoch_tasks)

        for batch_start in range(0, len(epoch_tasks), tasks_per_step):
            task_batch = epoch_tasks[batch_start : batch_start + tasks_per_step]
            n_batch_tasks = len(task_batch)
            opt_step += 1

            rollout_items: List[Dict[str, Any]] = []
            for task in task_batch:
                prompt_text = format_chat_prompt(
                    tokenizer,
                    build_fixed_prompt(task),
                )
                prompt_ids = tokenizer.encode(
                    prompt_text, add_special_tokens=False, return_tensors="pt"
                )

                model.set_adapter("policy")
                full_ids, attn_mask, comp_mask, texts = sample_group_rollouts(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    group_size=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )

                reward_res = compute_group_diversity_rewards(
                    responses=texts,
                    task=task,
                    cfg=reward_cfg,
                )
                advantages = torch.tensor(
                    compute_grpo_advantages(reward_res.rewards),
                    dtype=torch.float32,
                    device=device,
                )

                model.eval()
                model.config.use_cache = False
                with torch.no_grad():
                    old_logprobs = compute_completion_logprobs_chunked(
                        model, full_ids, attn_mask, comp_mask, mb_step
                    )
                    if ref_adapter_name is not None:
                        model.set_adapter(ref_adapter_name)
                        ref_logprobs = compute_completion_logprobs_chunked(
                            model, full_ids, attn_mask, comp_mask, mb_step
                        )
                    else:
                        with model.disable_adapter():
                            ref_logprobs = compute_completion_logprobs_chunked(
                                model, full_ids, attn_mask, comp_mask, mb_step
                            )

                rollout_items.append(
                    {
                        "task_id": task["observation_set_id"],
                        "task_uid": task["task_uid"],
                        "n_gt": int(task.get("n_compatible_expressions", 1)),
                        "full_ids": full_ids,
                        "attn_mask": attn_mask,
                        "comp_mask": comp_mask,
                        "advantages": advantages,
                        "old_logprobs": old_logprobs,
                        "ref_logprobs": ref_logprobs,
                        "reward_res": reward_res,
                    }
                )

            model.set_adapter("policy")
            model.train()
            model.config.use_cache = False

            step_loss_val = 0.0
            step_kl_val = 0.0
            for _ in range(args.ppo_epochs):
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_kl = 0.0

                for item in rollout_items:
                    full_ids = item["full_ids"]
                    attn_mask = item["attn_mask"]
                    comp_mask = item["comp_mask"]
                    advantages = item["advantages"]
                    old_lp = item["old_logprobs"]
                    ref_lp = item["ref_logprobs"]

                    total_group = full_ids.shape[0]
                    token_counts = comp_mask.sum(dim=-1).clamp(min=1.0)
                    denom_scale = float(total_group * n_batch_tasks)

                    for start in range(0, total_group, mb_step):
                        end = min(total_group, start + mb_step)
                        cur_lp = compute_completion_logprobs(
                            model,
                            full_ids[start:end],
                            attn_mask[start:end],
                            comp_mask[start:end],
                        )
                        ratio = torch.exp(cur_lp - old_lp[start:end])
                        adv_exp = advantages[start:end].unsqueeze(-1)
                        surr1 = ratio * adv_exp
                        surr2 = (
                            torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                            * adv_exp
                        )
                        pg_loss_per_token = -torch.min(surr1, surr2)

                        log_ratio_ref = ref_lp[start:end] - cur_lp
                        kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

                        mb_mask = comp_mask[start:end]
                        mb_counts = token_counts[start:end]
                        loss_per_seq = (
                            (pg_loss_per_token + args.kl_coef * kl_per_token) * mb_mask
                        ).sum(dim=-1) / mb_counts

                        chunk_loss = loss_per_seq.sum() / denom_scale
                        chunk_loss.backward()

                        accum_loss += float(chunk_loss.item())
                        chunk_kl = (
                            ((kl_per_token * mb_mask).sum(dim=-1) / mb_counts).sum()
                            / denom_scale
                        )
                        accum_kl += float(chunk_kl.item())

                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                scheduler.step()
                step_loss_val = accum_loss
                step_kl_val = accum_kl

            mean_reward = float(
                np.mean([r for it in rollout_items for r in it["reward_res"].rewards])
            )
            mean_valid = float(
                np.mean([it["reward_res"].valid_rate for it in rollout_items])
            )
            mean_uniq = float(
                np.mean([it["reward_res"].unique_valid_rate for it in rollout_items])
            )
            mean_parse = float(
                np.mean([it["reward_res"].parse_rate for it in rollout_items])
            )

            log_row: Dict[str, Any] = {
                "step": opt_step,
                "epoch": epoch + 1,
                "task_ids": [it["task_id"] for it in rollout_items],
                "task_uids": [it["task_uid"] for it in rollout_items],
                "reward_mode": args.reward_mode,
                "lr": float(scheduler.get_last_lr()[0]),
                "loss": step_loss_val,
                "kl": step_kl_val,
                "mean_reward": mean_reward,
                "valid_rate": mean_valid,
                "unique_valid_rate": mean_uniq,
                "parse_rate": mean_parse,
            }

            if opt_step % args.eval_every_steps == 0 or opt_step == total_opt_steps:
                val_stats = evaluate_val_split(
                    model=model,
                    tokenizer=tokenizer,
                    val_tasks=selected_val_tasks,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    val_seed=args.val_seed,
                    gen_batch_size=args.val_gen_batch_size,
                    device=device,
                )
                log_row.update(val_stats)
                # 严格大于 `>` 选优：防止在同分或微弱波动时替换掉 SFT 起点或更早稳健点
                if val_stats["val_recovery_group_mean"] > best_val_score:
                    best_val_score = float(val_stats["val_recovery_group_mean"])
                    best_step = opt_step
                    model.set_adapter("policy")
                    model.save_pretrained(best_dir, selected_adapters=["policy"])
                    tokenizer.save_pretrained(best_dir)

            with metrics_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row, allow_nan=False) + "\n")

            if opt_step % 2 == 0 or opt_step == 1 or opt_step == total_opt_steps:
                msg = (
                    f"[GRPO:{args.reward_mode}] step={opt_step}/{total_opt_steps} "
                    f"reward={mean_reward:.3f} valid={mean_valid:.2f} "
                    f"uniq={mean_uniq:.2f} kl={step_kl_val:.4f}"
                )
                if "val_recovery_group_mean" in log_row:
                    msg += (
                        f" | val_valid={log_row['val_valid_rate']:.3f} "
                        f"val_group_rec={log_row['val_recovery_group_mean']:.3f} "
                        f"(best={best_val_score:.3f}@step{best_step})"
                    )
                print(msg)

            if args.max_steps is not None and opt_step >= args.max_steps:
                break
        if args.max_steps is not None and opt_step >= args.max_steps:
            break

    final_dir = output_dir / "final_policy"
    model.set_adapter("policy")
    model.save_pretrained(final_dir, selected_adapters=["policy"])
    tokenizer.save_pretrained(final_dir)

    summary_payload = {
        "model_name": args.model_name,
        "reward_mode": args.reward_mode,
        "total_opt_steps": opt_step,
        "best_step": best_step,
        "best_val_recovery_group_mean": best_val_score,
        "step0_val_recovery_group_mean": float(step0_val_stats["val_recovery_group_mean"]),
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
        },
        "val_tasks_count": len(selected_val_tasks),
        "val_tasks_per_group": args.val_tasks_per_group,
    }
    (output_dir / "grpo_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    if best_step == 0:
        print(
            "\n" + "=" * 68 + "\n"
            f"WARNING [GRPO:{args.reward_mode}]: best_step == 0! "
            "No checkpoint strictly outperformed the Stage 1 SFT initialization "
            f"(val_recovery_group_mean={best_val_score:.4f}). "
            "best_policy/ retains the SFT initialization.\n"
            + "=" * 68
        )

    print(
        f"Stage 2 GRPO-LoRA saved to: {final_dir} "
        f"(best_policy={best_dir}, best_step={best_step}, best_val_group_rec={best_val_score:.4f})"
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    cfg = GRPOConfig()
    parser = argparse.ArgumentParser(description="Stage 2: Diversity-Aware GRPO-LoRA training (v2.1)")
    parser.add_argument("--model_name", type=str, default=cfg.model_name)
    parser.add_argument("--sft_adapter_dir", type=Path, default=cfg.sft_adapter_dir)
    parser.add_argument("--no_sft_adapter", action="store_true")
    parser.add_argument("--splits_path", type=Path, default=cfg.splits_path)
    parser.add_argument("--output_dir", type=Path, default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--max_steps", type=int, default=cfg.max_steps)
    parser.add_argument("--min_gt_for_grpo", type=int, default=cfg.min_gt_for_grpo)
    parser.add_argument("--group_size", type=int, default=cfg.group_size)
    parser.add_argument("--micro_batch_size", type=int, default=cfg.micro_batch_size)
    parser.add_argument("--tasks_per_step", type=int, default=cfg.tasks_per_step)
    parser.add_argument("--ppo_epochs", type=int, default=cfg.ppo_epochs)
    parser.add_argument("--clip_eps", type=float, default=cfg.clip_eps)
    parser.add_argument("--kl_coef", type=float, default=cfg.kl_coef)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--max_grad_norm", type=float, default=cfg.max_grad_norm)
    parser.add_argument("--temperature", type=float, default=cfg.temperature)
    parser.add_argument("--top_p", type=float, default=cfg.top_p)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.max_new_tokens)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default=cfg.reward_mode,
        choices=["diversity", "validity_only"],
    )
    parser.add_argument("--valid_floor_reward", type=float, default=cfg.reward.valid_floor_reward)
    parser.add_argument("--diversity_weight", type=float, default=cfg.reward.diversity_weight)
    parser.add_argument("--eval_every_steps", type=int, default=cfg.eval_every_steps)
    parser.add_argument("--val_tasks_per_group", type=int, default=cfg.val_tasks_per_group)
    parser.add_argument("--val_max_tasks", type=int, default=cfg.val_max_tasks)
    parser.add_argument("--val_seed", type=int, default=cfg.val_seed)
    parser.add_argument("--val_gen_batch_size", type=int, default=cfg.val_gen_batch_size)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--bf16", action="store_true", default=cfg.bf16)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument(
        "--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false"
    )
    args = parser.parse_args(argv)
    if args.no_sft_adapter:
        args.sft_adapter_dir = None
    return args


if __name__ == "__main__":
    train_grpo(parse_args())
