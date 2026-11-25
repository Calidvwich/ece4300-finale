"""
Stage 2：Diversity-Aware GRPO-LoRA 训练主脚本。

算法核心（纯 PyTorch + transformers + peft 实现，无 trl/vllm 强依赖）：
1. 针对同一欠定观测集 O，使用当前策略采样一组 G 个因果图响应 {h_1, ..., h_G}；
2. 调用 `compute_group_diversity_rewards` 计算集合级多样性奖励：
   - 组内唯一有效解：高基础分 + 平均边 Jaccard 距离新颖度奖励；
   - 组内重复有效解：负惩罚，直接打破模式崩塌；
   - 非法 DAG / 观测不匹配：负惩罚；
3. 计算组内标准化优势 A_i = (r_i - mean(r)) / (std(r) + eps)；
4. 在同一个 PeftModel 上挂载双适配器（可训练 `"policy"` 与冻结 `"ref"`），
   零额外基座显存开销计算无偏 KL 惩罚，优化 Clipped GRPO 代理目标：
   L = -E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)] + beta * KL(pi || pi_ref)
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig as PeftLoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import GRPOConfig, LoRAConfig, RewardConfig  # noqa: E402
from data_utils import build_fixed_prompt, format_chat_prompt  # noqa: E402
from verifier_reward import (  # noqa: E402
    compute_grpo_advantages,
    compute_group_diversity_rewards,
    parse_causal_graph,
    evaluate_observation_match,
)


def set_seed(seed: int) -> None:
    """固定随机种子以确保训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_completion_logprobs(
    model: PeftModel,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算批量样本在补全部分每个 token 上的对数概率 `log pi(y_t | x, y_<t)`。

    Args:
        full_ids: 形状 (G, L) 的完整序列（prompt + completion）。
        attention_mask: 形状 (G, L) 的注意力掩码。
        completion_mask: 形状 (G, L-1) 的布尔/浮点掩码，仅补全 token 对应的预测位置为 1。

    Returns:
        形状 (G, L-1) 的对数概率张量（非补全位置已清零）。
    """
    outputs = model(input_ids=full_ids, attention_mask=attention_mask)
    # logits[:, :-1, :] 预测 full_ids[:, 1:]；逐样本逐位置 log_softmax 前保持显存可控
    shift_logits = outputs.logits[:, :-1, :].float()
    shift_labels = full_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = torch.gather(
        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logprobs * completion_mask


def compute_completion_logprobs_chunked(
    model: PeftModel,
    full_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    micro_batch_size: int,
) -> torch.Tensor:
    """
    在无梯度模式下按 `micro_batch_size` 分块计算补全对数概率，避免整组同时展开大词表 logits 导致 OOM。
    """
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
    """
    使用当前 `"policy"` 适配器针对同一个 prompt 批量采样 `group_size` 条响应。

    Returns:
        full_ids (G, L), attention_mask (G, L), completion_mask (G, L-1), decoded_texts
    """
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

    # 构造 attention_mask 与 completion_mask：遇到首个 eos_id 之后均视为 padding
    attn_mask = torch.ones((group_size, seq_len), dtype=torch.long, device=device)
    comp_mask = torch.zeros((group_size, seq_len - 1), dtype=torch.float32, device=device)
    decoded_texts: List[str] = []

    for row in range(group_size):
        comp_tokens = generated[row, prompt_len:]
        valid_len = len(comp_tokens)
        for pos, tok in enumerate(comp_tokens.tolist()):
            if (eos_id is not None and tok == eos_id) or (tok == pad_id):
                # 保留首个 eos_id 本身参与监督，其后全部 mask 掉
                valid_len = pos + (1 if (eos_id is not None and tok == eos_id) else 0)
                break

        end_idx = prompt_len + valid_len
        if end_idx < seq_len:
            attn_mask[row, end_idx:] = 0

        # full_ids[t] 在 shift 后对应位置 t-1
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
    max_tasks: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, float]:
    """在留出验证集上快速评估 Fixed-Prompt 协议下的 Valid / Uniqueness / Recovery。"""
    model.eval()
    model.config.use_cache = True
    model.set_adapter("policy")
    subset = val_tasks[:max_tasks]

    valid_rates: List[float] = []
    unique_rates: List[float] = []
    recovery_rates: List[float] = []

    for task in subset:
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
        n_gt = max(1, int(task.get("n_compatible_graphs", 1)))
        gt_hashes = {
            frozenset(tuple(e) for e in g.get("edges", []))
            for g in task.get("ground_truth_graphs", [])
        }

        batch_prompt = prompt_ids.repeat(n_gt, 1)
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

        p_len = prompt_ids.shape[1]
        valid_count = 0
        unique_edge_sets = set()
        for row in range(n_gt):
            text = tokenizer.decode(generated[row, p_len:], skip_special_tokens=True).strip()
            g = parse_causal_graph(text, nodes=task["nodes"], max_edges=task.get("max_edges"))
            if g is not None:
                is_valid, _ = evaluate_observation_match(g, task["observations"])
                if is_valid:
                    valid_count += 1
                    unique_edge_sets.add(frozenset(g.edges))

        valid_rates.append(valid_count / n_gt)
        unique_rates.append(len(unique_edge_sets) / n_gt)
        recovered = len(unique_edge_sets & gt_hashes)
        recovery_rates.append(recovered / len(gt_hashes))

    return {
        "val_tasks": float(len(subset)),
        "val_valid_rate": float(np.mean(valid_rates)),
        "val_uniqueness_rate": float(np.mean(unique_rates)),
        "val_recovery_rate": float(np.mean(recovery_rates)),
    }


def load_policy_and_ref_model(
    model_name: str,
    sft_adapter_dir: Optional[Path],
    dtype: torch.dtype,
    device: torch.device,
    gradient_checkpointing: bool = True,
) -> Tuple[PeftModel, Optional[str]]:
    """
    加载基础模型与双 LoRA 适配器：
    - 若提供 `sft_adapter_dir`：加载 `"policy"`（可训练）与 `"ref"`（冻结），零显存复用基座；
    - 若未提供：新建随机初始化 `"policy"`，并通过 `disable_adapter()` 作为 `"ref"`。
    """
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

    if sft_adapter_dir is not None and Path(sft_adapter_dir).exists():
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
            lora_dropout=lora_defaults.lora_dropout,
            target_modules=list(lora_defaults.target_modules),
            bias="none",
        )
        model = get_peft_model(base_model, peft_cfg, adapter_name="policy")
        ref_adapter_name = None

    model.set_adapter("policy")
    model.to(device)
    return model, ref_adapter_name


def train_grpo(args: argparse.Namespace) -> None:
    """执行 Stage 2 Diversity-Aware GRPO-LoRA 主训练流程。"""
    set_seed(args.seed)
    splits_payload = json.loads(Path(args.splits_path).read_text(encoding="utf-8"))

    # 默认聚焦于存在多个合法解的欠定任务（|H_O| >= min_gt_for_grpo）
    all_train = splits_payload["splits"]["train"]
    train_tasks = [
        t for t in all_train if int(t.get("n_compatible_graphs", 0)) >= args.min_gt_for_grpo
    ]
    val_tasks = splits_payload["splits"]["val"]
    if not train_tasks:
        raise RuntimeError("No training tasks satisfy min_gt_for_grpo filter.")

    print(
        f"GRPO train tasks: {len(train_tasks)}/{len(all_train)} "
        f"(min_gt={args.min_gt_for_grpo}), val tasks: {len(val_tasks)}"
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

    reward_cfg = RewardConfig(
        diversity_weight=args.diversity_weight,
        duplicate_valid_reward=args.duplicate_penalty,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best_policy"
    metrics_file = output_dir / "grpo_metrics.jsonl"
    metrics_file.write_text("", encoding="utf-8")

    rng = random.Random(args.seed)
    total_steps = (
        args.max_steps
        if args.max_steps is not None
        else len(train_tasks) * args.epochs
    )
    global_step = 0
    best_val_recovery = -1.0

    for epoch in range(args.epochs):
        epoch_tasks = list(train_tasks)
        rng.shuffle(epoch_tasks)

        for task in epoch_tasks:
            global_step += 1
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
            )

            # 1) 在线组采样
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

            # 2) 精确验证 + 集合级多样性奖励 + 组内优势
            reward_res = compute_group_diversity_rewards(
                responses=texts,
                nodes=task["nodes"],
                observations=task["observations"],
                max_edges=task.get("max_edges"),
                cfg=reward_cfg,
            )
            advantages_list = compute_grpo_advantages(reward_res.rewards)
            advantages = torch.tensor(advantages_list, dtype=torch.float32, device=device)

            # 3) 按 micro_batch_size 分块计算参考策略 log_ref 与采样旧策略 log_old（均无需梯度）
            model.config.use_cache = False
            with torch.no_grad():
                old_logprobs = compute_completion_logprobs_chunked(
                    model, full_ids, attn_mask, comp_mask, args.micro_batch_size
                )
                if ref_adapter_name is not None:
                    model.set_adapter(ref_adapter_name)
                    ref_logprobs = compute_completion_logprobs_chunked(
                        model, full_ids, attn_mask, comp_mask, args.micro_batch_size
                    )
                else:
                    with model.disable_adapter():
                        ref_logprobs = compute_completion_logprobs_chunked(
                            model, full_ids, attn_mask, comp_mask, args.micro_batch_size
                        )

            # 4) Clipped GRPO 策略更新（按 micro_batch_size 分块反向传播，数学上与整组完全等价）
            model.set_adapter("policy")
            model.train()
            model.config.use_cache = False
            token_counts = comp_mask.sum(dim=-1).clamp(min=1.0)
            total_group = full_ids.shape[0]
            mb_step = max(1, args.micro_batch_size)

            step_loss_val = 0.0
            step_kl_val = 0.0
            for _ in range(args.ppo_epochs):
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_kl = 0.0

                for start in range(0, total_group, mb_step):
                    end = min(total_group, start + mb_step)
                    cur_lp = compute_completion_logprobs(
                        model,
                        full_ids[start:end],
                        attn_mask[start:end],
                        comp_mask[start:end],
                    )
                    ratio = torch.exp(cur_lp - old_logprobs[start:end])
                    adv_expanded = advantages[start:end].unsqueeze(-1)
                    surr1 = ratio * adv_expanded
                    surr2 = (
                        torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                        * adv_expanded
                    )
                    pg_loss_per_token = -torch.min(surr1, surr2)

                    # Schulman 无偏近似 KL(pi || pi_ref) >= 0
                    log_ratio_ref = ref_logprobs[start:end] - cur_lp
                    kl_per_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

                    mb_comp_mask = comp_mask[start:end]
                    mb_counts = token_counts[start:end]
                    loss_per_seq = (
                        (pg_loss_per_token + args.kl_coef * kl_per_token) * mb_comp_mask
                    ).sum(dim=-1) / mb_counts

                    chunk_loss = loss_per_seq.sum() / total_group
                    chunk_loss.backward()

                    accum_loss += float(chunk_loss.item())
                    chunk_kl = (
                        ((kl_per_token * mb_comp_mask).sum(dim=-1) / mb_counts).sum()
                        / total_group
                    )
                    accum_kl += float(chunk_kl.item())

                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()

                step_loss_val = accum_loss
                step_kl_val = accum_kl

            log_row: Dict[str, Any] = {
                "step": global_step,
                "epoch": epoch + 1,
                "task_id": task["observation_set_id"],
                "n_gt": int(task.get("n_compatible_graphs", 1)),
                "loss": step_loss_val,
                "kl": step_kl_val,
                "mean_reward": float(np.mean(reward_res.rewards)),
                "valid_rate": reward_res.valid_rate,
                "unique_valid_rate": reward_res.unique_valid_rate,
                "parse_rate": reward_res.parse_rate,
            }

            if global_step % args.eval_every_steps == 0 or global_step == total_steps:
                val_stats = evaluate_val_split(
                    model=model,
                    tokenizer=tokenizer,
                    val_tasks=val_tasks,
                    max_tasks=args.val_max_tasks,
                    temperature=0.7,
                    top_p=0.95,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
                log_row.update(val_stats)
                if val_stats["val_recovery_rate"] >= best_val_recovery:
                    best_val_recovery = val_stats["val_recovery_rate"]
                    model.save_pretrained(best_dir, selected_adapters=["policy"])
                    tokenizer.save_pretrained(best_dir)

            with metrics_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")

            if global_step % 5 == 0 or global_step == 1 or global_step == total_steps:
                print(
                    f"[GRPO] step={global_step}/{total_steps} "
                    f"task={task['observation_set_id']} "
                    f"reward={log_row['mean_reward']:.3f} "
                    f"valid={log_row['valid_rate']:.2f} "
                    f"uniq={log_row['unique_valid_rate']:.2f} "
                    f"kl={step_kl_val:.4f}"
                )

            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    final_dir = output_dir / "final_policy"
    model.set_adapter("policy")
    model.save_pretrained(final_dir, selected_adapters=["policy"])
    tokenizer.save_pretrained(final_dir)
    print(f"Stage 2 GRPO-LoRA saved to: {final_dir} (best val adapter: {best_dir})")


def parse_args() -> argparse.Namespace:
    cfg = GRPOConfig()
    parser = argparse.ArgumentParser(description="Stage 2: Diversity-Aware GRPO-LoRA training")
    parser.add_argument("--model_name", type=str, default=cfg.model_name)
    parser.add_argument("--sft_adapter_dir", type=Path, default=cfg.sft_adapter_dir)
    parser.add_argument("--no_sft_adapter", action="store_true")
    parser.add_argument("--splits_path", type=Path, default=cfg.splits_path)
    parser.add_argument("--output_dir", type=Path, default=cfg.output_dir)
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--max_steps", type=int, default=cfg.max_steps)
    parser.add_argument("--min_gt_for_grpo", type=int, default=cfg.min_gt_for_grpo)
    parser.add_argument("--group_size", type=int, default=cfg.group_size)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--ppo_epochs", type=int, default=cfg.ppo_epochs)
    parser.add_argument("--clip_eps", type=float, default=cfg.clip_eps)
    parser.add_argument("--kl_coef", type=float, default=cfg.kl_coef)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--max_grad_norm", type=float, default=cfg.max_grad_norm)
    parser.add_argument("--temperature", type=float, default=cfg.temperature)
    parser.add_argument("--top_p", type=float, default=cfg.top_p)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.max_new_tokens)
    parser.add_argument("--diversity_weight", type=float, default=cfg.reward.diversity_weight)
    parser.add_argument("--duplicate_penalty", type=float, default=cfg.reward.duplicate_valid_reward)
    parser.add_argument("--eval_every_steps", type=int, default=cfg.eval_every_steps)
    parser.add_argument("--val_max_tasks", type=int, default=cfg.val_max_tasks)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--bf16", action="store_true", default=cfg.bf16)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    args = parser.parse_args()
    if args.no_sft_adapter:
        args.sft_adapter_dir = None
    return args


if __name__ == "__main__":
    train_grpo(parse_args())
