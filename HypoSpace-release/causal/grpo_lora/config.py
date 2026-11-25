"""
Causal Diversity-Aware GRPO-LoRA 全局配置定义。

所有路径均基于本文件位置相对解析，避免硬编码绝对路径，
以便在本地 Windows（RTX 4070）与远程 Linux 服务器（RTX 5090 / 4090）之间无缝迁移。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


# 当前模块根目录：HypoSpace-release/causal/grpo_lora/
GRPO_LORA_DIR: Path = Path(__file__).resolve().parent
# Causal 任务根目录：HypoSpace-release/causal/
CAUSAL_DIR: Path = GRPO_LORA_DIR.parent


@dataclass
class DataConfig:
    """数据集与无泄露切分配置。"""

    n3_dataset_path: Path = CAUSAL_DIR / "datasets" / "n3_all_observations.json"
    n4_dataset_path: Path = CAUSAL_DIR / "datasets" / "n4_obs3_all.json"
    n5_ood_dataset_path: Path = CAUSAL_DIR / "datasets" / "n5_obs5_hybrid80.json"
    deepseek_baseline_path: Path = (
        CAUSAL_DIR / "benchmark_obs_count" / "full_repo_causal_n4_obs3_deepseek.json"
    )
    splits_output_path: Path = GRPO_LORA_DIR / "splits" / "causal_splits.json"
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    seed: int = 33550336


@dataclass
class RewardConfig:
    """
    集合级多样性奖励（Set-Level Diversity Reward）配置。

    针对同一个观测集 O 采样的一组 G 个输出，按以下分层规则赋分：
    1. 无法解析或非 DAG（含环 / 越界节点 / 超出 max_edges）：parse_fail_reward
    2. 合法 DAG 但不完全满足所有干预观测：
       invalid_base_reward + invalid_match_bonus * obs_match_ratio
    3. 完全满足所有干预观测（Valid）：
       - 组内首次出现（Unique Valid）：
         valid_unique_base + diversity_weight * mean_edge_jaccard_distance
       - 组内重复出现（Duplicate Valid）：
         duplicate_valid_reward（直接惩罚模式崩塌）
    """

    parse_fail_reward: float = -1.0
    invalid_base_reward: float = -0.5
    invalid_match_bonus: float = 0.3
    valid_unique_base: float = 1.0
    diversity_weight: float = 0.5
    duplicate_valid_reward: float = -0.3


@dataclass
class LoRAConfig:
    """LoRA 适配器结构超参数。"""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class SFTConfig:
    """Stage 1：轻量 SFT-LoRA Warm-up 超参数。"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    splits_path: Path = GRPO_LORA_DIR / "splits" / "causal_splits.json"
    output_dir: Path = GRPO_LORA_DIR / "outputs" / "sft_lora"
    epochs: int = 2
    max_steps: Optional[int] = None
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_seq_len: int = 512
    max_samples_per_task: int = 8
    shuffle_edges: bool = True
    seed: int = 42
    bf16: bool = True


@dataclass
class GRPOConfig:
    """Stage 2：Diversity-Aware GRPO-LoRA 训练超参数。"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_adapter_dir: Optional[Path] = GRPO_LORA_DIR / "outputs" / "sft_lora"
    splits_path: Path = GRPO_LORA_DIR / "splits" / "causal_splits.json"
    output_dir: Path = GRPO_LORA_DIR / "outputs" / "grpo_lora"
    epochs: int = 2
    max_steps: Optional[int] = None
    min_gt_for_grpo: int = 2
    group_size: int = 8
    ppo_epochs: int = 1
    clip_eps: float = 0.2
    kl_coef: float = 0.04
    lr: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 64
    eval_every_steps: int = 50
    val_max_tasks: int = 20
    seed: int = 42
    bf16: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class EvalConfig:
    """统一 Fixed-Prompt 评测超参数。"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_dir: Optional[Path] = None
    splits_path: Path = GRPO_LORA_DIR / "splits" / "causal_splits.json"
    split_name: str = "test"
    dataset_path: Optional[Path] = None
    compare_baseline_path: Optional[Path] = (
        CAUSAL_DIR / "benchmark_obs_count" / "full_repo_causal_n4_obs3_deepseek.json"
    )
    output_path: Path = GRPO_LORA_DIR / "results" / "eval_result.json"
    query_multiplier: float = 1.0
    gen_batch_size: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 64
    max_tasks: Optional[int] = None
    seed: int = 33550336
    bf16: bool = True
