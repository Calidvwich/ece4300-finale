"""
Causal Diversity-Aware GRPO-LoRA 全局配置定义（v2.1 测量口径统一版）。

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
    """数据集与零同构泄露切分配置。"""

    n3_dataset_path: Path = CAUSAL_DIR / "datasets" / "n3_all_observations.json"
    n4_dataset_path: Path = CAUSAL_DIR / "datasets" / "n4_obs3_all.json"
    n5_ood_dataset_path: Path = CAUSAL_DIR / "datasets" / "n5_obs5_hybrid80.json"
    deepseek_baseline_path: Path = (
        CAUSAL_DIR / "benchmark_obs_count" / "full_repo_causal_n4_obs3_deepseek.json"
    )
    splits_output_path: Path = GRPO_LORA_DIR / "splits" / "causal_splits.json"
    seed: int = 33550336


@dataclass
class SamplingConfig:
    """
    全局唯一采样默认参数（GRPO rollout、验证集选优、最终评测共用）。

    消除训练 rollout 与验证/评测解码分布不一致导致的测量错配。
    """

    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 64


@dataclass
class RewardConfig:
    """
    严格层级 + 排列不变的集合级多样性奖励配置（v2.0）。

    支持两种模式（用于严格消融对比）：
    - `diversity`: 本文主方法（逆频次覆盖衰减 + 边集合 Jaccard 新颖度）
    - `validity_only`: 纯合法性 RL 对照基线（只要合法即给固定 +1.0，无多样性奖励/去重衰减）
    """

    mode: str = "diversity"
    parse_fail_reward: float = -1.0
    invalid_base_reward: float = -0.7
    invalid_match_bonus: float = 0.3
    valid_floor_reward: float = 0.1
    valid_unique_bonus: float = 0.9
    diversity_weight: float = 0.5


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
    """Stage 1：轻量去偏 SFT-LoRA Warm-up 超参数。"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    splits_path: Path = GRPO_LORA_DIR / "splits" / "causal_splits.json"
    output_dir: Path = GRPO_LORA_DIR / "outputs" / "sft_lora"
    epochs: int = 2
    max_steps: Optional[int] = None
    batch_size: int = 1
    grad_accum: int = 16
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
    """Stage 2：Diversity-Aware GRPO-LoRA 训练超参数（v2.1）。"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_adapter_dir: Optional[Path] = GRPO_LORA_DIR / "outputs" / "sft_lora"
    splits_path: Path = GRPO_LORA_DIR / "splits" / "causal_splits.json"
    output_dir: Path = GRPO_LORA_DIR / "outputs" / "grpo_lora"
    epochs: int = 2
    max_steps: Optional[int] = None
    min_gt_for_grpo: int = 2
    group_size: int = 8
    micro_batch_size: int = 2
    tasks_per_step: int = 4
    ppo_epochs: int = 1
    clip_eps: float = 0.2
    kl_coef: float = 0.06
    lr: float = 1e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    temperature: float = SamplingConfig.temperature
    top_p: float = SamplingConfig.top_p
    max_new_tokens: int = SamplingConfig.max_new_tokens
    eval_every_steps: int = 20
    val_tasks_per_group: int = 12
    val_max_tasks: Optional[int] = None
    val_seed: int = 33550336
    val_gen_batch_size: int = 8
    reward_mode: str = "diversity"
    seed: int = 42
    bf16: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class EvalConfig:
    """统一 Fixed-Prompt 评测超参数（采样参数严格与 GRPOConfig 一致）。"""

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
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    temperature: float = SamplingConfig.temperature
    top_p: float = SamplingConfig.top_p
    max_new_tokens: int = SamplingConfig.max_new_tokens
    max_tasks: Optional[int] = None
    seed: int = 33550336
    bf16: bool = True
