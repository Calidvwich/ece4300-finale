"""
3D Voxel Reconstruction Diversity-Aware GRPO-LoRA 全局配置定义。

所有路径均基于本文件位置相对解析，兼容本地 Windows（RTX 4070）与远程 Linux（RTX 5090 / 4090）。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


# 当前模块根目录：HypoSpace-release/3d/grpo_lora/
GRPO_LORA_DIR: Path = Path(__file__).resolve().parent
# 3D 任务根目录：HypoSpace-release/3d/
TASK_3D_DIR: Path = GRPO_LORA_DIR.parent


@dataclass
class DataConfig:
    """3D 数据集与零 D4 同构泄露切分配置。"""

    id_dataset_path: Path = TASK_3D_DIR / "datasets" / "3d_g3_h3_b1to3.json"
    ood_raw_dataset_path: Path = TASK_3D_DIR / "datasets" / "3d_g3_h3_b4.json"
    ood_dataset_path: Path = TASK_3D_DIR / "datasets" / "3d_g3_h3_b4_ood.json"
    test_dataset_path: Path = TASK_3D_DIR / "datasets" / "3d_g3_h3_test.json"
    splits_output_path: Path = GRPO_LORA_DIR / "splits" / "3d_splits.json"
    seed: int = 33550336


@dataclass
class SamplingConfig:
    """
    全局唯一采样默认参数（GRPO rollout、验证集选优、最终评测共用）。

    max_new_tokens=128 足以覆盖 3 层 3x3 体素输出格式。
    """

    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 128


@dataclass
class RewardConfig:
    """严格层级 + 排列不变的集合级 3D 多样性奖励配置。"""

    mode: str = "diversity"
    parse_fail_reward: float = -1.0
    invalid_base_reward: float = -0.7
    invalid_match_bonus: float = 0.3
    valid_floor_reward: float = 0.1
    valid_unique_bonus: float = 0.9
    diversity_weight: float = 0.5


@dataclass
class LoRAConfig:
    """LoRA 适配器结构超参数（与 Causal 保持一致）。"""

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
    """Stage 1：轮换覆盖去偏 SFT-LoRA Warm-up 超参数。"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    splits_path: Path = GRPO_LORA_DIR / "splits" / "3d_splits.json"
    output_dir: Path = GRPO_LORA_DIR / "outputs" / "sft_lora"
    epochs: int = 2
    max_steps: Optional[int] = None
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_seq_len: int = 768
    max_samples_per_task: int = 8
    seed: int = 42
    bf16: bool = True


@dataclass
class GRPOConfig:
    """Stage 2：3D Diversity-Aware GRPO-LoRA 训练超参数。"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_adapter_dir: Optional[Path] = GRPO_LORA_DIR / "outputs" / "sft_lora"
    splits_path: Path = GRPO_LORA_DIR / "splits" / "3d_splits.json"
    output_dir: Path = GRPO_LORA_DIR / "outputs" / "grpo_lora"
    epochs: int = 5
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
    eval_every_steps: int = 15
    val_tasks_per_group: int = 6
    val_max_tasks: Optional[int] = None
    val_seed: int = 33550336
    val_gen_batch_size: int = 8
    reward_mode: str = "diversity"
    seed: int = 42
    bf16: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class EvalConfig:
    """统一 Fixed-Prompt 3D 评测超参数。"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_dir: Optional[Path] = None
    splits_path: Path = GRPO_LORA_DIR / "splits" / "3d_splits.json"
    split_name: str = "test"
    dataset_path: Optional[Path] = None
    compare_baseline_path: Optional[Path] = None
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
