"""
3D 体素重建精确解析器、官方一致物理/观测验证器与集合级多样性 GRPO 奖励模块。

官方代码复用与来源说明（未 import `3d/modules/llm_interface.py` 以免依赖 `openai`）：
- `Structure3D`: 复制自 `HypoSpace-release/3d/run_3d_benchmark.py:21-118`
- `parse_top_view`: 复制自 `Benchmark3D._parse_observation` (`run_3d_benchmark.py:165-182`)
- `validate_structure_matches_observations`: 复制自 `run_3d_benchmark.py:308-358`
- `parse_llm_response`: 复制自 `run_3d_benchmark.py:395-510`

奖励层级差异说明：
- 评测阶段 `Validity` 严格使用官方 `validate_structure_matches_observations`（不限最大高度），
  保证与 `DeepSeek` 官方 runner 100% 可比；
- 训练奖励在进入合法判定前检查 `norm.height <= max_height`、网格尺寸一致且第 1 层非全空，
  越界样本一律判为 `-1.0`。
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import RewardConfig  # noqa: E402


class Structure3D:
    """Represents a 3D structure (copied from `3d/run_3d_benchmark.py:21-118`)."""

    def __init__(self, layers: List[List[List[int]]]):
        if not layers:
            self.layers: List[np.ndarray] = []
            self.height = 0
            self.shape = (0, 0)
            return

        max_rows = 0
        max_cols = 0
        for layer in layers:
            if layer:
                max_rows = max(max_rows, len(layer))
                for row in layer:
                    if row:
                        max_cols = max(max_cols, len(row))

        self.layers = []
        for layer in layers:
            padded_layer = np.zeros((max_rows, max_cols), dtype=int)
            for i, row in enumerate(layer[:max_rows]):
                for j, val in enumerate(row[:max_cols]):
                    padded_layer[i, j] = val
            self.layers.append(padded_layer)

        self.height = len(self.layers)
        self.shape = (max_rows, max_cols) if self.layers else (0, 0)

    def to_string(self) -> str:
        """Convert to string representation for LLM."""
        result = []
        for i, layer in enumerate(self.layers):
            result.append(f"Layer {i+1}:")
            for row in layer:
                result.append(" ".join(str(cell) for cell in row))
        return "\n".join(result)

    def get_top_view(self) -> np.ndarray:
        """Get top view (OR of all layers)."""
        if not self.layers:
            return np.zeros((0, 0), dtype=int)

        top = np.zeros_like(self.layers[0], dtype=int)
        for layer in self.layers:
            if layer.shape == top.shape:
                top |= (layer.astype(bool)).astype(int)
            else:
                min_rows = min(layer.shape[0], top.shape[0])
                min_cols = min(layer.shape[1], top.shape[1])
                top[:min_rows, :min_cols] |= (
                    layer[:min_rows, :min_cols].astype(bool)
                ).astype(int)
        return top

    def normalize(self) -> "Structure3D":
        """Remove trailing all-zero layers from the top."""
        if not self.layers:
            return Structure3D([])

        last_non_zero = -1
        for i in range(len(self.layers) - 1, -1, -1):
            if np.any(self.layers[i] != 0):
                last_non_zero = i
                break

        if last_non_zero == -1:
            return Structure3D([self.layers[0] * 0])

        return Structure3D([layer.tolist() for layer in self.layers[: last_non_zero + 1]])

    def get_hash(self) -> str:
        """Get hash for comparison - normalized to ignore trailing zero layers."""
        normalized = self.normalize()
        if not normalized.layers:
            return "empty000"

        arr = np.stack(normalized.layers, axis=0).astype(np.uint8)
        h = hashlib.md5()
        h.update(arr.tobytes())
        h.update(np.array(arr.shape, dtype=np.int64).tobytes())
        return h.hexdigest()[:8]


@dataclass
class GroupRewardResult:
    """单个 3D 观测题上采样一组 G 个响应后的完整评估与奖励统计。"""

    rewards: List[float]
    parsed_structures: List[Optional[Structure3D]]
    parse_flags: List[bool]
    valid_flags: List[bool]
    unique_valid_flags: List[bool]
    obs_match_ratios: List[float]
    valid_rate: float
    unique_valid_rate: float
    parse_rate: float


def parse_top_view(obs_data: Any) -> np.ndarray:
    """Parse observation data (copied from `run_3d_benchmark.py:165-182`)."""
    if isinstance(obs_data, str):
        length = len(obs_data)
        grid_size = int(length**0.5)
        observation = []
        for i in range(grid_size):
            row = [int(obs_data[i * grid_size + j]) for j in range(grid_size)]
            observation.append(row)
        return np.array(observation, dtype=int)
    if isinstance(obs_data, list):
        return np.array(obs_data, dtype=int)
    return np.array(obs_data.get("observation", []), dtype=int)


def decode_compact_layers(layers_compact: Sequence[Union[str, Sequence[Sequence[int]]]]) -> Structure3D:
    """Decode compact dataset GT layer strings (or nested lists) into normalized `Structure3D`."""
    if not layers_compact:
        return Structure3D([])
    if isinstance(layers_compact[0], str):
        layers: List[List[List[int]]] = []
        grid_size = int(len(layers_compact[0]) ** 0.5)
        for layer_str in layers_compact:
            row_list = [
                [int(layer_str[r * grid_size + c]) for c in range(grid_size)]
                for r in range(grid_size)
            ]
            layers.append(row_list)
        return Structure3D(layers).normalize()
    return Structure3D(list(layers_compact)).normalize()


def check_physical_support(structure: Structure3D) -> bool:
    """Check whether every block above Layer 1 is supported from below."""
    for z in range(1, len(structure.layers)):
        current = structure.layers[z]
        below = structure.layers[z - 1]
        for r in range(current.shape[0]):
            for c in range(current.shape[1]):
                if current[r, c] == 1 and below[r, c] != 1:
                    return False
    return True


def validate_structure_matches_observations(structure: Structure3D, observations: Any) -> bool:
    """Official Validity predicate copied from `run_3d_benchmark.py:308-358`."""
    if isinstance(observations, str):
        observed = parse_top_view(observations)
        generated = structure.get_top_view()
        if generated.shape != observed.shape or not np.array_equal(generated, observed):
            return False
    elif isinstance(observations, list):
        for obs in observations:
            if isinstance(obs, dict):
                view_type = obs.get("view_type", "top")
                observed = np.array(obs["observation"], dtype=int)
            else:
                observed = parse_top_view(obs)
                view_type = "top"
            if view_type != "top":
                return False
            generated = structure.get_top_view()
            if generated.shape != observed.shape or not np.array_equal(generated, observed):
                return False
    else:
        observed = parse_top_view(observations)
        generated = structure.get_top_view()
        if generated.shape != observed.shape or not np.array_equal(generated, observed):
            return False

    return check_physical_support(structure)


def parse_llm_response(response: str) -> Optional[Structure3D]:
    """Official 3D parser copied from `run_3d_benchmark.py:395-510`."""
    if not isinstance(response, str):
        return None

    try:
        lines = response.strip().split("\n")
        structure_start = -1
        for i, line in enumerate(lines):
            if "Structure:" in line:
                structure_start = i
                break

        if structure_start == -1:
            for i, line in enumerate(lines):
                if line.strip().startswith("Layer 1:"):
                    structure_start = i
                    break

        if structure_start == -1:
            return None

        layers: List[List[List[int]]] = []
        current_layer: List[List[int]] = []
        in_layer = False
        layer_num = 0

        for line in lines[structure_start:]:
            line = line.strip()
            if not line or line.startswith("Note:") or line.startswith("Reasoning:") or line.startswith("Explanation:"):
                continue

            if line.startswith("Layer "):
                match = re.match(r"Layer\s+(\d+)", line)
                if match:
                    new_layer_num = int(match.group(1))
                    if new_layer_num == layer_num + 1 or (new_layer_num == 1 and layer_num == 0):
                        if current_layer:
                            layers.append(current_layer)
                            current_layer = []
                        in_layer = True
                        layer_num = new_layer_num
                    else:
                        break
            elif in_layer and line:
                reasoning_indicators = [
                    "therefore", "because", "since", "thus", "hence",
                    "reasoning:", "explanation:", "note:", "this means",
                    "this structure", "the structure", "i chose", "i created",
                ]
                if any(indicator in line.lower() for indicator in reasoning_indicators):
                    break

                try:
                    if len(line) > 100:
                        continue

                    if "," in line:
                        parts = [x.strip() for x in line.split(",")]
                    elif "|" in line:
                        parts = [x.strip() for x in line.split("|")]
                    elif any(c in line for c in ["[", "]"]):
                        stripped = line.strip("[]")
                        if "," in stripped:
                            parts = [x.strip() for x in stripped.split(",")]
                        else:
                            parts = stripped.split()
                    else:
                        parts = line.split()

                    row: List[int] = []
                    bad_val = False
                    for x in parts:
                        if x.strip():
                            val = int(x)
                            if val not in (0, 1):
                                bad_val = True
                                break
                            row.append(val)

                    if bad_val:
                        continue
                    if row:
                        current_layer.append(row)
                except Exception:
                    pass

        if current_layer:
            layers.append(current_layer)

        if layers:
            return Structure3D(layers)
    except Exception:
        return None

    return None


def structure_to_voxels(structure: Structure3D) -> Set[Tuple[int, int, int]]:
    """Convert normalized `Structure3D` to set of occupied `(z, r, c)` voxel coordinates."""
    norm = structure.normalize()
    voxels: Set[Tuple[int, int, int]] = set()
    for z, layer in enumerate(norm.layers):
        rows, cols = np.where(layer == 1)
        for r, c in zip(rows.tolist(), cols.tolist()):
            voxels.add((z, r, c))
    return voxels


def voxel_jaccard_distance(s1: Structure3D, s2: Structure3D) -> float:
    """Compute Jaccard distance `1 - |V1 & V2| / |V1 | V2|` over occupied 3D voxels."""
    v1 = structure_to_voxels(s1)
    v2 = structure_to_voxels(s2)
    union_sz = len(v1 | v2)
    if union_sz == 0:
        return 0.0
    return 1.0 - (len(v1 & v2) / float(union_sz))


def compute_group_diversity_rewards(
    responses: Sequence[str],
    observation: Any,
    grid_size: int = 3,
    max_height: int = 3,
    cfg: Optional[RewardConfig] = None,
) -> GroupRewardResult:
    """
    计算 3D 任务单组采样响应的严格层级 + 排列不变集合级多样性奖励：

    - 解析失败 / 网格尺寸 != `(grid_size, grid_size)` / `norm.height > max_height` / Layer 1 全空：`-1.0`
    - 在假设空间内但违反俯视图或物理支撑：`[-0.7, -0.4]`（部分匹配率 = 俯视图逐格一致率，违反支撑再 * 0.5）
    - 合法：`diversity` 为 `0.1 + (0.9 + 0.5 * d_bar) / count(hash)`，`validity_only` 固定 `+1.0`
    """
    reward_cfg = cfg if cfg is not None else RewardConfig()
    if reward_cfg.mode not in ("diversity", "validity_only"):
        raise ValueError(f"Unsupported reward mode: {reward_cfg.mode}")

    obs_top = parse_top_view(observation)
    n_resp = len(responses)
    parsed_structures: List[Optional[Structure3D]] = []
    parse_flags: List[bool] = []
    valid_flags: List[bool] = []
    obs_match_ratios: List[float] = []
    unique_valid_flags: List[bool] = [False] * n_resp
    row_hashes: List[Optional[str]] = [None] * n_resp
    in_space_flags: List[bool] = [False] * n_resp

    valid_counts: Counter = Counter()
    unique_valid_structs: Dict[str, Structure3D] = {}

    for idx, text in enumerate(responses):
        raw_struct = parse_llm_response(text)
        if raw_struct is None:
            parsed_structures.append(None)
            parse_flags.append(False)
            valid_flags.append(False)
            obs_match_ratios.append(0.0)
            continue

        norm = raw_struct.normalize()
        parsed_structures.append(norm)
        parse_flags.append(True)

        # 检查假设空间边界：网格尺寸一致、高度不超过 max_height、Layer 1 非全空
        in_space = (
            norm.shape == (grid_size, grid_size)
            and 1 <= norm.height <= max_height
            and bool(np.any(norm.layers[0] != 0))
        )
        in_space_flags[idx] = in_space
        if not in_space:
            valid_flags.append(False)
            obs_match_ratios.append(0.0)
            continue

        top_view = norm.get_top_view()
        cell_match = float(np.mean(top_view == obs_top))
        support_ok = check_physical_support(norm)
        top_ok = bool(np.array_equal(top_view, obs_top))
        is_valid = top_ok and support_ok

        effective_match = cell_match if support_ok else (0.5 * cell_match)
        valid_flags.append(is_valid)
        obs_match_ratios.append(effective_match)

        if is_valid:
            s_hash = norm.get_hash()
            row_hashes[idx] = s_hash
            valid_counts[s_hash] += 1
            if s_hash not in unique_valid_structs:
                unique_valid_structs[s_hash] = norm
                unique_valid_flags[idx] = True

    diversity_by_hash: Dict[str, float] = {}
    unique_keys = list(unique_valid_structs.keys())
    if len(unique_keys) <= 1:
        for k in unique_keys:
            diversity_by_hash[k] = 0.0
    else:
        for k_i in unique_keys:
            s_i = unique_valid_structs[k_i]
            dists = [
                voxel_jaccard_distance(s_i, unique_valid_structs[k_j])
                for k_j in unique_keys
                if k_j != k_i
            ]
            diversity_by_hash[k_i] = float(np.mean(dists))

    rewards: List[float] = []
    for idx in range(n_resp):
        if not parse_flags[idx] or not in_space_flags[idx]:
            rewards.append(reward_cfg.parse_fail_reward)
        elif not valid_flags[idx]:
            rewards.append(
                reward_cfg.invalid_base_reward
                + reward_cfg.invalid_match_bonus * obs_match_ratios[idx]
            )
        else:
            if reward_cfg.mode == "validity_only":
                rewards.append(
                    reward_cfg.valid_floor_reward + reward_cfg.valid_unique_bonus
                )
            else:
                s_hash = row_hashes[idx]
                assert s_hash is not None
                count = valid_counts[s_hash]
                mean_dist = diversity_by_hash[s_hash]
                rewards.append(
                    reward_cfg.valid_floor_reward
                    + (
                        reward_cfg.valid_unique_bonus
                        + reward_cfg.diversity_weight * mean_dist
                    )
                    / float(count)
                )

    return GroupRewardResult(
        rewards=rewards,
        parsed_structures=parsed_structures,
        parse_flags=parse_flags,
        valid_flags=valid_flags,
        unique_valid_flags=unique_valid_flags,
        obs_match_ratios=obs_match_ratios,
        valid_rate=(sum(valid_flags) / n_resp) if n_resp else 0.0,
        unique_valid_rate=(len(unique_valid_structs) / n_resp) if n_resp else 0.0,
        parse_rate=(sum(parse_flags) / n_resp) if n_resp else 0.0,
    )


def compute_grpo_advantages(rewards: Sequence[float], eps: float = 1e-6) -> List[float]:
    """GRPO 组内 z-score 相对优势；方差极小时返回全 0 防止梯度爆炸。"""
    if not rewards:
        return []
    arr = np.asarray(rewards, dtype=np.float64)
    std = float(np.std(arr))
    if std < eps:
        return [0.0] * len(rewards)
    mean = float(np.mean(arr))
    return [float((x - mean) / (std + eps)) for x in arr]
