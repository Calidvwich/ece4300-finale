"""
3D Voxel Reconstruction 数据加载、Fixed-Prompt 构造与零 D4 同构泄露切分模块。

核心设计：
- 对 3x3 网格使用二面体群 D4（4 次旋转 × 原图/水平翻转 = 8 个等价变换）计算规范同构键；
- `3d_g3_h3_b1to3.json` 共 129 题，归并后精确为 27 个 D4 等价类（b=1:3, b=2:8, b=3:16）；
- 按方块数分层，将完整等价类切入 train / val / test（保证每档方块数在 test/val 中均至少有 1 类），
  确保 train / val / test 与 4-block OOD 集在 D4 同构键层级上零重叠；
- `build_fixed_prompt` 与官方 `Benchmark3D.create_prompt(..., prior_structures=None)`
  （`HypoSpace-release/3d/run_3d_benchmark.py:184-306`）逐字符严格一致。
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import DataConfig  # noqa: E402
from verifier_reward import decode_compact_layers, parse_top_view  # noqa: E402


def load_3d_dataset(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """加载官方 3D 数据集 JSON，并统一补齐规范字段与 `observation_set_id`。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = dict(raw.get("metadata", {}))
    default_grid = int(metadata.get("grid_size", 3))
    default_height = int(metadata.get("max_height", 3))
    raw_sets = raw.get("observation_sets", [])

    tasks: List[Dict[str, Any]] = []
    source_stem = Path(path).stem
    for item in raw_sets:
        rec = dict(item)
        obs_raw = rec.get("observation", rec.get("observations"))
        top_grid = parse_top_view(obs_raw)
        grid_size = int(rec.get("grid_size", top_grid.shape[0] if top_grid.size else default_grid))
        max_height = int(rec.get("max_height", default_height))
        n_blocks = int(rec.get("n_blocks", int(np.sum(top_grid))))
        obs_id = str(rec.get("observation_id", rec.get("observation_set_id", "unknown")))
        gts = list(rec.get("ground_truth_structures", []))
        n_gt = int(rec.get("n_compatible_structures", len(gts)))

        rec["observation_id"] = obs_id
        rec["observation_set_id"] = obs_id
        rec["grid_size"] = grid_size
        rec["max_height"] = max_height
        rec["n_blocks"] = n_blocks
        rec["n_compatible_structures"] = n_gt
        rec["n_ground_truths"] = n_gt
        rec.setdefault("source_dataset", source_stem)
        tasks.append(rec)

    return metadata, tasks


def d4_transforms(grid: np.ndarray) -> List[np.ndarray]:
    """生成二维方阵在 D4 对称群下的 8 个等距变换（4 个旋转 × 是否水平翻转）。"""
    variants: List[np.ndarray] = []
    flipped = np.fliplr(grid)
    for k in range(4):
        variants.append(np.rot90(grid, k))
        variants.append(np.rot90(flipped, k))
    return variants


def canonical_isomorphism_key(task: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    计算 3D 俯视图在 D4 二面体群下的最小规范签名：
    `(grid_size, max_height, min over 8 D4 transforms of flattened tuple)`。
    """
    obs_raw = task.get("observation", task.get("observations"))
    top_grid = parse_top_view(obs_raw)
    grid_size = int(task.get("grid_size", top_grid.shape[0]))
    max_height = int(task.get("max_height", 3))
    best_flat = min(tuple(int(v) for v in mat.flatten()) for mat in d4_transforms(top_grid))
    return (grid_size, max_height, best_flat)


def build_fixed_prompt(
    observations: Any,
    grid_size: Optional[int] = None,
    max_height: int = 3,
) -> str:
    """
    构造与官方 `Benchmark3D.create_prompt(observations, prior_structures=None)`
    （`HypoSpace-release/3d/run_3d_benchmark.py:184-306`）逐字完全一致的 Fixed-Prompt。
    """
    if isinstance(observations, str):
        inferred_grid = int(len(observations) ** 0.5)
    elif isinstance(observations, list) and observations:
        if isinstance(observations[0], dict):
            obs_data = observations[0].get("observation", [])
            inferred_grid = len(obs_data) if obs_data else 3
        else:
            obs_data = parse_top_view(observations[0])
            inferred_grid = obs_data.shape[0] if hasattr(obs_data, "shape") else 3
    else:
        obs_data = parse_top_view(observations)
        inferred_grid = obs_data.shape[0] if hasattr(obs_data, "shape") else 3

    gs = int(grid_size) if grid_size is not None else inferred_grid

    prompt = f"""You are given observations of a 3D structure made of unit blocks on a {gs}x{gs} grid.
        Each observation shows a view of the structure from a specific angle.
        The maximum height of the structure is {max_height} layers.

        Observations (Top View - shows 1 if ANY layer has a block at that position):
        """
    if isinstance(observations, str):
        observation = parse_top_view(observations)
        prompt += f"\nTop view:\n"
        for row in observation:
            prompt += " ".join(str(cell) for cell in row) + "\n"
    elif isinstance(observations, list):
        for obs in observations:
            if isinstance(obs, dict):
                observation = np.array(obs["observation"], dtype=int)
            else:
                observation = parse_top_view(obs)
            prompt += f"\nTop view:\n"
            for row in observation:
                prompt += " ".join(str(cell) for cell in row) + "\n"
    else:
        observation = parse_top_view(observations)
        prompt += f"\nTop view:\n"
        for row in observation:
            prompt += " ".join(str(cell) for cell in row) + "\n"

    prompt += f"""
        
        Task: Infer the complete 3D structure that could produce these observations.

        Structure specifications:
        - Grid size: {gs}x{gs}
        - Maximum height: {max_height} layers
        - The structure consists of layers stacked from bottom to top, where each layer is a {gs}x{gs} grid with 0 (empty) or 1 (block).
        
        Important constraints:
        1. Layer 1 is the BOTTOM layer (ground level) - it should contain blocks, not be all zeros
        2. Blocks must be supported from below (a block at height h requires a block at height h-1 in the same position)
        3. Do not add unnecessary empty layers at the bottom or top
        4. Layers are numbered from bottom (Layer 1) to top (Layer N)
        5. Do not exceed the maximum height of {max_height} layers
        """

    prompt += f"""

        Provide your answer as a 3D structure with each layer specified.

        Output format:
        Structure:
        Layer 1:  (bottom layer - should contain at least one block)
        [row 1 values separated by spaces]
        [row 2 values separated by spaces]
        ...
        Layer 2:
        [row 1 values separated by spaces]
        [row 2 values separated by spaces]
        ...
        (continue for all layers needed, up to maximum {max_height} layers)
        
        Example of CORRECT output format for a 3x3 grid:
        Structure:
        Layer 1:
        1 0 1
        0 0 0
        1 0 1
        Layer 2:
        1 0 0
        0 0 0
        0 0 1
        
        Note: This represents a 3x3x2 structure where:
        - Layer 1 (bottom) has blocks at corners: (0,0), (0,2), (2,0), (2,2)
        - Layer 2 (top) has blocks at (0,0) and (2,2), both supported by blocks in Layer 1 below
        - Every '1' in Layer 2 has a '1' directly below it in Layer 1 (physical support requirement)
        - Uses spaces between digits (not commas or other separators)
        
        INCORRECT formats to avoid:
        1. Don't start with empty bottom layer: Layer 1 should contain blocks
        2. Don't use commas: "1,0,1" - use spaces instead: "1 0 1"
        3. Don't add blocks without support below them (physics violation)
        4. Don't exceed {max_height} layers in height
        """
    return prompt


def format_structure_target(gt_entry: Dict[str, Any]) -> str:
    """将 GT 样本格式化为官方标准 SFT 目标文本（`Structure:\\nLayer 1:\\n...`）。"""
    struct = decode_compact_layers(gt_entry.get("layers", []))
    return f"Structure:\n{struct.to_string()}"


def get_gt_bucket(n_gt: int) -> str:
    """将假设空间大小 |H_O| 划分为统一的 4 个分层桶（1 / 2-4 / 5-8 / 9+）。"""
    if n_gt <= 1:
        return "1"
    if n_gt <= 4:
        return "2-4"
    if n_gt <= 8:
        return "5-8"
    return "9+"


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """使用模型自带 chat template 构造输入文本。"""
    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt + "\n"


def _bucket_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"1": 0, "2-4": 0, "5-8": 0, "9+": 0}
    for it in items:
        counts[get_gt_bucket(int(it["n_compatible_structures"]))] += 1
    return counts


def _block_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for it in items:
        counts[f"b{int(it['n_blocks'])}"] += 1
    return dict(sorted(counts.items()))


def create_leakage_free_splits(
    id_dataset_path: Path,
    ood_raw_dataset_path: Path,
    splits_output_path: Path,
    test_dataset_output_path: Path,
    ood_dataset_output_path: Path,
    seed: int = 33550336,
) -> Dict[str, Any]:
    """
    构建零 D4 同构泄露的 3D `train / val / test` 切分与子集文件：
    1. 读取 `3d_g3_h3_b1to3.json`（129 题 -> 27 个 D4 等价类：b1=3, b2=8, b3=16）；
    2. 按方块数分层切分等价类（每档保证 train/val/test >= 1 类，总体约 60%/15%/25%）；
    3. 对 `3d_g3_h3_b4.json` 按 D4 等价类去重（每类按 `observation_id` 取首题）输出 `3d_g3_h3_b4_ood.json`；
    4. 严格断言 train/val/test 与 OOD 之间 D4 规范键零重叠，并导出官方格式 `3d_g3_h3_test.json`。
    """
    rng = random.Random(seed)
    id_meta, id_tasks = load_3d_dataset(id_dataset_path)
    ood_meta, ood_raw_tasks = load_3d_dataset(ood_raw_dataset_path)

    assert len(id_tasks) == 129, f"Expected 129 ID tasks, got {len(id_tasks)}"

    iso_by_blocks: Dict[int, Dict[Tuple[Any, ...], List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    all_iso_keys = set()
    for task in id_tasks:
        key = canonical_isomorphism_key(task)
        all_iso_keys.add(key)
        iso_by_blocks[int(task["n_blocks"])][key].append(task)

    expected_classes = {1: 3, 2: 8, 3: 16}
    actual_classes = {b: len(iso_by_blocks[b]) for b in sorted(iso_by_blocks.keys())}
    print(f"ID pool tasks={len(id_tasks)}, total D4 classes={len(all_iso_keys)}, per blocks={actual_classes}")
    assert len(all_iso_keys) == 27, f"Expected 27 D4 classes, got {len(all_iso_keys)}"
    assert actual_classes == expected_classes, f"Expected {expected_classes}, got {actual_classes}"

    # 每档方块数 ~60% / 15% / 25%，保证 test/val 在每个方块数至少有 1 类
    quotas = {
        1: (1, 1, 1),   # test=1, val=1, train=1
        2: (2, 1, 5),   # test=2, val=1, train=5
        3: (4, 2, 10),  # test=4, val=2, train=10
    }

    train_tasks: List[Dict[str, Any]] = []
    val_tasks: List[Dict[str, Any]] = []
    test_tasks: List[Dict[str, Any]] = []

    for n_blk in (1, 2, 3):
        groups = [
            (k, sorted(v, key=lambda x: x["observation_id"]))
            for k, v in iso_by_blocks[n_blk].items()
        ]
        groups.sort(key=lambda item: item[1][0]["observation_id"])
        rng.shuffle(groups)
        n_test_g, n_val_g, n_train_g = quotas[n_blk]
        assert n_test_g + n_val_g + n_train_g == len(groups)

        for _, grp in groups[:n_test_g]:
            test_tasks.extend(grp)
        for _, grp in groups[n_test_g : n_test_g + n_val_g]:
            val_tasks.extend(grp)
        for _, grp in groups[n_test_g + n_val_g :]:
            train_tasks.extend(grp)

    train_tasks.sort(key=lambda x: x["observation_id"])
    val_tasks.sort(key=lambda x: x["observation_id"])
    test_tasks.sort(key=lambda x: x["observation_id"])

    # 构造 D4 代表集 OOD 集（每个等价类按 observation_id 排序取第一个）
    ood_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for t in ood_raw_tasks:
        item = dict(t)
        item["source_dataset"] = ood_dataset_output_path.stem
        ood_groups[canonical_isomorphism_key(item)].append(item)

    ood_rep_tasks = [
        sorted(grp, key=lambda x: x["observation_id"])[0]
        for grp in ood_groups.values()
    ]
    ood_rep_tasks.sort(key=lambda x: x["observation_id"])

    # 零重叠断言
    train_keys = {canonical_isomorphism_key(t) for t in train_tasks}
    val_keys = {canonical_isomorphism_key(t) for t in val_tasks}
    test_keys = {canonical_isomorphism_key(t) for t in test_tasks}
    ood_keys = {canonical_isomorphism_key(t) for t in ood_rep_tasks}

    assert not (train_keys & val_keys), "Train/Val D4 equivalence class overlap!"
    assert not (train_keys & test_keys), "Train/Test D4 equivalence class overlap!"
    assert not (val_keys & test_keys), "Val/Test D4 equivalence class overlap!"
    assert not (train_keys & ood_keys), "Train/OOD D4 equivalence class overlap!"
    assert not (test_keys & ood_keys), "Test/OOD D4 equivalence class overlap!"

    train_ids = {t["observation_id"] for t in train_tasks}
    val_ids = {t["observation_id"] for t in val_tasks}
    test_ids = {t["observation_id"] for t in test_tasks}
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)

    print(
        f"Verified ZERO D4 overlap across train({len(train_keys)} cls/{len(train_tasks)} tasks), "
        f"val({len(val_keys)} cls/{len(val_tasks)} tasks), "
        f"test({len(test_keys)} cls/{len(test_tasks)} tasks), "
        f"and OOD({len(ood_keys)} cls/{len(ood_rep_tasks)} tasks)."
    )

    splits_payload: Dict[str, Any] = {
        "metadata": {
            "split_strategy": "d4_isomorphism_group_stratified",
            "seed": seed,
            "id_source": id_dataset_path.name,
            "ood_source": ood_dataset_output_path.name,
            "total_id_d4_classes": len(all_iso_keys),
            "counts": {
                "train_tasks": len(train_tasks),
                "val_tasks": len(val_tasks),
                "test_tasks": len(test_tasks),
                "ood_tasks": len(ood_rep_tasks),
                "train_groups": len(train_keys),
                "val_groups": len(val_keys),
                "test_groups": len(test_keys),
                "ood_groups": len(ood_keys),
                "test_queries": sum(int(t["n_compatible_structures"]) for t in test_tasks),
                "ood_queries": sum(int(t["n_compatible_structures"]) for t in ood_rep_tasks),
            },
            "block_distribution": {
                "train": _block_counts(train_tasks),
                "val": _block_counts(val_tasks),
                "test": _block_counts(test_tasks),
                "ood": _block_counts(ood_rep_tasks),
            },
            "bucket_distribution": {
                "train": _bucket_counts(train_tasks),
                "val": _bucket_counts(val_tasks),
                "test": _bucket_counts(test_tasks),
                "ood": _bucket_counts(ood_rep_tasks),
            },
        },
        "split_ids": {
            "train": [t["observation_id"] for t in train_tasks],
            "val": [t["observation_id"] for t in val_tasks],
            "test": [t["observation_id"] for t in test_tasks],
        },
        "splits": {
            "train": train_tasks,
            "val": val_tasks,
            "test": test_tasks,
        },
    }

    splits_output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_output_path.write_text(json.dumps(splits_payload, indent=2), encoding="utf-8")

    # 导出官方格式测试子集（供 DeepSeek runner 使用）
    def _strip_extra_for_official(t: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "observation_id": t["observation_id"],
            "observation": t["observation"],
            "ground_truth_structures": t["ground_truth_structures"],
            "n_compatible_structures": t["n_compatible_structures"],
        }

    test_official_payload = {
        "metadata": {
            **id_meta,
            "subset": "unseen_d4_test",
            "seed": seed,
            "n_observation_sets": len(test_tasks),
        },
        "observation_sets": [_strip_extra_for_official(t) for t in test_tasks],
    }
    test_dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_dataset_output_path.write_text(
        json.dumps(test_official_payload, indent=2), encoding="utf-8"
    )

    ood_official_payload = {
        "metadata": {
            **ood_meta,
            "subset": "d4_representatives_b4_ood",
            "n_raw_b4_tasks": len(ood_raw_tasks),
            "n_observation_sets": len(ood_rep_tasks),
        },
        "observation_sets": [_strip_extra_for_official(t) for t in ood_rep_tasks],
    }
    ood_dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    ood_dataset_output_path.write_text(
        json.dumps(ood_official_payload, indent=2), encoding="utf-8"
    )

    return splits_payload


def main() -> None:
    cfg = DataConfig()
    payload = create_leakage_free_splits(
        id_dataset_path=cfg.id_dataset_path,
        ood_raw_dataset_path=cfg.ood_raw_dataset_path,
        splits_output_path=cfg.splits_output_path,
        test_dataset_output_path=cfg.test_dataset_path,
        ood_dataset_output_path=cfg.ood_dataset_path,
        seed=cfg.seed,
    )
    print("\n=== 3D Zero-Leakage Splits Summary ===")
    print(json.dumps(payload["metadata"], indent=2))


if __name__ == "__main__":
    main()
