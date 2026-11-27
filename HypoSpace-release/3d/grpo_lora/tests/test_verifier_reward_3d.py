"""
3D Voxel Reconstruction 解析器、物理验证器、奖励函数与 D4 切分单元测试（CPU 即可运行）。
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent
_GRPO_DIR = _TESTS_DIR.parent
if str(_GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(_GRPO_DIR))

from config import DataConfig, RewardConfig  # noqa: E402
from data_utils import (  # noqa: E402
    build_fixed_prompt,
    canonical_isomorphism_key,
    d4_transforms,
    load_3d_dataset,
)
from verifier_reward import (  # noqa: E402
    Structure3D,
    compute_group_diversity_rewards,
    decode_compact_layers,
    parse_llm_response,
    parse_top_view,
    validate_structure_matches_observations,
)


def _official_create_prompt_golden(observations: str, max_height: int = 3) -> str:
    """Verbatim reference prompt from `run_3d_benchmark.py:184-306` with prior_structures=None."""
    grid_size = int(len(observations) ** 0.5)
    obs_arr = np.array(
        [[int(observations[i * grid_size + j]) for j in range(grid_size)] for i in range(grid_size)]
    )
    prompt = f"""You are given observations of a 3D structure made of unit blocks on a {grid_size}x{grid_size} grid.
        Each observation shows a view of the structure from a specific angle.
        The maximum height of the structure is {max_height} layers.

        Observations (Top View - shows 1 if ANY layer has a block at that position):
        """
    prompt += "\nTop view:\n"
    for row in obs_arr:
        prompt += " ".join(str(cell) for cell in row) + "\n"
    prompt += f"""
        
        Task: Infer the complete 3D structure that could produce these observations.

        Structure specifications:
        - Grid size: {grid_size}x{grid_size}
        - Maximum height: {max_height} layers
        - The structure consists of layers stacked from bottom to top, where each layer is a {grid_size}x{grid_size} grid with 0 (empty) or 1 (block).
        
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


class Test3DVerifierReward(unittest.TestCase):
    """覆盖 Part A.8 规定的全部 8 项核心单元测试。"""

    def test_parsing_formats_and_garbage(self) -> None:
        official = "Structure:\nLayer 1:\n1 0 1\n0 0 0\n0 0 0\nLayer 2:\n1 0 0\n0 0 0\n0 0 0"
        comma = "Structure:\nLayer 1:\n1, 0, 1\n0, 0, 0\n0, 0, 0"
        bracket = "Layer 1:\n[1 0 1]\n[0 0 0]\n[0 0 0]"

        s1 = parse_llm_response(official)
        s2 = parse_llm_response(comma)
        s3 = parse_llm_response(bracket)
        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        self.assertIsNotNone(s3)
        assert s1 is not None and s2 is not None and s3 is not None
        self.assertEqual(s1.height, 2)
        self.assertEqual(s2.get_hash(), s3.get_hash())

        self.assertIsNone(parse_llm_response("completely invalid gibberish output"))
        self.assertIsNone(parse_llm_response(""))

    def test_support_constraint_and_trailing_zero_hash(self) -> None:
        obs = "100000000"
        floating = parse_llm_response("Structure:\nLayer 1:\n0 0 0\n0 0 0\n0 0 0\nLayer 2:\n1 0 0\n0 0 0\n0 0 0")
        self.assertIsNotNone(floating)
        assert floating is not None
        self.assertFalse(validate_structure_matches_observations(floating, obs))

        base_struct = parse_llm_response("Structure:\nLayer 1:\n1 0 0\n0 0 0\n0 0 0")
        padded_struct = parse_llm_response(
            "Structure:\nLayer 1:\n1 0 0\n0 0 0\n0 0 0\nLayer 2:\n0 0 0\n0 0 0\n0 0 0"
        )
        assert base_struct is not None and padded_struct is not None
        self.assertEqual(base_struct.get_hash(), padded_struct.get_hash())

    def test_reward_hierarchy_strict_order(self) -> None:
        obs = "100000001"
        valid_a = "Structure:\nLayer 1:\n1 0 0\n0 0 0\n0 0 1"
        valid_dup = valid_a
        invalid_partial = "Structure:\nLayer 1:\n1 0 0\n0 0 0\n0 0 0"
        invalid_support = "Structure:\nLayer 1:\n1 0 0\n0 0 0\n0 0 0\nLayer 2:\n0 0 1\n0 0 0\n0 0 0"
        out_of_height = (
            "Structure:\nLayer 1:\n1 0 0\n0 0 0\n0 0 1\n"
            "Layer 2:\n1 0 0\n0 0 0\n0 0 0\n"
            "Layer 3:\n1 0 0\n0 0 0\n0 0 0\n"
            "Layer 4:\n1 0 0\n0 0 0\n0 0 0"
        )
        garbage = "not a structure"

        res = compute_group_diversity_rewards(
            [valid_a, valid_dup, invalid_partial, invalid_support, out_of_height, garbage],
            observation=obs,
            grid_size=3,
            max_height=3,
        )
        r_valid_1, r_valid_2, r_inv_part, r_inv_sup, r_tall, r_garb = res.rewards
        self.assertGreaterEqual(r_valid_1, 0.1)
        self.assertEqual(r_valid_1, r_valid_2)
        self.assertGreater(r_valid_1, r_inv_part)
        self.assertGreater(r_valid_1, r_inv_sup)
        self.assertTrue(-0.7 <= r_inv_part <= -0.4)
        self.assertTrue(-0.7 <= r_inv_sup <= -0.4)
        self.assertGreater(r_inv_part, r_tall)
        self.assertEqual(r_tall, -1.0)
        self.assertEqual(r_garb, -1.0)

    def test_permutation_invariance(self) -> None:
        obs = "110000000"
        s_a = "Structure:\nLayer 1:\n1 1 0\n0 0 0\n0 0 0"
        s_b = "Structure:\nLayer 1:\n1 1 0\n0 0 0\n0 0 0\nLayer 2:\n1 0 0\n0 0 0\n0 0 0"
        res1 = compute_group_diversity_rewards([s_a, s_a, s_b, s_a], obs, 3, 3)
        res2 = compute_group_diversity_rewards([s_b, s_a, s_a, s_a], obs, 3, 3)

        self.assertAlmostEqual(res1.rewards[0], res1.rewards[1], places=8)
        self.assertAlmostEqual(res1.rewards[0], res1.rewards[3], places=8)
        self.assertAlmostEqual(res1.rewards[0], res2.rewards[1], places=8)
        self.assertAlmostEqual(res1.rewards[2], res2.rewards[0], places=8)

    def test_validity_only_mode(self) -> None:
        obs = "100000000"
        s1 = "Structure:\nLayer 1:\n1 0 0\n0 0 0\n0 0 0"
        s2 = "Structure:\nLayer 1:\n1 0 0\n0 0 0\n0 0 0\nLayer 2:\n1 0 0\n0 0 0\n0 0 0"
        res = compute_group_diversity_rewards(
            [s1, s1, s2],
            observation=obs,
            grid_size=3,
            max_height=3,
            cfg=RewardConfig(mode="validity_only"),
        )
        self.assertEqual(res.rewards, [1.0, 1.0, 1.0])

    def test_d4_canonical_isomorphism_key(self) -> None:
        base_task = {"observation": "100000000", "grid_size": 3, "max_height": 3}
        base_grid = parse_top_view(base_task["observation"])
        base_key = canonical_isomorphism_key(base_task)
        for t in d4_transforms(base_grid):
            flat = "".join("".join(str(x) for x in row) for row in t)
            self.assertEqual(
                canonical_isomorphism_key({"observation": flat, "grid_size": 3, "max_height": 3}),
                base_key,
            )

        center_task = {"observation": "000010000", "grid_size": 3, "max_height": 3}
        self.assertNotEqual(base_key, canonical_isomorphism_key(center_task))

    def test_build_fixed_prompt_exact_match(self) -> None:
        for obs in ("100000000", "101010001", "010110000"):
            self.assertEqual(
                build_fixed_prompt(obs, grid_size=3, max_height=3),
                _official_create_prompt_golden(obs, max_height=3),
            )

    def test_gt_enumeration_and_validity(self) -> None:
        cfg = DataConfig()
        _, tasks = load_3d_dataset(cfg.id_dataset_path)
        rng = random.Random(33550336)
        sample = rng.sample(tasks, 5)
        for task in sample:
            k = int(task["n_blocks"])
            h = int(task["max_height"])
            gts = task["ground_truth_structures"]
            self.assertEqual(len(gts), h**k)
            for gt in gts:
                st = decode_compact_layers(gt["layers"])
                self.assertTrue(validate_structure_matches_observations(st, task["observation"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
