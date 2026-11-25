"""
零显存单元测试：验证解析器、DAG 干预可达性验证器、集合级多样性奖励、
GRPO 组内优势归一化以及无泄露数据切分逻辑。
"""

import sys
import tempfile
import unittest
from pathlib import Path

_GRPO_DIR = Path(__file__).resolve().parent.parent
if str(_GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(_GRPO_DIR))

from config import DataConfig, RewardConfig  # noqa: E402
from data_utils import (  # noqa: E402
    build_fixed_prompt,
    create_leakage_free_splits,
    format_graph_target,
    load_causal_dataset,
)
from verifier_reward import (  # noqa: E402
    compute_grpo_advantages,
    compute_group_diversity_rewards,
    edge_jaccard_distance,
    evaluate_observation_match,
    parse_causal_graph,
)


class TestVerifierAndReward(unittest.TestCase):
    """核心验证器与奖励函数测试。"""

    def test_parse_causal_graph_cases(self) -> None:
        nodes = ["A", "B", "C", "D"]
        # 正常解析
        g1 = parse_causal_graph("Graph: A->B, B->C", nodes=nodes, max_edges=4)
        self.assertIsNotNone(g1)
        self.assertEqual(g1.edges, [("A", "B"), ("B", "C")])

        # 空图解析
        g_empty = parse_causal_graph("Graph: No edges", nodes=nodes, max_edges=4)
        self.assertIsNotNone(g_empty)
        self.assertEqual(g_empty.edges, [])

        # 含环拒绝
        self.assertIsNone(parse_causal_graph("Graph: A->B, B->A", nodes=nodes))
        # 自环拒绝
        self.assertIsNone(parse_causal_graph("Graph: A->A", nodes=nodes))
        # 未知节点拒绝
        self.assertIsNone(parse_causal_graph("Graph: A->Z", nodes=nodes))
        # 超出最大边数拒绝
        self.assertIsNone(parse_causal_graph("Graph: A->B, B->C, C->D", nodes=nodes, max_edges=2))

    def test_real_dataset_ground_truths_pass_verifier(self) -> None:
        cfg = DataConfig()
        for ds_path in (cfg.n3_dataset_path, cfg.n4_dataset_path):
            _, tasks = load_causal_dataset(ds_path)
            # 抽检前 30 题的全部 ground truth 图，确保与官方语义 100% 一致
            for task in tasks[:30]:
                prompt = build_fixed_prompt(
                    nodes=task["nodes"],
                    observations=task["observations"],
                    max_edges=task.get("max_edges"),
                )
                self.assertIn("Prior predictions (do not repeat if avoidable):\nNone", prompt)
                for gt in task["ground_truth_graphs"]:
                    text = format_graph_target(gt["edges"])
                    parsed = parse_causal_graph(
                        text, nodes=task["nodes"], max_edges=task.get("max_edges")
                    )
                    self.assertIsNotNone(parsed)
                    is_valid, ratio = evaluate_observation_match(parsed, task["observations"])
                    self.assertTrue(is_valid, f"GT failed verifier in {task['observation_set_id']}")
                    self.assertEqual(ratio, 1.0)

    def test_group_diversity_rewards_and_advantages(self) -> None:
        nodes = ["A", "B", "C"]
        # 仅观测 Perturb(A) 与 Perturb(C)，此时 A->B, A->C 与 A->B, B->C 与 A->B, A->C, B->C 均合法
        observations = [
            {"perturbed_node": "A", "effects": {"A": 0, "B": 1, "C": 1}},
            {"perturbed_node": "C", "effects": {"A": 0, "B": 0, "C": 0}},
        ]
        responses = [
            "Graph: A->B, A->C",           # 首次唯一有效（基础分 + Jaccard 奖励）
            "Graph: A->B, A->C",           # 重复有效（惩罚）
            "Graph: A->B, B->C",           # 第二个唯一有效（基础分 + Jaccard 奖励）
            "Graph: B->A",                 # 合法 DAG 但不满足观测（部分惩罚）
            "invalid cyclic A->B, B->A",   # 无法解析/含环（最高惩罚）
        ]
        cfg = RewardConfig()
        res = compute_group_diversity_rewards(responses, nodes, observations, max_edges=3, cfg=cfg)

        self.assertEqual(res.parse_flags, [True, True, True, True, False])
        self.assertEqual(res.valid_flags, [True, True, True, False, False])
        self.assertEqual(res.unique_valid_flags, [True, False, True, False, False])

        # 验证奖励大小排序：唯一有效 > 0 > 重复有效 & 不满足观测 > 解析失败
        self.assertGreater(res.rewards[0], 1.0)
        self.assertGreater(res.rewards[2], 1.0)
        self.assertEqual(res.rewards[1], cfg.duplicate_valid_reward)
        self.assertGreater(res.rewards[3], res.rewards[4])
        self.assertEqual(res.rewards[4], cfg.parse_fail_reward)

        adv = compute_grpo_advantages(res.rewards)
        self.assertAlmostEqual(sum(adv), 0.0, places=5)
        self.assertGreater(adv[0], adv[1])

        # 方差为 0 时安全返回全 0
        self.assertEqual(compute_grpo_advantages([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])

    def test_jaccard_distance(self) -> None:
        nodes = ["A", "B", "C"]
        g1 = parse_causal_graph("Graph: A->B", nodes)
        g2 = parse_causal_graph("Graph: A->B, B->C", nodes)
        g3 = parse_causal_graph("Graph: No edges", nodes)
        self.assertAlmostEqual(edge_jaccard_distance(g1, g1), 0.0)
        self.assertAlmostEqual(edge_jaccard_distance(g1, g2), 0.5)
        self.assertAlmostEqual(edge_jaccard_distance(g3, g3), 0.0)

    def test_leakage_free_split_disjoint(self) -> None:
        cfg = DataConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "splits.json"
            payload = create_leakage_free_splits(
                n3_path=cfg.n3_dataset_path,
                n4_path=cfg.n4_dataset_path,
                output_path=out_path,
            )
            train_n4 = set(payload["split_ids"]["train_n4"])
            val_n4 = set(payload["split_ids"]["val"])
            test_n4 = set(payload["split_ids"]["test"])

            self.assertEqual(len(train_n4 & val_n4), 0)
            self.assertEqual(len(train_n4 & test_n4), 0)
            self.assertEqual(len(val_n4 & test_n4), 0)
            self.assertEqual(len(train_n4) + len(val_n4) + len(test_n4), 392)


if __name__ == "__main__":
    unittest.main(verbosity=2)
