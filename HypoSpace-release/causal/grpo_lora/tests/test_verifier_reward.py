"""
零显存单元测试（v2.1）：验证解析器、DAG 干预可达性验证器、严格层级与排列不变的多样性奖励、
`validity_only` 消融模式、GRPO 组内优势归一化、零同构泄露切分，以及测量修正逻辑
（采样参数统一、SFT 轮换覆盖、分层验证抽样与组等权平均）。
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

_GRPO_DIR = Path(__file__).resolve().parent.parent
if str(_GRPO_DIR) not in sys.path:
    sys.path.insert(0, str(_GRPO_DIR))

from config import DataConfig, EvalConfig, GRPOConfig, RewardConfig, SamplingConfig  # noqa: E402
from data_utils import (  # noqa: E402
    build_fixed_prompt,
    canonical_isomorphism_key,
    create_leakage_free_splits,
    format_graph_target,
    get_gt_bucket,
    load_causal_dataset,
)
from evaluate_lora import parse_args as eval_parse_args  # noqa: E402
from train_grpo import (  # noqa: E402
    compute_group_weighted_val_metrics,
    parse_args as grpo_parse_args,
    select_stratified_val_tasks,
)
from train_sft import build_sft_epoch_examples, init_task_gt_orders  # noqa: E402
from verifier_reward import (  # noqa: E402
    compute_grpo_advantages,
    compute_group_diversity_rewards,
    edge_jaccard_distance,
    evaluate_observation_match,
    parse_causal_graph,
)


class TestVerifierAndReward(unittest.TestCase):
    """核心验证器、奖励函数、切分与测量修正单元测试。"""

    def test_parse_causal_graph_cases(self) -> None:
        nodes = ["A", "B", "C", "D"]
        g1 = parse_causal_graph("Graph: A->B, B->C", nodes=nodes, max_edges=4)
        self.assertIsNotNone(g1)
        self.assertEqual(g1.edges, [("A", "B"), ("B", "C")])

        g_empty = parse_causal_graph("Graph: No edges", nodes=nodes, max_edges=4)
        self.assertIsNotNone(g_empty)
        self.assertEqual(g_empty.edges, [])

        self.assertIsNone(parse_causal_graph("Graph: A->B, B->A", nodes=nodes))
        self.assertIsNone(parse_causal_graph("Graph: A->A", nodes=nodes))
        self.assertIsNone(parse_causal_graph("Graph: A->Z", nodes=nodes))
        self.assertIsNone(
            parse_causal_graph("Graph: A->B, B->C, C->D", nodes=nodes, max_edges=2)
        )

    def test_real_dataset_ground_truths_pass_verifier(self) -> None:
        cfg = DataConfig()
        for ds_path in (cfg.n3_dataset_path, cfg.n4_dataset_path):
            _, tasks = load_causal_dataset(ds_path)
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

    def test_strict_hierarchy_and_order_invariance(self) -> None:
        nodes = ["A", "B", "C"]
        observations = [
            {"perturbed_node": "A", "effects": {"A": 0, "B": 1, "C": 1}},
            {"perturbed_node": "C", "effects": {"A": 0, "B": 0, "C": 0}},
        ]
        responses = [
            "Graph: A->B, A->C",           # g1 (freq=3)
            "Graph: A->B, A->C",           # g1 (freq=3)
            "Graph: A->B, A->C",           # g1 (freq=3)
            "Graph: A->B, B->C",           # g2 (freq=1)
            "Graph: A->B",                 # partial invalid DAG
            "invalid cyclic A->B, B->A",   # parse fail
        ]
        cfg = RewardConfig(mode="diversity")
        res = compute_group_diversity_rewards(responses, nodes, observations, max_edges=3, cfg=cfg)

        self.assertEqual(res.parse_flags, [True, True, True, True, True, False])
        self.assertEqual(res.valid_flags, [True, True, True, True, False, False])
        self.assertEqual(res.unique_valid_flags, [True, False, False, True, False, False])

        self.assertGreater(res.rewards[0], cfg.valid_floor_reward)
        self.assertGreater(res.rewards[0], 0.0)
        self.assertLess(res.rewards[4], 0.0)
        self.assertGreater(res.rewards[0], res.rewards[4])
        self.assertGreater(res.rewards[4], res.rewards[5])

        self.assertGreater(res.rewards[3], res.rewards[0])
        self.assertAlmostEqual(res.rewards[0], res.rewards[1])
        self.assertAlmostEqual(res.rewards[1], res.rewards[2])

        rev_res = compute_group_diversity_rewards(
            list(reversed(responses)), nodes, observations, max_edges=3, cfg=cfg
        )
        self.assertEqual(
            [round(x, 8) for x in res.rewards],
            [round(x, 8) for x in reversed(rev_res.rewards)],
        )

        adv = compute_grpo_advantages(res.rewards)
        self.assertAlmostEqual(sum(adv), 0.0, places=5)
        self.assertGreater(adv[3], adv[0])
        self.assertGreater(adv[0], adv[4])
        self.assertEqual(compute_grpo_advantages([1.0, 1.0, 1.0]), [0.0, 0.0, 0.0])

    def test_validity_only_ablation_mode(self) -> None:
        nodes = ["A", "B", "C"]
        observations = [
            {"perturbed_node": "A", "effects": {"A": 0, "B": 1, "C": 1}},
            {"perturbed_node": "C", "effects": {"A": 0, "B": 0, "C": 0}},
        ]
        responses = [
            "Graph: A->B, A->C",
            "Graph: A->B, A->C",
            "Graph: A->B, B->C",
            "Graph: A->B",
        ]
        cfg = RewardConfig(mode="validity_only")
        res = compute_group_diversity_rewards(responses, nodes, observations, max_edges=3, cfg=cfg)
        self.assertAlmostEqual(res.rewards[0], 1.0)
        self.assertAlmostEqual(res.rewards[1], 1.0)
        self.assertAlmostEqual(res.rewards[2], 1.0)
        self.assertLess(res.rewards[3], 0.0)

    def test_jaccard_distance(self) -> None:
        nodes = ["A", "B", "C"]
        g1 = parse_causal_graph("Graph: A->B", nodes)
        g2 = parse_causal_graph("Graph: A->B, B->C", nodes)
        g3 = parse_causal_graph("Graph: No edges", nodes)
        self.assertAlmostEqual(edge_jaccard_distance(g1, g1), 0.0)
        self.assertAlmostEqual(edge_jaccard_distance(g1, g2), 0.5)
        self.assertAlmostEqual(edge_jaccard_distance(g3, g3), 0.0)

    def test_zero_isomorphism_leakage_split(self) -> None:
        cfg = DataConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "splits.json"
            payload = create_leakage_free_splits(
                n3_path=cfg.n3_dataset_path,
                n4_path=cfg.n4_dataset_path,
                output_path=out_path,
            )
            train_n4 = [
                t for t in payload["splits"]["train"] if t.get("source_dataset") == "n4_obs3_all"
            ]
            val_n4 = payload["splits"]["val"]
            test_n4 = payload["splits"]["test"]

            id_tr = {t["observation_set_id"] for t in train_n4}
            id_va = {t["observation_set_id"] for t in val_n4}
            id_te = {t["observation_set_id"] for t in test_n4}
            self.assertEqual(len(id_tr & id_va), 0)
            self.assertEqual(len(id_tr & id_te), 0)
            self.assertEqual(len(id_va & id_te), 0)
            self.assertEqual(len(id_tr) + len(id_va) + len(id_te), 392)

            iso_tr = {canonical_isomorphism_key(t) for t in train_n4}
            iso_va = {canonical_isomorphism_key(t) for t in val_n4}
            iso_te = {canonical_isomorphism_key(t) for t in test_n4}
            self.assertEqual(len(iso_tr & iso_va), 0)
            self.assertEqual(len(iso_tr & iso_te), 0)
            self.assertEqual(len(iso_va & iso_te), 0)
            self.assertEqual(len(iso_tr) + len(iso_va) + len(iso_te), 22)

            for split_items in (train_n4, val_n4, test_n4):
                buckets = {get_gt_bucket(t["n_compatible_graphs"]) for t in split_items}
                self.assertEqual(buckets, {"1", "2-4", "5-8", "9+"})

    def test_unified_sampling_defaults(self) -> None:
        s_cfg = SamplingConfig()
        g_cfg = GRPOConfig()
        e_cfg = EvalConfig()
        g_args = grpo_parse_args([])
        e_args = eval_parse_args([])

        for obj in (g_cfg, e_cfg, g_args, e_args):
            self.assertEqual(obj.temperature, s_cfg.temperature)
            self.assertEqual(obj.top_p, s_cfg.top_p)
            self.assertEqual(obj.max_new_tokens, s_cfg.max_new_tokens)

    def test_sft_epoch_rotation_coverage(self) -> None:
        # 构造一道 K=16 的合成任务，验证 max_samples_per_task=8 时 2 个 epoch 覆盖全部 16 个 GT
        fake_gts = [{"nodes": ["A", "B"], "edges": [["A", "B"]] if i % 2 else []} for i in range(16)]
        tasks = [
            {
                "observation_set_id": "fake_k16",
                "source_dataset": "n4_obs3_all",
                "nodes": ["A", "B"],
                "max_edges": 1,
                "observations": [{"perturbed_node": "A", "effects": {"A": 0, "B": 0}, "string": "Perturb(A) -> A:0 B:0"}],
                "ground_truth_graphs": fake_gts,
            }
        ]
        orders = init_task_gt_orders(tasks, seed=42)
        ex0, cov0 = build_sft_epoch_examples(tasks, 8, 0, orders, shuffle_edges=False, seed=42)
        ex1, cov1 = build_sft_epoch_examples(tasks, 8, 1, orders, shuffle_edges=False, seed=42)

        self.assertEqual(len(ex0), 8)
        self.assertEqual(len(ex1), 8)
        uid = "n4_obs3_all:fake_k16"
        self.assertEqual(len(cov0[uid]), 8)
        self.assertEqual(len(cov1[uid]), 8)
        self.assertEqual(cov0[uid] | cov1[uid], set(range(16)))

    def test_stratified_val_sampling(self) -> None:
        splits_path = _GRPO_DIR / "splits" / "causal_splits.json"
        val_tasks = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]["val"]

        sub1 = select_stratified_val_tasks(val_tasks, tasks_per_group=12, seed=33550336)
        sub2 = select_stratified_val_tasks(val_tasks, tasks_per_group=12, seed=33550336)
        self.assertEqual(len(sub1), 48)
        self.assertEqual(
            [t["observation_set_id"] for t in sub1],
            [t["observation_set_id"] for t in sub2],
        )

        group_counts = {}
        for t in sub1:
            gid = t["_val_group_id"]
            group_counts[gid] = group_counts.get(gid, 0) + 1
        self.assertEqual(len(group_counts), 4)
        self.assertTrue(all(cnt == 12 for cnt in group_counts.values()))

    def test_group_equal_weighted_val_metrics(self) -> None:
        # 组 A 有 1 题 (recovery=1.0)，组 B 有 3 题 (recovery=0.0, 0.0, 0.2 -> 组均值 1/15? 取 0.2, 0.2, 0.2 -> 0.2)
        records = [
            {"group_id": "A", "n_ground_truths": 1, "valid_rate": 1.0, "uniqueness_rate": 1.0, "recovery_rate": 1.0},
            {"group_id": "B", "n_ground_truths": 4, "valid_rate": 0.5, "uniqueness_rate": 0.2, "recovery_rate": 0.2},
            {"group_id": "B", "n_ground_truths": 4, "valid_rate": 0.5, "uniqueness_rate": 0.2, "recovery_rate": 0.2},
            {"group_id": "B", "n_ground_truths": 4, "valid_rate": 0.5, "uniqueness_rate": 0.2, "recovery_rate": 0.2},
        ]
        metrics = compute_group_weighted_val_metrics(records)
        # 按题平均为 (1.0 + 0.2*3)/4 = 0.40；按组等权平均必须为 (1.0 + 0.2)/2 = 0.60
        self.assertAlmostEqual(metrics["val_recovery_rate"], 0.4)
        self.assertAlmostEqual(metrics["val_recovery_group_mean"], 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
