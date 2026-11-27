"""CPU tests covering the shipped datasets and official Boolean protocol."""

import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from benchmark_protocol import BooleanPrompt
from config import BOOLEAN_DIR, DataConfig, RewardConfig
from data_utils import (
    build_fixed_prompt, canonical_isomorphism_key, create_leakage_free_splits,
    format_expression_target, ground_truth_keys, load_boolean_dataset, make_observations,
)
from verifier_reward import (
    compute_grpo_advantages, compute_group_diversity_rewards, inspect_expression,
    mechanistic_subtrees, score_task_responses, subtree_jaccard_distance,
)


def load_official_benchmark():
    # Keep the official runner's implementation intact while removing network clients.
    stub = ModuleType("modules.llm_interface")
    for name in ("LLMInterface", "OpenRouterLLM", "OpenAILLM", "AnthropicLLM"):
        setattr(stub, name, object)
    path = BOOLEAN_DIR / "boolean_benchmark.py"
    spec = importlib.util.spec_from_file_location("official_boolean", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modules.llm_interface": stub}):
        spec.loader.exec_module(module)
    return module.BooleanBenchmarkRefined


class BooleanVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = DataConfig()
        cls.paths = (cls.cfg.basic_dataset_path, cls.cfg.id_dataset_path, cls.cfg.ood_dataset_path)
        cls.basic, cls.extended, cls.ood = [load_boolean_dataset(path)[1] for path in cls.paths]
        cls.official = load_official_benchmark()

    def task_with_observations(self, dataset, rows):
        task = dict(dataset[0])
        task["observations"] = [{"inputs": {"x": x, "y": y}, "output": out}
                                for x, y, out in rows]
        return task

    def test_official_prompt_byte_parity(self):
        for path, tasks in zip(self.paths, (self.basic, self.extended, self.ood)):
            task = tasks[0]
            with self.subTest(dataset=path.stem):
                benchmark = self.official(path)
                benchmark.operators = benchmark.metadata["operators"]
                golden = benchmark.create_prompt(make_observations(task), [])
                self.assertEqual(build_fixed_prompt(task), golden)
                self.assertIn("            None\n", golden)

    def test_official_parse_response(self):
        official = self.official(self.cfg.ood_dataset_path).parse_llm_response
        parse = BooleanPrompt(self.ood[0]).parse_llm_response
        examples = (
            ("Expression: x AND y", "x AND y"),
            ("  Expression:     x AND   y  ", "x AND   y"),
            ("expression: NOR(x, y)", "NOR(x, y)"),
            (r"\(x \land y\)", "x AND y"),
            (chr(96) + "x AND y" + chr(96), "x AND y"),
            ("garbage", "garbage"),
            ("@?!", None),
            (None, None),
        )
        for response, expected in examples:
            with self.subTest(response=response):
                self.assertEqual(parse(response), expected)
                self.assertEqual(parse(response), official(response))

    def test_official_runner_independence_and_metric_parity(self):
        class FakeLLM:
            def __init__(self, responses):
                self.responses = iter(responses)
                self.prompts = []

            def query(self, prompt):
                self.prompts.append(prompt)
                return next(self.responses)

        task = self.basic[0]
        benchmark = self.official(self.cfg.basic_dataset_path)
        benchmark.operators = benchmark.metadata["operators"]

        responses = ["x", "y", "x AND y", "@?!"]
        fixed = FakeLLM(responses)
        official = benchmark.evaluate_single_observation_set(
            fixed, task, len(responses), independent_samples=True,
        )
        self.assertEqual(fixed.prompts, [build_fixed_prompt(task)] * len(responses))
        metrics = score_task_responses(responses, task)
        for metric in ("n_valid", "n_unique_all", "n_unique_valid", "n_recovered_gts",
                       "parse_success_rate", "recovery_rate", "valid_rate"):
            self.assertEqual(metrics[metric], official[metric])
        self.assertEqual(metrics["uniqueness_rate"], official["novelty_rate"])

        history = FakeLLM(["@?!", "x", "y"])
        benchmark.evaluate_single_observation_set(history, task, 2)
        self.assertEqual(history.prompts[:2], [build_fixed_prompt(task)] * 2)
        self.assertIn("            Expression: x\n", history.prompts[2])

    def test_structural_keys_and_subtrees(self):
        task = self.extended[0]

        def key(formula):
            return inspect_expression(formula, task).key

        self.assertEqual(key("x AND y"), key("y AND x"))
        self.assertNotEqual(key("(x AND y) OR x"), key("x AND (y OR x)"))
        self.assertGreater(subtree_jaccard_distance(key("(x AND y) OR x"), key("x")), 0)
        self.assertEqual(mechanistic_subtrees(key("NOT (NOT y)")), {
            ("VAR", "y"), ("NOT", ("VAR", "y")), ("NOT", ("NOT", ("VAR", "y"))),
        })
        self.assertTrue(inspect_expression("NOR(x, y)", self.ood[0]).in_space)

    def test_space_constraints_and_reward_tiers(self):
        task = self.task_with_observations(self.basic, [(0, 0, 0), (0, 1, 0)])
        bad = ["***", "Expression: NOT x", "Expression: True", "Expression: x AND False",
               "Expression: x AND (y OR (x AND y))", "Expression: z",
               "Expression: x.anything()", "Expression: garbage"]
        scores = compute_group_diversity_rewards(["x", "y", *bad], task)
        self.assertAlmostEqual(scores.rewards[0], 1.0)
        self.assertAlmostEqual(scores.rewards[1], -0.55)
        self.assertTrue(all(reward == -1.0 for reward in scores.rewards[2:]))

    def test_group_reward_is_permutation_invariant(self):
        task = self.task_with_observations(self.basic, [(0, 0, 0)])
        responses = ["x AND y", "y AND x", "x", "***"]
        scores = compute_group_diversity_rewards(responses, task).rewards
        self.assertAlmostEqual(scores[0], 0.1 + (0.9 + 0.5 * (2 / 3)) / 2)
        self.assertEqual(scores[0], scores[1])
        self.assertEqual(compute_group_diversity_rewards(responses[::-1], task).rewards, scores[::-1])
        self.assertEqual(
            compute_group_diversity_rewards(responses, task, RewardConfig(mode="validity_only")).rewards,
            [1.0, 1.0, 1.0, -1.0],
        )
        self.assertEqual(compute_grpo_advantages([1.0, 1.0]), [0.0, 0.0])

    def test_recovery_uses_keys_and_uniqueness_includes_invalid_in_space(self):
        task = self.task_with_observations(self.basic, [(0, 1, 0)])
        task["ground_truth_expressions"] = [{"mechanistic_key": "('VAR', 'x')"}]
        record = score_task_responses(["x OR (x AND y)", "y", "x", "x"], task)
        self.assertEqual(record["n_unique_all"], 3)
        self.assertEqual(record["n_unique_valid"], 2)
        self.assertEqual(record["n_recovered_gts"], 1)
        self.assertEqual(record["uniqueness_rate"], 0.75)
        semantically_equivalent = score_task_responses(["x OR (x AND y)"], task)
        self.assertEqual(semantically_equivalent["valid_rate"], 1.0)
        self.assertEqual(semantically_equivalent["recovery_rate"], 0.0)

    def test_variable_swap_ignores_operator_set(self):
        first = self.task_with_observations(self.basic, [(0, 1, 0), (0, 0, 1)])
        second = self.task_with_observations(self.ood, [(0, 0, 1), (1, 0, 0)])
        self.assertEqual(canonical_isomorphism_key(first), canonical_isomorphism_key(second))

    def test_split_artifacts_and_cross_dataset_leakage(self):
        cfg = self.cfg
        payload, exported = create_leakage_free_splits(*self.paths, cfg.seed)
        self.assertEqual(json.loads(json.dumps(payload)), json.loads(cfg.splits_output_path.read_text(encoding="utf-8")))
        self.assertEqual(exported, json.loads(cfg.test_output_path.read_text(encoding="utf-8")))
        splits = payload["splits"]
        keys = {name: {canonical_isomorphism_key(task) for task in data} for name, data in splits.items()}
        self.assertFalse(keys["train"] & (keys["val"] | keys["test"]))
        self.assertFalse(keys["val"] & keys["test"])
        self.assertEqual(payload["metadata"]["basic_tasks_dropped"], 14)
        self.assertEqual(exported["metadata"]["total_observation_sets"], len(splits["test"]))
        self.assertEqual(len({task["task_uid"] for task in splits["train"]}), len(splits["train"]))

    def test_sft_swap_preserves_mechanistic_key(self):
        task = self.extended[0]
        formula = "NOT y OR (x AND y)"
        targets = {format_expression_target(formula, task, True, random.Random(seed))
                   for seed in range(16)}
        self.assertGreater(len(targets), 1)
        for target in targets:
            augmented = inspect_expression(target.removeprefix("Expression: "), task)
            self.assertTrue(augmented.in_space)
            self.assertEqual(augmented, inspect_expression(formula, task))

    def test_random_ground_truth_consistency_and_upstream_nor_exception(self):
        exception = "NOT (x  OR  y)"
        consistent = [task for task in self.extended
                      if all(gt["formula"] != exception for gt in task["ground_truth_expressions"])]
        for task in random.Random(self.cfg.seed).sample(consistent, 5):
            with self.subTest(task=task["task_uid"]):
                formulas = [gt["formula"] for gt in task["ground_truth_expressions"]]
                self.assertEqual([inspect_expression(f, task).key for f in formulas], ground_truth_keys(task))
                metrics = score_task_responses([f"Expression: {f}" for f in formulas], task)
                self.assertEqual(metrics["recovery_rate"], 1.0)
        for path, task, formula in (
            (self.cfg.id_dataset_path, self.extended[0], exception),
            (self.cfg.ood_dataset_path, self.ood[0], exception),
            (self.cfg.ood_dataset_path, self.ood[0], "x AND NOT (x OR y)"),
        ):
            expected = self.official(path)._in_space(formula)
            self.assertEqual(inspect_expression(formula, task).in_space, expected)
        self.assertFalse(inspect_expression(exception, self.extended[0]).in_space)
        self.assertTrue(inspect_expression(exception, self.ood[0]).in_space)


if __name__ == "__main__":
    unittest.main()
