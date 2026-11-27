"""Official Boolean evaluation semantics and grouped GRPO rewards."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Optional

import numpy as np
import sympy as sp

from benchmark_protocol import BooleanPrompt
from boolean_dataset import BooleanExpression
from config import RewardConfig
from data_utils import ground_truth_keys, make_observations
from formula_syntax import is_safe_formula


@dataclass(frozen=True)
class ExpressionResult:
    in_space: bool = False
    key: Optional[tuple] = None
    truth_values: tuple[int, ...] = ()


# Adapted from boolean_benchmark.py:239-361 at 75cf9f5. Parse once,
# retaining official raw AST depth and the Not(Or(...)) NOR treatment.
def _ast_depth(node) -> int:
    children = getattr(node, "args", ())
    return 1 + max(map(_ast_depth, children)) if children else 0


def _operators_allowed(node, allowed: tuple[str, ...]) -> bool:
    if isinstance(node, sp.Not) and isinstance(node.args[0], sp.Or):
        return "NOR" in allowed

    names = {
        sp.Not: "NOT", sp.And: "AND", sp.Or: "OR", sp.Xor: "XOR",
        sp.Implies: "IMPLIES", sp.Equivalent: "EQUIV",
    }
    operator = names.get(type(node))
    if operator and operator not in allowed:
        return False
    return all(_operators_allowed(child, allowed) for child in node.args)


@lru_cache(maxsize=8192)
def _inspect(formula, variables, operators, max_depth, opts) -> ExpressionResult:
    if not is_safe_formula(formula, variables):
        return ExpressionResult()
    try:
        expr = BooleanExpression(formula, list(variables), set(operators))
        node = expr.sympy_expr
        if node is None or not node.free_symbols.issubset({sp.Symbol(v) for v in variables}):
            return ExpressionResult()
        if not _operators_allowed(node, operators) or _ast_depth(node) > max_depth:
            return ExpressionResult()
        for child in sp.preorder_traversal(node):
            if isinstance(child, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                return ExpressionResult()
        return ExpressionResult(True, expr.mechanistic_key(**dict(opts)), tuple(expr.truth_table.values()))
    except (AttributeError, TypeError, ValueError, RecursionError):
        return ExpressionResult()


def inspect_expression(formula: str, task: dict) -> ExpressionResult:
    return _inspect(
        formula, tuple(sorted(task["variables"])), tuple(task["operators"]),
        task["max_depth"], tuple(sorted(task["mechanistic_opts"].items())),
    )


def _score_responses(responses, task):
    parser = BooleanPrompt(task)
    observations = make_observations(task)
    scores = []
    for response in responses:
        formula = parser.parse_llm_response(response)
        expr = inspect_expression(formula, task) if formula else ExpressionResult()
        matched = 0
        if expr.in_space:
            for obs in observations:
                x, y = (obs.inputs[name] for name in sorted(task["variables"]))
                matched += expr.truth_values[2 * x + y] == obs.output
        scores.append((formula, expr, matched / len(observations)))
    return scores


def score_task_responses(responses: list[str], task: dict) -> dict:
    """Compute official in-space novelty and stored mechanistic-key recovery."""
    scores = _score_responses(responses, task)
    valid = [expr.key for _, expr, ratio in scores if expr.in_space and ratio == 1.0]
    unique_all = {expr.key for _, expr, _ in scores if expr.in_space}
    unique_valid = set(valid)
    ground_truths = ground_truth_keys(task)
    recovered = sum(key in unique_valid for key in ground_truths)
    n_queries = len(responses)
    n_parsed = sum(formula is not None for formula, _, _ in scores)
    return {
        "observation_set_id": task["observation_set_id"],
        "task_uid": task["task_uid"],
        "n_observations": len(task["observations"]),
        "n_ground_truths": len(ground_truths),
        "n_queries": n_queries,
        "n_parsed": n_parsed,
        "n_valid": len(valid),
        "n_unique_all": len(unique_all),
        "n_unique_valid": len(unique_valid),
        "n_recovered_gts": recovered,
        "parse_success_rate": n_parsed / n_queries,
        "valid_rate": len(valid) / n_queries,
        "uniqueness_rate": len(unique_all) / n_queries,
        "recovery_rate": recovered / len(ground_truths),
        "raw_responses": responses,
    }


def mechanistic_subtrees(key: tuple) -> set[tuple]:
    """Respect the official key's unary, n-ary, and leaf representations."""
    if key[0] in {"VAR", "CONST"}:
        return {key}
    children = (key[1],) if key[0] == "NOT" else key[1]
    return {key}.union(*(mechanistic_subtrees(child) for child in children))


def subtree_jaccard_distance(first: tuple, second: tuple) -> float:
    a, b = mechanistic_subtrees(first), mechanistic_subtrees(second)
    return 1.0 - len(a & b) / len(a | b)


@dataclass
class GroupRewardResult:
    rewards: list[float]
    parse_rate: float
    valid_rate: float
    unique_valid_rate: float


def compute_group_diversity_rewards(
    responses: list[str], task: dict, cfg: Optional[RewardConfig] = None,
) -> GroupRewardResult:
    cfg = cfg or RewardConfig()
    if cfg.mode not in {"diversity", "validity_only"}:
        raise ValueError(f"Unknown reward mode {cfg.mode!r}")
    scores = _score_responses(responses, task)
    counts = Counter(expr.key for _, expr, ratio in scores if expr.in_space and ratio == 1.0)
    distance_sums = dict.fromkeys(counts, 0.0)
    if cfg.mode == "diversity":
        # Sort keys so even floating point summation is permutation-invariant.
        for first, second in combinations(sorted(counts), 2):
            distance = subtree_jaccard_distance(first, second)
            distance_sums[first] += distance
            distance_sums[second] += distance

    rewards = []
    for _, expr, ratio in scores:
        if not expr.in_space:
            reward = cfg.parse_fail_reward
        elif ratio < 1.0:
            reward = cfg.invalid_base_reward + cfg.invalid_match_bonus * ratio
        elif cfg.mode == "validity_only":
            reward = 1.0
        else:
            distance = distance_sums[expr.key] / max(1, len(counts) - 1)
            reward = cfg.valid_floor_reward + (
                cfg.valid_unique_bonus + cfg.diversity_weight * distance
            ) / counts[expr.key]
        rewards.append(reward)

    n_queries = max(1, len(responses))
    return GroupRewardResult(
        rewards, sum(formula is not None for formula, _, _ in scores) / n_queries,
        sum(counts.values()) / n_queries, len(counts) / n_queries,
    )


def compute_grpo_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    values = np.asarray(rewards, dtype=np.float32)
    return ((values - values.mean()) / max(float(values.std()), eps)).tolist()
