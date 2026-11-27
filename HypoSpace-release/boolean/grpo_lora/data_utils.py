"""Boolean datasets, variable-swap groups, fixed prompts, and SFT targets."""

import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import DataConfig
from benchmark_protocol import BooleanObservation, BooleanPrompt


GT_BUCKETS = ("1", "2-4", "5-8", "9+")


def get_gt_bucket(n_gt: int) -> str:
    if n_gt <= 1:
        return "1"
    if n_gt <= 4:
        return "2-4"
    if n_gt <= 8:
        return "5-8"
    return "9+"


def load_boolean_dataset(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    tasks = []
    for group in payload["datasets_by_n_observations"].values():
        for record in group:
            task = {**record, **{key: metadata[key] for key in (
                "variables", "operators", "max_depth", "mechanistic_opts",
            )}}
            task["source_dataset"] = path.stem
            task["task_uid"] = f"{path.stem}:{task['observation_set_id']}"
            count = len(task["ground_truth_expressions"])
            if not count or count != task["n_compatible_expressions"]:
                raise ValueError(f"Inconsistent ground-truth count in {task['task_uid']}")
            tasks.append(task)
    if len({task["task_uid"] for task in tasks}) != len(tasks):
        raise ValueError(f"Duplicate observation IDs in {path}")
    return payload, tasks


def canonical_isomorphism_key(task: dict) -> tuple:
    rows = [
        ((obs["inputs"]["x"], obs["inputs"]["y"]), obs["output"])
        for obs in task["observations"]
    ]
    original = tuple(sorted(rows))
    swapped = tuple(sorted(((y, x), out) for (x, y), out in rows))
    return min(original, swapped)


def make_observations(task: dict) -> list[BooleanObservation]:
    return [BooleanObservation(obs["inputs"], obs["output"]) for obs in task["observations"]]


def build_fixed_prompt(task: dict) -> str:
    return BooleanPrompt(task).create_prompt(make_observations(task), [])


def format_chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def ground_truth_keys(task: dict) -> list[tuple]:
    return [ast.literal_eval(gt["mechanistic_key"]) for gt in task["ground_truth_expressions"]]


def format_expression_target(
    formula: str, task: dict, shuffle_operands: bool, rng: random.Random,
) -> str:
    if shuffle_operands and task["mechanistic_opts"].get("apply_commutativity", True):
        # SFT consumes the basic and extended datasets. The AST round trip
        # keeps the precedence of NOT and mixed AND/OR expressions.
        from formula_syntax import translate_formula
        from verifier_reward import inspect_expression

        original = inspect_expression(formula, task)
        tree = ast.parse(translate_formula(formula), mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
                if rng.getrandbits(1):
                    node.left, node.right = node.right, node.left
        augmented = ast.unparse(tree).replace("&", "AND").replace("|", "OR").replace("~", "NOT ")
        candidate = inspect_expression(augmented, task)
        if candidate.in_space and candidate.key == original.key:
            formula = augmented
    return f"Expression: {formula}"


def describe_tasks(tasks: list[dict]) -> dict:
    counts = Counter(task["n_compatible_expressions"] for task in tasks)
    buckets = Counter(get_gt_bucket(task["n_compatible_expressions"]) for task in tasks)
    return {
        "n_tasks": len(tasks),
        "n_equivalence_classes": len({canonical_isomorphism_key(task) for task in tasks}),
        "n_queries": sum(task["n_compatible_expressions"] for task in tasks),
        "gt_counts": dict(sorted(counts.items())),
        "gt_buckets": {bucket: buckets[bucket] for bucket in GT_BUCKETS},
    }


def create_leakage_free_splits(
    basic_path: Path, id_path: Path, ood_path: Path, seed: int,
) -> tuple[dict, dict]:
    _, basic = load_boolean_dataset(basic_path)
    original_id, extended = load_boolean_dataset(id_path)
    _, ood = load_boolean_dataset(ood_path)
    groups = defaultdict(list)
    for task in extended:
        groups[canonical_isomorphism_key(task)].append(task)

    by_bucket = defaultdict(list)
    for key, tasks in sorted(groups.items()):
        # A swap preserves the dataset's hypothesis count. Fail loudly if a future
        # source breaks this assumption rather than stratifying one class twice.
        counts = {task["n_compatible_expressions"] for task in tasks}
        if len(counts) != 1:
            raise ValueError(f"Inconsistent ground-truth counts within swap class {key}")
        by_bucket[get_gt_bucket(next(iter(counts)))].append(key)

    rng = random.Random(seed)
    split_keys = {"train": set(), "val": set(), "test": set()}
    for bucket in GT_BUCKETS:
        keys = by_bucket[bucket]
        rng.shuffle(keys)
        n = len(keys)
        n_val = max(1, round(0.15 * n)) if n >= 3 else 0
        n_test = max(1, round(0.25 * n)) if n >= 2 else 0
        split_keys["val"].update(keys[:n_val])
        split_keys["test"].update(keys[n_val:n_val + n_test])
        split_keys["train"].update(keys[n_val + n_test:])

    splits = {name: [task for task in extended if canonical_isomorphism_key(task) in keys]
              for name, keys in split_keys.items()}
    held_out = split_keys["test"] | split_keys["val"]
    basic_train = [task for task in basic if canonical_isomorphism_key(task) not in held_out]
    splits["train"].extend(basic_train)
    actual_keys = {name: {canonical_isomorphism_key(t) for t in tasks}
                   for name, tasks in splits.items()}
    overlap = sum(len(actual_keys[a] & actual_keys[b])
                  for a, b in (("train", "val"), ("train", "test"), ("val", "test")))
    if overlap or sum(map(len, splits.values())) != len(extended) + len(basic_train):
        raise AssertionError("Boolean ID split leaks classes or loses extended tasks")

    metadata = {
        "seed": seed,
        "split_ratios_target": {"train": 0.60, "val": 0.15, "test": 0.25},
        "canonical_symmetry": "x/y swap on observations, ignoring operators",
        "extended_equivalence_classes": len(groups),
        "basic_tasks_kept": len(basic_train),
        "basic_tasks_dropped": len(basic) - len(basic_train),
        "id_class_overlap": overlap,
        "ood_classes_seen_in_train": len(
            {canonical_isomorphism_key(t) for t in ood} & actual_keys["train"]
        ),
        "counts": {name: describe_tasks(tasks) for name, tasks in {**splits, "ood": ood}.items()},
    }
    test_ids = {task["observation_set_id"] for task in splits["test"]}
    test_export = {
        **original_id,
        "metadata": {**original_id["metadata"], "total_observation_sets": len(test_ids)},
        "datasets_by_n_observations": {
            count: [task for task in tasks if task["observation_set_id"] in test_ids]
            for count, tasks in original_id["datasets_by_n_observations"].items()
        },
    }
    return {"metadata": metadata, "splits": splits}, test_export


def write_json(path: Path, data: dict, *, protect_existing: bool = False) -> None:
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if protect_existing and path.exists():
        if path.read_text(encoding="utf-8") == text:
            return
        raise FileExistsError(f"Choose a new --test-output path; {path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    cfg = DataConfig()
    parser = argparse.ArgumentParser(description="Create disjoint Boolean ID splits by variable swap")
    parser.add_argument("--basic", type=Path, default=cfg.basic_dataset_path)
    parser.add_argument("--id", type=Path, default=cfg.id_dataset_path)
    parser.add_argument("--ood", type=Path, default=cfg.ood_dataset_path)
    parser.add_argument("--output", type=Path, default=cfg.splits_output_path)
    parser.add_argument("--test-output", type=Path, default=cfg.test_output_path)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()
    payload, test_export = create_leakage_free_splits(args.basic, args.id, args.ood, args.seed)
    write_json(args.test_output, test_export, protect_existing=True)
    write_json(args.output, payload)
    print(json.dumps(payload["metadata"], indent=2))
    print("PASS: train/val/test share zero equivalence classes.")


if __name__ == "__main__":
    main()
