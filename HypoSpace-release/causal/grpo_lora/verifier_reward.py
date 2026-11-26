"""
Causal 任务精确解析器、DAG 验证器与集合级多样性 GRPO 奖励模块（v2.0）。

v2.0 核心修复与改进：
1. **严格单调奖惩层级**：确保任意合法解（即使在组内重复）奖励 >= `valid_floor_reward` (+0.1)，
   严格高于所有不满足观测的非法 DAG（<= -0.4）与无法解析的输出（-1.0），杜绝激励倒挂。
2. **排列不变的逆频次覆盖奖励（Order-Invariant Inverse-Frequency Reward）**：
   旧版仅惩罚“第 2 次及以后出现”，导致同一张图的首个样本得高正分、其余同图样本得负分，
   使模型梯度受采样顺序影响；v2.0 改为按整组出现频次 `count(g)` 均摊：
   `reward = valid_floor + (valid_unique_bonus + diversity_weight * jaccard_dist) / count(g)`，
   既严格奖励覆盖更多不同合法图、惩罚过度集中，又做到 100% 排列顺序无关。
3. **支持 `validity_only` 消融模式**：用于对照验证“无多样性奖励的普通 RL”是否加剧模式崩塌。
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

# 确保无论从哪个工作目录执行，均可导入 `causal/modules/models.py`
_CAUSAL_DIR = Path(__file__).resolve().parent.parent
if str(_CAUSAL_DIR) not in sys.path:
    sys.path.insert(0, str(_CAUSAL_DIR))

from modules.models import CausalGraph  # noqa: E402
from config import RewardConfig  # noqa: E402


@dataclass
class GroupRewardResult:
    """单个观测集上采样一组 G 个响应后的完整评估与奖励结果。"""

    rewards: List[float]
    parsed_graphs: List[Optional[CausalGraph]]
    parse_flags: List[bool]
    valid_flags: List[bool]
    unique_valid_flags: List[bool]
    obs_match_ratios: List[float]
    valid_rate: float
    unique_valid_rate: float
    parse_rate: float


def parse_causal_graph(
    response: str,
    nodes: Sequence[str],
    max_edges: Optional[int] = None,
) -> Optional[CausalGraph]:
    """
    将 LLM 文本输出解析为 `CausalGraph`（与官方 `run_causal_benchmark.py` 完全一致）。

    校验规则：
    - 提取 `Graph: ...` 行（若缺失则取首行）；
    - 支持 `Graph: No edges` 空图；
    - 仅允许 `nodes` 内节点，禁止自环、禁止超出 `max_edges`、必须构成有向无环图（DAG）。
    """
    if not isinstance(response, str):
        return None

    cleaned = response.replace("```", "").strip()
    match = re.search(r"(?i)\bgraph\s*:\s*(.+)$", cleaned, flags=re.MULTILINE)
    if match:
        line = match.group(1).strip()
    elif cleaned.splitlines():
        line = cleaned.splitlines()[0].strip()
    else:
        return None

    if not line:
        return None

    if (line.startswith('"') and line.endswith('"')) or (
        line.startswith("'") and line.endswith("'")
    ):
        line = line[1:-1].strip()

    line = (
        line.replace("→", "->")
        .replace("-->", "->")
        .replace("=>", "->")
        .rstrip(" .;")
    )

    node_list = list(nodes)
    node_set = set(node_list)

    if re.search(r"\b(no\s+edges?|empty|none|null)\b", line, re.I):
        return CausalGraph(nodes=node_list, edges=[])

    parts = [p.strip() for p in line.split(",") if p.strip()]
    if not parts:
        return None

    edges: List[Tuple[str, str]] = []
    for part in parts:
        edge_match = re.fullmatch(r"([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)", part)
        if not edge_match:
            return None
        u, v = edge_match.group(1), edge_match.group(2)
        if u not in node_set or v not in node_set or u == v:
            return None
        edges.append((u, v))

    # 保序去重
    edges = list(dict.fromkeys(edges))

    if max_edges is not None and len(edges) > max_edges:
        return None

    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(node_list)
    graph_nx.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph_nx):
        return None

    return CausalGraph(nodes=node_list, edges=edges)


def compute_perturbation_effects(
    graph: CausalGraph,
    perturbed_node: str,
) -> Dict[str, int]:
    """
    计算在给定因果图上扰动 `perturbed_node` 后的节点状态。

    官方语义：
    - 被扰动节点自身记为 0；
    - 其在 DAG 中的所有下游可达后代记为 1；
    - 其余节点记为 0。
    """
    adj: Dict[str, List[str]] = {n: [] for n in graph.nodes}
    for u, v in graph.edges:
        adj[u].append(v)

    descendants: Set[str] = set()
    stack = list(adj.get(perturbed_node, []))
    while stack:
        curr = stack.pop()
        if curr not in descendants:
            descendants.add(curr)
            stack.extend(adj.get(curr, []))

    return {
        node: (1 if (node != perturbed_node and node in descendants) else 0)
        for node in graph.nodes
    }


def evaluate_observation_match(
    graph: CausalGraph,
    observations: Sequence[Dict],
) -> Tuple[bool, float]:
    """
    校验因果图是否满足全部干预观测，同时返回观测匹配比例（0.0 ~ 1.0）。
    """
    if not observations:
        return True, 1.0

    matched_count = 0
    for obs in observations:
        predicted = compute_perturbation_effects(graph, obs["perturbed_node"])
        if predicted == obs["effects"]:
            matched_count += 1

    match_ratio = matched_count / len(observations)
    return matched_count == len(observations), match_ratio


def edge_jaccard_distance(g1: CausalGraph, g2: CausalGraph) -> float:
    """
    计算两张因果图边集合之间的 Jaccard 距离：`1 - |E1 ∩ E2| / |E1 ∪ E2|`。
    """
    e1 = set(g1.edges)
    e2 = set(g2.edges)
    union_size = len(e1 | e2)
    if union_size == 0:
        return 0.0
    return 1.0 - (len(e1 & e2) / union_size)


def compute_group_diversity_rewards(
    responses: Sequence[str],
    nodes: Sequence[str],
    observations: Sequence[Dict],
    max_edges: Optional[int] = None,
    cfg: Optional[RewardConfig] = None,
) -> GroupRewardResult:
    """
    计算 GRPO 单组采样响应的集合级奖励（v2.0：严格层级 + 排列不变逆频次衰减）。

    公式说明：
    - `diversity` 模式：对合法图 g，设其在组内出现次数为 c = count(g) >= 1，
      平均边 Jaccard 距离为 d_bar(g)（若组内仅有 1 种合法图则 d_bar=0）：
      r(g) = valid_floor_reward + (valid_unique_bonus + diversity_weight * d_bar(g)) / c
      注意：总组内合法奖励之和 = c * valid_floor + (valid_unique_bonus + diversity_weight * d_bar)
      既确保所有合法解严格高于所有非法解，又使“采样出 N 种不同合法解”比“重复采样同一合法解 N 次”
      获得高得多的组内与个体优势。
    - `validity_only` 模式：所有合法解均获得固定 `valid_floor_reward + valid_unique_bonus` (+1.0)。
    """
    reward_cfg = cfg if cfg is not None else RewardConfig()
    if reward_cfg.mode not in ("diversity", "validity_only"):
        raise ValueError(f"Unsupported reward mode: {reward_cfg.mode}")

    n_resp = len(responses)
    parsed_graphs: List[Optional[CausalGraph]] = []
    parse_flags: List[bool] = []
    valid_flags: List[bool] = []
    obs_match_ratios: List[float] = []
    unique_valid_flags: List[bool] = [False] * n_resp
    row_hashes: List[Optional[str]] = [None] * n_resp

    # 1) 解析与验证组内全部响应，统计各有效图哈希的出现频次
    valid_counts: Counter = Counter()
    unique_valid_graphs: Dict[str, CausalGraph] = {}

    for idx, text in enumerate(responses):
        graph = parse_causal_graph(text, nodes=nodes, max_edges=max_edges)
        parsed_graphs.append(graph)
        if graph is None:
            parse_flags.append(False)
            valid_flags.append(False)
            obs_match_ratios.append(0.0)
            continue

        parse_flags.append(True)
        is_valid, match_ratio = evaluate_observation_match(graph, observations)
        valid_flags.append(is_valid)
        obs_match_ratios.append(match_ratio)

        if is_valid:
            g_hash = graph.get_hash()
            row_hashes[idx] = g_hash
            valid_counts[g_hash] += 1
            if g_hash not in unique_valid_graphs:
                unique_valid_graphs[g_hash] = graph
                unique_valid_flags[idx] = True

    # 2) 计算每个唯一有效图相对组内其余唯一有效图的平均 Jaccard 距离
    diversity_by_hash: Dict[str, float] = {}
    unique_keys = list(unique_valid_graphs.keys())
    if len(unique_keys) <= 1:
        for k in unique_keys:
            diversity_by_hash[k] = 0.0
    else:
        for k_i in unique_keys:
            g_i = unique_valid_graphs[k_i]
            dists = [
                edge_jaccard_distance(g_i, unique_valid_graphs[k_j])
                for k_j in unique_keys
                if k_j != k_i
            ]
            diversity_by_hash[k_i] = float(np.mean(dists))

    # 3) 按严格层级与排列不变公式赋分
    rewards: List[float] = []
    for idx in range(n_resp):
        if not parse_flags[idx]:
            rewards.append(float(reward_cfg.parse_fail_reward))
        elif not valid_flags[idx]:
            partial = (
                reward_cfg.invalid_base_reward
                + reward_cfg.invalid_match_bonus * obs_match_ratios[idx]
            )
            rewards.append(float(partial))
        elif reward_cfg.mode == "validity_only":
            flat_valid = reward_cfg.valid_floor_reward + reward_cfg.valid_unique_bonus
            rewards.append(float(flat_valid))
        else:
            g_hash = row_hashes[idx]
            assert g_hash is not None
            freq = valid_counts[g_hash]
            div_bonus = reward_cfg.diversity_weight * diversity_by_hash[g_hash]
            r = reward_cfg.valid_floor_reward + (
                reward_cfg.valid_unique_bonus + div_bonus
            ) / float(freq)
            rewards.append(float(r))

    denom = max(1, n_resp)
    return GroupRewardResult(
        rewards=rewards,
        parsed_graphs=parsed_graphs,
        parse_flags=parse_flags,
        valid_flags=valid_flags,
        unique_valid_flags=unique_valid_flags,
        obs_match_ratios=obs_match_ratios,
        valid_rate=sum(valid_flags) / denom,
        unique_valid_rate=sum(unique_valid_flags) / denom,
        parse_rate=sum(parse_flags) / denom,
    )


def compute_grpo_advantages(
    rewards: Sequence[float],
    eps: float = 1e-6,
) -> List[float]:
    """
    计算 GRPO 组内标准化相对优势：
        A_i = (r_i - mean(r)) / (std(r) + eps)

    若整组奖励方差接近 0（所有样本表现完全相同），则直接返回全 0 优势，避免数值放大噪声。
    """
    if not rewards:
        return []
    arr = np.asarray(rewards, dtype=np.float64)
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr))
    if std_r < eps:
        return [0.0 for _ in rewards]
    adv = (arr - mean_r) / (std_r + eps)
    return [float(x) for x in adv]
