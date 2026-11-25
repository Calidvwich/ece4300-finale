"""
Causal 任务精确解析器、DAG 验证器与集合级多样性 GRPO 奖励模块。

设计说明：
1. 严格复用官方 `CausalGraph` 数据模型及干预可达性语义，保证训练奖励与最终评测零偏差。
2. 对每个样本群（Group of size G）计算集合级奖励（Set-Level Diversity Reward）：
   - 合法且唯一的新解：基础高分 + 与组内其他唯一有效解的平均边 Jaccard 距离奖励；
   - 合法但与组内已有解重复：明确惩罚，抑制模式崩塌；
   - 非法 DAG 或观测冲突：给予平滑或惩罚性负分，引导模型同时学到“格式/合法性”与“多样性”。
"""

import re
import sys
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
    计算 GRPO 单组采样响应的集合级多样性奖励。

    核心机制：
    - 解析出组内所有合法且互不重复的图集合 U_valid；
    - 对组内首次出现的每个唯一有效解，奖励基础分 + 结构新颖度分（与组内其余唯一解的平均 Jaccard 距离）；
    - 对组内后续重复出现的相同有效解，给予负惩罚（duplicate_valid_reward），迫使策略分散采样概率质量。
    """
    reward_cfg = cfg if cfg is not None else RewardConfig()
    n_resp = len(responses)

    parsed_graphs: List[Optional[CausalGraph]] = []
    parse_flags: List[bool] = []
    valid_flags: List[bool] = []
    obs_match_ratios: List[float] = []
    unique_valid_flags: List[bool] = [False] * n_resp

    # 先解析与验证所有响应，并记录每张有效图的首次出现下标
    seen_valid_hashes: Set[str] = set()
    unique_valid_items: List[Tuple[int, CausalGraph]] = []

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
            if g_hash not in seen_valid_hashes:
                seen_valid_hashes.add(g_hash)
                unique_valid_flags[idx] = True
                unique_valid_items.append((idx, graph))

    # 预先计算每个唯一有效解相对组内其他唯一有效解的平均结构距离
    diversity_scores: Dict[int, float] = {}
    if len(unique_valid_items) <= 1:
        for idx, _ in unique_valid_items:
            diversity_scores[idx] = 0.0
    else:
        for idx, g_i in unique_valid_items:
            dists = [
                edge_jaccard_distance(g_i, g_j)
                for other_idx, g_j in unique_valid_items
                if other_idx != idx
            ]
            diversity_scores[idx] = float(np.mean(dists))

    # 汇总每条响应的最终标量奖励
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
        elif unique_valid_flags[idx]:
            r = (
                reward_cfg.valid_unique_base
                + reward_cfg.diversity_weight * diversity_scores[idx]
            )
            rewards.append(float(r))
        else:
            rewards.append(float(reward_cfg.duplicate_valid_reward))

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
