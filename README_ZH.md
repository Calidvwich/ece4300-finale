# EEC 4300 课程大作业 — 基于两阶段 Diversity-Aware GRPO-LoRA 突破 HypoSpace 欠定假设覆盖瓶颈

> **语言**: [English](README.md) | **简体中文**

本仓库为 **EEC 4300** 课程大作业项目，基于 **HypoSpace**（*ICML 2026*, [PMLR Vol. 306](https://proceedings.mlr.press/v306/chen26bz.html)）基准。该基准将大语言模型（LLM）视为**欠定科学假设空间**上的采样器，核心评估三大指标：**Validity（有效性）**、**Uniqueness（去重率）** 与 **Recovery（覆盖率）**。

---

## 1. 核心问题与动机

在 HypoSpace 官方 **Fixed-Prompt** 独立采样协议（`--independent-samples`，每题预算 $N = |H_O|$，且不将历史输出塞进提示词）下，即使是强闭源大模型也会遭遇严重的**模式崩塌（Mode Collapse）与次线性覆盖（Sublinear Coverage）**：即随着假设空间 $|H_O|$ 增大，模型实际覆盖的有效解个数相对 $|H_O|$ 呈次线性增长，导致 Recovery 随问题歧义度上升而快速衰减（见图 2）。

- 在 4 节点未见结构测试集上，队友预先录得的 **`DeepSeek-Flash`** 闭源基线（`T = 0.7`，从 392 题全量结果 [full_repo_causal_n4_obs3_deepseek.json](HypoSpace-release/causal/benchmark_obs_count/full_repo_causal_n4_obs3_deepseek.json) 按 76 题对齐截取）**Validity 高达 100.00%**，但 **Macro Recovery 跌至 38.34%**，在 $|H_O| \ge 9$ 难题上更是降至 **18.52%**；
- 标准监督微调（**SFT-LoRA**）贡献了相对 `DeepSeek-Flash` 的大部分 Recovery 原始提升，但缺少集合级去重反馈；而仅奖励合法性的 **Validity-Only GRPO** 消融组在 RL 过程中合法率飙升（`77.52% → 96.27%`），验证集组均值覆盖率却单调下滑（`55.82% → 52.17%`），直接印证了无多样性惩罚时的模式坍缩问题。

为此，我们在 `Qwen2.5-7B-Instruct` 上建立**零同构泄露切分**与 Qwen 统一采样口径（`T = 0.9, top_p = 0.95`），提出 **Two-Stage Diversity-Aware GRPO-LoRA**，在不依赖 Prompt 历史技巧、严禁验证器特征泄露的前提下重塑模型内部分布。

---

## 2. 核心实测结果（v2.1 零同构泄露全量评测）

### 2.1 零同构泄露切分与有限采样参考上界

`n4_obs3_all.json` 的 392 题经节点全排列群 $S_4$ 规范签名（[data_utils.py](HypoSpace-release/causal/grpo_lora/data_utils.py) 中的 `canonical_isomorphism_key`）归并后，实质上仅由 **22 个因果观测同构类**构成。我们以**同构类为最小不可分割单元**分层整组切分为：
- **Train**：13 组（`256` 题 4 节点 + `19` 题 3 节点 = `275` 题）
- **Val**：4 组（`60` 题；验证时按每组 12 题等权抽样 `48` 题）
- **Unseen-Structure ID Test**：5 组（`76` 题，共 `400` 次独立查询，与训练集零节点换名同构重叠）

在固定预算 $K = |H_O|$ 独立均匀采样下，**i.i.d. 均匀采样下的期望 Recovery 上界**（*finite-sample reference ceiling*）为 $1 - (1 - 1/K)^K$（ID 测试集均值 `70.38%`；5 节点 OOD 为 `81.41%`）。需注意这是独立均匀采样的期望参考上界而非绝对上限，单题或小组的归一化 Recovery 可超过 100%（例如 OOD `G15` 达 `133.33%`、`G09` 达 `109.71%`，见 [ood_n5_v2_stats.md](results_analysis/ood_n5_v2_stats.md)）。我们通过 [group_stats.py](results_analysis/group_stats.py) 同步汇报**原始指标**、**参考上界归一化覆盖率**（`recovery_norm`）、**同构结构组均值**与 **10,000 次聚类自助法 95% 置信区间**。

### 2.2 4 节点未见结构 ID 测试集（76 题 / 5 结构组）

每个开源方法均**只训练 1 次**；下表 Qwen 结果为 **3 个评测采样种子**（`33550336, 1, 2`，`T = 0.9, top_p = 0.95`）的平均，反映推理采样方差而非训练种子方差；`DeepSeek-Flash` 为队友已有 392 题完整运行（`T = 0.7`，见 [config_deepseek.yaml](HypoSpace-release/causal/config/config_deepseek.yaml)）按同一批 76 题对齐截取。

| 模型 / 方法 | 评测来源 | Parse | Macro Valid | **Macro Rec (= Uniq)** | Task Norm Rec | **Group Rec (mean ± std)** | Group Norm Rec | **Pooled Rec** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen2.5-7B Base`（Zero-shot, `T=0.9`） | 3 评测种子 | 100.00% | 29.47% | 11.29% | 14.97% | 18.34% ± 22.45% | 21.26% ± 20.89% | 8.42% |
| `DeepSeek-Flash`（闭源 API, `T=0.7`） | 1 次已有运行 | **100.00%** | **100.00%** | 38.34% | 52.47% | 47.80% ± 33.61% | 59.40% ± 31.47% | 27.00% |
| `SFT-LoRA`（`= Validity-GRPO best`）* | 3 评测种子 | 97.69% | 86.71% | 55.63% | 78.70% | 61.81% ± 17.15% | 81.26% ± 6.98% | 52.33% |
| **`Diversity-GRPO-LoRA`（本文, `T=0.9`）** | 3 评测种子 | 98.42% | **91.91%** | **58.38%** | **82.50%** | **64.37% ± 16.88%** | **84.78% ± 8.10%** | **53.50%** |

*\*注：（1）因为每题查询预算 $N = |H_O|$ 且 `ground_truth_graphs` 为穷举全集，故 `Uniqueness` 与 `Recovery` 数值完全相等。（2）`Validity-Only GRPO` 的模式坍缩证据来自验证曲线（图 4 中图：`val group recovery` 从 `55.82%` 跌至 `52.17%`），因触发 `best_step == 0` 保护退回，该行测试分数即 SFT 初始化权重，未在测试集上评测其最终 checkpoint。*

- **增益归因与聚类显著性检验（见 [id_v2_stats.md](results_analysis/id_v2_stats.md)）**：
  - **增益归因**：在 4 节点 ID 测试集上，相对 `DeepSeek-Flash` 的 `+20.04%` 提升中，**Stage 1 SFT 已贡献 `+17.29%`**（Group `+14.00%`）。Stage 2 在 Recovery 上相对 SFT 多提升 `+2.75%`（Group `+2.56%`，3 个评测种子 `57.75% vs 54.20%`、`58.13% vs 54.33%`、`59.27% vs 58.35%` 方向一致），但在 $n = 5$ 个结构组下组级 95% CI `[-0.31%, +5.43%]` 跨 0；在 **Validity** 上 Stage 2 提升显著（Task `+5.20%`、Group `+4.46%`，95% CI **`[+0.42%, +8.73%]`**）。Stage 2 多样性优势更有力的证据见下文 22 组 OOD 5 节点结果。
  - `Diversity-GRPO-LoRA` vs. `DeepSeek-Flash`：Task Macro Recovery **+20.04%**（57 胜 / 11 平 / 8 负），Group Recovery **+16.57%**（聚类自助法 95% CI **`[+2.71%, +29.52%]`**），归一化组恢复率 **+25.38%**（CI `[+6.27%, +44.49%]`）。

#### 按假设空间难度 `|H_O|` 分桶对比（Macro Recovery）

| 难度分桶 (`\|H_O\|`) | 题数（组数） | 参考上界 Ceiling | `Base 7B` | `DeepSeek-Flash` | `SFT-LoRA` | **`Diversity-GRPO-LoRA`** | 归一化恢复率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `1`（唯一解） | 4 (1) | 100.00% | 58.33% | **100.00%** | 91.67% | **91.67%** | 91.67% |
| `2–4` | 36 (2) | 71.91% | 10.03% | 51.85% | 57.10% | **63.43%** | **88.22%** |
| `5–8` | 24 (1) | 65.64% | 8.51% | 17.71% | **47.92%** | 46.53% | 70.88% |
| **`9+`（最难大空间）** | 12 (1) | 65.36% | 4.94% | 18.52% | 54.63% | **55.86%** | **85.48%** |

*在 `9+` 分桶（仅 `1` 个结构组、`12` 题）上，**SFT（`54.63%`）与 Diversity-GRPO（`55.86%`）均约为 `DeepSeek-Flash`（`18.52%`）的 3 倍**。*

<p align="center">
  <img src="results_analysis/figures/fig1_id_ablation.png" width="49%" alt="未见结构 ID 四方消融"/>
  <img src="results_analysis/figures/fig2_difficulty_scaling.png" width="49%" alt="复杂度分桶与参考上界"/>
</p>

### 2.3 跨节点规模 OOD 5 节点泛化测试（80 题 / 22 个全新 5 节点结构组）

在训练从未见过的 5 节点因果系统上使用**单个评测采样种子（`33550336`，`T = 0.9, top_p = 0.95`）**零样本迁移评测，覆盖 [n5_obs5_hybrid80.json](HypoSpace-release/causal/datasets/n5_obs5_hybrid80.json)（80 题，22 结构组，参考上界 `81.41%`）与 [causal_n5_gt160.json](HypoSpace-release/causal/benchmark_obs_count/paper_scale/causal_n5_gt160.json)（7 题，$|H_O|=160$）：

| 数据集 | 模型 | Macro Valid | **Macro Rec (= Uniq)** | **Group Rec (mean ± std)** | Group Norm Rec | **Pooled Rec** | `Bucket 9+` Rec |
|---|---|---:|---:|---:|---:|---:|---:|
| `n5_obs5_hybrid80` (80 题) | `Qwen2.5-7B Base` | 28.14% | 26.58% | 17.78% ± 33.72% | 18.71% ± 33.90% | 1.69% | 0.07% |
| `n5_obs5_hybrid80` (80 题) | `SFT-LoRA`（`= Validity-GRPO*`） | 67.54% | 56.97% | 50.92% ± 28.47% | 61.69% ± 26.17% | 28.28% | 26.43% |
| `n5_obs5_hybrid80` (80 题) | **`Diversity-GRPO-LoRA`（本文）** | **70.33%** | **61.70%** | **63.21% ± 26.63%** | **78.17% ± 24.67%** | **31.38%** | **29.28%** |
| `causal_n5_gt160` (7 题, `\|H_O\|=160`) | **`Diversity-GRPO-LoRA`（本文）** *(仅本文方法，无基线，规模压力测试参考)* | 19.29% | 10.71% | — | — | 10.71% | 10.71% |

- **22 个结构组跨组显著（单评测种子，见 [ood_n5_v2_stats.md](results_analysis/ood_n5_v2_stats.md)）**：
  - `Diversity-GRPO-LoRA` 全面超越 `SFT-LoRA`：Task Macro Recovery **+4.73%**、Pooled Recovery **+3.10%**；结构组均值 Recovery 提升 **+12.29% ± 24.88%**（**12 胜 / 8 平 / 2 负**，精确双侧二项符号检验 **$p = 0.0129$**，聚类自助法 95% CI **`[+3.50%, +23.68%]`**），Group Norm Recovery 提升 **+16.49%**（95% CI **`[+5.01%, +31.49%]`**），并在所有多解分桶全面领先（`2–4`：`63.24% vs 47.06%`，`5–8`：`47.92% vs 41.67%`，`9+`：`29.28% vs 26.43%`）。需注意 22 组中多数（11 组）仅有 1 题，而 `G22` 一组占 19 题。
  - **单解分桶（`|H_O|=1`）反例**：在最大的单解组 `G02`（13 题）上，`Diversity-GRPO` 得分 **76.92%** 低于 `SFT` 的 **92.31%**；在 `G03`（`100.00% vs 66.67%`）与 `G04`（`100.00% vs 75.00%`）上的增益抵消后，两者的 `|H_O|=1` 桶均值持平（均为 `85.29%`）。
  - 相比 `Base 7B`：Group Recovery 提升 **+45.43%**（19 胜 / 3 平 / 0 负，精确符号检验 **$p < 10^{-4}$**，95% CI `[+33.81%, +57.77%]`）。

<p align="center">
  <img src="results_analysis/figures/fig3_ood_transfer.png" width="54%" alt="OOD 5 节点迁移对比"/>
  <img src="results_analysis/figures/fig4_training_curves.png" width="44%" alt="训练与消融动态曲线"/>
</p>

---

## 3. 方法架构（v2.1 核心设计）

1. **Stage 1 — 轮换全覆盖去偏 SFT-LoRA Warm-up**（[train_sft.py](HypoSpace-release/causal/grpo_lora/train_sft.py)）：零样本 7B 合法率仅 29.47%，直接做 RL 易因全错陷入零方差梯度消失。Stage 1 采用 Prompt Loss Masking、边序随机打乱以及保持每题预算不变（$\min(K, 8)$）的 **Epoch 循环轮换抽样**，2 个 Epoch 即完成全部 275 题 `1,353` 个标准答案 **100.00% 覆盖**，既奠定稳健冷启动起点，又构成严格对齐的监督消融基线。
2. **Stage 2 — 排列不变逆频次 Diversity-GRPO-LoRA**（[train_grpo.py](HypoSpace-release/causal/grpo_lora/train_grpo.py)、[verifier_reward.py](HypoSpace-release/causal/grpo_lora/verifier_reward.py)）：在统一采样参数（`T = 0.9, top_p = 0.95, max_new_tokens = 64`）下每题采样 $G=8$ 个候选图，采用**严格分层且 100% 排列不变的逆频次集合奖励**：
   $$r_i = \begin{cases}
   -1.0, & \text{无法解析 / 含环 / 越界节点或超边数上限} \\
   -0.7 + 0.3 \times \text{ObsMatchRatio}(g_i, O), & \text{合法 DAG 但未满足全部观测 } \in [-0.7, -0.4] \\
   0.1 + \frac{0.9 + 0.5 \cdot \bar{d}_{\text{Jaccard}}(g_i, \mathcal{U}\setminus\{g_i\})}{\text{count}(g_i)}, & \text{满足全部观测的合法图 } (\ge +0.1)
   \end{cases}$$
   工程上启用 **Zero-Dropout**（实测首步 KL 精确为 `0.0000`）、多题梯度累积（`4` 题/步）、Cosine LR（`1e-5`）、同基座双适配器无偏 KL（$\beta = 0.06$）及 Step 0 SFT 保底 + 结构组等权严格 `>` 选优。

---

## 4. Limitations（局限性说明）

- **4 节点测试集同构组数量有限**：严格消除置换同构泄露后，4 节点 ID 测试集仅剩 `5` 个同构结构组（`76` 题），导致 ID Recovery 相对 SFT 的组级统计检验功效受限（`[-0.31%, +5.43%]` 跨 0）。
- **单次训练运行与单种子 OOD 评测**：所有微调模型仅完整训练 1 次，未度量不同训练种子的方差；4 节点 ID 的 3 个种子（`33550336, 1, 2`）仅反映推理采样方差，5 节点 OOD 则仅使用单评测种子（`33550336`）。
- **验证选优存在波动且策略偏移温和**：`best_policy` 在 `step 20` 选出（组均值 `57.99%`），与末步 `step 102`（`57.81%`）仅差 `0.18pp`（基于 48 题 / 4 组验证集）；训练过程单步最大 KL 散度为 `0.020`（10 步滑动均值约 `0.011`），策略整体仍贴近 SFT 起点。
- **未在测试集评测 `Validity-Only` 末步权重**：`Validity-Only GRPO` 在验证集出现模式坍缩（`55.82% → 52.17%`，见图 4）后触发 `best_step == 0` 保底回退，未对其最终坍缩 checkpoint 单独运行测试集评测。
- **闭源对比基线温度不同**：队友已有 `DeepSeek-Flash` 评测采样温度为 `0.7`，而 Qwen 系列统一采用 `T = 0.9, top_p = 0.95`。
- **`causal_n5_gt160` 缺少基线对照**：该 7 题 `|H_O|=160` 子集仅运行了本文 `Diversity-GRPO-LoRA`，仅作超大空间压力测试参考。

---

## 5. 仓库结构与快速复现

```text
.
├── README.md / README_ZH.md                   # 中英文双语全局说明
├── results_analysis/
│   ├── group_stats.py                         # 同构结构组统计、参考上界与聚类自助法显著性脚本
│   ├── generate_charts.py                     # 基于 v2.1 结果直接重绘 4 张 300 DPI 图表
│   ├── id_v2_stats.md / .json                 # 4 节点未见结构 ID 统计报告
│   ├── ood_n5_v2_stats.md / .json             # 5 节点 OOD 统计报告
│   └── figures/                               # fig1_id_ablation.png .. fig4_training_curves.png
├── qwen7b_v2_results_and_logs/                # v2.1 零同构泄露全套 5090 评测 JSON 与训练日志
├── qwen7b_results_and_logs/                   # v1.0 历史基线留档
└── HypoSpace-release/causal/grpo_lora/        # 两阶段 Diversity-GRPO-LoRA 源码与 10 阶段流水线
```

```bash
# 1. 单元测试与零同构泄露切分验证（CPU，约 1 秒）
python HypoSpace-release/causal/grpo_lora/tests/test_verifier_reward.py
python HypoSpace-release/causal/grpo_lora/data_utils.py

# 2. 重新生成 ID 与 OOD 结构组统计报告，以及全套 4 张图表
python results_analysis/group_stats.py --model diversity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_grpo_diversity_seed*.json --model validity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_grpo_validity_seed*.json --model sft=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_sft_seed*.json --model base=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_unseen_base_seed*.json --deepseek_baseline HypoSpace-release/causal/benchmark_obs_count/full_repo_causal_n4_obs3_deepseek.json --splits_path HypoSpace-release/causal/grpo_lora/splits/causal_splits.json --pairs diversity:sft --pairs diversity:validity --pairs diversity:deepseek --pairs sft:deepseek --pairs diversity:base --pairs sft:base --output_md results_analysis/id_v2_stats.md --output_json results_analysis/id_v2_stats.json

python results_analysis/group_stats.py --model diversity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_ood_n5_grpo_diversity_seed33550336.json --model validity=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_ood_n5_grpo_validity_seed33550336.json --model sft=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_ood_n5_sft_seed33550336.json --model base=qwen7b_v2_results_and_logs/causal/grpo_lora/results/qwen7b_v2_ood_n5_base_seed33550336.json --dataset_path HypoSpace-release/causal/datasets/n5_obs5_hybrid80.json --pairs diversity:sft --pairs diversity:validity --pairs diversity:base --pairs sft:base --output_md results_analysis/ood_n5_v2_stats.md --output_json results_analysis/ood_n5_v2_stats.json

python results_analysis/generate_charts.py

# 3. RTX 5090 / 4090 全量 10 阶段训练与多种子评测流水线（位于 HypoSpace-release/）
bash causal/grpo_lora/run_server_all.sh
```
完整服务器部署命令请参阅 [RUN_GUIDE.md](HypoSpace-release/causal/grpo_lora/RUN_GUIDE.md)。
