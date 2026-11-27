# 3D Voxel Reconstruction GRPO-LoRA 交付说明（HANDOFF）

> 本文件汇总 `HypoSpace-release/3d/grpo_lora/` 及配套统计/可视化脚本的完整实现、D4 零同构泄露切分细节、与 Causal 版本的差异、已知限制及 RTX 4070 `Qwen2.5-0.5B-Instruct` 端到端冒烟验证结果。

---

## 1. 已交付文件清单（严格遵守第 0 部分路径归属）

| 路径 | 说明 |
|---|---|
| `HypoSpace-release/3d/datasets/3d_g3_h3_b1to3.json` | 官方生成器产出的 3×3×3、1–3 方块 ID 完整数据池（129 题，2,619 个结构） |
| `HypoSpace-release/3d/datasets/3d_g3_h3_b4.json` | 官方生成器产出的 3×3×3、固定 4 方块 OOD 原始池（126 题，10,206 个结构） |
| `HypoSpace-release/3d/datasets/3d_g3_h3_b4_ood.json` | 按 D4 等价类去重后的 4 方块 OOD 测试集（23 类各取首题，23 题，1,863 次查询） |
| `HypoSpace-release/3d/datasets/3d_g3_h3_test.json` | 与 `3d_splits.json` 对齐的官方格式 Unseen-D4 测试子集（供 DeepSeek 基线用） |
| `HypoSpace-release/3d/run_3d_benchmark.py` | 仅新增 `--independent-samples` 开关（默认关闭，行为与原版 100% 一致） |
| `HypoSpace-release/3d/grpo_lora/config.py` | 数据、统一采样、奖励、LoRA、SFT、GRPO 与评测配置 |
| `HypoSpace-release/3d/grpo_lora/verifier_reward.py` | 复用官方 3D 解析/验证逻辑 + 体素 Jaccard 逆频次集合多样性奖励 |
| `HypoSpace-release/3d/grpo_lora/data_utils.py` | D4 规范键、官方逐字匹配 `build_fixed_prompt` 与零泄露分层切分 |
| `HypoSpace-release/3d/grpo_lora/train_sft.py` | Stage 1 轮换覆盖去偏 SFT-LoRA |
| `HypoSpace-release/3d/grpo_lora/train_grpo.py` | Stage 2 Diversity/Validity-Only GRPO-LoRA（同时保存 `best_policy` 与 `final_policy`） |
| `HypoSpace-release/3d/grpo_lora/evaluate_lora.py` | 统一 Fixed-Prompt 评测（兼容 `adapter_dir/policy/` 嵌套目录） |
| `HypoSpace-release/3d/grpo_lora/run_server_all.sh` | 10 阶段 RTX 5090 全自动训练/消融/评测流水线（LF 换行） |
| `HypoSpace-release/3d/grpo_lora/RUN_GUIDE.md` | 服务器运行指令与 DeepSeek Fixed-Prompt 基线命令 |
| `HypoSpace-release/3d/grpo_lora/tests/test_verifier_reward_3d.py` | 8 项 CPU 单元测试（已全通过） |
| `results_analysis/stats_3d.py` | 仅依赖 `numpy` + stdlib 的 D4 聚类显著性与理论参考上界统计脚本 |
| `results_analysis/generate_charts_3d.py` | 基于 `stats_3d` JSON 与训练日志生成 3D 4 张 300 DPI 图表 |

*(未修改 `causal/**`、`boolean/**`、`results_analysis/group_stats.py`、`results_analysis/generate_charts.py`、`README*.md`、`WORKLOG.md` 或 `.gitignore`。)*

---

## 2. D4 零泄露切分与数据集统计（`seed=33550336`）

- **对称群与等价类**：3×3 俯视图经 D4（4 旋转 × 翻转，共 8 变换）规范化后，`3d_g3_h3_b1to3.json` 的 129 题精确归并为 **27 个 D4 等价类**：
  - `b=1`（3 个类，每题 $|H_O|=3$）
  - `b=2`（8 个类，每题 $|H_O|=9$）
  - `b=3`（16 个类，每题 $|H_O|=27$）
- **切分数量与 `|H_O|` 分布**（已断言 Train/Val/Test 与 OOD 之间 D4 规范键交集严格为空）：

| Split | D4 等价类数 | 题数 | 方块数分布 | `|H_O|` 分桶分布 | 评测查询数（$1\times|H_O|$） |
|---|---:|---:|---|---|---:|
| `train` | 16 | 73 | `b1: 1, b2: 28, b3: 44` | `2-4: 1, 9+: 72` | — |
| `val` | 4 | 18 | `b1: 4, b2: 2, b3: 12` | `2-4: 4, 9+: 14` | — |
| `test` | 7 | 38 | `b1: 4, b2: 6, b3: 28` | `2-4: 4, 9+: 34` | **822**（每种子） |
| `ood (b4)` | 23 | 23 | `b4: 23` | `9+: 23`（均 $K=81$） | **1,863**（每种子） |

- **GT Token 实测长度**（使用 `Qwen2.5-0.5B-Instruct` tokenizer，无截断风险）：
  - ID 池（`3d_g3_h3_b1to3.json`）最长 GT 输出为 **`67` tokens**；
  - OOD 池（`3d_g3_h3_b4_ood.json`）最长 GT 输出为 **`67` tokens**；
  - 统一设置 `SamplingConfig.max_new_tokens = 128`（约为最长 GT 的 1.9 倍），`temperature = 0.9`、`top_p = 0.95` 在 rollout、验证、评测三处完全一致。

---

## 3. 与 Causal 版本的关键差异

1. **无序增强移除**：3D 层结构（`Layer 1 -> Layer 2 -> Layer 3`）具有唯一固定物理层序，SFT 目标不做随机打乱。
2. **评测 Validity vs. 训练奖励高度校验**：
   - **评测阶段**：`evaluate_lora.py` 的 `valid_rate` 严格调用从官方 `run_3d_benchmark.py:308-358` 复制的 `validate_structure_matches_observations`（俯视图一致 + 物理支撑，不限制最大层数 `max_height`），以保证与 `DeepSeek-Flash` 基线 100% 同口径；
   - **训练奖励**：`compute_group_diversity_rewards` 在进入部分匹配前额外检查 `1 <= norm.height <= max_height`、网格尺寸一致且 Layer 1 非全空，越界样本直接给 `-1.0`。
3. **新增 `final_policy/` 评测**：`train_grpo.py` 与 `run_server_all.sh` 对 `Validity-Only` 与 `Diversity` 两组均同时保留并评测 `best_policy/` 与 `final_policy/`（含 OOD 对 `validity_final` 的评测），便于在 `best_step == 0` 时仍可直接观测末步权重是否发生模式坍缩。
4. **GRPO Epochs 调整**：因 3D 训练集仅 73 题（Causal 为 275 题中取 200+ 多解题），`GRPOConfig.epochs` 默认由 2 调至 **5**（每 epoch `19` 步，总计 **95** 个优化步，接近 Causal 的 102 步）。
5. **代码独立性**：未导入 `HypoSpace-release/3d/modules/llm_interface.py`（因其依赖未安装的 `openai`），而是将 `Structure3D`、`parse_top_view`、`validate_structure_matches_observations`、`parse_llm_response` 原样复制至 `verifier_reward.py` 并注明行号来源。

---

## 4. 已知限制（供最终整合与汇报披露）

1. **采样温度口径差异**：官方 `3d/config/config_deepseek.yaml`（引用自 Causal/3D 默认配置）使用 `temperature: 0.7`，而 Qwen 系列统一采用 `T = 0.9, top_p = 0.95`。
2. **`|H_O|` 分布偏重 `9+`**：由于 3×3×3 空间在 $b \ge 2$ 时即有 $|H_O| \in \{9, 27\}$，切分中 `1` 和 `5-8` 两个分桶均为 0 题（`b1 -> K=3` 落入 `2-4` 桶，`b2/b3 -> K=9/27` 落入 `9+` 桶），ID Test 共 7 个 D4 类。
3. **OOD `b4` 每类仅取 1 题**：为控制评测总查询预算，4 方块 OOD 从 126 题按 D4 类去重至 23 个代表题（1,863 次查询），默认运行单评测种子 `33550336`。

---

## 5. 验证与 RTX 4070 `0.5B` 端到端冒烟摘要

1. **单元测试 & 语法校验**：
   - `HypoSpace-release/3d/grpo_lora/tests/test_verifier_reward_3d.py`：**8/8 全部通过**（`0.019s`）；
   - 所有新增/修改 `.py` 文件 `py_compile` 通过；`run_server_all.sh` 为 LF 换行且 `bash -n` 通过。
2. **RTX 4070 `Qwen2.5-0.5B-Instruct` 端到端冒烟测试**（0 报错、0 NaN）：
   - **Stage 1 SFT（10 steps）**：`loss` 从 `0.4061` 收敛至 `0.0813`，覆盖 53.32% GT；
   - **Stage 2 Diversity-GRPO（4 steps）**：`val_valid` 从 `0.778` 升至 `0.870`，`val_group_rec` 从 `0.333` 升至 `0.370`（`best_step=4`），Step 1/2 `kl=0.0000`；
   - **Stage 2 Validity-Only GRPO（4 steps）**：`val_valid` 从 `0.778` 升至 `0.880`，`val_group_rec` 为 `0.352`；
   - **5 题 Test 子集冒烟评测 + `stats_3d.py` 读取验证**：
     - `sft` (`smoke_eval_sft.json`)：Parse=`100.00%`, Valid=`77.78%`, Uniq=`57.78%`, Recovery=`57.78%`；
     - `diversity best` (`smoke_eval_grpo_diversity.json`)：Parse=`100.00%`, Valid=`77.78%`, Uniq=`57.78%`, Recovery=`57.78%`；
     - `validity_final` (`smoke_eval_grpo_validity_final.json`)：Parse=`100.00%`, Valid=`80.00%`, Uniq=`57.78%`, Recovery=`57.78%`；
     - `results_analysis/stats_3d.py` 成功对齐 5 题 / 2 个 D4 组并输出 Markdown + JSON 报告。

---

## 6. 建议集成步骤加入 `.gitignore` 的规则

为防止后续提交把冒烟产物、checkpoint 或大日志推上仓库，建议在集成阶段向 `.gitignore` 补充：

```gitignore
HypoSpace-release/3d/grpo_lora/outputs/
HypoSpace-release/3d/grpo_lora/checkpoints/
HypoSpace-release/3d/grpo_lora/results/smoke_*
HypoSpace-release/3d/grpo_lora/HANDOFF.md
```
