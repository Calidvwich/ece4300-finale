# 3D Voxel Reconstruction Diversity-GRPO-LoRA 运行指南

本目录将两阶段 **Diversity-Aware GRPO-LoRA** 迁移到 HypoSpace 3D 体素重建任务，保留 **零 D4 同构泄露切分**、**统一采样参数（`T=0.9, top_p=0.95, max_new_tokens=128`）**、**排列不变逆频次奖励** 与 **`Validity-Only` (`best_policy` + `final_policy`) 双权重对照评测**。

---

## 1. 数据生成与 CPU 校验

所有命令建议在 `HypoSpace-release/` 下执行（Windows 本地使用 `D:\Environment\Python\DeepLearning\python.exe`，Linux 服务器使用 `.venv/bin/python`）：

```bash
# 1) 生成官方 3D 数据集（如尚未生成）
python 3d/generate_3d_dataset_complete.py --grid-size 3 --max-height 3 --max-blocks 3 --output 3d/datasets/3d_g3_h3_b1to3.json
python 3d/generate_3d_dataset_complete.py --grid-size 3 --max-height 3 --max-blocks 4 --fixed --output 3d/datasets/3d_g3_h3_b4.json

# 2) 生成零 D4 泄露切分 + 测试/OOD 子集，并运行单元测试
python 3d/grpo_lora/data_utils.py
python 3d/grpo_lora/tests/test_verifier_reward_3d.py
```

---

## 2. RTX 5090 服务器全自动 10 阶段运行

```bash
cd /opt/data/private/machine-learning/HypoSpace-release
chmod +x 3d/grpo_lora/run_server_all.sh

nohup env \
  PYTHON_BIN=.venv/bin/python \
  HF_HOME=/opt/data/private/machine-learning/hf_cache \
  RUN_TAG=qwen7b_3d \
  MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
  bash 3d/grpo_lora/run_server_all.sh \
  > run_qwen7b_3d.log 2>&1 &
```

### 常用可选开关
- `SKIP_VALIDITY=1`：显存或机时紧张时跳过第 5、6 步 `Validity-Only GRPO` 训练与评测。
- `OOD_MULTI_SEED=1`：让第 9 步 4-Block OOD 评测也跑满 `33550336 1 2` 三个种子（默认仅跑单种子 `33550336`）。
- `SKIP_BASE_EVAL=1`：若 Base ID 评测已生成完 3 份 JSON，中断续跑时可跳过第 2 步。

### 规模与 GPU 时间粗估（RTX 5090, Qwen2.5-7B-Instruct）
- **ID 测试集**（`3d_g3_h3_test.json`）：`38` 题，`822` 次查询/种子（3 种子 = `2,466` 查询/模型）；
- **OOD 测试集**（`3d_g3_h3_b4_ood.json`）：`23` 个 D4 代表题（`K=81`），`1,863` 次查询/种子；
- **训练步数**：Stage 1 SFT（`73` 题，`2` epochs ≈ `74` optimizer steps）；Stage 2 GRPO（`73` 题，`5` epochs ≈ `95` optimizer steps，`group_size=8`）；
- **预计耗时**（`Qwen2.5-7B`, `bfloat16`, `max_new_tokens=128`）：含 `Validity-Only` 与 `Diversity` 双组训练 + 多轮评测总计约 **1.5 ~ 2.5 小时**（设置 `SKIP_VALIDITY=1` 可压缩至 ~1.0 ~ 1.5 小时）。

---

## 3. DeepSeek-Flash Fixed-Prompt 基线命令（由用户在本地通过环境变量运行）

切勿将 API Key 写入任何脚本或配置。运行前请先设置环境变量 `HYPOSPACE_DEEPSEEK_API_KEY`，且注意官方 [config_deepseek.yaml](../causal/config/config_deepseek.yaml) 默认 `temperature: 0.7`：

```powershell
# PowerShell 设置 Key（当前会话有效）
$env:HYPOSPACE_DEEPSEEK_API_KEY = "<YOUR_KEY>"

# 1) 3D Unseen-D4 ID Test 全量（38 题，--n-samples 38，共 822 次 API 查询）
D:\Environment\Python\DeepLearning\python.exe HypoSpace-release\3d\run_3d_benchmark.py `
  --dataset HypoSpace-release\3d\datasets\3d_g3_h3_test.json `
  --config HypoSpace-release\3d\config\config_deepseek.yaml `
  --n-samples 38 `
  --query-multiplier 1.0 `
  --independent-samples `
  --seed 33550336 `
  --checkpoint-dir HypoSpace-release\3d\grpo_lora\checkpoints\deepseek_id `
  --output HypoSpace-release\3d\grpo_lora\results\deepseek_3d_id_fixed.json

# 2) 3D 4-Block OOD 全量（23 题，--n-samples 23，共 1,863 次 API 查询）
D:\Environment\Python\DeepLearning\python.exe HypoSpace-release\3d\run_3d_benchmark.py `
  --dataset HypoSpace-release\3d\datasets\3d_g3_h3_b4_ood.json `
  --config HypoSpace-release\3d\config\config_deepseek.yaml `
  --n-samples 23 `
  --query-multiplier 1.0 `
  --independent-samples `
  --seed 33550336 `
  --checkpoint-dir HypoSpace-release\3d\grpo_lora\checkpoints\deepseek_ood `
  --output HypoSpace-release\3d\grpo_lora\results\deepseek_3d_ood_b4_fixed.json
```

> **查询总数估计**：ID Test `822` 次 + OOD B4 `1,863` 次 = **共 `2,685` 次 DeepSeek-Flash API 调用**。

---

## 4. 回传结果后生成统计报告与图表

```powershell
D:\Environment\Python\DeepLearning\python.exe results_analysis\stats_3d.py `
  --model diversity=HypoSpace-release\3d\grpo_lora\results\qwen7b_3d_id_grpo_diversity_best_seed*.json `
  --model validity_final=HypoSpace-release\3d\grpo_lora\results\qwen7b_3d_id_grpo_validity_final_seed*.json `
  --model sft=HypoSpace-release\3d\grpo_lora\results\qwen7b_3d_id_sft_seed*.json `
  --model base=HypoSpace-release\3d\grpo_lora\results\qwen7b_3d_id_base_seed*.json `
  --deepseek_baseline HypoSpace-release\3d\grpo_lora\results\deepseek_3d_id_fixed.json `
  --output_md results_analysis\id_3d_stats.md `
  --output_json results_analysis\id_3d_stats.json
```
