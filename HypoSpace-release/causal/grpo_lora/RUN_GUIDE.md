# Causal Diversity-Aware GRPO-LoRA 启动指南

本目录实现了针对 HypoSpace Causal 任务的 **两阶段微调与评测框架**：
1. **Stage 1（`train_sft.py`）**：轻量 SFT-LoRA 冷启动，同时产出重要消融对照基线 `SFT-LoRA`；
2. **Stage 2（`train_grpo.py`）**：**Diversity-Aware GRPO-LoRA**，利用集合级多样性奖励（Unique Valid 基础分 + 边结构 Jaccard 新颖度 − 重复惩罚）抑制模式崩塌，直接优化 Recovery 与 Uniqueness；
3. **统一评测（`evaluate_lora.py`）**：严格按 Fixed-Prompt（`N = |H_O|`）协议评测，并自动与队友保存的 `DeepSeek-Flash` 结果做逐题配对对比。

---

## 1. 环境准备

所有命令均建议在 `HypoSpace-release/` 目录下运行。

### A. 本地 Windows（`DeepLearning` 环境，RTX 4070 8GB 用于冒烟测试）
```powershell
D:\Environment\Python\DeepLearning\python.exe -m pip install -r causal/grpo_lora/requirements_lora.txt
```

### B. 远程 Linux 服务器（RTX 5090 32GB / RTX 4090 24GB 用于全量训练）
```bash
pip install -r causal/grpo_lora/requirements_lora.txt
```
> 若服务器处于国内网络环境拉取 HuggingFace 模型较慢，可提前设置镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

---

## 2. Step 0：运行单元测试 & 生成无泄露数据集切分

运行单元测试（零显存需求，几秒完成）：
```powershell
D:\Environment\Python\DeepLearning\python.exe causal/grpo_lora/tests/test_verifier_reward.py
```

生成严格按 `observation_set_id` 隔离的 `Train / Val / Test` 切分，并打印配对的 `DeepSeek-Flash` 基准分数：
```powershell
D:\Environment\Python\DeepLearning\python.exe causal/grpo_lora/data_utils.py
```
输出文件：`causal/grpo_lora/splits/causal_splits.json`。

---

## 3. 本地 RTX 4070 (8GB) 端到端冒烟测试（可选）

使用 `Qwen/Qwen2.5-0.5B-Instruct` 快速验证 Stage 1 + Stage 2 + 评测脚本全链路零报错：

```powershell
# 1) Stage 1 SFT-LoRA 冒烟（20 步）
D:\Environment\Python\DeepLearning\python.exe causal/grpo_lora/train_sft.py `
  --model_name Qwen/Qwen2.5-0.5B-Instruct `
  --max_steps 20 --batch_size 2 --grad_accum 2 `
  --output_dir causal/grpo_lora/outputs/smoke_sft

# 2) Stage 2 Diversity-GRPO-LoRA 冒烟（10 步）
D:\Environment\Python\DeepLearning\python.exe causal/grpo_lora/train_grpo.py `
  --model_name Qwen/Qwen2.5-0.5B-Instruct `
  --sft_adapter_dir causal/grpo_lora/outputs/smoke_sft `
  --max_steps 10 --group_size 4 --eval_every_steps 10 --val_max_tasks 5 `
  --output_dir causal/grpo_lora/outputs/smoke_grpo

# 3) 评测冒烟（测 10 题）
D:\Environment\Python\DeepLearning\python.exe causal/grpo_lora/evaluate_lora.py `
  --model_name Qwen/Qwen2.5-0.5B-Instruct `
  --adapter_dir causal/grpo_lora/outputs/smoke_grpo/final_policy `
  --max_tasks 10 `
  --output_path causal/grpo_lora/results/smoke_eval.json
```

---

## 4. 服务器一键上传与全量训练（RTX 5090 / 4090）

### 第一步：解压代码并选中 RTX 5090 专用 Python 环境
5090 服务器镜像内存在两个 Python 环境，**切勿直接使用默认 `python3` / `pip`**，必须使用专为 5090 编译了 `torch 2.12.1+cu130` 的解释器：`/opt/envs/torch5090/bin/python`。

```bash
unzip hypo_grpo_server_bundle.zip
cd HypoSpace-release

# 1) 校验 5090 环境（预期输出 No broken requirements found 与 2.12.1+cu130 / 13.0 / RTX 5090 / 128.0）
PY=/opt/envs/torch5090/bin/python
"$PY" -m pip check
"$PY" -c 'import torch; x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), (x @ x)[0, 0].item())'

# 2) 安装 LoRA 依赖（建议直接用 --system-site-packages 建持久盘 .venv，完整继承 5090 torch 且不污染原环境）
"$PY" -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r causal/grpo_lora/requirements_lora.txt

# 3) 再次校验依赖与 5090 CUDA 未被覆盖破坏
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch, transformers, peft; x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16); print(torch.__version__, torch.cuda.get_device_name(0), transformers.__version__, peft.__version__, (x @ x)[0, 0].item())'

chmod +x causal/grpo_lora/run_server_all.sh
```

> **持久盘与网络提示**：
> - 若系统盘空间有限，可将模型缓存目录指到持久盘/数据盘（使模型权重与 Python 依赖分离）：
>   `export HF_HOME=/你的持久盘路径/hf_cache`
> - 若服务器访问 HuggingFace 官方源较慢，可保留脚本默认的 `HF_ENDPOINT=https://hf-mirror.com`；若直连海外更快，运行前执行 `export HF_ENDPOINT=https://huggingface.co`。
> - `run_server_all.sh` 会自动按 `/opt/envs/torch5090/bin/python` → `.venv/bin/python` → `python3` 顺序选择解释器；若你是通过上述 `.venv` 安装的依赖，请显式传入 `PYTHON_BIN=.venv/bin/python`（若直接装进了 `/opt/envs/torch5090`，则无需传）。

### 第二步：一条命令跑完“基线 → SFT → GRPO → ID/OOD 四方对比”（推荐挂在 `tmux` 中运行）

- **RTX 5090 (32GB) 全量运行（默认 `Qwen2.5-7B-Instruct`, `GROUP_SIZE=8`）**：
  ```bash
  tmux new -s grpo
  PYTHON_BIN=.venv/bin/python bash causal/grpo_lora/run_server_all.sh 2>&1 | tee server_run_qwen7b.log
  ```

- **RTX 4090 (24GB) 推荐运行（`Qwen2.5-3B-Instruct`, `GROUP_SIZE=8`；若跑 7B 请设 `GROUP_SIZE=4`）**：
  ```bash
  tmux new -s grpo
  PYTHON_BIN=.venv/bin/python MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" GROUP_SIZE=8 RUN_TAG="qwen3b" \
    bash causal/grpo_lora/run_server_all.sh 2>&1 | tee server_run_qwen3b.log
  ```

> 脚本第 `[0/8]` 步会自动先跑一遍 `torch 5090 bfloat16` 与依赖自检，通过后依次执行：① 单元测试 → ② Base 模型评测 → ③ Stage 1 SFT-LoRA 训练 → ④ SFT-LoRA 评测 → ⑤ Stage 2 GRPO-LoRA 训练 → ⑥ GRPO-LoRA ID 测试集评测（自动配对 DeepSeek-Flash）→ ⑦ 5 节点 `n5_obs5_hybrid80` OOD 评测 → ⑧ 5 节点 `|H_O|=160` 高难 OOD 评测。

### （可选）手动分步运行命令

- **RTX 5090 (32GB)** 推荐：`Qwen/Qwen2.5-7B-Instruct`
- **RTX 4090 (24GB)** 推荐：`Qwen/Qwen2.5-3B-Instruct`（若跑 7B 建议将 `--group_size` 设为 `4` 或 `6`）

### Step 1：Stage 1 轻量 SFT-LoRA 训练
```bash
python causal/grpo_lora/train_sft.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --epochs 2 --batch_size 4 --grad_accum 4 --lr 2e-4 \
  --output_dir causal/grpo_lora/outputs/qwen7b_sft
```

### Step 2：Stage 2 Diversity-Aware GRPO-LoRA 训练
```bash
python causal/grpo_lora/train_grpo.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --sft_adapter_dir causal/grpo_lora/outputs/qwen7b_sft \
  --epochs 2 --group_size 8 --lr 2e-5 --kl_coef 0.04 \
  --diversity_weight 0.5 --duplicate_penalty -0.3 \
  --eval_every_steps 50 --val_max_tasks 39 \
  --output_dir causal/grpo_lora/outputs/qwen7b_grpo
```
训练过程日志将实时写入 `causal/grpo_lora/outputs/qwen7b_grpo/grpo_metrics.jsonl`。

### Step 3：四方消融对比评测（ID Test + 5 节点 OOD Test）

```bash
# 1) Base Model (Zero-shot)
python causal/grpo_lora/evaluate_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --output_path causal/grpo_lora/results/id_base.json

# 2) Stage 1 SFT-LoRA（监督微调对照）
python causal/grpo_lora/evaluate_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir causal/grpo_lora/outputs/qwen7b_sft \
  --output_path causal/grpo_lora/results/id_sft.json

# 3) Stage 2 Diversity-GRPO-LoRA（本文主方法，自动打印与 DeepSeek-Flash 的配对对比）
python causal/grpo_lora/evaluate_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir causal/grpo_lora/outputs/qwen7b_grpo/best_policy \
  --output_path causal/grpo_lora/results/id_grpo.json

# 4) Out-of-Distribution：5 节点大空间泛化测试（训练从未见过 5 节点图）
python causal/grpo_lora/evaluate_lora.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir causal/grpo_lora/outputs/qwen7b_grpo/best_policy \
  --dataset_path causal/datasets/n5_obs5_hybrid80.json \
  --output_path causal/grpo_lora/results/ood_n5_grpo.json
```

---

## 5. 输出说明与常见问题排查

- **输出文件**：
  - `splits/causal_splits.json`：无泄露数据切分
  - `outputs/<run>/grpo_metrics.jsonl`：每步 reward / valid_rate / unique_valid_rate / KL / 验证集指标（可直接用于画训练曲线）
  - `results/id_*.json`：包含 `summary.macro`、`summary.pooled`、`summary.by_gt_bucket`（按 `|H_O|` 分桶）以及 `summary.paired_baseline`（同题下的 DeepSeek-Flash 分数）
- **若在 24GB 显存上遇到 OOM**：
  - 优先减小 `--group_size`（从 `8` 降到 `4`），或将基座模型改为 `Qwen/Qwen2.5-3B-Instruct`。
- **若出现 KL 散度过大或 Validity 下降**：
  - 适当调大 `--kl_coef`（如从 `0.04` 调到 `0.08`），或降低 `--lr`（如 `1e-5`）。
