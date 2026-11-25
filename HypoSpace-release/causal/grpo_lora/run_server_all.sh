#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Causal Diversity-Aware GRPO-LoRA 服务器一键全量训练与评测脚本
#
# 适用显卡：
#   - RTX 5090 (32GB): 默认自动优先检测 /opt/envs/torch5090/bin/python，
#                      使用 Qwen/Qwen2.5-7B-Instruct, GROUP_SIZE=8, MICRO_BATCH_SIZE=2
#   - RTX 4090 (24GB): 可覆盖为 MODEL_NAME="Qwen/Qwen2.5-3B-Instruct" 或 MICRO_BATCH_SIZE=1
#
# 断点续跑 / 跳过 Base 评测示例：
#   SKIP_BASE_EVAL=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#     MODEL_NAME="/opt/data/private/machine-learning/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>" \
#     bash causal/grpo_lora/run_server_all.sh
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# 自动优先选用 5090 专用 Python 环境，其次本地 .venv，最后回退 python3
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "/opt/envs/torch5090/bin/python" ]]; then
    PYTHON_BIN="/opt/envs/torch5090/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
RUN_TAG="${RUN_TAG:-qwen7b}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"
KL_COEF="${KL_COEF:-0.04}"
DIV_WEIGHT="${DIV_WEIGHT:-0.5}"
DUP_PENALTY="${DUP_PENALTY:--0.3}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-0}"

SFT_DIR="causal/grpo_lora/outputs/${RUN_TAG}_sft"
GRPO_DIR="causal/grpo_lora/outputs/${RUN_TAG}_grpo"
RES_DIR="causal/grpo_lora/results"
mkdir -p "${RES_DIR}" "causal/grpo_lora/outputs"

# 若已存在本地完整 HuggingFace 缓存且未显式传入本地路径，自动切换为离线本地快照目录，避免 429 限流
if [[ ! -d "${MODEL_NAME}" ]] && [[ "${MODEL_NAME}" == */* ]]; then
  CACHE_HUB_DIR="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
  SAFE_REPO="models--${MODEL_NAME//\//--}"
  if [[ -d "${CACHE_HUB_DIR}/${SAFE_REPO}/snapshots" ]]; then
    CANDIDATE_SNAP="$(ls -dt "${CACHE_HUB_DIR}/${SAFE_REPO}/snapshots"/* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${CANDIDATE_SNAP}" ]] && [[ -f "${CANDIDATE_SNAP}/config.json" ]]; then
      echo "Detected local HF snapshot, switching to offline local model: ${CANDIDATE_SNAP}"
      MODEL_NAME="${CANDIDATE_SNAP}"
      export HF_HUB_OFFLINE=1
      export TRANSFORMERS_OFFLINE=1
    fi
  fi
fi

echo "============================================================"
echo "Repo Root   : ${REPO_ROOT}"
echo "Python      : ${PYTHON_BIN}"
echo "Model       : ${MODEL_NAME}"
echo "Run Tag     : ${RUN_TAG}"
echo "SFT Batch   : bs=${SFT_BATCH_SIZE} x grad_accum=${SFT_GRAD_ACCUM}"
echo "GRPO Batch  : group_size=${GROUP_SIZE}, micro_batch_size=${MICRO_BATCH_SIZE}"
echo "HF_HOME     : ${HF_HOME:-<default ~/.cache/huggingface>}"
echo "============================================================"

echo "[0/8] Preflight Check (5090 CUDA bfloat16 & dependencies)..."
"${PYTHON_BIN}" -c '
import importlib, sys, torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in", sys.executable)
    sys.exit(1)
x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
print("CUDA Check:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), (x @ x)[0, 0].item())
required = ["transformers", "peft", "accelerate", "safetensors", "networkx", "numpy", "scipy"]
missing = []
for pkg in required:
    try:
        importlib.import_module(pkg)
    except Exception:
        missing.append(pkg)
if missing:
    print("ERROR: Missing packages:", ", ".join(missing))
    print(f"Please run: {sys.executable} -m pip install -r causal/grpo_lora/requirements_lora.txt")
    sys.exit(1)
print("All dependencies verified OK.")
'

echo "[1/8] Running unit tests..."
"${PYTHON_BIN}" causal/grpo_lora/tests/test_verifier_reward.py

if [[ "${SKIP_BASE_EVAL}" == "1" ]] && [[ -f "${RES_DIR}/${RUN_TAG}_id_base.json" ]]; then
  echo "[2/8] Skipping Base Model eval (found existing ${RES_DIR}/${RUN_TAG}_id_base.json)."
else
  echo "[2/8] Evaluating Base Model (Zero-shot, Fixed-Prompt ID Test)..."
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --output_path "${RES_DIR}/${RUN_TAG}_id_base.json"
fi

echo "[3/8] Training Stage 1: SFT-LoRA Warm-up (gradient checkpointing ON)..."
"${PYTHON_BIN}" causal/grpo_lora/train_sft.py \
  --model_name "${MODEL_NAME}" \
  --epochs "${SFT_EPOCHS}" \
  --batch_size "${SFT_BATCH_SIZE}" \
  --grad_accum "${SFT_GRAD_ACCUM}" \
  --lr 2e-4 \
  --output_dir "${SFT_DIR}"

echo "[4/8] Evaluating Stage 1 SFT-LoRA (ID Test)..."
"${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
  --model_name "${MODEL_NAME}" \
  --adapter_dir "${SFT_DIR}" \
  --output_path "${RES_DIR}/${RUN_TAG}_id_sft.json"

echo "[5/8] Training Stage 2: Diversity-Aware GRPO-LoRA (micro-batched + gradient checkpointing)..."
"${PYTHON_BIN}" causal/grpo_lora/train_grpo.py \
  --model_name "${MODEL_NAME}" \
  --sft_adapter_dir "${SFT_DIR}" \
  --epochs "${GRPO_EPOCHS}" \
  --group_size "${GROUP_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --lr 2e-5 \
  --kl_coef "${KL_COEF}" \
  --diversity_weight "${DIV_WEIGHT}" \
  --duplicate_penalty "${DUP_PENALTY}" \
  --eval_every_steps 50 \
  --val_max_tasks 39 \
  --output_dir "${GRPO_DIR}"

BEST_GRPO_ADAPTER="${GRPO_DIR}/best_policy"
if [[ ! -d "${BEST_GRPO_ADAPTER}" ]]; then
  BEST_GRPO_ADAPTER="${GRPO_DIR}/final_policy"
fi

echo "[6/8] Evaluating Stage 2 Diversity-GRPO-LoRA (ID Test, paired with DeepSeek-Flash)..."
"${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
  --model_name "${MODEL_NAME}" \
  --adapter_dir "${BEST_GRPO_ADAPTER}" \
  --output_path "${RES_DIR}/${RUN_TAG}_id_grpo.json"

echo "[7/8] Evaluating OOD 5-node dataset (n5_obs5_hybrid80.json)..."
"${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
  --model_name "${MODEL_NAME}" \
  --adapter_dir "${BEST_GRPO_ADAPTER}" \
  --dataset_path causal/datasets/n5_obs5_hybrid80.json \
  --output_path "${RES_DIR}/${RUN_TAG}_ood_n5.json"

echo "[8/8] Evaluating OOD 5-node large-space dataset (causal_n5_gt160.json)..."
"${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
  --model_name "${MODEL_NAME}" \
  --adapter_dir "${BEST_GRPO_ADAPTER}" \
  --dataset_path causal/benchmark_obs_count/paper_scale/causal_n5_gt160.json \
  --output_path "${RES_DIR}/${RUN_TAG}_ood_n5_gt160.json"

echo "============================================================"
echo "All 8 pipeline stages completed! Generated result files:"
ls -lh "${RES_DIR}/${RUN_TAG}_"*.json
echo "Training curve log: ${GRPO_DIR}/grpo_metrics.jsonl"
echo "============================================================"
