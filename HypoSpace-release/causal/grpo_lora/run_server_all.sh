#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Causal Diversity-Aware GRPO-LoRA v2.1 服务器全自动实验流水线（10 阶段）
#
# 核心特性（零同构泄露切分 + 统一采样参数 + 多种子评测 + 5 方消融）：
#   - 统一采样参数：SAMPLE_TEMPERATURE=0.9, SAMPLE_TOP_P=0.95 贯穿训练、验证与全部评测；
#   - 多种子 ID 评测：默认在 EVAL_SEEDS="33550336 1 2" 上分别评测 4 个开源模型；
#   - 组等权验证选优：--val_tasks_per_group 12，含 step=0 保底验证与严格大于 `>` 选优。
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

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
RUN_TAG="${RUN_TAG:-qwen7b_v2}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.9}"
SAMPLE_TOP_P="${SAMPLE_TOP_P:-0.95}"
VAL_TASKS_PER_GROUP="${VAL_TASKS_PER_GROUP:-12}"
EVAL_SEEDS="${EVAL_SEEDS:-33550336 1 2}"
OOD_MULTI_SEED="${OOD_MULTI_SEED:-0}"

GROUP_SIZE="${GROUP_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GRPO_TASKS_PER_STEP="${GRPO_TASKS_PER_STEP:-4}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
GRPO_EPOCHS="${GRPO_EPOCHS:-2}"
GRPO_LR="${GRPO_LR:-1e-5}"
KL_COEF="${KL_COEF:-0.06}"
DIV_WEIGHT="${DIV_WEIGHT:-0.5}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-0}"

SFT_DIR="causal/grpo_lora/outputs/${RUN_TAG}_sft"
VAL_GRPO_DIR="causal/grpo_lora/outputs/${RUN_TAG}_grpo_validity"
DIV_GRPO_DIR="causal/grpo_lora/outputs/${RUN_TAG}_grpo_diversity"
RES_DIR="causal/grpo_lora/results"
mkdir -p "${RES_DIR}" "causal/grpo_lora/outputs"

# 若已存在本地完整 HuggingFace 缓存，自动切换离线本地快照目录，避免 429 限流
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

read -ra SEED_ARRAY <<< "${EVAL_SEEDS}"
FIRST_SEED="${SEED_ARRAY[0]}"
if [[ "${OOD_MULTI_SEED}" == "1" ]]; then
  OOD_SEED_ARRAY=("${SEED_ARRAY[@]}")
else
  OOD_SEED_ARRAY=("${FIRST_SEED}")
fi

echo "============================================================"
echo "Repo Root   : ${REPO_ROOT}"
echo "Python      : ${PYTHON_BIN}"
echo "Model       : ${MODEL_NAME}"
echo "Run Tag     : ${RUN_TAG}"
echo "Sampling    : T=${SAMPLE_TEMPERATURE}, top_p=${SAMPLE_TOP_P}"
echo "Eval Seeds  : ${EVAL_SEEDS} (OOD multi-seed=${OOD_MULTI_SEED})"
echo "SFT Batch   : bs=${SFT_BATCH_SIZE} x grad_accum=${SFT_GRAD_ACCUM}"
echo "GRPO Batch  : tasks_per_step=${GRPO_TASKS_PER_STEP}, group_size=${GROUP_SIZE}, micro_bs=${MICRO_BATCH_SIZE}"
echo "GRPO Val    : val_tasks_per_group=${VAL_TASKS_PER_GROUP}"
echo "============================================================"

echo "[0/10] Preflight Check (5090 CUDA bfloat16 & dependencies)..."
"${PYTHON_BIN}" -c '
import importlib, sys, torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in", sys.executable)
    sys.exit(1)
x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
print("CUDA Check:", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), (x @ x)[0, 0].item())
required = ["transformers", "peft", "accelerate", "safetensors", "networkx", "numpy", "scipy"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
if missing:
    print("ERROR: Missing packages:", ", ".join(missing))
    sys.exit(1)
print("All dependencies verified OK.")
'

echo "[1/10] Running unit tests & verifying zero-isomorphism-leakage split..."
"${PYTHON_BIN}" causal/grpo_lora/tests/test_verifier_reward.py
"${PYTHON_BIN}" causal/grpo_lora/data_utils.py

echo "[2/10] Evaluating Base Model (Zero-shot, Unseen-Structure 4-Node Test, multi-seed)..."
for SEED in "${SEED_ARRAY[@]}"; do
  OUT_FILE="${RES_DIR}/${RUN_TAG}_unseen_base_seed${SEED}.json"
  if [[ "${SKIP_BASE_EVAL}" == "1" ]] && [[ -f "${OUT_FILE}" ]]; then
    echo "  Skipping existing ${OUT_FILE}"
  else
    "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
      --model_name "${MODEL_NAME}" \
      --temperature "${SAMPLE_TEMPERATURE}" \
      --top_p "${SAMPLE_TOP_P}" \
      --seed "${SEED}" \
      --output_path "${OUT_FILE}"
  fi
done

echo "[3/10] Training Stage 1: Debiased SFT-LoRA Warm-up (epoch-rotating GT coverage)..."
"${PYTHON_BIN}" causal/grpo_lora/train_sft.py \
  --model_name "${MODEL_NAME}" \
  --epochs "${SFT_EPOCHS}" \
  --batch_size "${SFT_BATCH_SIZE}" \
  --grad_accum "${SFT_GRAD_ACCUM}" \
  --lr 2e-4 \
  --output_dir "${SFT_DIR}"

echo "[4/10] Evaluating Stage 1 SFT-LoRA (Unseen-Structure 4-Node Test, multi-seed)..."
for SEED in "${SEED_ARRAY[@]}"; do
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${SFT_DIR}" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_unseen_sft_seed${SEED}.json"
done

echo "[5/10] Training Stage 2a Ablation: Validity-Only GRPO-LoRA..."
"${PYTHON_BIN}" causal/grpo_lora/train_grpo.py \
  --model_name "${MODEL_NAME}" \
  --sft_adapter_dir "${SFT_DIR}" \
  --reward_mode validity_only \
  --epochs "${GRPO_EPOCHS}" \
  --tasks_per_step "${GRPO_TASKS_PER_STEP}" \
  --group_size "${GROUP_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --temperature "${SAMPLE_TEMPERATURE}" \
  --top_p "${SAMPLE_TOP_P}" \
  --lr "${GRPO_LR}" \
  --kl_coef "${KL_COEF}" \
  --eval_every_steps 20 \
  --val_tasks_per_group "${VAL_TASKS_PER_GROUP}" \
  --output_dir "${VAL_GRPO_DIR}"

BEST_VAL_GRPO="${VAL_GRPO_DIR}/best_policy"
if [[ ! -d "${BEST_VAL_GRPO}" ]]; then
  BEST_VAL_GRPO="${VAL_GRPO_DIR}/final_policy"
fi

echo "[6/10] Evaluating Stage 2a Validity-Only GRPO-LoRA (Unseen-Structure Test, multi-seed)..."
for SEED in "${SEED_ARRAY[@]}"; do
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${BEST_VAL_GRPO}" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_unseen_grpo_validity_seed${SEED}.json"
done

echo "[7/10] Training Stage 2b Ours: Diversity-Aware GRPO-LoRA..."
"${PYTHON_BIN}" causal/grpo_lora/train_grpo.py \
  --model_name "${MODEL_NAME}" \
  --sft_adapter_dir "${SFT_DIR}" \
  --reward_mode diversity \
  --epochs "${GRPO_EPOCHS}" \
  --tasks_per_step "${GRPO_TASKS_PER_STEP}" \
  --group_size "${GROUP_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --temperature "${SAMPLE_TEMPERATURE}" \
  --top_p "${SAMPLE_TOP_P}" \
  --lr "${GRPO_LR}" \
  --kl_coef "${KL_COEF}" \
  --diversity_weight "${DIV_WEIGHT}" \
  --eval_every_steps 20 \
  --val_tasks_per_group "${VAL_TASKS_PER_GROUP}" \
  --output_dir "${DIV_GRPO_DIR}"

BEST_DIV_GRPO="${DIV_GRPO_DIR}/best_policy"
if [[ ! -d "${BEST_DIV_GRPO}" ]]; then
  BEST_DIV_GRPO="${DIV_GRPO_DIR}/final_policy"
fi

echo "[8/10] Evaluating Stage 2b Diversity-GRPO-LoRA (Unseen-Structure Test, multi-seed)..."
for SEED in "${SEED_ARRAY[@]}"; do
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${BEST_DIV_GRPO}" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_unseen_grpo_diversity_seed${SEED}.json"
done

echo "[9/10] Evaluating OOD 5-node dataset (n5_obs5_hybrid80.json) across all 4 open models..."
for SEED in "${OOD_SEED_ARRAY[@]}"; do
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --dataset_path causal/datasets/n5_obs5_hybrid80.json \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_ood_n5_base_seed${SEED}.json"
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${SFT_DIR}" \
    --dataset_path causal/datasets/n5_obs5_hybrid80.json \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_ood_n5_sft_seed${SEED}.json"
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${BEST_VAL_GRPO}" \
    --dataset_path causal/datasets/n5_obs5_hybrid80.json \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_ood_n5_grpo_validity_seed${SEED}.json"
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${BEST_DIV_GRPO}" \
    --dataset_path causal/datasets/n5_obs5_hybrid80.json \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_ood_n5_grpo_diversity_seed${SEED}.json"
done

echo "[10/10] Evaluating OOD 5-node large-space dataset (causal_n5_gt160.json)..."
for SEED in "${OOD_SEED_ARRAY[@]}"; do
  "${PYTHON_BIN}" causal/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${BEST_DIV_GRPO}" \
    --dataset_path causal/benchmark_obs_count/paper_scale/causal_n5_gt160.json \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_ood_n5_gt160_seed${SEED}.json"
done

echo "============================================================"
echo "All 10 pipeline stages completed! Generated result files:"
ls -lh "${RES_DIR}/${RUN_TAG}_"*.json
echo "============================================================"
