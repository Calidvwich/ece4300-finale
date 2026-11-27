#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 3D Voxel Reconstruction Diversity-Aware GRPO-LoRA 服务器 10 阶段全自动流水线
#
# 遵循 Part A.7 & 第 0 部分规范：
#   - 统一采样参数：SAMPLE_TEMPERATURE=0.9, SAMPLE_TOP_P=0.95, max_new_tokens=128
#   - 多种子 ID 评测：EVAL_SEEDS="33550336 1 2"
#   - Validity-Only 与 Diversity 两组均评测 best_policy 与 final_policy（验证集/测试集对齐）
#   - 支持 SKIP_VALIDITY=1 跳过第 5、6 步，支持 OOD_MULTI_SEED=1
# ==============================================================================

CALLER_PWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
if [[ -n "${PYTHON_BIN:-}" && "${PYTHON_BIN}" == */* && "${PYTHON_BIN}" != /* && -x "${CALLER_PWD}/${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${CALLER_PWD}/${PYTHON_BIN}"
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  elif [[ -x "/opt/envs/torch5090/bin/python" ]]; then
    PYTHON_BIN="/opt/envs/torch5090/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${HF_HOME:-}" ]]; then
  export HF_HOME
  if [[ "${MODEL_NAME}" == "Qwen/Qwen2.5-7B-Instruct" ]] && compgen -G "${HF_HOME}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/config.json" > /dev/null; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  fi
fi

RUN_TAG="${RUN_TAG:-qwen7b_3d}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-0.9}"
SAMPLE_TOP_P="${SAMPLE_TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
VAL_TASKS_PER_GROUP="${VAL_TASKS_PER_GROUP:-6}"
EVAL_SEEDS="${EVAL_SEEDS:-33550336 1 2}"
OOD_MULTI_SEED="${OOD_MULTI_SEED:-0}"
SKIP_VALIDITY="${SKIP_VALIDITY:-0}"
SKIP_BASE_EVAL="${SKIP_BASE_EVAL:-0}"

GROUP_SIZE="${GROUP_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GRPO_TASKS_PER_STEP="${GRPO_TASKS_PER_STEP:-4}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-1}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-16}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
GRPO_EPOCHS="${GRPO_EPOCHS:-5}"
GRPO_LR="${GRPO_LR:-1e-5}"
KL_COEF="${KL_COEF:-0.06}"
DIV_WEIGHT="${DIV_WEIGHT:-0.5}"

SFT_DIR="3d/grpo_lora/outputs/${RUN_TAG}_sft"
VAL_GRPO_DIR="3d/grpo_lora/outputs/${RUN_TAG}_grpo_validity"
DIV_GRPO_DIR="3d/grpo_lora/outputs/${RUN_TAG}_grpo_diversity"
RES_DIR="3d/grpo_lora/results"
mkdir -p "${RES_DIR}" "3d/grpo_lora/outputs"

# 检测本地 HF 缓存快照并自动切换离线加载
if [[ ! -d "${MODEL_NAME}" ]] && [[ "${MODEL_NAME}" == */* ]]; then
  CACHE_HUB_DIR="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
  SAFE_REPO="models--${MODEL_NAME//\//--}"
  if [[ -d "${CACHE_HUB_DIR}/${SAFE_REPO}/snapshots" ]]; then
    CANDIDATE_SNAP="$(ls -dt "${CACHE_HUB_DIR}/${SAFE_REPO}/snapshots"/* 2>/dev/null | head -n 1 || true)"
    if [[ -n "${CANDIDATE_SNAP}" ]] && [[ -f "${CANDIDATE_SNAP}/config.json" ]]; then
      echo "Detected local HF snapshot, switching to offline model: ${CANDIDATE_SNAP}"
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
echo "3D GRPO-LoRA Server Pipeline"
echo "Python       : ${PYTHON_BIN}"
echo "Model        : ${MODEL_NAME}"
echo "Run Tag      : ${RUN_TAG}"
echo "Sampling     : T=${SAMPLE_TEMPERATURE}, top_p=${SAMPLE_TOP_P}, max_new_tokens=${MAX_NEW_TOKENS}"
echo "Eval Seeds   : ${EVAL_SEEDS} (OOD multi-seed=${OOD_MULTI_SEED})"
echo "Skip Validity: ${SKIP_VALIDITY}"
echo "============================================================"

echo "[1/10] Checking environment, generating 3D datasets/splits & running unit tests..."
"${PYTHON_BIN}" -c '
import importlib, sys, torch
if not torch.cuda.is_available():
    print("ERROR: CUDA unavailable in", sys.executable); sys.exit(1)
x = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
print("CUDA Check:", torch.__version__, torch.cuda.get_device_name(0), (x @ x)[0, 0].item())
for pkg in ["transformers", "peft", "accelerate", "safetensors", "numpy"]:
    if importlib.util.find_spec(pkg) is None:
        print("ERROR: Missing package:", pkg); sys.exit(1)
'
if [[ ! -f "3d/datasets/3d_g3_h3_b1to3.json" ]]; then
  "${PYTHON_BIN}" 3d/generate_3d_dataset_complete.py --grid-size 3 --max-height 3 --max-blocks 3 --output 3d/datasets/3d_g3_h3_b1to3.json
fi
if [[ ! -f "3d/datasets/3d_g3_h3_b4.json" ]]; then
  "${PYTHON_BIN}" 3d/generate_3d_dataset_complete.py --grid-size 3 --max-height 3 --max-blocks 4 --fixed --output 3d/datasets/3d_g3_h3_b4.json
fi
"${PYTHON_BIN}" 3d/grpo_lora/data_utils.py
"${PYTHON_BIN}" 3d/grpo_lora/tests/test_verifier_reward_3d.py

echo "[2/10] Evaluating Base Model (Zero-shot, Unseen-D4 ID Test, multi-seed)..."
for SEED in "${SEED_ARRAY[@]}"; do
  OUT_FILE="${RES_DIR}/${RUN_TAG}_id_base_seed${SEED}.json"
  if [[ "${SKIP_BASE_EVAL}" == "1" ]] && [[ -f "${OUT_FILE}" ]]; then
    echo "  Skipping existing ${OUT_FILE}"
  else
    "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
      --model_name "${MODEL_NAME}" \
      --temperature "${SAMPLE_TEMPERATURE}" \
      --top_p "${SAMPLE_TOP_P}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --output_path "${OUT_FILE}"
  fi
done

echo "[3/10] Training Stage 1: Debiased 3D SFT-LoRA..."
"${PYTHON_BIN}" 3d/grpo_lora/train_sft.py \
  --model_name "${MODEL_NAME}" \
  --epochs "${SFT_EPOCHS}" \
  --batch_size "${SFT_BATCH_SIZE}" \
  --grad_accum "${SFT_GRAD_ACCUM}" \
  --lr 2e-4 \
  --output_dir "${SFT_DIR}"

echo "[4/10] Evaluating Stage 1 SFT-LoRA (Unseen-D4 ID Test, multi-seed)..."
for SEED in "${SEED_ARRAY[@]}"; do
  "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${SFT_DIR}" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_id_sft_seed${SEED}.json"
done

if [[ "${SKIP_VALIDITY}" != "1" ]]; then
  echo "[5/10] Training Stage 2a Ablation: Validity-Only 3D GRPO-LoRA..."
  "${PYTHON_BIN}" 3d/grpo_lora/train_grpo.py \
    --model_name "${MODEL_NAME}" \
    --sft_adapter_dir "${SFT_DIR}" \
    --reward_mode validity_only \
    --epochs "${GRPO_EPOCHS}" \
    --tasks_per_step "${GRPO_TASKS_PER_STEP}" \
    --group_size "${GROUP_SIZE}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --lr "${GRPO_LR}" \
    --kl_coef "${KL_COEF}" \
    --eval_every_steps 15 \
    --val_tasks_per_group "${VAL_TASKS_PER_GROUP}" \
    --output_dir "${VAL_GRPO_DIR}"

  echo "[6/10] Evaluating Stage 2a Validity-Only best_policy & final_policy (multi-seed)..."
  for SEED in "${SEED_ARRAY[@]}"; do
    "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
      --model_name "${MODEL_NAME}" \
      --adapter_dir "${VAL_GRPO_DIR}/best_policy" \
      --temperature "${SAMPLE_TEMPERATURE}" \
      --top_p "${SAMPLE_TOP_P}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --output_path "${RES_DIR}/${RUN_TAG}_id_grpo_validity_best_seed${SEED}.json"
    "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
      --model_name "${MODEL_NAME}" \
      --adapter_dir "${VAL_GRPO_DIR}/final_policy" \
      --temperature "${SAMPLE_TEMPERATURE}" \
      --top_p "${SAMPLE_TOP_P}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --output_path "${RES_DIR}/${RUN_TAG}_id_grpo_validity_final_seed${SEED}.json"
  done
else
  echo "[5-6/10] SKIP_VALIDITY=1 set; skipping Validity-Only GRPO training and ID eval."
fi

echo "[7/10] Training Stage 2b Ours: Diversity-Aware 3D GRPO-LoRA..."
"${PYTHON_BIN}" 3d/grpo_lora/train_grpo.py \
  --model_name "${MODEL_NAME}" \
  --sft_adapter_dir "${SFT_DIR}" \
  --reward_mode diversity \
  --epochs "${GRPO_EPOCHS}" \
  --tasks_per_step "${GRPO_TASKS_PER_STEP}" \
  --group_size "${GROUP_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --temperature "${SAMPLE_TEMPERATURE}" \
  --top_p "${SAMPLE_TOP_P}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --lr "${GRPO_LR}" \
  --kl_coef "${KL_COEF}" \
  --diversity_weight "${DIV_WEIGHT}" \
  --eval_every_steps 15 \
  --val_tasks_per_group "${VAL_TASKS_PER_GROUP}" \
  --output_dir "${DIV_GRPO_DIR}"

echo "[8/10] Evaluating Stage 2b Diversity best_policy & final_policy (multi-seed)..."
for SEED in "${SEED_ARRAY[@]}"; do
  "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${DIV_GRPO_DIR}/best_policy" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_id_grpo_diversity_best_seed${SEED}.json"
  "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${DIV_GRPO_DIR}/final_policy" \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_id_grpo_diversity_final_seed${SEED}.json"
done

echo "[9/10] Evaluating 4-Block OOD (3d_g3_h3_b4_ood.json: 23 D4 classes, K=81)..."
for SEED in "${OOD_SEED_ARRAY[@]}"; do
  "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --dataset_path 3d/datasets/3d_g3_h3_b4_ood.json \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_ood_b4_base_seed${SEED}.json"
  "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${SFT_DIR}" \
    --dataset_path 3d/datasets/3d_g3_h3_b4_ood.json \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_ood_b4_sft_seed${SEED}.json"
  if [[ "${SKIP_VALIDITY}" != "1" ]]; then
    "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
      --model_name "${MODEL_NAME}" \
      --adapter_dir "${VAL_GRPO_DIR}/final_policy" \
      --dataset_path 3d/datasets/3d_g3_h3_b4_ood.json \
      --temperature "${SAMPLE_TEMPERATURE}" \
      --top_p "${SAMPLE_TOP_P}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --seed "${SEED}" \
      --output_path "${RES_DIR}/${RUN_TAG}_ood_b4_grpo_validity_final_seed${SEED}.json"
  fi
  "${PYTHON_BIN}" 3d/grpo_lora/evaluate_lora.py \
    --model_name "${MODEL_NAME}" \
    --adapter_dir "${DIV_GRPO_DIR}/best_policy" \
    --dataset_path 3d/datasets/3d_g3_h3_b4_ood.json \
    --temperature "${SAMPLE_TEMPERATURE}" \
    --top_p "${SAMPLE_TOP_P}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --seed "${SEED}" \
    --output_path "${RES_DIR}/${RUN_TAG}_ood_b4_grpo_diversity_best_seed${SEED}.json"
done

echo "[10/10] Printing summary and generated result files..."
ls -lh "${RES_DIR}/${RUN_TAG}_"*.json
if [[ -f "${PROJECT_ROOT}/results_analysis/stats_3d.py" ]]; then
  echo "You can now run stats_3d.py on the generated files under ${RES_DIR}/."
fi
echo "All 10 stages for 3D GRPO-LoRA completed!"
