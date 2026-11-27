#!/usr/bin/env bash
set -euo pipefail

CALLER_PWD="$(pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
PYTHON="${PYTHON_BIN:-${PYTHON:-}}"
if [[ -n "$PYTHON" && "$PYTHON" == */* && "$PYTHON" != /* && -x "${CALLER_PWD}/${PYTHON}" ]]; then
    PYTHON="${CALLER_PWD}/${PYTHON}"
fi
if [[ -z "$PYTHON" ]]; then
    if [[ -x "../../.venv/bin/python" ]]; then
        PYTHON="../../.venv/bin/python"
    elif [[ -x "../../../.venv/bin/python" ]]; then
        PYTHON="../../../.venv/bin/python"
    elif [[ -x "/opt/envs/torch5090/bin/python" ]]; then
        PYTHON="/opt/envs/torch5090/bin/python"
    else
        PYTHON="python3"
    fi
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
    if [[ "$MODEL_NAME" == "Qwen/Qwen2.5-7B-Instruct" ]] && compgen -G "${HF_HOME}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*/config.json" > /dev/null; then
        export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
        export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    fi
fi

RUN_TAG="${RUN_TAG:-qwen7b_boolean}"
SKIP_VALIDITY="${SKIP_VALIDITY:-0}"
OOD_MULTI_SEED="${OOD_MULTI_SEED:-0}"
GRPO_EPOCHS="${GRPO_EPOCHS:-6}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-8}"
OOD_PATH="../../causal/benchmark_obs_count/paper_scale/boolean_full_d2.json"
SFT_DIR="outputs/${RUN_TAG}_sft"
VALIDITY_DIR="outputs/${RUN_TAG}_grpo_validity"
DIVERSITY_DIR="outputs/${RUN_TAG}_grpo_diversity"
SEEDS=(33550336 1 2)
OOD_SEEDS=(33550336)
if [[ "$OOD_MULTI_SEED" == 1 ]]; then OOD_SEEDS=("${SEEDS[@]}"); fi

evaluate() {
    local split="$1" label="$2" adapter="$3"
    shift 3
    local args=() seed
    if [[ -n "$adapter" ]]; then args+=(--adapter_dir "$adapter"); fi
    if [[ "$split" == ood ]]; then args+=(--dataset_path "$OOD_PATH"); fi
    for seed in "$@"; do
        "$PYTHON" -u evaluate_lora.py --model_name "$MODEL_NAME" \
            --gen_batch_size "$GEN_BATCH_SIZE" --seed "$seed" \
            --output_path "results/${RUN_TAG}_${split}_${label}_seed${seed}.json" "${args[@]}"
    done
}

grpo() {
    "$PYTHON" -u train_grpo.py --model_name "$MODEL_NAME" --sft_adapter_dir "$SFT_DIR" \
        --epochs "$GRPO_EPOCHS" --micro_batch_size "$MICRO_BATCH_SIZE" \
        --val_gen_batch_size "$GEN_BATCH_SIZE" --reward_mode "$1" --output_dir "$2"
}

"$PYTHON" -u data_utils.py
"$PYTHON" -m unittest discover -s tests -v

evaluate id base "" "${SEEDS[@]}"
"$PYTHON" -u train_sft.py --model_name "$MODEL_NAME" --epochs "$SFT_EPOCHS" --output_dir "$SFT_DIR"
evaluate id sft "$SFT_DIR" "${SEEDS[@]}"

if [[ "$SKIP_VALIDITY" != 1 ]]; then
    grpo validity_only "$VALIDITY_DIR"
    evaluate id validity_best "$VALIDITY_DIR/best_policy" "${SEEDS[@]}"
    evaluate id validity_final "$VALIDITY_DIR/final_policy" "${SEEDS[@]}"
fi

grpo diversity "$DIVERSITY_DIR"
evaluate id diversity "$DIVERSITY_DIR/best_policy" "${SEEDS[@]}"
evaluate id diversity_final "$DIVERSITY_DIR/final_policy" "${SEEDS[@]}"

evaluate ood base "" "${OOD_SEEDS[@]}"
evaluate ood sft "$SFT_DIR" "${OOD_SEEDS[@]}"
if [[ "$SKIP_VALIDITY" != 1 ]]; then
    evaluate ood validity_final "$VALIDITY_DIR/final_policy" "${OOD_SEEDS[@]}"
fi
evaluate ood diversity "$DIVERSITY_DIR/best_policy" "${OOD_SEEDS[@]}"

printf '\nBoolean run %s complete. Checkpoints in %s/outputs and evaluations in %s/results.\n' \
    "$RUN_TAG" "$HERE" "$HERE"
