#!/bin/bash
# ==============================================================================
# harmbench (HEx-PHI): generate completions
# ==============================================================================

set -euo pipefail

UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARM_ROOT="${UNIFIED_ROOT}/harmbench"
cd "$HARM_ROOT"

formatted_date=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_DIR="${UNIFIED_ROOT}/results/harmbench/logs"
mkdir -p "$LOG_DIR"
JOB_LOG="${LOG_DIR}/hex_phi_generate_${formatted_date}.log"
JOB_ERR="${LOG_DIR}/hex_phi_generate_${formatted_date}.err"

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS # use XFORMERS for better reproducibility

temperature=0.6
method=DirectRequest

behaviors_path=${HARM_ROOT}/data/behavior_datasets/hex-phi.csv
base_save_dir=${UNIFIED_ROOT}/results/harmbench/results_hex-phi_temp${temperature}
mkdir -p "$base_save_dir"

# TODO: add your model aliases in harmbench/configs/model_configs/models.yaml
models=(
    model_alias1
    model_alias2
)

for model in "${models[@]}"; do
    echo "Processing model: $model"
    python ./scripts/run_pipeline.py \
        --methods "$method" \
        --models "$model" \
        --base_save_dir "$base_save_dir" \
        --behaviors_path "$behaviors_path" \
        --step all \
        --mode local \
        --max_new_tokens 16384 \
        --temperature "$temperature" \
        --cls_path cais/HarmBench-Llama-2-13b-cls >>"$JOB_LOG" 2>>"$JOB_ERR"
done
