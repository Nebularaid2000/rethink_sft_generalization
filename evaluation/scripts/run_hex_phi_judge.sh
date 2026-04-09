#!/bin/bash
# ==============================================================================
# harmbench (HEx-PHI): judge completions and compute ASR
# ==============================================================================

set -euo pipefail

UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HARM_ROOT="${UNIFIED_ROOT}/harmbench"
cd "$HARM_ROOT"

formatted_date=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_DIR="${UNIFIED_ROOT}/results/harmbench/logs"
mkdir -p "$LOG_DIR"
JOB_LOG="${LOG_DIR}/hex_phi_judge_${formatted_date}.log"
JOB_ERR="${LOG_DIR}/hex_phi_judge_${formatted_date}.err"

export VLLM_USE_MULTIPROCESSING_SPAWN=1
export PYTHON_MULTIPROCESSING_METHOD=spawn

behaviors_path=data/behavior_datasets/hex-phi.csv
base_save_dir=../results/harmbench/results_hex-phi_temp0.6
save_dir=results_gpt4.1

num_tokens=16384
num_workers=100 # TODO: configure max workers for your API quota

# TODO: add your model aliases in harmbench/configs/model_configs/models.yaml
model_names=(
    model_alias1
    model_alias2
)

for model_name in "${model_names[@]}"; do
    echo "=============================================="
    echo "Processing model: $model_name"
    echo "=============================================="

    python3 -u evaluate_completions_api_parallel_resume.py \
        --behaviors_path "$behaviors_path" \
        --completions_path "./${base_save_dir}/DirectRequest/default/completions/${num_tokens}_tokens/${model_name}.json" \
        --save_path "./${base_save_dir}/DirectRequest/default/${save_dir}/${num_tokens}_tokens/${model_name}_eval.json" \
        --save_asr_path "./${base_save_dir}/DirectRequest/default/${save_dir}/${num_tokens}_tokens/${model_name}_asr_result.json" \
        --num_tokens "$num_tokens" \
        --max_workers "$num_workers" \
        --resume \
        --remove_think >>"$JOB_LOG" 2>>"$JOB_ERR"
done
