#!/bin/bash
# ==============================================================================
# math_eval: MATH500, AIME24
# ==============================================================================

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn


# -------------------- Config (MODIFY HERE) --------------------
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${UNIFIED_ROOT}/math_eval"
cd "$ROOT"

OUTPUT="${UNIFIED_ROOT}/results/math"

datasets=(
    ${UNIFIED_ROOT}/data/math/MATH500
    ${UNIFIED_ROOT}/data/math/converted_aime_dataset
)

models=(
    # TODO: Add your model paths here
    /path/to/model/merged_step10
    /path/to/model/merged_step40
)

mode=(
    no_instruct
)

# -------------------- Run --------------------
for path in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for m in "${mode[@]}"; do
            echo "==== [math] model=$(basename "$path") dataset=$(basename "$dataset") mode=$m ===="
            start_time=$(date +%s)
            python math_eval_budget.py \
                --model_path=${path} \
                --dataset=${dataset} \
                --types=${m} \
                --output_dir=$OUTPUT \
                --num_gpus=${NUM_GPUS}
            end_time=$(date +%s)
            elapsed_time=$((end_time - start_time))
            echo "  Time elapsed: ${elapsed_time}s"
            echo "------------------------------------------------------------"
        done
    done
done
