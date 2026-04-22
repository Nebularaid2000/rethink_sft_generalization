#!/bin/bash
# ==============================================================================
# evalchemy: GPQADiamond, LiveCodeBench (code)
# ==============================================================================

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EVALCHEMY_ROOT="${UNIFIED_ROOT}/evalchemy"
cd "$EVALCHEMY_ROOT"
export PYTHONPATH="${EVALCHEMY_ROOT}:$PYTHONPATH"

# -------------------- Config (MODIFY HERE) --------------------
export CUDA_VISIBLE_DEVICES=0,1
gpu_num=2

EVAL_TASKS=(
    "GPQADiamond"
    "LiveCodeBench"
)
TASKS_STR=$(IFS=, ; echo "${EVAL_TASKS[*]}")

MODELS=(
    # TODO: Add your model paths here
    /path/to/model/merged_step10
    /path/to/model/merged_step40
)

RESULT_ROOT="${UNIFIED_ROOT}/results/evalchemy"
LOGS_DIR="${RESULT_ROOT}/logs"
OUTPUT_DIR="${RESULT_ROOT}/outputs"
mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"

default_max_output_length=32768
batch_size="auto"

# -------------------- Run --------------------
overall_log="${LOGS_DIR}/eval_$(date +"%Y%m%d_%H%M%S").log"

for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    MODEL_NAME="$(basename $(dirname "$MODEL_PATH"))_$(basename "$MODEL_PATH")"

    if [[ "$MODEL_PATH" == *"Qwen2.5-7B-Instruct"* ]] || \
       [[ "$MODEL_PATH" == *"Qwen2.5-Math-7B-Instruct"* ]] || \
       [[ "$MODEL_PATH" == *"Qwen-2.5-Math-7B-SimpleRL-Zoo"* ]]; then
        max_output_length=32768
    else
        max_output_length=$default_max_output_length
    fi

    current_time=$(date +"%Y%m%d_%H%M%S")
    log_file="${LOGS_DIR}/${MODEL_NAME}_eval_${current_time}.log"

    echo "==== [evalchemy] model=$MODEL_NAME tasks=$TASKS_STR ====" | tee -a "$overall_log"

    python -m eval.eval \
        --model vllm \
        --tasks "${TASKS_STR}" \
        --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${gpu_num},gpu_memory_utilization=0.85,dtype=bfloat16,max_model_len=${max_output_length}" \
        --batch_size "$batch_size" \
        --output_path "$OUTPUT_DIR" 2>&1 | tee "$log_file"
done
