#!/bin/bash
# ==============================================================================
# lm-evaluation-harness: ifeval, mmlu_pro, truthfulqa, halueval
# ==============================================================================

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# -------------------- Install --------------------
UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LM_EVAL_PATH="${UNIFIED_ROOT}/lm-evaluation-harness"
cd $LM_EVAL_PATH

# -------------------- Config (MODIFY HERE) --------------------
export CUDA_VISIBLE_DEVICES=0,1
gpu_num=2

tasks=(
    "truthfulqa"
    "ifeval"
    "mmlu_pro"
    "halueval"
)

models=(
    # TODO: Add your model paths here
    /path/to/model/merged_step10
    /path/to/model/merged_step40
)

output_base="${UNIFIED_ROOT}/results/lm_eval"

max_model_tokens=32768
max_gen_tokens=32768
base_model_args="tensor_parallel_size=$gpu_num,data_parallel_size=1,gpu_memory_utilization=0.85,dtype=bfloat16"
batch_size="auto"

# -------------------- Run --------------------

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        output_file="${output_base}/${task}"
        mkdir -p "$output_file"

        model_args="$base_model_args,max_model_len=$max_model_tokens"

        echo "==== [lm_eval] model=$(basename $(dirname "$model"))/$(basename "$model") task=$task ===="
        lm_eval --model vllm \
            --model_args pretrained=${model},$model_args \
            --gen_kwargs max_gen_toks=$max_gen_tokens \
            --tasks "$task" \
            --batch_size "$batch_size" \
            --log_samples \
            --trust_remote_code \
            --apply_chat_template \
            --output_path "$output_file"
    done
done
