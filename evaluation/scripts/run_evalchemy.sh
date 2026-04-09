#!/bin/bash
# ==============================================================================
# evalchemy: GPQADiamond, LiveCodeBench (code)
# ==============================================================================
source /mnt/shared-storage-user/wangpeng/.bashrc

export PATH="$HOME/anaconda3/bin:$PATH"

sudo mkdir -p /models
sudo chmod 777 /models
sudo mkdir -p /cfs_oss/shared/ai4good1
sudo chmod 777 /cfs_oss/shared/ai4good1
sudo mkdir -p /cfs_oss/shared/ai4good_shared
sudo chmod 777 /cfs_oss/shared/ai4good_shared

cd /mnt/shared-storage-user/wangpeng
export AWS_ACCESS_KEY_ID=bqihrarkoe8y0nimkmxc
export AWS_SECRET_ACCESS_KEY=r1is8lvr26tpzmchvz9jndww5mwtik1cg3g793e1

./s3mount  ailab-public-shared /models --endpoint-url http://hdd1.h.pjlab.org.cn:8060 --force-path-style --no-sign-request
./s3mount ai4good-h-hdd-1 /cfs_oss/shared/ai4good1 --endpoint-url http://hdd1.h.pjlab.org.cn:8060 --allow-delete --allow-overwrite --force-path-style
./s3mount ai4good-h-hdd-shared /cfs_oss/shared/ai4good_shared --endpoint-url http://hdd1.h.pjlab.org.cn:8060 --allow-delete --allow-overwrite --force-path-style

set -e

# -------------------- Environment --------------------
export PATH=/usr/bin:/usr/local/cuda/bin:/home/renqihan/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/usr/bin/python:$PYTHONPATH
export PYTHONUSERBASE=/home/renqihan/.local

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# -------------------- Install --------------------
UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Install lm-evaluation-harness (evalchemy dependency)
PIP_PATH="${UNIFIED_ROOT}/lm-evaluation-harness"
cd "$PIP_PATH"
pip install -e .
pip install bespokelabs bespokelabs-curator sqlalchemy

PROJECT_ROOT="${UNIFIED_ROOT}/evalchemy"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"

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
    # /path/to/model/merged_step10
    # /path/to/model/merged_step40
    # /mnt/shared-storage-user/ai4good1-share/hf_hub/Qwen/Qwen3-8B-Base
    /cfs_oss/shared/ai4good1/renqihan/ckpt/offline_H/sft_qw3-base-14b_new-c1-20.5k-16384_vanilla_lr5e-5_ep8_token-mean_bs256/merged_step640
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
