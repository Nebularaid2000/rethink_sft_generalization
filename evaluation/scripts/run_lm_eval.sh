#!/bin/bash
# ==============================================================================
# lm-evaluation-harness: ifeval, mmlu_pro, truthfulqa, halueval
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
export HF_HUB_OFFLINE=1

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# -------------------- Install --------------------
UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIP_PATH="${UNIFIED_ROOT}/lm-evaluation-harness"
cd "$PIP_PATH"
pip install -e .

# -------------------- Config (MODIFY HERE) --------------------
export CUDA_VISIBLE_DEVICES=0,1
gpu_num=2

tasks=(
    # "truthfulqa"
    "ifeval"
    "mmlu_pro"
    "halueval"
)

models=(
    # TODO: Add your model paths here
    # /path/to/model/merged_step10
    # /path/to/model/merged_step40
    # /mnt/shared-storage-user/ai4good1-share/hf_hub/Qwen/Qwen3-14B-Base
    /cfs_oss/shared/ai4good1/renqihan/ckpt/offline_H/sft_qw3-base-14b_new-c1-20.5k-16384_vanilla_lr5e-5_ep8_token-mean_bs256/merged_step640
)

output_base="${UNIFIED_ROOT}/results/lm_eval"

default_max_model_tokens=32768
default_max_gen_tokens=32768
base_model_args="tensor_parallel_size=$gpu_num,data_parallel_size=1,gpu_memory_utilization=0.85,dtype=bfloat16"
batch_size="auto"

# -------------------- Run --------------------
cd "$PIP_PATH"

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        output_file="${output_base}/${task}"
        mkdir -p "$output_file"

        if [[ "$model" == *"Qwen2.5-7B-Instruct"* ]] || \
           [[ "$model" == *"Qwen2.5-Math-7B-Instruct"* ]] || \
           [[ "$model" == *"Qwen-2.5-Math-7B-SimpleRL-Zoo"* ]]; then
            max_model_tokens=4096
            max_gen_tokens=4096
        else
            max_model_tokens=$default_max_model_tokens
            max_gen_tokens=$default_max_gen_tokens
        fi

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
