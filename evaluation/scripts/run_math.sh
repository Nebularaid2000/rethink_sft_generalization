#!/bin/bash
# ==============================================================================
# math_eval: MATH500, AIME24
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

export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# -------------------- Install --------------------
pip install word2number

# -------------------- Config (MODIFY HERE) --------------------
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ROOT="${UNIFIED_ROOT}/math_eval"
cd "$ROOT"

OUTPUT="${UNIFIED_ROOT}/results/math"

datasets=(
    ${UNIFIED_ROOT}/data/math/MATH500
    # ${UNIFIED_ROOT}/data/math/converted_aime_dataset
)

models=(
    # TODO: Add your model paths here
    # /path/to/model/merged_step10
    # /path/to/model/merged_step40
    # /mnt/shared-storage-user/ai4good1-share/hf_hub/Qwen/Qwen3-14B-Base
    /cfs_oss/shared/ai4good1/renqihan/ckpt/offline_H/sft_qw3-base-14b_new-c1-20.5k-16384_vanilla_lr5e-5_ep8_token-mean_bs256/merged_step640
)

limit_tokens=(
    512
)

mode=(
    no_instruct
)

# -------------------- Run --------------------
for path in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for tokens in "${limit_tokens[@]}"; do
            for m in "${mode[@]}"; do
                echo "==== [math] model=$(basename "$path") dataset=$(basename "$dataset") mode=$m tokens=$tokens ===="
                start_time=$(date +%s)
                python math_eval_budget.py \
                    --model_path=${path} \
                    --dataset=${dataset} \
                    --tok_limit_instruct=${tokens} \
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
done
