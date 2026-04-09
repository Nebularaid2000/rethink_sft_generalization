#!/bin/bash
# ==============================================================================
# alpaca_eval: generate responses + evaluate with reward model
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
source /mnt/shared-storage-user/wangpeng/.bashrc
export PATH=/usr/bin:/usr/local/cuda/bin:/home/renqihan/.local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=/usr/bin/python:$PYTHONPATH
export PYTHONUSERBASE=/home/renqihan/.local
export HF_HUB_CACHE=/mnt/shared-storage-user/renqihan/models
export HF_HUB_OFFLINE=1

# -------------------- Config (MODIFY HERE) --------------------
export CUDA_VISIBLE_DEVICES=0,1

UNIFIED_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${UNIFIED_ROOT}/alpaca_eval/code"
cd "$ROOT"

OUTPUT_DIR="${UNIFIED_ROOT}/results/alpaca/outputs"
REWARD_DIR="${UNIFIED_ROOT}/results/alpaca/rewards"
mkdir -p "$OUTPUT_DIR" "$REWARD_DIR"

USE_SYS_PROMPT=False

models=(
    # TODO: Add your model paths here
    /path/to/model/merged_step10
    /path/to/model/merged_step40
)

# -------------------- Step 1: Generate --------------------
echo "==== [alpaca] Step 1: Generate responses ===="
for path in "${models[@]}"; do
    echo "  Generating for: $(basename $(dirname "$path"))/$(basename "$path")"
    python generate_alpaca.py \
        --pretrained_model_path=${path} \
        --output_path=${OUTPUT_DIR} \
        --use_system_prompt=${USE_SYS_PROMPT}
done

# -------------------- Step 2: Evaluate with reward model --------------------
echo "==== [alpaca] Step 2: Evaluate with reward model ===="
for path in "${models[@]}"; do
    echo "  Evaluating for: $(basename $(dirname "$path"))/$(basename "$path")"
    python3 evaluate_reward.py --task_name 'alpaca' --input_name=${path}
done

echo "==== [alpaca] Done ===="
