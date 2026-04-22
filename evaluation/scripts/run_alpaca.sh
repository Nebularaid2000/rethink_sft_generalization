#!/bin/bash
# ==============================================================================
# alpaca_eval: generate responses + evaluate with reward model
# ==============================================================================

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
