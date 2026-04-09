formatted_date=$(date "+%Y-%m-%d-%H-%M-%S")

ROOT_DIR=/path/to/this/repo
cd $ROOT_DIR
JOB_LOG="${ROOT_DIR}/log/${formatted_date}.log"
JOB_ERR="${ROOT_DIR}/log/${formatted_date}.err"

export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export CUDA_LAUNCH_BLOCKING=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

export WANDB_API_KEY=your_wandb_key
export WANDB_MODE=offline
PROJ_NAME=sft_generalization

MODEL_PATH=Qwen/Qwen3-8B-Base
TRAIN_DATA=/path/to/Math-CoT-20k
VAL_DATA=$TRAIN_DATA
EPOCHS=8
TBS=256
micro_bsz=4
LR=5e-5
IMPORTANCE=vanilla
CLIP_RATIO_LOW=1.0
CLIP_RATIO_HIGH=1.0
LOSS_AGG_MODE=token-mean
ref_log_prob_enable=True

OFFLOAD=False

RUN_NAME=Qwen3-8B_Math-CoT-20k_lr5e-5_ep8_bs256
SAVE_PATH=ckpt/${RUN_NAME}

MASTER_PORT=$((RANDOM % 1001 + 20000))

torchrun --nnodes=$NODE_COUNT --nproc_per_node=$PROC_PER_NODE \
    --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer_ours \
    data.eos_token=null \
    data.eos_token_id=null \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.prompt_key=message \
    data.response_key=response \
    data.logprob_key=logprob \
    data.advantage_key=advantage \
    data.train_batch_size=$TBS \
    data.micro_batch_size_per_gpu=$micro_bsz \
    data.max_length=20000 \
    data.truncation=right \
    data.shuffle_train=False \
    data.ref_log_prob.enable=$ref_log_prob_enable \
    data.ref_log_prob.only_return_mean_logprob=False \
    model.partial_pretrain=$MODEL_PATH \
    model.fsdp_config.model_dtype=bf16 \
    model.trust_remote_code=True \
    model.fsdp_config.cpu_offload=$OFFLOAD \
    optim.lr=$LR \
    optim.betas="[0.9,0.999]" \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.nnodes=$NODE_COUNT \
    trainer.project_name=$PROJ_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.total_epochs=$EPOCHS \
    trainer.save_freq=10 \
    trainer.logger='["console","wandb"]' \
    trainer.importance_sampling_mode=$IMPORTANCE \
    trainer.clip_ratio_low=$CLIP_RATIO_LOW \
    trainer.clip_ratio_high=$CLIP_RATIO_HIGH \
    trainer.loss_agg_mode=$LOSS_AGG_MODE \
    trainer.checkpoint.save_contents='["model"]' \
    trainer.resume_mode=disable \
    >> $JOB_LOG 2>> $JOB_ERR
