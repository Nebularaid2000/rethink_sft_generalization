#!/bin/bash

ckpt_path_list=(    
    /path/to/model/global_step_10
    /path/to/model/global_step_20
    /path/to/model/global_step_40
    # ...
)

for ckpt_path in "${ckpt_path_list[@]}"; do
    local_dir=$ckpt_path

    # find the last global_step_xxx directory
    while [[ $(basename "$ckpt_path") != global_step_* && "$ckpt_path" != "/" ]]; do
        ckpt_path=$(dirname "$ckpt_path")
    done

    if [[ $(basename "$ckpt_path") == global_step_* ]]; then
        parent_dir=$(dirname "$ckpt_path")
        base_dir=$(basename "$parent_dir")
        step=$(basename "$ckpt_path" | sed 's/^global_step_//')
        new_path="${parent_dir}/merged_step${step}"
        echo "$new_path"
    else
        echo "Error: No global_step_xxx directory found in path"
    fi

    echo "local dir: $local_dir"
    echo "target dir: $new_path"

    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir $local_dir \
        --target_dir $new_path \
        --trust_remote_code

done
