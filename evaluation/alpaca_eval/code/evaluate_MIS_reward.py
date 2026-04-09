import os
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
import argparse
from datasets import disable_caching
disable_caching()
from utils import ListDataset


import warnings
import logging
logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
warnings.simplefilter("ignore")

import re


def extract_answer(text: str) -> str:
    """
    删除最后一个 </think> 及其之前的所有内容，返回剩下的文本。
    如果没有 </think>，则返回原始文本。
    """
    parts = re.split(r'</think>', text)
    if len(parts) > 1:
        return parts[-1].strip()
    else:
        return text.strip()


def main():
    logging.info(f'cuda is available {torch.cuda.is_available()}')
    logging.info(f'cuda device count {torch.cuda.device_count()}')
    logging.info(f'cuda device name {torch.cuda.get_device_name()}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, choices=['ultrafeedback', 'alpaca'])
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input json/jsonl files')
    args = parser.parse_args()

    # model_dir = '/mnt/shared-storage-user/ai4good1-share/wangpeng/Skywork-Reward-V2-Llama-3.1-8B'
    model_dir = '/mnt/shared-storage-user/ai4good1-share/wangpeng/RM-Mistral-7B'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=model_dir,
        device_map='auto',
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 1,
        "num_workers": 20,
    }

    output_dir = f'/mnt/shared-storage-user/wangpeng/LLM-Extrapolation/reward/MISRM_rewards_{args.task_name}'
    os.makedirs(output_dir, exist_ok=True)

    # 获取指定文件夹下所有json/jsonl文件
    if args.task_name == "ultrafeedback":
        input_files = sorted([
            f for f in os.listdir(args.input_dir)
            if f.endswith('.jsonl')
        ])
    elif args.task_name == "alpaca":
        input_files = sorted([
            f for f in os.listdir(args.input_dir)
            if f.endswith('.json')
        ])

    logging.info(f"Found {len(input_files)} files in {args.input_dir}: {input_files}")

    for input_file in input_files:
        # 从文件名获取fname（去掉扩展名）
        fname = os.path.splitext(input_file)[0]
        input_path = os.path.join(args.input_dir, input_file)
        output_path = os.path.join(output_dir, f" MISRM_{fname}.jsonl")

        # 如果输出文件已存在，跳过
        if os.path.exists(output_path):
            logging.info(f"Skipping {fname}, output already exists: {output_path}")
            continue

        logging.info(f"Processing file: {input_path}")

        prompts = []
        outputs = []

        if args.task_name == "ultrafeedback":
            with open(input_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    prompts.append(data["prompt"])
                    outputs.append(data["output"])
        elif args.task_name == "alpaca":
            with open(input_path, "r") as f:
                data = json.load(f)
            for d in data:
                prompts.append(d["instruction"])
                outputs.append(extract_answer(d["output"]))

        chats = [[
            {'role': 'user', 'content': str(prompt)},
            {'role': 'assistant', 'content': str(output)},
        ] for prompt, output in zip(prompts, outputs)]

        texts = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "")
            for chat in chats
        ]

        logging.info(f"Running {fname} ({len(texts)} samples)")
        text_dataset = ListDataset(texts)
        pipe_outputs = [e for e in tqdm(rm_pipe(text_dataset, **pipe_kwargs), total=len(texts), dynamic_ncols=True, desc=fname)]
        logging.info(f"Done {fname}")

        rewards = [output[0]["score"] for output in pipe_outputs]
        with open(output_path, "w") as f:
            for reward in rewards:
                f.write(json.dumps(reward) + "\n")

        logging.info(f"Saved rewards to {output_path}")

    logging.info("All files processed.")


if __name__ == '__main__':
    main()