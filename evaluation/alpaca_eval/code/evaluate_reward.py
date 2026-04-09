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
    parser.add_argument('--input_name', type=str, required=True)
    args = parser.parse_args()

    model_dir = '/cfs_oss/shared/ai4good1/wangpeng/allenai/Llama-3.1-8B-Instruct-RM-RB2'

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

    reward_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'results', 'alpaca', 'rewards')
    reward_base = os.path.abspath(reward_base)
    os.makedirs(os.path.join(reward_base, f'rewards_{args.task_name}'), exist_ok=True)
    prompts = []
    outputs = []
    fname = args.input_name.replace('/', '-')
    if args.task_name == "ultrafeedback":
        with open(f"outputs_{args.task_name}/{args.input_name}.jsonl", "r") as f:
            for line in f:
                data = json.loads(line)
                prompts.append(data["prompt"])
                outputs.append(data["output"])
    elif args.task_name == "alpaca":
        output_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'results', 'alpaca', 'outputs')
        output_base = os.path.abspath(output_base)
        with open(f"{output_base}/{fname}.json", "r") as f:
            data = json.load(f)
        for d in data:
            prompts.append(d["instruction"])
            outputs.append(extract_answer(d["output"]))

    chats = [[{'role': 'user', 'content': str(prompt)}, {'role': 'assistant', 'content': str(output)},
              ] for prompt, output in zip(prompts, outputs)]
    texts = [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(tokenizer.bos_token, "") for chat in chats]

    logging.info(f"Running {fname}")
    text_dataset = ListDataset(texts)
    pipe_outputs = [e for e in tqdm(rm_pipe(text_dataset, **pipe_kwargs), total=len(texts), dynamic_ncols=True)]
    logging.info(f"Done {fname}")
    rewards = [output[0]["score"] for output in pipe_outputs]
    with open(f"{reward_base}/rewards_{args.task_name}/RB2_fullleft_{fname}.jsonl", "w") as f:
        for reward in rewards:
            f.write(json.dumps(reward) + "\n")


if __name__ == '__main__':
    main()
