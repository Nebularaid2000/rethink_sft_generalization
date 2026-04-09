#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

TOKENIZER_PATH = "/mnt/shared-storage-user/ai4good1-share/hf_hub/Qwen/Qwen3-8B-Base"

def total_response_tokens(parquet_path: Path, tokenizer, batch_size: int = 256) -> int:
    df = pd.read_parquet(parquet_path, columns=["response"])
    if "response" not in df.columns:
        raise KeyError(f"'response' column not found in {parquet_path}")

    responses = df["response"].fillna("").astype(str).tolist()
    total = 0

    for i in range(0, len(responses), batch_size):
        batch = responses[i : i + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        total += sum(len(ids) for ids in encoded["input_ids"])

    return total


def max_prompt_response_tokens(parquet_path: Path, tokenizer, batch_size: int = 256) -> tuple[int, int]:
    df = pd.read_parquet(parquet_path, columns=["prompt", "response"])

    if "prompt" not in df.columns:
        raise KeyError(f"'prompt' column not found in {parquet_path}")
    if "response" not in df.columns:
        raise KeyError(f"'response' column not found in {parquet_path}")

    prompts = df["prompt"].fillna("").astype(str).tolist()
    responses = df["response"].fillna("").astype(str).tolist()

    max_prompt = 0
    max_response = 0

    for i in range(0, len(prompts), batch_size):
        prompt_batch = prompts[i : i + batch_size]
        response_batch = responses[i : i + batch_size]

        prompt_encoded = tokenizer(
            prompt_batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        response_encoded = tokenizer(
            response_batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        if prompt_encoded["input_ids"]:
            max_prompt = max(max_prompt, max(len(ids) for ids in prompt_encoded["input_ids"]))
        if response_encoded["input_ids"]:
            max_response = max(max_response, max(len(ids) for ids in response_encoded["input_ids"]))

    return max_prompt, max_response


def main():
    parser = argparse.ArgumentParser(
        description="Count total tokens in parquet 'response' column and compare ratio."
    )
    parser.add_argument("--a", required=True, help="Path to parquet A (numerator).")
    parser.add_argument("--b", required=True, help="Path to parquet B (denominator).")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for tokenization.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    total_a = total_response_tokens(Path(args.a), tokenizer, args.batch_size)
    total_b = total_response_tokens(Path(args.b), tokenizer, args.batch_size)
    max_prompt_a, max_response_a = max_prompt_response_tokens(Path(args.a), tokenizer, args.batch_size)
    max_prompt_b, max_response_b = max_prompt_response_tokens(Path(args.b), tokenizer, args.batch_size)

    ratio = total_a / total_b if total_b else float("inf")

    print(f"A: {args.a}")
    print(f"B: {args.b}")
    print(f"tokenizer = {TOKENIZER_PATH}")
    print(f"total_response_tokens_A = {total_a}")
    print(f"total_response_tokens_B = {total_b}")
    print(f"max_prompt_tokens_A = {max_prompt_a}")
    print(f"max_response_tokens_A = {max_response_a}")
    print(f"max_prompt_tokens_B = {max_prompt_b}")
    print(f"max_response_tokens_B = {max_response_b}")
    print(f"A_div_B = {ratio:.10f}")


if __name__ == "__main__":
    main()
