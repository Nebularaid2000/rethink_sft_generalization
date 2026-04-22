import os, time
import json
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
from utils import DATASET_KEYS, RESPONSE_EXTRACTOR
import pandas as pd
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer
from math_verify import parse, verify
import matplotlib.pyplot as plt
import random
import numpy as np
import torch

DRAW_DIR_NAME = "draw"

MAX_AIME=5000
MAX_MATH500=2000
MAX_GSM8K=1000
MAX_AMC=3000
MAX_OLYMPIADBENCH=5000
MAX_MINERVA=3000


budget_nums = 10

DATASET_CONFIGS = {
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/converted_aime_dataset': {
        'loader': lambda name: load_from_disk(name),
        'test_n': 10,
        'temperature': 0.6, 
        'max_samples': 32768,
        'top_p': 0.95,
        'budget_list': np.arange(int(MAX_AIME/budget_nums), int(MAX_AIME/budget_nums + MAX_AIME), int(MAX_AIME/budget_nums)).tolist(),
        'save_name': 'aime'
    },
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/MATH500': {
        'loader': lambda name: load_dataset(name),
        'test_n': 3, 
        'temperature': 0.6,
        'top_p': 0.95,
        'max_samples': 32768,
        'budget_list': np.arange(int(MAX_MATH500/budget_nums), int(MAX_MATH500/budget_nums + MAX_MATH500), int(MAX_MATH500/budget_nums)).tolist(),
        'save_name': 'math500'
    },
    'openai/gsm8k': {
        'loader': lambda name: load_dataset(name, 'main'),
        'test_n': 1, 'temperature': 0.6, 'max_samples': 1319,'top_p': 1.0,
        'budget_list': np.arange(int(MAX_GSM8K/budget_nums), int(MAX_GSM8K/budget_nums + MAX_GSM8K), int(MAX_GSM8K/budget_nums)).tolist(),
        'save_name': 'gsm8k'
    },
    'amc': {
        'loader': lambda _: read_jsonl_file("datasets/aimo-validation-amc.jsonl"),
        'test_n': 10, 'temperature': 0.6, 'max_samples': -1,'top_p': 1.0,
        'budget_list': np.arange(int(MAX_AMC/budget_nums), int(MAX_AMC/budget_nums + MAX_AMC), int(MAX_AMC/budget_nums)).tolist(),
        'save_name': 'amc'
    },
    'olympiadbench': {
        'loader': lambda _: read_jsonl_file("datasets/olympiadbench/test.jsonl"),
        'test_n': 1, 'temperature': 0.6, 'top_p': 1.0,
        'max_samples': 32768,
        'budget_list': np.arange(int(MAX_OLYMPIADBENCH/budget_nums), int(MAX_OLYMPIADBENCH/budget_nums + MAX_OLYMPIADBENCH), int(MAX_OLYMPIADBENCH/budget_nums)).tolist(),
        'save_name': 'olympiadbench'
    },
    'minerva_math': {
        'loader': lambda _: read_jsonl_file("datasets/minerva_math/test.jsonl"),
        'test_n': 10, 'temperature': 0.6, 'max_samples': -1,'top_p': 1.0,
        'budget_list': np.arange(int(MAX_MINERVA/budget_nums), int(MAX_MINERVA/budget_nums + MAX_MINERVA), int(MAX_MINERVA/budget_nums)).tolist(),
        'save_name': 'minerva'
    },
    'humanoid_openr1_math': {
        'loader': lambda _: read_jsonl_file("/mnt/shared-storage-user/wangpeng/eval/humanoid_data_1k_fortest.jsonl"),
        'test_n': 3, 
        'temperature': 0.6,
        'top_p': 0.95,
        'max_samples': -1,
        'budget_list': np.arange(int(MAX_MATH500/budget_nums), int(MAX_MATH500/budget_nums + MAX_MATH500), int(MAX_MATH500/budget_nums)).tolist(),
        'save_name': 'humanoid_openr1_math'
    },
}

def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def build_prompt(x):
    message = [
        {"role": "user", "content": "{}\n Please reason step by step, and put your final answer within \\boxed{{}}.".format(x[QUESTION_KEY])}
    ]

    return message


def get_most_common(solns):
    soln_counts = {}
    for soln in solns:
        if len(soln) == 0:
            continue
        found = False
        for key in soln_counts:
            if verify(soln, key):
                soln_counts[key] += 1
                found = True
                break
        if not found:
            try:
                soln_counts[soln[0]] = 1
            except:
                soln_counts[soln[1]] = 1
    if len(soln_counts) == 0:
        return ""
    return max(soln_counts, key=soln_counts.get)


def truncate_responses_by_budget(responses, token_ids_list, budget, tokenizer):
    result = [None] * len(responses)
    to_decode = []
    indices_to_decode = []

    for i, (resp, token_ids) in enumerate(zip(responses, token_ids_list)):
        if len(token_ids) <= budget:
            result[i] = resp
        else:
            to_decode.append(token_ids[:budget])
            indices_to_decode.append(i)

    if to_decode:
        decoded_batch = tokenizer.batch_decode(to_decode, skip_special_tokens=True)
        for idx, decoded in zip(indices_to_decode, decoded_batch):
            result[idx] = decoded

    return result


def save_results_markdown_and_plot(scores, budgets, save_dir, dataset, model, types, limit_tokens):
    # Print markdown table
    markdown_lines = ["| Budget | pass@1 | pass@k(majority) | average_pass_rate |",
                      "|--------|--------|------------------|--------------------|"]
    for b in budgets:
        line = f"| {b} | {scores[b]['pass@1']:.4f} | {scores[b]['pass@k(majority)']:.4f} | {scores[b]['average_pass_rate']:.4f} |"
        markdown_lines.append(line)

    markdown_text = "\n".join(markdown_lines)
    print("\nMarkdown Table:\n")
    print(markdown_text)

    # Save to markdown file
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, DRAW_DIR_NAME, f"{dataset}_{types}"), exist_ok=True)
    with open(os.path.join(args.output_dir, DRAW_DIR_NAME, f"{dataset}_{types}/{model}.md"), 'w') as f:
        f.write(markdown_text)

    # Draw plot
    plt.figure()
    plt.plot(budgets, [scores[b]['pass@1'] for b in budgets], marker='o', label="pass@1")
    plt.plot(budgets, [scores[b]['pass@k(majority)'] for b in budgets], marker='s', label="pass@k(majority)")
    plt.plot(budgets, [scores[b]['average_pass_rate'] for b in budgets], marker='^', label="average_pass_rate")
    plt.xlabel("Budget")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Budget")
    plt.legend()
    plt.grid(True)

    # Save figure
    plot_path = os.path.join(save_dir, "accuracy_vs_budget.png")
    plt.savefig(plot_path)
    plt.close()


def save_results_for_native(scores, save_dir, title):
    # Print markdown table
    markdown_lines = ["| pass@1 | pass@k(majority) | average_pass_rate | avearage_length |",
                      "|--------|------------------|--------------------| -----------------|"]
    
    line = f"| {scores['pass@1']:.4f} | {scores['pass@k(majority)']:.4f} | {scores['average_pass_rate']:.4f} | {scores['average_length']:.4f} |"
    markdown_lines.append(line)

    markdown_text = "\n".join(markdown_lines)
    print("\nMarkdown Table:\n")
    print(markdown_text)

    # Save to markdown file
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"{title}_native.md"), 'w') as f:
        f.write(markdown_text)


def set_seed(seed) -> None:
    """ set random seed for all related packages
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'backends'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # ============= Main function ==============
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--tok_limit', type=int, default=32768)
    parser.add_argument('--tok_limit_instruct', type=int, default=512)
    parser.add_argument('--types', type=str, default="no_instruct")
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default="eval_results")
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    # Init args
    model_path = args.model_path
    dataset_name = args.dataset
    tok_limit = args.tok_limit
    tok_limit_instruct = args.tok_limit_instruct
    types = args.types
    results = {}

    print("Dataset:", dataset_name)

    QUESTION_KEY = DATASET_KEYS[dataset_name]["question"]
    ANSWER_KEY = DATASET_KEYS[dataset_name]["answer"]
    # eq = RESPONSE_COMPARATOR[dataset_name]  # use math-verify instead

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    except:
        base_model_path = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print("Using base model tokenizer")


    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset config for {dataset_name} is not defined.")

    config = DATASET_CONFIGS[dataset_name]
    print("loading dataset...")
    dataset = config['loader'](dataset_name)
    TEST_N = config['test_n']
    MAX_TOKENS = tok_limit
    TEST_TEMPERATURE = config['temperature']
    MAX_TEST_SAMPLES = config['max_samples']
    budget_list = config['budget_list']
    top_p = config['top_p']
    dataset_save_name = config['save_name']


    def parse_output(ds, outputs, save_file_name=None):
        predictions, golds, results = [], [], []
        for input, output in zip(ds, outputs):
            gold = input[ANSWER_KEY]
            if isinstance(gold, list):
                gold = gold[0]
            gold = str(gold)
            ans_format = "\\boxed{{{}}}"
            if "boxed" in gold:
                gold_extracted = parse(gold)
            else:
                gold_extracted = parse(ans_format.format(gold))
            prediction = [RESPONSE_EXTRACTOR[dataset_name](resp.text) for resp in output.outputs]
            predictions.append(prediction)
            golds.append(gold)
            results.append({
                QUESTION_KEY: input[QUESTION_KEY],
                ANSWER_KEY: input[ANSWER_KEY],
                "responses": [resp.text for resp in output.outputs],
                "response_ids": [resp.token_ids for resp in output.outputs],
                "prediction": prediction,
                "gold": gold,
                "tokens": [len(resp.token_ids) for resp in output.outputs],
                "accuracy": [verify(gold_extracted, pred) for pred in prediction]
            })
        if save_file_name:
            with open(save_file_name, 'w') as f:
                json.dump(results, f, indent=4)
        return results


    def process_single_result(args):
        result, budget, extractor_name = args

        if  dataset_name == "olympiadbench":
            gold = str(result['final_answer'][0])
        else:
            gold = str(result['gold'])

        ans_format = "\\boxed{{{}}}"
        if "boxed" in gold:
            gold_extracted = parse(gold)
        else:
            gold_extracted = parse(ans_format.format(gold))

        responses = result["responses"]
        response_ids = result["response_ids"]

        truncated = truncate_responses_by_budget(responses, response_ids, budget, tokenizer)

        preds = []
        for r in truncated:
            try:
                preds.append(parse(r))
            except:
                preds.append(r)

        match_flags = [verify(gold_extracted, p) for p in preds] # gold first, pred second

        pass_1 = int(match_flags[0]) if match_flags else 0
        maj = int(verify(gold_extracted, get_most_common(preds))) if preds else 0
        avg = float(np.mean(match_flags)) if match_flags else 0

        return pass_1, maj, avg


    def process_single_result_no_truncation(result):

        if  dataset_name == "olympiadbench":
            gold = str(result['final_answer'][0])
        else:
            gold = str(result['gold'])

        
        ans_format = "\\boxed{{{}}}"
        if "boxed" in gold:
            gold_extracted = parse(gold)
        else:
            gold_extracted = parse(ans_format.format(gold))


        responses = result["responses"]
        response_ids = result["response_ids"]

        preds = []
        for r in responses:
            try:
                preds.append(parse(r))
            except:
                preds.append(r)

        match_flags = [verify(gold_extracted, p) for p in preds]

        pass_1 = int(match_flags[0]) if match_flags else 0
        maj = int(verify(gold_extracted, get_most_common(preds))) if preds else 0
        avg = float(np.mean(match_flags)) if match_flags else 0
        avg_lengths = np.mean([len(r) for r in response_ids]) if response_ids else 0

        return pass_1, maj, avg, avg_lengths


    def get_scores_parallel(results, budget_list=None, num_workers=None):

        print("cpu_count():",cpu_count)
        if budget_list is None:
            budget_list = list(range(200, 1500, 300))
        if num_workers is None:
            num_workers = max(cpu_count() - 1, 1)

        scores_per_budget = {}

        for budget in budget_list:
            args_list = [(result, budget, dataset_name) for result in results]

            with Pool(num_workers) as pool:
                all_scores = list(pool.imap_unordered(process_single_result, args_list))

            pass1, maj_acc, avg_pass = zip(*all_scores)

            scores_per_budget[budget] = {
                'pass@1': np.mean(pass1),
                'pass@k(majority)': np.mean(maj_acc),
                'average_pass_rate': np.mean(avg_pass),
            }
        
        # no budget and compute average lengths
        with Pool(num_workers) as pool:
            native_scores = list(pool.imap_unordered(process_single_result_no_truncation, results))
        pass1, maj_acc, avg_pass, avg_lengths = zip(*native_scores)
        
        saved_native_scores = {
            'pass@1': np.mean(pass1),
            'pass@k(majority)': np.mean(maj_acc),
            'average_pass_rate': np.mean(avg_pass),
            'average_length': np.mean(avg_lengths)
        }

        
        scores_per_budget['native'] = saved_native_scores
        return scores_per_budget


    def evaluate_model(model_name):
        saved_model_name = clean_model_name(model_name)
        print(f"Evaluating model: {saved_model_name}")
        save_path = f"{args.output_dir}/new_outputs/{saved_model_name}/{dataset_save_name}_budget{budget_list[-1]}/{types}_{tok_limit_instruct}.json"
        if os.path.exists(save_path):
            print("Found existing results, loading...")
            start_time = time.time()
            test_scores = get_scores_from_file(model_name)
            end_time = time.time()
            print("Test:", test_scores)
            print("Time taken:", end_time - start_time)
            test_scores['time'] = end_time - start_time
            return test_scores
        
        test_prompts = []
        model = LLM(model_name, gpu_memory_utilization=0.85, tensor_parallel_size=args.num_gpus, trust_remote_code=True)
        
        try:
            test_ds = dataset['test'].shuffle(seed=0).select(range(min(MAX_TEST_SAMPLES, len(dataset['test']))))
        except:
            test_ds = dataset
        
        for x in test_ds:
            prompt = build_prompt(x)
            prompt_tokens = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            test_prompts.append(prompt_tokens)

        sampling_params = SamplingParams(temperature=TEST_TEMPERATURE, 
                                        max_tokens=MAX_TOKENS, 
                                        n=TEST_N,
                                        top_p=top_p,
                                        seed=1234)
        sampling_params.stop_token_ids = [tokenizer.eos_token_id]

        print("Generating test outputs...")
        start_time = time.time()
        test_outputs = model.generate(test_prompts, sampling_params=sampling_params, use_tqdm=True)

        os.makedirs(f"{args.output_dir}/new_outputs/{saved_model_name}/{dataset_save_name}_budget{budget_list[-1]}", exist_ok=True)

        save_path = f"{args.output_dir}/new_outputs/{saved_model_name}/{dataset_save_name}_budget{budget_list[-1]}/{types}_{tok_limit_instruct}.json"
        results = parse_output(test_ds, test_outputs, save_path)
        print("cpu_count:",cpu_count())
        test_scores = get_scores_parallel(results, budget_list=budget_list,num_workers=max(cpu_count()-1,1))
        end_time = time.time()
        print("Test:", test_scores)
        print("Time taken:", end_time - start_time)
        test_scores['time'] = end_time - start_time
        return test_scores


    print("Found model_path:", model_path)
    print("This is not a checkpoint, will evaluate directly...")


    def clean_model_name(model_path):
        from pathlib import Path
        if "global_step" in model_path:
            step_note = model_path.split("/")[-1]
            assert step_note.startswith("global_step")
            step_num = step_note.split("_")[1]
            model_path_clean = model_path.removesuffix(f"/ckpt/{step_note}")
            saved_model_name = Path(model_path_clean).name
            saved_model_name = saved_model_name + f"_{step_num}"
        else:
            saved_model_name = model_path.split("/")[-2] + model_path.split("/")[-1]
        return saved_model_name.replace("/", "_")



    def get_scores_from_file(model_name):
        start_time = time.time()

        saved_model_name = clean_model_name(model_name)

        save_path = f"{args.output_dir}/new_outputs/{saved_model_name}/{dataset_save_name}_budget{budget_list[-1]}/{types}_{tok_limit_instruct}.json"
        with open(save_path, 'r') as f:
            results = json.load(f)
        test_scores = get_scores_parallel(results, budget_list=budget_list, num_workers=max(cpu_count() - 1,1))
        end_time = time.time()
        print("Test:", test_scores)
        print("Time taken:", end_time - start_time)
        test_scores['time'] = end_time - start_time
        return test_scores


    scores = evaluate_model(model_path)


    if scores:
        scores["model_path"] = model_path
        saved_model_name = clean_model_name(model_path)
        os.makedirs(f"{args.output_dir}/results/{saved_model_name}/{dataset_save_name}_budget{budget_list[-1]}", exist_ok=True)
        print(scores)

        save_path = f"{args.output_dir}/results/{saved_model_name}/{dataset_save_name}_budget{budget_list[-1]}/{types}_{tok_limit_instruct}.json"
        save_results_markdown_and_plot(scores, budget_list, f"{args.output_dir}/results/{saved_model_name}/{dataset_save_name}_budget{budget_list[-1]}", dataset_save_name, saved_model_name, types, tok_limit_instruct)
        if 'native' in scores:
            save_results_for_native(scores['native'], f"{args.output_dir}/results/{saved_model_name}/{dataset_save_name}_budget{budget_list[-1]}", f"{types}_{tok_limit_instruct}")

        with open(save_path, 'w') as f:
            json.dump(scores, f, indent=4)
