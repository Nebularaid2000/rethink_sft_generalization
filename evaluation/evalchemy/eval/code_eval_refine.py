# # """
# # 只跑评测，跳过生成步骤
# # """
# # import json
# # import logging

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # from eval.chat_benchmarks.LiveCodeBench.eval_instruct import LiveCodeBenchBenchmark

# # # 1. 实例化 benchmark
# # benchmark = LiveCodeBenchBenchmark(
# #     debug=False,
# #     logger=logger,
# # )

# # # 2. 加载之前保存的生成结果
# # with open("/mnt/shared-storage-user/wangpeng/Transferability-of-LLM-Reasoning/eval/evalchemy/real_results/outputs/GPQA_new/__cfs_oss__shared__ai4good1__renqihan__ckpt__offline_H__sft_gemma3-base-12b_new-c1-20.5k-16384_vanilla_lr5e-5_ep8_token-mean_bs256__merged_step10/results_2025-12-28T02-25-54.771372.json", "r") as f:
# #     generation_results = json.load(f)

# # # 3. 直接评测
# # eval_results = benchmark.evaluate_responses(generation_results["results"]["LiveCodeBench"])

# # print(json.dumps(
# #     {k: v for k, v in eval_results.items() if k not in ["examples", "raw_metrics"]},
# #     indent=2, default=str
# # ))


# """
# eval/code_eval_refine.py
# 从已有JSON结果文件 + 原始数据集，重新评测 LiveCodeBench
# """
# import sys
# import os
# import json
# import copy
# import logging
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import numpy as np
# from datasets import load_dataset, concatenate_datasets

# # 路径设置
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, ROOT_DIR)

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# from eval.chat_benchmarks.LiveCodeBench.eval_instruct import (
#     LiveCodeBenchBenchmark,
#     has_code,
#     calc_stats,
# )
# from eval.chat_benchmarks.LiveCodeBench.livecodebench_utils import (
#     translate_private_test_cases,
#     map_to_example,
# )


# def load_and_process_dataset():
#     """加载并处理原始数据集，与 load_questions() 逻辑一致"""
#     logger.info("Loading original dataset...")
#     ds = load_dataset(
#         "/mnt/shared-storage-user/renqihan/dataset/livecodebench/code_generation_lite",
#         version_tag="release_v2",
#         split="test",
#         trust_remote_code=True,
#     )
#     logger.info(f"Raw dataset size: {len(ds)}")

#     cpu_count = os.cpu_count()
#     processed_shards = []
#     num_shards = 4
#     for i in range(num_shards):
#         shard = ds.shard(num_shards=num_shards, index=i)
#         shard = shard.map(
#             lambda example: {
#                 "private_test_cases": translate_private_test_cases(example["private_test_cases"])
#             },
#             num_proc=cpu_count,
#         )
#         shard = shard.map(map_to_example, remove_columns=ds.column_names)
#         processed_shards.append(shard)
#     ds = concatenate_datasets(processed_shards)
#     logger.info(f"Processed dataset size: {len(ds)}")
#     return ds


# def build_dataset_map(ds):
#     """
#     构建 prompt -> dataset_example 的映射
#     用 prompt 前500字符作为key（避免完全相同prompt的极端情况）
#     """
#     dataset_map = {}
#     for ex in ds:
#         # 用完整 prompt 做 key
#         key = ex["prompt"].strip()
#         dataset_map[key] = dict(ex)
#     logger.info(f"Dataset map size: {len(dataset_map)}")
#     return dataset_map


# def load_json_results(json_path: str) -> list:
#     """加载JSON评测结果文件"""
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     data = data["results"]["LiveCodeBench"]

#     if isinstance(data, dict) and "examples" in data:
#         return data["examples"]
#     elif isinstance(data, list):
#         return data
#     else:
#         raise ValueError(f"Unrecognized JSON format in {json_path}")


# def match_json_to_dataset(json_examples, dataset_map):
#     """
#     将JSON中的example与数据集匹配
#     返回匹配成功的 (json_ex, ds_ex) 列表
#     """
#     matched = []
#     unmatched = 0

#     for json_ex in json_examples:
#         key = json_ex["prompt"].strip()
#         ds_ex = dataset_map.get(key)

#         if ds_ex is None:
#             # 尝试模糊匹配（前200字符）
#             for ds_key, ds_val in dataset_map.items():
#                 if ds_key[:200] == key[:200]:
#                     ds_ex = ds_val
#                     break

#         if ds_ex is not None:
#             matched.append((json_ex, ds_ex))
#         else:
#             unmatched += 1
#             logger.warning(f"Unmatched prompt (first 80 chars): {key[:80]}...")

#     logger.info(f"Matched: {len(matched)}, Unmatched: {unmatched}")
#     return matched


# def evaluate_from_json(
#     json_path: str,
#     max_workers: int = 32,
#     output_path: str = "eval_results.json",
# ):
#     """
#     主评测函数：
#     1. 加载JSON结果
#     2. 加载原始数据集（获取 private_test_cases）
#     3. 匹配 & 逐代码块评测
#     """
#     # ========== 1. 加载JSON ==========
#     json_examples = load_json_results(json_path)
#     logger.info(f"Loaded {len(json_examples)} examples from JSON")

#     # 看一下第一个example的结构
#     first = json_examples[0]
#     logger.info(f"JSON example keys: {list(first.keys())}")
#     logger.info(f"Number of code blocks in content: {len(first['content'])}")

#     # ========== 2. 加载数据集 ==========
#     ds = load_and_process_dataset()
#     dataset_map = build_dataset_map(ds)

#     # ========== 3. 匹配 ==========
#     matched = match_json_to_dataset(json_examples, dataset_map)

#     # ========== 4. 构建评测任务 ==========
#     benchmark = LiveCodeBenchBenchmark(debug=False, logger=logger)

#     eval_tasks = []
#     task_meta = []  # 记录每个task属于哪个example的哪个code block

#     for match_idx, (json_ex, ds_ex) in enumerate(matched):
#         contents = json_ex["content"]  # list of code strings

#         for code_idx, code_str in enumerate(contents):
#             # 合并数据集字段 + JSON字段
#             task = copy.deepcopy(ds_ex)
#             task["difficulty"] = json_ex["difficulty"]
#             task["model_output"] = json_ex.get("output", "")

#             # model_answer: evaluate_single_example 会取 [-1]
#             if isinstance(code_str, list):
#                 task["model_answer"] = code_str
#             else:
#                 task["model_answer"] = [code_str]

#             eval_tasks.append(task)
#             task_meta.append({
#                 "match_idx": match_idx,
#                 "code_idx": code_idx,
#                 "difficulty": json_ex["difficulty"],
#             })

#     logger.info(f"Total evaluation tasks: {len(eval_tasks)}")
#     logger.info(f"({len(matched)} examples × ~{len(json_examples[0]['content'])} code blocks each)")

#     # ========== 5. 并发评测 ==========
#     logger.warning("Expect some stdout leaks from code execution...")

#     ordered_results = [None] * len(eval_tasks)

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_idx = {}
#         for i, task in enumerate(eval_tasks):
#             future = executor.submit(benchmark.evaluate_single_example, task)
#             future_to_idx[future] = i

#         done_count = 0
#         for future in as_completed(future_to_idx):
#             idx = future_to_idx[future]
#             try:
#                 result = future.result()
#                 ordered_results[idx] = result
#             except Exception as e:
#                 logger.error(f"Error evaluating task {idx}: {e}")
#                 ordered_results[idx] = {
#                     "correctness": False,
#                     "reason": f"Error: {e}",
#                     "difficulty": task_meta[idx]["difficulty"],
#                 }
#             done_count += 1
#             if done_count % 100 == 0:
#                 logger.info(f"Progress: {done_count}/{len(eval_tasks)}")

#     # ========== 6. 统计结果 ==========

#     # 6a. 总体统计（所有代码块）
#     total_correct = sum(1 for r in ordered_results if r and r.get("correctness"))
#     total = len(ordered_results)

#     per_difficulty_correct = defaultdict(int)
#     per_difficulty_total = defaultdict(int)
#     for r, meta in zip(ordered_results, task_meta):
#         diff = meta["difficulty"]
#         per_difficulty_total[diff] += 1
#         if r and r.get("correctness"):
#             per_difficulty_correct[diff] += 1

#     logger.info(f"\n{'='*60}")
#     logger.info(f"[All code blocks] Total: {total_correct}/{total} = {total_correct/total:.2%}")
#     for diff in sorted(per_difficulty_total.keys()):
#         c = per_difficulty_correct[diff]
#         t = per_difficulty_total[diff]
#         logger.info(f"  {diff}: {c}/{t} = {c/t:.2%}")

#     # 6b. pass@k 统计（按 example 分组）
#     # 对每个 example，只要有一个 code block 通过就算 pass
#     example_results = defaultdict(list)
#     for r, meta in zip(ordered_results, task_meta):
#         example_results[meta["match_idx"]].append(r)

#     pass_at_1_correct = 0  # 第一个代码块通过
#     pass_at_any_correct = 0  # 任意代码块通过（pass@k）
#     per_diff_pass_any = defaultdict(int)
#     per_diff_example_total = defaultdict(int)

#     for match_idx, results_list in example_results.items():
#         diff = task_meta[match_idx * len(json_examples[0]["content"])]["difficulty"]
#         per_diff_example_total[diff] += 1

#         # pass@1: 第一个代码块
#         if results_list[0] and results_list[0].get("correctness"):
#             pass_at_1_correct += 1

#         # pass@any: 任意一个通过
#         if any(r and r.get("correctness") for r in results_list):
#             pass_at_any_correct += 1
#             per_diff_pass_any[diff] += 1

#     num_examples = len(example_results)
#     logger.info(f"\n[Per example] pass@1: {pass_at_1_correct}/{num_examples} = {pass_at_1_correct/num_examples:.2%}")
#     logger.info(f"[Per example] pass@any: {pass_at_any_correct}/{num_examples} = {pass_at_any_correct/num_examples:.2%}")
#     for diff in sorted(per_diff_example_total.keys()):
#         c = per_diff_pass_any[diff]
#         t = per_diff_example_total[diff]
#         logger.info(f"  {diff} pass@any: {c}/{t} = {c/t:.2%}")
#     logger.info(f"{'='*60}")

#     # ========== 7. 保存结果 ==========
#     save_results = {
#         "all_code_blocks": {
#             "total_correct": total_correct,
#             "total": total,
#             "accuracy": total_correct / total,
#             "per_difficulty": {
#                 diff: {
#                     "correct": per_difficulty_correct[diff],
#                     "total": per_difficulty_total[diff],
#                     "accuracy": per_difficulty_correct[diff] / per_difficulty_total[diff],
#                 }
#                 for diff in per_difficulty_total
#             },
#         },
#         "per_example": {
#             "num_examples": num_examples,
#             "pass_at_1": pass_at_1_correct / num_examples,
#             "pass_at_any": pass_at_any_correct / num_examples,
#             "per_difficulty_pass_any": {
#                 diff: {
#                     "correct": per_diff_pass_any[diff],
#                     "total": per_diff_example_total[diff],
#                     "accuracy": per_diff_pass_any[diff] / per_diff_example_total[diff],
#                 }
#                 for diff in per_diff_example_total
#             },
#         },
#     }

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(save_results, f, indent=2, ensure_ascii=False)
#     logger.info(f"Results saved to {output_path}")

#     return save_results


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Evaluate LiveCodeBench from JSON results")
#     parser.add_argument("--json_path", type=str, required=True,
#                         help="Path to JSON results file")
#     parser.add_argument("--max_workers", type=int, default=32,
#                         help="Number of parallel threads")
#     parser.add_argument("--output", type=str, default="eval_results.json",
#                         help="Output results file")
#     args = parser.parse_args()

#     evaluate_from_json(
#         json_path=args.json_path,
#         max_workers=args.max_workers,
#         output_path=args.output,
#     )

"""
eval/code_eval_refine.py
从已有JSON结果文件 + 原始数据集，重新评测 LiveCodeBench
（优化版：只评测最后一个代码块，与原始 evaluate_responses 逻辑一致）
"""
import sys
import os
import json
import copy
import re
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset, concatenate_datasets

# 路径设置
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from eval.chat_benchmarks.LiveCodeBench.eval_instruct import (
    LiveCodeBenchBenchmark,
    has_code,
    calc_stats,
)
from eval.chat_benchmarks.LiveCodeBench.livecodebench_utils import (
    translate_private_test_cases,
    map_to_example,
    post_process_code,
)


# def fix_underscore_indentation(code: str) -> str:
#     """
#     将每行 \\n 之后连续的、数量为4的倍数的下划线替换为等量空格。
#     例如:
#         ____return 1       -> '    return 1'
#         ________x = 2      -> '        x = 2'
#         ___y = 3           -> '___y = 3'  (3个不是4的倍数，不替换)
#         ____def foo():     -> '    def foo():'
#     """
#     def replace_leading_underscores(line):
#         # 匹配行首连续的下划线
#         match = re.match(r'^(_+)(.*)', line)
#         if match:
#             underscores = match.group(1)
#             rest = match.group(2)
#             count = len(underscores)
#             if count > 0 and count % 4 == 0:
#                 return ' ' * count + rest
#         return line

#     lines = code.split('\n')
#     fixed_lines = [replace_leading_underscores(line) for line in lines]
#     return '\n'.join(fixed_lines)

def fix_underscore_indentation(code: str) -> str:
    """
    将每行行首连续的、数量为4的倍数的 ▁ 或 _ 替换为等量空格。
    """
    def replace_leading_underscores(line: str) -> str:
        # 同时匹配普通下划线 _ 和 ▁ (U+2581)
        match = re.match(r'^([_▁]+)(.*)', line)
        if match:
            underscores = match.group(1)
            rest = match.group(2)
            count = len(underscores)
            if count > 0 and count % 4 == 0:
                return ' ' * count + rest
        return line
    lines = code.split('\n')
    fixed_lines = [replace_leading_underscores(line) for line in lines]
    return '\n'.join(fixed_lines)


def load_and_process_dataset():
    """加载并处理原始数据集，与 load_questions() 逻辑一致"""
    logger.info("Loading original dataset...")
    ds = load_dataset(
        "/mnt/shared-storage-user/renqihan/dataset/livecodebench/code_generation_lite",
        version_tag="release_v2",
        split="test",
        trust_remote_code=True,
    )
    logger.info(f"Raw dataset size: {len(ds)}")

    cpu_count = os.cpu_count()
    processed_shards = []
    num_shards = 4
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i)
        shard = shard.map(
            lambda example: {
                "private_test_cases": translate_private_test_cases(example["private_test_cases"])
            },
            num_proc=cpu_count,
        )
        shard = shard.map(map_to_example, remove_columns=ds.column_names)
        processed_shards.append(shard)
    ds = concatenate_datasets(processed_shards)
    logger.info(f"Processed dataset size: {len(ds)}")
    return ds


def build_dataset_map(ds):
    """构建 prompt -> dataset_example 的映射"""
    dataset_map = {}
    for ex in ds:
        key = ex["prompt"].strip()
        dataset_map[key] = dict(ex)
    logger.info(f"Dataset map size: {len(dataset_map)}")
    return dataset_map


def load_json_results(json_path: str) -> list:
    """加载JSON评测结果文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data["results"]["LiveCodeBench"]

    if isinstance(data, dict) and "examples" in data:
        return data["examples"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unrecognized JSON format in {json_path}")


def match_json_to_dataset(json_examples, dataset_map):
    """将JSON中的example与数据集匹配"""
    matched = []
    unmatched = 0

    for json_ex in json_examples:
        key = json_ex["prompt"].strip()
        ds_ex = dataset_map.get(key)

        if ds_ex is None:
            for ds_key, ds_val in dataset_map.items():
                if ds_key[:200] == key[:200]:
                    ds_ex = ds_val
                    break

        if ds_ex is not None:
            matched.append((json_ex, ds_ex))
        else:
            unmatched += 1
            logger.warning(f"Unmatched prompt (first 80 chars): {key[:80]}...")

    logger.info(f"Matched: {len(matched)}, Unmatched: {unmatched}")
    return matched


def evaluate_single_example_fast(benchmark, task):
    """
    评测单个 example，与原始 evaluate_single_example 逻辑完全一致：
    只取 code_filter_result[-1]（最后一个代码块）
    新增：修复下划线缩进问题
    """
    try:
        code_filter_result = task["model_answer"]

        if not code_filter_result or len(code_filter_result) == 0:
            return {
                "prompt": task["prompt"],
                "difficulty": task["difficulty"],
                "correctness": False,
                "reason": "Does not contain code component.",
            }

        # ====== 只取最后一个代码块 ======
        last_code = code_filter_result[-1]

        # ====== 修复下划线缩进：\n后连续4的倍数个下划线 → 空格 ======
        last_code = fix_underscore_indentation(last_code)

        try:
            problem_to_check = copy.deepcopy(task)

            curr_res = benchmark.check_correctness(
                problem=problem_to_check,
                completion=post_process_code(last_code),
                timeout=6,
                is_extracted=not task["is_stdin"],
            )

            return {
                "prompt": task["prompt"],
                "difficulty": task["difficulty"],
                "correctness": curr_res,
                "reason": "" if curr_res else "Code is incorrect.",
            }

        except Exception as e:
            return {
                "prompt": task["prompt"],
                "difficulty": task["difficulty"],
                "correctness": False,
                "reason": f"Evaluation error: {str(e)}",
            }

    except Exception as outer_e:
        return {
            "prompt": task.get("prompt", ""),
            "difficulty": task.get("difficulty", "unknown"),
            "correctness": False,
            "reason": f"Critical error: {str(outer_e)}",
        }


def evaluate_from_json(
    json_path: str,
    max_workers: int = 32,
    output_path: str = "eval_results.json",
):
    """
    主评测函数（优化版）：
    - 每个 example 只评测最后一个代码块（与原始 evaluate_responses 一致）
    - 修复下划线缩进问题
    - 减少不必要的 deepcopy
    """
    # ========== 1. 加载JSON ==========
    json_examples = load_json_results(json_path)
    logger.info(f"Loaded {len(json_examples)} examples from JSON")

    first = json_examples[0]
    logger.info(f"JSON example keys: {list(first.keys())}")
    if isinstance(first.get("content"), list):
        logger.info(f"Number of code blocks in first example's content: {len(first['content'])}")

    # ========== 2. 加载数据集 ==========
    ds = load_and_process_dataset()
    dataset_map = build_dataset_map(ds)

    # ========== 3. 匹配 ==========
    matched = match_json_to_dataset(json_examples, dataset_map)

    # ========== 4. 构建评测任务（每个 example 只 1 个任务）==========
    benchmark = LiveCodeBenchBenchmark(debug=False, logger=logger)

    eval_tasks = []
    for match_idx, (json_ex, ds_ex) in enumerate(matched):
        contents = json_ex["content"]

        task = dict(ds_ex)  # 浅拷贝
        task["difficulty"] = json_ex["difficulty"]
        task["model_output"] = json_ex.get("output", "")

        if isinstance(contents, list):
            task["model_answer"] = contents
        elif isinstance(contents, str):
            task["model_answer"] = [contents]
        else:
            task["model_answer"] = []

        eval_tasks.append(task)

    logger.info(f"Total evaluation tasks: {len(eval_tasks)} (1 per example, last code block only)")

    # ========== 5. 并发评测 ==========
    logger.warning("Expect some stdout leaks from code execution...")

    ordered_results = [None] * len(eval_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for i, task in enumerate(eval_tasks):
            future = executor.submit(evaluate_single_example_fast, benchmark, task)
            future_to_idx[future] = i

        done_count = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                ordered_results[idx] = result
            except Exception as e:
                logger.error(f"Error evaluating task {idx}: {e}")
                ordered_results[idx] = {
                    "correctness": False,
                    "reason": f"Error: {e}",
                    "difficulty": eval_tasks[idx].get("difficulty", "unknown"),
                }
            done_count += 1
            if done_count % 50 == 0:
                logger.info(f"Progress: {done_count}/{len(eval_tasks)}")

    # ========== 6. 统计结果 ==========
    total_correct = sum(1 for r in ordered_results if r and r.get("correctness"))
    total = len(ordered_results)

    per_difficulty_correct = defaultdict(int)
    per_difficulty_total = defaultdict(int)

    for r in ordered_results:
        if r is None:
            continue
        diff = r["difficulty"]
        per_difficulty_total[diff] += 1
        if r.get("correctness"):
            per_difficulty_correct[diff] += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Overall Accuracy: {total_correct}/{total} = {total_correct/total:.2%}")
    for diff in sorted(per_difficulty_total.keys()):
        c = per_difficulty_correct[diff]
        t = per_difficulty_total[diff]
        logger.info(f"  {diff}: {c}/{t} = {c/t:.2%}")
    logger.info(f"{'='*60}")

    # ========== 7. 保存结果 ==========
    save_results = {
        "num_total": total,
        "num_correct": total_correct,
        "accuracy": total_correct / total if total > 0 else 0,
        "per_difficulty": {
            diff: {
                "correct": per_difficulty_correct[diff],
                "total": per_difficulty_total[diff],
                "accuracy": per_difficulty_correct[diff] / per_difficulty_total[diff],
            }
            for diff in sorted(per_difficulty_total.keys())
        },
        "details": [
            {
                "idx": i,
                "difficulty": r["difficulty"],
                "correctness": r["correctness"],
                "reason": r.get("reason", ""),
            }
            for i, r in enumerate(ordered_results)
            if r is not None
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    return save_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LiveCodeBench from JSON results")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to JSON results file")
    parser.add_argument("--max_workers", type=int, default=32,
                        help="Number of parallel threads")
    parser.add_argument("--output", type=str, default="eval_results.json",
                        help="Output results file")
    args = parser.parse_args()

    evaluate_from_json(
        json_path=args.json_path,
        max_workers=args.max_workers,
        output_path=args.output,
    )