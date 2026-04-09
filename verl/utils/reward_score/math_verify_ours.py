# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
    from latex2sympy2_extended.latex2sympy2 import NormalizationConfig
    from math_verify.errors import TimeoutException
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")
from typing import Optional
import re


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression (including the \\boxed{} wrapper) or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def format_reward_func(completion, coef=0.5):
    pattern1 = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
    )
    matches1 = re.search(pattern1, completion, re.DOTALL)
    pattern2 = (
        r"^(?=(?:.*<\|think\|>){1})(?=(?:.*<\|\/think\|>){1})"
        r"(?!.*<\|think\|>.*<\|think\|>)"
        r"(?!.*<\|\/think\|>.*<\|\/think\|>)"
    )
    matches2 = re.search(pattern2, completion, re.DOTALL)
    if matches1 or matches2:
        return coef
    else:
        return 0.0


def compute_score(solution_str: str, ground_truth: str, timeout_score: float = 0.0, parse_gold_fail_score: float = 0.0, use_format_reward: bool = False) -> float:
    format_reward = format_reward_func(solution_str)

    response = last_boxed_only_string(solution_str)
    
    if response is not None:
        response = response
    else:
        match = re.search(r"(<answer>(.*?)</answer>)", solution_str, re.DOTALL)
        if match:
            response = match.group(2).strip()
        else:
            response = solution_str.split("\n")[-1]
    
    # print("extracted response: ", response)

    ground_truth = f"${str(ground_truth)}$" # renqihan 20250803: Important!! If $ $ is not added to the answer, parsing may fail
    # An alternative way is to wrap ground_truth with \boxed{}
    # if "boxed" not in ground_truth:
    #     ground_truth = f"\\boxed{{{ground_truth}}}"

    gold_parsed = parse(ground_truth)
   

    ret_score = 0.0
    if len(gold_parsed) != 0:
        response_parsed = parse(
            response,
            extraction_config=[
                LatexExtractionConfig(
                    try_extract_without_anchor=True,
                    boxed_match_priority=0,  # maybe we don't need this because the response is already extracted from \boxed{}
                    normalization_config=NormalizationConfig(
                        basic_latex=True,
                        units=True,
                        malformed_operators=False,
                        nits=False,
                        boxed="all",
                        equations=False
                    )
                ), 
                ExprExtractionConfig()
            ],
        )

        try:
            ret_score = float(verify(gold_parsed, response_parsed)) # According to the math-verify docs, the order must be gold first, then answer; otherwise some cases can fail. The previous order was incorrect.
        except Exception:
            pass
        except TimeoutException:
            ret_score = timeout_score
    
    else:
        ret_score = parse_gold_fail_score
        print(f"Failed to parse gold solution: {ground_truth}. Returning {parse_gold_fail_score} score.")
    
    if use_format_reward:
        ret_score = ret_score + format_reward
        return {"score": ret_score, "format_reward": format_reward}
    else:
        return ret_score


def compute_score_wrapper(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        timeout_score=extra_info.get("timeout_score", 0.0),
        parse_gold_fail_score=extra_info.get("parse_gold_fail_score", 0.0)
    )


def compute_score_wrapper_with_format(data_source, solution_str, ground_truth, extra_info) -> float:
    return compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        timeout_score=extra_info.get("timeout_score", 0.0),
        parse_gold_fail_score=extra_info.get("parse_gold_fail_score", 0.0),
        use_format_reward=True
    )
