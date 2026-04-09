import copy
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM

from eval.task import BaseBenchmark

from .livecodebench_utils import lcb_run, map_to_example, post_process_code, translate_private_test_cases

# HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
# if not HF_HUB_CACHE:
#     print(
#         "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for LiveCodeBenchv5 benchmark"
#     )


def has_code(response):
    # pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    pattern = r"```python\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    if matches: 
        return matches
    else:
        return response 


# Calculate mean and standard error for all metrics
def calc_stats(values):
    mean = np.mean(values)
    stderr = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, stderr


class LiveCodeBenchV5Benchmark(BaseBenchmark):
    """
    LiveCodeBench v5 - v2 Benchmark for evaluating the math reasoning of LLMs.

    Follows the evaluation logic of hendrycks_math answer extraction.
    """

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize LiveCodeBenchV5 benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.debug = debug
        self.max_new_tokens = 32768  # set higher to avoid truncation for reasoning models
        self.seed = seed
        # self.n_repeat = 3
        self.n_repeat = 1

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.load_questions()
        if self.debug:
            examples = examples[:2]
            examples = Dataset.from_dict(examples)

        all_outputs = []

        # for i in range(self.n_repeat):
        #     all_instances = []
        #     seed = [s + i for s in self.seed]

        #     for idx, example in enumerate(examples):
        #         if example["is_stdin"]:
        #             prompt_text = (
        #                 "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
        #                 + example["prompt"]
        #             )
        #         else:
        #             prompt_text = (
        #                 "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
        #                 + example["prompt"]
        #             )
        #         messages = [{"role": "user", "content": prompt_text}]

        for i in range(self.n_repeat): #设置成 1 不动，改重复次数的话在下面vllm参数里面改
            all_instances = []
            seed = [s + i for s in self.seed]

            for idx, example in enumerate(examples):
                if example["is_stdin"]:
                    prompt_text = (
                        "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
                        # + "Generate an executable Python function generated from the given prompt. The function should take stdin as input and print the output. Simply call the function after the definition."
                        + example["prompt"]
                        + "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows: ```python\n# YOUR CODE HERE\n```. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
                    )
                    # print(f"stdin_case:{prompt_text}\n")
                    # print(f"caonima{idx}\n")
                else:
                    prompt_text = (
                        "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
                        # + "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
                        + example["prompt"]
                        + "Do not directly test on the sample inputs. Enclose your code within delimiters as follows: ```python\n# YOUR CODE HERE\n```. Return the function body without invoking it at the final solution."
                        # + "Attention: You must mind that you should write your own code instead of just ```# YOUR CODE HERE```"
                    )

                    # prompt_text = (
                    #     "Question:"
                    #     +"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
                    #     # + "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
                    #     + example["prompt"]
                    #     + "Do not directly test on the sample inputs. Enclose your code within delimiters as follows: ```python\n# YOUR CODE HERE\n```. Return the function body without invoking it at the final solution."
                    #     # + "Attention: You must mind that you should write your own code instead of just ```# YOUR CODE HERE```"
                    #     +"Answer:"
                    # )

                    # print(f"no_case:{prompt_text}\n")
                messages = [
                    # {"role": "sys", "content": "A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <|think|> <|/think|>, followed directly by the final answer, like this: <|think|> reasoning process here <|/think|> final answer here."},
                    {"role": "user", "content": prompt_text}
                    ]
                templated_messages = self._prepare_messages(messages, model)
                # ## new chat_template
                # prompt_text = messages[0]['content']
                # prompt_token_ids = model.tokenizer.encode(prompt_text, add_special_tokens=False)                                        
                # # 构造成 vllm 能识别的格式                                                                               
                # from vllm.inputs import TokensPrompt                                                                                   
                # templated_messages = TokensPrompt(prompt_token_ids=prompt_token_ids)

                # print(f"template_messages:{templated_messages}\n")

                # templated_messages = templated_messages['prompt_token_ids']
                # ##

                # print(templated_messages)
                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": True,
                            "max_new_tokens": self.max_new_tokens,
                            # "temperature": 0.7,
                            "temperature": 0.6,
                            "top_p":0.95,
                            "seed": seed,
                            "n": 3,
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = i
                all_instances.append(instance)

            # # Generate model responses
            # self.logger.info("Generating responses for LiveCodeBenchV5...")
            # outputs = self.compute(model, all_instances)
            # all_outputs.append(outputs)
            # print("fuck you",len(all_instances))
            # Generate model responses
            self.logger.info("Generating responses for LiveCodeBenchV5...")
            outputs = self.compute(model, all_instances)
            # print(f"output_from_compute:{outputs}\n")
            # all_outputs.append(outputs)
            # print(outputs)
            for j in range(len(all_instances)):
                all_outputs.append(outputs[j][0])
                # print(outputs[0][0])
                all_outputs.append(outputs[j][1])
                all_outputs.append(outputs[j][2])
        def reorder_list(lst, n):
            """
            将list按列重排：
            取出位置为0, n, 2n,...的元素放前面；
            再取1, n+1, 2n+1,...，以此类推。
            """
            result = []
            for i in range(n):
                result.extend(lst[i::n])  # lst[i::n] 表示从i开始每隔n取一个
            return result
        temp = reorder_list(all_outputs,3)
        all_outputs = []
        n = len(all_instances)
        all_outputs.append(temp[:n])
        all_outputs.append(temp[n:n*2])
        all_outputs.append(temp[n*2:])
        # print(all_outputs)
        print("\n",len(all_outputs),len(all_outputs[0]))
        # print(all_outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        examples_list = []

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [has_code(o) for o in outputs]
            examples_list.append(example)

        return {"examples": examples_list}

    @staticmethod
    def check_correctness(problem: Dict, completion: str, timeout: float, is_extracted: bool = False) -> Dict:
        """
        Evaluates the functional correctness of a completion by running the test
        suite provided in the problem.

        :param completion_id: an optional completion ID so we can match
            the results later even if execution finishes asynchronously.
        """
        result_list = lcb_run(problem, completion, timeout, is_extracted)
        details = [r[0] for r in result_list]
        all_passed = all(details)

        result = ""
        if result_list and all_passed:
            result = "passed"

        return result == "passed"

    def evaluate_single_example(self, example):
        """Helper function to evaluate a single example"""
        try:
            response_entry = {
                "prompt": example["prompt"],
                "content": example["model_answer"],
                "difficulty": example["difficulty"],
                "correctness": None,
                "reason": None,
                "output": example["model_output"],
            }

            code_filter_result = example["model_answer"]

            if not code_filter_result or len(code_filter_result) == 0:
                response_entry["correctness"] = False
                response_entry["reason"] = "Does not contain code component."
                return response_entry

            try:
                last_code = code_filter_result[-1]
                problem_to_check = copy.deepcopy(example)

                # Add debugging
                self.logger.debug(f"Evaluating {example['difficulty']} problem...")

                # Add timeout handling
                curr_res = self.check_correctness(
                    problem=problem_to_check,
                    completion=post_process_code(last_code),
                    timeout=6,
                    is_extracted=not problem_to_check["is_stdin"],
                )

                # Log the result
                self.logger.debug(f"Result for {example['difficulty']}: {curr_res}")

                response_entry["correctness"] = curr_res
                response_entry["reason"] = "" if curr_res else "Code is incorrect."

            except Exception as e:
                self.logger.error(f"Error evaluating {example['difficulty']} example: {str(e)}")
                response_entry["correctness"] = False
                response_entry["reason"] = f"Evaluation error: {str(e)}"

            return response_entry

        except Exception as outer_e:
            self.logger.error(f"Outer error in evaluate_single_example: {str(outer_e)}")
            return {
                "prompt": example["prompt"],
                "content": example.get("model_answer"),
                "difficulty": example.get("difficulty"),
                "correctness": False,
                "reason": f"Critical error: {str(outer_e)}",
                "output": example["model_output"],
            }

    def evaluate_responses(self, responses: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions in parallel using threads."""
        # Handle None result from non-primary ranks
        if responses is None:
            return None

        self.logger.info(f"Evaluating {len(responses['examples'])} examples...")
        self.logger.warning(f"Expect some output leaks from the code / test execution into stdout")

        # First, organize completions by repeat index
        examples_by_repeat = defaultdict(list)
        for example in responses["examples"]:
            for i, (output, answers) in enumerate(zip(example["model_outputs"], example["model_answers"])):
                # Create a copy of the original example and update with the specific completion
                example_copy = example.copy()  # Make a shallow copy of the example
                example_copy["model_answer"] = answers
                example_copy["model_output"] = output
                # Remove the lists of all outputs/answers to avoid confusion
                example_copy.pop("model_outputs", None)
                example_copy.pop("model_answers", None)
                examples_by_repeat[i].append(example_copy)

        # Evaluate each set of completions separately
        all_metrics = []
        run_stats = []
        num_questions = len(responses["examples"])

        for repeat_idx, examples in examples_by_repeat.items():
            # Use ThreadPoolExecutor with limited concurrency
            results = []
            with ThreadPoolExecutor(max_workers=32) as executor:
                future_to_example = {}
                for i, example in enumerate(examples):
                    future = executor.submit(self.evaluate_single_example, example)
                    future_to_example[future] = (i, example)

                # Collect results as they complete
                results = [None] * len(examples)
                for future in as_completed(future_to_example):
                    idx, example = future_to_example[future]
                    try:
                        result = future.result()
                        results[idx] = (result, example)
                    except Exception as e:
                        self.logger.error(f"Future error for example {idx}: {str(e)}")
                        results[idx] = (
                            {
                                "prompt": example["prompt"],
                                "content": example["model_answer"],
                                "difficulty": example["difficulty"],
                                "correctness": False,
                                "reason": f"Future error: {str(e)}",
                                "output": example["model_output"],
                            },
                            example,
                        )

            # Calculate metrics for this repeat
            total_correct = sum(1 for result, _ in results if result["correctness"])
            total_finish = len(results)

            per_difficulty_correct = defaultdict(int)
            per_difficulty_total = defaultdict(int)

            for result, example in results:
                per_difficulty_correct[example["difficulty"]] += result["correctness"]
                per_difficulty_total[example["difficulty"]] += 1

            metrics = {
                "total_correct": total_correct,
                "total_finish": total_finish,
                "accuracy": total_correct / total_finish,
                "per_difficulty_correct": dict(per_difficulty_correct),
                "per_difficulty_total": dict(per_difficulty_total),
            }

            # Add per-difficulty accuracies
            for difficulty in per_difficulty_correct.keys():
                metrics[f"accuracy_{difficulty}"] = (
                    per_difficulty_correct[difficulty] / per_difficulty_total[difficulty]
                )

            all_metrics.append(metrics)

            # Add to run_stats for precomputed_hf_lm.py compatibility
            run_stats.append(
                {
                    "repetition": repeat_idx + 1,
                    "num_total": total_finish,
                    "num_solved": total_correct,
                    "accuracy": total_correct / total_finish,
                }
            )

        final_metrics = {}

        # Calculate stats for overall accuracy
        acc_values = [m["accuracy"] for m in all_metrics]
        mean_acc, stderr_acc = calc_stats(acc_values)
        final_metrics["accuracy_avg"] = mean_acc
        final_metrics["accuracy_std_err"] = stderr_acc
        self.logger.info(f"Overall accuracy: {mean_acc:.2%} ± {stderr_acc:.2%}")

        # Calculate stats for each difficulty level
        difficulties = all_metrics[0]["per_difficulty_correct"].keys()
        for diff in difficulties:
            acc_values = [m[f"accuracy_{diff}"] for m in all_metrics]
            mean_acc, stderr_acc = calc_stats(acc_values)
            final_metrics[f"accuracy_{diff}_avg"] = mean_acc
            final_metrics[f"accuracy_{diff}_std_err"] = stderr_acc

        # Log results
        for diff in difficulties:
            mean = final_metrics[f"accuracy_{diff}_avg"]
            stderr = final_metrics[f"accuracy_{diff}_std_err"]
            self.logger.info(f"Accuracy {diff}: {mean:.2%} ± {stderr:.2%}")

        # Include raw results and examples in final metrics
        final_metrics["raw_metrics"] = all_metrics
        final_metrics["examples"] = [result for result, _ in results]  # Include last run's examples

        # Add compatibility with precomputed_hf_lm.py
        solved_avg = np.mean([result["num_solved"] for result in run_stats])
        final_metrics.update(
            {
                "num_total": num_questions,
                "solved_avg": solved_avg,
                "run_stats": run_stats,
                "num_repeat": self.n_repeat,
            }
        )

        return final_metrics

    # def load_questions(self) -> Dataset:
    #     """Load LiveCodeBenchV5 questions from source."""
    #     self.logger.info("Loading LiveCodeBenchV5 questions from source and converting to dataset...")
    #     cpu_count = os.cpu_count()
    #     from datasets import load_dataset
    #     ds = load_dataset("/mnt/shared-storage-user/renqihan/dataset/livecodebench/code_generation_lite", 
    #                        version_tag="release_v5",
    #                        split="test",
    #                        trust_remote_code=True)
    #     # ds = load_dataset("mlfoundations-dev/LCBv5-v2", split="test", trust_remote_code=True, cache_dir=HF_HUB_CACHE)
    #     # Avoids "pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays" when mapping
    #     processed_shards = []
    #     num_shards = 4
    #     for i in range(num_shards):
    #         shard = ds.shard(num_shards=num_shards, index=i)
    #         shard = shard.map(
    #             lambda example: {"private_test_cases": translate_private_test_cases(example["private_test_cases"])},
    #             num_proc=cpu_count,
    #             writer_batch_size=10,
    #         )
    #         shard = shard.map(map_to_example, remove_columns=ds.column_names)
    #         processed_shards.append(shard)
    #     ds = concatenate_datasets(processed_shards)
    #     return ds

    def load_questions(self) -> Dataset:
        """Load LiveCodeBenchV5 questions from source."""
        self.logger.info("Loading LiveCodeBenchV5 questions from source and converting to dataset...")
        
        from datasets import load_dataset, Dataset as HFDataset
        ds = load_dataset("/mnt/shared-storage-user/renqihan/dataset/livecodebench/code_generation_lite", 
                        version_tag="v5",
                        split="test",
                        trust_remote_code=True)

        # 完全绕过 Arrow map，用纯 Python 循环处理
        records = []
        for example in ds:
            example["private_test_cases"] = translate_private_test_cases(example["private_test_cases"])
            record = map_to_example(example)
            records.append(record)
        
        ds = HFDataset.from_list(records)
        return ds
