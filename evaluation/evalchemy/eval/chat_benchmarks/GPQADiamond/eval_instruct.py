import logging
import os
import random
from typing import Any, Dict, List, Optional

import lm_eval.models
import numpy as np
from datasets import load_dataset
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from datasets import Dataset

from eval.task import BaseBenchmark

from .testing_utils import get_multiple_choice_answer

# PROMPT = """Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.
# Problem: {problem}
# Options: {options}
# Answer:"""

PROMPT = """Problem: {problem}
Options: {options}
Please reason step by step and return your final answer within \\boxed{{}}. Only include the letter choice (A, B, C, or D) as your final answer."""

# PROMPT = """Question: {problem}                                                                                                                                    
# Options: {options}                                                                                                                                                Please reason step by step and return your final answer within \\boxed{{}}. Only include the letter choice (A, B, C, or D) as your final answer.
# Answer:"""     

HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE")
if not HF_HUB_CACHE:
    print(
        "WARNING: HF_HUB_CACHE environment variable is not set, using default cache directory ~/.cache/huggingface/hub for GPQADiamond benchmark"
    )


class GPQADiamondBenchmark(BaseBenchmark):
    """
    GPQADiamond Benchmark for evaluating multiple choice reasoning of LLMs.
    Link: https://huggingface.co/datasets/Idavidrein/gpqa
    """

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize GPQADiamond benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.dataset_name = "Idavidrein/gpqa"
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = 32768
        self.n_repeat = 3

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and examples
        """
        examples = self.load_questions()

        if self.debug:
            examples = examples[:2]
        # print(examples.keys())
            # examples = Dataset.from_dict(examples)
        # examples = examples[:4]

        for example in examples:
            multiple_choice_string, correct_answer = self.generate_multiple_choice_answers(example)
            example["multiple_choice_string"] = multiple_choice_string
            example["answer"] = correct_answer

        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        else:
            model_name = model.model_args["model"]

        all_outputs = []

        for i in range(1):
            all_instances = []
            seed = [s + i for s in self.seed]
            # seed = [1234 for s in self.seed]

            for idx, example in enumerate(examples):
                messages = [
                    # {
                    #     "role": "system", 
                    #     "content": "A conversation between the User and Assistant. The User asks a question, and the Assistant provides a solution. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <|think|> <|/think|>, followed directly by the final answer, like this: <|think|> reasoning process here <|/think|> final answer here."
                    # },
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            problem=example["Question"], options=example["multiple_choice_string"]
                        ),
                    },
                ]

                templated_messages = self._prepare_messages(messages, model)

                # # ## no chat_template
                
                # # te_messages = messages[0]['content']
                # prompt_text = PROMPT.format(                                                                                           
                # problem=example["Question"], options=example["multiple_choice_string"]                                             
                # )                                                                                                 
                # # 手动 tokenize 成 token_ids                                                                                        
                # prompt_token_ids = model.tokenizer.encode(prompt_text, add_special_tokens=False)                                        
                # # 构造成 vllm 能识别的格式                                                                               
                # from vllm.inputs import TokensPrompt                                                                                   
                # templated_messages = TokensPrompt(prompt_token_ids=prompt_token_ids)

                # print(f"template_messages:{templated_messages}\n")

                # templated_messages = templated_messages['prompt_token_ids']
                # ##

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": True,
                            "temperature": 0.6,
                            "max_new_tokens": self.max_new_tokens,
                            "seed": seed,
                            "top_p": 0.95,
                            "n": 3,
                        },
                    ),
                    idx,
                )
                instance.repeat_idx = i
                all_instances.append(instance)
                # print(f"instance:{instance}\n")

            # Generate model responses
            self.logger.info("Generating responses for GPQADiamond...")
            print(f"all_instances length: {len(all_instances)}")
            outputs = self.compute(model, all_instances)

            # all_outputs.append(outputs)

            print("check transformers n_sample: \n")
            print(len(outputs[0]))
            print(len(outputs))
            print(outputs)
            print("check over!")

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

        # print(f"example_length:{len(examples)}")
        # print(f"all_outputs_length:{len(all_outputs[0])}")

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [get_multiple_choice_answer(o) for o in outputs]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = sum([example["answer"] == example["model_answers"][i] for example in examples])

            all_results.append(
                {
                    "repetition": i + 1,
                    "num_total": num_questions,
                    "num_solved": solved,
                    "accuracy": solved / num_questions,
                }
            )

        # Calculate overall statistics
        solved_avg = np.mean([result["num_solved"] for result in all_results])
        accuracy_avg = np.mean([result["accuracy"] for result in all_results])
        accuracy_std = np.std([result["accuracy"] for result in all_results])
        accuracy_std_err = np.std([result["accuracy"] for result in all_results]) / np.sqrt(self.n_repeat)

        results.update(
            {
                "num_total": num_questions,
                "solved_avg": solved_avg,
                "run_stats": all_results,
                "accuracy_avg": accuracy_avg,
                "accuracy_std_err": accuracy_std_err,
                "num_repeat": self.n_repeat,
            }
        )

        return results

    # def load_questions(self) -> List[Dict[str, Any]]:
    #     """Load GPQADiamond questions from the dataset."""
    #     dataset = load_dataset(self.dataset_name, "gpqa_diamond", cache_dir=HF_HUB_CACHE)
    #     questions = [row for row in dataset["train"]]
    #     if self.debug:
    #         questions = questions[:2]
    #     self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
    #     return questions

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load GPQADiamond questions from the dataset."""
        local_csv_path = "/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/gpqa/gpqa_diamond.csv"  
        dataset = load_dataset("csv", data_files=local_csv_path)
        questions = [row for row in dataset["train"]]
        
        if self.debug:
            questions = questions[:2]
        
        self.logger.info(f"Loaded {len(questions)} questions from local CSV: {local_csv_path}")
        return questions

    # def generate_multiple_choice_answers(self, data: Dict[str, Any]) -> tuple[str, str]:
    #     """Generate multiple choice string and correct answer letter."""
    #     answers = [
    #         data["Correct Answer"],
    #         data["Incorrect Answer 1"],
    #         data["Incorrect Answer 2"],
    #         data["Incorrect Answer 3"],
    #     ]
    #     rnd = random.Random(42)
    #     rnd.shuffle(answers)

    #     options = ["A", "B", "C", "D"]
    #     options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

    #     multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)
    #     correct_answer_letter = next(
    #         letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"]
    #     )

    #     return multiple_choice_string, correct_answer_letter

    def generate_multiple_choice_answers(self, data: Dict[str, Any]) -> tuple[str, str]:
        """修复版本：确保正确答案位置随机分布"""
        answers = [
            {"text": data["Correct Answer"], "is_correct": True},
            {"text": data["Incorrect Answer 1"], "is_correct": False},
            {"text": data["Incorrect Answer 2"], "is_correct": False},
            {"text": data["Incorrect Answer 3"], "is_correct": False},
        ]
        
        # 使用问题ID或内容创建确定性种子
        question_hash = hash(data["Question"]) % 10000
        seed = 42 + question_hash  # 不同的种子
        
        rnd = random.Random(seed)
        rnd.shuffle(answers)  # 现在answers是字典列表
        
        options = ["A", "B", "C", "D"]
        
        # 构建选项字符串并找到正确答案
        correct_letter = None
        option_strings = []
        
        for i, option in enumerate(options):
            answer = answers[i]
            option_strings.append(f"{option}) {answer['text']}")
            if answer['is_correct']:
                correct_letter = option
        
        multiple_choice_string = ", ".join(option_strings)
        
        return multiple_choice_string, correct_letter
