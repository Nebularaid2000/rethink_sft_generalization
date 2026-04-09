#!/usr/bin/env python3
"""
LiveCodeBench Standalone Evaluator

This script evaluates pre-generated code solutions from a JSON file.

Usage:
    python evaluate_livecodebench.py --input results.json --output evaluation_results.json
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm import tqdm

# Import your utility functions
from livecodebench_utils import lcb_run, map_to_example, post_process_code, translate_private_test_cases


class LiveCodeBenchEvaluator:
    """
    Standalone evaluator for LiveCodeBench responses.
    """

    def __init__(
        self,
        max_workers: int = 32,
        timeout: float = 6.0,
        logger: logging.Logger = None,
    ):
        """
        Initialize evaluator.

        Args:
            max_workers: Number of parallel workers for evaluation
            timeout: Timeout for code execution in seconds
            logger: Logger instance
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.logger = logger or self._setup_logger()

    def load_reference_test_cases(self) -> List[Dict]:
        """
        Load reference test cases from LiveCodeBench JSONL files.

        Returns:
            List of example data with test cases (in same order as dataset)
        """
        self.logger.info("Loading reference test cases from LiveCodeBench dataset...")

        # Load from JSONL files directly
        data_dir = "/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/livecodebench"

        # Load all test shards
        import json
        all_data = []
        jsonl_files = [
            os.path.join(data_dir, "test.jsonl"),
            os.path.join(data_dir, "test2.jsonl"),
            os.path.join(data_dir, "test3.jsonl"),
            os.path.join(data_dir, "test4.jsonl"),
        ]

        for jsonl_file in jsonl_files:
            if os.path.exists(jsonl_file):
                self.logger.debug(f"Loading {jsonl_file}...")
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            all_data.append(json.loads(line))

        self.logger.info(f"Loaded {len(all_data)} raw examples from JSONL files")

        # Process examples using chunked approach to avoid PyArrow overflow
        self.logger.info("Processing test cases in chunks to avoid memory issues...")
        
        # Method 1: Process data directly without using Dataset.map with multiprocessing
        processed_data = []
        
        # First translate private test cases
        self.logger.info("Translating private test cases...")
        for example in tqdm(all_data, desc="Translating test cases"):
            try:
                translated = translate_private_test_cases(example["private_test_cases"])
                example["private_test_cases"] = translated
            except Exception as e:
                self.logger.warning(f"Failed to translate test case: {e}")
        
        # Then map to example format
        self.logger.info("Mapping to example format...")
        for example in tqdm(all_data, desc="Mapping examples"):
            try:
                mapped = map_to_example(example)
                processed_data.append(mapped)
            except Exception as e:
                self.logger.warning(f"Failed to map example: {e}")
        
        self.logger.info(f"Processed {len(processed_data)} reference test cases")
        return processed_data

    @staticmethod
    def _get_content_list(example: Dict) -> List:
        """
        Get content list from example, supporting both 'content' and 'model_answers' fields.

        Args:
            example: Example dictionary

        Returns:
            List of code strings
        """
        content = example.get("content")
        if content is None:
            # Fallback to model_answers field (used by eval_instruct.py)
            content = example.get("model_answers", [])

        if isinstance(content, str):
            return [content]
        elif isinstance(content, list):
            return content
        else:
            return []

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Setup default logger."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

    @staticmethod
    def clean_code(code: str) -> str:
        """
        Clean code by replacing special characters.
        
        Replace ▁ characters that appear after newlines in multiples of 4
        with spaces (representing Python indentation).
        Only processes ▁ that appear at the beginning of lines.
        
        Args:
            code: Raw code string
            
        Returns:
            Cleaned code string
        """
        import re
        
        def replace_indent(match):
            prefix = match.group(1)  # Newline or start of string
            underscores = match.group(2)
            num_underscores = len(underscores)
            
            # Calculate how many underscores form valid indentation (multiple of 4)
            indent_count = (num_underscores // 4) * 4
            
            if indent_count > 0:
                # Replace valid indentation with spaces
                spaces = ' ' * indent_count
                # Keep any remaining underscores as-is
                remaining_underscores = underscores[indent_count:]
                return prefix + spaces + remaining_underscores
            else:
                # Less than 4 underscores, keep original
                return match.group(0)
        
        # Pattern: (newline OR start of string) followed by one or more ▁
        pattern = r'(\n|^)(▁+)'
        code = re.sub(pattern, replace_indent, code, flags=re.MULTILINE)
        
        return code

    def validate_and_fix_examples(self, examples: List[Dict]) -> List[Dict]:
        """
        Validate and fix example data structure.
        """
        fixed_examples = []
        
        for idx, example in enumerate(examples):
            fixed_ex = copy.deepcopy(example)
            
            # Check content
            if 'content' not in fixed_ex:
                self.logger.error(f"Example {idx}: Missing 'content' field")
                continue
            
            # Ensure content is a list
            if isinstance(fixed_ex['content'], str):
                fixed_ex['content'] = [fixed_ex['content']]
            
            # Check test cases
            test_case_fields = ['public_test_cases', 'private_test_cases', 'generated_test_cases']
            has_tests = any(field in fixed_ex for field in test_case_fields)
            
            if not has_tests:
                self.logger.error(f"Example {idx} (question_id: {fixed_ex.get('question_id')}): No test cases!")
                continue
            
            # Translate test cases
            if 'private_test_cases' in fixed_ex:
                try:
                    fixed_ex = translate_private_test_cases(fixed_ex)
                except Exception as e:
                    self.logger.warning(f"Failed to translate test cases for example {idx}: {e}")
            
            fixed_examples.append(fixed_ex)
        
        self.logger.info(f"Validated {len(fixed_examples)}/{len(examples)} examples")
        return fixed_examples

    @staticmethod
    def check_correctness(
        problem: Dict,
        completion: str,
        timeout: float,
        is_extracted: bool = False
    ) -> bool:
        """
        Evaluates the functional correctness of a completion.

        Args:
            problem: Problem dictionary with test cases
            completion: Generated code solution
            timeout: Execution timeout in seconds
            is_extracted: Whether the code is a function (True) or stdin-based (False)

        Returns:
            True if all tests pass, False otherwise
        """
        try:
            result_list = lcb_run(problem, completion, timeout, is_extracted)
            if not result_list:
                return False
            details = [r[0] for r in result_list]
            all_passed = all(details)
            return all_passed
        except Exception as e:
            logging.debug(f"Correctness check failed with error: {str(e)}")
            return False

    def evaluate_single_code(self, example: Dict, code: str, code_idx: int) -> Dict:
        """
        Evaluate a single code solution.

        Args:
            example: Example dictionary containing problem info
            code: Code string to evaluate
            code_idx: Index of this code in the content list

        Returns:
            Dictionary with evaluation results
        """
        try:
            # Clean the code
            cleaned_code = self.clean_code(code)
            
            response_entry = {
                "code_index": code_idx,
                "prompt": example.get("prompt", ""),
                "content": cleaned_code,
                "difficulty": example.get("difficulty", "unknown"),
                "correctness": None,
                "reason": None,
                "question_id": example.get("question_id", ""),
            }

            # Check if code exists and is not empty
            if not cleaned_code or len(cleaned_code.strip()) == 0:
                response_entry["correctness"] = False
                response_entry["reason"] = "Empty code."
                return response_entry

            # Check if it's just a placeholder
            if cleaned_code.strip() == "# YOUR CODE HERE":
                response_entry["correctness"] = False
                response_entry["reason"] = "Code is just a placeholder."
                return response_entry

            try:
                problem_to_check = copy.deepcopy(example)

                # Run correctness check
                curr_res = self.check_correctness(
                    problem=problem_to_check,
                    completion=post_process_code(cleaned_code),
                    timeout=self.timeout,
                    is_extracted=not problem_to_check.get("is_stdin", False),
                )

                response_entry["correctness"] = curr_res
                response_entry["reason"] = "" if curr_res else "Code is incorrect."

            except Exception as e:
                self.logger.debug(f"Error evaluating code: {str(e)}")
                response_entry["correctness"] = False
                response_entry["reason"] = f"Evaluation error: {str(e)}"

            return response_entry

        except Exception as outer_e:
            self.logger.error(f"Critical error in evaluate_single_code: {str(outer_e)}")
            return {
                "code_index": code_idx,
                "prompt": example.get("prompt", ""),
                "content": code,
                "difficulty": example.get("difficulty", "unknown"),
                "correctness": False,
                "reason": f"Critical error: {str(outer_e)}",
                "question_id": example.get("question_id", ""),
            }

    def evaluate_example(self, example: Dict, example_idx: int) -> List[Dict]:
        """
        Evaluate all code solutions for a single example.

        Args:
            example: Example dictionary
            example_idx: Index of this example

        Returns:
            List of evaluation results for each code variant
        """
        content_list = self._get_content_list(example)

        results = []
        for code_idx, code in enumerate(content_list):
            result = self.evaluate_single_code(example, code, code_idx)
            result["example_index"] = example_idx
            results.append(result)

        return results

    def evaluate_examples_parallel(self, examples: List[Dict]) -> List[List[Dict]]:
        """
        Evaluate all examples in parallel.

        Args:
            examples: List of example dictionaries

        Returns:
            List of lists, where each inner list contains results for one example
        """
        all_results = [None] * len(examples)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for idx, example in enumerate(examples):
                future = executor.submit(self.evaluate_example, example, idx)
                future_to_idx[future] = idx

            # Progress bar
            with tqdm(total=len(examples), desc="Evaluating examples") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results = future.result()
                        all_results[idx] = results
                    except Exception as e:
                        self.logger.error(f"Future error for example {idx}: {str(e)}")
                        # Create error result
                        all_results[idx] = [{
                            "example_index": idx,
                            "code_index": 0,
                            "correctness": False,
                            "reason": f"Future error: {str(e)}",
                            "difficulty": examples[idx].get("difficulty", "unknown"),
                        }]
                    pbar.update(1)

        return all_results

    def calculate_metrics(self, all_results: List[List[Dict]], examples: List[Dict]) -> Dict:
        """
        Calculate metrics from evaluation results.

        Args:
            all_results: List of lists of evaluation results
            examples: Original examples

        Returns:
            Dictionary of metrics
        """
        # For each example, check if ANY of the code variants passed
        example_passed = []
        per_difficulty_correct = defaultdict(int)
        per_difficulty_total = defaultdict(int)

        for example_results, example in zip(all_results, examples):
            # Check if any code variant passed
            any_passed = any(r["correctness"] for r in example_results)
            example_passed.append(any_passed)
            
            difficulty = example.get("difficulty", "unknown")
            per_difficulty_correct[difficulty] += int(any_passed)
            per_difficulty_total[difficulty] += 1

        total_correct = sum(example_passed)
        total_finish = len(examples)

        metrics = {
            "total_correct": total_correct,
            "total_finish": total_finish,
            "accuracy": total_correct / total_finish if total_finish > 0 else 0.0,
            "per_difficulty_correct": dict(per_difficulty_correct),
            "per_difficulty_total": dict(per_difficulty_total),
        }

        # Add per-difficulty accuracies
        for difficulty in per_difficulty_correct.keys():
            if per_difficulty_total[difficulty] > 0:
                metrics[f"accuracy_{difficulty}"] = (
                    per_difficulty_correct[difficulty] / per_difficulty_total[difficulty]
                )
            else:
                metrics[f"accuracy_{difficulty}"] = 0.0

        # Add pass@k metrics (how many code variants passed for each example)
        code_variants_count = [len(results) for results in all_results]
        max_variants = max(code_variants_count) if code_variants_count else 0
        
        for k in range(1, max_variants + 1):
            pass_at_k = sum(
                1 for results in all_results 
                if any(r["correctness"] for r in results[:k])
            )
            metrics[f"pass@{k}"] = pass_at_k / total_finish if total_finish > 0 else 0.0

        return metrics

    @staticmethod
    def calc_stats(values: List[float]) -> tuple:
        """Calculate mean and standard error."""
        if not values:
            return 0.0, 0.0
        mean = np.mean(values)
        if len(values) > 1:
            stderr = np.std(values, ddof=1) / np.sqrt(len(values))
        else:
            stderr = 0.0
        return mean, stderr

    def evaluate_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate responses from a dictionary.

        Args:
            data: Dictionary containing 'examples' key with list of examples

        Returns:
            Dictionary containing evaluation metrics
        """
        examples = data["results"]["LiveCodeBench"]["examples"]
        self.logger.info(f"Evaluating {len(examples)} examples...")

        # Check if examples have test cases; if not, load from reference dataset
        needs_test_cases = any(
            "private_test_cases" not in ex and "public_test_cases" not in ex
            for ex in examples
        )

        if needs_test_cases:
            self.logger.info("Examples missing test cases, loading from LiveCodeBench dataset...")
            ref_list = self.load_reference_test_cases()

            # Create prompt -> test case mapping
            ref_by_prompt = {}
            for ref in ref_list:
                prompt = ref.get("prompt", "")
                if prompt:
                    ref_by_prompt[prompt] = ref

            # Merge test cases into examples by matching prompts
            matched = 0
            for i, ex in enumerate(examples):
                prompt = ex.get("prompt", "")
                if prompt in ref_by_prompt:
                    ref = ref_by_prompt[prompt]
                    # Copy test case fields
                    for key in ["private_test_cases", "public_test_cases", "generated_test_cases",
                               "is_stdin", "starter_code", "function_name", "question_id"]:
                        if key in ref:
                            ex[key] = ref[key]
                    matched += 1
                else:
                    # Try fallback by index (assuming same order)
                    if i < len(ref_list):
                        ref = ref_list[i]
                        ref_prompt = ref.get("prompt", "")
                        # Check if prompts are similar (first 100 chars match)
                        if prompt[:100] == ref_prompt[:100]:
                            for key in ["private_test_cases", "public_test_cases", "generated_test_cases",
                                       "is_stdin", "starter_code", "function_name", "question_id"]:
                                if key in ref:
                                    ex[key] = ref[key]
                            matched += 1
                        else:
                            self.logger.warning(f"Example {i}: prompt mismatch, using index-based fallback")
                    else:
                        self.logger.warning(f"Example {i}: no matching test case found")

            self.logger.info(f"Matched {matched}/{len(examples)} examples with test cases")

        # Count total code variants (support both content and model_answers fields)
        total_variants = sum(len(self._get_content_list(ex)) for ex in examples)
        self.logger.info(f"Total code variants to evaluate: {total_variants}")

        # Evaluate all examples
        all_results = self.evaluate_examples_parallel(examples)

        # Calculate metrics
        metrics = self.calculate_metrics(all_results, examples)

        # Add detailed results
        metrics["detailed_results"] = all_results
        metrics["num_examples"] = len(examples)
        metrics["num_total_variants"] = total_variants

        return metrics

    def print_summary(self, metrics: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total Examples: {metrics['total_finish']}")
        print(f"Correct Examples: {metrics['total_correct']}")
        print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
        print("-" * 70)

        # Print pass@k metrics
        pass_at_keys = [k for k in metrics.keys() if k.startswith("pass@")]
        if pass_at_keys:
            print("Pass@k Metrics:")
            for key in sorted(pass_at_keys, key=lambda x: int(x.split("@")[1])):
                print(f"  {key}: {metrics[key]:.2%}")
            print("-" * 70)

        # Print per-difficulty metrics
        difficulties = sorted(set(
            k.replace("accuracy_", "").replace("_avg", "").replace("_std_err", "")
            for k in metrics.keys() 
            if k.startswith("accuracy_") and k != "accuracy" and "avg" not in k and "std_err" not in k
        ))
        
        if difficulties:
            print("Per-Difficulty Accuracy:")
            for diff in difficulties:
                acc = metrics.get(f"accuracy_{diff}", 0.0)
                total = metrics.get("per_difficulty_total", {}).get(diff, 0)
                correct = metrics.get("per_difficulty_correct", {}).get(diff, 0)
                print(f"  {diff.capitalize()}: {acc:.2%} ({correct}/{total})")
        
        print("=" * 70)


def load_data(input_path: str) -> Dict:
    """
    Load data from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary containing data
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def save_results(results: Dict, output_path: str):
    """
    Save evaluation results to JSON file.

    Args:
        results: Results dictionary
        output_path: Path to output JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function for standalone evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate LiveCodeBench responses from JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_livecodebench.py --input results.json
  python evaluate_livecodebench.py --input results.json --output eval_results.json
  python evaluate_livecodebench.py --input results.json --workers 64 --timeout 10
        """
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input JSON file containing model responses"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output JSON file for results (default: input_path + '_eval.json')"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)"
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=6.0,
        help="Timeout for code execution in seconds (default: 6.0)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Determine output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_eval.json")

    try:
        # Load data
        logger.info(f"Loading data from: {args.input}")
        data = load_data(args.input)
        
        logger.info(f"Loaded {len(data['results']['LiveCodeBench']['examples'])} examples")

        # Initialize evaluator
        evaluator = LiveCodeBenchEvaluator(
            max_workers=args.workers,
            timeout=args.timeout,
            logger=logger,
        )

        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.evaluate_from_dict(data)

        # Print summary
        evaluator.print_summary(results)

        # Save results
        save_results(results, args.output)

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()