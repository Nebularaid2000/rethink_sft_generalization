"""
使用 vLLM 加载模型并生成回复的示例代码（参数对齐 lm-eval 配置）
"""

from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer
import random
import numpy as np
import torch


def generate_with_vllm():
    """使用 vLLM 生成文本回复"""

    # ===== 0. 设置各类随机种子（对齐 lm-eval 的 random_seed / numpy_seed / torch_seed）=====
    random.seed(0)          # 对应 random_seed
    np.random.seed(1234)    # 对应 numpy_seed
    torch.manual_seed(1234) # 对应 torch_seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    # ===== 1. 初始化模型 =====
    model_path = "/mnt/shared-storage-user/wangpeng/Ministral-3-8B-Base-2512_copy"

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",      # 与 lm-eval model_args 中的 dtype 对齐
        max_model_len=32768,   # 与 lm-eval model_args 中的 max_model_len 对齐
    )

    # ===== 2. 初始化 tokenizer =====
    from transformers import MistralCommonBackend
    tokenizer = MistralCommonBackend.from_pretrained(
        model_path,
        # tokenizer_mode=tokenizer_mode,
        # trust_remote_code=trust_remote_code,
        # revision=tokenizer_revision,
        # add_bos_token=add_bos_token,
    )

    # ===== 3. 设置采样参数（对齐 gen_kwargs） =====
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=32768,
        seed=0,  # vLLM 采样随机种子
        n = 3,
    )

    # ===== 4. 准备输入提示 =====
    prompts = [
        "You are given a strip of paper $s$ that is $n$ cells long. Each cell is either black or white. In an operation you can take any $k$ consecutive cells and make them all white.\n\nFind the minimum number of operations needed to remove all black cells.\n\nInput\n\nThe first line contains a single integer $t$ ($1 \\leq t \\leq 1000$) — the number of test cases.\n\nThe first line of each test case contains two integers $n$ and $k$ ($1 \\leq k \\leq n \\leq 2 \\cdot 10^5$) — the length of the paper and the integer used in the operation.\n\nThe second line of each test case contains a string $s$ of length $n$ consisting of characters $\\texttt{B}$ (representing a black cell) or $\\texttt{W}$ (representing a white cell).\n\nThe sum of $n$ over all test cases does not exceed $2 \\cdot 10^5$.\n\nOutput\n\nFor each test case, output a single integer — the minimum number of operations needed to remove all black cells.Sample Input 1:\n8\n\n6 3\n\nWBWWWB\n\n7 3\n\nWWBWBWW\n\n5 4\n\nBWBWB\n\n5 5\n\nBBBBB\n\n8 2\n\nBWBWBBBB\n\n10 2\n\nWBBWBBWBBW\n\n4 1\n\nBBBB\n\n3 2\n\nWWW\n\n\n\nSample Output 1:\n\n2\n1\n2\n1\n4\n3\n4\n0\n\n\nNote\n\nIn the first test case you can perform the following operations: $$\\color{red}{\\texttt{WBW}}\\texttt{WWB} \\to \\texttt{WWW}\\color{red}{\\texttt{WWB}} \\to \\texttt{WWWWWW}$$\n\nIn the second test case you can perform the following operations: $$\\texttt{WW}\\color{red}{\\texttt{BWB}}\\texttt{WW} \\to \\texttt{WWWWWWW}$$\n\nIn the third test case you can perform the following operations: $$\\texttt{B}\\color{red}{\\texttt{WBWB}} \\to \\color{red}{\\texttt{BWWW}}\\texttt{W} \\to \\texttt{WWWWW}$$"
    ]

    # ===== 5. 构造 tokenized prompts =====
    token_prompts = []
    for idx, example in enumerate(prompts):
        # prompt_text = (
        #     "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests."
        #     # + "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
        #     + example
        #     + "Do not directly test on the sample inputs. Enclose your code within delimiters as follows: ```python\n# YOUR CODE HERE\n```. Return the function body without invoking it at the final solution."
        # )
        prompt_text = (
            "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Do not directly test on the sample inputs. Enclose your code within delimiters as follows: ```python\n# YOUR CODE HERE\n```. Return the function body without invoking it at the final solution."
            # + "Generate an executable Python function generated from the given prompt. Return the function body without invoking it at the final solution."
            + example
            + "Attention: You must mind that you should write your own code instead of just ```# YOUR CODE HERE```"
        )

        messages = [{"role": "user", "content": prompt_text}]

        # 应用 chat template 并 tokenize
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,  # 直接返回 list 而不是 dict
        )

        # 调试信息（可按需注释掉）
        print(f"Prompt {idx} token IDs length: {len(token_ids)}")
        print(token_ids)
        decoded_prompt = tokenizer.decode(token_ids)
        print(f"Decoded prompt:\n{decoded_prompt}\n")

        # 创建 TokensPrompt 对象
        token_prompt = TokensPrompt(prompt_token_ids=token_ids)
        token_prompts.append(token_prompt)

    # ===== 6. 生成回复 =====
    outputs = llm.generate(token_prompts, sampling_params)

    # ===== 7. 打印结果 =====
    for idx, output in enumerate(outputs):
        print(f"\n{'=' * 80}")
        print(f"Prompt {idx} 的所有候选：")
        for k, out_k in enumerate(output.outputs):
            generated_text = out_k.text
            print(f"\n--- 候选 {k} ---")
            print(generated_text)
        print('=' * 80)



if __name__ == "__main__":
    print("=" * 80)
    print("示例: 使用 chat_template + tokenize（参数对齐 lm-eval）")
    print("=" * 80)
    generate_with_vllm()
