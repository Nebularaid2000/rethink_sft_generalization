from functools import partial


CATEGORIES = ["Business", "Humanities", "Medical", "Other", "STEM", "Social Sciences"]

choices = ["A", "B", "C", "D"]  # 如果原文件已有就不要重复定义

def format_cot_example(example, including_answer=True):
    """
    构造 CoT 提示：
    - 使用你数据里的字段：question / option_a~d / answer
    - 不依赖 example["options"] 或 example["cot_content"]
    """
    prompt = "Question:\n"
    question = example["question"]
    prompt += question + "\n"

    # 用 option_a / b / c / d 拼出 options 列表
    options = [
        example["option_a"],
        example["option_b"],
        example["option_c"],
        example["option_d"],
    ]

    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)

    if including_answer:
        # 假设 example["answer"] 是 "A"/"B"/"C"/"D"
        gold = str(example["answer"]).strip().upper()

        # 这里给一个非常简单的 CoT 模板，你可以按需要改复杂些
        prompt += f"\nAnswer: Let's think step by step.\nThe correct answer is {gold}.\n\n"
    else:
        # 保留原来的推理指示
        prompt += r"Please reason step by step and return your final answer within \boxed{}. Only include the letter choice (A, B, C, D, E, F, G, H, I or J) as your final answer."

    return prompt



doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)

def process_docs(dataset, category):
    return dataset.filter(lambda x: x["subject_category"] == category)


process_functions = {
    f"process_{category.lower().replace(' ', '_')}": partial(
        process_docs, category=category
    )
    for category in CATEGORIES
}

globals().update(process_functions)
