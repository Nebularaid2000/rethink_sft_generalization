import dataclasses
import re
from typing import Dict, Optional, Union

from lm_eval.tasks.ifeval import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]
    # Reasoning models modification : Remove the thinking part
    # response = re.sub(r".*?<\/think>(\\n)*", "", response, flags=re.DOTALL).strip()
    # if isinstance(response, list):
    #     response = response[0] if response else ""
    # elif not isinstance(response, str):
    #     response = str(response) if response is not None else ""
    # def extract_boxed_content(text: str) -> str:
    #     """
    #     Extract content from \boxed{} in the text.
    #     Handles nested braces correctly.
    #     """
    #     # 查找 \boxed{ 的位置
    #     boxed_pattern = r'\\boxed\{'
    #     match = re.search(boxed_pattern, text)
        
    #     if not match:
    #         return text  # 如果没有找到 \boxed{，返回原文本
        
    #     start_idx = match.end()  # \boxed{ 后面的位置
    #     brace_count = 1
    #     end_idx = start_idx
        
    #     # 遍历字符串，匹配括号
    #     while end_idx < len(text) and brace_count > 0:
    #         if text[end_idx] == '{':
    #             brace_count += 1
    #         elif text[end_idx] == '}':
    #             brace_count -= 1
    #         end_idx += 1
        
    #     if brace_count == 0:
    #         # 成功找到匹配的右括号
    #         return text[start_idx:end_idx-1]
    #     else:
    #         # 括号不匹配，返回原文本
    #         return text
    # response = extract_boxed_content(response)
    response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL).strip()
    # response = re.sub(r"^.*?</think>", "", response, flags=re.DOTALL).lstrip()
    response = re.sub(r"^.*</think>", "", response, flags=re.DOTALL).lstrip() # 这里的 .* 是贪婪匹配，会匹配到最后一个 </think>. 如果要用非贪婪匹配，可以用 .*?
    response = re.sub(r"<think>.*?<\|/think\|>\s*", "", response, flags=re.DOTALL).strip()
    response = re.sub(r"^.*<\|/think\|>", "", response, flags=re.DOTALL).lstrip()

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc

def build_prompt(doc):
    """
    构建自定义 prompt 给 LLM 使用
    doc: dict, 单条 IFEval 样本
    """


    # user prompt：来自 IFEval 数据集的 prompt 字段
    user_prompt = doc["prompt"]
    # user_prompt += "You should first think through the reasoning process internally and then give the final answer. The reasoning process should be enclosed within <think></think>, followed directly by the final answer, like this: <think> reasoning process here </think> final answer here. DO NOT output anything else other than the thinking process and the final answer -- no explanations, no extra comments."
    # user_prompt += "DO NOT output anything else other than the thinking process and the final answer -- no explanations, no extra comments."
    # user_prompt += "\nPlease put your final answer within \\boxed{{}}."
    return user_prompt

# import dataclasses
# from typing import Dict, Optional, Union

# from lm_eval.tasks.ifeval import instructions_registry


# @dataclasses.dataclass
# class InputExample:
#     key: int
#     instruction_id_list: list[str]
#     prompt: str
#     kwargs: list[Dict[str, Optional[Union[str, int]]]]


# @dataclasses.dataclass
# class OutputExample:
#     instruction_id_list: list[str]
#     prompt: str
#     response: str
#     follow_all_instructions: bool
#     follow_instruction_list: list[bool]


# def test_instruction_following_strict(
#     inp,
#     response,
# ):
#     """Tests response to see if instructions are followed."""
#     instruction_list = inp.instruction_id_list
#     is_following_list = []

#     for index, instruction_id in enumerate(instruction_list):
#         instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
#         instruction = instruction_cls(instruction_id)

#         # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
#         kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
#         instruction.build_description(**kwargs)
#         args = instruction.get_instruction_args()
#         if args and "prompt" in args:
#             instruction.build_description(prompt=inp.prompt)

#         if response.strip() and instruction.check_following(response):
#             is_following_list.append(True)
#         else:
#             is_following_list.append(False)

#     return OutputExample(
#         instruction_id_list=inp.instruction_id_list,
#         prompt=inp.prompt,
#         response=response,
#         follow_all_instructions=all(is_following_list),
#         follow_instruction_list=is_following_list,
#     )


# def test_instruction_following_loose(
#     inp,
#     response,
# ):
#     """Tests response for an upper bound for following instructions."""
#     r = response.split("\n")
#     response_remove_first = "\n".join(r[1:]).strip()
#     response_remove_last = "\n".join(r[:-1]).strip()
#     response_remove_both = "\n".join(r[1:-1]).strip()
#     revised_response = response.replace("*", "")
#     revised_response_remove_first = response_remove_first.replace("*", "")
#     revised_response_remove_last = response_remove_last.replace("*", "")
#     revised_response_remove_both = response_remove_both.replace("*", "")
#     all_responses = [
#         response,
#         revised_response,
#         response_remove_first,
#         response_remove_last,
#         response_remove_both,
#         revised_response_remove_first,
#         revised_response_remove_last,
#         revised_response_remove_both,
#     ]
#     instruction_list = inp.instruction_id_list
#     is_following_list = []

#     for index, instruction_id in enumerate(instruction_list):
#         instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
#         instruction = instruction_cls(instruction_id)

#         # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
#         kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
#         instruction.build_description(**kwargs)
#         args = instruction.get_instruction_args()
#         if args and "prompt" in args:
#             instruction.build_description(prompt=inp.prompt)

#         is_following = False
#         for r in all_responses:
#             if r.strip() and instruction.check_following(r):
#                 is_following = True
#                 break

#         is_following_list.append(is_following)

#     return OutputExample(
#         instruction_id_list=inp.instruction_id_list,
#         prompt=inp.prompt,
#         response=response,
#         follow_all_instructions=all(is_following_list),
#         follow_instruction_list=is_following_list,
#     )


# def process_results(doc, results):
#     inp = InputExample(
#         key=doc["key"],
#         instruction_id_list=doc["instruction_id_list"],
#         prompt=doc["prompt"],
#         kwargs=doc["kwargs"],
#     )
#     response = results[0]

#     out_strict = test_instruction_following_strict(inp, response)
#     out_loose = test_instruction_following_loose(inp, response)

#     return {
#         "prompt_level_strict_acc": out_strict.follow_all_instructions,
#         "inst_level_strict_acc": out_strict.follow_instruction_list,
#         "prompt_level_loose_acc": out_loose.follow_all_instructions,
#         "inst_level_loose_acc": out_loose.follow_instruction_list,
#     }


# def agg_inst_level_acc(items):
#     flat_items = [item for sublist in items for item in sublist]
#     inst_level_acc = sum(flat_items) / len(flat_items)
#     return inst_level_acc
