"""
The logic in this file largely borrows from Qwen2.5-Math codebase at https://github.com/QwenLM/Qwen2.5-Math:
"""

# import re


# def get_multiple_choice_answer(pred: str):
#     pred=last_boxed_only_string(pred)
#     if not pred:
#         return ""
#     # print(pred,"\n")
#     tmp = re.findall(r"\b(A|B|C|D)\b", pred.upper())
#     if tmp:
#         pred = tmp
#     else:
#         pred = [pred.strip().strip(".")]

#     if len(pred) == 0:
#         pred = ""
#     else:
#         pred = pred[-1]

#     # Remove the period at the end, again!
#     pred = pred.rstrip(".").rstrip("/")

#     return pred



# def last_boxed_only_string(string: str) :
#     """Extract the last LaTeX boxed expression from a string.

#     Args:
#         string: Input string containing LaTeX code

#     Returns:
#         The last boxed expression (including the \\boxed{} wrapper) or None if not found
#     """
#     idx = string.rfind("\\boxed{")
#     if idx < 0:
#         return None

#     i = idx
#     right_brace_idx = None
#     num_left_braces_open = 0

#     while i < len(string):
#         if string[i] == "{":
#             num_left_braces_open += 1
#         if string[i] == "}":
#             num_left_braces_open -= 1
#             if num_left_braces_open == 0:
#                 right_brace_idx = i
#                 break
#         i += 1

#     return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


# print(get_multiple_choice_answer(""))



import re


def get_multiple_choice_answer(pred: str):
    # 先尝试提取 boxed 内容
    boxed_content = last_boxed_only_string(pred)
    
    if boxed_content:
        # 提取 boxed 内的内容
        inner = re.search(r'\\boxed\{(.*)\}', boxed_content)
        pred = inner.group(1) if inner else boxed_content
    # 如果没有 boxed，直接使用原始字符串
    
    if not pred:
        return ""
    
    # 多种模式提取 A/B/C/D
    # 模式1: 明确的答案表述（优先级最高）
    patterns = [
        r"(?:correct answer|right answer|answer)\s*(?:is|:)\s*([A-D])",
        r"(?:choose|select|pick)\s*(?:option)?\s*([A-D])",
        r"(?:option|choice)\s*([A-D])\s*(?:is correct|is right)",
        r"\b([A-D])\s*(?:is correct|is right|is the answer)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # # 模式2: 标准单词边界匹配
    # tmp = re.findall(r"\b([A-D])\b", pred.upper())
    
    # # 模式3: 如果还是失败，查找所有大写字母 A-D
    # if not tmp:
    #     tmp = re.findall(r"([A-D])", pred.upper())
    
    # if tmp:
    #     pred = tmp[-1]
    # else:
    #     pred = pred.strip().strip(".").strip()
    #     if len(pred) > 1:
    #         pred = ""
    # pred = pred.strip().strip(".").strip()
    # if len(pred) == 0:
    #     pred = ""

    # 清理末尾符号
    pred = pred.rstrip(".").rstrip("/").strip()

    # return pred
    # pred = pred.strip().strip(".").strip()
    # 添加验证：必须是 A-D 之一
    if len(pred) == 1 and pred.upper() in ['A', 'B', 'C', 'D']:
        return pred.upper()
    return ""



def last_boxed_only_string(string: str):
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression (including the \\boxed{} wrapper) or None if not found
    """
    if not string:
        return None
        
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
