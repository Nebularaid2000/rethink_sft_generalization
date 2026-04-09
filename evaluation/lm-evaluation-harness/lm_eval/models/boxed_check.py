import re

def extract_boxed_content(text: str) -> str:
    """
    Extract content from \boxed{} in the text.
    Handles nested braces correctly.
    """
    boxed_pattern = r'\\boxed\{'
    match = re.search(boxed_pattern, text)
    
    if not match:
        return text
    
    start_idx = match.end()
    brace_count = 1
    end_idx = start_idx
    
    while end_idx < len(text) and brace_count > 0:
        if text[end_idx] == '{':
            brace_count += 1
        elif text[end_idx] == '}':
            brace_count -= 1
        end_idx += 1
    
    if brace_count == 0:
        return text[start_idx:end_idx-1]
    else:
        return text

# 测试你的例子
test_text = "B. Mutation and recombination during T-cell maturation in the thymus result in cells that recognize foreign antigens, contributing to tissue graft rejection.\n\nDuring T-cell maturation in the thymus, a process called positive and negative selection occurs. Positive selection ensures that T-cells can recognize self-MHC molecules, while negative selection eliminates T-cells that react too strongly to self-antigens. However, some T-cells that recognize foreign antigens are also produced through mutation and recombination. When these T-cells encounter the same antigen presented by the graft, they can mount an immune response leading to graft rejection.\n\nThe final answer is \\boxed{B}"
result = extract_boxed_content(test_text)
print(f"原文: {test_text}")
print(f"提取结果: {result}")
