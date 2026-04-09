import re
def extract_boxed_content(text: str) -> str:
        """
        Extract content from \boxed{} in the text.
        Handles nested braces correctly.
        """
        # 查找 \boxed{ 的位置
        boxed_pattern = r'\\boxed\{'
        match = re.search(boxed_pattern, text)
        
        if not match:
            return text  # 如果没有找到 \boxed{，返回原文本
        
        start_idx = match.end()  # \boxed{ 后面的位置
        brace_count = 1
        end_idx = start_idx
        
        # 遍历字符串，匹配括号
        while end_idx < len(text) and brace_count > 0:
            if text[end_idx] == '{':
                brace_count += 1
            elif text[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
        
        if brace_count == 0:
            # 成功找到匹配的右括号
            return text[start_idx:end_idx-1]
        else:
            # 括号不匹配，返回原文本
            return text

# 测试样例
test_cases = [
    # 简单情况
    r"\boxed{42}",
    
    # 包含数学表达式
    r"\boxed{x + y = 10}",
    
    # 嵌套括号
    r"\boxed{a{b}c}",
    r"\boxed{\frac{1}{2}}",
    
    # 多层嵌套
    r"\boxed{f(x) = \sqrt{x^{2}}}",
    
    # 前后有其他内容
    r"答案是 \boxed{100} 分",
    r"Therefore, the result is \boxed{2^{10} = 1024}.",
    
    # 没有 \boxed
    r"Just plain text",
    
    # 括号不匹配（边界情况）
    r"\boxed{incomplete",
    
    # 空内容
    r"\boxed{}",
    
    # 复杂嵌套
    r"\boxed{{a}{b}{c}}",
    r"\boxed{\text{answer is } \{5, 6\}}"
]

# 你可以这样测试
for i, test in enumerate(test_cases, 1):
    result = extract_boxed_content(test)
    print(f"{i}. 输入: {test}")
    print(f"   输出: {result}")
    print()
