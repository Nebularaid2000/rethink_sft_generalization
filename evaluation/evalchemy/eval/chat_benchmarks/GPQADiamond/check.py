import logging
import random
from typing import Dict, Any, List
from datasets import load_dataset

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class TestGPQA:
    """测试GPQADiamond功能"""
    
    def __init__(self, debug=False, logger=None):
        self.debug = debug
        self.logger = logger or logging.getLogger(__name__)
    
    def load_questions(self) -> List[Dict[str, Any]]:
        """Load GPQADiamond questions from the dataset."""
        local_csv_path = "eval/chat_benchmarks/GPQADiamond/data/gpqa_diamond.csv"  
        dataset = load_dataset("csv", data_files=local_csv_path)
        questions = [row for row in dataset["train"]]
        
        # if self.debug:
        #     questions = questions[:2]
        
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
    #     # 使用固定种子确保可重现性
    #     rnd = random.Random(42)
    #     # rnd = random.Random(242)
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

    
    def test_data_structure(self):
        """测试数据加载功能"""
        print("\n" + "="*60)
        print("测试数据加载功能")
        print("="*60)
        
        try:
            # 加载问题
            questions = self.load_questions()
            print(f"✓ 成功加载 {len(questions)} 个问题")
            
            if questions:
                # 打印第一个问题的结构
                sample = questions[0]
                print(f"\n第一个问题的结构：")
                for key, value in sample.items():
                    value_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"  {key}: {value_preview}")
                
                # 检查必需的字段
                required_fields = ["Question", "Correct Answer", "Incorrect Answer 1", 
                                  "Incorrect Answer 2", "Incorrect Answer 3"]
                missing_fields = [field for field in required_fields if field not in sample]
                
                if missing_fields:
                    print(f"\n✗ 缺少必要的字段: {missing_fields}")
                    return False
                else:
                    print(f"\n✓ 所有必需字段都存在")
                    return True
            else:
                print("✗ 没有加载到任何问题")
                return False
                
        except Exception as e:
            print(f"✗ 数据加载失败: {e}")
            return False
    
    def test_answer_generation(self, num_test_cases=3):
        """测试答案生成功能"""
        print("\n" + "="*60)
        print("测试选项打乱和答案生成功能")
        print("="*60)
        
        questions = self.load_questions()
        
        if len(questions) < num_test_cases:
            num_test_cases = len(questions)
            print(f"警告：只有 {num_test_cases} 个问题可测试")
        
        for i in range(num_test_cases):
            print(f"\n--- 测试用例 {i+1} ---")
            
            test_data = questions[i]
            print(f"问题: {test_data['Question'][:100]}...")
            print(f"正确答案: {test_data['Correct Answer']}")
            print(f"错误答案1: {test_data['Incorrect Answer 1']}")
            print(f"错误答案2: {test_data['Incorrect Answer 2']}")
            print(f"错误答案3: {test_data['Incorrect Answer 3']}")
            
            # 测试多次调用以确保可重复性
            print(f"\n多次调用验证可重复性：")
            
            all_results = []
            for attempt in range(3):
                mc_string, correct_letter = self.generate_multiple_choice_answers(test_data)
                print(f"  尝试 {attempt+1}:")
                print(f"    选项字符串: {mc_string}")
                print(f"    正确答案字母: {correct_letter}")
                all_results.append((mc_string, correct_letter))
            
            # 验证可重复性
            if len(set(all_results)) == 1:
                print("  ✓ 所有尝试结果相同（可重复性验证通过）")
            else:
                print("  ✗ 结果不一致！（可重复性验证失败）")
                return False
            
            # 验证正确字母与正确答案的对应关系
            options_dict = {
                letter: answer for part in mc_string.split(", ") 
                for letter, answer in [part.split(") ", 1)]
            }
            
            if options_dict[correct_letter] == test_data["Correct Answer"]:
                print(f"  ✓ 正确字母映射验证通过")
            else:
                print(f"  ✗ 正确字母映射错误！")
                print(f"    选项 {correct_letter} 对应的是: {options_dict[correct_letter]}")
                print(f"    但正确答案应该是: {test_data['Correct Answer']}")
                return False
            
            # 验证四个选项都不同
            if len(options_dict) == 4:
                print(f"  ✓ 4个选项都生成成功")
            else:
                print(f"  ✗ 选项数量错误: {len(options_dict)}")
                return False
        
        print(f"\n✓ 所有 {num_test_cases} 个测试用例通过")
        return True
    
    def test_option_shuffling_randomness(self):
        """测试选项打乱的随机性（使用不同种子）"""
        print("\n" + "="*60)
        print("测试选项打乱的随机性（不同种子）")
        print("="*60)
        
        # 使用模拟数据进行测试
        test_cases = [
            {
                "Question": "化学问题示例",
                "Correct Answer": "正确答案",
                "Incorrect Answer 1": "错误答案1",
                "Incorrect Answer 2": "错误答案2",
                "Incorrect Answer 3": "错误答案3"
            }
        ]
        
        print("测试不同随机种子对选项顺序的影响：")
        
        # 测试多个不同种子
        test_seeds = [42, 123, 456, 789]
        results_by_seed = {}
        
        for seed in test_seeds:
            # 临时修改方法以测试不同种子
            def custom_generate_multiple_choice_answers(data, seed_value):
                answers = [
                    data["Correct Answer"],
                    data["Incorrect Answer 1"],
                    data["Incorrect Answer 2"],
                    data["Incorrect Answer 3"],
                ]
                rnd = random.Random(seed_value)
                rnd.shuffle(answers)
                
                options = ["A", "B", "C", "D"]
                options_to_answers = {letter: answer for letter, answer in zip(options, answers)}
                multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)
                correct_answer_letter = next(
                    letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"]
                )
                return multiple_choice_string, correct_answer_letter
            
            mc_string, correct_letter = custom_generate_multiple_choice_answers(test_cases[0], seed)
            results_by_seed[seed] = (mc_string, correct_letter)
            print(f"种子 {seed}: {mc_string} → 正确答案: {correct_letter}")
        
        # 检查不同种子是否产生不同结果
        unique_results = set(results_by_seed.values())
        if len(unique_results) > 1:
            print(f"\n✓ 不同种子产生不同结果（随机性验证通过）")
        else:
            print(f"\n⚠ 警告：不同种子产生相同结果（可能应该不同）")
        
        return True
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始测试 GPQA Diamond 功能")
        print("="*60)
        
        # 测试数据加载
        data_ok = self.test_data_structure()
        
        # 测试答案生成
        if data_ok:
            answer_gen_ok = self.test_answer_generation()
        else:
            print("\n数据加载失败，跳过后续测试")
            answer_gen_ok = False
        
        # 测试随机性
        randomness_ok = self.test_option_shuffling_randomness()
        
        # 总结
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        print(f"数据加载测试: {'通过' if data_ok else '失败'}")
        print(f"答案生成测试: {'通过' if answer_gen_ok else '失败'}")
        print(f"随机性测试: {'通过' if randomness_ok else '通过（有警告）'}")
        
        all_ok = data_ok and answer_gen_ok
        if all_ok:
            print(f"\n🎉 所有核心测试通过！")
        else:
            print(f"\n😟 部分测试失败，请检查")
        
        return all_ok

# 运行测试
if __name__ == "__main__":
    # 创建一个测试实例
    test_gpqa = TestGPQA(debug=True, logger=logger)
    
    # 运行所有测试
    success = test_gpqa.run_all_tests()
    
    # 额外详细测试 - 查看不同问题的选项分布
    print("\n" + "="*60)
    print("查看多个问题的选项分布")
    print("="*60)
    
    questions = test_gpqa.load_questions()
    if questions:
        # print(f"分析前 {min(5, len(questions))} 个问题的正确答案位置分布：")
        print(f"分析前 {len(questions)} 个问题的正确答案位置分布：")
        
        positions = []
        for i in range(len(questions)):
            _, correct_letter = test_gpqa.generate_multiple_choice_answers(questions[i])
            positions.append(correct_letter)
        
        from collections import Counter
        pos_counts = Counter(positions)
        print(f"正确答案位置统计：")
        for letter in ["A", "B", "C", "D"]:
            count = pos_counts.get(letter, 0)
            print(f"  {letter}: {count}/{len(positions)} ({count/len(positions)*100:.1f}%)")
        
        # 理想情况下，正确答案应该均匀分布在各个位置
        print(f"\n理想情况下每个位置应有约 25% 的分布")


# import random
# # 快速测试问题
# def quick_test():
#     print("快速测试随机性问题")
    
#     # 模拟数据
#     questions = [
#         {
#             "Question": f"问题{i}",
#             "Correct Answer": f"正确{i}",
#             "Incorrect Answer 1": f"错误{i}_1",
#             "Incorrect Answer 2": f"错误{i}_2",
#             "Incorrect Answer 3": f"错误{i}_3"
#         }
#         for i in range(5)
#     ]
    
#     # 使用固定种子42
#     positions = []
#     for q in questions:
#         answers = [q["Correct Answer"], q["Incorrect Answer 1"], 
#                   q["Incorrect Answer 2"], q["Incorrect Answer 3"]]
        
#         rnd = random.Random(1242)
#         rnd.shuffle(answers)
        
#         # 找到正确答案的新位置
#         for i, letter in enumerate(["A", "B", "C", "D"]):
#             if answers[i] == q["Correct Answer"]:
#                 positions.append(letter)
#                 print(f"'{q['Question']}' -> 位置: {letter}")
#                 break
    
#     # 统计分布
#     print(f"\n分布统计: {positions}")
#     from collections import Counter
#     print("计数:", Counter(positions))

# quick_test()

# import logging
# import random
# import sys
# from typing import Dict, Any, List
# from datasets import load_dataset
# from collections import Counter

# # 创建日志记录器
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(handler)

# class TestGPQA:
#     """测试GPQADiamond功能"""
    
#     def __init__(self, debug=False, logger=None):
#         self.debug = debug
#         self.logger = logger or logging.getLogger(__name__)
    
#     def load_questions(self) -> List[Dict[str, Any]]:
#         """Load GPQADiamond questions from the dataset."""
#         # 这里我用示例数据代替，你需要替换为实际路径
#         # local_csv_path = "eval/chat_benchmarks/GPQADiamond/data/gpqa_diamond.csv"  
        
#         # 如果没有实际数据，创建一些示例数据用于测试
#         print("注意：使用示例数据代替实际CSV文件")
#         example_data = [
#             {
#                 "Question": "碳元素的化学符号是什么？",
#                 "Correct Answer": "C",
#                 "Incorrect Answer 1": "CO",
#                 "Incorrect Answer 2": "CO2",
#                 "Incorrect Answer 3": "C6H12O6"
#             },
#             {
#                 "Question": "水的化学式是什么？",
#                 "Correct Answer": "H2O",
#                 "Incorrect Answer 1": "CO2",
#                 "Incorrect Answer 2": "NaCl",
#                 "Incorrect Answer 3": "O2"
#             },
#             {
#                 "Question": "盐酸的化学式是什么？",
#                 "Correct Answer": "HCl",
#                 "Incorrect Answer 1": "H2SO4",
#                 "Incorrect Answer 2": "NaOH",
#                 "Incorrect Answer 3": "NaCl"
#             },
#             {
#                 "Question": "硫酸的化学式是什么？",
#                 "Correct Answer": "H2SO4",
#                 "Incorrect Answer 1": "HCl",
#                 "Incorrect Answer 2": "NaOH",
#                 "Incorrect Answer 3": "NaCl"
#             },
#             {
#                 "Question": "钠的化学符号是什么？",
#                 "Correct Answer": "Na",
#                 "Incorrect Answer 1": "K",
#                 "Incorrect Answer 2": "Ca",
#                 "Incorrect Answer 3": "Fe"
#             }
#         ]
        
#         # 如果你想测试真实数据，取消注释下面的代码并修改路径
#         """
#         try:
#             local_csv_path = "eval/chat_benchmarks/GPQADiamond/data/gpqa_diamond.csv"  
#             dataset = load_dataset("csv", data_files=local_csv_path)
#             questions = [row for row in dataset["train"]]
#         except Exception as e:
#             print(f"加载CSV失败: {e}，改用示例数据")
#             questions = example_data
#         """
        
#         questions = example_data
        
#         if self.debug:
#             questions = questions[:2]
        
#         self.logger.info(f"Loaded {len(questions)} questions")
#         return questions
    
#     def generate_multiple_choice_answers(self, data: Dict[str, Any]) -> tuple[str, str]:
#         """Generate multiple choice string and correct answer letter."""
#         answers = [
#             data["Correct Answer"],
#             data["Incorrect Answer 1"],
#             data["Incorrect Answer 2"],
#             data["Incorrect Answer 3"],
#         ]
#         # 使用固定种子确保可重现性
#         rnd = random.Random(42)
#         rnd.shuffle(answers)

#         options = ["A", "B", "C", "D"]
#         options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

#         multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)
#         correct_answer_letter = next(
#             letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"]
#         )

#         return multiple_choice_string, correct_answer_letter

#     def debug_generate_answers(self):
#         """调试答案生成函数"""
#         print("\n" + "="*60)
#         print("深入调试答案生成")
#         print("="*60)
        
#         # 加载真实数据
#         questions = self.load_questions()
#         if not questions:
#             print("没有加载到问题数据")
#             return
        
#         # 检查前5个问题的答案内容
#         correct_answers = []
#         incorrect_answers = []
        
#         for i in range(min(5, len(questions))):
#             q = questions[i]
#             print(f"\n问题 {i+1}:")
#             print(f"  Correct Answer: '{q['Correct Answer']}'")
#             print(f"  Incorrect 1:    '{q['Incorrect Answer 1']}'")
#             print(f"  Incorrect 2:    '{q['Incorrect Answer 2']}'")
#             print(f"  Incorrect 3:    '{q['Incorrect Answer 3']}'")
            
#             # 检查是否有答案相同
#             all_answers = [
#                 q["Correct Answer"],
#                 q["Incorrect Answer 1"],
#                 q["Incorrect Answer 2"],
#                 q["Incorrect Answer 3"]
#             ]
            
#             if len(set(all_answers)) != 4:
#                 print(f"  ⚠ 警告: 有重复的答案!")
            
#             # 调试打乱过程
#             self.debug_single_shuffle(q)
    
#     def debug_single_shuffle(self, data):
#         """调试单个问题的打乱过程"""
#         answers = [
#             data["Correct Answer"],
#             data["Incorrect Answer 1"],
#             data["Incorrect Answer 2"],
#             data["Incorrect Answer 3"],
#         ]
        
#         print(f"  原始顺序: {answers}")
        
#         # 使用不同种子测试
#         for seed in [42, 43, 44]:
#             answers_copy = answers.copy()
#             rnd = random.Random(seed)
#             rnd.shuffle(answers_copy)
            
#             options = ["A", "B", "C", "D"]
#             mapping = dict(zip(options, answers_copy))
            
#             # 找到正确答案的位置
#             for letter, answer in mapping.items():
#                 if answer == data["Correct Answer"]:
#                     print(f"  种子{seed:3d}: 正确答案在 {letter} ({answer})")
#                     break
    
#     def test_shuffle_mechanism(self):
#         """调试为什么正确答案总在同一位置"""
#         print("\n" + "="*60)
#         print("调试选项打乱机制")
#         print("="*60)
        
#         # 创建一个简单的测试用例
#         test_data = {
#             "Question": "测试问题",
#             "Correct Answer": "A",
#             "Incorrect Answer 1": "B",
#             "Incorrect Answer 2": "C",
#             "Incorrect Answer 3": "D"
#         }
        
#         # 多次运行测试
#         print("测试1: 多次运行相同的数据")
#         for i in range(3):
#             answers = [
#                 test_data["Correct Answer"],
#                 test_data["Incorrect Answer 1"],
#                 test_data["Incorrect Answer 2"],
#                 test_data["Incorrect Answer 3"],
#             ]
#             print(f"第{i+1}次运行前 answers: {answers}")
            
#             rnd = random.Random(42)
#             rnd.shuffle(answers)
            
#             options = ["A", "B", "C", "D"]
#             options_to_answers = {letter: answer for letter, answer in zip(options, answers)}
            
#             print(f"  打乱后 answers: {answers}")
#             print(f"  选项映射: {options_to_answers}")
            
#             # 找到正确答案对应的字母
#             correct_answer_letter = next(
#                 letter for letter, answer in options_to_answers.items() 
#                 if answer == test_data["Correct Answer"]
#             )
#             print(f"  正确答案字母: {correct_answer_letter}")
#             print()
    
#     def analyze_position_distribution(self):
#         """分析正确答案位置分布"""
#         print("\n" + "="*60)
#         print("分析正确答案位置分布")
#         print("="*60)
        
#         questions = self.load_questions()
#         if not questions:
#             print("没有加载到问题数据")
#             return
        
#         print(f"分析前 {min(5, len(questions))} 个问题的正确答案位置分布：")
        
#         positions = []
#         option_details = []
        
#         for i in range(min(5, len(questions))):
#             q = questions[i]
            
#             # 使用固定的generate_multiple_choice_answers方法
#             mc_string, correct_letter = self.generate_multiple_choice_answers(q)
#             positions.append(correct_letter)
            
#             # 解析选项字符串
#             options_text = {}
#             for part in mc_string.split(", "):
#                 letter, answer = part.split(") ", 1)
#                 options_text[letter] = answer
            
#             option_details.append({
#                 "question": q["Question"][:30] + "..." if len(q["Question"]) > 30 else q["Question"],
#                 "options": options_text,
#                 "correct_letter": correct_letter
#             })
            
#             print(f"\n问题 {i+1}: {q['Question'][:40]}...")
#             print(f"  生成选项: {mc_string}")
#             print(f"  正确答案: {correct_letter}")
        
#         # 统计分布
#         print("\n正确答案位置统计：")
#         pos_counts = Counter(positions)
#         total = len(positions)
        
#         for letter in ["A", "B", "C", "D"]:
#             count = pos_counts.get(letter, 0)
#             percentage = count/total*100 if total > 0 else 0
#             print(f"  {letter}: {count}/{total} ({percentage:.1f}%)")
        
#         print(f"理想情况下每个位置应有约 25% 的分布")
        
#         # 显示详细信息
#         if all(pos == "D" for pos in positions):
#             print("\n⚠ 警告：所有正确答案都在D位置！")
#             print("分析可能的原因：")
#             print("1. 所有问题的正确答案在shuffle后恰好都落入D位置")
#             print("2. 正确答案的标识可能有问题")
            
#             # 验证每个问题的正确答案映射
#             print("\n验证每个问题的正确答案映射：")
#             for i, detail in enumerate(option_details):
#                 print(f"\n问题 {i+1}: {detail['question']}")
#                 for letter, answer in detail['options'].items():
#                     mark = "✓" if letter == detail['correct_letter'] else " "
#                     print(f"  {mark} {letter}) {answer}")
    
#     def run_complete_debug(self):
#         """运行完整的调试流程"""
#         print("开始完整调试流程")
#         print("="*60)
        
#         # 1. 测试基本的shuffle机制
#         self.test_shuffle_mechanism()
        
#         # 2. 调试答案生成函数
#         self.debug_generate_answers()
        
#         # 3. 分析位置分布
#         self.analyze_position_distribution()
        
#         # 4. 额外测试：不同种子的效果
#         print("\n" + "="*60)
#         print("测试不同随机种子的影响")
#         print("="*60)
        
#         questions = self.load_questions()
#         if questions:
#             q = questions[0]  # 使用第一个问题
            
#             print(f"测试问题: {q['Question'][:40]}...")
#             print(f"正确答案: {q['Correct Answer']}")
            
#             # 测试多个随机种子
#             seeds = [42, 100, 500, 1000]
#             positions = {}
            
#             for seed in seeds:
#                 # 手动复制原函数的逻辑，但使用不同种子
#                 answers = [
#                     q["Correct Answer"],
#                     q["Incorrect Answer 1"],
#                     q["Incorrect Answer 2"],
#                     q["Incorrect Answer 3"],
#                 ]
                
#                 rnd = random.Random(seed)
#                 rnd.shuffle(answers)
                
#                 options = ["A", "B", "C", "D"]
#                 mapping = dict(zip(options, answers))
                
#                 # 找到正确答案的位置
#                 for letter, answer in mapping.items():
#                     if answer == q["Correct Answer"]:
#                         positions[seed] = letter
#                         print(f"种子 {seed:4d} -> 正确答案位置: {letter}")
#                         break

# # 主程序入口
# def main():
#     """主函数"""
#     # 创建测试实例
#     test_gpqa = TestGPQA(debug=True, logger=logger)
    
#     # 运行完整调试
#     test_gpqa.run_complete_debug()

# if __name__ == "__main__":
#     main()
