import json
import os

os.chdir("/fs-computility/liudongrui/renqihan/cairuikun/math_eval/eval_results/STEP_sft_qw3-base-14b_all-correct-8-57k_vanilla_lr5e-5_ep8/new_outputs")
# /fs-computility/liudongrui/renqihan/cairuikun/math_eval/eval_results/new_outputs
# os.chdir("/fs-computility/liudongrui/renqihan/cairuikun/math_eval/eval_results/math500_3repeats/new_outputs")

task = "aime_budget5000" # "aime_budget5000","math500_budget2000","olympiadbench_budget5000"
n_repeat = 10

models = [
    "merged_sft_qw3-base-14b_all-correct-8-57k_vanilla_lr5e-5_ep8_step110",
    "merged_sft_qw3-base-14b_all-correct-8-57k_vanilla_lr5e-5_ep8_step220",
    "merged_sft_qw3-base-14b_all-correct-8-57k_vanilla_lr5e-5_ep8_step330",
    "merged_sft_qw3-base-14b_all-correct-8-57k_vanilla_lr5e-5_ep8_step550",
    "merged_sft_qw3-base-14b_all-correct-8-57k_vanilla_lr5e-5_ep8_step660"
]

# models = [
#     'merged_sft_qw3-base-14b_all-correct-1-57k_cispo-0.8-0.8_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_all-correct-1-57k_dft_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_all-correct-1-57k_vanilla_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_all-correct-1-57k_vanilla_lr5e-5_ep8_step440', 
#     'merged_sft_qw3-base-14b_all-correct-8-57k_cispo-0.8-0.8_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_all-correct-8-57k_dft-clip-0.8-0.8_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_all-correct-8-57k_dft_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_all-correct-8-57k_ppo-0.8-0.8_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_all-correct-8-57k_vanilla_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_all-correct-8-57k_vanilla_lr5e-5_ep1_token-sum-norm_step111',
#     'merged_sft_qw3-base-14b_correct-7-wrong-1-57k_adv-only_lr5e-5_ep1_step111', 
#     'merged_sft_qw3-base-14b_correct-7-wrong-1-57k_cispo_lr5e-5_ep1_step111'
# ]

result_txt = "result.txt"

with open(result_txt, 'a+', encoding='utf-8') as f:
    f.write(f"task:{task}\n模型\t平均token数\nn_repeat={n_repeat}\n")  # 添加标题行

for model in models:
    path = os.path.join(model, task, "no_instruct_32768.json")

    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        token_num_lists = [item.get("tokens") for item in data]
    
        token = 0 
        num = 0

        for token_num_list in token_num_lists:
            small = 0
            sm = 0
            # print("             ",token_num_list)
            for token_num in token_num_list:
                small += token_num
                token += token_num
                num += 1
                sm += 1
            
            # print("             token:",small,"num:",sm)

        ave_num = token/num

        print("token:",token,"num:",num)

        with open(result_txt, 'a+', encoding='utf-8') as result_file:
            # 使用格式化字符串写入
            result_file.write(f"model:{model} \t ave_num:{ave_num:.2f}\n")



