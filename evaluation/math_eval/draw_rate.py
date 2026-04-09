import os
import re
import pandas as pd
import matplotlib.pyplot as plt

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from math_eval_budget import MAX_AIME, MAX_GSM8K, MAX_AMC, MAX_MATH500, MAX_MINERVA, MAX_OLYMPIADBENCH

# todo: 替换为你实际的文件夹路径
# path_prefix = "draw"
path_prefix = "draw_20250430"


label_dict = {
    # "binary_cmu1_bin_adv_0.1_0.9_global_step_24_actor_huggingface": {
    #     "label": "Bin 0.9",
    #     "mark": "*",
    #     "line_style": "-"
    # }, 
    #  "binary_cmu1_bin_adv_0.1_0.9_global_step_72_actor_huggingface": {
    #     "label": "Bin 0.9(3)",
    #     "mark": "o",
    #     "line_style": "-"
    # }, 
    # "binary_cmu1_bin_adv_0.1_global_step_24_actor_huggingface": {
    #     "label": "Bin 0.8",
    #     "mark": "o",
    #     "line_style": "-"
    # },
    # "8b_binary_0.1_4873687_global_step_24_actor_huggingface" : {
    #     "label": "Soft Penalty",
    #     "mark": "1",
    #     "line_style": "--"
    # }, 
    # "DeepSeek-R1-Distill-Llama-8B" : {
    #     "label": "Base Model",
    #     "mark": "o",
    #     "line_style": "dotted"
    # }, 
    # "compression_dataset" : {
    #     "label": "CMU1",
    #     "mark": "*",
    #     "line_style": "--"
    # },
    # "sig_length_group" : {
    #     "label": "Group Reward",
    #     "mark": "1",
    #     "line_style": "--"
    # },

    "Qwen2.5-VL-7B-Instruct" : {
        "label": "Base Model",
        "mark": "o",
        "line_style": "dotted"
    },
    "multinode-mm-eureka-qwen2.5-7b_K12_onlinefilter_episode10" : {
        "label": "mm:k12",
        "mark": "s",
        "line_style": "--"
    },
    "multinode-mm-eureka-qwen2.5-7b_K12_cmu1-3.2k_onlinefilter_episode10" : {
        "label": "mm:k12_cmu1-3.2k",
        "mark": "s",
        "line_style": "--"
    },
    "4889344_20250411_1542_eureka_1node15k_qwen25vl7b_k12_cmu1_nofilter" : {
        "label": "cmu1_nofilter_ep1:k12",
        "mark": "x",
        "line_style": "-"
    },
    "4921336_20250416_1736_eureka_1node15k_qwenvl7b_k12_cmu1_nofilter_3episode_step244" : {
        "label": "cmu1_nofilter_ep2:k12",
        "mark": "x",
        "line_style": "-"
    },
    "4921336_20250416_1736_eureka_1node15k_qwenvl7b_k12_cmu1_nofilter_3episode_step366" : {
        "label": "cmu1_nofilter_ep3:k12",
        "mark": "x",
        "line_style": "-"
    },
    "tp-qwen2.5-7b_K12_onlinefilter_episode4_len600_step366" : {
        "label": "tp600_ep3:k12",
        "mark": "^",
        "line_style": "-"
    },
    "tp-qwen2.5-7b_K12_onlinefilter_episode3_len400_from600" : {
        "label": "tp600->400_ep3:k12",
        "mark": "^",
        "line_style": "-"
    },
    "tp-qwen2.5-7b_K12_onlinefilter_episode3_len400_from600_step244" : {
        "label": "tp600->400_ep2:k12",
        "mark": "^",
        "line_style": "-"
    },
    "tp-qwen2.5-7b_K12_nofilter_episode3_len600" : {
        "label": "tp600_nofilter_ep3:k12",
        "mark": "^",
        "line_style": "-"
    },
}


def process_files(directory_path, limit):
    # 创建空列表存储结果
    all_data = []

    # 读取目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".md"):  # 处理 Markdown 文件
            file_path = os.path.join(directory_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 使用正则表达式提取表格内容
            table_pattern = r'\| (.*?) \| (.*?) \| (.*?) \| (.*?) \|'  # 匹配表格行
            matches = re.findall(table_pattern, content)

            # 如果表格存在
            if matches:
                budget_list = []
                average_pass_rate_list = []
                pass_1_list = []
                
                for match in matches:
                    if match[0] == 'Budget':
                        continue
                    # match 是一个元组，包含 [Budget, pass@1, pass@k(majority), average_pass_rate]
                    budget, pass_1, pass_k, average_pass_rate = match
                    budget_list.append(float(budget)/limit)
                    # print(budget, limit)
                    average_pass_rate_list.append(float(average_pass_rate) * 100)
                    # pass_1_list.append(float(pass_1))
                
                # 存储每个文件的数据
                # breakpoint()
                all_data.append((filename[:filename.find('.md')], budget_list, average_pass_rate_list))

    return all_data


def plot_results(all_data, title):
    # 创建图形
    plt.figure(figsize=(10, 6))
    # 按照filename字母排序
    all_data.sort(key=lambda x: x[0])

    # 遍历所有数据并绘制每条线
    for filename, budget_list, average_pass_rate_list in all_data:
        print(f"filename: {filename}")
        print("budget_list: ", budget_list)
        print("average_pass_rate_list: ", average_pass_rate_list)
        # if "sig_group" in filename:
        #     setting = "GSR (Ours)"
        #     mark = '*'
        #     line_style = "-"
        #     continue
        # elif "cos_group" in filename:
        #     setting = "GCR (Ours)"
        #     mark = '*'
        #     line_style = "-"
        #     continue
        # elif "group" in filename:
        #     setting = "Group Reward (Ours)"
        #     mark = '*'
        #     line_style = "-"
        #     continue

        # elif "7B" in filename:
        #     setting = "SOTA1"
        #     mark = '*'
        #     line_style = "--"
        #     continue

        # elif "all_exact" in filename:
        #     setting = "SOTA2"
        #     mark = 'o'
        #     line_style = "--"
        #     # continue

        # elif "Distill" in filename:
        #     setting = "DS-Dist-8B"
        #     mark = '1'
        #     line_style = "dotted"
        # elif "more" in filename:
        #     if 'bin_' in filename:
        #         if '0.7' in filename:
        #             setting = "Soft Penalty - Bin(0.7) - More (Ours)"
        #             mark = 'o'
        #             line_style = "-"
        #             # continue
        #         else:
        #             setting = "Soft Penalty - Bin(0.8) - More"
        #             mark = 'o'
        #             line_style = "-"
        #             # continue
        #     else:
        #         setting = "Soft Penalty - More"
        #         mark = 'o'
        #         line_style = "-"
        #     # continue
        # elif "less" in filename:
        #     if 'bin_' in filename:
        #         if '0.7' in filename:
        #             setting = "Soft Penalty - Bin(0.7) - Less (Ours)"
        #             mark = 'o'
        #             line_style = "-"
        #             # continue
        #         else:
        #             setting = "Soft Penalty - Bin(0.8) - Less"
        #             mark = 'o'
        #             line_style = "-"
        #             # continue
        #     else:
        #         setting = "Soft Penalty - Less"
        #         mark = 'o'
        #         line_style = "-"
        #         # continue

        # elif "binary" in filename:
        #     if 'bin_' in filename:
        #         if '0.7' in filename:
        #             setting = "Soft Penalty - Bin(0.7) - No Instruct (Ours)"
        #             mark = 'o'
        #             line_style = "-"
        #             # continue
        #         else:
        #             setting = "Soft Penalty - Bin(0.8) - No Instruct"
        #             mark = 'o'
        #             line_style = "-"
        #             # continue
        #     else:
        #         setting = "Soft Penalty - No Instruct"
        #         mark = 'o'
        #         line_style = "-"
   

        # elif "adaptive_length" in filename:
        #     setting = "Length w/ adap std"
        #     mark = 'o'
        #     line_style = "--"
        # elif "length" in filename:
        #     setting = "Length w/o std"
        #     mark = 'o'
        #     line_style = "--"
        # elif "Math-7B" in filename:
        #     setting = "Base Model"
        #     mark = '1'
        #     line_style = "dotted"
        # if "dr_grpo" in filename:
        #     setting = "DR_GRPO"
        #     mark = 'o'
        #     line_style = "--"
        # elif "adaptive_grpo" in filename:
        #     setting = "ADAP_GRPO"
        #     mark = 'o'
        #     line_style = "--"
        # elif "grpo" in filename:
        #     setting = "GRPO"
        #     mark = 'o'
        #     line_style = "--"

        # elif "length_std" in filename:
        #     setting = "Length w/ std"
        #     mark = '*'
        #     line_style = "-"
        # elif "adaptive_length" in filename:
        #     setting = "Length w/ adap std"
        #     mark = '*'
        #     line_style = "-"
        # elif "length" in filename:
        #     setting = "Length w/o std"
        #     mark = '*'
        #     line_style = "-"
        # elif "Math-7B" in filename:
        #     setting = "Base Model"
        #     mark = '1'
        #     line_style = "dotted"
        # else:
        #     if "dyn" in filename:
        #         setting = "Dynamic"
        #         mark = 'o'
        #         line_style = "--"
        #     else:
        #         setting = "GRPO"
        #         mark = 'o'
        #         line_style = "-"

        type_label = filename.split("_")[-1]
        if type_label == "instruct":
            type_label = ""
        
        key_name = filename[:filename.rfind("_")]

        if key_name not in label_dict:
            continue

        setting = label_dict[key_name]

        print(filename, setting)

        plt.plot(budget_list, 
                 average_pass_rate_list, 
                 label=f"{setting['label']}  {type_label}", 
                 marker=setting['mark'], 
                 linestyle=setting['line_style'],
                 linewidth=0.8)
        # plt.plot(budget_list, pass_1_list, label=f'{filename} - Pass@1', marker='o')

    handles, labels = plt.gca().get_legend_handles_labels()

    #specify order of items in legend
    # order = [0,1,3,2]

    plt.xlabel("Budget Ratio")
    plt.ylabel("Accuracy (%)")
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    # plt.ylim(20,65)
    os.makedirs(f"{path_prefix}/results", exist_ok=True)  # 创建目录
    plt.savefig(f"{path_prefix}/results/{title}.png", dpi=200, bbox_inches='tight')  # 保存图像
    # 展示图表
    plt.close()

if __name__ == "__main__":
    # 设定你的文件夹路径

    dataset_to_title = {
        # "gsm8k_no_instruct": "GSM8K",
        # "math500_no_instruct": "Math500",
        # "aime_no_instruct": "AIME",
        # "amc_no_instruct": "AMC",
        # "olympiadbench_no_instruct": "OlympiadBench",
        # "minerva_no_instruct": "Minerva",
        # "gsm8k_more": "GSM8K",
        # "math500_more": "Math500",
        # "aime_more": "AIME",
        # "amc_more": "AMC",
        # "olympiadbench_more": "OlympiadBench",
        # "minerva_more": "Minerva",
        # "gsm8k_less": "GSM8K",
        # "math500_less": "Math500",
        # "aime_less": "AIME",
        # "amc_less": "AMC",
        # "olympiadbench_less" :"OlympiadBench",
        # "minerva_less": "Minerva",
        "gsm8k": "GSM8K",
        "math500": "Math500",
        "aime": "AIME",
        "amc": "AMC",
        "olympiadbench": "OlympiadBench",
        "minerva": "Minerva",
    }

    dataset_to_budget = {
        "gsm8k": MAX_GSM8K,
        "math500": MAX_MATH500,
        "aime": MAX_AIME,
        "amc": MAX_AMC,
        "olympiadbench": MAX_OLYMPIADBENCH,
        "minerva": MAX_MINERVA,

        # "gsm8k_no_instruct": 900,
        # "math500_no_instruct": 6000,
        # "aime_no_instruct": 12000,
        # "amc_no_instruct": 8000,
        # "olympiadbench_no_instruct": 8000,
        # "minerva_no_instruct": 6000,
        #  "gsm8k_more": 900,
        # "math500_more":  6000,
        # "aime_more":  12000,
        # "amc_more": 8000,
        # "olympiadbench_more": 8000,
        # "minerva_more": 6000,
        # "gsm8k_less": 900,
        # "math500_less": 6000,
        # "aime_less": 12000,
        # "amc_less": 8000,
        # "olympiadbench_less" : 8000,
        # "minerva_less": 6000,
    }

    paths = [
        "gsm8k",
        "math500",
        "aime",
        "amc",
        "olympiadbench",
        "minerva",

        # "gsm8k_no_instruct",
        # "math500_no_instruct",
        # "aime_no_instruct",
        # "amc_no_instruct",
        # "olympiadbench_no_instruct",
        # "minerva_no_instruct",
        # "gsm8k_more",
        # "math500_more",
        # "aime_more",
        # "amc_more",
        # "olympiadbench_more",
        # "minerva_more",
        # "gsm8k_less",
        # "math500_less",
        # "aime_less",
        # "amc_less",
        # "olympiadbench_less",
        # "minerva_less",
        # "gsm8k_easy",
        # "math500_easy",
        # "aime_easy",
        # "amc_easy",
        # "olympiadbench_easy",
        # "minerva_easy",
        # "gsm8k_hard",
        # "math500_hard",
        # "aime_hard",
        # "amc_hard",
        # "olympiadbench_hard",
        # "minerva_hard",
        # "gsm8k",
        # "math500",
        # "aime",
        # "amc",
        # "olympiadbench",
        # "minerva",
    ]

    
    # 处理文件并获取结果
    df_lst = {}
    # 使用df_lst存储并计算跨dataset的平均分数

    for directory_path in paths:

        df = process_files(f"{path_prefix}/{directory_path}", dataset_to_budget[directory_path])
        plot_results(df, dataset_to_title[directory_path])

        for filename, budget_list, average_pass_rate_list in df:
            if filename in df_lst:
                df_lst[filename].append(average_pass_rate_list)
            else:
                df_lst[filename] = [average_pass_rate_list]

    import numpy as np
    df = []
    for filename, df_list in df_lst.items():
        average_pass_rate_list = np.mean(df_list, axis=0)
        budget_list = [(i+1) / 10 for i in range(10)]
        df.append((filename, budget_list, average_pass_rate_list[:20][::2]))
    
    

    plot_results(df, "Average_Score")


