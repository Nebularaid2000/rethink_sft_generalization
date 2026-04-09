from utils.parser import extract_answer
# from utils.grader import math_equal

DATASET_KEYS = {
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/converted_aime_dataset': {'question': 'problem', 'answer': 'solution'},
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/MATH500': {'question': 'problem', 'answer': 'solution'},
}

RESPONSE_EXTRACTOR = {
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/MATH500': lambda x: extract_answer(x, data_name='math',use_last_number=False),
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/converted_aime_dataset': lambda x: extract_answer(x, data_name='math',use_last_number=False),
}

# RESPONSE_COMPARATOR = {
#     'openai/gsm8k': lambda x, y: math_equal(x, y, timeout=True),
#     'hendrycks/competition_math': lambda x, y: math_equal(x, y, timeout=True),
#     'di-zhang-fdu/MATH500': lambda x, y: math_equal(x, y, timeout=True),
#     'datasets/compression_dataset': lambda x, y: math_equal(x, y, timeout=True),
#     'datasets/converted_aime_dataset': lambda x, y: math_equal(x, y, timeout=True),
#     "cais/mmlu": lambda x, y:  math_equal(x, y, timeout=True),
#     "amc": lambda x, y: math_equal(x, y, timeout=True),
#     'olympiadbench': lambda x, y: math_equal(x, y, timeout=True),
#     "minerva_math": lambda x, y: math_equal(x, y, timeout=True),
#     "cmu1": lambda x, y: math_equal(x, y, timeout=True),
# }
