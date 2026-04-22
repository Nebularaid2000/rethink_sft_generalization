from utils.parser import extract_answer

DATASET_KEYS = {
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/converted_aime_dataset': {'question': 'problem', 'answer': 'solution'},
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/MATH500': {'question': 'problem', 'answer': 'solution'},
}

RESPONSE_EXTRACTOR = {
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/MATH500': lambda x: extract_answer(x, data_name='math',use_last_number=False),
    '/mnt/shared-storage-user/renqihan/sft_generalization/evaluation/data/math/converted_aime_dataset': lambda x: extract_answer(x, data_name='math',use_last_number=False),
}
