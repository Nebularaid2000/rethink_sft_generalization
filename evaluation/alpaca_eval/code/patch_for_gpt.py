import alpaca_eval.utils as utils

# 保存原函数
_original_string_to_dict = utils._string_to_dict

# 打补丁
def _patched_string_to_dict(to_convert):
    return {
        s.split("=", 1)[0]: s.split("=", 1)[1]
        for s in to_convert.split(" ")
        if len(s) > 0 and "=" in s
    }

utils._string_to_dict = _patched_string_to_dict

# 然后正常调用 evaluate
from alpaca_eval.main import evaluate
import fire
fire.Fire(evaluate)