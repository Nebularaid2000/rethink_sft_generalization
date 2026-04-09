import os
import tiktoken

# 模拟你的环境变量
os.environ["TIKTOKEN_CACHE_DIR"] = "/mnt/shared-storage-user/ai4good1-share/hf_hub/OpenAI/tiktoken"

try:
    # 尝试加载，看看是否报错
    enc = tiktoken.get_encoding("cl100k_base")
    print("✅ Tiktoken 加载成功！路径配置正确。")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print(f"请确保文件 /mnt/shared-storage-user/ai4good1-share/hf_hub/OpenAI/tiktoken/9b5ad71b2ce5302211f9c61530b329a4922fc6a4 存在。")