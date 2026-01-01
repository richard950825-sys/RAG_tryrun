# debug.py
import sys
print(f"当前 Python 路径: {sys.executable}")

try:
    print("1. 尝试导入 openai llm...")
    from llama_index.llms.openai import OpenAI
    print("✅ llama-index-llms-openai 导入成功")
except Exception as e:
    print(f"❌ 失败! 详细错误: {e}")

try:
    print("2. 尝试导入 openai embedding...")
    from llama_index.embeddings.openai import OpenAIEmbedding
    print("✅ llama-index-embeddings-openai 导入成功")
except Exception as e:
    print(f"❌ 失败! 详细错误: {e}")