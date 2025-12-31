import os
import sys
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

# 🔴 1. 强制设置代理 (解决 WinError 10053)
# 如果你用的是 v2rayN，通常是 10809；Clash 通常是 7890。请根据实际情况修改！
PROXY_URL = "http://127.0.0.1:10808"  
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL

# --- 模型配置 ---
LLM_MODEL_NAME = "models/gemini-3-pro-preview" # 推荐用 1.5 Pro，稳定且支持 Vision
EMBED_MODEL_NAME = "models/gemini-embedding-001" # 谷歌最新的 Embedding 模型

# 🔴 2. Rerank 模型配置 (本地运行，首次下载约 500MB)
# BGE-Reranker 是目前开源界效果最好的重排序模型之一
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

def init_settings():
    """初始化 LlamaIndex 全局设置"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ 未找到 GOOGLE_API_KEY，请检查 .env 文件！")

    print(f"⚙️  正在初始化模型: {LLM_MODEL_NAME} & {RERANK_MODEL_NAME}")

    try:
        # 1. LLM
        Settings.llm = GoogleGenAI(
            model=LLM_MODEL_NAME,
            api_key=api_key,
            temperature=0.1,
        )

        # 2. Embedding
        Settings.embedding = GoogleGenAIEmbedding(
            model_name=EMBED_MODEL_NAME,
            api_key=api_key
        )
        
        print("✅ 全局模型设置成功 (已覆盖默认 OpenAI)")
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        raise