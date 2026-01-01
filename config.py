# --- START OF FILE config.py ---

import os
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv

load_dotenv()

# --- 1. 网络配置 (GA加速环境) ---
# 如果你的 GA 加速是“透明代理”（直接能连外网），这里不需要设 HTTP_PROXY。
# 如果 GA 加速提供了本地端口（比如 127.0.0.1:7890），请在 .env 里填上。
PROXY_URL = os.getenv("HTTP_PROXY")
if PROXY_URL:
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    print(f"🌍 检测到代理配置，已应用: {PROXY_URL}")

# --- 2. 模型配置 ---
# 设备配置
DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")

# Google API Key
API_KEY = os.getenv("GOOGLE_API_KEY")

# 模型名称 (回归 Google 官方命名，如 models/gemini-1.5-pro)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "models/gemini-1.5-pro")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "models/text-embedding-004")

# Rerank 模型
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

def init_settings():
    """初始化 LlamaIndex 全局设置 (Google 原生模式)"""
    if not API_KEY:
        raise ValueError("❌ 未找到 GOOGLE_API_KEY，请检查 .env 文件！")

    print(f"🚀 初始化 Google 原生 SDK 模式...")
    print(f"🧠 LLM: {LLM_MODEL_NAME}")
    print(f"🧬 Embedding: {EMBED_MODEL_NAME}")

    try:
        # 1. 初始化 LLM
        # GoogleGenAI 类会自动处理 "models/" 前缀，也会自动读取系统环境变量中的代理
        Settings.llm = GoogleGenAI(
            model=LLM_MODEL_NAME,
            api_key=API_KEY,
            temperature=0.1,
            # Google 原生 SDK 不需要像 OpenAI 那样显式配置 http_client，它会自动读取 os.environ
        )

        # 2. 初始化 Embedding
        Settings.embedding = GoogleGenAIEmbedding(
            model_name=EMBED_MODEL_NAME,
            api_key=API_KEY
        )
        
        print(f"✅ 模型初始化成功 (Device: {DEVICE})")
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        raise