import json
from llama_index.core import Settings

def extract_metadata_from_text(text_content, filename):
    """
    使用 LLM 提取文档元数据
    """
    # 截取前 3000 字，通常包含标题、摘要和关键信息
    context = text_content[:3000]
    
    prompt = f"""
    你是一个专业的文档分析助手。请分析以下文档片段（来自文件：{filename}）：
    
    ---文档开始---
    {context}
    ---文档结束---
    
    请提取以下信息，并严格以 JSON 格式输出，不要包含 Markdown 格式标记（如 ```json）：
    1. "doc_type": 文档类型 (如: 合同, 技术手册, 财报, 简历, 政策文件, 其他)
    2. "summary": 50字以内的简短摘要
    3. "keywords": [关键实体1, 关键实体2, 年份, 核心术语] (最多5个)
    4. "doc_date": 文档中提到的主要日期 (格式 YYYY-MM-DD，如果找不到则为 null)
    
    输出示例:
    {{
        "doc_type": "合同",
        "summary": "关于甲乙双方采购服务器的协议",
        "keywords": ["服务器", "采购", "2025"],
        "doc_date": "2025-01-01"
    }}
    """
    
    try:
        response = Settings.llm.complete(prompt)
        # 清洗数据，防止 LLM 返回 ```json 包裹
        json_str = response.text.replace("```json", "").replace("```", "").strip()
        metadata = json.loads(json_str)
        return metadata
    except Exception as e:
        print(f"⚠️ 元数据提取失败: {e}")
        # 返回兜底数据
        return {
            "doc_type": "General",
            "summary": "元数据提取失败",
            "keywords": [],
            "doc_date": None
        }