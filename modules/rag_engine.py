import os
import fitz  # PyMuPDF
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.schema import ImageDocument

# 🔴 修正导入路径：从 sbert_rerank 导入
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from config import RERANK_MODEL_NAME
from modules.ingestion import get_client, COLLECTION_NAME

# 初始化 Reranker (单例模式)
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print(f"📥 正在加载重排序模型 ({RERANK_MODEL_NAME})...首次运行需下载")
        try:
            # 初始化
            _reranker = SentenceTransformerRerank(
                model=RERANK_MODEL_NAME,
                top_n=5,
                device="cpu" # 如果你有NVIDIA显卡改为 "cuda"
            )
            print("✅ Reranker 加载完成")
        except Exception as e:
            print(f"❌ Reranker 加载失败: {e}")
            return None
    return _reranker

def get_retriever_engine():
    """获取查询引擎 (Vector + Rerank)"""
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        return None

    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embedding)
    
    # 1. 初筛：先检索 20 条 (Vector Search)
    base_retriever = index.as_retriever(similarity_top_k=20)
    
    return base_retriever

def query_with_vision(query_text, pdf_path_map):
    """
    Args:
        query_text: 用户问题
        pdf_path_map: dict, {filename: full_path} 用于查找图片
    """
    retriever = get_retriever_engine()
    if not retriever:
        return "⚠️ 知识库为空，请先上传文档。"

    # 1. 初步检索 (Vector Search)
    nodes = retriever.retrieve(query_text)
    if not nodes:
        return "⚠️ 未找到相关文档内容。"

    # 2. 重排序 (Rerank) - 核心升级点
    reranker = get_reranker()
    if reranker:
        print("⚖️ 正在进行重排序 (Reranking)...")
        # Rerank 可能会剔除不相关的节点，只保留 top_n
        nodes = reranker.postprocess_nodes(nodes, query_str=query_text)
    
    # 3. 整理上下文 & 准备截图
    context_str = ""
    related_files_pages = [] # 格式: (file_name, page_idx)
    
    print(f"🎯 最终选定的 {len(nodes)} 个片段来源:")
    for n in nodes:
        # 获取文件名 (Docling 通常会保留文件名在 metadata)
        # 兼容性处理：如果 metadata 里没有 file_name，尝试从 node id 或其他字段推断，或者由 app 传入上下文
        f_name = n.metadata.get('file_name', 'unknown')
        page_label = n.metadata.get('page_label', '1')
        
        print(f"   - {f_name} (Page {page_label}): {n.score if n.score else 'N/A'}")
        
        context_str += f"--- 文档: {f_name} [第 {page_label} 页] ---\n{n.text}\n\n"
        
        try:
            p_idx = int(page_label) - 1
            related_files_pages.append((f_name, p_idx))
        except:
            pass

    # 4. 动态截图 (VisRAG)
    image_docs = []
    # 去重并只取前 2 张图，防止 Token 爆炸
    unique_pages = list(set(related_files_pages))[:2] 
    
    for f_name, p_idx in unique_pages:
        full_path = pdf_path_map.get(f_name)
        # 如果找不到文件名映射，尝试去 data 目录下模糊匹配一下
        if not full_path and os.path.exists("data"):
            possible_path = os.path.join("data", f_name)
            if os.path.exists(possible_path):
                full_path = possible_path

        if full_path and os.path.exists(full_path):
            try:
                doc = fitz.open(full_path)
                if 0 <= p_idx < len(doc):
                    print(f"🖼️ 正在截取: {f_name} 第 {p_idx+1} 页")
                    pix = doc[p_idx].get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    image_docs.append(ImageDocument(image=img_bytes))
                doc.close()
            except Exception as e:
                print(f"截图失败: {e}")

    # 5. 发送给 Gemini
    prompt = f"""
    请根据【上下文文本】和【附图】(文档原始页面)回答用户问题。
    如果图片中有表格或图表，请优先参考图片内容。
    
    用户问题: {query_text}
    
    上下文文本:
    {context_str}
    """
    
    print("🤖 正在请求 Gemini...")
    response = Settings.llm.complete(prompt, image_documents=image_docs)
    
    return response.text