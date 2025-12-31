import os
import qdrant_client
import streamlit as st  # 🔴 新增：引入 streamlit
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.docling import DoclingReader
from llama_index.core import SimpleDirectoryReader
from qdrant_client.http import models

# 引入新模块
from modules.database import add_document_start, update_document_success, update_document_failed
from modules.metadata import extract_metadata_from_text

STORAGE_PATH = "./storage_db"
COLLECTION_NAME = "gemini_rag"

# 🔴 核心修改：使用缓存装饰器，确保全局只有一个 client 实例
@st.cache_resource
def get_client():
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    # 本地模式下，只需初始化一次
    return qdrant_client.QdrantClient(path=STORAGE_PATH)

def delete_file_from_vector_db(filename):
    """从 Qdrant 中物理删除指定文件的所有向量"""
    # 这里调用的 get_client() 会返回缓存的同一个实例，不会触发文件锁
    client = get_client() 
    try:
        # 使用 Filter 删除
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename),
                        ),
                    ],
                )
            ),
        )
        print(f"🗑️ 已从向量库删除: {filename}")
    except Exception as e:
        print(f"⚠️ 向量库删除失败 (可能是集合不存在): {e}")

def process_single_file(file_path):
    filename = os.path.basename(file_path)
    
    # 1. 数据库占位
    add_document_start(filename, file_path)
    print(f"🔄 [开始处理] {filename} ...")
    
    try:
        # 2. Docling 解析
        reader = DoclingReader(export_type="markdown")
        file_extractor = {".pdf": reader}
        dir_reader = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor
        )
        documents = dir_reader.load_data()
        
        if not documents:
            update_document_failed(filename, "解析为空")
            return False, "解析结果为空", 0

        # 3. AI 提取元数据
        full_text_preview = "\n".join([d.text for d in documents])[:5000]
        meta = extract_metadata_from_text(full_text_preview, filename)
        
        # 4. 注入元数据
        for doc in documents:
            doc.metadata.update(meta)
            doc.metadata["filename"] = filename

        # 5. 存入向量库
        client = get_client() # 🔴 获取全局单例 Client
        
        # ⚠️ 注意：每次 process 都要检查集合是否存在，如果不存在会自动创建
        # 但 QdrantVectorStore 可能会尝试重新初始化，这里我们要小心处理
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # 先删除旧向量 (如果重新上传同名文件)
        delete_file_from_vector_db(filename)

        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=Settings.embedding
        )
        
        # 6. 更新数据库状态
        update_document_success(filename, meta, len(documents))
        return True, "成功入库", len(documents)

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_document_failed(filename, str(e))
        return False, f"处理失败: {str(e)}", 0