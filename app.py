import streamlit as st
import os
import shutil
import pandas as pd
from config import init_settings
# 确保 ingestion 和 rag_engine 模块存在且路径正确
from modules.ingestion import process_single_file, delete_file_from_vector_db
from modules.rag_engine import query_with_vision
from modules.database import init_db, get_all_documents, delete_document_record
import nest_asyncio

# 解决异步循环问题
nest_asyncio.apply()

st.set_page_config(page_title="Gemini 知识库 Pro", page_icon="🧠", layout="wide")
st.title("🧠 Gemini 智能知识库 Pro (结构化版)")

# 初始化 Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# 初始化 DB 和 模型
init_db()
try:
    with st.spinner("🚀 系统初始化中..."):
        init_settings()
except Exception as e:
    st.error(f"❌ 初始化失败: {e}")
    st.stop()

# --- 侧边栏：上传 ---
with st.sidebar:
    st.header("📤 文档上传")
    uploaded_files = st.file_uploader("支持 PDF", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button(f"开始处理 {len(uploaded_files)} 个文件"):
            os.makedirs("data", exist_ok=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            success_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"正在处理 ({i+1}/{len(uploaded_files)}): {uploaded_file.name} ...")
                
                # 保存文件
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 调用处理
                st.toast(f"正在分析元数据: {uploaded_file.name}...")
                is_success, msg, _ = process_single_file(file_path)
                
                if is_success:
                    success_count += 1
                else:
                    st.error(f"{uploaded_file.name}: {msg}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("处理完成！")
            if success_count > 0:
                st.success(f"🎉 成功入库 {success_count} 个文件")
                st.rerun()

# --- 主界面：Tabs 分区 ---
tab_chat, tab_manage = st.tabs(["💬 智能问答", "🗂️ 语料库管理"])

# Tab 1: 问答
with tab_chat:
    # 构建 pdf_map (用于截图)
    pdf_map = {}
    if os.path.exists("data"):
        for f in os.listdir("data"):
            pdf_map[f] = os.path.join("data", f)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("问点什么... (例如: 总结一下2025年的合同)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 深度思考中 (检索 + 视觉 + 结构化数据)..."):
                try:
                    response = query_with_vision(prompt, pdf_map)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"出错: {e}")

# Tab 2: 管理 (结构化数据的核心展示)
with tab_manage:
    st.subheader("📚 知识库全景")
    
    docs = get_all_documents()
    if docs:
        # 转为 DataFrame 展示
        df = pd.DataFrame(docs)
        
        # 挑选展示列 (确保列名与 database.py 中定义的一致)
        # 假设 database.py 返回的 keys 包含: id, filename, doc_type, summary, tags, upload_time, status
        if not df.empty:
            display_columns = ["id", "filename", "doc_type", "summary", "tags", "upload_time", "status"]
            # 过滤掉不存在的列以防报错
            existing_cols = [c for c in display_columns if c in df.columns]
            display_df = df[existing_cols]
            
            st.dataframe(
                display_df, 
                column_config={
                    "summary": st.column_config.TextColumn("AI 摘要", width="medium"),
                    "tags": st.column_config.ListColumn("关键词"),
                    "status": st.column_config.Column("状态"),
                    "doc_type": st.column_config.Column("类型"),
                    "upload_time": st.column_config.Column("上传时间")
                },
                # 🔴 修正点：use_container_width=True -> width="stretch"
                width="stretch",
                hide_index=True
            )
            
            # 删除功能
            st.divider()
            st.caption("🗑️ 数据管理区")
            col1, col2 = st.columns([3, 1])
            with col1:
                file_to_delete = st.selectbox("选择要删除的文件", df["filename"].unique())
            with col2:
                # 稍微加点样式让按钮对其
                st.write("") 
                st.write("")
                if st.button("彻底删除选中文件", type="primary"):
                    if file_to_delete:
                        with st.spinner(f"正在删除 {file_to_delete}..."):
                            # 1. 删向量
                            delete_file_from_vector_db(file_to_delete)
                            # 2. 删数据库
                            delete_document_record(file_to_delete)
                            # 3. 删物理文件 (可选)
                            try:
                                file_path = os.path.join("data", file_to_delete)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                            except Exception as e:
                                st.warning(f"物理文件删除失败: {e}")
                        
                        st.success(f"已删除: {file_to_delete}")
                        st.rerun()
    else:
        st.info("📭 知识库暂无数据，请在侧边栏上传文档。")