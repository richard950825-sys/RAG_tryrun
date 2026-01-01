# --- START OF FILE app.py ---
import os
# 🔴 核心修复：设置 Hugging Face 镜像地址 (必须放在最前面)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import streamlit as st
import shutil
import pandas as pd
from dotenv import load_dotenv
import nest_asyncio

# 引入模块
from config import init_settings
from modules.ingestion import process_single_file, delete_file_from_vector_db
from modules.rag_engine import query_with_vision
from modules.database import init_db, get_all_documents, delete_document_record

# 加载环境变量
load_dotenv()
nest_asyncio.apply()

st.set_page_config(page_title="Gemini 知识库 Pro", page_icon="🧠", layout="wide")

# 🔴 安全增强：密码认证函数
def check_password():
    """如果未认证，返回 False 并显示登录框；否则返回 True"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.title("🔒 系统登录")
    
    # 获取环境变量中的密码，默认 admin123
    SYSTEM_PASSWORD = os.getenv("APP_PASSWORD", "admin123")
    
    password_input = st.text_input("请输入访问密码", type="password")
    
    if st.button("登录"):
        if password_input == SYSTEM_PASSWORD:
            st.session_state.authenticated = True
            st.rerun() # 重新运行以加载主界面
        else:
            st.error("❌ 密码错误")
    
    return False

# --- 主程序入口 ---
def main():
    st.title("🧠 Gemini 智能知识库 Pro (Ubuntu Server版)")

    # 初始化 Session 消息
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 初始化 DB 和 模型
    init_db()
    try:
        # 使用 spinner 防止页面跳动
        with st.spinner("🚀 系统初始化中..."):
            init_settings()
    except Exception as e:
        st.error(f"❌ 初始化失败: {e}")
        st.stop()

    # --- 侧边栏：上传 ---
    with st.sidebar:
        st.header("📤 文档上传")
        # 此处代码与原版保持一致...
        uploaded_files = st.file_uploader("支持 PDF", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            if st.button(f"开始处理 {len(uploaded_files)} 个文件"):
                os.makedirs("data", exist_ok=True)
                progress_bar = st.progress(0)
                status_text = st.empty()
                success_count = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"正在处理 ({i+1}/{len(uploaded_files)}): {uploaded_file.name} ...")
                    file_path = os.path.join("data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.toast(f"正在分析: {uploaded_file.name}...")
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
            
            # 添加退出登录按钮
            st.divider()
            if st.button("🚪 退出登录"):
                st.session_state.authenticated = False
                st.rerun()

    # --- 主界面 Logic (Tab 1 & 2) ---
    # 此处代码与原版保持一致，直接复制你的 Tab 逻辑...
    tab_chat, tab_manage = st.tabs(["💬 智能问答", "🗂️ 语料库管理"])
    
    with tab_chat:
        pdf_map = {}
        if os.path.exists("data"):
            for f in os.listdir("data"):
                pdf_map[f] = os.path.join("data", f)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("问点什么..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🧠 深度思考中..."):
                    try:
                        response = query_with_vision(prompt, pdf_map)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"出错: {e}")

    with tab_manage:
        # 此处代码与原版保持一致...
        st.subheader("📚 知识库全景")
        docs = get_all_documents()
        if docs:
            df = pd.DataFrame(docs)
            if not df.empty:
                # 仅展示存在的列
                display_columns = ["id", "filename", "doc_type", "summary", "tags", "upload_time", "status"]
                existing_cols = [c for c in display_columns if c in df.columns]
                st.dataframe(df[existing_cols], width=1000, hide_index=True) # 修正 width 参数

                st.divider()
                file_to_delete = st.selectbox("选择要删除的文件", df["filename"].unique())
                if st.button("彻底删除选中文件", type="primary"):
                    if file_to_delete:
                        delete_file_from_vector_db(file_to_delete)
                        delete_document_record(file_to_delete)
                        try:
                            fp = os.path.join("data", file_to_delete)
                            if os.path.exists(fp): os.remove(fp)
                        except: pass
                        st.success(f"已删除: {file_to_delete}")
                        st.rerun()
        else:
            st.info("暂无数据")

# 🔴 只有通过密码检查才执行 main()
if check_password():
    main()