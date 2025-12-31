import sqlite3
import json
from datetime import datetime

DB_PATH = "corpus.db"

def init_db():
    """初始化 SQLite 数据库"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 创建文档表
    # status: 'processing', 'completed', 'failed'
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            file_path TEXT,
            upload_time TEXT,
            doc_type TEXT,
            summary TEXT,
            tags TEXT,
            page_count INTEGER,
            status TEXT,
            error_msg TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_document_start(filename, file_path):
    """开始处理前，先占位"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT OR REPLACE INTO documents (filename, file_path, upload_time, status)
            VALUES (?, ?, ?, ?)
        ''', (filename, file_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "processing"))
        conn.commit()
        return c.lastrowid
    finally:
        conn.close()

def update_document_success(filename, meta_data, page_count):
    """处理成功，更新元数据"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # meta_data 是个字典，我们需要拆解存入列中，或者存为 JSON 字符串
    # 这里演示拆解存入关键字段
    doc_type = meta_data.get("doc_type", "Unknown")
    summary = meta_data.get("summary", "")
    tags = ", ".join(meta_data.get("keywords", []))
    
    c.execute('''
        UPDATE documents 
        SET status = 'completed', doc_type = ?, summary = ?, tags = ?, page_count = ?, error_msg = NULL
        WHERE filename = ?
    ''', (doc_type, summary, tags, page_count, filename))
    conn.commit()
    conn.close()

def update_document_failed(filename, error_msg):
    """处理失败"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        UPDATE documents 
        SET status = 'failed', error_msg = ?
        WHERE filename = ?
    ''', (str(error_msg), filename))
    conn.commit()
    conn.close()

def get_all_documents():
    """获取所有文档列表 (用于前端展示)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让结果像字典一样访问
    c = conn.cursor()
    c.execute("SELECT * FROM documents ORDER BY upload_time DESC")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def delete_document_record(filename):
    """删除数据库记录"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()