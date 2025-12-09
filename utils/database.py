# utils/database.py

"""
SQLite database helpers for Hyperbolic caching.
Hyperbolic 缓存用的 SQLite 数据库工具函数。
"""

import os
import sqlite3
import json
from typing import Optional


def get_db_path() -> str:
    """
    Resolve absolute path to the SQLite database file.
    解析 SQLite 数据库文件的绝对路径（不依赖当前工作目录）。
    """
    utils_dir = os.path.dirname(os.path.abspath(__file__))   # utils/
    project_root = os.path.dirname(utils_dir)                # project root
    return os.path.join(project_root, "ICM.sqlite3")


def init_db() -> None:
    """
    Initialize the SQLite database and ensure required tables and indices exist.
    初始化 SQLite 数据库，确保所需的表和索引存在。
    """
    db_path = get_db_path()
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_requests (
            id INTEGER PRIMARY KEY,
            model_name TEXT,
            prompt TEXT,
            top_logprobs INTEGER,
            max_tokens INTEGER,
            echo INTEGER,
            timeout REAL,
            response TEXT
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON llm_requests(model_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt ON llm_requests(prompt)")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_model_prompt_params
        ON llm_requests(model_name, prompt, top_logprobs, max_tokens, echo, timeout)
    """)

    conn.commit()
    conn.close()


def fetch_cached_result(
    model: str,
    prompt: str,
    top_logprobs: int,
    max_tokens: int,
    echo: bool,
    timeout: float
) -> Optional[dict]:
    """
    Fetch cached API result from database if exists.
    从数据库中查询是否存在已缓存的 API 结果。
    """
    init_db()
    db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = (
        "SELECT response FROM llm_requests "
        "WHERE model_name = ? AND prompt = ? AND top_logprobs = ? "
        "AND max_tokens = ? AND echo = ? AND timeout = ?"
    )
    params = (model, prompt, int(top_logprobs), int(max_tokens), int(echo), float(timeout))

    try:
        row = cursor.execute(query, params).fetchone()
    finally:
        conn.close()

    if not row:
        return None

    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return row[0]


def save_result_to_db(
    model: str,
    prompt: str,
    top_logprobs: int,
    max_tokens: int,
    echo: bool,
    timeout: float,
    data: dict
) -> None:
    """
    Save new API result to database.
    将新的 API 结果写入数据库。
    """
    init_db()
    db_path = get_db_path()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO llm_requests (model_name, prompt, top_logprobs, max_tokens, echo, timeout, response) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            model,
            prompt,
            int(top_logprobs),
            int(max_tokens),
            int(echo),
            float(timeout),
            json.dumps(data, ensure_ascii=False),
        )
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("[DB] ICM.sqlite3 initialized successfully.")
