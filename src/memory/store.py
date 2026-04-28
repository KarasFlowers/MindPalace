"""Memory 存储层 — 用户观点 + 认知画像的持久化。

Phase B: 新增向量 embedding 字段，find_related_memories 改为向量召回 + 关键词回退。
"""

import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import sqlite3

from src.config import DB_PATH
from src.memory.profiler import CognitiveProfile

logger = logging.getLogger(__name__)

_CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      INTEGER,
    article_title   TEXT NOT NULL,
    user_response   TEXT NOT NULL,
    stance_summary  TEXT,
    topic_keywords  TEXT,
    core_preference TEXT,
    reasoning_style TEXT,
    emotional_tone  TEXT,
    embedding       BLOB,
    embed_model     TEXT,
    created_at      TEXT NOT NULL
);
"""

_MIGRATION_COLUMNS = [
    ("embedding", "BLOB"),
    ("embed_model", "TEXT"),
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_memories_table():
    """确保 memories 表存在，并执行必要的列迁移。"""
    with _get_conn() as conn:
        conn.execute(_CREATE_MEMORIES_TABLE)
        _migrate_columns(conn)


def _migrate_columns(conn: sqlite3.Connection):
    """为旧表添加 Phase B 新增的列（ALTER TABLE ADD COLUMN 幂等）。"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    for col_name, col_type in _MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type}")
            logger.info("Migrated memories table: added column '%s'", col_name)


def save_memory(
    article_id: int | None,
    article_title: str,
    user_response: str,
    profile: CognitiveProfile,
) -> int:
    """保存一条用户记忆（观点 + 认知画像 + embedding）。返回新记录的 ID。

    如果 embedding 计算失败（API 不可用等），记忆仍会写入但 embedding 为 NULL。
    """
    init_memories_table()
    now = datetime.now(timezone.utc).isoformat()

    embed_blob, embed_model = _safe_embed(user_response)

    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO memories
            (article_id, article_title, user_response, stance_summary,
             topic_keywords, core_preference, reasoning_style, emotional_tone,
             embedding, embed_model, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id,
                article_title,
                user_response,
                profile.stance_summary,
                json.dumps(profile.topic_keywords, ensure_ascii=False),
                json.dumps(profile.core_preference, ensure_ascii=False),
                profile.reasoning_style,
                profile.emotional_tone,
                embed_blob,
                embed_model,
                now,
            ),
        )
        new_id = cursor.lastrowid

    logger.info(
        "Saved memory #%d for article '%s' (embedded=%s)",
        new_id, article_title[:40], embed_blob is not None,
    )
    return new_id


def _safe_embed(text: str) -> tuple[bytes | None, str | None]:
    """尝试计算 embedding，失败时返回 (None, None) 不阻塞存储。"""
    try:
        from src.memory.embedder import get_embedder, vec_to_blob
        embedder = get_embedder()
        vecs = embedder.embed([text])
        return vec_to_blob(vecs[0]), embedder.model_name
    except Exception as exc:
        logger.warning("Embedding failed (will store without vector): %s", exc)
        return None, None


def find_related_memories(
    query_text: str,
    exclude_id: int | None = None,
    limit: int = 5,
    min_similarity: float = 0.35,
) -> list[dict]:
    """向量召回 + 关键词回退，查找与 query_text 语义相关的历史记忆。

    1. 对 query_text 做 embedding，与所有有 embedding 的记忆算余弦相似度。
    2. 若向量召回结果为 0（例如旧记录无 embedding 或 API 不可用），
       则回退到 topic_keywords LIKE 匹配。
    """
    init_memories_table()

    if not query_text or not query_text.strip():
        return []

    # --- 1. 向量召回 ---
    results = _vector_search(query_text, exclude_id, limit, min_similarity)
    if results:
        logger.info("Found %d related memories via vector search", len(results))
        return results

    # --- 2. 关键词回退 ---
    results = _keyword_fallback(query_text, exclude_id, limit)
    logger.info("Found %d related memories via keyword fallback", len(results))
    return results


def _vector_search(
    query_text: str,
    exclude_id: int | None,
    limit: int,
    min_similarity: float,
) -> list[dict]:
    """用 embedding 余弦相似度召回。"""
    try:
        from src.memory.embedder import get_embedder, blob_to_vec, cosine_similarity
        embedder = get_embedder()
        query_vec = embedder.embed([query_text])[0]
    except Exception as exc:
        logger.warning("Embedding query failed, skipping vector search: %s", exc)
        return []

    where = "embedding IS NOT NULL"
    params: list = []
    if exclude_id is not None:
        where += " AND id != ?"
        params.append(exclude_id)

    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM memories WHERE {where}", params
        ).fetchall()

    scored: list[tuple[float, dict]] = []
    for row in rows:
        vec = blob_to_vec(row["embedding"])
        sim = cosine_similarity(query_vec, vec)
        if sim >= min_similarity:
            d = _row_to_dict(row)
            d["similarity"] = round(sim, 4)
            scored.append((sim, d))

    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:limit]]


def _keyword_fallback(
    query_text: str,
    exclude_id: int | None,
    limit: int,
) -> list[dict]:
    """从 query_text 提取关键词，用 LIKE 匹配 topic_keywords 列。"""
    keywords = _extract_keywords(query_text)
    if not keywords:
        return []

    conditions = []
    params: list = []
    for kw in keywords:
        conditions.append("topic_keywords LIKE ?")
        params.append(f"%{kw}%")

    where_clause = " OR ".join(conditions)
    if exclude_id is not None:
        where_clause = f"({where_clause}) AND id != ?"
        params.append(exclude_id)

    query = f"""
        SELECT * FROM memories
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ?
    """
    params.append(limit)

    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    return [_row_to_dict(row) for row in rows]


def _extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """从文本中提取简易关键词。

    英文：提取 >= 2 字母的完整单词。
    中文：对连续汉字生成 2-gram（双字切片），确保短关键词能 LIKE 匹配。
    """
    en_tokens = re.findall(r'[a-zA-Z]{2,}', text)
    zh_seqs = re.findall(r'[\u4e00-\u9fff]{2,}', text)
    zh_bigrams: list[str] = []
    for seq in zh_seqs:
        for i in range(len(seq) - 1):
            zh_bigrams.append(seq[i : i + 2])
    combined = list(dict.fromkeys(en_tokens + zh_bigrams))  # 去重保序
    combined.sort(key=len, reverse=True)
    return combined[:max_keywords]


def _row_to_dict(row: sqlite3.Row) -> dict:
    """将 sqlite3.Row 转为 dict，解析 JSON 字段，移除 embedding blob。"""
    d = dict(row)
    d["topic_keywords"] = json.loads(d.get("topic_keywords") or "[]")
    d["core_preference"] = json.loads(d.get("core_preference") or "[]")
    d.pop("embedding", None)  # 不在 API 层暴露原始 blob
    return d


def get_all_memories(limit: int = 50) -> list[dict]:
    """获取所有记忆，按时间倒序。"""
    init_memories_table()

    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_latest_memory_id() -> int | None:
    """返回最新 memory 的 id，没有记录时返回 None。"""
    init_memories_table()
    with _get_conn() as conn:
        row = conn.execute("SELECT MAX(id) AS mid FROM memories").fetchone()
    return row["mid"] if row and row["mid"] is not None else None


def count_memories_since(since_id: int) -> int:
    """统计 id > since_id 的 memory 条数。"""
    init_memories_table()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM memories WHERE id > ?", (since_id,)
        ).fetchone()
    return row["cnt"] if row else 0


def get_recent_memories(limit: int = 10) -> list[dict]:
    """获取最近 N 条记忆（用于 crystallize），按 id 升序。"""
    init_memories_table()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(row) for row in reversed(rows)]
