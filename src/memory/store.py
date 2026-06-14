"""Memory 存储层 — 用户观点 + 认知画像的持久化。

Phase B: 新增向量 embedding 字段，find_related_memories 改为向量召回 + 关键词回退。
"""

import json
import logging
import re
from datetime import datetime, timezone

import numpy as np
import sqlite3

from src.memory.profiler import CognitiveProfile
from src.storage.db import _get_conn

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
    source_type     TEXT NOT NULL DEFAULT 'article',
    source_id       TEXT,
    links           TEXT,                          -- JSON: {"neighbor_id": weight, ...}（A-MEM 演化链接）
    retrieval_count INTEGER NOT NULL DEFAULT 0,    -- 被召回次数（A-MEM 访问统计）
    created_at      TEXT NOT NULL
);
"""

_MIGRATION_COLUMNS = [
    ("embedding", "BLOB"),
    ("embed_model", "TEXT"),
    ("source_type", "TEXT NOT NULL DEFAULT 'article'"),
    ("source_id", "TEXT"),
    # A-MEM 演化升级新增：
    ("links", "TEXT"),
    ("retrieval_count", "INTEGER NOT NULL DEFAULT 0"),
]



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
    source_type: str = "article",
    source_id: str | None = None,
) -> int:
    """保存一条用户记忆（观点 + 认知画像 + embedding）。返回新记录的 ID。

    如果 embedding 计算失败（API 不可用等），记忆仍会写入但 embedding 为 NULL。
    """
    init_memories_table()
    now = datetime.now(timezone.utc).isoformat()

    embed_blob, embed_model = _safe_embed(
        user_response,
        profile=profile,
    )

    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO memories
            (article_id, article_title, user_response, stance_summary,
             topic_keywords, core_preference, reasoning_style, emotional_tone,
             embedding, embed_model, source_type, source_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                source_type,
                source_id,
                now,
            ),
        )
        new_id = cursor.lastrowid

    logger.info(
        "Saved memory #%d for article '%s' (embedded=%s)",
        new_id, article_title[:40], embed_blob is not None,
    )
    return new_id


def _safe_embed(
    text: str,
    profile: "CognitiveProfile | None" = None,
    stance_summary: str = "",
    topic_keywords: list[str] | None = None,
    core_preference: list[str] | None = None,
) -> tuple[bytes | None, str | None]:
    """尝试计算 embedding，失败时返回 (None, None) 不阻塞存储。

    采用 A-MEM 式元数据增强：嵌入拼接后的 `content + stance + keywords + preferences`
    而非原始文本。若提供 profile 则从中提取元数据，否则使用显式参数（查询端用）。
    """
    from src.memory.embedder import build_enhanced_text, get_embedder, vec_to_blob

    if profile is not None:
        enhanced = build_enhanced_text(
            content=text,
            stance_summary=profile.stance_summary or stance_summary,
            topic_keywords=profile.topic_keywords or topic_keywords or [],
            core_preference=profile.core_preference or core_preference or [],
        )
    else:
        enhanced = build_enhanced_text(
            content=text,
            stance_summary=stance_summary,
            topic_keywords=topic_keywords or [],
            core_preference=core_preference or [],
        )

    try:
        embedder = get_embedder()
        vecs = embedder.embed([enhanced])
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
    """用 embedding 余弦相似度召回。

    查询端同样用增强文本格式嵌入（元数据字段留空），保证与存储端格式一致，
    使余弦相似度可比。
    """
    try:
        from src.memory.embedder import build_enhanced_text, get_embedder, blob_to_vec, cosine_similarity
        embedder = get_embedder()
        enhanced_query = build_enhanced_text(content=query_text)
        query_vec = embedder.embed([enhanced_query])[0]
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
    d["links"] = json.loads(d.get("links") or "{}")
    if d.get("retrieval_count") is None:
        d["retrieval_count"] = 0
    d.pop("embedding", None)  # 不在 API 层暴露原始 blob
    return d


def _increment_retrieval_count(memory_ids: list[int]) -> None:
    """批量递增被召回记忆的 retrieval_count（A-MEM 访问统计）。"""
    if not memory_ids:
        return
    init_memories_table()
    ids = [int(mid) for mid in memory_ids if mid is not None]
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE memories SET retrieval_count = retrieval_count + 1 "
            f"WHERE id IN ({placeholders})",
            ids,
        )


def get_memory(memory_id: int) -> dict | None:
    """按 id 获取单条记忆（含 links/retrieval_count）。"""
    init_memories_table()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def update_memory_links(memory_id: int, links: dict) -> None:
    """更新指定记忆的 links 字段（A-MEM strengthen 用）。"""
    init_memories_table()
    with _get_conn() as conn:
        conn.execute(
            "UPDATE memories SET links = ? WHERE id = ?",
            (json.dumps(links, ensure_ascii=False), memory_id),
        )


def update_memory_tags(memory_id: int, topic_keywords: list[str] | None = None,
                       stance_summary: str | None = None) -> None:
    """轻量演化：更新邻居记忆的 topic_keywords / stance_summary（A-MEM update_neighbor 用）。"""
    init_memories_table()
    sets: list[str] = []
    params: list = []
    if topic_keywords is not None:
        sets.append("topic_keywords = ?")
        params.append(json.dumps(topic_keywords, ensure_ascii=False))
    if stance_summary is not None:
        sets.append("stance_summary = ?")
        params.append(stance_summary)
    if not sets:
        return
    params.append(memory_id)
    with _get_conn() as conn:
        conn.execute(
            f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params
        )


def get_all_memories(limit: int = 50) -> list[dict]:
    """获取所有记忆，按时间倒序。"""
    init_memories_table()

    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_memories_by_source(
    source_types: str | list[str] | tuple[str, ...] | None = None,
    limit: int = 50,
) -> list[dict]:
    """按来源类型获取记忆；未传 source_types 时返回非 article 来源。"""
    init_memories_table()

    with _get_conn() as conn:
        if source_types is None:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE source_type != 'article'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            types = [source_types] if isinstance(source_types, str) else list(source_types)
            if not types:
                return []
            placeholders = ",".join("?" for _ in types)
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE source_type IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [*types, limit],
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


def rebuild_embeddings(limit: int = 0) -> int:
    """用 A-MEM 增强格式重新嵌入历史记忆（存量迁移）。

    对 embedding 为 NULL 或需要用新格式重新计算的记录，基于其完整 profile
    重新构建增强文本并嵌入。

    Args:
        limit: 最多迁移的记录数；0 表示全部。

    Returns:
        成功重新嵌入的记录数。
    """
    init_memories_table()

    from src.memory.profiler import CognitiveProfile

    sql = "SELECT * FROM memories WHERE embedding IS NULL ORDER BY id"
    params: list = []
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)

    with _get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    updated = 0
    for row in rows:
        profile = CognitiveProfile(
            core_preference=json.loads(row["core_preference"] or "[]"),
            reasoning_style=row["reasoning_style"] or "",
            emotional_tone=row["emotional_tone"] or "",
            topic_keywords=json.loads(row["topic_keywords"] or "[]"),
            stance_summary=row["stance_summary"] or "",
        )
        embed_blob, embed_model = _safe_embed(
            row["user_response"], profile=profile
        )
        if embed_blob is None:
            continue
        with _get_conn() as conn:
            conn.execute(
                "UPDATE memories SET embedding = ?, embed_model = ? WHERE id = ?",
                (embed_blob, embed_model, row["id"]),
            )
        updated += 1

    logger.info("Rebuilt embeddings for %d memories", updated)
    return updated
