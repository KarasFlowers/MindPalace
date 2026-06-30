"""SQLite 存储层。轻量、零配置，够用即可。"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from src.config import DB_PATH

logger = logging.getLogger(__name__)
_db_initialized_for: str | None = None

_CREATE_ARTICLES_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url           TEXT UNIQUE NOT NULL,
    title         TEXT NOT NULL,
    source        TEXT NOT NULL,
    summary       TEXT,
    clean_content TEXT,
    source_lang   TEXT,
    translated    INTEGER NOT NULL DEFAULT 0,
    scores_json   TEXT,
    total_score   REAL,
    reasoning     TEXT,
    is_favorite   INTEGER NOT NULL DEFAULT 0,
    favorited_at  TEXT,
    favorite_note TEXT,
    created_at    TEXT NOT NULL
);
"""

_CREATE_ARTICLE_TAGS_TABLE = """
CREATE TABLE IF NOT EXISTS article_tags (
    article_id  INTEGER NOT NULL,
    tag         TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (article_id, tag),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
"""

_ARTICLE_MIGRATION_COLUMNS = [
    ("clean_content", "TEXT"),
    ("source_lang", "TEXT"),
    ("translated", "INTEGER NOT NULL DEFAULT 0"),
    ("is_favorite", "INTEGER NOT NULL DEFAULT 0"),
    ("favorited_at", "TEXT"),
    ("favorite_note", "TEXT"),
]

_CREATE_SCANNED_TABLE = """
CREATE TABLE IF NOT EXISTS scanned_urls (
    url         TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL
);
"""

_CREATE_CHAT_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    mode        TEXT NOT NULL,
    history     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""

_CREATE_PROFILE_CRYSTALS_TABLE = """
CREATE TABLE IF NOT EXISTS profile_crystals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    content           TEXT NOT NULL,
    anchor_memory_id  INTEGER NOT NULL,
    window            INTEGER NOT NULL,
    created_at        TEXT NOT NULL,
    type              TEXT,
    status            TEXT DEFAULT 'candidate',
    confidence        REAL,
    sources           TEXT,
    tags              TEXT
);
"""

# 旧 profile_crystals 表的列迁移（Axiomind 结构化输出升级）
_CRYSTAL_MIGRATION_COLUMNS = [
    ("type", "TEXT"),
    ("status", "TEXT DEFAULT 'candidate'"),
    ("confidence", "REAL"),
    ("sources", "TEXT"),
    ("tags", "TEXT"),
]

_CREATE_FEEDBACK_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id       INTEGER NOT NULL,
    rating          TEXT NOT NULL,
    adopted_role    TEXT,
    note            TEXT,
    created_at      TEXT NOT NULL
);
"""

_CREATE_DEBATES_TABLE = """
CREATE TABLE IF NOT EXISTS debates (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id         INTEGER,
    article_title      TEXT NOT NULL,
    article_summary    TEXT,
    difficulty         TEXT NOT NULL,
    active_roles       TEXT NOT NULL,     -- JSON array
    turns              TEXT NOT NULL,     -- JSON array of Turn dicts
    consensus          TEXT,              -- JSON object from Judge.finalize
    terminated_by      TEXT NOT NULL,
    total_rounds       INTEGER NOT NULL,
    disagreement_score REAL,
    routing_reasoning  TEXT,
    paradigm           TEXT,              -- 讨论范式：debate | report | ...
    created_at         TEXT NOT NULL
);
"""

# 旧 debates 表的列迁移（范式抽象升级）
_DEBATE_MIGRATION_COLUMNS = [
    ("paradigm", "TEXT"),
]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """初始化数据库表。"""
    global _db_initialized_for
    db_path = str(DB_PATH)
    if _db_initialized_for == db_path:
        return

    with _get_conn() as conn:
        conn.execute(_CREATE_ARTICLES_TABLE)
        _migrate_article_columns(conn)
        conn.execute(_CREATE_ARTICLE_TAGS_TABLE)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_article_tags_article_id ON article_tags(article_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_article_tags_tag ON article_tags(tag)")
        conn.execute(_CREATE_SCANNED_TABLE)
        conn.execute(_CREATE_CHAT_SESSIONS_TABLE)
        conn.execute(_CREATE_DEBATES_TABLE)
        _migrate_debate_columns(conn)
        conn.execute(_CREATE_PROFILE_CRYSTALS_TABLE)
        _migrate_crystal_columns(conn)
        conn.execute(_CREATE_FEEDBACK_TABLE)

    _db_initialized_for = db_path
    logger.info("Database initialized at %s", DB_PATH)


def _migrate_article_columns(conn: sqlite3.Connection):
    """为旧 articles 表补齐后续版本新增列。"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
    for col_name, col_type in _ARTICLE_MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type}")
            logger.info("Migrated articles table: added column '%s'", col_name)


def _migrate_crystal_columns(conn: sqlite3.Connection):
    """为旧 profile_crystals 表补齐结构化输出列（Axiomind 升级）。"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(profile_crystals)").fetchall()}
    for col_name, col_type in _CRYSTAL_MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE profile_crystals ADD COLUMN {col_name} {col_type}")
            logger.info("Migrated profile_crystals table: added column '%s'", col_name)


def _migrate_debate_columns(conn: sqlite3.Connection):
    """为旧 debates 表补齐 paradigm 列（范式抽象升级）。"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(debates)").fetchall()}
    for col_name, col_type in _DEBATE_MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE debates ADD COLUMN {col_name} {col_type}")
            logger.info("Migrated debates table: added column '%s'", col_name)


def _row_to_article_dict(row) -> dict:
    d = dict(row)
    d["scores"] = json.loads(d.pop("scores_json", "{}"))
    d["is_favorite"] = bool(d.get("is_favorite", 0))
    d["translated"] = bool(d.get("translated", 0))
    d["tags"] = []
    return d


def _row_to_debate_dict(row) -> dict:
    d = dict(row)
    d["active_roles"] = json.loads(d.get("active_roles") or "[]")
    d["turns"] = json.loads(d.get("turns") or "[]")
    d["consensus"] = json.loads(d["consensus"]) if d.get("consensus") else None
    # paradigm 是范式抽象升级新增的列，旧记录可能为 NULL，默认 debate
    if not d.get("paradigm"):
        d["paradigm"] = "debate"
    return d


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _article_exists(conn: sqlite3.Connection, article_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM articles WHERE id = ?", (article_id,)).fetchone()
    return row is not None


def _normalize_tags(tags: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if not tags:
        return []

    raw_items = [tags] if isinstance(tags, str) else list(tags)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        for candidate in str(item).replace("，", ",").split(","):
            tag = candidate.strip()
            if not tag:
                continue
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(tag)
    return normalized


def _attach_tags(conn: sqlite3.Connection, articles: list[dict]) -> list[dict]:
    if not articles:
        return articles

    article_ids = [article["id"] for article in articles]
    placeholders = ",".join("?" for _ in article_ids)
    rows = conn.execute(
        f"""
        SELECT article_id, tag
        FROM article_tags
        WHERE article_id IN ({placeholders})
        ORDER BY tag COLLATE NOCASE ASC, tag ASC
        """,
        article_ids,
    ).fetchall()

    tag_map = {article_id: [] for article_id in article_ids}
    for row in rows:
        tag_map[row["article_id"]].append(row["tag"])

    for article in articles:
        article["tags"] = tag_map.get(article["id"], [])
    return articles


def save_articles(articles) -> None:
    """保存评分后的文章。重复 URL 自动跳过。"""
    init_db()
    now = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        for a in articles:
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO articles
                    (url, title, source, summary, clean_content, source_lang, translated,
                     scores_json, total_score, reasoning, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        a.url,
                        a.title,
                        a.source,
                        a.summary,
                        a.clean_content,
                        getattr(a, "source_lang", None),
                        1 if getattr(a, "translated", False) else 0,
                        json.dumps(a.scores, ensure_ascii=False),
                        a.total_score,
                        a.reasoning,
                        now,
                    ),
                )
            except sqlite3.Error:
                logger.exception("Failed to save article: %s", a.url)


def get_existing_urls() -> set[str]:
    """获取数据库中已经存在的所有文章 URL（包括加精文章和已扫描文章）。"""
    init_db()
    urls = set()
    with _get_conn() as conn:
        rows_a = conn.execute("SELECT url FROM articles").fetchall()
        rows_s = conn.execute("SELECT url FROM scanned_urls").fetchall()

    for row in rows_a:
        urls.add(row["url"])
    for row in rows_s:
        urls.add(row["url"])
    return urls


def mark_as_scanned(urls: list[str]) -> None:
    """批量记录已扫描过的文章 URL，避免重复浪费。"""
    if not urls:
        return

    init_db()
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        for url in urls:
            conn.execute(
                "INSERT OR IGNORE INTO scanned_urls (url, created_at) VALUES (?, ?)",
                (url, now),
            )


def get_article(article_id: int) -> dict | None:
    """按 ID 查询文章。"""
    init_db()
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        if row is None:
            return None
        article = _row_to_article_dict(row)
        _attach_tags(conn, [article])
    return article


def list_articles(
    limit: int = 20,
    favorites_only: bool = False,
    query: str | None = None,
    tags: str | list[str] | tuple[str, ...] | None = None,
    source: str | None = None,
    days: int | None = None,
) -> list[dict]:
    """列出文章，支持按收藏、关键词、标签、来源和时间筛选。"""
    init_db()

    filters: list[str] = []
    params: list = []
    normalized_tags = _normalize_tags(tags)

    if favorites_only:
        filters.append("is_favorite = 1")

    if query and query.strip():
        like = f"%{query.strip().casefold()}%"
        filters.append(
            """
            (
                LOWER(title) LIKE ?
                OR LOWER(source) LIKE ?
                OR LOWER(COALESCE(summary, '')) LIKE ?
                OR LOWER(COALESCE(clean_content, '')) LIKE ?
                OR LOWER(COALESCE(favorite_note, '')) LIKE ?
                OR EXISTS (
                    SELECT 1
                    FROM article_tags tag_search
                    WHERE tag_search.article_id = articles.id
                      AND LOWER(tag_search.tag) LIKE ?
                )
            )
            """
        )
        params.extend([like, like, like, like, like, like])

    if source and source.strip():
        filters.append("LOWER(source) LIKE ?")
        params.append(f"%{source.strip().casefold()}%")

    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        filters.append("created_at >= ?")
        params.append(cutoff.isoformat())

    for tag in normalized_tags:
        filters.append(
            """
            EXISTS (
                SELECT 1
                FROM article_tags required_tag
                WHERE required_tag.article_id = articles.id
                  AND LOWER(required_tag.tag) = ?
            )
            """
        )
        params.append(tag.casefold())

    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    order_sql = (
        "ORDER BY favorited_at DESC, total_score DESC, created_at DESC"
        if favorites_only
        else "ORDER BY total_score DESC, created_at DESC"
    )

    with _get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM articles {where_sql} {order_sql} LIMIT ?",
            [*params, limit],
        ).fetchall()
        articles = [_row_to_article_dict(row) for row in rows]
        _attach_tags(conn, articles)
    return articles


def set_article_favorite(article_id: int, favorite: bool = True, note: str | None = None) -> bool:
    """收藏或取消收藏文章。返回是否成功找到文章。"""
    init_db()
    now = datetime.now(timezone.utc).isoformat() if favorite else None
    with _get_conn() as conn:
        if favorite:
            cursor = conn.execute(
                """
                UPDATE articles
                SET is_favorite = 1,
                    favorited_at = COALESCE(favorited_at, ?),
                    favorite_note = COALESCE(?, favorite_note)
                WHERE id = ?
                """,
                (now, note, article_id),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE articles
                SET is_favorite = 0, favorited_at = NULL
                WHERE id = ?
                """,
                (article_id,),
            )
    return cursor.rowcount > 0


def set_article_note(article_id: int, note: str | None) -> bool:
    """更新文章备注；空字符串会被视为清空。"""
    init_db()
    cleaned_note = note.strip() if isinstance(note, str) else note
    cleaned_note = cleaned_note if cleaned_note else None
    with _get_conn() as conn:
        cursor = conn.execute(
            "UPDATE articles SET favorite_note = ? WHERE id = ?",
            (cleaned_note, article_id),
        )
    return cursor.rowcount > 0


def get_article_tags(article_id: int) -> list[str]:
    """返回文章标签列表。"""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT tag
            FROM article_tags
            WHERE article_id = ?
            ORDER BY tag COLLATE NOCASE ASC, tag ASC
            """,
            (article_id,),
        ).fetchall()
    return [row["tag"] for row in rows]


def replace_article_tags(article_id: int, tags: str | list[str] | tuple[str, ...] | None) -> bool:
    """用一组新标签覆盖文章原有标签。"""
    init_db()
    normalized_tags = _normalize_tags(tags)
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        if not _article_exists(conn, article_id):
            return False
        conn.execute("DELETE FROM article_tags WHERE article_id = ?", (article_id,))
        for tag in normalized_tags:
            conn.execute(
                """
                INSERT OR IGNORE INTO article_tags (article_id, tag, created_at)
                VALUES (?, ?, ?)
                """,
                (article_id, tag, now),
            )
    return True


def add_article_tags(article_id: int, tags: str | list[str] | tuple[str, ...] | None) -> bool:
    """为文章追加标签。"""
    init_db()
    normalized_tags = _normalize_tags(tags)
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        if not _article_exists(conn, article_id):
            return False
        for tag in normalized_tags:
            conn.execute(
                """
                INSERT OR IGNORE INTO article_tags (article_id, tag, created_at)
                VALUES (?, ?, ?)
                """,
                (article_id, tag, now),
            )
    return True


def remove_article_tags(article_id: int, tags: str | list[str] | tuple[str, ...] | None) -> bool:
    """删除文章的指定标签。"""
    init_db()
    normalized_tags = _normalize_tags(tags)
    with _get_conn() as conn:
        if not _article_exists(conn, article_id):
            return False
        for tag in normalized_tags:
            conn.execute(
                "DELETE FROM article_tags WHERE article_id = ? AND LOWER(tag) = ?",
                (article_id, tag.casefold()),
            )
    return True


def cleanup_old_articles(
    retention_days: int,
    dry_run: bool = False,
    keep_discussed: bool = True,
    keep_tagged: bool = True,
) -> dict:
    """清理超过保留期的普通文章，收藏/标签/讨论上下文默认保留。"""
    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_iso = cutoff.isoformat()

    with _get_conn() as conn:
        protection_clauses = []
        if keep_tagged and _table_exists(conn, "article_tags"):
            protection_clauses.append("id NOT IN (SELECT article_id FROM article_tags)")
        if keep_discussed and _table_exists(conn, "debates"):
            protection_clauses.append(
                "id NOT IN (SELECT article_id FROM debates WHERE article_id IS NOT NULL)"
            )
        if keep_discussed and _table_exists(conn, "memories"):
            protection_clauses.append(
                "id NOT IN (SELECT article_id FROM memories WHERE article_id IS NOT NULL)"
            )
        protected_sql = "".join(f" AND {clause}" for clause in protection_clauses)

        select_sql = f"""
            SELECT id, title, source, created_at, total_score
            FROM articles
            WHERE is_favorite = 0
              AND created_at < ?
              {protected_sql}
            ORDER BY created_at ASC
        """
        delete_sql = f"""
            DELETE FROM articles
            WHERE is_favorite = 0
              AND created_at < ?
              {protected_sql}
        """

        candidates = [dict(row) for row in conn.execute(select_sql, (cutoff_iso,)).fetchall()]
        deleted_count = 0
        if not dry_run and candidates:
            cursor = conn.execute(delete_sql, (cutoff_iso,))
            deleted_count = cursor.rowcount

    return {
        "cutoff": cutoff_iso,
        "dry_run": dry_run,
        "deleted_count": deleted_count,
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


# ---------------- debates ----------------


def _serialize_turns(turns: list) -> str:
    """把 DebateState.turns 序列化为 JSON 字符串。"""
    serialized = []
    for turn in turns:
        serialized.append(
            {
                "role_key": turn.role_key,
                "round_idx": turn.round_idx,
                "phase": turn.phase.value if hasattr(turn.phase, "value") else str(turn.phase),
                "content": turn.content,
                "force_closing": turn.force_closing,
                "tool_calls_used": getattr(turn, "tool_calls_used", 0),
                "tool_log": getattr(turn, "tool_log", []),
            }
        )
    return json.dumps(serialized, ensure_ascii=False)


def save_debate(state, article_id: int | None = None) -> int:
    """把 DebateState 落库，返回新记录 id。"""
    init_db()
    now = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO debates
            (article_id, article_title, article_summary, difficulty, active_roles,
             turns, consensus, terminated_by, total_rounds, disagreement_score,
             routing_reasoning, paradigm, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                article_id if article_id is not None else state.article_id,
                state.article_title,
                state.article_summary,
                state.difficulty,
                json.dumps(state.active_roles, ensure_ascii=False),
                _serialize_turns(state.turns),
                json.dumps(state.consensus, ensure_ascii=False) if state.consensus else None,
                state.terminated_by,
                state.round_idx,
                state.disagreement_score,
                state.routing_reasoning,
                getattr(state, "paradigm", None) or "debate",
                now,
            ),
        )
        new_id = cursor.lastrowid

    logger.info(
        "Saved debate #%d: %s [rounds=%d, terminated_by=%s]",
        new_id,
        state.article_title[:40],
        state.round_idx,
        state.terminated_by,
    )
    return new_id


def get_debate(debate_id: int) -> dict | None:
    """按 ID 读取一次辩论。"""
    init_db()
    with _get_conn() as conn:
        row = conn.execute("SELECT * FROM debates WHERE id = ?", (debate_id,)).fetchone()
    return _row_to_debate_dict(row) if row else None


def list_debates(limit: int = 20) -> list[dict]:
    """列出最近的 debates，按时间倒序。"""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM debates ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_debate_dict(row) for row in rows]


def list_recent_debates_for_article(article_id: int, limit: int = 3) -> list[dict]:
    """返回某篇文章最近的几次讨论。"""
    init_db()
    with _get_conn() as conn:
        if not _table_exists(conn, "debates"):
            return []
        rows = conn.execute(
            """
            SELECT *
            FROM debates
            WHERE article_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (article_id, limit),
        ).fetchall()
    return [_row_to_debate_dict(row) for row in rows]


# ---------------- profile_crystals ----------------

def list_crystals(limit: int = 50) -> list[dict]:
    """列出最近的结构化认知洞察，按时间倒序。"""
    init_db()
    with _get_conn() as conn:
        if not _table_exists(conn, "profile_crystals"):
            return []
        rows = conn.execute(
            "SELECT * FROM profile_crystals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_crystal_dict(row) for row in rows]


def _row_to_crystal_dict(row) -> dict:
    """将 profile_crystals 行转为 dict，解析 JSON 字段。"""
    d = dict(row)
    d["sources"] = json.loads(d.get("sources") or "[]")
    d["tags"] = json.loads(d.get("tags") or "[]")
    if not d.get("type"):
        d["type"] = "observation"
    if not d.get("status"):
        d["status"] = "candidate"
    if d.get("confidence") is None:
        d["confidence"] = 0.0
    return d
