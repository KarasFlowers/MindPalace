"""SQLite 存储层。轻量、零配置，够用即可。"""

import json
import sqlite3
import logging
from datetime import datetime, timedelta, timezone

from src.config import DB_PATH

logger = logging.getLogger(__name__)
_db_initialized = False

_CREATE_ARTICLES_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT UNIQUE NOT NULL,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL,
    summary     TEXT,
    clean_content TEXT,
    scores_json TEXT,
    total_score REAL,
    reasoning   TEXT,
    is_favorite INTEGER NOT NULL DEFAULT 0,
    favorited_at TEXT,
    favorite_note TEXT,
    created_at  TEXT NOT NULL
);
"""

_ARTICLE_MIGRATION_COLUMNS = [
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

_CREATE_LEARNING_PROGRESS_TABLE = """
CREATE TABLE IF NOT EXISTS learning_progress (
    module_name     TEXT PRIMARY KEY,
    current_stage   INTEGER NOT NULL,
    mastery_level   INTEGER NOT NULL,
    last_context    TEXT,
    updated_at      TEXT NOT NULL
);
"""

_CREATE_PROFILE_CRYSTALS_TABLE = """
CREATE TABLE IF NOT EXISTS profile_crystals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    content           TEXT NOT NULL,
    anchor_memory_id  INTEGER NOT NULL,
    window            INTEGER NOT NULL,
    created_at        TEXT NOT NULL
);
"""

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
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      INTEGER,
    article_title   TEXT NOT NULL,
    article_summary TEXT,
    difficulty      TEXT NOT NULL,
    active_roles    TEXT NOT NULL,     -- JSON array
    turns           TEXT NOT NULL,     -- JSON array of Turn dicts
    consensus       TEXT,              -- JSON object from Judge.finalize
    terminated_by   TEXT NOT NULL,
    total_rounds    INTEGER NOT NULL,
    disagreement_score REAL,
    routing_reasoning  TEXT,
    created_at      TEXT NOT NULL
);
"""


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表。"""
    global _db_initialized
    if _db_initialized:
        return
    with _get_conn() as conn:
        conn.execute(_CREATE_ARTICLES_TABLE)
        _migrate_article_columns(conn)
        conn.execute(_CREATE_SCANNED_TABLE)
        conn.execute(_CREATE_CHAT_SESSIONS_TABLE)
        conn.execute(_CREATE_LEARNING_PROGRESS_TABLE)
        conn.execute(_CREATE_DEBATES_TABLE)
        conn.execute(_CREATE_PROFILE_CRYSTALS_TABLE)
        conn.execute(_CREATE_FEEDBACK_TABLE)
    _db_initialized = True
    logger.info("Database initialized at %s", DB_PATH)


def _migrate_article_columns(conn: sqlite3.Connection):
    """为旧 articles 表补齐收藏相关列。"""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(articles)").fetchall()}
    for col_name, col_type in _ARTICLE_MIGRATION_COLUMNS:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type}")
            logger.info("Migrated articles table: added column '%s'", col_name)


def _row_to_article_dict(row) -> dict:
    d = dict(row)
    d["scores"] = json.loads(d.pop("scores_json", "{}"))
    d["is_favorite"] = bool(d.get("is_favorite", 0))
    return d


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


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
                    (url, title, source, summary, clean_content, scores_json, total_score, reasoning, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        a.url,
                        a.title,
                        a.source,
                        a.summary,
                        a.clean_content,
                        json.dumps(a.scores, ensure_ascii=False),
                        a.total_score,
                        a.reasoning,
                        now,
                    ),
                )
            except Exception:
                logger.exception("Failed to save article: %s", a.url)


def get_existing_urls() -> set[str]:
    """获取数据库中已经存在的所有文章 URL（包括加精文章和已扫描文章）。"""
    init_db()
    urls = set()
    with _get_conn() as conn:
        rows_a = conn.execute("SELECT url FROM articles").fetchall()
        rows_s = conn.execute("SELECT url FROM scanned_urls").fetchall()
    
    for r in rows_a: urls.add(r["url"])
    for r in rows_s: urls.add(r["url"])
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
        row = conn.execute(
            "SELECT * FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
    if row is None:
        return None
    return _row_to_article_dict(row)


def list_articles(limit: int = 20, favorites_only: bool = False) -> list[dict]:
    """列出最近的文章，按总分降序。"""
    init_db()
    with _get_conn() as conn:
        if favorites_only:
            rows = conn.execute(
                """
                SELECT * FROM articles
                WHERE is_favorite = 1
                ORDER BY favorited_at DESC, total_score DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM articles ORDER BY total_score DESC LIMIT ?", (limit,)
            ).fetchall()
    return [_row_to_article_dict(row) for row in rows]


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
                SET is_favorite = 0, favorited_at = NULL, favorite_note = NULL
                WHERE id = ?
                """,
                (article_id,),
            )
    return cursor.rowcount > 0


def cleanup_old_articles(
    retention_days: int,
    dry_run: bool = False,
    keep_discussed: bool = True,
) -> dict:
    """清理超过保留期的普通文章，收藏文章永远保留。"""
    init_db()
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_iso = cutoff.isoformat()

    with _get_conn() as conn:
        protection_clauses = []
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
    for t in turns:
        serialized.append({
            "role_key": t.role_key,
            "round_idx": t.round_idx,
            "phase": t.phase.value if hasattr(t.phase, "value") else str(t.phase),
            "content": t.content,
            "force_closing": t.force_closing,
            "tool_calls_used": getattr(t, "tool_calls_used", 0),
            "tool_log": getattr(t, "tool_log", []),
        })
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
             routing_reasoning, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                now,
            ),
        )
        new_id = cursor.lastrowid

    logger.info(
        "Saved debate #%d: %s [rounds=%d, terminated_by=%s]",
        new_id, state.article_title[:40], state.round_idx, state.terminated_by,
    )
    return new_id


def _row_to_debate_dict(row) -> dict:
    d = dict(row)
    d["active_roles"] = json.loads(d.get("active_roles") or "[]")
    d["turns"] = json.loads(d.get("turns") or "[]")
    d["consensus"] = json.loads(d["consensus"]) if d.get("consensus") else None
    return d


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
            "SELECT * FROM debates ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_debate_dict(r) for r in rows]
