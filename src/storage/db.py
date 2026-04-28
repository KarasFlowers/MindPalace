"""SQLite 存储层。轻量、零配置，够用即可。"""

import json
import sqlite3
import logging
from datetime import datetime, timezone

from src.config import DB_PATH

logger = logging.getLogger(__name__)

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
    created_at  TEXT NOT NULL
);
"""

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
    with _get_conn() as conn:
        conn.execute(_CREATE_ARTICLES_TABLE)
        conn.execute(_CREATE_SCANNED_TABLE)
        conn.execute(_CREATE_CHAT_SESSIONS_TABLE)
        conn.execute(_CREATE_LEARNING_PROGRESS_TABLE)
        conn.execute(_CREATE_DEBATES_TABLE)
        conn.execute(_CREATE_PROFILE_CRYSTALS_TABLE)
        conn.execute(_CREATE_FEEDBACK_TABLE)
    logger.info("Database initialized at %s", DB_PATH)


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
    result = dict(row)
    result["scores"] = json.loads(result.pop("scores_json", "{}"))
    return result


def list_articles(limit: int = 20) -> list[dict]:
    """列出最近的文章，按总分降序。"""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM articles ORDER BY total_score DESC LIMIT ?", (limit,)
        ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["scores"] = json.loads(d.pop("scores_json", "{}"))
        results.append(d)
    return results


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
