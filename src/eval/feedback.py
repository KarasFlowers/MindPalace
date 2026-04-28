"""用户反馈收集与存储。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.storage.db import _get_conn, init_db

logger = logging.getLogger(__name__)


def save_feedback(
    debate_id: int,
    rating: str,
    adopted_role: str | None = None,
    note: str | None = None,
) -> int:
    """保存用户对一次 Council 讨论的反馈。

    Args:
        debate_id: 关联的 debate ID。
        rating: "up" | "down" | "adopted"。
        adopted_role: 如果 rating == "adopted"，记录采纳的角色 key。
        note: 可选的用户备注。

    Returns:
        新反馈记录的 ID。
    """
    init_db()
    now = datetime.now(timezone.utc).isoformat()

    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback (debate_id, rating, adopted_role, note, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (debate_id, rating, adopted_role, note, now),
        )
        new_id = cursor.lastrowid

    logger.info("Saved feedback #%d for debate #%d (rating=%s)", new_id, debate_id, rating)
    return new_id


def get_feedback_stats(days: int = 7) -> dict:
    """统计最近 N 天的反馈分布。"""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT rating, COUNT(*) as cnt
            FROM feedback
            WHERE created_at >= datetime('now', ?)
            GROUP BY rating
            """,
            (f"-{days} days",),
        ).fetchall()
    return {row["rating"]: row["cnt"] for row in rows}


def collect_feedback_interactive(debate_id: int) -> str | None:
    """交互式收集用户反馈。返回 rating 或 None（用户跳过）。"""
    print("\n本次 Council 讨论如何？")
    print("  [1] 👍 有启发  [2] 👎 无意义  [3] 📌 采纳某观点  [Enter] 跳过")

    try:
        choice = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    mapping = {"1": "up", "2": "down", "3": "adopted"}
    rating = mapping.get(choice)

    if not rating:
        return None

    adopted_role = None
    note = None

    if rating == "adopted":
        print("  采纳了哪个角色的观点？(critic/synthesizer/mentor)")
        try:
            adopted_role = input("  > ").strip() or None
        except (EOFError, KeyboardInterrupt):
            pass

    try:
        note_input = input("  备注（可选，直接 Enter 跳过）: ").strip()
        if note_input:
            note = note_input
    except (EOFError, KeyboardInterrupt):
        pass

    save_feedback(debate_id, rating, adopted_role, note)
    print(f"  ✅ 反馈已记录 ({rating})")
    return rating
