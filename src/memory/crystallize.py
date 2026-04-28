"""认知画像结晶 — 每 N 条 memory 压缩成一条可读的用户画像片段。

生成的片段追加到 data/user_profile.md，被 llm/client.py 自动注入所有 LLM 调用。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import PROJECT_ROOT, CRYSTAL_WINDOW
from src.llm.client import chat
from src.memory.store import (
    count_memories_since,
    get_latest_memory_id,
    get_recent_memories,
)
from src.storage.db import _get_conn, init_db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CRYSTAL_PROMPT = """\
你是 MindPalace 的"认知画像结晶器"。

输入是用户最近 N 次发言的画像标签和核心立场。
请把它们压缩成一段可读的用户画像片段（markdown，< 300 字），要求：
1. 用第二人称（"你倾向于..."）
2. 指出稳定的价值偏好 + 明显的思维漂移
3. 用一句话给出"这个阶段你最适合被哪种论点挑战"

不要输出多余的标题或分割线，直接输出 markdown 段落。
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crystallize_if_needed(
    window: int | None = None,
    provider_config: dict | None = None,
) -> str | None:
    """若自上次结晶后新增 >= window 条 memory，生成一次画像结晶。

    返回结晶文本（写入数据库和 user_profile.md），若未触发则返回 None。
    """
    window = window or CRYSTAL_WINDOW
    init_db()

    last_anchor = _get_last_anchor()
    new_count = count_memories_since(last_anchor)

    if new_count < window:
        logger.debug(
            "Crystallize skipped: %d new memories < window %d",
            new_count, window,
        )
        return None

    logger.info(
        "Crystallize triggered: %d new memories (window=%d)",
        new_count, window,
    )

    recent = get_recent_memories(limit=window)
    if not recent:
        return None

    prompt_input = _format_memories_for_prompt(recent)

    from src.config import get_memory_config
    cfg = provider_config or get_memory_config()
    crystal = chat(CRYSTAL_PROMPT, prompt_input, provider_config=cfg)

    anchor_id = get_latest_memory_id() or 0
    _save_crystal(crystal, anchor_id, window)
    _append_to_user_profile(crystal)

    logger.info("Crystal saved (anchor=%d, len=%d chars)", anchor_id, len(crystal))
    return crystal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_last_anchor() -> int:
    """获取最近一次结晶的 anchor_memory_id，若无记录返回 0。"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT anchor_memory_id FROM profile_crystals "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["anchor_memory_id"] if row else 0


def _save_crystal(content: str, anchor_id: int, window: int) -> int:
    """写入 profile_crystals 表。"""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO profile_crystals (content, anchor_memory_id, window, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (content, anchor_id, window, now),
        )
        return cursor.lastrowid


def _format_memories_for_prompt(memories: list[dict]) -> str:
    """将 memory 列表格式化为 LLM 输入文本。"""
    parts = []
    for i, m in enumerate(memories, 1):
        parts.append(
            f"--- 记忆 #{i} ---\n"
            f"文章: {m.get('article_title', '?')}\n"
            f"立场: {m.get('stance_summary', '?')}\n"
            f"核心偏好: {m.get('core_preference', [])}\n"
            f"推理模式: {m.get('reasoning_style', '?')}\n"
            f"情感底色: {m.get('emotional_tone', '?')}\n"
            f"原文摘要: {(m.get('user_response') or '')[:200]}"
        )
    return "\n\n".join(parts)


def _append_to_user_profile(crystal: str) -> None:
    """将结晶追加到 data/user_profile.md，并清除 llm/client.py 的缓存。"""
    profile_path = PROJECT_ROOT / "data" / "user_profile.md"
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    separator = "\n\n---\n\n"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    block = f"<!-- Crystal {timestamp} -->\n{crystal}"

    with open(profile_path, "a", encoding="utf-8") as f:
        f.write(separator + block)

    logger.info("Appended crystal to %s", profile_path)

    from src.llm.client import reset_user_profile_cache
    reset_user_profile_cache()
