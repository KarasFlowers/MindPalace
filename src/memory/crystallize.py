"""认知画像结晶 — 每 N 条 memory 压缩成结构化的认知洞察。

借鉴 Axiomind 的知识金字塔：将用户近期发言压缩为结构化洞察，
按 observation / principle / axiom 三层分类，带 candidate/active 状态。
结构化结果写入数据库和 data/user_profile.md，被 llm/client.py 自动注入所有 LLM 调用。
"""

from __future__ import annotations

import json
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

输入是用户最近 N 次发言的画像标签和核心立场。请把它们压缩成一个结构化的认知洞察。

按 Axiomind 知识金字塔分类：
- **observation**: 从近期发言中观察到的稳定模式（最常见）
- **principle**: 跨多次发言反复出现、可指导未来行动的规则（较抽象）
- **axiom**: 最深层的身份级信念（最抽象、最稳定，需人类才能激活）

判定原则：
1. 单次或少数几次发言 → observation
2. 多次重复出现的稳定价值取向 → principle
3. 反复出现且触及"我是谁"的根本信念 → axiom

以 JSON 输出（严格遵守 schema）：
{
  "type": "observation" | "principle" | "axiom",
  "content": "一句话核心陈述（用第二人称‘你倾向于...’）",
  "confidence": 0.0 到 1.0 之间的浮点数,
  "reasoning": "为什么从这些记忆中提炼出这个结论（一句话）",
  "tags": ["1-3 个中文标签"]
}

只输出 JSON，不要输出多余标题或 markdown 围栏。
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crystallize_if_needed(
    window: int | None = None,
    provider_config: dict | None = None,
) -> dict | None:
    """若自上次结晶后新增 >= window 条 memory，生成一次结构化认知洞察。

    返回结构化字典（同时写入数据库和 user_profile.md），若未触发则返回 None。
    返回字段：type / content / confidence / reasoning / tags / sources / status。
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

    source_ids = [m["id"] for m in recent if m.get("id") is not None]
    prompt_input = _format_memories_for_prompt(recent)

    from src.config import get_memory_config
    cfg = provider_config or get_memory_config()
    raw = chat(CRYSTAL_PROMPT, prompt_input, provider_config=cfg)
    crystal = _parse_crystal(raw, source_ids)

    anchor_id = get_latest_memory_id() or 0
    _save_crystal(crystal, anchor_id, window)
    _append_to_user_profile(crystal)

    logger.info(
        "Crystal saved (anchor=%d, type=%s, confidence=%.2f)",
        anchor_id, crystal["type"], crystal["confidence"],
    )
    return crystal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TYPE_EMOJI = {
    "axiom": "🏔️",
    "principle": "📋",
    "observation": "🔍",
}

_TYPE_LABEL = {
    "axiom": "Axiom",
    "principle": "Principle",
    "observation": "Observation",
}


def _parse_crystal(raw: str, source_ids: list[int]) -> dict:
    """将 LLM 输出解析为结构化字典；解析失败时降级为 observation。"""
    text = (raw or "").strip()
    # 剥离可能的 markdown 围栏
    if text.startswith("```"):
        text = text.split("```", 2)
        text = text[1] if len(text) > 1 else text[0]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Crystal JSON parse failed; falling back to observation. raw[:120]=%r", (raw or "")[:120])
        return {
            "type": "observation",
            "content": (raw or "").strip()[:300] or "（无法生成有效洞察）",
            "confidence": 0.3,
            "reasoning": "LLM 未返回有效 JSON，降级为原始文本",
            "tags": [],
            "sources": source_ids,
            "status": "candidate",
        }

    crystal_type = str(data.get("type", "observation")).strip().lower()
    if crystal_type not in ("observation", "principle", "axiom"):
        crystal_type = "observation"

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    tags = data.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    return {
        "type": crystal_type,
        "content": str(data.get("content", "")).strip(),
        "confidence": confidence,
        "reasoning": str(data.get("reasoning", "")).strip(),
        "tags": [str(t).strip() for t in tags if str(t).strip()],
        "sources": source_ids,
        "status": "candidate",
    }


def _get_last_anchor() -> int:
    """获取最近一次结晶的 anchor_memory_id，若无记录返回 0。"""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT anchor_memory_id FROM profile_crystals "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return row["anchor_memory_id"] if row else 0


def _save_crystal(crystal: dict, anchor_id: int, window: int) -> int:
    """写入 profile_crystals 表（含结构化字段）。"""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO profile_crystals
            (content, anchor_memory_id, window, created_at,
             type, status, confidence, sources, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                crystal["content"],
                anchor_id,
                window,
                now,
                crystal["type"],
                crystal["status"],
                crystal["confidence"],
                json.dumps(crystal["sources"], ensure_ascii=False),
                json.dumps(crystal["tags"], ensure_ascii=False),
            ),
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


def _render_crystal_markdown(crystal: dict) -> str:
    """将结构化洞察渲染为可读的 Markdown 块（注入 user_profile.md）。"""
    emoji = _TYPE_EMOJI.get(crystal["type"], "🔍")
    label = _TYPE_LABEL.get(crystal["type"], crystal["type"])
    status = crystal.get("status", "candidate")
    status_tag = "（候选）" if status == "candidate" else ""

    tags_str = " ".join(f"#{t}" for t in crystal.get("tags", []))
    source_count = len(crystal.get("sources", []))

    lines = [
        f"### {emoji} {label}{status_tag}",
        f"**置信度**: {crystal['confidence']:.2f} | **来源**: {source_count} 条记忆{(' | **标签**: ' + tags_str) if tags_str else ''}",
        "",
        crystal["content"],
    ]
    if crystal.get("reasoning"):
        lines.append("")
        lines.append(f"> {crystal['reasoning']}")
    return "\n".join(lines)


# 终端展示用的颜色映射（按洞察类型）
_TYPE_COLOR = {
    "axiom": "\033[35m",        # MAGENTA
    "principle": "\033[36m",    # CYAN
    "observation": "\033[33m",  # YELLOW
}


def render_crystal_terminal(crystal: dict, colors: dict | None = None) -> str:
    """将结构化洞察渲染为终端彩色文本（用于 CLI 展示）。

    与 `_render_crystal_markdown` 区别：带 ANSI 颜色、缩进、终端友好格式。
    """
    c = colors or {}
    BOLD = c.get("BOLD", "\033[1m")
    DIM = c.get("DIM", "\033[2m")
    RESET = c.get("RESET", "\033[0m")

    type_color = _TYPE_COLOR.get(crystal["type"], DIM)
    emoji = _TYPE_EMOJI.get(crystal["type"], "🔍")
    label = _TYPE_LABEL.get(crystal["type"], crystal["type"])
    status = crystal.get("status", "candidate")
    status_tag = "（候选）" if status == "candidate" else ""
    tags_str = " ".join(f"#{t}" for t in crystal.get("tags", []))
    source_count = len(crystal.get("sources", []))

    lines = [
        f"{BOLD}{type_color}{emoji} {label}{status_tag}{RESET}",
        f"{DIM}置信度 {crystal['confidence']:.2f} · 来源 {source_count} 条记忆"
        + (f" · {tags_str}" if tags_str else "") + RESET,
        "",
        crystal["content"],
    ]
    if crystal.get("reasoning"):
        lines.append("")
        lines.append(f"{DIM}> {crystal['reasoning']}{RESET}")
    return "\n".join(lines)


def _append_to_user_profile(crystal: dict) -> None:
    """将结构化洞察渲染为 Markdown 并追加到 data/user_profile.md，清除 client 缓存。"""
    profile_path = PROJECT_ROOT / "data" / "user_profile.md"
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    separator = "\n\n---\n\n"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    block = f"<!-- Crystal {timestamp} -->\n{_render_crystal_markdown(crystal)}"

    with open(profile_path, "a", encoding="utf-8") as f:
        f.write(separator + block)

    logger.info("Appended crystal to %s", profile_path)

    from src.llm.client import reset_user_profile_cache
    reset_user_profile_cache()
