"""记忆演化引擎 — 借鉴 A-MEM 的 process_memory 算法。

每次添加新记忆时，可选地触发演化：
1. 找 5 个最近邻居（向量召回）
2. LLM 判断是否应演化（should_evolve）+ 执行动作（strengthen / update_neighbor）
3. strengthen：在新记忆和邻居之间建立链接 + 可选更新自身 tags
4. update_neighbor：根据新记忆更新邻居的 topic_keywords / stance_summary（轻量演化）

设计原则（与 A-MEM 的差异）：
- 精简为 4 字段输出（should_evolve / action / suggested_links / tags_to_update），
  更聚焦、token 更省
- 失败不阻断核心流程（沿用项目降级策略）
- 邻居权重基于向量相似度，存入 links 字段
"""

from __future__ import annotations

import json
import logging

from src.config import get_memory_config
from src.llm.client import chat_json
from src.memory.store import (
    find_related_memories,
    get_memory,
    update_memory_links,
    update_memory_tags,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EVOLUTION_PROMPT = """\
你是 MindPalace 的"记忆演化引擎"（借鉴 A-MEM 设计）。

输入是一条新记忆和它的 5 个最近邻居（按相似度排序）。你要判断：这条新记忆是否应该与邻居建立关联，并触发轻量演化。

判定原则：
- 如果新记忆与某邻居讨论同一主题、延续同一立场、或形成对比 → 应演化
- 如果新记忆与邻居完全不相关（向量近但语义远）→ 不演化
- 如果新记忆是对旧观点的更新/修正 → 用 update_neighbor 更新旧记忆

以 JSON 输出（严格遵守 schema）：
{
  "should_evolve": <bool>,
  "action": "strengthen" | "update_neighbor" | "none",
  "suggested_links": [<邻居序号，如 "1,3" 表示第 1 和第 3 个邻居应被链接>],
  "tags_to_update": {"<邻居序号>": ["新关键词1", "新关键词2"]}
}

字段说明：
- should_evolve：是否触发演化
- action：演化动作（strengthen 建链接；update_neighbor 改邻居标签；none 不动）
- suggested_links：用逗号分隔的邻居序号字符串，指出哪些邻居应与新记忆建立链接（仅 strengthen 用）
- tags_to_update：邻居序号 → 应补充的关键词列表（update_neighbor 用，最多 2 个邻居）

只输出 JSON，不要多余解释。
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def link_memories(
    new_memory_id: int,
    provider_config: dict | None = None,
    neighbor_limit: int = 5,
) -> dict:
    """对一条新记忆执行 A-MEM 式演化：找邻居 → LLM 决策 → 建链接/更新邻居。

    Args:
        new_memory_id: 新保存的记忆 id。
        provider_config: MEMORY 档 provider 配置。
        neighbor_limit: 查找的最近邻居数量（默认 5）。

    Returns:
        演化结果 dict：
        - evolved: bool — 是否触发了演化
        - action: str — 实际执行的动作
        - links_created: int — 建立的链接数
        - neighbors_updated: int — 更新的邻居数
        - error: str — 失败时的错误信息（成功时无此字段）
    """
    new_memory = get_memory(new_memory_id)
    if not new_memory:
        return {"evolved": False, "action": "none", "links_created": 0,
                "neighbors_updated": 0, "error": f"memory #{new_memory_id} not found"}

    neighbors = find_related_memories(
        new_memory.get("user_response", ""),
        exclude_id=new_memory_id,
        limit=neighbor_limit,
    )
    if not neighbors:
        logger.debug("[Evolution] memory #%d: no neighbors, skip", new_memory_id)
        return {"evolved": False, "action": "none", "links_created": 0,
                "neighbors_updated": 0}

    cfg = provider_config or get_memory_config()
    prompt_input = _format_evolution_input(new_memory, neighbors)

    try:
        result = chat_json(EVOLUTION_PROMPT, prompt_input, provider_config=cfg)
    except Exception as exc:
        logger.warning("[Evolution] memory #%d: LLM decision failed: %s", new_memory_id, exc)
        return {"evolved": False, "action": "none", "links_created": 0,
                "neighbors_updated": 0, "error": str(exc)}

    decision = _parse_decision(result, neighbors)
    if not decision["should_evolve"] or decision["action"] == "none":
        logger.debug("[Evolution] memory #%d: LLM decided not to evolve", new_memory_id)
        return {"evolved": False, "action": "none", "links_created": 0,
                "neighbors_updated": 0}

    links_created, neighbors_updated = _apply_decision(
        new_memory_id, decision, neighbors
    )

    logger.info(
        "[Evolution] memory #%d: evolved (action=%s, links=%d, neighbors_updated=%d)",
        new_memory_id, decision["action"], links_created, neighbors_updated,
    )
    return {
        "evolved": True,
        "action": decision["action"],
        "links_created": links_created,
        "neighbors_updated": neighbors_updated,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_evolution_input(new_memory: dict, neighbors: list[dict]) -> str:
    """格式化新记忆 + 邻居列表为 LLM 输入文本。"""
    parts = [
        "=== 新记忆 ===",
        f"内容: {new_memory.get('user_response', '')}",
        f"立场: {new_memory.get('stance_summary', '')}",
        f"关键词: {new_memory.get('topic_keywords', [])}",
        f"偏好: {new_memory.get('core_preference', [])}",
        "",
        f"=== 最近邻居（共 {len(neighbors)} 条）===",
    ]
    for i, n in enumerate(neighbors, 1):
        parts.append(
            f"\n--- 邻居 #{i} (id={n.get('id')}, 相似度={n.get('similarity', 0):.2f}) ---\n"
            f"内容: {(n.get('user_response') or '')[:200]}\n"
            f"立场: {n.get('stance_summary', '')}\n"
            f"关键词: {n.get('topic_keywords', [])}"
        )
    return "\n".join(parts)


def _parse_decision(result: dict, neighbors: list[dict]) -> dict:
    """解析 LLM 决策为结构化 dict。失败时回退到不演化。"""
    try:
        should_evolve = bool(result.get("should_evolve", False))
    except (TypeError, ValueError):
        should_evolve = False

    action = str(result.get("action", "none")).strip().lower()
    if action not in ("strengthen", "update_neighbor", "none"):
        action = "none"
    if not should_evolve:
        action = "none"

    # suggested_links 可能是 "1,3" 字符串或 [1,3] 列表
    raw_links = result.get("suggested_links", "")
    link_indices: list[int] = []
    if isinstance(raw_links, str):
        for part in raw_links.split(","):
            part = part.strip()
            if part.isdigit():
                link_indices.append(int(part))
    elif isinstance(raw_links, list):
        for part in raw_links:
            try:
                link_indices.append(int(part))
            except (TypeError, ValueError):
                continue

    # tags_to_update: {"1": ["kw1"], "3": ["kw2"]}
    raw_tags = result.get("tags_to_update", {}) or {}
    tags_map: dict[int, list[str]] = {}
    if isinstance(raw_tags, dict):
        for k, v in raw_tags.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if isinstance(v, list):
                tags_map[idx] = [str(t).strip() for t in v if str(t).strip()]

    return {
        "should_evolve": should_evolve,
        "action": action,
        "link_indices": link_indices,  # 1-based 邻居序号
        "tags_map": tags_map,          # 1-based 邻居序号 → 关键词
    }


def _apply_decision(
    new_memory_id: int,
    decision: dict,
    neighbors: list[dict],
) -> tuple[int, int]:
    """执行演化决策，返回 (links_created, neighbors_updated)。"""
    links_created = 0
    neighbors_updated = 0

    action = decision["action"]

    # strengthen：在新记忆的 links 字段中加入选中的邻居 id
    if action == "strengthen":
        new_memory = get_memory(new_memory_id) or {}
        existing_links: dict = new_memory.get("links") or {}
        for idx in decision["link_indices"]:
            if 1 <= idx <= len(neighbors):
                neighbor = neighbors[idx - 1]
                nid = neighbor.get("id")
                if nid is None:
                    continue
                nid_key = str(nid)
                # 权重 = 向量相似度（已存在则取较大值，不覆盖）
                weight = round(neighbor.get("similarity", 0.5), 4)
                if nid_key in existing_links:
                    existing_links[nid_key] = max(existing_links[nid_key], weight)
                else:
                    existing_links[nid_key] = weight
                links_created += 1
        if links_created:
            update_memory_links(new_memory_id, existing_links)

    # update_neighbor：更新邻居的 topic_keywords（合并而非覆盖）
    if action in ("strengthen", "update_neighbor"):
        for idx, new_tags in decision["tags_map"].items():
            if 1 <= idx <= len(neighbors):
                neighbor = neighbors[idx - 1]
                nid = neighbor.get("id")
                if nid is None:
                    continue
                # 合并：旧关键词 + 新关键词，去重保序，最多 8 个
                old_tags = neighbor.get("topic_keywords") or []
                merged = list(dict.fromkeys(old_tags + new_tags))[:8]
                if merged != old_tags:
                    update_memory_tags(nid, topic_keywords=merged)
                    neighbors_updated += 1

    return links_created, neighbors_updated
