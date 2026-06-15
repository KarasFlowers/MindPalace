"""心智漫游基线 diff — 借鉴 Axiomind 的暂存区模式。

核心思想：用户对同一问题卡的多次回答，应先 diff 出增量（新出现的模式），
而非每次都当作全新记忆保存。这避免认知档案被重复内容淹没。

两个职责：
1. find_similar_answers：在历史回答中找与当前回答相似的（向量召回 + source_id 过滤）
2. compute_diff：LLM 比较两次回答，返回是否相似 + 变化了什么
"""

from __future__ import annotations

import logging

from src.config import get_memory_config
from src.llm.client import chat_json
from src.memory.store import find_related_memories

logger = logging.getLogger(__name__)


# 相似度阈值：超过此值才认为是"重复回答"
SIMILAR_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

DIFF_PROMPT = """\
你是 MindPalace 的"回答 diff 分析师"（借鉴 Axiomind 基线去噪）。

用户对同一问题给出了两次回答（可能间隔数周/数月）。你要判断：
1. 两次回答是否本质上相似（is_similar）
2. 如果相似，发生了什么变化（what_changed）
3. 新回答相比旧回答的新颖度（novelty，0.0=完全重复，1.0=全新视角）

判定原则：
- 若两次回答的核心立场、理由、例子都一致 → is_similar=true, novelty 低
- 若立场一致但理由/例子有变化 → is_similar=true, novelty 中等，what_changed 指出变化
- 若立场完全不同 → is_similar=false, novelty 高

以 JSON 输出：
{
  "is_similar": <bool>,
  "what_changed": "一句话说明变化（中文，不超过 60 字；若不相似则留空）",
  "novelty": <0.0-1.0 浮点数>
}

只输出 JSON。
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_similar_answers(
    card_id: str,
    current_response: str,
    limit: int = 3,
    min_similarity: float = SIMILAR_THRESHOLD,
) -> list[dict]:
    """在同一问题卡的历史回答中，找与 current_response 相似的。

    通过 find_related_memories（向量召回）找到相似记忆后，再用 source_id 过滤，
    确保只返回对同一问题卡的历史回答。

    Args:
        card_id: 当前问题卡的 id（source_id）。
        current_response: 用户的当前回答文本。
        limit: 最多返回几条历史回答。
        min_similarity: 向量相似度下限。

    Returns:
        历史回答列表（按相似度降序），每条含 id/similarity/user_response/created_at。
    """
    if not current_response or not current_response.strip():
        return []

    # 向量召回范围放宽（跨所有记忆），再用 source_id 收窄
    candidates = find_related_memories(
        current_response,
        limit=limit * 3,  # 多召回一些，source_id 过滤后可能剩很少
        min_similarity=min_similarity,
    )

    # 过滤：必须是同一问题卡的历史回答
    same_card = [
        m for m in candidates
        if m.get("source_id") == card_id and m.get("source_type") != "article"
    ]
    return same_card[:limit]


def compute_diff(
    current_response: str,
    historical_response: str,
    similarity: float | None = None,
    provider_config: dict | None = None,
) -> dict:
    """LLM 比较两次回答，返回 {is_similar, what_changed, novelty}。

    失败时回退到基于 similarity 的启发式判断（不阻塞流程）。

    Args:
        current_response: 当前回答。
        historical_response: 历史回答。
        similarity: 向量相似度（可选，用于回退判断）。
        provider_config: MEMORY 档 provider 配置。

    Returns:
        {is_similar: bool, what_changed: str, novelty: float}
    """
    if not current_response.strip() or not historical_response.strip():
        return {"is_similar": False, "what_changed": "", "novelty": 1.0}

    cfg = provider_config or get_memory_config()
    prompt_input = (
        f"=== 历史回答 ===\n{historical_response}\n\n"
        f"=== 当前回答 ===\n{current_response}"
    )

    try:
        result = chat_json(DIFF_PROMPT, prompt_input, provider_config=cfg)
    except Exception as exc:
        logger.warning("compute_diff LLM failed, falling back to heuristic: %s", exc)
        return _heuristic_diff(current_response, historical_response, similarity)

    try:
        novelty = float(result.get("novelty", 0.5))
    except (TypeError, ValueError):
        novelty = 0.5
    novelty = max(0.0, min(1.0, novelty))

    return {
        "is_similar": bool(result.get("is_similar", False)),
        "what_changed": str(result.get("what_changed", "")).strip(),
        "novelty": novelty,
    }


def check_and_describe_similarity(
    card_id: str,
    current_response: str,
    provider_config: dict | None = None,
) -> dict | None:
    """一站式：找相似历史回答 + diff。供 save_inquiry_memory 调用。

    Returns:
        若找到相似历史回答，返回 {historical: dict, diff: dict}；
        否则返回 None。
    """
    similars = find_similar_answers(card_id, current_response, limit=1)
    if not similars:
        return None

    historical = similars[0]
    diff = compute_diff(
        current_response,
        historical.get("user_response", ""),
        similarity=historical.get("similarity"),
        provider_config=provider_config,
    )
    return {"historical": historical, "diff": diff}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _heuristic_diff(
    current: str,
    historical: str,
    similarity: float | None,
) -> dict:
    """LLM 不可用时的回退：基于相似度的粗略判断。"""
    if similarity is None:
        # 无相似度信息，保守认为不相似
        return {"is_similar": False, "what_changed": "", "novelty": 1.0}
    is_similar = similarity > 0.8
    return {
        "is_similar": is_similar,
        "what_changed": "" if not is_similar else "（细节可能有变化）",
        "novelty": round(1.0 - similarity, 2),
    }
