"""The Judge —— 主审 Agent。

两个职责：
1. midcheck(state): 在每轮 rebuttal 结束时评估分歧度，决定是否继续下一轮。
2. finalize(state): 辩论结束时给出最终共识（headline / key_points / tensions）。

Midcheck 用 **fast/router** 档模型（便宜，只要能出 0-1 之间的分歧度和布尔判断即可）；
Finalize 用 **judge** 档模型（最强，保证结论质量）。
"""

from __future__ import annotations

import json
import logging

from src.config import get_judge_config, get_router_config
from src.council.state import DebateState, Turn
from src.llm.client import chat_json

logger = logging.getLogger(__name__)


# ---------------- prompts ----------------

MIDCHECK_SYSTEM_PROMPT = """\
你是 MindPalace 议事厅的"中期裁判"。你只需要完成两件事：

1. 评估本轮辩论的分歧度 disagreement_score（0.0 表示完全共识，1.0 表示完全对立）。
2. 判断是否值得再开一轮（should_continue）。

判断规则：
- 如果各方关键论点已经趋同或对手的反驳已被充分回应，should_continue=false。
- 如果仍有实质性分歧未被解决，且继续讨论可能收敛，should_continue=true。
- 你不需要自己发表观点，只做评估。

严格以如下 JSON 输出：
{
  "disagreement_score": <float 0.0-1.0>,
  "should_continue": <bool>,
  "next_focus": "如果建议继续，下一轮应聚焦哪个分歧点（中文，不超过 60 字；否则空串）"
}
"""


FINALIZE_SYSTEM_PROMPT = """\
你是 MindPalace 议事厅的"主审 Agent"（The Judge）。

你会读到一篇文章的背景，以及多个角色在一轮或多轮辩论中的全部发言。
你的任务是收敛整场辩论，产出一份可被读者直接使用的结论。

严格要求：
- headline 必须是一句话（≤60 字），浓缩本次讨论最值得记住的一点。
- key_points 列出 3-5 条经过辩论后仍然成立的核心洞察（每条≤50 字），要求互不重复。
- remaining_tensions 列出仍未解决的实质性分歧，若已收敛可为空数组。
- recommended_stance 给读者一个可采纳的立场 + 简要理由（中文，≤120 字）。
- 不要再引入文章之外的全新事实；你的角色是整理和裁定，不是补充论据。

严格以如下 JSON 输出：
{
  "headline": "一句话结论",
  "key_points": ["观点 1", "观点 2", "观点 3"],
  "remaining_tensions": ["张力 1", ...],
  "recommended_stance": "推荐读者采纳的立场 + 理由"
}
"""


# ---------------- helpers ----------------

def _format_turn(turn: Turn) -> str:
    """把一个 Turn 序列化为给 Judge 看的文本块。"""
    header = f"[{turn.role_key}] round={turn.round_idx} phase={turn.phase.value}"
    body = json.dumps(turn.content, ensure_ascii=False, indent=2)
    return f"{header}\n{body}"


def _format_transcript(state: DebateState) -> str:
    """把整场辩论串成一段转录文本。"""
    lines = [
        f"文章标题: {state.article_title}",
        f"文章摘要: {state.article_summary}",
        f"难度: {state.difficulty} | 活跃角色: {state.active_roles}",
        f"已完成轮次: {state.round_idx}",
        "",
        "=== 发言记录 ===",
    ]
    for idx, t in enumerate(state.turns):
        lines.append(f"\n--- Turn #{idx} ---")
        lines.append(_format_turn(t))
    return "\n".join(lines)


# ---------------- public API ----------------

def midcheck(state: DebateState, provider_config: dict | None = None) -> dict:
    """中期分歧度检查。失败时返回保守的默认值（继续讨论）。"""
    cfg = provider_config or get_router_config()  # 用便宜档
    transcript = _format_transcript(state)

    try:
        result = chat_json(
            MIDCHECK_SYSTEM_PROMPT,
            transcript,
            provider_config=cfg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Judge.midcheck] 评估失败，保守选择继续讨论: %s", exc)
        return {"disagreement_score": 0.5, "should_continue": True, "next_focus": ""}

    try:
        score = float(result.get("disagreement_score", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))

    should = bool(result.get("should_continue", True))
    next_focus = str(result.get("next_focus", "")).strip()

    logger.info(
        "[Judge.midcheck] disagreement=%.2f continue=%s focus=%s",
        score, should, next_focus[:40],
    )
    return {
        "disagreement_score": score,
        "should_continue": should,
        "next_focus": next_focus,
    }


def finalize(state: DebateState, provider_config: dict | None = None) -> dict:
    """最终共识收敛。失败时返回带 error 标记的降级结果。"""
    cfg = provider_config or get_judge_config()
    transcript = _format_transcript(state)

    try:
        result = chat_json(
            FINALIZE_SYSTEM_PROMPT,
            transcript,
            provider_config=cfg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[Judge.finalize] 收敛失败: %s", exc)
        return {
            "headline": "Judge 未能生成共识（调用失败）。",
            "key_points": [],
            "remaining_tensions": [],
            "recommended_stance": "",
            "error": str(exc),
        }

    # 防御性规整
    return {
        "headline": str(result.get("headline", "")).strip(),
        "key_points": list(result.get("key_points", []) or []),
        "remaining_tensions": list(result.get("remaining_tensions", []) or []),
        "recommended_stance": str(result.get("recommended_stance", "")).strip(),
    }
