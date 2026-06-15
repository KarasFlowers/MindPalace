"""心智漫游回答分析。"""

from __future__ import annotations

import logging

from src.config import get_memory_config
from src.inquiry.types import PromptCard
from src.llm.client import chat_json

logger = logging.getLogger(__name__)

_ANALYSIS_SYSTEM_PROMPT = """\
你是 MindPalace 的心智漫游引导者。你不会做心理诊断，也不会给人生定论。
你的任务是基于用户对一个问题的回答，提炼其中的价值排序、隐藏前提和可继续思考的方向。

根据问题类型调整语气：
- self: 温和、镜像、帮助用户看见自己的模式。
- philosophy: 澄清立场，指出隐藏前提，给一个反例或挑战。
- thought_experiment: 关注选择背后的价值排序，并提出一个条件变体。

严格输出 JSON：
{
  "core_stance": "用户核心立场/自我线索，中文，不超过80字",
  "hidden_assumption": "隐藏前提或反复模式，中文，不超过80字",
  "reflection": "一段简短反馈，中文，不超过160字",
  "followup_question": "一个值得继续思考的问题，中文，不超过80字"
}
"""


def analyze_response(
    card: PromptCard,
    user_response: str,
    provider_config: dict | None = None,
) -> dict:
    """分析用户回答，失败时返回可展示的降级结构。"""
    context = f"\n设定:\n{card.context}" if card.context else ""
    followups = "\n".join(f"- {item}" for item in card.followups[:4])
    twists = "\n".join(f"- {item}" for item in card.twists[:3])
    user_prompt = f"""
问题类型: {card.kind}
标题: {card.title}{context}
问题: {card.prompt}
参考追问:
{followups or '- 无'}
条件变体:
{twists or '- 无'}

用户回答:
{user_response}
""".strip()

    try:
        result = chat_json(
            _ANALYSIS_SYSTEM_PROMPT,
            user_prompt,
            provider_config=provider_config or get_memory_config(),
        )
    except Exception as exc:
        logger.warning("Inquiry analysis failed: %s", exc)
        return {
            "core_stance": "暂时无法自动提炼核心立场。",
            "hidden_assumption": "分析失败，但原始回答仍可保存。",
            "reflection": "这次回答已经记录了一个值得回看的自我切片。",
            "followup_question": card.followups[0] if card.followups else "这个回答中，哪一句最接近你的真实想法？",
            "error": str(exc),
        }

    return {
        "core_stance": str(result.get("core_stance", "")).strip(),
        "hidden_assumption": str(result.get("hidden_assumption", "")).strip(),
        "reflection": str(result.get("reflection", "")).strip(),
        "followup_question": str(result.get("followup_question", "")).strip(),
    }
