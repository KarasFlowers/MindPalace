"""Cognitive Profiler — 从用户发言中提取心智模式标签。"""

import logging
from dataclasses import dataclass
from src.llm.client import chat_json
from src.config import get_memory_config

logger = logging.getLogger(__name__)

PROFILER_SYSTEM_PROMPT = """\
你是 MindPalace 的"认知画像分析师"（Cognitive Profiler）。

你的任务是分析用户对某篇文章的回应文本，从中提取出用户的**思维模式画像**。

请从以下三个维度进行分析：

1. **核心偏好 (core_preference)**
   用户在这段发言中展现的价值取向。从以下标签中选择最贴切的 1-2 个：
   - 实用主义 (pragmatist): 关注"有没有用"、"怎么落地"
   - 技术决定论 (techno-determinist): 相信技术进步能解决大部分问题
   - 人文关怀 (humanist): 优先考虑人的感受、伦理、公平
   - 怀疑论 (skeptic): 对新概念和权威说法持质疑态度
   - 理想主义 (idealist): 追求完美方案和宏大叙事
   - 经验主义 (empiricist): 靠数据和实证说话

2. **推理模式 (reasoning_style)**
   用户使用了什么样的逻辑方法。选择最贴切的 1 个：
   - 演绎推理 (deductive): 从原理到结论
   - 归纳猜想 (inductive): 从案例到规律
   - 类比联想 (analogical): 用其他领域的经验来推断
   - 直觉判断 (intuitive): 凭感觉下结论，缺少显式推理链
   - 系统性思维 (systems-thinking): 关注整体关系和反馈循环

3. **情感底色 (emotional_tone)**
   这段发言的整体情感基调。选择 1 个：
   - 乐观激进 (optimistic-aggressive): 积极拥抱变化
   - 审慎乐观 (cautiously-optimistic): 看好但有保留
   - 冷静客观 (neutral-analytical): 不带情感地分析
   - 审慎悲观 (cautiously-pessimistic): 担忧但愿意被说服
   - 悲观防守 (pessimistic-defensive): 强烈质疑或抵触

同时输出：
- **topic_keywords**: 从用户发言中提取的 3-5 个核心话题关键词（中文）
- **stance_summary**: 用一句话概括用户在这个话题上的立场（中文，不超过 60 字）

以 JSON 格式输出：
{
  "core_preference": ["标签1", "标签2"],
  "reasoning_style": "标签",
  "emotional_tone": "标签",
  "topic_keywords": ["关键词1", "关键词2", ...],
  "stance_summary": "一句话立场概括"
}
"""


@dataclass
class CognitiveProfile:
    """用户的认知画像。"""

    core_preference: list[str]
    reasoning_style: str
    emotional_tone: str
    topic_keywords: list[str]
    stance_summary: str


def profile_response(
    user_response: str,
    article_title: str = "",
    article_summary: str = "",
    provider_config: dict | None = None,
) -> CognitiveProfile:
    """分析用户回应，提取认知画像。

    Args:
        user_response: 用户的回应文本。
        article_title: 相关文章标题（提供上下文）。
        article_summary: 相关文章摘要（提供上下文）。
        provider_config: 任务特定的 Provider 配置。

    Returns:
        CognitiveProfile 对象。
    """
    context_parts = []
    if article_title:
        context_parts.append(f"讨论的文章标题: {article_title}")
    if article_summary:
        context_parts.append(f"文章摘要: {article_summary}")
    context_parts.append(f"\n用户的回应:\n{user_response}")

    user_prompt = "\n".join(context_parts)

    logger.info("[Profiler] Analyzing user response...")
    cfg = provider_config or get_memory_config()
    result = chat_json(PROFILER_SYSTEM_PROMPT, user_prompt, provider_config=cfg)

    profile = CognitiveProfile(
        core_preference=result.get("core_preference", []),
        reasoning_style=result.get("reasoning_style", ""),
        emotional_tone=result.get("emotional_tone", ""),
        topic_keywords=result.get("topic_keywords", []),
        stance_summary=result.get("stance_summary", ""),
    )

    logger.info(
        "[Profiler] Tags: pref=%s, reasoning=%s, tone=%s",
        profile.core_preference,
        profile.reasoning_style,
        profile.emotional_tone,
    )
    return profile
