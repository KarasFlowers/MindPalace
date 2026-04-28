"""Echo Location — 回声定位 + 偏见预警。

对比用户在不同时间点的认知画像变化，生成深度反思报告。
"""

import logging
from dataclasses import dataclass

from src.llm.client import chat_json
from src.config import get_memory_config

logger = logging.getLogger(__name__)

ECHO_SYSTEM_PROMPT = """\
你是 MindPalace 的"回声定位仪"（Echo Location System）。

你的任务是对比同一用户在不同时间点对相似话题的认知画像，生成一份深度洞察报告。

你会收到：
- 用户本次的观点和认知标签
- 用户过去在相似话题下的观点和认知标签

请你分析：

1. **观点变迁 (stance_shift)**
   用户的立场是否发生了变化？朝什么方向变了？可能是什么原因？
   
2. **思维模式对比 (reasoning_shift)**
   用户的推理方式是否有变化？（例如从直觉判断变成了系统性分析）

3. **情感底色漂移 (tone_drift)**
   用户的情感基调发生了什么变化？（例如从悲观变成审慎乐观）

4. **偏见预警 (bias_alert)**
   用户是否表现出明显的思维定式？是否一直待在认知舒适区？
   如果是，给出打破舒适区的具体建议。
   如果不是，设为 null。

5. **成长洞察 (growth_insight)**
   一句话点评用户的认知进化轨迹（中文，不超过100字）。要犀利、有穿透力。

以 JSON 格式输出：
{
  "stance_shift": "观点变迁描述",
  "reasoning_shift": "推理模式变化描述",
  "tone_drift": "情感底色漂移描述",
  "bias_alert": "偏见预警（可为 null）",
  "growth_insight": "一句话成长洞察"
}
"""


@dataclass
class EchoReport:
    """回声定位报告。"""

    stance_shift: str
    reasoning_shift: str
    tone_drift: str
    bias_alert: str | None
    growth_insight: str
    has_history: bool  # 是否有历史记录可对比


def generate_echo_report(
    current_response: str,
    current_tags: dict,
    historical_memories: list[dict],
    provider_config: dict | None = None,
) -> EchoReport:
    """生成回声定位报告。

    Args:
        current_response: 用户本次的回应文本。
        current_tags: 本次的认知画像标签。
        historical_memories: 历史相关记忆列表。
        provider_config: 任务特定的 Provider 配置。

    Returns:
        EchoReport 对象。
    """
    if not historical_memories:
        return EchoReport(
            stance_shift="",
            reasoning_shift="",
            tone_drift="",
            bias_alert=None,
            growth_insight="这是你在此话题下的第一次发言，暂无历史参照。继续表达，系统会记住你的心智轨迹。",
            has_history=False,
        )

    # 构建历史上下文
    history_parts = []
    for i, mem in enumerate(historical_memories[:3], 1):  # 最多对比 3 条
        history_parts.append(
            f"--- 历史记录 #{i} ({mem.get('created_at', '?')}) ---\n"
            f"文章: {mem.get('article_title', '?')}\n"
            f"用户立场: {mem.get('stance_summary', '?')}\n"
            f"核心偏好: {mem.get('core_preference', [])}\n"
            f"推理模式: {mem.get('reasoning_style', '?')}\n"
            f"情感底色: {mem.get('emotional_tone', '?')}\n"
            f"原文摘要: {mem.get('user_response', '')[:200]}"
        )

    user_prompt = (
        f"=== 用户本次发言 ===\n"
        f"{current_response}\n\n"
        f"本次认知标签:\n"
        f"  核心偏好: {current_tags.get('core_preference', [])}\n"
        f"  推理模式: {current_tags.get('reasoning_style', '?')}\n"
        f"  情感底色: {current_tags.get('emotional_tone', '?')}\n"
        f"  立场概括: {current_tags.get('stance_summary', '?')}\n\n"
        f"=== 历史画像 ===\n"
        + "\n\n".join(history_parts)
    )

    logger.info("[Echo] Generating echo report with %d historical records...", len(historical_memories))
    cfg = provider_config or get_memory_config()
    result = chat_json(ECHO_SYSTEM_PROMPT, user_prompt, provider_config=cfg)

    return EchoReport(
        stance_shift=result.get("stance_shift", ""),
        reasoning_shift=result.get("reasoning_shift", ""),
        tone_drift=result.get("tone_drift", ""),
        bias_alert=result.get("bias_alert"),
        growth_insight=result.get("growth_insight", ""),
        has_history=True,
    )


def format_echo_report(report: EchoReport, colors: dict | None = None) -> str:
    """将 EchoReport 格式化为终端输出。"""
    c = colors or {}
    BOLD = c.get("BOLD", "")
    DIM = c.get("DIM", "")
    CYAN = c.get("CYAN", "")
    YELLOW = c.get("YELLOW", "")
    GREEN = c.get("GREEN", "")
    RED = c.get("RED", "")
    MAGENTA = c.get("MAGENTA", "")
    RESET = c.get("RESET", "")

    lines = []
    sep = f"  {DIM}{'─' * 54}{RESET}"

    lines.append(f"\n{BOLD}{MAGENTA}{'=' * 60}")
    lines.append(f"  [Echo Location] -- Cognitive Reflection")
    lines.append(f"{'=' * 60}{RESET}")

    if not report.has_history:
        lines.append(f"\n  {DIM}{report.growth_insight}{RESET}\n")
        return "\n".join(lines)

    if report.stance_shift:
        lines.append(f"\n  {BOLD}{CYAN}[Stance Shift]{RESET}")
        lines.append(f"  {report.stance_shift}")

    if report.reasoning_shift:
        lines.append(f"\n  {BOLD}{GREEN}[Reasoning Shift]{RESET}")
        lines.append(f"  {report.reasoning_shift}")

    if report.tone_drift:
        lines.append(f"\n  {BOLD}{YELLOW}[Tone Drift]{RESET}")
        lines.append(f"  {report.tone_drift}")

    if report.bias_alert:
        lines.append(f"\n  {BOLD}{RED}[!! Bias Alert !!]{RESET}")
        lines.append(f"  {RED}{report.bias_alert}{RESET}")

    lines.append(sep)
    lines.append(f"\n  {BOLD}{MAGENTA}* {report.growth_insight}{RESET}")
    lines.append("")

    return "\n".join(lines)
