"""Prompt 迭代器 — 基于评估报告和用户反馈生成 prompt 改进建议。"""

from __future__ import annotations

import logging

from src.config import get_judge_config
from src.llm.client import chat
from src.eval.judge_debates import judge_recent_debates, generate_weekly_report
from src.eval.feedback import get_feedback_stats

logger = logging.getLogger(__name__)

ITERATE_PROMPT = """\
你是 MindPalace 的 Prompt 改进顾问。

输入包含：
1. 周度评估报告（各角色的打分和弱点）
2. 用户反馈统计（👍/👎/📌 分布）

请生成 2-3 条**具体可操作**的 prompt 改进建议。每条建议指明：
- 针对哪个角色（Critic / Synthesizer / Mentor / Judge）
- 具体修改什么（增加/删除/修改哪部分 prompt 指令）
- 预期效果

用中文输出，格式：
1. **[角色]**: [建议内容]
2. ...
"""


def generate_iteration_suggestions(days: int = 7) -> str:
    """生成基于评估和反馈的 prompt 改进建议。"""
    reports = judge_recent_debates(days=days)
    weekly = generate_weekly_report(reports, days=days)
    feedback = get_feedback_stats(days=days)

    feedback_summary = (
        f"👍有启发: {feedback.get('up', 0)} | "
        f"👎无意义: {feedback.get('down', 0)} | "
        f"📌采纳: {feedback.get('adopted', 0)}"
    )

    user_prompt = f"周度评估报告:\n{weekly}\n\n用户反馈统计: {feedback_summary}"

    try:
        return chat(
            system_prompt=ITERATE_PROMPT,
            user_prompt=user_prompt,
            provider_config=get_judge_config(),
        )
    except Exception as exc:
        logger.warning("Prompt iteration failed: %s", exc)
        return f"[Prompt 迭代失败: {exc}]"
