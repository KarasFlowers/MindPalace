"""心智漫游单次会话流程。"""

from __future__ import annotations

import logging

from src.inquiry.analysis import analyze_response
from src.inquiry.library import choose_random_card
from src.inquiry.types import PromptCard
from src.memory.echo import format_echo_report, generate_echo_report
from src.memory.profiler import CognitiveProfile, profile_response
from src.memory.store import find_related_memories, save_memory
from src.config import get_memory_config
from src.ux import Spinner, collect_multiline

logger = logging.getLogger(__name__)

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

_KIND_LABELS = {
    "self": "认识自己",
    "philosophy": "哲思问题",
    "thought_experiment": "思想实验",
}


def format_card(card: PromptCard) -> str:
    """格式化问题卡展示文本。"""
    label = _KIND_LABELS.get(card.kind, card.kind)
    parts = [f"{BOLD}{CYAN}{label}: {card.title}{RESET}"]
    if card.context:
        parts.append(f"\n{DIM}设定:{RESET}\n{card.context}")
    parts.append(f"\n{BOLD}问题:{RESET}\n{card.prompt}")
    if card.tags:
        parts.append(f"\n{DIM}标签: {' '.join('#' + tag for tag in card.tags)}{RESET}")
    return "\n".join(parts)


def collect_multiline_response() -> str | None:
    """收集多行回答；连续两次空行提交，skip 跳过。

    复用统一的 src.ux.collect_multiline，保持与 council / daily 一致的交互规则。
    """
    hint = f"{DIM}请写下第一反应。连续两次空行提交，输入 skip 跳过。{RESET}\n"
    return collect_multiline(prompt=f"{GREEN}>{RESET} ", allow_skip=True, hint=hint)


def print_analysis(analysis: dict) -> None:
    """展示回答分析。"""
    print(f"\n{BOLD}{MAGENTA}[心智镜像]{RESET}")
    if analysis.get("core_stance"):
        print(f"  - 核心线索: {analysis['core_stance']}")
    if analysis.get("hidden_assumption"):
        print(f"  - 隐藏前提: {analysis['hidden_assumption']}")
    if analysis.get("reflection"):
        print(f"\n  {analysis['reflection']}")
    if analysis.get("followup_question"):
        print(f"\n  {YELLOW}继续追问: {analysis['followup_question']}{RESET}")


def _print_similarity_prompt(similar_info: dict) -> None:
    """当检测到与历史回答相似时，打印提示（Axiomind 基线 diff）。"""
    historical = similar_info["historical"]
    diff = similar_info["diff"]
    created = (historical.get("created_at") or "")[:10]
    prev_response = (historical.get("user_response") or "").replace("\n", " ").strip()[:80]
    what_changed = diff.get("what_changed", "")

    print(f"\n{YELLOW}\U0001f501 你在 {created} 也回答过类似问题{RESET}")
    if prev_response:
        print(f"{DIM}  当时你说：{prev_response}{RESET}")
    if what_changed:
        print(f"{YELLOW}  变化：{what_changed}{RESET}")
    print()


def save_inquiry_memory(card: PromptCard, user_response: str, provider_config: dict | None = None) -> int:
    """将一次心智漫游回答写入长期记忆。"""
    cfg = provider_config or get_memory_config()
    try:
        profile = profile_response(
            user_response=user_response,
            article_title=card.title,
            article_summary=card.prompt,
            provider_config=cfg,
        )
    except Exception as exc:
        logger.warning("Inquiry profiling failed; saving raw response only: %s", exc)
        profile = CognitiveProfile(
            core_preference=[],
            reasoning_style="",
            emotional_tone="",
            topic_keywords=card.tags,
            stance_summary=user_response[:60],
        )

    # 基线 diff：检查是否有相似的历史回答（Axiomind 暂存区模式）
    similar_info = None
    try:
        from src.inquiry.diff import check_and_describe_similarity
        similar_info = check_and_describe_similarity(card.id, user_response, provider_config=cfg)
    except Exception as exc:
        logger.warning("Inquiry diff check failed: %s", exc)

    if similar_info and similar_info["diff"]["is_similar"]:
        _print_similarity_prompt(similar_info)

    memory_id = save_memory(
        article_id=None,
        article_title=f"[心智漫游] {card.title}",
        user_response=user_response,
        profile=profile,
        source_type=card.kind,
        source_id=card.id,
        link_after_save=True,
        provider_config=cfg,
    )
    try:
        related = find_related_memories(user_response, exclude_id=memory_id)
        current_tags = {
            "core_preference": profile.core_preference,
            "reasoning_style": profile.reasoning_style,
            "emotional_tone": profile.emotional_tone,
            "stance_summary": profile.stance_summary,
        }
        echo = generate_echo_report(user_response, current_tags, related, provider_config=cfg)
        print(format_echo_report(echo))
    except Exception as exc:
        logger.warning("Inquiry echo report failed after save: %s", exc)
    return memory_id


def run_inquiry_session(kind: str | None = None, card: PromptCard | None = None) -> int | None:
    """运行一次心智漫游会话。"""
    selected = card or choose_random_card(kind)
    print("\n" + format_card(selected) + "\n")
    user_response = collect_multiline_response()
    if not user_response:
        print(f"\n{DIM}已跳过。{RESET}\n")
        return None

    with Spinner(
        "正在提炼你的回答...",
        style="pulse",
        success_text="回答提炼完成",
        failure_text="回答提炼失败",
    ):
        analysis = analyze_response(selected, user_response)
    print_analysis(analysis)

    try:
        choice = input(f"\n{DIM}保存这次回答到长期记忆？[Y/n]: {RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if choice in ("", "y", "yes"):
        memory_id = save_inquiry_memory(selected, user_response)
        print(f"\n{GREEN}✓ 已保存为记忆 #{memory_id}{RESET}\n")
        return memory_id

    print(f"\n{DIM}未保存。{RESET}\n")
    return None
