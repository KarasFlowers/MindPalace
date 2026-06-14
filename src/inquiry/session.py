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
    """收集多行回答；Enter 两次提交，skip 跳过。"""
    print(f"{DIM}请写下第一反应。按 Enter 两次提交，输入 skip 跳过。{RESET}\n")
    lines: list[str] = []
    while True:
        try:
            line = input(f"{GREEN}>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if line.strip().lower() == "skip":
            return None
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    response = "\n".join(lines).strip()
    return response or None


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
    memory_id = save_memory(
        article_id=None,
        article_title=f"[心智漫游] {card.title}",
        user_response=user_response,
        profile=profile,
        source_type=card.kind,
        source_id=card.id,
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

    print(f"\n{DIM}正在提炼你的回答...{RESET}")
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
