"""日常思维训练营流 (Daily Session Flow)。一键串联 Scout -> Council -> Memory。"""

import logging
import sys
import time

from src.scout.pipeline import run_scout
from src.council.flow import run_council
from src.council.output import format_council_result
from src.memory.profiler import profile_response
from src.memory.store import save_memory, find_related_memories
from src.memory.echo import generate_echo_report, format_echo_report
from src.storage.db import _get_conn, save_debate
from src.config import get_scout_config, get_council_config, get_memory_config
from src.obs import span

logger = logging.getLogger(__name__)

# ANSI Colors
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

COLORS = {
    "BOLD": BOLD, "DIM": DIM, "CYAN": CYAN, "YELLOW": YELLOW,
    "GREEN": GREEN, "RED": RED, "MAGENTA": MAGENTA, "RESET": RESET,
}


def _get_article_id_by_url(url: str) -> int | None:
    """根据 URL 查找数据库中的文章 ID。"""
    with _get_conn() as conn:
        row = conn.execute("SELECT id FROM articles WHERE url = ?", (url,)).fetchone()
        if row:
            return row["id"]
    return None


def run_daily_session():
    """执行端到端每日训练营工作流。"""
    _session_span = span("daily.session")
    _session_span.__enter__()
    try:
        print(f"\n{BOLD}{MAGENTA}")
        print("  __  __ _           _ _____      _                 ")
        print(" |  \\/  (_)         | |  __ \\    | |                ")
        print(" | \\  / |_ _ __   __| | |__) |_ _| | __ _  ___ ___  ")
        print(" | |\\/| | | '_ \\ / _` |  ___/ _` | |/ _` |/ __/ _ \\ ")
        print(" | |  | | | | | | (_| | |  | (_| | | (_| | (_|  __/ ")
        print(" |_|  |_|_|_| |_|\\__,_|_|   \\__,_|_|\\__,_|\\___\\___| ")
        print(f"{RESET}")
        print(f"  {DIM}Initializing Daily Cognitive Session...{RESET}\n")
        time.sleep(1)

        print(f"  {BOLD}{CYAN}[1/3] SCOUTING{RESET} {DIM}Scouring the web for high-density signals...{RESET}")

        # 1. Scout 阶段
        scout_cfg = get_scout_config()
        scout_results = run_scout(top_k=5, provider_config=scout_cfg)
        if not scout_results:
            print(f"\n  {YELLOW}No new signals found today. The noise is too high. Rest for now.{RESET}\n")
            return

        top_article = scout_results[0]
        article_id = _get_article_id_by_url(top_article.url)

        print(f"  {GREEN}✓ Found top signal: {BOLD}{top_article.title}{RESET}")
        print(f"  {DIM}  Score: {top_article.total_score}/10 | SRC: {top_article.source}{RESET}\n")

        # 2. Council 阶段
        print(f"  {BOLD}{CYAN}[2/3] COUNCIL{RESET} {DIM}Assembling the MindPalace Council...{RESET}")
        time.sleep(1)

        # 抑制原版详细日志，提供沉浸感
        logging.getLogger("src.council.flow").setLevel(logging.WARNING)

        sys.stdout.write(f"  {DIM}The Critic is analyzing flaws...{RESET}\r")
        sys.stdout.flush()

        # 运行包含请求过程
        council_cfg = get_council_config()
        result = run_council(
            title=top_article.title,
            summary=top_article.summary,
            content=top_article.summary,  # 使用摘要作为正文，提高速度
            provider_config=council_cfg,
        )

        sys.stdout.write(" " * 50 + "\r")  # clear line
        print(format_council_result(result, colors=COLORS))
        logging.getLogger("src.council.flow").setLevel(logging.INFO)

        # 落库辩论
        debate_id = None
        try:
            debate_id = save_debate(result, article_id=article_id)
        except Exception:
            logger.exception("Failed to persist debate state")

        # 3. Memory 阶段 (用户交互)
        print(f"  {BOLD}{CYAN}[3/3] YOUR TURN{RESET} {DIM}Break the illusion. Form your own thesis.{RESET}")
        print(f"  {DIM}(Enter your thoughts. Press Enter twice to submit, or type 'skip' to skip){RESET}\n")

        lines = []
        while True:
            try:
                line = input(f"  {GREEN}>{RESET} ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.strip().lower() == "skip":
                print(f"\n  {DIM}Skipped. Sometimes silence is an answer.{RESET}\n")
                return
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)

        user_response = "\n".join(lines).strip()
        if not user_response:
            print(f"\n  {DIM}No response recorded.{RESET}\n")
            return

        # 认知剖析
        print(f"\n  {DIM}Extracting Cognitive Profile...{RESET}")
        logging.getLogger("src.memory.profiler").setLevel(logging.WARNING)
        memory_cfg = get_memory_config()
        profile = profile_response(
            user_response=user_response,
            article_title=top_article.title,
            article_summary=top_article.summary,
            provider_config=memory_cfg,
        )
        logging.getLogger("src.memory.profiler").setLevel(logging.INFO)

        # 保存并生成回声
        memory_id = save_memory(
            article_id=article_id,
            article_title=top_article.title,
            user_response=user_response,
            profile=profile,
            # daily_session 末尾批量触发演化（见下方 link_memories 调用），
            # 此处不开 link_after_save 以避免与 echo 报告并行竞争。
        )

        # 回声定位
        print(f"  {DIM}Running Echo Location against historical patterns...{RESET}")
        logging.getLogger("src.memory.echo").setLevel(logging.WARNING)

        current_tags = {
            "core_preference": profile.core_preference,
            "reasoning_style": profile.reasoning_style,
            "emotional_tone": profile.emotional_tone,
            "stance_summary": profile.stance_summary,
        }

        related = find_related_memories(user_response, exclude_id=memory_id)
        echo = generate_echo_report(user_response, current_tags, related, provider_config=memory_cfg)

        logging.getLogger("src.memory.echo").setLevel(logging.INFO)

        print(format_echo_report(echo, colors=COLORS))

        # 记忆演化（A-MEM）：为本次新增记忆与历史记忆建立链接
        try:
            from src.memory.evolution import link_memories

            result = link_memories(memory_id, provider_config=memory_cfg)
            if result.get("evolved"):
                print(
                    f"  {DIM}\U0001f9e0 记忆演化：建立 {result['links_created']} 条链接，"
                    f"更新 {result['neighbors_updated']} 个邻居{RESET}"
                )
        except Exception:
            logger.exception("Memory evolution failed")

        # 认知固化检查
        try:
            from src.memory.crystallize import crystallize_if_needed, render_crystal_terminal

            crystal = crystallize_if_needed(provider_config=memory_cfg)
            if crystal:
                print(f"\n  {BOLD}{MAGENTA}\u2728 认知洞察已结晶{RESET}")
                print("  " + render_crystal_terminal(crystal, colors=COLORS).replace("\n", "\n  "))
                print()
        except Exception:
            logger.exception("Crystallize failed")

        # 用户反馈收集
        if debate_id:
            try:
                from src.eval.feedback import collect_feedback_interactive

                collect_feedback_interactive(debate_id)
            except Exception:
                logger.debug("Feedback collection skipped")

        print(f"  {BOLD}{GREEN}Session Complete. Memory indexed. You have grown.{RESET}\n")

        return
    finally:
        _session_span.__exit__(None, None, None)
