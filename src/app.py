"""MindPalace CLI 入口。"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path

# Windows 终端默认 GBK 编码，强制 UTF-8 避免 emoji/中文输出报错
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import questionary
    from questionary import Style
except ImportError:
    class _MissingQuestionaryPrompt:
        def ask(self):
            raise RuntimeError(
                "Interactive mode requires 'questionary'. Run `pip install -e .` first."
            )

    class _QuestionaryFallback:
        @staticmethod
        def select(*args, **kwargs):
            return _MissingQuestionaryPrompt()

        @staticmethod
        def text(*args, **kwargs):
            return _MissingQuestionaryPrompt()

        @staticmethod
        def confirm(*args, **kwargs):
            return _MissingQuestionaryPrompt()

        class Separator(str):
            def __new__(cls, text: str = "----------------"):
                return str.__new__(cls, text)

    def Style(style_config):
        return style_config

    questionary = _QuestionaryFallback()

from src.scout.pipeline import run_scout
from src.council.flow import run_council
from src.council.output import format_council_result
from src.storage.db import (
    add_article_tags,
    cleanup_old_articles,
    get_article,
    list_articles,
    list_recent_debates_for_article,
    remove_article_tags,
    replace_article_tags,
    save_debate,
    set_article_favorite,
    set_article_note,
)
from src.eval.feedback import collect_feedback_interactive
from src.memory.profiler import profile_response
from src.memory.store import save_memory, find_related_memories, get_all_memories
from src.memory.echo import generate_echo_report, format_echo_report
from src.workflows.daily_session import run_daily_session
from src.inquiry.cli import run_inquiry_menu
from src.config import (
    ARTICLE_RETENTION_DAYS,
    FEED_PRESETS,
    PROJECT_ROOT,
    get_default_feeds,
    get_scout_config,
    get_council_config,
    get_memory_config,
)
from dotenv import set_key, get_key
import shutil

# ANSI 颜色
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

PROVIDER_PREFIXES = {
    "global": "OPENAI",
    "scout": "SCOUT",
    "council": "COUNCIL",
    "memory": "MEMORY",
    "fast": "FAST",
    "router": "ROUTER",
    "judge": "JUDGE",
    "embedding": "EMBEDDING",
}

PROVIDER_LABELS = {
    "OPENAI": "Global Default",
    "SCOUT": "Scout",
    "COUNCIL": "Council",
    "MEMORY": "Memory",
    "FAST": "Fast",
    "ROUTER": "Router",
    "JUDGE": "Judge",
    "EMBEDDING": "Embedding",
}

PROVIDER_CLI_NAMES = {prefix: cli_name for cli_name, prefix in PROVIDER_PREFIXES.items()}

# Questionary 自定义样式
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),
    ('question', 'bold'),
    ('answer', 'fg:#f44336 bold'),
    ('pointer', 'fg:#673ab7 bold'),
    ('highlighted', 'fg:#673ab7 bold'),
    ('selected', 'fg:#cc5454'),
    ('separator', 'fg:#cc5454'),
    ('instruction', ''),
    ('text', ''),
    ('disabled', 'fg:#858585 italic')
])


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=f"{DIM}%(asctime)s{RESET} %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _format_score_bar(score: float, max_score: int = 10) -> str:
    """生成一个迷你分数条。进度条按整数，数字保留小数。"""
    filled = int(score * 10 / max_score)
    bar = "█" * filled + "░" * (10 - filled)
    color = GREEN if score >= 7 else YELLOW if score >= 4 else RED
    return f"{color}{bar} {score:.1f}/{max_score}{RESET}"


def _format_score_bar_plain(score: float, max_score: int = 10) -> str:
    """生成一个迷你分数条（纯文本，无颜色）。进度条按整数，数字保留小数。"""
    filled = int(score * 10 / max_score)
    bar = "█" * filled + "░" * (10 - filled)
    return f"{bar} {score:.1f}/{max_score}"


def _parse_tag_text(text: str | None) -> list[str]:
    if not text:
        return []

    tags = []
    seen = set()
    for candidate in text.replace("，", ",").split(","):
        tag = candidate.strip()
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(tag)
    return tags


def _format_tags(tags: list[str] | None) -> str:
    if not tags:
        return "无"
    return " ".join(f"#{tag}" for tag in tags)


def _article_filters_from_args(args) -> dict:
    return {
        "query": getattr(args, "query", None),
        "tags": getattr(args, "tag", None) or [],
        "source": getattr(args, "source", None),
        "days": getattr(args, "days", None),
    }


def _empty_article_filters() -> dict:
    return {"query": "", "tags": [], "source": "", "days": None}


def _normalize_article_filters(filters: dict | None) -> dict:
    normalized = _empty_article_filters()
    if not filters:
        return normalized
    normalized["query"] = (filters.get("query") or "").strip()
    normalized["tags"] = list(filters.get("tags") or [])
    normalized["source"] = (filters.get("source") or "").strip()
    normalized["days"] = filters.get("days")
    return normalized


def _has_active_filters(filters: dict | None) -> bool:
    current = _normalize_article_filters(filters)
    return any(
        [
            current.get("query"),
            current.get("tags"),
            current.get("source"),
            current.get("days") is not None,
        ]
    )


def _format_filter_summary(filters: dict | None, favorites_only: bool = False) -> str:
    current = _normalize_article_filters(filters)
    parts = ["档案库" if favorites_only else "全部文章"]
    if current["query"]:
        parts.append(f"关键词={current['query']}")
    if current["tags"]:
        parts.append(f"标签={', '.join(current['tags'])}")
    if current["source"]:
        parts.append(f"来源={current['source']}")
    if current["days"] is not None:
        parts.append(f"近 {current['days']} 天")
    return " | ".join(parts)


def _load_articles_for_filters(limit: int, favorites_only: bool, filters: dict | None) -> list[dict]:
    current = _normalize_article_filters(filters)
    return list_articles(
        limit=limit,
        favorites_only=favorites_only,
        query=current["query"] or None,
        tags=current["tags"],
        source=current["source"] or None,
        days=current["days"],
    )


def _print_article_card(article: dict, show_note: bool = True):
    badges = []
    if article.get("is_favorite"):
        badges.append(f"{YELLOW}[收藏]{RESET}")
    if article.get("tags"):
        badges.append(f"{GREEN}[已标记]{RESET}")
    badge_text = f" {' '.join(badges)}" if badges else ""

    print(f"  {BOLD}{CYAN}ID:{article['id']}{RESET}{badge_text} {BOLD}{article['title']}{RESET}")
    print(
        f"     {DIM}[SRC] {article['source']}  |  "
        f"Score: {article['total_score']:.1f}/10  |  "
        f"Created: {article['created_at'][:10]}{RESET}"
    )
    if article.get("tags"):
        print(f"     {GREEN}Tags: {_format_tags(article.get('tags'))}{RESET}")
    if show_note and article.get("favorite_note"):
        print(f"     {YELLOW}Note: {article['favorite_note']}{RESET}")
    print(f"     {MAGENTA}> {article.get('summary', '')}{RESET}")
    print(f"     {DIM}{'─' * 50}{RESET}\n")


def _print_article_collection(
    title: str,
    articles: list[dict],
    filters: dict | None,
    favorites_only: bool = False,
):
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  [MindPalace] -- {title}")
    print(f"{'=' * 60}{RESET}")
    print(f"  {DIM}{_format_filter_summary(filters, favorites_only=favorites_only)}{RESET}\n")
    for article in articles:
        _print_article_card(article)


def _print_recent_debate_summary(debate: dict):
    consensus = debate.get("consensus") or {}
    headline = consensus.get("headline") or debate.get("article_title") or "未命名讨论"
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  [Recent Discussion] #{debate['id']}")
    print(f"{'=' * 60}{RESET}")
    print(f"  {BOLD}{headline}{RESET}")
    print(
        f"  {DIM}时间: {debate['created_at'][:16]} | 难度: {debate.get('difficulty', '?')} | "
        f"结束方式: {debate.get('terminated_by', '?')}{RESET}"
    )

    key_points = consensus.get("key_points") or []
    if key_points:
        print(f"\n  {BOLD}{MAGENTA}[核心洞察]{RESET}")
        for point in key_points[:4]:
            print(f"  - {point}")

    tensions = consensus.get("remaining_tensions") or []
    if tensions:
        print(f"\n  {BOLD}{MAGENTA}[未解决张力]{RESET}")
        for tension in tensions[:3]:
            print(f"  - {tension}")

    stance = consensus.get("recommended_stance")
    if stance:
        print(f"\n  {BOLD}{MAGENTA}[推荐立场]{RESET}")
        print(f"  {stance}")

    if not consensus and debate.get("turns"):
        print(f"\n  {BOLD}{MAGENTA}[讨论片段]{RESET}")
        for turn in debate["turns"][-3:]:
            role = turn.get("role_key", "role")
            raw = turn.get("content")
            if isinstance(raw, dict):
                content = json.dumps(raw, ensure_ascii=False)
            else:
                content = str(raw or "")
            print(f"  - {role}: {content[:120]}")

    print(f"\n  {DIM}{'─' * 56}{RESET}")


def _show_article_overview(article: dict):
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {article['title']}")
    print(f"{'=' * 60}{RESET}")
    print(
        f"  {DIM}来源: {article['source']} | 评分: {article['total_score']:.1f}/10 | "
        f"入库: {article['created_at'][:10]}{RESET}"
    )
    print(f"  {DIM}链接: {article['url']}{RESET}")
    if article.get("tags"):
        print(f"  {GREEN}标签: {_format_tags(article['tags'])}{RESET}")
    if article.get("favorite_note"):
        print(f"  {YELLOW}备注: {article['favorite_note']}{RESET}")
    recent_discussions = list_recent_debates_for_article(article["id"], limit=3)
    if recent_discussions:
        latest = recent_discussions[0]
        headline = (latest.get("consensus") or {}).get("headline") or "有历史讨论可回看"
        print(f"  {DIM}最近讨论: #{latest['id']} {headline}{RESET}")
    print(f"\n  {MAGENTA}摘要: {article.get('summary', '')}{RESET}\n")
    return recent_discussions


def _truncate_text(text: str | None, limit: int = 90) -> str:
    content = (text or "").strip().replace("\n", " ")
    if len(content) <= limit:
        return content
    return content[: limit - 3].rstrip() + "..."


def _build_council_snapshot(result) -> dict:
    """从 Council 结果提取一屏能读完的对话抓手。"""
    consensus = getattr(result, "consensus", None) or {}
    critic = getattr(result, "critic", None) or {}
    synth = getattr(result, "synthesizer", None) or {}
    mentor = getattr(result, "mentor", None) or {}

    vulnerabilities = critic.get("vulnerabilities") or []
    first_vulnerability = vulnerabilities[0] if vulnerabilities else {}
    connections = synth.get("connections") or []
    first_connection = connections[0] if connections else {}
    questions = mentor.get("questions") or []
    first_question = questions[0] if questions else {}
    key_points = consensus.get("key_points") or []

    summary = consensus.get("headline") or (key_points[0] if key_points else "")
    if not summary:
        summary = critic.get("verdict") or synth.get("synthesis") or mentor.get("provocation") or result.article_title

    challenge = ""
    if first_vulnerability:
        challenge = first_vulnerability.get("assumption") or first_vulnerability.get("counter") or ""
    if not challenge:
        challenge = critic.get("verdict") or "这次讨论里没有形成强烈反驳。"

    bridge = ""
    if first_connection:
        bridge = first_connection.get("insight") or first_connection.get("analogy") or ""
    if not bridge:
        bridge = synth.get("synthesis") or "这次讨论更偏向直接思辨，没有明显跨界补充。"

    question = ""
    if first_question:
        question = first_question.get("question") or ""
    if not question:
        question = mentor.get("provocation") or "你最想继续追问哪一点？"

    stance = consensus.get("recommended_stance") or ""
    return {
        "summary": _truncate_text(summary, 120),
        "challenge": _truncate_text(challenge, 120),
        "bridge": _truncate_text(bridge, 120),
        "question": _truncate_text(question, 120),
        "stance": _truncate_text(stance, 120),
    }


def _print_council_snapshot(result):
    """先给用户一屏内的摘要，再决定要不要展开。"""
    snapshot = _build_council_snapshot(result)
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print("  [Council Snapshot] -- 先抓最重要的四点")
    print(f"{'=' * 60}{RESET}")
    print(f"  {BOLD}一句话结论{RESET} {snapshot['summary']}")
    print(f"  {RED}最大争议{RESET} {snapshot['challenge']}")
    print(f"  {GREEN}新视角{RESET} {snapshot['bridge']}")
    print(f"  {YELLOW}最值得回应的问题{RESET} {snapshot['question']}")
    if snapshot["stance"]:
        print(f"  {MAGENTA}当前建议立场{RESET} {snapshot['stance']}")
    print()


def _print_council_role_detail(role_key: str, payload: dict):
    """按角色把结构化内容展开成人能快速扫读的形式。"""
    if role_key == "critic":
        print(f"\n{BOLD}{RED}[The Critic] -- 展开批判视角{RESET}")
        vulnerabilities = payload.get("vulnerabilities") or []
        for item in vulnerabilities[:3]:
            severity = str(item.get("severity", "?")).upper()
            print(f"  - [{severity}] {item.get('assumption', '')}")
            counter = item.get("counter")
            if counter:
                print(f"    {DIM}崩塌条件: {counter}{RESET}")
        if payload.get("missing_counterexample"):
            print(f"  {MAGENTA}反例: {payload['missing_counterexample']}{RESET}")
        if payload.get("verdict"):
            print(f"  {DIM}一句话判断: {payload['verdict']}{RESET}")
    elif role_key == "synthesizer":
        print(f"\n{BOLD}{GREEN}[The Synthesizer] -- 展开连接视角{RESET}")
        connections = payload.get("connections") or []
        for item in connections[:3]:
            print(f"  - [{item.get('domain', '?')}] {item.get('analogy', '')}")
            insight = item.get("insight")
            if insight:
                print(f"    {DIM}启发: {insight}{RESET}")
        if payload.get("synthesis"):
            print(f"  {DIM}综合洞察: {payload['synthesis']}{RESET}")
    elif role_key == "mentor":
        print(f"\n{BOLD}{YELLOW}[The Mentor] -- 展开追问视角{RESET}")
        questions = payload.get("questions") or []
        for item in questions[:3]:
            print(f"  - [{item.get('level', '追问')}] {item.get('question', '')}")
        if payload.get("provocation"):
            print(f"  {MAGENTA}刺激点: {payload['provocation']}{RESET}")
    print(f"\n  {DIM}{'─' * 56}{RESET}\n")


def _offer_council_deep_dive(result):
    """用户按需展开某个视角，而不是默认吃下全部长文。"""
    critic = getattr(result, "critic", None) or {}
    synth = getattr(result, "synthesizer", None) or {}
    mentor = getattr(result, "mentor", None) or {}

    while True:
        print(
            f"{DIM}想继续展开哪一部分？"
            f" [1] 批判视角 [2] 连接视角 [3] 追问视角 [4] 完整讨论 [Enter] 继续回应{RESET}"
        )
        try:
            choice = input(f"  {GREEN}>{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if choice in ("", "c", "continue"):
            return
        if choice == "1":
            _print_council_role_detail("critic", critic)
        elif choice == "2":
            _print_council_role_detail("synthesizer", synth)
        elif choice == "3":
            _print_council_role_detail("mentor", mentor)
        elif choice == "4":
            print(format_council_result(result, colors=COLORS))
        else:
            print(f"{YELLOW}请输入 1/2/3/4，或直接回车继续。{RESET}")


def _collect_guided_user_response() -> str | None:
    """用回应模板降低用户开口成本。"""
    starters = {
        "1": "我同意这里最有说服力的一点，因为",
        "2": "我不同意这里的一个关键前提，因为",
        "3": "我现在最困惑的问题是",
        "4": "这让我想到另一个案例：",
        "5": "请继续追问我这个点：",
    }

    print(f"  {BOLD}{CYAN}{'=' * 60}")
    print("  [Your Turn] -- 选一个起手句，更容易开口")
    print(f"  {'=' * 60}{RESET}")
    print("  [1] 我同意，因为...")
    print("  [2] 我不同意，因为...")
    print("  [3] 我最困惑的是...")
    print("  [4] 这让我想到...")
    print("  [5] 请继续追问我...")
    print("  [Enter] 自由输入  [skip] 跳过")
    print(f"  {DIM}输入内容后，连续两次回车提交。{RESET}\n")

    try:
        choice = input(f"  {GREEN}>{RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if choice == "skip":
        return None

    starter = starters.get(choice)
    lines: list[str] = []
    blank_count = 0
    first_input = True

    while True:
        try:
            line = input(f"  {GREEN}>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if first_input and line.strip().lower() == "skip":
            return None

        if first_input and starter:
            lines.append(f"{starter}{line.strip()}" if line.strip() else starter)
            first_input = False
            continue

        first_input = False
        if line == "":
            blank_count += 1
            if blank_count >= 2:
                break
            continue

        blank_count = 0
        lines.append(line)

    user_response = "\n".join(lines).strip()
    return user_response or None


def _process_council_reflection(article: dict, user_response: str):
    """把用户回应接到记忆与 echo 流程里。"""
    print(f"\n  {DIM}Analyzing your cognitive pattern...{RESET}")
    mem_cfg = get_memory_config()
    profile = profile_response(
        user_response=user_response,
        article_title=article["title"],
        article_summary=article.get("summary", ""),
        provider_config=mem_cfg,
    )

    print(f"\n  {BOLD}{MAGENTA}[Cognitive Profile]{RESET}")
    print(f"    Preference:  {', '.join(profile.core_preference)}")
    print(f"    Reasoning:   {profile.reasoning_style}")
    print(f"    Tone:        {profile.emotional_tone}")
    print(f"    Stance:      {profile.stance_summary}")
    print(f"    Keywords:    {', '.join(profile.topic_keywords)}")

    memory_id = save_memory(
        article_id=article.get("id"),
        article_title=article["title"],
        user_response=user_response,
        profile=profile,
    )
    print(f"  {DIM}Memory saved (#{memory_id}){RESET}")

    related = find_related_memories(user_response, exclude_id=memory_id)
    current_tags = {
        "core_preference": profile.core_preference,
        "reasoning_style": profile.reasoning_style,
        "emotional_tone": profile.emotional_tone,
        "stance_summary": profile.stance_summary,
    }
    echo = generate_echo_report(user_response, current_tags, related, provider_config=mem_cfg)
    print(format_echo_report(echo, colors=COLORS))


def _run_council_experience(article: dict, pause_at_end: bool = False, paradigm: str = "debate"):
    """统一 Council 的展示、回应和反馈体验。"""
    print(f"\n{BOLD}Starting Council discussion for: {article['title']}{RESET}\n")

    cfg = get_council_config()
    result = run_council(
        title=article["title"],
        summary=article.get("summary", ""),
        content=article.get("summary", ""),
        provider_config=cfg,
        paradigm=paradigm,
    )

    debate_id = None
    try:
        debate_id = save_debate(result, article_id=article.get("id"))
    except Exception:
        logging.getLogger(__name__).exception("Failed to persist debate state")

    _print_council_snapshot(result)
    _offer_council_deep_dive(result)

    user_response = _collect_guided_user_response()
    if user_response:
        _process_council_reflection(article, user_response)
    else:
        print(f"\n  {DIM}No response recorded this time.{RESET}\n")

    if debate_id is not None:
        try:
            collect_feedback_interactive(debate_id)
        except Exception:
            logging.getLogger(__name__).exception("Failed to collect council feedback")

    if pause_at_end:
        input(f"\n{DIM}按 Enter 继续...{RESET}")


def _display_results(articles):
    """以富文本格式展示评分结果。"""
    if not articles:
        print(f"\n{YELLOW}未找到任何文章。请检查 RSS 源或网络连接。{RESET}")
        return

    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  [MindPalace Scout] -- Top {len(articles)} Picks")
    print(f"{'=' * 60}{RESET}\n")

    for i, a in enumerate(articles, 1):
        scores = a.scores
        print(f"  {BOLD}{CYAN}#{i}{RESET} {BOLD}{a.title}{RESET}")
        print(f"     {DIM}[SRC] {a.source}  |  [URL] {a.url}{RESET}")
        print(f"     {MAGENTA}> {a.summary}{RESET}")
        print()
        print(f"     信息密度  {_format_score_bar(scores.get('information_density', 0))}")
        print(f"     原理深度  {_format_score_bar(scores.get('principle_depth', 0))}")
        print(f"     因果链    {_format_score_bar(scores.get('causal_chain', 0))}")
        print(f"     {BOLD}综合评分  {GREEN}{a.total_score:.1f}/10{RESET}")
        print()
        print(f"     {DIM}* {a.reasoning}{RESET}")
        print(f"     {DIM}{'─' * 50}{RESET}\n")


def cmd_scout(args):
    """执行 Scout 流水线。"""
    if args.feeds:
        feeds = args.feeds
    elif args.preset:
        feeds = get_default_feeds(args.preset)
    else:
        feeds = None
    cfg = get_scout_config()
    results = run_scout(feed_urls=feeds, top_k=args.top, provider_config=cfg)
    _display_results(results)


def cmd_list(args):
    """列出数据库中的文章。"""
    filters = _article_filters_from_args(args)
    if filters["days"] is not None and filters["days"] < 1:
        print(f"\n{RED}--days 必须大于等于 1。{RESET}\n")
        return
    articles = _load_articles_for_filters(limit=args.limit, favorites_only=False, filters=filters)
    if not articles:
        if _has_active_filters(filters):
            print(f"\n{YELLOW}当前筛选下没有文章：{_format_filter_summary(filters)}{RESET}\n")
        else:
            print(f"\n{YELLOW}数据库中暂无文章。请先运行 scout 命令。{RESET}\n")
        return

    _print_article_collection("Saved Articles", articles, filters)


def cmd_favorite(args):
    """收藏文章。"""
    article = get_article(args.item)
    if not article:
        print(f"\n{RED}Article ID {args.item} not found. Use 'list' to see available articles.{RESET}")
        return

    set_article_favorite(args.item, favorite=True, note=args.note)
    if args.tags:
        add_article_tags(args.item, args.tags)
    article = get_article(args.item)
    print(f"\n{GREEN}已收藏：{article['title']}{RESET}")
    if article and article.get("tags"):
        print(f"{GREEN}标签：{_format_tags(article['tags'])}{RESET}")
    if article and article.get("favorite_note"):
        print(f"{YELLOW}备注：{article['favorite_note']}{RESET}")
    print()


def cmd_unfavorite(args):
    """取消收藏文章。"""
    article = get_article(args.item)
    if not article:
        print(f"\n{RED}Article ID {args.item} not found. Use 'favorites' to see saved articles.{RESET}")
        return

    set_article_favorite(args.item, favorite=False)
    print(f"\n{YELLOW}已取消收藏：{article['title']}{RESET}\n")


def cmd_favorites(args):
    """列出收藏夹文章。"""
    filters = _article_filters_from_args(args)
    if filters["days"] is not None and filters["days"] < 1:
        print(f"\n{RED}--days 必须大于等于 1。{RESET}\n")
        return
    articles = _load_articles_for_filters(limit=args.limit, favorites_only=True, filters=filters)
    if not articles:
        if _has_active_filters(filters):
            print(f"\n{YELLOW}当前筛选下没有档案文章：{_format_filter_summary(filters, favorites_only=True)}{RESET}\n")
        else:
            print(f"\n{YELLOW}收藏夹还没有文章。看到值得反复讨论的材料，可以用 favorite 收藏。{RESET}\n")
        return

    _print_article_collection("Personal Archive", articles, filters, favorites_only=True)


def cmd_note(args):
    """编辑文章备注。"""
    article = get_article(args.item)
    if not article:
        print(f"\n{RED}Article ID {args.item} not found. Use 'list' to see available articles.{RESET}")
        return

    note = None if args.clear else args.text
    set_article_note(args.item, note)
    article = get_article(args.item)
    if args.clear:
        print(f"\n{YELLOW}已清空备注：{article['title']}{RESET}\n")
    else:
        print(f"\n{GREEN}已更新备注：{article['title']}{RESET}")
        print(f"{YELLOW}{article.get('favorite_note', '')}{RESET}\n")


def cmd_tag(args):
    """编辑文章标签。"""
    article = get_article(args.item)
    if not article:
        print(f"\n{RED}Article ID {args.item} not found. Use 'list' to see available articles.{RESET}")
        return

    if args.clear:
        replace_article_tags(args.item, [])
    elif args.set is not None:
        replace_article_tags(args.item, args.set)
    elif args.add is not None:
        add_article_tags(args.item, args.add)
    elif args.remove is not None:
        remove_article_tags(args.item, args.remove)

    updated = get_article(args.item)
    print(f"\n{GREEN}已更新标签：{updated['title']}{RESET}")
    print(f"{GREEN}当前标签：{_format_tags(updated.get('tags'))}{RESET}\n")


def cmd_cleanup(args):
    """清理旧文章。"""
    if args.days < 1:
        print(f"\n{RED}保留天数必须大于等于 1。{RESET}\n")
        return

    result = cleanup_old_articles(
        retention_days=args.days,
        dry_run=args.dry_run,
        keep_discussed=not args.include_discussed,
        keep_tagged=not args.include_tagged,
    )

    action = "可清理" if args.dry_run else "已清理"
    count = result["candidate_count"] if args.dry_run else result["deleted_count"]
    print(f"\n{BOLD}{CYAN}[Article Cleanup]{RESET}")
    print(f"  保留天数: {args.days}")
    print(f"  收藏文章: 永远保留")
    print(f"  已打标签文章: {'也会清理' if args.include_tagged else '默认保留'}")
    print(f"  已讨论/已记忆文章: {'也会清理' if args.include_discussed else '默认保留'}")
    print(f"  {action}: {count} 篇\n")

    candidates = result.get("candidates", [])[:10]
    if candidates:
        print(f"  {DIM}候选预览:{RESET}")
        for item in candidates:
            print(f"  - ID:{item['id']} {item['title'][:60]}")
        if len(result.get("candidates", [])) > 10:
            print(f"  {DIM}... 还有 {len(result['candidates']) - 10} 篇{RESET}")
        print()


def cmd_view(args):
    """查看文章完整内容。"""
    article = get_article(args.item)
    if not article:
        print(f"\n{RED}Article ID {args.item} not found. Use 'list' to see available articles.{RESET}")
        return

    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  [Article View]")
    print(f"{'=' * 60}{RESET}\n")
    
    print(f"  {BOLD}{article['title']}{RESET}")
    print(f"  {DIM}来源: {article['source']}{RESET}")
    print(f"  {DIM}链接: {article['url']}{RESET}")
    print(f"  {DIM}评分: {article['total_score']:.1f}/10{RESET}")
    if article.get("tags"):
        print(f"  {GREEN}标签: {_format_tags(article['tags'])}{RESET}")
    if article.get("favorite_note"):
        print(f"  {YELLOW}备注: {article['favorite_note']}{RESET}")

    recent_debates = list_recent_debates_for_article(article["id"], limit=2)
    if recent_debates:
        latest = recent_debates[0]
        headline = (latest.get("consensus") or {}).get("headline") or "可回看历史讨论"
        print(f"  {DIM}最近讨论: #{latest['id']} {headline}{RESET}")

    print(f"\n  {BOLD}{MAGENTA}[摘要]{RESET}")
    print(f"  {article.get('summary', '')}")

    if recent_debates:
        print(f"\n  {BOLD}{MAGENTA}[最近讨论摘要]{RESET}")
        consensus = recent_debates[0].get("consensus") or {}
        key_points = consensus.get("key_points") or []
        for point in key_points[:3]:
            print(f"  - {point}")

    print(f"\n  {BOLD}{MAGENTA}[正文]{RESET}")
    
    # 从数据库读取的是 clean_content（已清洗的纯文本）
    # 但数据库中存储的字段名可能不同，需要检查
    content = article.get('clean_content', '')
    if not content:
        # 如果没有 clean_content，尝试从 summary 获取
        content = article.get('summary', '正文内容未保存')
    
    # 分段显示，提高可读性
    paragraphs = content.split('\n')
    for para in paragraphs:
        if para.strip():
            print(f"  {para.strip()}")
            print()
    
    print(f"  {DIM}{'─' * 56}{RESET}\n")


def cmd_brief(args):
    """生成文章导读精炼版。"""
    article = get_article(args.item)
    if not article:
        print(f"\n{RED}Article ID {args.item} not found. Use 'list' to see available articles.{RESET}")
        return

    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  [Generating Brief...]")
    print(f"{'=' * 60}{RESET}\n")
    
    print(f"  {BOLD}{article['title']}{RESET}")
    print(f"  {DIM}来源: {article['source']} | 评分: {article['total_score']:.1f}/10{RESET}\n")
    
    # 调用 LLM 生成导读
    from src.llm.client import chat
    
    brief_prompt = f"""请为以下文章生成一份导读精炼版，帮助读者快速理解核心内容。

文章标题: {article['title']}
文章摘要: {article.get('summary', '')}

请按以下结构输出（中文）：

1. **一句话概括**（20字以内）
2. **核心论点**（3-5个要点，每个不超过30字）
3. **关键证据或案例**（如果有）
4. **值得思考的问题**（1-2个）

要求：
- 简洁明了，去除冗余
- 突出逻辑主线
- 保留关键信息密度
"""
    
    print(f"  {DIM}正在生成导读...{RESET}\n")
    
    cfg = get_council_config()
    brief_content = chat(
        system_prompt="你是一个专业的内容精炼师，擅长提取文章核心要点。",
        user_prompt=brief_prompt,
        provider_config=cfg,
    )
    
    print(f"  {MAGENTA}{brief_content}{RESET}\n")
    print(f"  {DIM}{'─' * 56}{RESET}")
    print(f"  {DIM}提示: 使用 'python -m src view --item {args.item}' 查看完整内容{RESET}\n")


def cmd_council(args):
    """运行议事厅讨论，然后收集用户回应。"""
    article = get_article(args.item)
    if not article:
        print(f"\n{RED}Article ID {args.item} not found. Use 'list' to see available articles.{RESET}")
        return

    paradigm = getattr(args, "paradigm", "debate")
    _run_council_experience(article, pause_at_end=False, paradigm=paradigm)


def cmd_reflect(args):
    """查看认知历史 — 你的心智进化轨迹。"""
    memories = get_all_memories(limit=args.limit)
    if not memories:
        print(f"\n{YELLOW}No memories yet. Use 'council' to start recording your thoughts.{RESET}")
        return

    print(f"\n{BOLD}{MAGENTA}{'=' * 60}")
    print(f"  [Echo Location] -- Cognitive History")
    print(f"{'=' * 60}{RESET}\n")

    for m in memories:
        print(f"  {BOLD}{CYAN}#{m['id']}{RESET} {BOLD}{m['article_title']}{RESET}")
        print(f"     {DIM}{m['created_at']}{RESET}")
        print(f"     {MAGENTA}Stance: {m.get('stance_summary', '')}{RESET}")
        print(f"     Preference: {', '.join(m.get('core_preference', []))}")
        print(f"     Reasoning:  {m.get('reasoning_style', '')}")
        print(f"     Tone:       {m.get('emotional_tone', '')}")
        print(f"     Keywords:   {', '.join(m.get('topic_keywords', []))}")
        print(f"     {DIM}{'─' * 50}{RESET}\n")


def cmd_daily(args):
    """一键触发每日沉浸式心智训练。"""
    run_daily_session()

def cmd_resolve(args):
    """进入解惑空间。"""
    if args.list:
        from src.resolve.engine import run_sessions_list
        run_sessions_list()
    elif args.delete:
        from src.resolve.engine import delete_session
        if delete_session(args.delete):
            print(f"\n{GREEN}会话已删除。{RESET}\n")
        else:
            print(f"\n{RED}会话不存在。{RESET}\n")
    else:
        role = args.role if args.role else None
        session = args.session if args.session else None
        run_repl(role_key=role, session_id=session)


def _configure_provider(env_path: Path, prefix: str):
    """交互式配置特定的 Provider。"""
    is_global = (prefix == "OPENAI")
    display_prefix = "GLOBAL DEFAULT" if is_global else prefix
    
    current_key = get_key(env_path, f"{prefix}_API_KEY") 
    current_url = get_key(env_path, f"{prefix}_BASE_URL")
    current_models = get_key(env_path, f"{prefix}_MODEL_NAMES") or get_key(env_path, f"{prefix}_MODEL_NAME")

    # 如果特定配置为空，显示全局值作为预览
    if not is_global:
        if not current_key: current_key = get_key(env_path, "OPENAI_API_KEY") or ""
        if not current_url: current_url = get_key(env_path, "OPENAI_BASE_URL") or "https://api.openai.com/v1"
        if not current_models: current_models = get_key(env_path, "MODEL_NAMES") or get_key(env_path, "MODEL_NAME") or "gpt-4o-mini"

    mask_key = f"{current_key[:5]}...{current_key[-3:]}" if len(current_key) > 8 else "***"

    print(f"\n  {BOLD}{CYAN}--- Configuring {display_prefix} ---{RESET}")
    print(f"  {DIM}(Press Enter to keep current/global values){RESET}\n")

    new_key = input(f"    API_KEY [{mask_key}]: ").strip()
    if new_key:
        set_key(env_path, f"{prefix}_API_KEY", new_key)
        os.environ[f"{prefix}_API_KEY"] = new_key
        current_key = new_key

    new_url = input(f"    BASE_URL [{current_url}]: ").strip()
    if new_url:
        set_key(env_path, f"{prefix}_BASE_URL", new_url)
        os.environ[f"{prefix}_BASE_URL"] = new_url
        current_url = new_url

    new_models = input(f"    MODEL_NAMES [{current_models}]: ").strip()
    if new_models:
        set_key(env_path, f"{prefix}_MODEL_NAMES", new_models)
        os.environ[f"{prefix}_MODEL_NAMES"] = new_models
        current_models = new_models

    return _read_provider_config_from_env_file(env_path, prefix)


def _read_provider_config_from_env_file(env_path: Path, prefix: str) -> dict:
    """从 .env 读取指定 Provider，保持与 src.config 的回退规则一致。"""
    api_key = get_key(env_path, f"{prefix}_API_KEY")
    if not api_key and prefix != "OPENAI":
        api_key = get_key(env_path, "OPENAI_API_KEY")

    base_url = get_key(env_path, f"{prefix}_BASE_URL")
    if not base_url and prefix != "OPENAI":
        base_url = get_key(env_path, "OPENAI_BASE_URL") or "https://api.openai.com/v1"
    elif not base_url:
        base_url = "https://api.openai.com/v1"

    models_raw = get_key(env_path, f"{prefix}_MODEL_NAMES") or get_key(env_path, f"{prefix}_MODEL_NAME")
    if not models_raw and prefix == "OPENAI":
        models_raw = get_key(env_path, "MODEL_NAMES") or get_key(env_path, "MODEL_NAME")
    if not models_raw and prefix != "OPENAI":
        models_raw = get_key(env_path, "MODEL_NAMES") or get_key(env_path, "MODEL_NAME")

    models = [m.strip() for m in (models_raw or "").split(",") if m.strip()]
    return {
        "api_key": api_key,
        "base_url": base_url,
        "models": models,
    }


def _test_provider_config(prefix: str, provider_config: dict) -> bool:
    """对指定 Provider 发起最小 LLM 调用，检测 API Key/Base URL/模型是否可用。"""
    display_prefix = "GLOBAL DEFAULT" if prefix == "OPENAI" else prefix
    api_key = provider_config.get("api_key")
    base_url = provider_config.get("base_url") or "https://api.openai.com/v1"
    models = provider_config.get("models") or []

    print(f"\n  {BOLD}{CYAN}--- Testing {display_prefix} ---{RESET}")
    print(f"  {DIM}BASE_URL: {base_url}{RESET}")

    if not api_key:
        print(f"  {RED}API Key 未设置。{RESET}\n")
        _print_reconfigure_hint(prefix)
        return False
    if not models:
        print(f"  {RED}MODEL_NAMES 未设置。{RESET}\n")
        _print_reconfigure_hint(prefix)
        return False

    if prefix == "EMBEDDING":
        return _test_embedding_provider_config(display_prefix, api_key, base_url, models)

    from src.llm.client import chat

    ok = True
    for model in models:
        print(f"  {DIM}Testing model: {model} ...{RESET}")
        try:
            response = chat(
                system_prompt="You are a connection test endpoint. Reply briefly.",
                user_prompt="请只回复 OK。",
                model=model,
                max_retries=1,
                provider_config={
                    "api_key": api_key,
                    "base_url": base_url,
                    "models": [model],
                },
            )
            preview = (response or "").replace("\n", " ").strip()[:80]
            print(f"  {GREEN}✓ {model} 可用{RESET} {DIM}{preview}{RESET}")
        except Exception as exc:
            ok = False
            print(f"  {RED}✗ {model} 不可用：{exc}{RESET}")

    print()
    if not ok:
        _print_reconfigure_hint(prefix)
    return ok


def _test_embedding_provider_config(display_prefix: str, api_key: str, base_url: str, models: list[str]) -> bool:
    """检测 Embedding Provider，走 embeddings 接口而不是 chat 接口。"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    ok = True

    for model in models:
        print(f"  {DIM}Testing embedding model: {model} ...{RESET}")
        try:
            resp = client.embeddings.create(model=model, input=["MindPalace embedding test"])
            vector_dim = len(resp.data[0].embedding) if resp.data else 0
            print(f"  {GREEN}✓ {model} 可用{RESET} {DIM}vector_dim={vector_dim}{RESET}")
        except Exception as exc:
            ok = False
            print(f"  {RED}✗ {model} 不可用：{exc}{RESET}")

    print()
    if not ok:
        prefix = "OPENAI" if display_prefix == "GLOBAL DEFAULT" else display_prefix
        _print_reconfigure_hint(prefix)
    return ok


def _print_reconfigure_hint(prefix: str) -> None:
    """在检测失败后打印重配引导。"""
    label = PROVIDER_LABELS.get(prefix, prefix)
    cli_name = PROVIDER_CLI_NAMES.get(prefix, "global")

    print(f"  {YELLOW}建议重新配置 {label}。{RESET}")
    print(f"  {DIM}菜单路径：设置 -> 配置 API / 模型 -> {label}{RESET}")
    print(f"  {DIM}命令行检测：python -m src config --test --provider {cli_name}{RESET}")
    print(f"  {DIM}重点检查：API_KEY / BASE_URL / MODEL_NAMES{RESET}\n")


def _test_all_provider_configs(env_path: Path) -> dict:
    """批量检测所有 Provider，重复配置只真实调用一次。"""
    print(f"\n  {BOLD}{CYAN}--- Testing All Providers ---{RESET}")
    print(f"  {DIM}会按各 Provider 的最终生效配置检测；重复配置会复用结果。{RESET}\n")

    seen_signatures: dict[tuple, tuple[bool, str]] = {}
    results: list[tuple[str, bool]] = []
    failed_prefixes: list[str] = []

    for _, prefix in PROVIDER_PREFIXES.items():
        cfg = _read_provider_config_from_env_file(env_path, prefix)
        display_prefix = "GLOBAL DEFAULT" if prefix == "OPENAI" else prefix
        mode = "embedding" if prefix == "EMBEDDING" else "chat"
        signature = (
            mode,
            cfg.get("api_key") or "",
            cfg.get("base_url") or "https://api.openai.com/v1",
            tuple(cfg.get("models") or []),
        )

        if signature in seen_signatures:
            ok, source_prefix = seen_signatures[signature]
            marker = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
            print(f"  {marker} {display_prefix}  {DIM}(复用 {source_prefix} 的检测结果){RESET}")
        else:
            ok = _test_provider_config(prefix, cfg)
            seen_signatures[signature] = (ok, display_prefix)

        results.append((display_prefix, ok))
        if not ok:
            failed_prefixes.append(prefix)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  {BOLD}Summary:{RESET} {passed}/{total} providers passed.\n")
    if failed_prefixes:
        failed_labels = ", ".join(PROVIDER_LABELS.get(prefix, prefix) for prefix in failed_prefixes)
        print(f"  {YELLOW}检测失败的 Provider：{failed_labels}{RESET}")
        print(f"  {DIM}你可以回到“设置 -> 配置 API / 模型”逐个重配，再重新执行全部检测。{RESET}\n")

    return {
        "all_ok": passed == total,
        "failed_prefixes": failed_prefixes,
    }


def cmd_eval(args):
    """运行 LLM-as-a-Judge 周度评估。"""
    days = args.days
    print(f"\n  {BOLD}{CYAN}Running LLM-as-a-Judge evaluation (last {days} days)...{RESET}")

    from src.eval.judge_debates import judge_recent_debates, generate_weekly_report, save_weekly_report

    reports = judge_recent_debates(days=days)
    if not reports:
        print(f"  {YELLOW}No debates found in the last {days} days.{RESET}\n")
        return

    weekly = generate_weekly_report(reports, days=days)
    path = save_weekly_report(weekly)
    print(f"\n{weekly}")
    print(f"\n  {DIM}Report saved to: {path}{RESET}")

    # 可选: 生成 prompt 改进建议
    if args.iterate:
        print(f"\n  {DIM}Generating prompt improvement suggestions...{RESET}")
        from src.eval.prompt_iterator import generate_iteration_suggestions
        suggestions = generate_iteration_suggestions(days=days)
        print(f"\n{suggestions}")

    print()


def cmd_config(args):
    """交互式配置 API。"""
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"

    if not env_path.exists():
        if example_path.exists():
            shutil.copy(example_path, env_path)
        else:
            env_path.touch()

    if getattr(args, "test", False):
        if args.provider == "all":
            _test_all_provider_configs(env_path)
            return

        prefix = PROVIDER_PREFIXES.get(args.provider, "OPENAI")
        cfg = _read_provider_config_from_env_file(env_path, prefix)
        _test_provider_config(prefix, cfg)
        return

    print(f"\n{BOLD}{MAGENTA}{'=' * 60}")
    print(f"  [MindPalace] -- Interactive Configuration")
    print(f"{'=' * 60}{RESET}")
    print(f"\n  Which provider would you like to configure?")
    print(f"  1. {BOLD}Global Default{RESET} (Fallback for all tasks)")
    print(f"  2. {BOLD}Scout{RESET} (Scoring/Ranking - e.g. DeepSeek)")
    print(f"  3. {BOLD}Council{RESET} (Persona/Discussion - e.g. Gemini)")
    print(f"  4. {BOLD}Memory{RESET} (Profiling - defaults to Council)")
    
    choice = input(f"\n  Choose [1-4, Default: 1]: ").strip()
    
    prefix_map = {
        "1": "OPENAI",
        "2": "SCOUT",
        "3": "COUNCIL",
        "4": "MEMORY"
    }
    
    prefix = prefix_map.get(choice, "OPENAI")
    cfg = _configure_provider(env_path, prefix)

    print(f"\n  {BOLD}{GREEN}✓ Configuration saved securely to .env!{RESET}\n")

    test_now = input("  Test this provider now? [Y/n]: ").strip().lower()
    if test_now in ("", "y", "yes"):
        _test_provider_config(prefix, cfg)


def interactive_menu():
    """交互式主菜单。"""
    print(f"\n{BOLD}{MAGENTA}")
    print("  __  __ _           _ _____      _                 ")
    print(" |  \\/  (_)         | |  __ \\    | |                ")
    print(" | \\  / |_ _ __   __| | |__) |_ _| | __ _  ___ ___  ")
    print(" | |\\/| | | '_ \\ / _` |  ___/ _` | |/ _` |/ __/ _ \\ ")
    print(" | |  | | | | | | (_| | |  | (_| | | (_| | (_|  __/ ")
    print(" |_|  |_|_|_| |_|\\__,_|_|   \\__,_|_|\\__,_|\\___\\___| ")
    print(f"{RESET}")
    print(f"  {DIM}你的私人认知进化实验室{RESET}\n")

    while True:
        action = questionary.select(
            "请选择功能：",
            choices=[
                "🚀 今日练习",
                "📚 文章库",
                "🧭 心智漫游",
                "💬 深度对话",
                "🧠 认知回顾",
                "⚙️  设置",
                questionary.Separator(),
                "❌ 退出"
            ],
            style=custom_style
        ).ask()

        if not action or action == "❌ 退出":
            print(f"\n{DIM}再见！{RESET}\n")
            break

        try:
            if action.startswith("🚀"):
                run_daily_session()
            elif action.startswith("📚"):
                _interactive_library()
            elif action.startswith("🧭"):
                run_inquiry_menu()
            elif action.startswith("💬"):
                _interactive_resolve()
            elif action.startswith("🧠"):
                _interactive_cognition()
            elif action.startswith("⚙️"):
                _interactive_config()
        except KeyboardInterrupt:
            print(f"\n{DIM}操作已取消{RESET}\n")
            continue
        except Exception as e:
            print(f"\n{RED}错误: {e}{RESET}\n")
            logging.exception("Error in interactive menu")


def _interactive_library():
    """文章库子菜单。"""
    while True:
        action = questionary.select(
            "文章库：",
            choices=[
                "🎯 发现新文章",
                "📚 浏览文章",
                "⭐ 档案库",
                "🧹 清理旧文章",
                questionary.Separator(),
                "🔙 返回主菜单",
            ],
            style=custom_style,
        ).ask()

        if not action or action.startswith("🔙"):
            return

        try:
            if action.startswith("🎯"):
                _interactive_scout()
            elif action.startswith("📚"):
                _interactive_list()
            elif action.startswith("⭐"):
                _interactive_list(favorites_only=True)
            elif action.startswith("🧹"):
                _interactive_cleanup()
        except KeyboardInterrupt:
            print(f"\n{DIM}操作已取消{RESET}\n")
            return


def _interactive_cognition():
    """认知回顾子菜单。"""
    while True:
        action = questionary.select(
            "认知回顾：",
            choices=[
                "🧠 查看认知历史",
                "💎 查看认知洞察",
                "📦 导出认知档案",
                "📊 周度评估报告",
                questionary.Separator(),
                "🔙 返回主菜单",
            ],
            style=custom_style,
        ).ask()

        if not action or action.startswith("🔙"):
            return

        try:
            if action.startswith("🧠"):
                _interactive_memory()
            elif action.startswith("💎"):
                _interactive_crystals()
            elif action.startswith("📦"):
                _interactive_export_brain()
            elif action.startswith("📊"):
                _interactive_eval()
        except KeyboardInterrupt:
            print(f"\n{DIM}操作已取消{RESET}\n")
            return


def _interactive_crystals():
    """展示已结晶的结构化认知洞察。"""
    from src.storage.db import list_crystals
    from src.memory.crystallize import render_crystal_terminal

    crystals = list_crystals(limit=50)
    if not crystals:
        print(f"\n{YELLOW}还没有认知洞察。多参与 Council 辩论或心智漫游，积累足够后会自动结晶。{RESET}\n")
        return

    print(f"\n{BOLD}{MAGENTA}{'=' * 60}")
    print(f"  [Cognitive Crystals] -- 认知洞察（共 {len(crystals)} 条）")
    print(f"{'=' * 60}{RESET}\n")

    for cr in crystals:
        crystal_dict = {
            "type": cr.get("type", "observation"),
            "content": cr.get("content", ""),
            "confidence": cr.get("confidence", 0.0),
            "reasoning": "",
            "tags": cr.get("tags", []),
            "sources": cr.get("sources", []),
            "status": cr.get("status", "candidate"),
        }
        print(render_crystal_terminal(crystal_dict, colors=COLORS))
        print(f"  {DIM}{cr.get('created_at', '')[:10]}{RESET}")
        print(f"  {DIM}{'─' * 54}{RESET}\n")

    try:
        input(f"{DIM}按 Enter 继续...{RESET}")
    except EOFError:
        return


def _interactive_export_brain():
    """将认知洞察导出为 Markdown brain 目录。"""
    from src.memory.brain_export import export_brain

    try:
        count = export_brain()
    except Exception as exc:
        print(f"\n{RED}导出失败: {exc}{RESET}\n")
        return

    if count == 0:
        print(f"\n{YELLOW}还没有认知洞察可导出。{RESET}\n")
        return

    print(f"\n{GREEN}✓ 已导出 {count} 条认知洞察到 data/brain/ 目录{RESET}")
    print(f"{DIM}  ├─ axioms/        （身份级信念）")
    print(f"  ├─ principles/    （可复用规则）")
    print(f"  └─ observations/  （日常观察）{RESET}")
    print(f"\n{DIM}文件带 YAML frontmatter，可用 Obsidian 打开。{RESET}\n")

    try:
        input(f"{DIM}按 Enter 继续...{RESET}")
    except EOFError:
        return


def _interactive_scout():
    """交互式 Scout。"""
    top_k = questionary.text(
        "返回前几篇文章？",
        default="5",
        style=custom_style
    ).ask()
    
    if not top_k:
        return
    
    try:
        top_k = int(top_k)
    except ValueError:
        print(f"{RED}请输入有效数字{RESET}")
        return
    
    cfg = get_scout_config()
    results = run_scout(feed_urls=None, top_k=top_k, provider_config=cfg)
    _display_results(results)


def _prompt_article_filters(current_filters: dict | None) -> dict:
    """交互式设置文章筛选。"""
    current = _normalize_article_filters(current_filters)
    print(f"\n{DIM}留空表示不筛选；标签请用逗号分隔。{RESET}")

    query = questionary.text(
        "关键词（标题/摘要/备注/标签）：",
        default=current["query"],
        style=custom_style,
    ).ask()
    if query is None:
        return current

    tag_text = questionary.text(
        "标签：",
        default=", ".join(current["tags"]),
        style=custom_style,
    ).ask()
    if tag_text is None:
        return current

    source = questionary.text(
        "来源关键词：",
        default=current["source"],
        style=custom_style,
    ).ask()
    if source is None:
        return current

    days_text = questionary.text(
        "最近几天（留空不限）：",
        default="" if current["days"] is None else str(current["days"]),
        style=custom_style,
    ).ask()
    if days_text is None:
        return current

    days = None
    if days_text.strip():
        try:
            days = int(days_text.strip())
        except ValueError:
            print(f"\n{RED}最近天数必须是数字。将保留原筛选条件。{RESET}\n")
            return current
        if days < 1:
            print(f"\n{RED}最近天数必须大于等于 1。将保留原筛选条件。{RESET}\n")
            return current

    return {
        "query": query.strip(),
        "tags": _parse_tag_text(tag_text),
        "source": source.strip(),
        "days": days,
    }


def _show_recent_discussions(article: dict):
    """查看某篇文章最近的讨论记录。"""
    debates = list_recent_debates_for_article(article["id"], limit=5)
    if not debates:
        print(f"\n{YELLOW}这篇文章还没有历史讨论。可以先发起一次议事厅讨论。{RESET}\n")
        input(f"{DIM}按 Enter 继续...{RESET}")
        return

    choices = []
    for debate in debates:
        headline = (debate.get("consensus") or {}).get("headline") or debate.get("article_title", "")
        title = headline[:42] + "..." if len(headline) > 42 else headline
        choices.append(f"[#{debate['id']}] {debate['created_at'][:10]} {title}")
    choices.extend([questionary.Separator(), "🔙 返回"])

    selected = questionary.select(
        "选择要回看的讨论：",
        choices=choices,
        style=custom_style,
    ).ask()

    if not selected or selected.startswith("🔙"):
        return

    debate_id = int(selected.split("]")[0].strip("[#"))
    debate = next((item for item in debates if item["id"] == debate_id), None)
    if not debate:
        return

    _print_recent_debate_summary(debate)
    input(f"{DIM}按 Enter 继续...{RESET}")


def _interactive_edit_note(article: dict):
    """交互式编辑文章备注。"""
    note = questionary.text(
        "输入备注（留空表示清空）：",
        default=article.get("favorite_note") or "",
        style=custom_style,
    ).ask()
    if note is None:
        return

    set_article_note(article["id"], note)
    updated = get_article(article["id"])
    if updated and updated.get("favorite_note"):
        print(f"\n{GREEN}已更新备注。{RESET}")
        print(f"{YELLOW}{updated['favorite_note']}{RESET}\n")
    else:
        print(f"\n{YELLOW}已清空备注。{RESET}\n")
    input(f"{DIM}按 Enter 继续...{RESET}")


def _interactive_edit_tags(article: dict):
    """交互式编辑文章标签。"""
    tag_text = questionary.text(
        "输入标签（逗号分隔，留空表示清空）：",
        default=", ".join(article.get("tags") or []),
        style=custom_style,
    ).ask()
    if tag_text is None:
        return

    replace_article_tags(article["id"], _parse_tag_text(tag_text))
    updated = get_article(article["id"])
    print(f"\n{GREEN}已更新标签：{_format_tags(updated.get('tags'))}{RESET}\n")
    input(f"{DIM}按 Enter 继续...{RESET}")


def _interactive_list(favorites_only: bool = False, filters: dict | None = None):
    """交互式文章库，支持筛选、收藏、标签、备注和历史讨论。"""
    current_filters = _normalize_article_filters(filters)
    empty_message = (
        "收藏档案里还没有文章。可以先在 Browse 里收藏或打标签。"
        if favorites_only
        else "数据库中暂无文章。请先运行 Scout。"
    )

    while True:
        articles = _load_articles_for_filters(
            limit=20, favorites_only=favorites_only, filters=current_filters
        )

        if not articles:
            if not _has_active_filters(current_filters):
                print(f"\n{YELLOW}{empty_message}{RESET}\n")
                return

            print(
                f"\n{YELLOW}当前筛选下没有文章："
                f"{_format_filter_summary(current_filters, favorites_only)}{RESET}\n"
            )
            action = questionary.select(
                "你想怎么做？",
                choices=["🔎 调整筛选", "🧹 清空筛选", "🔙 返回主菜单"],
                style=custom_style,
            ).ask()
            if action and action.startswith("🔎"):
                current_filters = _prompt_article_filters(current_filters)
                continue
            if action and action.startswith("🧹"):
                current_filters = _empty_article_filters()
                continue
            return

        print(
            f"\n{BOLD}{CYAN}[文章筛选]{RESET} "
            f"{_format_filter_summary(current_filters, favorites_only)}\n"
        )

        choices = []
        for article in articles:
            score_bar = _format_score_bar_plain(article["total_score"])
            title = (
                article["title"][:46] + "..."
                if len(article["title"]) > 46
                else article["title"]
            )
            archive_mark = (
                "[收藏] " if article.get("is_favorite")
                else "[标签] " if article.get("tags")
                else ""
            )
            note_mark = " 📝" if article.get("favorite_note") else ""
            tag_preview = ""
            if article.get("tags"):
                preview_tags = " ".join(f"#{tag}" for tag in article["tags"][:2])
                tag_preview = f" {preview_tags}"
            choices.append(
                f"[ID:{article['id']}] {archive_mark}{title} "
                f"{score_bar}{note_mark}{tag_preview}"
            )

        choices.append(questionary.Separator())
        choices.append("🔎 设置筛选")
        if _has_active_filters(current_filters):
            choices.append("🧹 清空筛选")
        choices.append("🔙 返回主菜单")

        selected = questionary.select(
            "选择文章：",
            choices=choices,
            style=custom_style,
        ).ask()

        if not selected:
            return
        if selected.startswith("🔎"):
            current_filters = _prompt_article_filters(current_filters)
            continue
        if selected.startswith("🧹"):
            current_filters = _empty_article_filters()
            continue
        if selected.startswith("🔙"):
            return

        article_id = int(selected.split("]")[0].split(":")[1])
        article = get_article(article_id)
        if not article:
            print(f"\n{RED}文章不存在。{RESET}\n")
            continue

        recent_discussions = _show_article_overview(article)
        action = questionary.select(
            "你想做什么？",
            choices=[
                "📖 生成导读精炼版",
                "🌐 查看原文（浏览器打开）",
                "🏛️  发起议事厅讨论",
                f"🕘 查看最近讨论（{len(recent_discussions)}）",
                (
                    "⭐ 取消收藏"
                    if article.get("is_favorite")
                    else "⭐ 收藏到收藏夹"
                ),
                "📝 编辑备注",
                "🏷️ 编辑标签",
                questionary.Separator(),
                "🔙 返回文章列表",
            ],
            style=custom_style,
        ).ask()

        if not action or action.startswith("🔙"):
            continue

        if action.startswith("📖"):
            _show_brief(article)
        elif action.startswith("🌐"):
            _open_in_browser(article)
        elif action.startswith("🏛️"):
            _start_council(article)
        elif action.startswith("🕘"):
            _show_recent_discussions(article)
        elif action.startswith("⭐"):
            set_article_favorite(
                article["id"], favorite=not article.get("is_favorite")
            )
            status = (
                "已收藏" if not article.get("is_favorite") else "已取消收藏"
            )
            color = GREEN if not article.get("is_favorite") else YELLOW
            print(f"\n{color}{status}: {article['title']}{RESET}\n")
            input(f"{DIM}按 Enter 继续...{RESET}")
        elif action.startswith("📝"):
            _interactive_edit_note(article)
        elif action.startswith("🏷️"):
            _interactive_edit_tags(article)


def _show_brief(article):
    """显示文章导读。"""
    print(f"\n{DIM}正在生成导读...{RESET}\n")
    
    from src.llm.client import chat
    
    brief_prompt = f"""请为以下文章生成一份导读精炼版，帮助读者快速理解核心内容。

文章标题: {article['title']}
文章摘要: {article.get('summary', '')}

请按以下结构输出（中文）：

1. **一句话概括**（20字以内）
2. **核心论点**（3-5个要点，每个不超过30字）
3. **关键证据或案例**（如果有）
4. **值得思考的问题**（1-2个）

要求：
- 简洁明了，去除冗余
- 突出逻辑主线
- 保留关键信息密度
"""
    
    try:
        cfg = get_council_config()
        brief_content = chat(
            system_prompt="你是一个专业的内容精炼师，擅长提取文章核心要点。",
            user_prompt=brief_prompt,
            provider_config=cfg,
        )
        
        print(f"{BOLD}{MAGENTA}[导读精炼版]{RESET}\n")
        print(f"{brief_content}\n")
        print(f"{DIM}{'─' * 56}{RESET}\n")
        
    except RuntimeError as e:
        error_msg = str(e)
        if "blocked" in error_msg.lower() or "403" in error_msg:
            print(f"{YELLOW}⚠️  内容安全过滤触发，无法生成导读。{RESET}")
            print(f"{DIM}这可能是因为文章内容触发了 API 的安全策略。{RESET}")
            print(f"{DIM}建议：直接查看原文或发起讨论。{RESET}\n")
        else:
            print(f"{RED}生成导读失败: {e}{RESET}\n")
    except Exception as e:
        print(f"{RED}生成导读失败: {e}{RESET}\n")
    
    input(f"{DIM}按 Enter 继续...{RESET}")


def _open_in_browser(article):
    """在浏览器中打开文章原文。"""
    import webbrowser
    
    url = article.get('url', '')
    if not url:
        print(f"\n{RED}文章链接不存在。{RESET}\n")
        return
    
    print(f"\n{DIM}正在打开浏览器...{RESET}")
    print(f"{CYAN}{url}{RESET}\n")
    
    try:
        webbrowser.open(url)
        print(f"{GREEN}✓ 已在浏览器中打开原文{RESET}\n")
    except Exception as e:
        print(f"{RED}无法打开浏览器: {e}{RESET}\n")
        print(f"请手动访问: {url}\n")
    
    input(f"{DIM}按 Enter 继续...{RESET}")


def _start_council(article):
    """发起议事厅讨论。"""
    paradigm = _prompt_paradigm()
    _run_council_experience(article, pause_at_end=True, paradigm=paradigm)


def _interactive_view():
    """交互式 View - 已整合到 List 中。"""
    print(f"\n{YELLOW}此功能已整合到 List 中，请使用 List 功能。{RESET}\n")


def _interactive_brief():
    """交互式 Brief - 已整合到 List 中。"""
    print(f"\n{YELLOW}此功能已整合到 List 中，请使用 List 功能。{RESET}\n")


def _interactive_council():
    """交互式 Council - 已整合到 List 中。"""
    print(f"\n{YELLOW}此功能已整合到 List 中，请使用 List 功能选择文章后发起讨论。{RESET}\n")


def _prompt_paradigm() -> str:
    """交互式选择讨论范式。"""
    try:
        choice = questionary.select(
            "选择讨论范式:",
            choices=[
                questionary.Choice(
                    title="🥊 辩论 (Debate) — 对抗式多轮反驳",
                    value="debate",
                ),
                questionary.Choice(
                    title="📝 报告 (Report) — 中心化起草 + 审阅",
                    value="report",
                ),
            ],
            use_arrow_keys=True,
        ).ask()
    except (EOFError, KeyboardInterrupt):
        return "debate"
    return choice or "debate"


def _interactive_memory():
    """交互式 Memory。"""
    memories = get_all_memories(limit=20)
    if not memories:
        print(f"\n{YELLOW}No memories yet. Use Council to start recording your thoughts.{RESET}\n")
        return

    print(f"\n{BOLD}{MAGENTA}{'=' * 60}")
    print(f"  [Echo Location] -- Cognitive History")
    print(f"{'=' * 60}{RESET}\n")

    for m in memories:
        print(f"  {BOLD}{CYAN}#{m['id']}{RESET} {BOLD}{m['article_title']}{RESET}")
        print(f"     {DIM}{m['created_at']}{RESET}")
        print(f"     {MAGENTA}Stance: {m.get('stance_summary', '')}{RESET}")
        print(f"     Preference: {', '.join(m.get('core_preference', []))}")
        print(f"     Reasoning:  {m.get('reasoning_style', '')}")
        print(f"     Tone:       {m.get('emotional_tone', '')}")
        print(f"     Keywords:   {', '.join(m.get('topic_keywords', []))}")
        print(f"     {DIM}{'─' * 50}{RESET}\n")


def _interactive_resolve():
    """交互式 Resolve。"""
    from src.resolve.engine import list_sessions, run_repl
    
    action = questionary.select(
        "Resolve 模式：",
        choices=[
            "🆕 开始新对话（议事厅模式）",
            "👤 开始新对话（单角色模式）",
            "📜 查看并恢复历史会话",
            "🔙 返回主菜单"
        ],
        style=custom_style
    ).ask()
    
    if not action or action.startswith("🔙"):
        return
    
    if action.startswith("🆕"):
        run_repl()
    elif action.startswith("👤"):
        role = questionary.select(
            "选择角色：",
            choices=["critic", "synthesizer", "mentor"],
            style=custom_style
        ).ask()
        if role:
            run_repl(role_key=role)
    elif action.startswith("📜"):
        sessions = list_sessions()
        if not sessions:
            print(f"\n{YELLOW}暂无历史会话。{RESET}\n")
            return
        
        choices = [
            f"[{s['id'][:8]}...] {s['title']} ({s['mode']}) - {s['updated_at'][:10]}"
            for s in sessions
        ]
        choices.append(questionary.Separator())
        choices.append("🔙 返回")
        
        selected = questionary.select(
            "选择会话：",
            choices=choices,
            style=custom_style
        ).ask()
        
        if selected and not selected.startswith("🔙"):
            session_id = selected.split("]")[0].strip("[")
            run_repl(session_id=session_id)


def _interactive_eval():
    """交互式 Eval。"""
    from src.eval.judge_debates import judge_recent_debates, generate_weekly_report, save_weekly_report

    days_str = questionary.text(
        "评估最近几天的讨论？",
        default="7",
        style=custom_style,
    ).ask()
    if not days_str:
        return
    try:
        days = int(days_str)
    except ValueError:
        print(f"{RED}请输入有效数字{RESET}")
        return

    print(f"\n  {DIM}正在评估最近 {days} 天的讨论...{RESET}")
    reports = judge_recent_debates(days=days)
    if not reports:
        print(f"  {YELLOW}该时间段内没有讨论记录。{RESET}\n")
        return

    weekly = generate_weekly_report(reports, days=days)
    path = save_weekly_report(weekly)
    print(f"\n{weekly}")
    print(f"\n  {DIM}报告已保存至: {path}{RESET}\n")

    do_iterate = questionary.confirm(
        "是否生成 Prompt 改进建议？",
        default=False,
        style=custom_style,
    ).ask()
    if do_iterate:
        from src.eval.prompt_iterator import generate_iteration_suggestions
        print(f"\n  {DIM}正在生成...{RESET}")
        suggestions = generate_iteration_suggestions(days=days)
        print(f"\n{suggestions}\n")


def _interactive_cleanup():
    """交互式清理旧文章。"""
    days_str = questionary.text(
        "普通文章保留多少天？",
        default=str(ARTICLE_RETENTION_DAYS),
        style=custom_style,
    ).ask()
    if not days_str:
        return
    try:
        days = int(days_str)
    except ValueError:
        print(f"{RED}请输入有效数字{RESET}")
        return
    if days < 1:
        print(f"{RED}保留天数必须大于等于 1。{RESET}")
        return

    dry_run = questionary.confirm(
        "先预览，不实际删除？",
        default=True,
        style=custom_style,
    ).ask()
    if dry_run is None:
        return

    include_discussed = questionary.confirm(
        "也清理已讨论/已记忆的旧文章？",
        default=False,
        style=custom_style,
    ).ask()
    if include_discussed is None:
        return

    include_tagged = questionary.confirm(
        "也清理已打标签的旧文章？",
        default=False,
        style=custom_style,
    ).ask()
    if include_tagged is None:
        return

    result = cleanup_old_articles(
        retention_days=days,
        dry_run=dry_run,
        keep_discussed=not include_discussed,
        keep_tagged=not include_tagged,
    )

    count = result["candidate_count"] if dry_run else result["deleted_count"]
    print(f"\n{BOLD}{CYAN}[Article Cleanup]{RESET}")
    print(f"  {'可清理' if dry_run else '已清理'}: {count} 篇")
    print(f"  {DIM}收藏文章会始终保留；带标签文章默认也会保留。{RESET}\n")

    for item in result.get("candidates", [])[:10]:
        print(f"  - ID:{item['id']} {item['title'][:60]}")
    if len(result.get("candidates", [])) > 10:
        print(f"  {DIM}... 还有 {len(result['candidates']) - 10} 篇{RESET}")
    print()
    input(f"{DIM}按 Enter 继续...{RESET}")


def _interactive_config():
    """交互式 Config。"""
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"

    if not env_path.exists():
        if example_path.exists():
            shutil.copy(example_path, env_path)
        else:
            env_path.touch()

    action = questionary.select(
        "设置：",
        choices=[
            "🔧 配置 API / 模型",
            "🧪 检测 API 是否可用",
            questionary.Separator(),
            "🔙 返回主菜单",
        ],
        style=custom_style,
    ).ask()

    if not action or action.startswith("🔙"):
        return

    provider = _select_provider_for_config(allow_all=action.startswith("🧪"))
    if not provider:
        return

    if action.startswith("🔧"):
        cfg = _configure_provider(env_path, provider)
        print(f"\n  {BOLD}{GREEN}✓ Configuration saved!{RESET}\n")
        test_now = questionary.confirm(
            "是否立即检测这个配置？",
            default=True,
            style=custom_style,
        ).ask()
        if test_now:
            _test_provider_config(provider, cfg)
    elif action.startswith("🧪"):
        if provider == "ALL":
            result = _test_all_provider_configs(env_path)
            if not result["all_ok"]:
                _prompt_reconfigure_failed_provider(env_path, result["failed_prefixes"])
        else:
            cfg = _read_provider_config_from_env_file(env_path, provider)
            ok = _test_provider_config(provider, cfg)
            if not ok:
                reconfigure_now = questionary.confirm(
                    f"{PROVIDER_LABELS.get(provider, provider)} 检测失败，现在重新配置？",
                    default=True,
                    style=custom_style,
                ).ask()
                if reconfigure_now:
                    updated_cfg = _configure_provider(env_path, provider)
                    print(f"\n  {BOLD}{GREEN}✓ Configuration saved!{RESET}\n")
                    _test_provider_config(provider, updated_cfg)


def _select_provider_for_config(allow_all: bool = False) -> str | None:
    choices = [
        "Global Default (全局默认)",
        "Scout (评分/排序)",
        "Council (讨论/对话)",
        "Memory (认知画像)",
        "Fast (轻量任务)",
        "Router (难度路由)",
        "Judge (最终裁决/评估)",
        "Embedding (向量化)",
    ]
    if allow_all:
        choices.append("All Providers (全部检测)")
    choices.extend([questionary.Separator(), "🔙 返回"])

    provider = questionary.select(
        "选择 Provider：",
        choices=choices,
        style=custom_style,
    ).ask()

    if not provider or provider.startswith("🔙"):
        return None

    prefix_map = {
        "Global Default (全局默认)": "OPENAI",
        "Scout (评分/排序)": "SCOUT",
        "Council (讨论/对话)": "COUNCIL",
        "Memory (认知画像)": "MEMORY",
        "Fast (轻量任务)": "FAST",
        "Router (难度路由)": "ROUTER",
        "Judge (最终裁决/评估)": "JUDGE",
        "Embedding (向量化)": "EMBEDDING",
        "All Providers (全部检测)": "ALL",
    }
    return prefix_map.get(provider, "OPENAI")


def _prompt_reconfigure_failed_provider(env_path: Path, failed_prefixes: list[str]) -> None:
    """在全部检测失败后，引导用户直接进入重配流程。"""
    if not failed_prefixes:
        return

    choices = [f"{PROVIDER_LABELS.get(prefix, prefix)}" for prefix in failed_prefixes]
    choices.extend([questionary.Separator(), "暂时不配置"])

    selected = questionary.select(
        "选择一个失败的 Provider 立即重配：",
        choices=choices,
        style=custom_style,
    ).ask()

    if not selected or selected == "暂时不配置":
        return

    label_to_prefix = {PROVIDER_LABELS.get(prefix, prefix): prefix for prefix in failed_prefixes}
    prefix = label_to_prefix[selected]
    updated_cfg = _configure_provider(env_path, prefix)
    print(f"\n  {BOLD}{GREEN}✓ Configuration saved!{RESET}\n")
    _test_provider_config(prefix, updated_cfg)


def main():
    parser = argparse.ArgumentParser(
        prog="mindpalace",
        description="MindPalace Agent",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive menu")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # config
    scout_parser = subparsers.add_parser("scout", help="Run the Scout pipeline")
    scout_parser.add_argument(
        "--feeds", nargs="*", help="Custom RSS feed URLs"
    )
    scout_parser.add_argument(
        "--preset",
        choices=sorted(FEED_PRESETS.keys()),
        help="Built-in feed preset: humanities, mixed, or tech",
    )
    scout_parser.add_argument(
        "--top", type=int, default=5, help="Return top N results (default: 5)"
    )
    scout_parser.set_defaults(func=cmd_scout)

    # list
    list_parser = subparsers.add_parser("list", help="List saved articles")
    list_parser.add_argument(
        "--limit", type=int, default=20, help="Max articles to show (default: 20)"
    )
    list_parser.add_argument(
        "--query", type=str, help="Search title, summary, note, and tags"
    )
    list_parser.add_argument(
        "--tag", action="append", help="Filter by tag (repeatable)"
    )
    list_parser.add_argument(
        "--source", type=str, help="Filter by source keyword"
    )
    list_parser.add_argument(
        "--days", type=int, help="Only show articles from the last N days"
    )
    list_parser.set_defaults(func=cmd_list)

    # favorite
    favorite_parser = subparsers.add_parser("favorite", help="Add an article to favorites")
    favorite_parser.add_argument(
        "--item", type=int, required=True, help="Article ID (use 'list' to see IDs)"
    )
    favorite_parser.add_argument(
        "--note", type=str, help="Optional note for why this article is worth keeping"
    )
    favorite_parser.add_argument(
        "--tags", type=str, help="Optional comma-separated tags to append"
    )
    favorite_parser.set_defaults(func=cmd_favorite)

    # unfavorite
    unfavorite_parser = subparsers.add_parser("unfavorite", help="Remove an article from favorites")
    unfavorite_parser.add_argument(
        "--item", type=int, required=True, help="Article ID (use 'favorites' to see IDs)"
    )
    unfavorite_parser.set_defaults(func=cmd_unfavorite)

    # favorites
    favorites_parser = subparsers.add_parser("favorites", help="List favorite articles")
    favorites_parser.add_argument(
        "--limit", type=int, default=20, help="Max articles to show (default: 20)"
    )
    favorites_parser.add_argument(
        "--query", type=str, help="Search title, summary, note, and tags"
    )
    favorites_parser.add_argument(
        "--tag", action="append", help="Filter favorites by tag (repeatable)"
    )
    favorites_parser.add_argument(
        "--source", type=str, help="Filter favorites by source keyword"
    )
    favorites_parser.add_argument(
        "--days", type=int, help="Only show favorites from the last N days"
    )
    favorites_parser.set_defaults(func=cmd_favorites)

    # note
    note_parser = subparsers.add_parser("note", help="Edit an article note")
    note_parser.add_argument(
        "--item", type=int, required=True, help="Article ID (use 'list' to see IDs)"
    )
    note_group = note_parser.add_mutually_exclusive_group(required=True)
    note_group.add_argument("--text", type=str, help="Set or replace the note text")
    note_group.add_argument("--clear", action="store_true", help="Clear the note")
    note_parser.set_defaults(func=cmd_note)

    # tag
    tag_parser = subparsers.add_parser("tag", help="Edit article tags")
    tag_parser.add_argument(
        "--item", type=int, required=True, help="Article ID (use 'list' to see IDs)"
    )
    tag_group = tag_parser.add_mutually_exclusive_group(required=True)
    tag_group.add_argument("--set", type=str, help="Replace tags with a comma-separated list")
    tag_group.add_argument("--add", type=str, help="Add comma-separated tags")
    tag_group.add_argument("--remove", type=str, help="Remove comma-separated tags")
    tag_group.add_argument("--clear", action="store_true", help="Remove all tags")
    tag_parser.set_defaults(func=cmd_tag)

    # cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean old non-favorite articles")
    cleanup_parser.add_argument(
        "--days",
        type=int,
        default=ARTICLE_RETENTION_DAYS,
        help=f"Keep non-favorite articles for N days (default: {ARTICLE_RETENTION_DAYS})",
    )
    cleanup_parser.add_argument(
        "--dry-run", action="store_true", help="Preview cleanup without deleting"
    )
    cleanup_parser.add_argument(
        "--include-discussed",
        action="store_true",
        help="Also delete old articles that have debates or memories",
    )
    cleanup_parser.add_argument(
        "--include-tagged",
        action="store_true",
        help="Also delete old articles that have archive tags",
    )
    cleanup_parser.set_defaults(func=cmd_cleanup)

    # view
    view_parser = subparsers.add_parser("view", help="View full article content")
    view_parser.add_argument(
        "--item", type=int, required=True, help="Article ID (use 'list' to see IDs)"
    )
    view_parser.set_defaults(func=cmd_view)

    # brief
    brief_parser = subparsers.add_parser("brief", help="Generate a concise reading guide for an article")
    brief_parser.add_argument(
        "--item", type=int, required=True, help="Article ID (use 'list' to see IDs)"
    )
    brief_parser.set_defaults(func=cmd_brief)

    # council
    council_parser = subparsers.add_parser("council", help="Run Council discussion on an article")
    council_parser.add_argument(
        "--item", type=int, required=True, help="Article ID (use 'list' to see IDs)"
    )
    council_parser.add_argument(
        "--paradigm",
        choices=("debate", "report"),
        default="debate",
        help="Discussion paradigm: debate (adversarial, default) or report (draft + review)",
    )
    council_parser.set_defaults(func=cmd_council)

    # reflect
    reflect_parser = subparsers.add_parser("reflect", help="View your cognitive evolution history")
    reflect_parser.add_argument(
        "--limit", type=int, default=20, help="Max memories to show (default: 20)"
    )
    reflect_parser.set_defaults(func=cmd_reflect)

    # daily
    daily_parser = subparsers.add_parser("daily", help="[Phase 4] Run the daily End-to-End session")
    daily_parser.set_defaults(func=cmd_daily)

    # resolve
    resolve_parser = subparsers.add_parser("resolve", help="Enter Resolve Space for deep conversation")
    resolve_parser.add_argument("--role", type=str, help="Talk to a specific role (e.g. mentor, critic)")
    resolve_parser.add_argument("--session", type=str, help="Resume a specific session by ID")
    resolve_parser.add_argument("--list", action="store_true", help="List all saved sessions")
    resolve_parser.add_argument("--delete", type=str, help="Delete a session by ID")
    resolve_parser.set_defaults(func=cmd_resolve)

    # eval
    eval_parser = subparsers.add_parser("eval", help="Run weekly LLM-as-a-Judge evaluation")
    eval_parser.add_argument(
        "--days", type=int, default=7, help="Evaluation window in days (default: 7)"
    )
    eval_parser.add_argument(
        "--iterate", action="store_true", help="Also generate prompt improvement suggestions"
    )
    eval_parser.set_defaults(func=cmd_eval)

    # config
    config_parser = subparsers.add_parser("config", help="Interactive setup for API Key and Models")
    config_parser.add_argument(
        "--test",
        action="store_true",
        help="Test whether a configured provider can complete a minimal LLM call",
    )
    config_parser.add_argument(
        "--provider",
        choices=["all", "global", "scout", "council", "memory", "fast", "router", "judge", "embedding"],
        default="global",
        help="Provider to test when using --test (default: global, use 'all' for every provider)",
    )
    config_parser.set_defaults(func=cmd_config)

    args = parser.parse_args()
    _setup_logging(args.verbose)

    # 初始化链路追踪（由 TRACING_ENABLED 环境变量控制，默认关闭）
    from src.obs import init_tracing
    init_tracing()

    # 如果没有提供任何命令，或者使用了 -i 参数，启动交互式菜单
    if args.command is None or args.interactive:
        interactive_menu()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        print(f"{YELLOW}This command is not yet implemented.{RESET}")


if __name__ == "__main__":
    main()
