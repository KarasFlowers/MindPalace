"""MindPalace CLI 入口。"""

import argparse
import io
import logging
import os
import sys
from pathlib import Path

# Windows 终端默认 GBK 编码，强制 UTF-8 避免 emoji/中文输出报错
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import questionary
from questionary import Style

from src.scout.pipeline import run_scout
from src.council.flow import run_council
from src.council.output import format_council_result
from src.storage.db import (
    cleanup_old_articles,
    get_article,
    list_articles,
    save_debate,
    set_article_favorite,
)
from src.memory.profiler import profile_response
from src.memory.store import save_memory, find_related_memories, get_all_memories
from src.memory.echo import generate_echo_report, format_echo_report
from src.workflows.daily_session import run_daily_session
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
    articles = list_articles(limit=args.limit)
    if not articles:
        print(f"\n{YELLOW}数据库中暂无文章。请先运行 scout 命令。{RESET}")
        return

    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  [MindPalace] -- Saved Articles")
    print(f"{'=' * 60}{RESET}\n")

    for a in articles:
        scores = a.get("scores", {})
        favorite = f" {YELLOW}[收藏]{RESET}" if a.get("is_favorite") else ""
        print(f"  {BOLD}{CYAN}ID:{a['id']}{RESET}{favorite} {BOLD}{a['title']}{RESET}")
        print(f"     {DIM}[SRC] {a['source']}  |  Score: {a['total_score']:.1f}/10{RESET}")
        print(f"     {MAGENTA}> {a.get('summary', '')}{RESET}")
        print(f"     {DIM}{'─' * 50}{RESET}\n")


def cmd_favorite(args):
    """收藏文章。"""
    article = get_article(args.item)
    if not article:
        print(f"\n{RED}Article ID {args.item} not found. Use 'list' to see available articles.{RESET}")
        return

    set_article_favorite(args.item, favorite=True, note=args.note)
    print(f"\n{GREEN}已收藏：{article['title']}{RESET}\n")


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
    articles = list_articles(limit=args.limit, favorites_only=True)
    if not articles:
        print(f"\n{YELLOW}收藏夹还没有文章。看到值得反复讨论的材料，可以用 favorite 收藏。{RESET}")
        return

    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  [MindPalace] -- Favorites")
    print(f"{'=' * 60}{RESET}\n")

    for a in articles:
        note = a.get("favorite_note")
        print(f"  {BOLD}{CYAN}ID:{a['id']}{RESET} {BOLD}{a['title']}{RESET}")
        print(f"     {DIM}[SRC] {a['source']}  |  Score: {a['total_score']:.1f}/10{RESET}")
        if note:
            print(f"     {YELLOW}Note: {note}{RESET}")
        print(f"     {MAGENTA}> {a.get('summary', '')}{RESET}")
        print(f"     {DIM}{'─' * 50}{RESET}\n")


def cmd_cleanup(args):
    """清理旧文章。"""
    if args.days < 1:
        print(f"\n{RED}保留天数必须大于等于 1。{RESET}\n")
        return

    result = cleanup_old_articles(
        retention_days=args.days,
        dry_run=args.dry_run,
        keep_discussed=not args.include_discussed,
    )

    action = "可清理" if args.dry_run else "已清理"
    count = result["candidate_count"] if args.dry_run else result["deleted_count"]
    print(f"\n{BOLD}{CYAN}[Article Cleanup]{RESET}")
    print(f"  保留天数: {args.days}")
    print(f"  收藏文章: 永远保留")
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
    print(f"\n  {BOLD}{MAGENTA}[摘要]{RESET}")
    print(f"  {article.get('summary', '')}")
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

    print(f"\n{BOLD}Starting Council discussion for: {article['title']}{RESET}\n")

    cfg = get_council_config()
    result = run_council(
        title=article["title"],
        summary=article.get("summary", ""),
        content=article.get("summary", ""),
        provider_config=cfg,
    )

    try:
        save_debate(result, article_id=article.get("id"))
    except Exception:
        logging.getLogger(__name__).exception("Failed to persist debate state")

    output = format_council_result(result, colors=COLORS)
    print(output)

    # 收集用户回应
    print(f"  {BOLD}{CYAN}{'=' * 60}")
    print(f"  [Your Turn] -- Share Your Thoughts")
    print(f"  {'=' * 60}{RESET}")
    print(f"  {DIM}(Enter your response. Press Enter twice to submit, or type 'skip' to skip){RESET}\n")

    lines = []
    while True:
        try:
            line = input(f"  {GREEN}>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() == "skip":
            print(f"\n  {DIM}Skipped. Your thoughts are your own.{RESET}\n")
            return
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    user_response = "\n".join(lines).strip()
    if not user_response:
        print(f"\n  {DIM}No response recorded.{RESET}\n")
        return

    # Cognitive Profiling
    print(f"\n  {DIM}Analyzing your cognitive pattern...{RESET}")
    mem_cfg = get_memory_config()
    profile = profile_response(
        user_response=user_response,
        article_title=article["title"],
        article_summary=article.get("summary", ""),
        provider_config=mem_cfg,
    )

    # 显示画像标签
    print(f"\n  {BOLD}{MAGENTA}[Cognitive Profile]{RESET}")
    print(f"    Preference:  {', '.join(profile.core_preference)}")
    print(f"    Reasoning:   {profile.reasoning_style}")
    print(f"    Tone:        {profile.emotional_tone}")
    print(f"    Stance:      {profile.stance_summary}")
    print(f"    Keywords:    {', '.join(profile.topic_keywords)}")

    # 保存记忆
    memory_id = save_memory(
        article_id=args.item,
        article_title=article["title"],
        user_response=user_response,
        profile=profile,
    )
    print(f"  {DIM}Memory saved (#{memory_id}){RESET}")

    # Echo Location
    related = find_related_memories(user_response, exclude_id=memory_id)
    current_tags = {
        "core_preference": profile.core_preference,
        "reasoning_style": profile.reasoning_style,
        "emotional_tone": profile.emotional_tone,
        "stance_summary": profile.stance_summary,
    }
    echo = generate_echo_report(user_response, current_tags, related, provider_config=mem_cfg)
    print(format_echo_report(echo, colors=COLORS))


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
                "⭐ 收藏夹",
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
            elif action.startswith("📊"):
                _interactive_eval()
        except KeyboardInterrupt:
            print(f"\n{DIM}操作已取消{RESET}\n")
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


def _interactive_list(favorites_only: bool = False):
    """交互式 List - 整合 View 和 Brief 功能。"""
    articles = list_articles(limit=20, favorites_only=favorites_only)
    if not articles:
        if favorites_only:
            print(f"\n{YELLOW}收藏夹还没有文章。可以先在 Browse 里收藏高价值文章。{RESET}\n")
        else:
            print(f"\n{YELLOW}数据库中暂无文章。请先运行 Scout。{RESET}\n")
        return

    # 构建选择列表（纯文本，无颜色代码）
    choices = []
    for a in articles:
        score_bar = _format_score_bar_plain(a['total_score'])
        # 截断标题避免过长
        title = a['title'][:50] + "..." if len(a['title']) > 50 else a['title']
        favorite_mark = "[收藏] " if a.get("is_favorite") else ""
        choice_text = f"[ID:{a['id']}] {favorite_mark}{title} {score_bar}"
        choices.append(choice_text)
    
    choices.append(questionary.Separator())
    choices.append("🔙 返回主菜单")
    
    selected = questionary.select(
        "选择文章：",
        choices=choices,
        style=custom_style
    ).ask()
    
    if not selected or selected.startswith("🔙"):
        return
    
    # 提取文章 ID
    article_id = int(selected.split("]")[0].split(":")[1])
    article = get_article(article_id)
    
    if not article:
        print(f"\n{RED}文章不存在。{RESET}\n")
        return
    
    # 显示文章基本信息（这里可以用颜色）
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {article['title']}")
    print(f"{'=' * 60}{RESET}")
    print(f"  {DIM}来源: {article['source']} | 评分: {article['total_score']:.1f}/10{RESET}")
    print(f"  {DIM}链接: {article['url']}{RESET}\n")
    print(f"  {MAGENTA}摘要: {article.get('summary', '')}{RESET}\n")
    
    # 选择操作
    action = questionary.select(
        "你想做什么？",
        choices=[
            "📖 生成导读精炼版",
            "🌐 查看原文（浏览器打开）",
            "🏛️  发起议事厅讨论",
            "⭐ 取消收藏" if article.get("is_favorite") else "⭐ 收藏到收藏夹",
            questionary.Separator(),
            "🔙 返回文章列表"
        ],
        style=custom_style
    ).ask()
    
    if not action or action.startswith("🔙"):
        _interactive_list(favorites_only=favorites_only)  # 递归返回列表
        return
    
    if action.startswith("📖"):
        _show_brief(article)
        _interactive_list(favorites_only=favorites_only)  # 操作完成后返回列表
    elif action.startswith("🌐"):
        _open_in_browser(article)
        _interactive_list(favorites_only=favorites_only)
    elif action.startswith("🏛️"):
        _start_council(article)
    elif action.startswith("⭐"):
        set_article_favorite(article["id"], favorite=not article.get("is_favorite"))
        status = "已收藏" if not article.get("is_favorite") else "已取消收藏"
        print(f"\n{GREEN if not article.get('is_favorite') else YELLOW}{status}: {article['title']}{RESET}\n")
        input(f"{DIM}按 Enter 继续...{RESET}")
        _interactive_list(favorites_only=favorites_only)


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
    print(f"\n{BOLD}Starting Council discussion for: {article['title']}{RESET}\n")

    cfg = get_council_config()
    result = run_council(
        title=article["title"],
        summary=article.get("summary", ""),
        content=article.get("summary", ""),
        provider_config=cfg,
    )

    try:
        save_debate(result, article_id=article.get("id"))
    except Exception:
        logging.getLogger(__name__).exception("Failed to persist debate state")

    output = format_council_result(result, colors=COLORS)
    print(output)

    # 收集用户回应
    print(f"  {BOLD}{CYAN}{'=' * 60}")
    print(f"  [Your Turn] -- Share Your Thoughts")
    print(f"  {'=' * 60}{RESET}")
    print(f"  {DIM}(Enter your response. Press Enter twice to submit, or type 'skip' to skip){RESET}\n")

    lines = []
    while True:
        try:
            line = input(f"  {GREEN}>{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() == "skip":
            print(f"\n  {DIM}Skipped. Your thoughts are your own.{RESET}\n")
            return
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)

    user_response = "\n".join(lines).strip()
    if not user_response:
        print(f"\n  {DIM}No response recorded.{RESET}\n")
        return

    # Cognitive Profiling
    print(f"\n  {DIM}Analyzing your cognitive pattern...{RESET}")
    mem_cfg = get_memory_config()
    profile = profile_response(
        user_response=user_response,
        article_title=article["title"],
        article_summary=article.get("summary", ""),
        provider_config=mem_cfg,
    )

    # 显示画像标签
    print(f"\n  {BOLD}{MAGENTA}[Cognitive Profile]{RESET}")
    print(f"    Preference:  {', '.join(profile.core_preference)}")
    print(f"    Reasoning:   {profile.reasoning_style}")
    print(f"    Tone:        {profile.emotional_tone}")
    print(f"    Stance:      {profile.stance_summary}")
    print(f"    Keywords:    {', '.join(profile.topic_keywords)}")

    # 保存记忆
    memory_id = save_memory(
        article_id=article['id'],
        article_title=article["title"],
        user_response=user_response,
        profile=profile,
    )
    print(f"  {DIM}Memory saved (#{memory_id}){RESET}")

    # Echo Location
    related = find_related_memories(user_response, exclude_id=memory_id)
    current_tags = {
        "core_preference": profile.core_preference,
        "reasoning_style": profile.reasoning_style,
        "emotional_tone": profile.emotional_tone,
        "stance_summary": profile.stance_summary,
    }
    echo = generate_echo_report(user_response, current_tags, related, provider_config=mem_cfg)
    print(format_echo_report(echo, colors=COLORS))
    
    input(f"\n{DIM}按 Enter 继续...{RESET}")


def _interactive_view():
    """交互式 View - 已整合到 List 中。"""
    print(f"\n{YELLOW}此功能已整合到 List 中，请使用 List 功能。{RESET}\n")


def _interactive_brief():
    """交互式 Brief - 已整合到 List 中。"""
    print(f"\n{YELLOW}此功能已整合到 List 中，请使用 List 功能。{RESET}\n")


def _interactive_council():
    """交互式 Council - 已整合到 List 中。"""
    print(f"\n{YELLOW}此功能已整合到 List 中，请使用 List 功能选择文章后发起讨论。{RESET}\n")


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

    result = cleanup_old_articles(
        retention_days=days,
        dry_run=dry_run,
        keep_discussed=not include_discussed,
    )

    count = result["candidate_count"] if dry_run else result["deleted_count"]
    print(f"\n{BOLD}{CYAN}[Article Cleanup]{RESET}")
    print(f"  {'可清理' if dry_run else '已清理'}: {count} 篇")
    print(f"  {DIM}收藏文章会始终保留。{RESET}\n")

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
    list_parser.set_defaults(func=cmd_list)

    # favorite
    favorite_parser = subparsers.add_parser("favorite", help="Add an article to favorites")
    favorite_parser.add_argument(
        "--item", type=int, required=True, help="Article ID (use 'list' to see IDs)"
    )
    favorite_parser.add_argument(
        "--note", type=str, help="Optional note for why this article is worth keeping"
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
    favorites_parser.set_defaults(func=cmd_favorites)

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
