"""心智漫游交互菜单。"""

from __future__ import annotations

try:
    import questionary
except ImportError:  # pragma: no cover - app.py 已有主入口提示
    questionary = None

from src.inquiry.library import InquiryLibraryError
from src.inquiry.session import run_inquiry_session
from src.memory.store import get_memories_by_source
from src.resolve.engine import run_repl

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


def _require_questionary():
    if questionary is None:
        raise RuntimeError("Interactive mode requires 'questionary'. Run `pip install -e .` first.")


def _print_inquiry_memories(limit: int = 30) -> None:
    """展示心智漫游回答——按问题卡分组显示演化轨迹（Axiomind cross-date pattern）。"""
    memories = get_memories_by_source(limit=limit)
    if not memories:
        print(f"\n{YELLOW}还没有心智漫游回答。{RESET}\n")
        return

    # 按 source_id（问题卡）分组
    groups: dict[str, list[dict]] = {}
    for item in memories:
        sid = item.get("source_id") or "(无问题卡)"
        groups.setdefault(sid, []).append(item)

    print(f"\n{BOLD}{CYAN}[心智漫游回答轨迹]{RESET}")
    print(f"{DIM}共 {len(memories)} 条回答，{len(groups)} 个主题{RESET}\n")

    for sid, items in groups.items():
        # 每组按时间正序（旧→新）展示变迁
        items.sort(key=lambda m: m.get("created_at") or "")
        first = items[0]
        label = _KIND_LABELS.get(first.get("source_type"), first.get("source_type", "unknown"))
        title = first.get("article_title", "").replace("[心智漫游] ", "")

        print(f"  {BOLD}{MAGENTA}▍ {title}{RESET} {DIM}[{label}] × {len(items)} 次{RESET}")
        for i, item in enumerate(items):
            created = (item.get("created_at") or "")[:10]
            response = (item.get("user_response") or "").replace("\n", " ").strip()
            stance = (item.get("stance_summary") or "").replace("\n", " ").strip()
            arrow = f"{GREEN}→{RESET}" if i == 0 else f"{YELLOW}↻{RESET}"
            line = f"    {arrow} {DIM}{created}{RESET} "
            if stance:
                line += f"{DIM}立场：{stance[:40]}{RESET}"
            print(line)
            if response:
                print(f"      {DIM}{response[:90]}{RESET}")
        print(f"  {DIM}{'─' * 50}{RESET}\n")


def run_inquiry_menu() -> None:
    """运行心智漫游主菜单。"""
    _require_questionary()

    while True:
        action = questionary.select(
            "心智漫游：",
            choices=[
                "🪞 认识自己",
                "🧠 哲思问题",
                "🧪 思想实验",
                "🎲 随机一题",
                "💬 自由对话",
                "📜 回看我的回答",
                questionary.Separator(),
                "🔙 返回主菜单",
            ],
        ).ask()

        if not action or action.startswith("🔙"):
            return

        try:
            if action.startswith("🪞"):
                run_inquiry_session(kind="self")
            elif action.startswith("🧠"):
                run_inquiry_session(kind="philosophy")
            elif action.startswith("🧪"):
                run_inquiry_session(kind="thought_experiment")
            elif action.startswith("🎲"):
                run_inquiry_session(kind=None)
            elif action.startswith("💬"):
                run_repl()
            elif action.startswith("📜"):
                _print_inquiry_memories()
                try:
                    input(f"{DIM}按 Enter 继续...{RESET}")
                except EOFError:
                    return
        except InquiryLibraryError as exc:
            print(f"\n{YELLOW}{exc}{RESET}\n")
        except KeyboardInterrupt:
            print(f"\n{DIM}操作已取消{RESET}\n")
            return
