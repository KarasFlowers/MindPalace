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
RESET = "\033[0m"

_KIND_LABELS = {
    "self": "认识自己",
    "philosophy": "哲思问题",
    "thought_experiment": "思想实验",
}


def _require_questionary():
    if questionary is None:
        raise RuntimeError("Interactive mode requires 'questionary'. Run `pip install -e .` first.")


def _print_inquiry_memories(limit: int = 20) -> None:
    memories = get_memories_by_source(limit=limit)
    if not memories:
        print(f"\n{YELLOW}还没有心智漫游回答。{RESET}\n")
        return

    print(f"\n{BOLD}{CYAN}[心智漫游回答]{RESET}\n")
    for item in memories:
        label = _KIND_LABELS.get(item.get("source_type"), item.get("source_type", "unknown"))
        title = item.get("article_title", "").replace("[心智漫游] ", "")
        response = (item.get("user_response") or "").replace("\n", " ")
        print(f"  {BOLD}#{item['id']} [{label}] {title}{RESET}")
        print(f"  {DIM}{item.get('created_at', '')[:16]} | {response[:100]}{RESET}\n")


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
