"""Resolve 引擎。提供一个支持长程对话、角色切换的 REPL 接口。"""

import json
import logging
import uuid
import sys
from datetime import datetime, timezone

from src.llm.client import chat
from src.council.roles import get_role, get_discussion_order
from src.config import get_council_config, get_fast_config
from src.storage.db import _get_conn, init_db

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

class ResolveSession:
    def __init__(self, session_id: str | None = None, title: str = "New Session", mode: str = "single"):
        self.session_id = session_id or str(uuid.uuid4())
        self.title = title
        self.mode = mode # "single" or "council"
        self.history = [] # 格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.cfg = get_council_config()
        self._load_history()

    def _load_history(self):
        with _get_conn() as conn:
            row = conn.execute("SELECT history, title, mode FROM chat_sessions WHERE id = ?", (self.session_id,)).fetchone()
            if row:
                self.title = row["title"]
                self.mode = row["mode"]
                self.history = json.loads(row["history"])
    
    def _save_history(self):
        now = datetime.now(timezone.utc).isoformat()
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, title, mode, history, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    history=excluded.history,
                    updated_at=excluded.updated_at
                """,
                (self.session_id, self.title, self.mode, json.dumps(self.history, ensure_ascii=False), now, now)
            )

    def speak_to_role(self, user_text: str, role_key: str) -> str:
        self._compress_history_if_needed()
        role = get_role(role_key)
        
        system_prompt = role["prompt"]
        response = chat(
            system_prompt=system_prompt,
            user_prompt=user_text,
            model=None,
            provider_config=self.cfg,
            history=self.history.copy()
        )

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": response})
        self._save_history()

        return response
    
    def speak_to_council(self, user_text: str) -> dict[str, str]:
        self._compress_history_if_needed()
        results = {}
        current_user_text = user_text
        
        for r_key in get_discussion_order():
            role = get_role(r_key)
            system_prompt = role["prompt"]
            response = chat(
                system_prompt=system_prompt,
                user_prompt=current_user_text,
                provider_config=self.cfg,
                history=self.history.copy()
            )
            results[r_key] = response
            # 拼接前面的回答作为上下文给下一个角色
            current_user_text = f"{current_user_text}\n\n[{role['name']} 的补充看法]:\n{response}"
            
        self.history.append({"role": "user", "content": user_text})
        combined_response = "\n\n".join([f"**{get_role(k)['name']}**:\n{resp}" for k, resp in results.items()])
        self.history.append({"role": "assistant", "content": combined_response})
        self._save_history()

        return results

    # ---- Phase B: history compression ----

    _COMPRESS_PROMPT = (
        "你是会话压缩器。把下面的对话历史压缩成一段摘要（< 500 字），"
        "保留关键论点、用户立场、未解决的问题。不要编造。"
    )
    _COMPRESS_THRESHOLD = 40   # messages (~ 20 rounds)
    _COMPRESS_KEEP_RECENT = 10  # keep last N messages intact

    def _compress_history_if_needed(self):
        """When history exceeds threshold, compress early messages into a summary."""
        if len(self.history) <= self._COMPRESS_THRESHOLD:
            return

        cut = len(self.history) - self._COMPRESS_KEEP_RECENT
        early = self.history[:cut]

        formatted = "\n".join(
            f"{m['role']}: {m['content'][:300]}" for m in early
        )
        try:
            summary = chat(
                system_prompt=self._COMPRESS_PROMPT,
                user_prompt=formatted,
                provider_config=get_fast_config(),
            )
        except Exception as exc:
            logger.warning("History compression failed: %s", exc)
            return

        self.history = [
            {"role": "system", "content": f"[历史摘要] {summary}"},
        ] + self.history[cut:]
        self._save_history()
        logger.info(
            "Compressed %d early messages into summary (%d chars)",
            cut, len(summary),
        )


def list_sessions(limit: int = 20) -> list[dict]:
    """列出所有会话。"""
    init_db()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, mode, updated_at FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: str) -> bool:
    """删除指定会话。"""
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0


def run_repl(role_key: str | None = None, session_id: str | None = None):
    """启动交互式终端。"""
    mode = "single" if role_key else "council"
    session = ResolveSession(session_id=session_id, mode=mode) if session_id else ResolveSession(mode=mode)
    
    print("\n" + "="*50)
    print(" 🧠 MindPalace Resolve Space ".center(50))
    print("="*50)
    
    if mode == "single":
        try:
            role = get_role(role_key)
            print(f"  当前正在与: {role['name']} 对话")
        except Exception as e:
            print(f"无法加载角色 '{role_key}': {e}")
            sys.exit(1)
    else:
        print(f"  当前正在与: The Council (智库模式) 对话")
    
    if session.history:
        print(f"  {DIM}[恢复历史会话，共 {len(session.history)//2} 轮对话]{RESET}")
    
    print(f"  会话ID: {DIM}{session.session_id[:8]}...{RESET}")
    print("  输入 'exit' 或 'quit' 退出。")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("\nYou> ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Good bye!")
                break
            if not user_input:
                continue

            if mode == "single":
                resp = session.speak_to_role(user_input, role_key)
                print(f"\n{get_role(role_key)['name']}>\n{resp}")
            else:
                resps = session.speak_to_council(user_input)
                for k, v in resps.items():
                    print(f"\n{get_role(k)['name']}>\n{v}")
                
        except KeyboardInterrupt:
            print("\nGood bye!")
            break
        except Exception as e:
            logger.exception("Error in REPL")
            print(f"\n[Error] System encountered an issue: {e}")


def run_sessions_list():
    """显示会话列表并允许用户选择恢复。"""
    sessions = list_sessions()
    
    if not sessions:
        print(f"\n{YELLOW}暂无历史会话。使用 'resolve' 开始新对话。{RESET}\n")
        return
    
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  [Resolve Sessions] -- 历史会话")
    print(f"{'=' * 60}{RESET}\n")
    
    for i, s in enumerate(sessions, 1):
        mode_display = "单角色" if s['mode'] == 'single' else "议事厅"
        print(f"  {BOLD}{CYAN}[{i}]{RESET} {s['title']}")
        print(f"       {DIM}ID: {s['id'][:8]}... | 模式: {mode_display} | 更新: {s['updated_at'][:10]}{RESET}")
    
    print(f"\n  {DIM}输入序号恢复会话，或按 Enter 开始新会话{RESET}")
    
    try:
        choice = input(f"\n  {GREEN}>{RESET} ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                selected = sessions[idx]
                print(f"\n  {DIM}恢复会话: {selected['title']}{RESET}\n")
                run_repl(session_id=selected['id'])
            else:
                print(f"\n{RED}无效序号。{RESET}\n")
        elif choice == "":
            run_repl()
    except KeyboardInterrupt:
        print("\n")
