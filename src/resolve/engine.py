"""Resolve 引擎。提供一个支持长程对话、角色切换的 REPL 接口。"""

import json
import logging
import uuid
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

ROLE_COLORS = {
    "critic": RED,
    "synthesizer": GREEN,
    "mentor": YELLOW,
}


def _truncate_text(text: str | None, limit: int = 100) -> str:
    content = (text or "").strip().replace("\n", " ")
    if len(content) <= limit:
        return content
    return content[: limit - 3].rstrip() + "..."


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_structured_response(text: str) -> dict | None:
    """尽量把角色输出解析成 JSON，失败则返回 None。"""
    candidate = _strip_code_fence(text)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_role_highlight(role_key: str, response: str) -> str:
    parsed = _parse_structured_response(response)
    if parsed:
        if role_key == "critic":
            vulnerabilities = parsed.get("vulnerabilities") or []
            first = vulnerabilities[0] if vulnerabilities else {}
            return (
                first.get("assumption")
                or first.get("counter")
                or parsed.get("verdict")
                or response
            )
        if role_key == "synthesizer":
            connections = parsed.get("connections") or []
            first = connections[0] if connections else {}
            return (
                first.get("insight")
                or first.get("analogy")
                or parsed.get("synthesis")
                or response
            )
        if role_key == "mentor":
            questions = parsed.get("questions") or []
            first = questions[0] if questions else {}
            return (
                first.get("question")
                or parsed.get("provocation")
                or response
            )

    lines = [line.strip("- ").strip() for line in response.splitlines() if line.strip()]
    return lines[0] if lines else response


def _print_role_response(role_key: str, response: str):
    """把角色输出尽量转换成可读结构，而不是直接把 JSON 扔给用户。"""
    role = get_role(role_key)
    color = ROLE_COLORS.get(role_key, CYAN)
    parsed = _parse_structured_response(response)

    print(f"\n{BOLD}{color}{role['name']}{RESET}")
    if not parsed:
        print(response)
        return

    if role_key == "critic":
        vulnerabilities = parsed.get("vulnerabilities") or []
        for item in vulnerabilities[:3]:
            severity = str(item.get("severity", "?")).upper()
            print(f"  - [{severity}] {item.get('assumption', '')}")
            if item.get("counter"):
                print(f"    {DIM}崩塌条件: {item['counter']}{RESET}")
        if parsed.get("missing_counterexample"):
            print(f"  {MAGENTA}反例: {parsed['missing_counterexample']}{RESET}")
        if parsed.get("verdict"):
            print(f"  {DIM}一句话判断: {parsed['verdict']}{RESET}")
    elif role_key == "synthesizer":
        connections = parsed.get("connections") or []
        for item in connections[:3]:
            print(f"  - [{item.get('domain', '?')}] {item.get('analogy', '')}")
            if item.get("insight"):
                print(f"    {DIM}启发: {item['insight']}{RESET}")
        if parsed.get("synthesis"):
            print(f"  {DIM}综合洞察: {parsed['synthesis']}{RESET}")
    elif role_key == "mentor":
        questions = parsed.get("questions") or []
        for item in questions[:3]:
            print(f"  - [{item.get('level', '追问')}] {item.get('question', '')}")
        if parsed.get("provocation"):
            print(f"  {MAGENTA}刺激点: {parsed['provocation']}{RESET}")
    else:
        print(response)


def _print_council_digest(results: dict[str, str]):
    """默认只展示一层综合摘要，减轻每轮三角色齐发的压迫感。"""
    print(f"\n{BOLD}{CYAN}[Council Composite]{RESET}")
    print(f"  {RED}批判者提醒你{RESET} {_truncate_text(_extract_role_highlight('critic', results.get('critic', '')), 110)}")
    print(f"  {GREEN}连接者补了一层{RESET} {_truncate_text(_extract_role_highlight('synthesizer', results.get('synthesizer', '')), 110)}")
    print(f"  {YELLOW}导师最想追问{RESET} {_truncate_text(_extract_role_highlight('mentor', results.get('mentor', '')), 110)}")
    print(f"  {DIM}你可以继续反驳一个前提、讲一个例子，或让我展开某个角色。{RESET}")


def _offer_role_expansion(results: dict[str, str]):
    """按需展开角色，避免每一轮都输出三大段。"""
    while True:
        print(
            f"{DIM}需要我展开哪个角色？"
            f" [1] Critic [2] Synthesizer [3] Mentor [4] 全部原文 [Enter] 继续{RESET}"
        )
        try:
            choice = input(f"{GREEN}>{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            return

        if choice in ("", "c", "continue"):
            return
        if choice == "1":
            _print_role_response("critic", results.get("critic", ""))
        elif choice == "2":
            _print_role_response("synthesizer", results.get("synthesizer", ""))
        elif choice == "3":
            _print_role_response("mentor", results.get("mentor", ""))
        elif choice == "4":
            for role_key in get_discussion_order():
                _print_role_response(role_key, results.get(role_key, ""))
        else:
            print(f"{YELLOW}请输入 1/2/3/4，或直接回车继续。{RESET}")

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
        except ValueError as e:
            # 拼错角色名不应让整个程序退出；返回让调用方决定下一步
            print(f"  {RED}无法加载角色 '{role_key}': {e}{RESET}")
            print(f"  {DIM}可用角色：critic / synthesizer / mentor{RESET}\n")
            return
    else:
        print(f"  当前正在与: The Council (智库模式) 对话")
        print(f"  {DIM}我会先给你综合回应，再按需展开具体角色。{RESET}")

    if session.history:
        print(f"  {DIM}[恢复历史会话，共 {len(session.history)//2} 轮对话]{RESET}")

    print(f"  会话ID: {DIM}{session.session_id[:8]}...{RESET}")
    print(f"  {DIM}输入 'exit' 或 'quit' 退出，Ctrl+C 也可中断。{RESET}")
    print(f"  {DIM}更容易聊开的方式：反驳一个前提、讲一个案例、或请我继续追问。{RESET}")
    print("="*50 + "\n")

    while True:
        try:
            # 统一提示符为 >（与 council / inquiry / daily 一致）
            user_input = input(f"\n{GREEN}>{RESET} ").strip()
            if user_input.lower() in ("exit", "quit"):
                print(f"{DIM}再见！{RESET}")
                break
            if not user_input:
                continue

            if mode == "single":
                resp = session.speak_to_role(user_input, role_key)
                _print_role_response(role_key, resp)
            else:
                resps = session.speak_to_council(user_input)
                _print_council_digest(resps)
                _offer_role_expansion(resps)

        except KeyboardInterrupt:
            print(f"\n{DIM}再见！{RESET}")
            break
        except Exception as e:
            logger.exception("Error in REPL")
            print(f"\n{RED}[Error] System encountered an issue: {e}{RESET}")


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
