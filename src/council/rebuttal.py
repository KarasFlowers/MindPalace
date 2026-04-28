"""反驳轮 prompt 构造。

两个核心职责：
1. opening 轮——保留旧版 `_build_user_prompt` 的"后序角色看到前序角色"行为。
2. rebuttal 轮——让角色针对对手最新发言做具体回应，并在 `force_closing=True`
   时注入 akashic 式的"强制落地"约束，防止死循环。
"""

from __future__ import annotations

import json

from src.council.state import DebateState, Turn


# ---------------- opening prompt ----------------

def build_opening_prompt(state: DebateState, role_key: str) -> str:
    """各角色首轮陈述的 prompt。

    为了兼容原有行为：Synthesizer 能看到 Critic 的分析；
    Mentor 能看到 Critic 和 Synthesizer 的分析。
    只有在该角色实际在 active_roles 中且已发过言时才注入。
    """
    parts = [
        f"文章标题: {state.article_title}",
        f"文章摘要: {state.article_summary}",
        f"\n文章正文:\n{state.article_content}",
    ]

    def _append_prior(prior_role: str, label: str, summarize: callable) -> None:
        turns = state.turns_of(prior_role)
        if not turns:
            return
        latest = turns[-1].content
        parts.append(f"\n--- {label} 的分析 ---\n{summarize(latest)}")

    if role_key == "synthesizer":
        _append_prior(
            "critic", "Critic",
            lambda c: (
                f"逻辑漏洞: {c.get('vulnerabilities', [])}\n"
                f"反面案例: {c.get('missing_counterexample', '')}\n"
                f"论证评价: {c.get('verdict', '')}"
            ),
        )

    if role_key == "mentor":
        _append_prior(
            "critic", "Critic",
            lambda c: (
                f"逻辑漏洞: {c.get('vulnerabilities', [])}\n"
                f"反面案例: {c.get('missing_counterexample', '')}\n"
                f"论证评价: {c.get('verdict', '')}"
            ),
        )
        _append_prior(
            "synthesizer", "Synthesizer",
            lambda s: (
                f"跨界连接: {s.get('connections', [])}\n"
                f"综合洞察: {s.get('synthesis', '')}"
            ),
        )

    return "\n".join(parts)


# ---------------- rebuttal prompt ----------------

def _summarize_turn(turn: Turn) -> str:
    """把 Turn 压成一段可读文字，供反驳上下文使用。"""
    c = turn.content
    # 按角色常见字段优先抽取，否则全文 JSON 序列化
    highlight = (
        c.get("verdict")
        or c.get("synthesis")
        or c.get("provocation")
        or ""
    )
    if highlight:
        return str(highlight)
    return json.dumps(c, ensure_ascii=False)[:500]


def build_rebuttal_prompt(
    state: DebateState,
    role_key: str,
    force_closing: bool,
) -> str:
    """构造反驳轮 prompt。

    Args:
        state: 当前辩论状态，已经包含本轮之前所有 Turn。
        role_key: 当前发言者。
        force_closing: 如果为 True，注入强制落地约束（仿 akashic 的 step N-1）。
    """
    # 取出**除自己以外**其他角色在本轮之前的最近一次发言
    opponents_latest: list[Turn] = []
    for opp_key in state.active_roles:
        if opp_key == role_key:
            continue
        opp_turns = state.turns_of(opp_key)
        if opp_turns:
            opponents_latest.append(opp_turns[-1])

    parts = [
        f"文章标题: {state.article_title}",
        f"文章摘要: {state.article_summary}",
        f"\n--- 当前是第 {state.round_idx} 轮反驳 ---",
    ]

    if opponents_latest:
        parts.append("\n对手最新观点：")
        for t in opponents_latest:
            parts.append(f"\n[{t.role_key}] 刚才说:\n{_summarize_turn(t)}")

    # 自己的历史发言，帮助保持立场一致性
    self_turns = state.turns_of(role_key)
    if self_turns:
        parts.append("\n你（本角色）此前已表达过的观点摘要：")
        for t in self_turns:
            label = "opening" if t.round_idx == 0 else f"round {t.round_idx}"
            parts.append(f"  - [{label}] {_summarize_turn(t)[:160]}")

    if force_closing:
        parts.append(
            "\n⚠️ 本轮是最后一次发言（强制收尾）。禁止引入任何新论点、新案例或新类比。你只能：\n"
            "  1. 针对对手最新观点做一次简短的最终回应（≤120 字）；\n"
            "  2. 将你本次辩论的最终立场浓缩进你角色 schema 的既有字段中，不要新增字段。\n"
            "如果继续追加新论点，将被视为违规。"
        )
    else:
        parts.append(
            "\n请保持你的角色特征，针对对手最新观点做具体、有针对性的反驳或推进。"
            "继续使用你角色 schema 规定的 JSON 字段输出，可以在字段内容中加入对对手观点的回应。"
        )

    return "\n".join(parts)
