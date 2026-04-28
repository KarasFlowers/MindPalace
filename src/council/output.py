"""Council 结果格式化输出。

接受 DebateState（新）或任何拥有 `.critic / .synthesizer / .mentor` 属性的对象（旧）。
DebateState 额外展示：难度徽章、反驳轮数、终止原因、Judge 共识。
"""


def _fmt_header_badges(result, colors: dict) -> list[str]:
    """在标题行后追加难度 / 轮次 / 终止原因徽章。"""
    BOLD = colors.get("BOLD", "")
    DIM = colors.get("DIM", "")
    CYAN = colors.get("CYAN", "")
    YELLOW = colors.get("YELLOW", "")
    GREEN = colors.get("GREEN", "")
    RED = colors.get("RED", "")
    RESET = colors.get("RESET", "")

    difficulty = getattr(result, "difficulty", None)
    active_roles = getattr(result, "active_roles", None) or []
    round_idx = getattr(result, "round_idx", 0)
    terminated_by = getattr(result, "terminated_by", "")

    if difficulty is None and not active_roles and round_idx == 0 and not terminated_by:
        return []

    diff_color = {"easy": GREEN, "medium": YELLOW, "hard": RED}.get(difficulty, CYAN)
    end_color = {"converged": GREEN, "max_rounds": YELLOW}.get(terminated_by, DIM)

    badges = []
    if difficulty:
        badges.append(f"{diff_color}[{difficulty.upper()}]{RESET}")
    if active_roles:
        badges.append(f"{DIM}roles={'/'.join(active_roles)}{RESET}")
    badges.append(f"{DIM}rebuttal_rounds={round_idx}{RESET}")
    if terminated_by:
        badges.append(f"{end_color}ended={terminated_by}{RESET}")

    return [f"  {BOLD}Council Badges:{RESET} " + "  ".join(badges)]


def _fmt_judge_section(result, colors: dict) -> list[str]:
    """展示 Judge 的最终共识。"""
    BOLD = colors.get("BOLD", "")
    DIM = colors.get("DIM", "")
    MAGENTA = colors.get("MAGENTA", "")
    CYAN = colors.get("CYAN", "")
    GREEN = colors.get("GREEN", "")
    YELLOW = colors.get("YELLOW", "")
    RESET = colors.get("RESET", "")

    consensus = getattr(result, "consensus", None)
    if not consensus:
        return []

    lines = [
        f"\n  {BOLD}{MAGENTA}[The Judge] -- Consensus{RESET}",
    ]
    headline = consensus.get("headline", "")
    if headline:
        lines.append(f"    {BOLD}{CYAN}▸ {headline}{RESET}")

    key_points = consensus.get("key_points") or []
    if key_points:
        lines.append(f"    {DIM}核心洞察:{RESET}")
        for kp in key_points:
            lines.append(f"      {GREEN}•{RESET} {kp}")

    tensions = consensus.get("remaining_tensions") or []
    if tensions:
        lines.append(f"    {DIM}未解决的张力:{RESET}")
        for tt in tensions:
            lines.append(f"      {YELLOW}⚠{RESET} {tt}")

    stance = consensus.get("recommended_stance", "")
    if stance:
        lines.append(f"\n    {BOLD}推荐立场:{RESET} {stance}")

    error = consensus.get("error")
    if error:
        lines.append(f"    {DIM}(judge 调用失败: {error}){RESET}")

    lines.append(f"  {DIM}{'─' * 54}{RESET}")
    return lines


def format_council_result(result, colors: dict | None = None) -> str:
    """将 DebateState 或 CouncilResult 格式化为终端富文本。

    Args:
        result: DebateState（新）或 CouncilResult（旧别名）。
        colors: ANSI 颜色 dict，包含 BOLD, DIM, CYAN, YELLOW, GREEN, RED, MAGENTA, RESET。
    """
    c = colors or {}
    BOLD = c.get("BOLD", "")
    DIM = c.get("DIM", "")
    CYAN = c.get("CYAN", "")
    YELLOW = c.get("YELLOW", "")
    GREEN = c.get("GREEN", "")
    RED = c.get("RED", "")
    MAGENTA = c.get("MAGENTA", "")
    RESET = c.get("RESET", "")

    lines = []
    sep = f"  {DIM}{'─' * 54}{RESET}"

    # Header
    lines.append(f"\n{BOLD}{CYAN}{'=' * 60}")
    lines.append(f"  [MindPalace Council] Discussion Result")
    lines.append(f"{'=' * 60}{RESET}")
    lines.append(f"  {BOLD}{result.article_title}{RESET}")
    lines.append(f"  {DIM}> {result.article_summary}{RESET}")
    lines.extend(_fmt_header_badges(result, c))
    lines.append("")

    # === Critic ===
    critic = result.critic or {}
    if critic:
        lines.append(f"  {BOLD}{RED}[The Critic] -- Vulnerabilities{RESET}")
        vulns = critic.get("vulnerabilities", [])
        for v in vulns:
            severity = v.get("severity", "?")
            sev_color = RED if severity == "high" else YELLOW if severity == "medium" else DIM
            lines.append(f"    {sev_color}[{str(severity).upper()}]{RESET} {v.get('assumption', '')}")
            lines.append(f"         {DIM}Counter: {v.get('counter', '')}{RESET}")
        counter_ex = critic.get("missing_counterexample", "")
        if counter_ex:
            lines.append(f"    {MAGENTA}Counterexample: {counter_ex}{RESET}")
        verdict = critic.get("verdict", "")
        if verdict:
            lines.append(f"    {BOLD}Verdict: {verdict}{RESET}")
        lines.append(sep)

    # === Synthesizer ===
    synth = result.synthesizer or {}
    if synth:
        lines.append(f"\n  {BOLD}{GREEN}[The Synthesizer] -- Cross-Domain Connections{RESET}")
        conns = synth.get("connections", [])
        for conn in conns:
            lines.append(f"    {CYAN}[{conn.get('domain', '?')}]{RESET} {conn.get('analogy', '')}")
            lines.append(f"         {DIM}Insight: {conn.get('insight', '')}{RESET}")
        synthesis = synth.get("synthesis", "")
        if synthesis:
            lines.append(f"    {BOLD}Synthesis: {synthesis}{RESET}")
        lines.append(sep)

    # === Mentor ===
    mentor = result.mentor or {}
    if mentor:
        lines.append(f"\n  {BOLD}{YELLOW}[The Mentor] -- Socratic Questions{RESET}")
        questions = mentor.get("questions", [])
        for q in questions:
            level = q.get("level", "")
            lines.append(f"    {MAGENTA}[{level}]{RESET} {q.get('question', '')}")
        provocation = mentor.get("provocation", "")
        if provocation:
            lines.append(f"\n    {BOLD}{RED}* {provocation}{RESET}")
        lines.append(sep)

    # === Judge consensus ===
    lines.extend(_fmt_judge_section(result, c))

    lines.append("")
    return "\n".join(lines)
