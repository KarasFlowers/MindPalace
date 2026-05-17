"""Council 辩论状态机。

流程:

    ROUTING  --(router.route)-->  OPENING
    OPENING  --(各角色首轮陈述)-->  REBUTTAL (若 active_roles >= 2) 或直接跳到 JUDGING
    REBUTTAL 循环 (max_rebuttal_rounds):
        ├── 每轮让所有角色针对对手最新观点发言
        ├── 轮次达到 max_rebuttal_rounds 时 force_closing=True（akashic 式强制落地）
        └── 每轮结束后由 Judge midcheck 评估分歧度，低于阈值则终止
    JUDGING  --(judge.finalize)-->  DONE

入口 `run_council` 保留原有签名，对外返回 DebateState。`CouncilResult` 作为类型别名
保留给依赖旧名的调用方。
"""

from __future__ import annotations

import logging

from src.config import (
    CONVERGE_THRESHOLD,
    MAX_REBUTTAL_ROUNDS,
    get_council_config,
)
from src.council.judge import finalize as judge_finalize
from src.council.judge import midcheck as judge_midcheck
from src.council.rebuttal import build_opening_prompt, build_rebuttal_prompt
from src.council.roles import get_role, TOOL_ENABLED_ROLES
from src.council.router import route as route_difficulty
from src.council.state import DebateState, Phase, Turn
from src.llm.client import chat_json, chat_with_tools
from src.obs import span

logger = logging.getLogger(__name__)


# 向后兼容：旧代码 `from src.council.flow import CouncilResult` 仍可工作
CouncilResult = DebateState


# ---------------- private ----------------

def _run_role_turn(
    state: DebateState,
    role_key: str,
    council_cfg: dict,
    phase: Phase,
    force_closing: bool,
) -> tuple[dict, int, list]:
    """执行单个角色的一次发言。

    Returns:
        (content_dict, tool_calls_used, tool_log)
    """
    role = get_role(role_key)
    if phase is Phase.OPENING:
        user_prompt = build_opening_prompt(state, role_key)
    else:
        user_prompt = build_rebuttal_prompt(state, role_key, force_closing=force_closing)

    logger.info(
        "[Council] %s speaking (phase=%s, round=%s, force_closing=%s)",
        role["name"], phase.value, state.round_idx, force_closing,
    )

    with span(
        f"council.role.{role_key}",
        role=role_key,
        phase=phase.value,
        round_idx=state.round_idx,
        force_closing=force_closing,
    ):

      # 工具启用的角色走 chat_with_tools
      if role_key in TOOL_ENABLED_ROLES and not force_closing:
          try:
              from src.tools import TOOLS, to_openai_schema
              if TOOLS:
                  result = chat_with_tools(
                      system_prompt=role["prompt"],
                      user_prompt=user_prompt,
                      tools_schema=to_openai_schema(),
                      tool_executor=TOOLS,
                      max_tool_calls=3,
                      provider_config=council_cfg,
                  )
                  content_str = result["content"]
                  # 尝试解析为 JSON
                  try:
                      import json
                      content = json.loads(content_str)
                  except (json.JSONDecodeError, TypeError):
                      content = {"raw_response": content_str}
                  return content, result["tool_calls_used"], result["tool_log"]
          except Exception as exc:
              logger.warning(
                  "[Council] tool-use path failed for %s, falling back to chat_json: %s",
                  role_key, exc,
              )

      # 普通路径（Mentor 或 fallback）
      content = chat_json(role["prompt"], user_prompt, provider_config=council_cfg)
      return content, 0, []


def _append_turn(
    state: DebateState,
    role_key: str,
    phase: Phase,
    content: dict,
    force_closing: bool,
    tool_calls_used: int = 0,
    tool_log: list | None = None,
) -> None:
    state.turns.append(
        Turn(
            role_key=role_key,
            round_idx=state.round_idx,
            phase=phase,
            content=content,
            force_closing=force_closing,
            tool_calls_used=tool_calls_used,
            tool_log=tool_log or [],
        )
    )


# ---------------- public ----------------

def run_council(
    title: str,
    summary: str,
    content: str,
    provider_config: dict | None = None,
    max_rebuttal_rounds: int | None = None,
    converge_threshold: float | None = None,
) -> DebateState:
    """运行一次完整的议事厅辩论。

    Args:
        title: 文章标题。
        summary: 文章摘要（通常来自 Scout）。
        content: 文章正文（清洗后）。
        provider_config: 辩论角色使用的 Provider（即 Council 档）。Router 和 Judge
            各自有独立 provider 配置，不受此参数影响。
        max_rebuttal_rounds: 覆盖默认的最大反驳轮数（来自 .env MAX_REBUTTAL_ROUNDS）。
        converge_threshold: 覆盖默认的分歧度阈值（来自 .env CONVERGE_THRESHOLD）。

    Returns:
        DebateState：包含完整辩论过程与 Judge 共识结果。
    """
    logger.info("=== Council Session Start: %s ===", title[:50])

    _debate_span = span("council.debate", article_title=title[:80])
    _s = _debate_span.__enter__()
    try:
        state = DebateState(
        article_title=title,
        article_summary=summary,
        article_content=content,
        max_rebuttal_rounds=(
            MAX_REBUTTAL_ROUNDS if max_rebuttal_rounds is None else max_rebuttal_rounds
        ),
        converge_threshold=(
            CONVERGE_THRESHOLD if converge_threshold is None else converge_threshold
        ),
    )

    council_cfg = provider_config or get_council_config()

    # --- Phase 0: Routing ---
    difficulty, active_roles, reasoning = route_difficulty(title, summary)
    state.difficulty = difficulty
    state.active_roles = active_roles
    state.routing_reasoning = reasoning

    # --- Phase 1: Opening ---
    state.phase = Phase.OPENING
    for role_key in state.active_roles:
        content_json, tc_used, tc_log = _run_role_turn(
            state, role_key, council_cfg, Phase.OPENING, force_closing=False
        )
        _append_turn(state, role_key, Phase.OPENING, content_json, force_closing=False,
                     tool_calls_used=tc_used, tool_log=tc_log)

    # --- Phase 2: Rebuttal loop（仅当多角色且允许反驳时）---
    if len(state.active_roles) >= 2 and state.max_rebuttal_rounds > 0:
        state.phase = Phase.REBUTTAL
        while state.round_idx < state.max_rebuttal_rounds:
            state.round_idx += 1
            force_closing = state.round_idx == state.max_rebuttal_rounds

            for role_key in state.active_roles:
                content_json, tc_used, tc_log = _run_role_turn(
                    state, role_key, council_cfg, Phase.REBUTTAL,
                    force_closing=force_closing,
                )
                _append_turn(
                    state, role_key, Phase.REBUTTAL, content_json,
                    force_closing=force_closing,
                    tool_calls_used=tc_used, tool_log=tc_log,
                )

            if force_closing:
                state.terminated_by = "max_rounds"
                break

            check = judge_midcheck(state)
            state.disagreement_score = check["disagreement_score"]
            if not check["should_continue"]:
                state.terminated_by = "converged"
                break
    else:
        state.terminated_by = (
            "single_role" if len(state.active_roles) == 1 else "no_rebuttal"
        )

    # --- Phase 3: Judge finalize ---
    state.phase = Phase.JUDGING
    state.consensus = judge_finalize(state)
    state.phase = Phase.DONE

    logger.info(
        "=== Council Session Done === [terminated_by=%s rounds=%d]",
        state.terminated_by, state.round_idx,
    )

        # 追加辩论结果属性
        try:
            _s.set_attribute("difficulty", state.difficulty or "")
            _s.set_attribute("terminated_by", state.terminated_by or "")
            _s.set_attribute("total_rounds", state.round_idx)
            _s.set_attribute("active_roles", ",".join(state.active_roles))
            total_tc = sum(t.tool_calls_used for t in state.turns)
            _s.set_attribute("total_tool_calls", total_tc)
        except Exception:
            pass

        return state
    finally:
        _debate_span.__exit__(None, None, None)
