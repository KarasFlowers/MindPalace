"""Council 辩论状态机。

流程（默认 Debate 范式）:

    ROUTING  --(router.route)-->  OPENING
    OPENING  --(各角色首轮陈述)-->  REBUTTAL (若 active_roles >= 2) 或直接跳到 JUDGING
    REBUTTAL 循环 (max_rebuttal_rounds):
        ├── 每轮让所有角色针对对手最新观点发言
        ├── 轮次达到 max_rebuttal_rounds 时 force_closing=True（akashac 式强制落地）
        └── 每轮结束后由 Judge midcheck 评估分歧度，低于阈值则终止
    JUDGING  --(judge.finalize)-->  DONE

讨论流程被抽象为可注册的范式（src/council/paradigms.py + registry.py），
借鉴 MALLM 的 discussion_paradigms 设计。`run_council` 通过 `paradigm` 参数
选择范式，默认为 "debate"（向后兼容）。

入口 `run_council` 保留原有签名，对外返回 DebateState。`CouncilResult` 作为类型别名
保留给依赖旧名的调用方。
"""

from __future__ import annotations

import logging

from src.config import (
    CONVERGE_THRESHOLD,
    COUNCIL_CONVERGENCE_PROTOCOL,
    MAX_REBUTTAL_ROUNDS,
    get_council_config,
)
from src.council.judge import finalize as judge_finalize
from src.council.registry import get_paradigm
from src.council.router import route as route_difficulty
from src.council.state import DebateState, Phase
from src.llm.client import chat_json, chat_with_tools  # noqa: F401  re-exported for monkeypatching
from src.obs import span

logger = logging.getLogger(__name__)


# 向后兼容：旧代码 `from src.council.flow import CouncilResult` 仍可工作
CouncilResult = DebateState


# ---------------- public ----------------

def run_council(
    title: str,
    summary: str,
    content: str,
    provider_config: dict | None = None,
    max_rebuttal_rounds: int | None = None,
    converge_threshold: float | None = None,
    paradigm: str = "debate",
    convergence_protocol: str | None = None,
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
        paradigm: 讨论范式（debate / report / ...）。未知名称回退到 debate。
            借鉴 MALLM 的 discussion_paradigms，通过 registry.py 注册。
        convergence_protocol: 收敛协议（midcheck / consensus_threshold / voting）。
            None 时从 .env COUNCIL_CONVERGENCE_PROTOCOL 读取（默认 midcheck）。
            借鉴 MALLM 的 decision_protocols，通过 protocol_registry.py 注册。

    Returns:
        DebateState：包含完整辩论过程与 Judge 共识结果。
    """
    logger.info("=== Council Session Start [paradigm=%s]: %s ===", paradigm, title[:50])

    _debate_span = span("council.debate", article_title=title[:80], paradigm=paradigm)
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
            paradigm=paradigm,
            convergence_protocol=convergence_protocol or COUNCIL_CONVERGENCE_PROTOCOL,
        )

        council_cfg = provider_config or get_council_config()

        # --- Phase 0: Routing ---
        difficulty, active_roles, reasoning = route_difficulty(title, summary)
        state.difficulty = difficulty
        state.active_roles = active_roles
        state.routing_reasoning = reasoning

        # --- Phases 1-2: 范式驱动讨论（opening + 可选 rebuttal）---
        paradigm_cls = get_paradigm(paradigm)
        paradigm_inst = paradigm_cls()
        paradigm_inst.run_discussion(state, council_cfg)

        # --- Phase 3: Judge finalize ---
        state.phase = Phase.JUDGING
        state.consensus = judge_finalize(state)
        state.phase = Phase.DONE

        logger.info(
            "=== Council Session Done [paradigm=%s] === [terminated_by=%s rounds=%d]",
            paradigm, state.terminated_by, state.round_idx,
        )

        # 追加辩论结果属性
        try:
            _s.set_attribute("paradigm", paradigm)
            _s.set_attribute("difficulty", state.difficulty or "")
            _s.set_attribute("terminated_by", state.terminated_by or "")
            _s.set_attribute("total_rounds", state.round_idx)
            _s.set_attribute("active_roles", ",".join(state.active_roles))
            total_tc = sum(t.tool_calls_used for t in state.turns)
            _s.set_attribute("total_tool_calls", total_tc)
        except Exception:
            logger.debug("Failed to set span attributes", exc_info=True)

        return state
    finally:
        _debate_span.__exit__(None, None, None)
