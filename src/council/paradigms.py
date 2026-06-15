"""讨论范式抽象 — 借鉴 MALLM 的 discussion_paradigms 设计。

将 Council 的"怎么讨论"从单一硬编码流程抽象为可注册的范式（paradigm）。
每个范式是一个类，实现 `run(state, council_cfg) -> DebateState`。

为保证测试可 mock（测试 monkeypatch `src.council.flow.chat_json` / `chat_with_tools`），
范式内部通过运行时访问 `src.council.flow` 模块的 chat_json / chat_with_tools，
而非在范式模块顶部直接 import，这样对 flow 模块的 patch 仍然生效。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.config import CONVERGE_THRESHOLD, MAX_REBUTTAL_ROUNDS
from src.council.judge import finalize as judge_finalize
from src.council.rebuttal import build_opening_prompt, build_rebuttal_prompt
from src.council.roles import get_role, TOOL_ENABLED_ROLES
from src.council.state import DebateState, Phase, Turn
from src.obs import span

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class DiscussionParadigm(ABC):
    """讨论范式的抽象基类（模板方法模式）。

    每个范式定义一次完整的讨论流程：opening → （可选）多轮 → judge 收敛。
    子类只需实现 `run_discussion()`，路由/初始化 state 与 judge finalize 由基类
    或调用方处理。`name` 属性用于注册表标识。
    """

    name: str = "base"

    @abstractmethod
    def run_discussion(self, state: DebateState, council_cfg: dict) -> DebateState:
        """执行讨论流程（opening + 可选 rebuttal），不调用 judge finalize。

        Args:
            state: 已完成 routing 的 DebateState（含 difficulty / active_roles）。
            council_cfg: Council 档 provider 配置。

        Returns:
            更新后的 state（含 turns，phase 推进到 JUDGING 之前）。
        """
        ...

    # ---- 共享辅助方法 ----

    def _run_role_turn(
        self,
        state: DebateState,
        role_key: str,
        council_cfg: dict,
        phase: Phase,
        force_closing: bool,
    ) -> tuple[dict, int, list]:
        """执行单个角色的一次发言（沿用旧 flow 的 tool-use 回退逻辑）。

        通过运行时访问 src.council.flow 的 chat_json / chat_with_tools，
        保证测试对 flow 模块的 monkeypatch 仍然生效。
        """
        # 运行时访问，避免硬绑定到本模块的 import 时刻
        import src.council.flow as _flow

        role = get_role(role_key)
        if phase is Phase.OPENING:
            user_prompt = build_opening_prompt(state, role_key)
        else:
            user_prompt = build_rebuttal_prompt(state, role_key, force_closing=force_closing)

        logger.info(
            "[Council:%s] %s speaking (phase=%s, round=%s, force_closing=%s)",
            self.name, role["name"], phase.value, state.round_idx, force_closing,
        )

        with span(
            f"council.{self.name}.role.{role_key}",
            paradigm=self.name,
            role=role_key,
            phase=phase.value,
            round_idx=state.round_idx,
            force_closing=force_closing,
        ):

            if role_key in TOOL_ENABLED_ROLES and not force_closing:
                try:
                    from src.tools import TOOLS, to_openai_schema
                    if TOOLS:
                        result = _flow.chat_with_tools(
                            system_prompt=role["prompt"],
                            user_prompt=user_prompt,
                            tools_schema=to_openai_schema(),
                            tool_executor=TOOLS,
                            max_tool_calls=3,
                            provider_config=council_cfg,
                        )
                        content_str = result["content"]
                        try:
                            import json
                            content = json.loads(content_str)
                        except (json.JSONDecodeError, TypeError):
                            content = {"raw_response": content_str}
                        return content, result["tool_calls_used"], result["tool_log"]
                except Exception as exc:
                    logger.warning(
                        "[Council:%s] tool-use path failed for %s, falling back to chat_json: %s",
                        self.name, role_key, exc,
                    )

            content = _flow.chat_json(role["prompt"], user_prompt, provider_config=council_cfg)
            return content, 0, []

    def _append_turn(
        self,
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


# ---------------------------------------------------------------------------
# Debate 范式（原有流程）
# ---------------------------------------------------------------------------

class DebateParadigm(DiscussionParadigm):
    """经典对抗式辩论：Opening → Rebuttal 循环（midcheck 收敛）。

    这是重构前的原 run_council 逻辑，提取为范式以支持可插拔。
    """

    name = "debate"

    def run_discussion(self, state: DebateState, council_cfg: dict) -> DebateState:
        # --- Phase 1: Opening ---
        state.phase = Phase.OPENING
        for role_key in state.active_roles:
            content_json, tc_used, tc_log = self._run_role_turn(
                state, role_key, council_cfg, Phase.OPENING, force_closing=False
            )
            self._append_turn(
                state, role_key, Phase.OPENING, content_json,
                force_closing=False, tool_calls_used=tc_used, tool_log=tc_log,
            )

        # --- Phase 2: Rebuttal loop（仅当多角色且允许反驳时）---
        if len(state.active_roles) >= 2 and state.max_rebuttal_rounds > 0:
            state.phase = Phase.REBUTTAL
            while state.round_idx < state.max_rebuttal_rounds:
                state.round_idx += 1
                force_closing = state.round_idx == state.max_rebuttal_rounds

                for role_key in state.active_roles:
                    content_json, tc_used, tc_log = self._run_role_turn(
                        state, role_key, council_cfg, Phase.REBUTTAL,
                        force_closing=force_closing,
                    )
                    self._append_turn(
                        state, role_key, Phase.REBUTTAL, content_json,
                        force_closing=force_closing, tool_calls_used=tc_used, tool_log=tc_log,
                    )

                if force_closing:
                    state.terminated_by = "max_rounds"
                    break

                # 收敛检查：通过可插拔协议（默认 midcheck，向后兼容）
                from src.council.protocol_registry import get_protocol
                protocol = get_protocol(state.convergence_protocol)
                result = protocol.check(state, council_cfg)
                state.disagreement_score = result["score"]
                if result["converged"]:
                    state.terminated_by = result["reason"] or "converged"
                    break
        else:
            state.terminated_by = (
                "single_role" if len(state.active_roles) == 1 else "no_rebuttal"
            )

        return state


# ---------------------------------------------------------------------------
# Report 范式（新增）
# ---------------------------------------------------------------------------

class ReportParadigm(DiscussionParadigm):
    """中心化审阅范式：一个主起草人生成完整报告，其他人给反馈。

    借鉴 MALLM 的 Report 范式。流程：
    1. 主起草人（默认 synthesizer）读取文章，生成完整分析报告。
    2. 其他角色针对报告给出简短反馈（1 轮，不做对抗式反驳）。
    3. 由 Judge 收敛最终结论。

    与 Debate 的区别：无 opening+rebuttal 多轮对抗，而是单向起草 → 反馈。
    适合需要结构化综述而非观点碰撞的场景。
    """

    name = "report"

    def run_discussion(self, state: DebateState, council_cfg: dict) -> DebateState:
        if not state.active_roles:
            state.terminated_by = "no_rebuttal"
            return state

        # 选择主起草人：优先 synthesizer，否则取第一个角色
        drafter = "synthesizer" if "synthesizer" in state.active_roles else state.active_roles[0]
        reviewers = [r for r in state.active_roles if r != drafter]

        # --- Phase 1: 主起草人生成报告（opening）---
        state.phase = Phase.OPENING
        content_json, tc_used, tc_log = self._run_role_turn(
            state, drafter, council_cfg, Phase.OPENING, force_closing=False
        )
        self._append_turn(
            state, drafter, Phase.OPENING, content_json,
            force_closing=False, tool_calls_used=tc_used, tool_log=tc_log,
        )

        # --- Phase 2: 审阅者给出反馈（单轮 rebuttal，非对抗）---
        if reviewers and state.max_rebuttal_rounds > 0:
            state.phase = Phase.REBUTTAL
            state.round_idx = 1
            for role_key in reviewers:
                content_json, tc_used, tc_log = self._run_role_turn(
                    state, role_key, council_cfg, Phase.REBUTTAL,
                    force_closing=False,
                )
                self._append_turn(
                    state, role_key, Phase.REBUTTAL, content_json,
                    force_closing=False, tool_calls_used=tc_used, tool_log=tc_log,
                )
            state.terminated_by = "reviewed"
        else:
            state.terminated_by = "single_role"

        return state
