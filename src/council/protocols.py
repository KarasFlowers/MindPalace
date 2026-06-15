"""收敛协议抽象 — 借鉴 MALLM 的 decision_protocols 设计。

将 Council 的"怎么判断收敛"从单一 midcheck 抽象为可注册的协议。
每个协议实现 `check(state, council_cfg) -> {converged, reason, score}`。

与 MALLM 的差异：
- 精简为 3 个协议（midcheck 默认 / consensus_threshold / voting）
- 接口聚焦"是否收敛"，不引入 MALLM 的 8 种投票变体（保持轻量）
- 默认 midcheck 完全向后兼容现有 DebateParadigm 行为
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from src.config import CONVERGE_THRESHOLD, get_router_config
from src.council.judge import midcheck as judge_midcheck
from src.council.state import DebateState
from src.llm.client import chat_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class ConvergenceProtocol(ABC):
    """收敛协议抽象基类。

    每个协议定义"一轮反驳结束后如何判断是否收敛"。
    DebateParadigm 在每轮 rebuttal 后调用 protocol.check()。
    """

    name: str = "base"

    @abstractmethod
    def check(self, state: DebateState, council_cfg: dict) -> dict:
        """检查一轮结束后是否收敛。

        Args:
            state: 当前辩论状态（含本轮所有 turns）。
            council_cfg: Council 档 provider 配置。

        Returns:
            {
                "converged": bool,    # 是否已收敛（应停止 rebuttal）
                "reason": str,        # 收敛原因（写入 terminated_by；未收敛则空）
                "score": float,       # 收敛分数（写入 disagreement_score）
            }
        """
        ...


# ---------------------------------------------------------------------------
# Midcheck 协议（默认，完全向后兼容）
# ---------------------------------------------------------------------------

class MidcheckProtocol(ConvergenceProtocol):
    """现有 midcheck 封装为协议（默认协议）。

    行为与重构前完全一致：调用 judge.midcheck，should_continue=False 即收敛。
    """

    name = "midcheck"

    def check(self, state: DebateState, council_cfg: dict) -> dict:
        result = judge_midcheck(state)
        converged = not result["should_continue"]
        return {
            "converged": converged,
            "reason": "converged" if converged else "",
            "score": result["disagreement_score"],
        }


# ---------------------------------------------------------------------------
# Consensus Threshold 协议
# ---------------------------------------------------------------------------

class ConsensusThresholdProtocol(ConvergenceProtocol):
    """共识阈值协议：分歧度低于 CONVERGE_THRESHOLD 即收敛。

    与 midcheck 的区别：不依赖 LLM 的 should_continue 布尔，而是用 LLM 给出的
    disagreement_score 直接与阈值比较。更可预测、更省 token（仍需一次 midcheck 调用，
    但忽略其 should_continue 字段）。
    """

    name = "consensus_threshold"

    def check(self, state: DebateState, council_cfg: dict) -> dict:
        result = judge_midcheck(state)
        score = result["disagreement_score"]
        converged = score < CONVERGE_THRESHOLD
        return {
            "converged": converged,
            "reason": "consensus_threshold" if converged else "",
            "score": score,
        }


# ---------------------------------------------------------------------------
# Voting 协议
# ---------------------------------------------------------------------------

VOTING_SYSTEM_PROMPT = """\
你是 MindPalace 议事厅的"投票协调员"。

各角色刚刚完成一轮发言。请评估：当前各角色的立场是否已经趋于一致，
足以停止进一步辩论。

判断依据：
- 若各方核心论点已趋同，或主要分歧已被充分回应 → confidence 高（> 0.7）
- 若仍有实质性未被解决的分歧 → confidence 低

严格以 JSON 输出：
{
  "consensus_confidence": <0.0-1.0 浮点，各方立场一致程度>,
  "main_remaining_divergence": "若未收敛，主要分歧点（中文，不超过 50 字）"
}
"""


class VotingProtocol(ConvergenceProtocol):
    """投票协议：LLM 评估各方立场一致程度，超阈值即收敛。

    与 midcheck 的区别：不计算"分歧度"，而是评估"共识置信度"。
    consensus_confidence >= CONVERGE_THRESHOLD 即收敛。
    用 router 档模型（便宜）。
    """

    name = "voting"

    def check(self, state: DebateState, council_cfg: dict) -> dict:
        from src.council.judge import _format_transcript

        transcript = _format_transcript(state)
        cfg = get_router_config()

        try:
            result = chat_json(VOTING_SYSTEM_PROMPT, transcript, provider_config=cfg)
        except Exception as exc:
            logger.warning("[Voting] 评估失败，保守继续: %s", exc)
            return {"converged": False, "reason": "", "score": 0.5}

        try:
            confidence = float(result.get("consensus_confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # consensus_confidence 高 = 分歧度低；转成 disagreement_score 以兼容 state 字段
        disagreement = round(1.0 - confidence, 4)
        converged = confidence >= CONVERGE_THRESHOLD

        return {
            "converged": converged,
            "reason": "voting_consensus" if converged else "",
            "score": disagreement,
        }
