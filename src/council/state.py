"""Council 辩论状态模型。

DebateState 是辩论流程的唯一数据结构，贯穿 router / opening / rebuttal / judging
各阶段。`flow.run_council` 构造并演化该对象；下游 `output.format_council_result`
和 `storage.db.save_debate` 读取它。

为了与原 CouncilResult 的调用方（app.py / workflows / tests / output.py）兼容，
DebateState 通过 @property 暴露 `critic / synthesizer / mentor` 三个字段，返回
对应角色**最新**一轮的发言内容（opening 或最后一次 rebuttal）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    """辩论生命周期的阶段。"""

    ROUTING = "routing"      # 难度分级，尚未有角色发言
    OPENING = "opening"      # 各活跃角色首轮陈述
    REBUTTAL = "rebuttal"    # 互相反驳循环
    JUDGING = "judging"      # Judge 收敛
    DONE = "done"


@dataclass
class Turn:
    """单次发言记录。"""

    role_key: str            # "critic" | "synthesizer" | "mentor"
    round_idx: int           # 0 表示 opening 轮；>=1 表示第 N 轮 rebuttal
    phase: Phase
    content: dict            # 角色 JSON 输出（保留角色原始 schema）
    force_closing: bool = False  # 是否处于"强制落地"的最后一轮
    tool_calls_used: int = 0     # 本轮使用了多少次工具调用
    tool_log: list = field(default_factory=list)  # 工具调用详情


# 不同难度对应的默认角色集
DIFFICULTY_ROLES: dict[str, list[str]] = {
    "easy":   ["mentor"],
    "medium": ["critic", "mentor"],
    "hard":   ["critic", "synthesizer", "mentor"],
}


@dataclass
class DebateState:
    """一次辩论的完整状态快照。"""

    # --- 输入 ---
    article_title: str
    article_summary: str
    article_content: str
    article_id: int | None = None

    # --- 路由结果 ---
    difficulty: str = "medium"
    active_roles: list[str] = field(default_factory=list)
    routing_reasoning: str = ""

    # --- 辩论过程 ---
    phase: Phase = Phase.ROUTING
    round_idx: int = 0               # 当前 rebuttal 轮次；opening 阶段保持 0
    turns: list[Turn] = field(default_factory=list)

    # --- 收敛控制 ---
    disagreement_score: float = 0.0  # [0, 1]，Judge 中期评估的分歧度
    consensus: dict | None = None    # Judge 最终产出（headline / key_points / ...）
    terminated_by: str = ""          # "converged" | "max_rounds" | "no_rebuttal" | "single_role"

    # --- 配置 ---
    max_rebuttal_rounds: int = 3
    converge_threshold: float = 0.3

    # ---------- 向后兼容访问器 ----------

    @property
    def critic(self) -> dict:
        """Critic 的最新发言内容，若未参与则返回空 dict。"""
        return self._latest_content("critic")

    @property
    def synthesizer(self) -> dict:
        return self._latest_content("synthesizer")

    @property
    def mentor(self) -> dict:
        return self._latest_content("mentor")

    # ---------- 辅助 ----------

    def _latest_content(self, role_key: str) -> dict:
        turns = self.turns_of(role_key)
        return turns[-1].content if turns else {}

    def turns_of(self, role_key: str) -> list[Turn]:
        """返回某个角色的全部发言，按时间顺序。"""
        return [t for t in self.turns if t.role_key == role_key]

    def spoken_roles_before(self, turn_index: int) -> list[str]:
        """给定 turn 索引，返回在它之前发过言的所有角色（去重、保持顺序）。"""
        seen: list[str] = []
        for t in self.turns[:turn_index]:
            if t.role_key not in seen:
                seen.append(t.role_key)
        return seen

    def total_rounds(self) -> int:
        """辩论进行了多少轮（opening 算 0 轮，rebuttal 从 1 开始）。"""
        return self.round_idx
