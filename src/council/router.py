"""难度路由。用轻量模型判定话题难度，决定派几个角色上场。

对应面试亮点：动态路由 —— easy 话题只派 1 个 Agent，hard 话题派全员，
直接体现"按成本/质量自适应派单"。
"""

from __future__ import annotations

import logging

from src.config import get_router_config
from src.council.state import DIFFICULTY_ROLES
from src.llm.client import chat_json

logger = logging.getLogger(__name__)


ROUTER_SYSTEM_PROMPT = """\
你是 MindPalace 议事厅的"难度路由器"。

你要根据一篇文章的标题和摘要，判断它适合派多少个 Agent 进行讨论。
判定标准：

- easy: 单一观点、无逻辑争议、偏事实性介绍或常识性陈述。派 1 个 Mentor 追问即可。
- medium: 有一定论证链，存在可挑战的假设，但观点整体一致。派 Critic + Mentor。
- hard: 多方观点冲突、涉及价值判断、因果链复杂，或含有模糊/争议的核心论断。派全员。

以 JSON 输出：
{
  "difficulty": "easy" | "medium" | "hard",
  "reasoning": "一句话说明你的判断依据（中文，不超过 50 字）"
}
"""


def route(
    title: str,
    summary: str,
    provider_config: dict | None = None,
) -> tuple[str, list[str], str]:
    """返回 (difficulty, active_roles, reasoning)。

    任何异常都回退到 medium 档，保证辩论始终能跑起来。
    """
    cfg = provider_config or get_router_config()
    user_prompt = f"文章标题: {title}\n文章摘要: {summary}"

    try:
        result = chat_json(ROUTER_SYSTEM_PROMPT, user_prompt, provider_config=cfg)
    except Exception as exc:  # noqa: BLE001 路由失败不该阻塞主流程
        logger.warning("[Router] 路由失败，回退到 medium: %s", exc)
        return "medium", DIFFICULTY_ROLES["medium"], "router_fallback"

    difficulty = str(result.get("difficulty", "medium")).lower().strip()
    if difficulty not in DIFFICULTY_ROLES:
        logger.warning("[Router] 未知难度 %r，回退到 medium", difficulty)
        difficulty = "medium"

    reasoning = str(result.get("reasoning", "")).strip()
    active_roles = DIFFICULTY_ROLES[difficulty]

    logger.info(
        "[Router] difficulty=%s roles=%s reason=%s",
        difficulty, active_roles, reasoning[:60],
    )
    return difficulty, active_roles, reasoning
