"""Tool 抽象 — 对齐 OpenAI function calling 的工具注册与调度。"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Tool(Protocol):
    """工具协议。实现此协议即可被 Council 角色通过 function calling 调用。"""

    name: str
    description: str
    parameters: dict  # JSON Schema

    def run(self, **kwargs) -> str:
        """执行工具并返回字符串结果。"""
        ...


# 全局工具注册表
TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    """将工具注册到全局注册表。"""
    TOOLS[tool.name] = tool
    logger.debug("Registered tool: %s", tool.name)


def get_tool(name: str) -> Tool:
    """按名称获取已注册的工具。"""
    if name not in TOOLS:
        raise KeyError(f"Tool '{name}' not registered. Available: {list(TOOLS.keys())}")
    return TOOLS[name]


def to_openai_schema() -> list[dict]:
    """将所有已注册工具转为 OpenAI function calling 格式。"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in TOOLS.values()
    ]
