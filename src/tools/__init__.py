"""Tools 子包 — Council 角色可调用的外部工具。"""

from src.tools.base import TOOLS, register, to_openai_schema, get_tool

# 自动注册内置工具（import 即注册）
import src.tools.web_search  # noqa: F401
import src.tools.fact_check   # noqa: F401
