"""Web Search 工具 — 基于 DuckDuckGo，供 Council 角色查证事实。"""

from __future__ import annotations

import json
import logging
import os

from src.tools.base import register

logger = logging.getLogger(__name__)

_SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()


class WebSearchTool:
    name = "web_search"
    description = (
        "当你对一个事实性论断不确定时，搜索网络获取佐证。"
        "返回前 3 条结果的标题、链接和摘要。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词（建议英文，提高召回率）",
            }
        },
        "required": ["query"],
    }

    def run(self, *, query: str) -> str:  # noqa: D401
        """执行搜索并返回 JSON 字符串。"""
        logger.info("[WebSearch] query=%s (provider=%s)", query, _SEARCH_PROVIDER)
        try:
            results = _search_ddg(query)
        except Exception as exc:
            logger.warning("[WebSearch] search failed: %s", exc)
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        return json.dumps(results, ensure_ascii=False)


def _search_ddg(query: str, max_results: int = 3) -> list[dict]:
    """DuckDuckGo 免费搜索。"""
    from duckduckgo_search import DDGS

    raw = list(DDGS().text(query, max_results=max_results))
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", r.get("link", "")),
            "snippet": r.get("body", r.get("snippet", "")),
        }
        for r in raw
    ]


register(WebSearchTool())
