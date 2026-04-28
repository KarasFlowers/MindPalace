"""Fact Check 工具 — 针对单条论断进行事实核查。

内部先调 web_search，再用 fast 模型对搜索结果做简短判定。
"""

from __future__ import annotations

import json
import logging

from src.tools.base import register, get_tool

logger = logging.getLogger(__name__)


class FactCheckTool:
    name = "fact_check"
    description = (
        "对一条具体的事实性论断进行核查。"
        "工具会搜索网络并返回结论：supported / refuted / inconclusive，附带佐证。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": "需要核查的事实性论断",
            }
        },
        "required": ["claim"],
    }

    def run(self, *, claim: str) -> str:  # noqa: D401
        """搜索 + LLM 判定。"""
        logger.info("[FactCheck] claim=%s", claim[:80])

        # 1. 用 web_search 拿素材
        search_tool = get_tool("web_search")
        search_raw = search_tool.run(query=claim)
        try:
            search_results = json.loads(search_raw)
        except (json.JSONDecodeError, TypeError):
            search_results = []

        if isinstance(search_results, dict) and "error" in search_results:
            return json.dumps(
                {"verdict": "inconclusive", "reason": search_results["error"], "sources": []},
                ensure_ascii=False,
            )

        # 2. 用 fast 模型做判定
        try:
            from src.llm.client import chat_json
            from src.config import get_fast_config

            sources_text = "\n".join(
                f"- {r.get('title', '?')}: {r.get('snippet', '')}" for r in search_results
            )
            result = chat_json(
                system_prompt=(
                    "你是事实核查助手。根据提供的搜索结果判断论断是否成立。\n"
                    "返回 JSON: {\"verdict\": \"supported|refuted|inconclusive\", \"reason\": \"...\"}"
                ),
                user_prompt=f"论断: {claim}\n\n搜索结果:\n{sources_text}",
                provider_config=get_fast_config(),
            )
            result["sources"] = [r.get("url", "") for r in search_results[:3]]
            return json.dumps(result, ensure_ascii=False)

        except Exception as exc:
            logger.warning("[FactCheck] LLM judge failed: %s", exc)
            return json.dumps(
                {
                    "verdict": "inconclusive",
                    "reason": f"LLM 判定失败: {exc}",
                    "sources": [r.get("url", "") for r in search_results[:3]],
                },
                ensure_ascii=False,
            )


register(FactCheckTool())
