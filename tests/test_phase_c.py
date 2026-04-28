"""Phase C 测试 — Tools / Feedback / Judge Eval。

全部用 mock 避开真实 LLM 和网络调用。
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest


# ────────────────── Tools 基础 ──────────────────

class TestToolBase:
    def test_tools_registered(self):
        from src.tools import TOOLS
        assert "web_search" in TOOLS
        assert "fact_check" in TOOLS

    def test_openai_schema_format(self):
        from src.tools import to_openai_schema
        schema = to_openai_schema()
        assert len(schema) >= 2
        for entry in schema:
            assert entry["type"] == "function"
            assert "name" in entry["function"]
            assert "parameters" in entry["function"]

    def test_get_tool_raises_on_missing(self):
        from src.tools.base import get_tool
        with pytest.raises(KeyError, match="no_such_tool"):
            get_tool("no_such_tool")


class TestWebSearch:
    def test_run_returns_json_string(self):
        from src.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        with patch("src.tools.web_search._search_ddg", return_value=[
            {"title": "T1", "url": "http://a.com", "snippet": "S1"},
        ]):
            result = tool.run(query="test query")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["title"] == "T1"

    def test_run_handles_exception(self):
        from src.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        with patch("src.tools.web_search._search_ddg", side_effect=Exception("network error")):
            result = tool.run(query="fail")
        parsed = json.loads(result)
        assert "error" in parsed


class TestFactCheck:
    def test_run_returns_verdict(self):
        from src.tools.fact_check import FactCheckTool
        tool = FactCheckTool()
        with patch("src.tools.web_search._search_ddg", return_value=[
            {"title": "Evidence", "url": "http://e.com", "snippet": "supports claim"},
        ]), patch("src.llm.client.chat_json", return_value={
            "verdict": "supported", "reason": "evidence found"
        }), patch("src.config.get_fast_config", return_value={}):
            result = tool.run(claim="Earth is round")
        parsed = json.loads(result)
        assert parsed["verdict"] == "supported"
        assert "sources" in parsed


# ────────────────── chat_with_tools ──────────────────

class TestChatWithTools:
    def test_no_tool_calls_returns_content(self):
        from src.llm.client import chat_with_tools

        mock_msg = MagicMock()
        mock_msg.tool_calls = None
        mock_msg.content = '{"answer": "42"}'

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=mock_msg)]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("src.llm.client._get_client_from_pool", return_value=mock_client), \
             patch("src.llm.client._get_user_profile", return_value=""):
            result = chat_with_tools(
                system_prompt="sys",
                user_prompt="user",
                tools_schema=[],
                tool_executor={},
                provider_config={"api_key": "k", "base_url": "u", "models": ["m"]},
            )

        assert result["content"] == '{"answer": "42"}'
        assert result["tool_calls_used"] == 0

    def test_tool_call_loop_executes(self):
        from src.llm.client import chat_with_tools

        # 第一次响应：工具调用
        tool_call = MagicMock()
        tool_call.id = "tc_1"
        tool_call.function.name = "web_search"
        tool_call.function.arguments = '{"query": "test"}'

        msg1 = MagicMock()
        msg1.tool_calls = [tool_call]
        msg1.content = None
        msg1.model_dump.return_value = {"role": "assistant", "tool_calls": []}

        resp1 = MagicMock()
        resp1.choices = [MagicMock(message=msg1)]

        # 第二次响应：最终答案
        msg2 = MagicMock()
        msg2.tool_calls = None
        msg2.content = '{"done": true}'

        resp2 = MagicMock()
        resp2.choices = [MagicMock(message=msg2)]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [resp1, resp2]

        fake_tool = MagicMock()
        fake_tool.run.return_value = '{"results": []}'

        with patch("src.llm.client._get_client_from_pool", return_value=mock_client), \
             patch("src.llm.client._get_user_profile", return_value=""):
            result = chat_with_tools(
                system_prompt="sys",
                user_prompt="user",
                tools_schema=[{"type": "function", "function": {"name": "web_search"}}],
                tool_executor={"web_search": fake_tool},
                provider_config={"api_key": "k", "base_url": "u", "models": ["m"]},
            )

        assert result["tool_calls_used"] == 1
        assert len(result["tool_log"]) == 1
        assert result["tool_log"][0]["tool"] == "web_search"
        fake_tool.run.assert_called_once_with(query="test")


# ────────────────── Feedback ──────────────────

class TestFeedback:
    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._patchers = [
            patch("src.storage.db.DB_PATH", self._tmp.name),
            patch("src.eval.feedback.init_db", self._init_test_db),
        ]
        for p in self._patchers:
            p.start()

    def teardown_method(self):
        for p in reversed(self._patchers):
            p.stop()
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def _init_test_db(self):
        conn = sqlite3.connect(self._tmp.name)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                debate_id INTEGER NOT NULL,
                rating TEXT NOT NULL,
                adopted_role TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def test_save_feedback(self):
        from src.eval.feedback import save_feedback
        fid = save_feedback(debate_id=1, rating="up", note="good")
        assert fid >= 1

    def test_get_feedback_stats(self):
        from src.eval.feedback import save_feedback, get_feedback_stats
        save_feedback(debate_id=1, rating="up")
        save_feedback(debate_id=2, rating="up")
        save_feedback(debate_id=3, rating="down")
        stats = get_feedback_stats(days=1)
        assert stats.get("up", 0) == 2
        assert stats.get("down", 0) == 1


# ────────────────── Judge Debates ──────────────────

class TestJudgeDebates:
    def test_generate_weekly_report_format(self):
        from src.eval.judge_debates import generate_weekly_report

        reports = [
            {
                "debate_id": 1,
                "article_title": "Test",
                "scores": {"logical_rigor": 8, "inspiration": 7, "coverage": 6, "groundedness": 5},
                "weaknesses": ["weak1"],
                "prompt_improvement_hint": "hint1",
            }
        ]
        report = generate_weekly_report(reports, days=7)
        assert "Weekly Eval Report" in report
        assert "logical_rigor" in report
        assert "weak1" in report
        assert "hint1" in report

    def test_empty_reports(self):
        from src.eval.judge_debates import generate_weekly_report
        report = generate_weekly_report([], days=7)
        assert "无有效评估数据" in report


# ────────────────── Turn 新字段 ──────────────────

class TestTurnToolFields:
    def test_turn_defaults(self):
        from src.council.state import Turn, Phase
        t = Turn(role_key="critic", round_idx=0, phase=Phase.OPENING, content={})
        assert t.tool_calls_used == 0
        assert t.tool_log == []

    def test_turn_with_tools(self):
        from src.council.state import Turn, Phase
        t = Turn(
            role_key="critic", round_idx=1, phase=Phase.REBUTTAL,
            content={"test": True},
            tool_calls_used=2,
            tool_log=[{"tool": "web_search", "args": {"query": "x"}, "result_preview": "..."}],
        )
        assert t.tool_calls_used == 2
        assert len(t.tool_log) == 1
