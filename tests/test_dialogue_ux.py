"""对话体验层测试。"""

from types import SimpleNamespace
from unittest.mock import patch


def _fake_council_result():
    return SimpleNamespace(
        article_title="Modernity and Care",
        article_summary="summary",
        consensus={
            "headline": "这篇文章最有价值的地方，是把制度与情感的张力摆到了台面上。",
            "key_points": ["制度与情感的张力"],
            "recommended_stance": "先承认张力，再讨论如何取舍。",
        },
        critic={
            "vulnerabilities": [
                {
                    "assumption": "作者默认制度一定压制关怀",
                    "counter": "如果制度设计得当，它也可能保护关怀",
                    "severity": "high",
                }
            ],
            "verdict": "论证尖锐，但前提还不够扎实。",
        },
        synthesizer={
            "connections": [
                {
                    "domain": "history",
                    "analogy": "像韦伯谈科层制一样",
                    "insight": "问题不是制度本身，而是制度如何塑造行动者。",
                }
            ],
            "synthesis": "这提醒我们别把效率和关怀简单对立起来。",
        },
        mentor={
            "questions": [
                {
                    "level": "立场挑战",
                    "question": "如果效率真的让更多人受益，你还会坚持优先保留关怀吗？",
                }
            ],
            "provocation": "你真正害怕失去的，到底是效率还是人的温度？",
        },
    )


def test_build_council_snapshot_picks_discussion_handles():
    from src import app

    snapshot = app._build_council_snapshot(_fake_council_result())

    assert "制度与情感的张力" in snapshot["summary"]
    assert "制度一定压制关怀" in snapshot["challenge"]
    assert "制度如何塑造行动者" in snapshot["bridge"]
    assert "效率真的让更多人受益" in snapshot["question"]


def test_collect_guided_user_response_uses_starter():
    from src import app

    inputs = iter(["1", "它把制度激励讲清楚了。", "", ""])
    with patch("builtins.input", side_effect=lambda *args, **kwargs: next(inputs)):
        response = app._collect_guided_user_response()

    assert response.startswith("我同意这里最有说服力的一点，因为")
    assert "制度激励讲清楚了" in response


def test_run_council_experience_collects_feedback_even_when_response_skipped():
    from src import app

    article = {"id": 1, "title": "Modernity and Care", "summary": "summary"}
    inputs = iter(["", "skip"])

    with patch("src.app.run_council", return_value=_fake_council_result()), \
         patch("src.app.save_debate", return_value=99), \
         patch("src.app.collect_feedback_interactive") as mock_feedback, \
         patch("builtins.input", side_effect=lambda *args, **kwargs: next(inputs)):
        app._run_council_experience(article, pause_at_end=False)

    mock_feedback.assert_called_once_with(99)


def test_parse_structured_response_handles_fenced_json():
    from src.resolve import engine

    parsed = engine._parse_structured_response(
        """```json
        {"questions":[{"level":"立场挑战","question":"你为什么这样想？"}],"provocation":"再往下走一步。"}
        ```"""
    )

    assert parsed["questions"][0]["question"] == "你为什么这样想？"


def test_extract_role_highlight_prefers_structured_fields():
    from src.resolve import engine

    critic_response = """
    {
      "vulnerabilities": [{"assumption": "作者把个体选择看得过于自由", "counter": "现实里结构约束更强", "severity": "high"}],
      "verdict": "论证偏理想化"
    }
    """

    highlight = engine._extract_role_highlight("critic", critic_response)
    assert "个体选择看得过于自由" in highlight


def test_interactive_resolve_restores_full_session_id():
    from src import app

    full_id = "12345678-abcd-4def-9000-abcdef123456"
    sessions = [
        {
            "id": full_id,
            "title": "旧会话",
            "mode": "council",
            "updated_at": "2026-06-24T12:00:00",
        }
    ]

    with patch("src.resolve.engine.list_sessions", return_value=sessions), \
         patch("src.resolve.engine.run_repl") as mock_run_repl, \
         patch("src.app.questionary.select") as mock_select:
        mock_select.return_value.ask.side_effect = [
            "📜 查看并恢复历史会话",
            "[12345678...] 旧会话 (council) - 2026-06-24",
        ]
        app._interactive_resolve()

    mock_run_repl.assert_called_once_with(session_id=full_id)
