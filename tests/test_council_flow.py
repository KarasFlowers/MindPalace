"""Council 状态机流程测试。

全部用 mock 避开真实 LLM 调用。覆盖：
- 路由（easy / medium / hard）→ 活跃角色数量
- opening 阶段各角色依次发言
- rebuttal 循环：converge / max_rounds 两条终止路径
- Judge finalize 总是被调用
- 强制落地（force_closing）标记正确传递到最后一轮
- 单角色 debate 跳过 rebuttal
- 向后兼容：DebateState.critic/synthesizer/mentor 属性仍可访问
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.council.flow import run_council, CouncilResult
from src.council.state import DebateState, Phase, Turn, DIFFICULTY_ROLES
from src.council.roles import get_roles, get_discussion_order


# ---------------- 角色注册（沿用旧测试） ----------------

class TestRoles:
    def test_all_roles_registered(self):
        roles = get_roles()
        assert "critic" in roles
        assert "synthesizer" in roles
        assert "mentor" in roles

    def test_discussion_order(self):
        # 保留默认顺序接口
        assert get_discussion_order() == ["critic", "synthesizer", "mentor"]

    def test_each_role_has_prompt(self):
        roles = get_roles()
        for key in ("critic", "synthesizer", "mentor"):
            assert len(roles[key]["prompt"]) > 100


# ---------------- DebateState 基础行为 ----------------

class TestDebateState:
    def test_backward_compat_accessors_return_empty_dict(self):
        s = DebateState(article_title="T", article_summary="S", article_content="C")
        assert s.critic == {}
        assert s.synthesizer == {}
        assert s.mentor == {}

    def test_latest_content_returns_most_recent(self):
        s = DebateState(article_title="T", article_summary="S", article_content="C")
        s.turns.append(Turn(role_key="critic", round_idx=0, phase=Phase.OPENING, content={"v": 1}))
        s.turns.append(Turn(role_key="critic", round_idx=1, phase=Phase.REBUTTAL, content={"v": 2}))
        assert s.critic == {"v": 2}

    def test_couincil_result_is_alias(self):
        # 向后兼容：旧代码 from src.council.flow import CouncilResult 仍工作
        assert CouncilResult is DebateState

    def test_difficulty_roles_map(self):
        assert DIFFICULTY_ROLES["easy"] == ["mentor"]
        assert len(DIFFICULTY_ROLES["hard"]) == 3


# ---------------- 完整 run_council ----------------

def _stub_role_output(role_key: str, round_idx: int) -> dict:
    """为角色生成可识别的占位 JSON。"""
    if role_key == "critic":
        return {
            "vulnerabilities": [{"assumption": f"A{round_idx}", "counter": f"C{round_idx}",
                                 "severity": "medium"}],
            "missing_counterexample": f"CE-{round_idx}",
            "verdict": f"verdict-r{round_idx}",
        }
    if role_key == "synthesizer":
        return {
            "connections": [{"domain": "bio", "analogy": f"ana-{round_idx}",
                             "insight": f"ins-{round_idx}"}],
            "synthesis": f"synthesis-r{round_idx}",
        }
    if role_key == "mentor":
        return {
            "questions": [
                {"level": "立场挑战", "question": f"Q1-r{round_idx}"},
                {"level": "价值观追问", "question": f"Q2-r{round_idx}"},
                {"level": "价值权衡", "question": f"Q3-r{round_idx}"},
            ],
            "provocation": f"prov-r{round_idx}",
        }
    return {}


@pytest.fixture
def patched_chat(monkeypatch):
    """统一 mock 各模块的 chat_json / chat_with_tools。返回 (flow_mock, router_mock, judge_mock).

    flow_mock 同时代理 chat_json（Mentor 路径）和 chat_with_tools（Critic/Synthesizer），
    call_count 统一计数以保持旧测试语义。
    """
    flow_mock = MagicMock()
    router_mock = MagicMock()
    judge_mock = MagicMock()

    # chat_with_tools 代理到同一个 flow_mock，包装返回值为 tool-use 格式
    import json as _json

    def _tools_shim(*args, **kwargs):
        result = flow_mock(*args, **kwargs)
        # chat_with_tools 返回 dict，需包装为 {"content": json, ...}
        return {
            "content": _json.dumps(result, ensure_ascii=False),
            "tool_calls_used": 0,
            "tool_log": [],
        }

    monkeypatch.setattr("src.council.flow.chat_json", flow_mock)
    monkeypatch.setattr("src.council.flow.chat_with_tools", _tools_shim)
    monkeypatch.setattr("src.council.router.chat_json", router_mock)
    monkeypatch.setattr("src.council.judge.chat_json", judge_mock)
    return flow_mock, router_mock, judge_mock


class TestRouting:
    def test_easy_only_runs_mentor(self, patched_chat):
        flow_mock, router_mock, judge_mock = patched_chat
        router_mock.return_value = {"difficulty": "easy", "reasoning": "简单事实介绍"}

        # easy 只派 mentor，所以只有 opening 的 1 次角色调用
        flow_mock.side_effect = [_stub_role_output("mentor", 0)]

        # Judge.finalize 仍然会跑
        judge_mock.return_value = {
            "headline": "短说", "key_points": ["p1"],
            "remaining_tensions": [], "recommended_stance": "就这样",
        }

        state = run_council("T", "S", "C")

        assert state.difficulty == "easy"
        assert state.active_roles == ["mentor"]
        assert state.terminated_by == "single_role"
        assert flow_mock.call_count == 1  # 只有 mentor opening
        assert state.round_idx == 0       # 没有进入 rebuttal
        assert state.phase == Phase.DONE
        assert state.consensus is not None
        assert state.consensus["headline"] == "短说"

    def test_hard_runs_three_roles(self, patched_chat):
        flow_mock, router_mock, judge_mock = patched_chat
        router_mock.return_value = {"difficulty": "hard", "reasoning": "多方观点"}

        # 3 角色 opening + 若干 rebuttal
        def side(*args, **kwargs):
            # 无限生成 stubs
            return _stub_role_output("critic", 0)

        flow_mock.side_effect = side
        # 中期立刻收敛，避免多轮
        judge_mock.side_effect = [
            {"disagreement_score": 0.1, "should_continue": False, "next_focus": ""},
            {"headline": "H", "key_points": [], "remaining_tensions": [],
             "recommended_stance": ""},
        ]

        state = run_council("T", "S", "C")

        assert state.difficulty == "hard"
        assert len(state.active_roles) == 3
        # opening (3) + rebuttal 第 1 轮 (3) = 6 次
        assert flow_mock.call_count == 6
        assert state.terminated_by == "converged"
        assert state.round_idx == 1


class TestRebuttalLoop:
    def test_max_rounds_forces_closing(self, patched_chat):
        flow_mock, router_mock, judge_mock = patched_chat
        router_mock.return_value = {"difficulty": "hard", "reasoning": "test"}
        flow_mock.side_effect = [_stub_role_output("c", i) for i in range(100)]
        # midcheck 永远说"继续"，逼到 max_rounds
        judge_mock.side_effect = [
            {"disagreement_score": 0.9, "should_continue": True, "next_focus": "x"},
            {"disagreement_score": 0.9, "should_continue": True, "next_focus": "x"},
            # finalize
            {"headline": "H", "key_points": [], "remaining_tensions": [],
             "recommended_stance": ""},
        ]

        state = run_council("T", "S", "C", max_rebuttal_rounds=3)

        assert state.terminated_by == "max_rounds"
        assert state.round_idx == 3

        # opening 3 + rebuttal 3 轮 × 3 角色 = 3 + 9 = 12
        assert flow_mock.call_count == 12

        # 最后一轮 3 个 Turn 的 force_closing 必须为 True
        last_turns = state.turns[-3:]
        assert all(t.force_closing for t in last_turns)
        assert all(t.round_idx == 3 for t in last_turns)
        # 之前的 rebuttal 轮则 force_closing=False
        prior_rebuttal_turns = [t for t in state.turns
                                if t.phase == Phase.REBUTTAL and t.round_idx < 3]
        assert all(not t.force_closing for t in prior_rebuttal_turns)

    def test_single_role_skips_rebuttal(self, patched_chat):
        flow_mock, router_mock, judge_mock = patched_chat
        router_mock.return_value = {"difficulty": "easy", "reasoning": "x"}
        flow_mock.side_effect = [_stub_role_output("mentor", 0)]
        judge_mock.return_value = {
            "headline": "H", "key_points": [], "remaining_tensions": [],
            "recommended_stance": "",
        }

        state = run_council("T", "S", "C")

        # 只有一次 opening，没有任何 midcheck 调用
        assert flow_mock.call_count == 1
        # midcheck 未被调用；finalize 被调用 1 次
        assert judge_mock.call_count == 1

    def test_converge_short_circuits(self, patched_chat):
        flow_mock, router_mock, judge_mock = patched_chat
        router_mock.return_value = {"difficulty": "medium", "reasoning": "x"}
        # medium = critic + mentor
        flow_mock.side_effect = [_stub_role_output(r, i) for i in range(20) for r in ("x",)]
        judge_mock.side_effect = [
            # midcheck 第 1 轮之后立即收敛
            {"disagreement_score": 0.1, "should_continue": False, "next_focus": ""},
            {"headline": "H", "key_points": [], "remaining_tensions": [],
             "recommended_stance": ""},
        ]

        state = run_council("T", "S", "C", max_rebuttal_rounds=5)

        assert state.terminated_by == "converged"
        assert state.round_idx == 1
        # opening (2) + rebuttal r1 (2) = 4 次角色调用
        assert flow_mock.call_count == 4
        # 最后一轮的 force_closing 一定是 False（是提前收敛，不是被逼收尾）
        assert not state.turns[-1].force_closing


class TestJudgeAlwaysRuns:
    def test_finalize_called_even_on_router_fallback(self, patched_chat):
        flow_mock, router_mock, judge_mock = patched_chat
        # router 失败：抛异常，route() 应兜底到 medium
        router_mock.side_effect = RuntimeError("router down")
        flow_mock.side_effect = [_stub_role_output("x", 0) for _ in range(10)]
        judge_mock.side_effect = [
            {"disagreement_score": 0.05, "should_continue": False, "next_focus": ""},
            {"headline": "fallback-headline", "key_points": [],
             "remaining_tensions": [], "recommended_stance": ""},
        ]

        state = run_council("T", "S", "C")

        assert state.difficulty == "medium"  # router 回退
        assert state.consensus["headline"] == "fallback-headline"

    def test_finalize_failure_yields_error_payload(self, patched_chat):
        flow_mock, router_mock, judge_mock = patched_chat
        router_mock.return_value = {"difficulty": "easy", "reasoning": "x"}
        flow_mock.side_effect = [_stub_role_output("mentor", 0)]
        judge_mock.side_effect = RuntimeError("judge down")

        state = run_council("T", "S", "C")

        assert state.consensus is not None
        assert "error" in state.consensus
