"""Memory 模块测试 — Profiler + Store + Echo Location。"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

from src.memory.profiler import profile_response, CognitiveProfile
from src.memory.store import (
    save_memory,
    find_related_memories,
    get_all_memories,
    init_memories_table,
)
from src.memory.echo import generate_echo_report, EchoReport


# === Cognitive Profiler 测试 ===


class TestProfiler:
    @patch("src.memory.profiler.chat_json")
    def test_extracts_cognitive_profile(self, mock_chat_json):
        """验证 Profiler 能从 LLM 返回中正确提取认知画像。"""
        mock_chat_json.return_value = {
            "core_preference": ["实用主义", "怀疑论"],
            "reasoning_style": "演绎推理",
            "emotional_tone": "冷静客观",
            "topic_keywords": ["AI", "就业", "自动化"],
            "stance_summary": "AI 会带来结构性失业，但新岗位会出现",
        }

        profile = profile_response(
            user_response="我认为 AI 会替代部分重复性工作，但创造力岗位暂时安全",
            article_title="AI 与就业的未来",
            article_summary="讨论 AI 对劳动市场的影响",
        )

        assert isinstance(profile, CognitiveProfile)
        assert "实用主义" in profile.core_preference
        assert profile.reasoning_style == "演绎推理"
        assert profile.emotional_tone == "冷静客观"
        assert len(profile.topic_keywords) >= 2
        assert len(profile.stance_summary) > 0

    @patch("src.memory.profiler.chat_json")
    def test_handles_minimal_response(self, mock_chat_json):
        """验证对极简回应也能提取画像。"""
        mock_chat_json.return_value = {
            "core_preference": ["怀疑论"],
            "reasoning_style": "直觉判断",
            "emotional_tone": "悲观防守",
            "topic_keywords": ["AI"],
            "stance_summary": "不看好",
        }

        profile = profile_response(user_response="不看好")
        assert profile.emotional_tone == "悲观防守"


# === Memory Store 测试 ===


def _make_fake_embedder(dim: int = 8):
    """返回一个确定性的假 embedder，用于测试。"""
    class FakeEmbedder:
        model_name = "fake-embed"
        def embed(self, texts):
            vecs = []
            for t in texts:
                rng = np.random.RandomState(hash(t) % 2**31)
                v = rng.randn(dim).astype(np.float32)
                v /= np.linalg.norm(v)
                vecs.append(v)
            return vecs
    return FakeEmbedder()


class TestStore:
    def setup_method(self):
        """每个测试用临时数据库 + mock embedder。"""
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._fake_embedder = _make_fake_embedder()
        self._patchers = [
            patch("src.memory.store.DB_PATH", self._tmp.name),
            patch("src.memory.embedder.get_embedder", return_value=self._fake_embedder),
        ]
        for p in self._patchers:
            p.start()

    def teardown_method(self):
        for p in reversed(self._patchers):
            p.stop()
        from src.memory.embedder import reset_embedder
        reset_embedder()
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_save_and_retrieve(self):
        """验证保存和检索的完整流程（向量召回）。"""
        profile = CognitiveProfile(
            core_preference=["实用主义"],
            reasoning_style="演绎推理",
            emotional_tone="冷静客观",
            topic_keywords=["AI", "就业", "自动化"],
            stance_summary="AI 会改变就业结构",
        )

        memory_id = save_memory(
            article_id=1,
            article_title="AI 与就业",
            user_response="我认为 AI 会改变就业市场",
            profile=profile,
        )

        assert memory_id > 0

        # 通过向量召回（相似文本）
        results = find_related_memories("我认为 AI 会改变就业市场")
        assert len(results) >= 1
        assert results[0]["article_title"] == "AI 与就业"
        assert "实用主义" in results[0]["core_preference"]
        assert "similarity" in results[0]

    def test_keyword_fallback(self):
        """验证关键词回退：当向量召回失败时用 LIKE 匹配。"""
        profile_ai = CognitiveProfile(
            core_preference=["技术决定论"],
            reasoning_style="系统性思维",
            emotional_tone="乐观激进",
            topic_keywords=["AI", "深度学习"],
            stance_summary="AI 是未来",
        )
        profile_food = CognitiveProfile(
            core_preference=["人文关怀"],
            reasoning_style="归纳猜想",
            emotional_tone="审慎乐观",
            topic_keywords=["食品安全", "有机农业"],
            stance_summary="有机食品更健康",
        )

        save_memory(1, "AI Revolution", "AI is great", profile_ai)
        save_memory(2, "Organic Food", "Organic is better", profile_food)

        # 强制向量搜索失败，触发 keyword fallback
        with patch("src.memory.store._vector_search", return_value=[]):
            ai_results = find_related_memories("AI 深度学习是未来")
            assert len(ai_results) >= 1
            assert ai_results[0]["article_title"] == "AI Revolution"

            food_results = find_related_memories("食品安全很重要")
            assert len(food_results) >= 1
            assert food_results[0]["article_title"] == "Organic Food"

    def test_get_all_memories(self):
        """验证获取全部记忆。"""
        profile = CognitiveProfile(
            core_preference=["怀疑论"],
            reasoning_style="直觉判断",
            emotional_tone="悲观防守",
            topic_keywords=["test"],
            stance_summary="test stance",
        )

        save_memory(1, "Article 1", "Response 1", profile)
        save_memory(2, "Article 2", "Response 2", profile)

        all_mems = get_all_memories()
        assert len(all_mems) == 2
        # embedding blob should not be exposed
        assert "embedding" not in all_mems[0]


# === Echo Location 测试 ===


class TestEchoLocation:
    def test_no_history_returns_graceful_response(self):
        """验证无历史记录时返回友好提示。"""
        report = generate_echo_report(
            current_response="Some response",
            current_tags={"core_preference": ["实用主义"]},
            historical_memories=[],
        )

        assert isinstance(report, EchoReport)
        assert report.has_history is False
        assert len(report.growth_insight) > 0

    @patch("src.memory.echo.chat_json")
    def test_with_history_generates_comparison(self, mock_chat_json):
        """验证有历史记录时能生成对比报告。"""
        mock_chat_json.return_value = {
            "stance_shift": "从悲观转向审慎乐观",
            "reasoning_shift": "从直觉判断升级为系统性思维",
            "tone_drift": "情感基调从防守转为开放",
            "bias_alert": "你在技术话题上一直偏乐观，建议多听反面声音",
            "growth_insight": "你的思维正在从二元对立走向辩证融合，这是认知成熟的标志。",
        }

        history = [
            {
                "id": 1,
                "article_title": "AI 取代画师",
                "user_response": "AI 永远不可能有真正的创造力",
                "stance_summary": "AI 无法替代人类创造力",
                "core_preference": ["人文关怀"],
                "reasoning_style": "直觉判断",
                "emotional_tone": "悲观防守",
                "created_at": "2026-01-15T00:00:00",
            }
        ]

        report = generate_echo_report(
            current_response="AI 辅助创作可能是一种新范式",
            current_tags={
                "core_preference": ["实用主义"],
                "reasoning_style": "系统性思维",
                "emotional_tone": "审慎乐观",
                "stance_summary": "AI 辅助创作是可接受的新范式",
            },
            historical_memories=history,
        )

        assert report.has_history is True
        assert "悲观" in report.stance_shift or "乐观" in report.stance_shift
        assert report.bias_alert is not None
        assert len(report.growth_insight) > 0

    @patch("src.memory.echo.chat_json")
    def test_no_bias_returns_null(self, mock_chat_json):
        """验证无偏见时 bias_alert 为 null。"""
        mock_chat_json.return_value = {
            "stance_shift": "无明显变化",
            "reasoning_shift": "保持一致",
            "tone_drift": "无漂移",
            "bias_alert": None,
            "growth_insight": "思维保持一致。",
        }

        report = generate_echo_report(
            current_response="test",
            current_tags={},
            historical_memories=[{"id": 1, "created_at": "2026-01-01"}],
        )

        assert report.bias_alert is None
