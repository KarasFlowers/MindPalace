"""心智漫游基线 diff 测试（Axiomind 暂存区模式）。

覆盖：
- find_similar_answers 按 source_id 过滤
- compute_diff 检测相似 + LLM 失败回退
- check_and_describe_similarity 一站式
- save_inquiry_memory 检测到相似历史时触发提示
"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from src.inquiry.diff import (
    compute_diff,
    find_similar_answers,
    check_and_describe_similarity,
    _heuristic_diff,
)
from src.memory.profiler import CognitiveProfile
from src.memory.store import save_memory


# ---------------------------------------------------------------------------
# Helpers（与 test_memory_evolution 共享的 token 叠加 embedder）
# ---------------------------------------------------------------------------

def _deterministic_embed(text: str, dim: int = 8) -> np.ndarray:
    import re as _re
    v = np.zeros(dim, dtype=np.float32)
    tokens = _re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", text.lower())
    for tok in tokens:
        rng = np.random.RandomState(hash(tok) % 2**31)
        v += rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class FakeEmbedder:
    model_name = "fake-embed-diff"
    _dim = 8

    def embed(self, texts):
        return [_deterministic_embed(t, self._dim) for t in texts]


@pytest.fixture(autouse=True)
def _isolated_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    fake = FakeEmbedder()
    # 重置 init_db 缓存，确保新临时 DB 一定会重新建表（防止跨测试文件污染）
    import src.storage.db as _dbmod
    saved_init = _dbmod._db_initialized_for
    _dbmod._db_initialized_for = None
    patchers = [
        patch("src.storage.db.DB_PATH", tmp.name),
        patch("src.memory.embedder.get_embedder", return_value=fake),
    ]
    for p in patchers:
        p.start()
    yield fake
    for p in reversed(patchers):
        p.stop()
    _dbmod._db_initialized_for = saved_init
    from src.memory.embedder import reset_embedder
    reset_embedder()
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


def _profile_for(text: str) -> CognitiveProfile:
    return CognitiveProfile(
        core_preference=["实用主义"],
        reasoning_style="演绎推理",
        emotional_tone="冷静客观",
        topic_keywords=["AI"],
        stance_summary=text[:40],
    )


# ---------------------------------------------------------------------------
# find_similar_answers
# ---------------------------------------------------------------------------

class TestFindSimilarAnswers:
    def test_filters_by_source_id(self):
        """只应返回同一问题卡的历史回答，不含其他卡的。"""
        # 同一问题卡 card_A 的历史回答
        save_memory(None, "[心智漫游] 卡A", "AI 深度学习改变一切", _profile_for("AI"),
                    source_type="self", source_id="card_A")
        # 不同问题卡 card_B 的回答（内容也相似，应被过滤掉）
        save_memory(None, "[心智漫游] 卡B", "AI 深度学习是未来", _profile_for("AI未来"),
                    source_type="self", source_id="card_B")

        # 用较低 min_similarity 确保召回（本测试聚焦 source_id 过滤，非向量质量）
        results = find_similar_answers("card_A", "AI 深度学习", limit=3, min_similarity=0.1)
        assert len(results) >= 1, "应至少召回 card_A 的历史回答"
        assert all(r.get("source_id") == "card_A" for r in results), "不应混入其他卡"

    def test_no_history_returns_empty(self):
        """无历史回答时返回空列表。"""
        results = find_similar_answers("card_X", "任何内容", limit=3)
        assert results == []

    def test_empty_response_returns_empty(self):
        """空回答不查询。"""
        results = find_similar_answers("card_A", "", limit=3)
        assert results == []


# ---------------------------------------------------------------------------
# compute_diff
# ---------------------------------------------------------------------------

class TestComputeDiff:
    @patch("src.inquiry.diff.chat_json")
    def test_detects_similarity(self, mock_chat_json):
        """LLM 返回相似时应识别。"""
        mock_chat_json.return_value = {
            "is_similar": True,
            "what_changed": "立场一致但理由更充分",
            "novelty": 0.3,
        }
        result = compute_diff("当前回答", "历史回答", similarity=0.7)
        assert result["is_similar"] is True
        assert "立场一致" in result["what_changed"]
        assert result["novelty"] == 0.3

    @patch("src.inquiry.diff.chat_json", side_effect=RuntimeError("LLM down"))
    def test_fallback_on_llm_failure(self, _mock):
        """LLM 失败时回退到启发式判断。"""
        result = compute_diff("当前", "历史", similarity=0.9)
        assert result["is_similar"] is True  # similarity > 0.8
        assert 0.0 <= result["novelty"] <= 1.0

    def test_heuristic_diff_high_similarity(self):
        """高相似度启发式判为相似。"""
        result = _heuristic_diff("a", "b", similarity=0.9)
        assert result["is_similar"] is True
        assert result["novelty"] < 0.3

    def test_heuristic_diff_low_similarity(self):
        """低相似度启发式判为不相似。"""
        result = _heuristic_diff("a", "b", similarity=0.5)
        assert result["is_similar"] is False
        assert result["novelty"] > 0.4

    def test_empty_input_returns_novel(self):
        """空输入应判为新颖（不相似）。"""
        result = compute_diff("", "历史回答")
        assert result["is_similar"] is False
        assert result["novelty"] == 1.0


# ---------------------------------------------------------------------------
# check_and_describe_similarity（一站式）
# ---------------------------------------------------------------------------

class TestCheckAndDescribeSimilarity:
    def test_returns_none_when_no_history(self):
        """无相似历史时返回 None。"""
        result = check_and_describe_similarity("card_none", "任何内容")
        assert result is None

    @patch("src.inquiry.diff.chat_json")
    def test_returns_historical_and_diff(self, mock_chat_json):
        """有相似历史时应返回 historical + diff。"""
        save_memory(None, "[心智漫游] 卡A", "AI 深度学习改变一切", _profile_for("AI"),
                    source_type="self", source_id="card_A")

        mock_chat_json.return_value = {
            "is_similar": True,
            "what_changed": "更乐观了",
            "novelty": 0.4,
        }
        result = check_and_describe_similarity("card_A", "AI 深度学习新进展")
        assert result is not None
        assert "historical" in result
        assert "diff" in result
        assert result["diff"]["is_similar"] is True


# ---------------------------------------------------------------------------
# save_inquiry_memory 集成（相似历史提示）
# ---------------------------------------------------------------------------

class TestSaveInquiryMemorySimilarityPrompt:
    def test_prompts_on_similar_history(self):
        """检测到相似历史回答时应打印提示（不阻断保存）。"""
        from src.inquiry.session import save_inquiry_memory
        from src.inquiry.library import get_card

        card = get_card("forgive_yourself")  # 用现有卡
        # 先存一条历史回答
        with patch("src.inquiry.session.profile_response") as mock_prof, \
             patch("src.inquiry.session.generate_echo_report"), \
             patch("src.inquiry.session.format_echo_report", return_value=""), \
             patch("src.memory.evolution.chat_json") as mock_evo:  # 演化不干扰
            mock_prof.return_value = _profile_for("原谅")
            mock_evo.return_value = {"should_evolve": False, "action": "none"}
            save_inquiry_memory(card, "我上次原谅自己是去年，因为一个误会")

        # 再存一条相似回答，应触发 diff 提示
        with patch("src.inquiry.session.profile_response") as mock_prof, \
             patch("src.inquiry.session.generate_echo_report"), \
             patch("src.inquiry.session.format_echo_report", return_value=""), \
             patch("src.inquiry.diff.chat_json") as mock_diff, \
             patch("src.memory.evolution.chat_json") as mock_evo, \
             patch("builtins.print") as mock_print:
            mock_prof.return_value = _profile_for("原谅")
            mock_diff.return_value = {
                "is_similar": True,
                "what_changed": "这次更宽容了",
                "novelty": 0.4,
            }
            mock_evo.return_value = {"should_evolve": False, "action": "none"}
            save_inquiry_memory(card, "我上次原谅自己是因为那个误会")

        # 应有打印包含"也回答过类似问题"的提示
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "也回答过类似问题" in printed
