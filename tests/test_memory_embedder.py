"""Phase B: Embedder + 向量召回 + Crystallize + Trajectory 测试。"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.memory.embedder import (
    OpenAIEmbedder,
    cosine_similarity,
    vec_to_blob,
    blob_to_vec,
    reset_embedder,
)
from src.memory.profiler import CognitiveProfile
from src.memory.store import (
    save_memory,
    find_related_memories,
    get_all_memories,
    get_latest_memory_id,
    count_memories_since,
    get_recent_memories,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_embed(text: str, dim: int = 8) -> np.ndarray:
    """为同一文本返回相同的单位向量。"""
    rng = np.random.RandomState(hash(text) % 2**31)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


class FakeEmbedder:
    """测试用 embedder，基于文本哈希的确定性向量。"""
    model_name = "fake-embed-test"
    _dim = 8

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [_deterministic_embed(t, self._dim) for t in texts]


_PROFILE_AI = CognitiveProfile(
    core_preference=["技术决定论"],
    reasoning_style="系统性思维",
    emotional_tone="乐观激进",
    topic_keywords=["AI", "深度学习", "神经网络"],
    stance_summary="AI 将彻底改变世界",
)

_PROFILE_FOOD = CognitiveProfile(
    core_preference=["人文关怀"],
    reasoning_style="归纳猜想",
    emotional_tone="审慎乐观",
    topic_keywords=["食品安全", "有机农业"],
    stance_summary="有机食品更健康",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_db():
    """每个测试用独立的临时 DB + 假 embedder。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    fake = FakeEmbedder()

    patchers = [
        patch("src.memory.store.DB_PATH", tmp.name),
        patch("src.storage.db.DB_PATH", tmp.name),
        patch("src.memory.embedder.get_embedder", return_value=fake),
    ]
    for p in patchers:
        p.start()

    yield fake

    for p in reversed(patchers):
        p.stop()
    reset_embedder()
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# cosine / blob roundtrip
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_cosine_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_cosine_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_cosine_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_blob_roundtrip(self):
        original = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        blob = vec_to_blob(original)
        restored = blob_to_vec(blob)
        np.testing.assert_array_almost_equal(original, restored)


# ---------------------------------------------------------------------------
# 向量召回 ranking
# ---------------------------------------------------------------------------

class TestVectorRecall:
    def test_similar_text_ranked_higher(self):
        """语义相近的 memory 排在前面。"""
        save_memory(1, "AI Future", "AI 深度学习改变一切", _PROFILE_AI)
        save_memory(2, "Organic", "有机蔬菜与健康", _PROFILE_FOOD)

        results = find_related_memories("AI 深度学习改变一切")
        assert len(results) >= 1
        assert results[0]["article_title"] == "AI Future"
        assert results[0]["similarity"] >= 0.35

    def test_unrelated_text_low_similarity(self):
        """完全不相关的查询不应召回。"""
        save_memory(1, "AI Future", "AI 深度学习改变一切", _PROFILE_AI)

        results = find_related_memories(
            "量子纠缠态的退相干时间与温度的非线性关系",
            min_similarity=0.9,
        )
        assert len(results) == 0

    def test_exclude_id(self):
        """exclude_id 应过滤掉指定记忆。"""
        mid = save_memory(1, "AI Future", "AI 深度学习改变一切", _PROFILE_AI)
        results = find_related_memories("AI 深度学习改变一切", exclude_id=mid)
        matching = [r for r in results if r.get("id") == mid]
        assert len(matching) == 0

    def test_empty_query(self):
        assert find_related_memories("") == []
        assert find_related_memories("   ") == []

    def test_embed_failure_falls_back_to_keywords(self):
        """embedder 异常时走关键词回退。"""
        save_memory(1, "AI Future", "AI 深度学习", _PROFILE_AI)

        with patch("src.memory.store._vector_search", return_value=[]):
            results = find_related_memories("AI 深度学习的研究进展")
            # keyword fallback should pick up "AI" or "深度学习"
            assert len(results) >= 1


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestStoreHelpers:
    def test_get_latest_memory_id_empty(self):
        assert get_latest_memory_id() is None

    def test_get_latest_memory_id(self):
        save_memory(1, "A", "resp", _PROFILE_AI)
        mid2 = save_memory(2, "B", "resp2", _PROFILE_AI)
        assert get_latest_memory_id() == mid2

    def test_count_memories_since(self):
        mid1 = save_memory(1, "A", "resp", _PROFILE_AI)
        save_memory(2, "B", "resp2", _PROFILE_AI)
        save_memory(3, "C", "resp3", _PROFILE_AI)
        assert count_memories_since(mid1) == 2

    def test_get_recent_memories(self):
        for i in range(5):
            save_memory(i, f"Art {i}", f"Response {i}", _PROFILE_AI)
        recent = get_recent_memories(limit=3)
        assert len(recent) == 3
        # should be in ascending id order
        assert recent[0]["id"] < recent[1]["id"] < recent[2]["id"]


# ---------------------------------------------------------------------------
# Crystallize
# ---------------------------------------------------------------------------

class TestCrystallize:
    def test_skipped_when_not_enough_memories(self):
        """不够 window 条时不触发。"""
        from src.memory.crystallize import crystallize_if_needed

        # mock db init to also create profile_crystals table
        from src.storage.db import init_db
        init_db_patcher = patch(
            "src.memory.crystallize.init_db",
            side_effect=lambda: None,
        )
        # We need the profile_crystals table — create it via direct SQL
        import sqlite3
        tmp_name = os.environ.get("_TEST_DB")  # not set, use the patched DB_PATH

        save_memory(1, "A", "resp", _PROFILE_AI)  # only 1 memory

        with patch("src.memory.crystallize._get_last_anchor", return_value=0):
            result = crystallize_if_needed(window=10)
        assert result is None

    @patch("src.memory.crystallize.chat")
    def test_triggers_when_enough_memories(self, mock_chat):
        """达到 window 阈值时触发结晶。"""
        from src.memory.crystallize import crystallize_if_needed

        mock_chat.return_value = "你倾向于用技术决定论看待问题。"

        for i in range(12):
            save_memory(i, f"Article {i}", f"Response about AI topic {i}", _PROFILE_AI)

        with patch("src.memory.crystallize._get_last_anchor", return_value=0), \
             patch("src.memory.crystallize._save_crystal", return_value=1) as mock_save, \
             patch("src.memory.crystallize._append_to_user_profile") as mock_append:
            result = crystallize_if_needed(window=10)

        assert result is not None
        assert "技术决定论" in result
        mock_save.assert_called_once()
        mock_append.assert_called_once()
