"""记忆演化引擎测试（A-MEM link_memories）。

覆盖：
- 无邻居时跳过演化
- strengthen 动作建立链接
- update_neighbor 动作更新邻居标签
- LLM 失败不阻断
- agentic 检索沿 links 遍历
- retrieval_count 自增
"""

import json
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from src.memory.profiler import CognitiveProfile
from src.memory.store import (
    find_related_memories,
    get_memory,
    save_memory,
    update_memory_links,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_embed(text: str, dim: int = 8) -> np.ndarray:
    """token 叠加向量：相似文本方向接近。"""
    import re as _re
    v = np.zeros(dim, dtype=np.float32)
    tokens = _re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", text.lower())
    for tok in tokens:
        rng = np.random.RandomState(hash(tok) % 2**31)
        v += rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class FakeEmbedder:
    model_name = "fake-embed-evolution"
    _dim = 8

    def embed(self, texts):
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
    _dbmod._db_initialized_for = saved_init  # 恢复缓存，避免污染后续测试
    from src.memory.embedder import reset_embedder
    reset_embedder()
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# link_memories
# ---------------------------------------------------------------------------

class TestLinkMemories:
    def test_no_neighbors_returns_not_evolved(self):
        """无邻居时（DB 空）应跳过演化。"""
        from src.memory.evolution import link_memories

        # 用一个不存在的 memory id，应返回 not found
        result = link_memories(9999)
        assert result["evolved"] is False
        assert "error" in result

    @patch("src.memory.evolution.chat_json")
    def test_strengthen_creates_links(self, mock_chat_json):
        """strengthen 动作应在新记忆建立指向邻居的链接。"""
        from src.memory.evolution import link_memories

        # 存一条相似的历史记忆
        old_id = save_memory(1, "AI 旧文", "AI 深度学习改变一切", _PROFILE_AI)
        # 存新记忆（内容相似，应能召回旧记忆）
        new_id = save_memory(2, "AI 新文", "AI 深度学习的新进展", _PROFILE_AI)

        # mock LLM 返回 strengthen，链接第 1 个邻居
        mock_chat_json.return_value = {
            "should_evolve": True,
            "action": "strengthen",
            "suggested_links": "1",
            "tags_to_update": {},
        }

        result = link_memories(new_id)
        assert result["evolved"] is True
        assert result["action"] == "strengthen"
        assert result["links_created"] >= 1

        # 验证新记忆的 links 字段包含旧记忆 id
        new_mem = get_memory(new_id)
        assert str(old_id) in new_mem["links"]

    @patch("src.memory.evolution.chat_json")
    def test_update_neighbor_modifies_tags(self, mock_chat_json):
        """update_neighbor 动作应合并邻居的 topic_keywords。"""
        from src.memory.evolution import link_memories

        old_id = save_memory(1, "AI 旧文", "AI 深度学习", _PROFILE_AI)
        new_id = save_memory(2, "AI 新文", "AI 深度学习新进展", _PROFILE_AI)

        mock_chat_json.return_value = {
            "should_evolve": True,
            "action": "update_neighbor",
            "suggested_links": "",
            "tags_to_update": {"1": ["新范式", "涌现"]},
        }

        result = link_memories(new_id)
        assert result["evolved"] is True
        assert result["neighbors_updated"] >= 1

        # 验证旧记忆的 keywords 被合并（旧 + 新）
        old_mem = get_memory(old_id)
        assert "新范式" in old_mem["topic_keywords"]
        assert "AI" in old_mem["topic_keywords"]  # 旧关键词保留

    @patch("src.memory.evolution.chat_json", side_effect=RuntimeError("LLM down"))
    def test_llm_failure_does_not_block(self, _mock):
        """LLM 决策失败应返回 not evolved + error，不抛异常。"""
        from src.memory.evolution import link_memories

        save_memory(1, "AI 文", "AI 深度学习", _PROFILE_AI)
        new_id = save_memory(2, "AI 文2", "AI 深度学习新进展", _PROFILE_AI)

        result = link_memories(new_id)
        assert result["evolved"] is False
        assert "error" in result

    @patch("src.memory.evolution.chat_json")
    def test_should_not_evolve_returns_none(self, mock_chat_json):
        """LLM 判定不演化时应返回 not evolved。"""
        from src.memory.evolution import link_memories

        save_memory(1, "AI 文", "AI 深度学习", _PROFILE_AI)
        new_id = save_memory(2, "AI 文2", "AI 深度学习新进展", _PROFILE_AI)

        mock_chat_json.return_value = {
            "should_evolve": False,
            "action": "none",
            "suggested_links": "",
            "tags_to_update": {},
        }

        result = link_memories(new_id)
        assert result["evolved"] is False
        assert result["action"] == "none"


# ---------------------------------------------------------------------------
# agentic 检索
# ---------------------------------------------------------------------------

class TestAgenticSearch:
    def test_agentic_search_traverses_links(self):
        """agentic=True 时应沿 links 追加邻居到结果。"""
        # 存两条相似记忆
        m1 = save_memory(1, "AI 1", "AI 深度学习改变一切", _PROFILE_AI)
        m2 = save_memory(2, "AI 2", "AI 深度学习新进展", _PROFILE_AI)

        # 手动给 m1 建一个指向 m2 的链接
        update_memory_links(m1, {str(m2): 0.9})

        # agentic 检索 "AI 深度学习" 应能通过 m1 的链接找到 m2
        results = find_related_memories("AI 深度学习", exclude_id=None, limit=5, agentic=True)
        ids = [r["id"] for r in results]
        assert m1 in ids
        assert m2 in ids

    def test_non_agentic_does_not_traverse(self):
        """agentic=False（默认）不应沿 links 遍历——对比 agentic=True 的差异。"""
        m1 = save_memory(1, "AI 1", "AI 深度学习改变一切", _PROFILE_AI)
        # 存一条完全不相似的记忆（向量召回不到，只能通过链接到达）
        m2 = save_memory(2, "Food", "有机食品健康饮食营养蔬菜", _PROFILE_FOOD)
        from src.memory.store import _get_conn
        with _get_conn() as conn:
            conn.execute("UPDATE memories SET embedding = NULL WHERE id = ?", (m2,))
            conn.commit()
        update_memory_links(m1, {str(m2): 0.9})

        # 非 agentic：纯向量召回，m2 不相关不应出现
        non_agentic = find_related_memories(
            "AI 深度学习", exclude_id=None, limit=5, agentic=False, min_similarity=0.35
        )
        # agentic：应通过 m1 的链接找到 m2
        agentic = find_related_memories(
            "AI 深度学习", exclude_id=None, limit=5, agentic=True, min_similarity=0.35
        )
        agentic_ids = {r["id"] for r in agentic}
        non_agentic_ids = {r["id"] for r in non_agentic}

        # 关键断言：agentic 比非 agentic 多召回了 m2（通过链接）
        assert m2 in agentic_ids, "agentic 应通过链接找到 m2"
        assert m2 not in non_agentic_ids, "非 agentic 不应通过链接遍历"


# ---------------------------------------------------------------------------
# retrieval_count
# ---------------------------------------------------------------------------

class TestRetrievalCount:
    def test_track_retrieval_increments_count(self):
        """track_retrieval=True 应递增命中记忆的 retrieval_count。"""
        m1 = save_memory(1, "AI 1", "AI 深度学习", _PROFILE_AI)

        # 召回并跟踪
        find_related_memories("AI 深度学习", track_retrieval=True)
        mem = get_memory(m1)
        assert mem["retrieval_count"] >= 1

        # 再召回一次，计数应增加
        find_related_memories("AI 深度学习", track_retrieval=True)
        mem = get_memory(m1)
        assert mem["retrieval_count"] >= 2

    def test_no_track_does_not_increment(self):
        """默认（track_retrieval=False）不应改变 retrieval_count。"""
        m1 = save_memory(1, "AI 1", "AI 深度学习", _PROFILE_AI)
        find_related_memories("AI 深度学习")  # 默认不跟踪
        mem = get_memory(m1)
        assert mem["retrieval_count"] == 0


# ---------------------------------------------------------------------------
# save_memory link_after_save 参数
# ---------------------------------------------------------------------------

class TestSaveWithLinking:
    @patch("src.memory.evolution.chat_json")
    def test_link_after_save_triggers_evolution(self, mock_chat_json):
        """link_after_save=True 应在保存后自动触发 link_memories。"""
        # 先存一条历史记忆
        save_memory(1, "旧", "AI 深度学习", _PROFILE_AI)

        mock_chat_json.return_value = {
            "should_evolve": True,
            "action": "strengthen",
            "suggested_links": "1",
            "tags_to_update": {},
        }

        new_id = save_memory(
            2, "新", "AI 深度学习新进展", _PROFILE_AI,
            link_after_save=True,
        )
        # link_memories 应被调用（mock 了 chat_json）
        mock_chat_json.assert_called_once()

    def test_no_link_after_save_skips_evolution(self):
        """link_after_save=False（默认）不应触发演化。"""
        save_memory(1, "旧", "AI 深度学习", _PROFILE_AI)
        with patch("src.memory.evolution.link_memories") as mock_link:
            save_memory(2, "新", "AI 新进展", _PROFILE_AI)
            mock_link.assert_not_called()
