"""Phase B: Embedder + 向量召回 + Crystallize + Trajectory 测试。"""

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.memory.embedder import (
    OpenAIEmbedder,
    cosine_similarity,
    get_embedder,
    vec_to_blob,
    blob_to_vec,
    reset_embedder,
    build_enhanced_text,
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
    """为相似文本返回方向相近的向量（token 叠加）。

    采用 token 叠加而非整段哈希：这样增强文本（content + 元数据）与原始
    content 共享大部分 token，向量方向接近，更贴近真实 embedding 模型，
    也能验证 A-MEM 增强嵌入对召回质量的提升。
    """
    import re as _re
    v = np.zeros(dim, dtype=np.float32)
    tokens = _re.findall(r"[\u4e00-\u9fff]|[a-zA-Z]+", text.lower())
    for tok in tokens:
        seed = hash(tok) % 2**31
        rng = np.random.RandomState(seed)
        v += rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


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
    def test_get_embedder_rejects_anthropic_provider_type(self):
        reset_embedder()
        cfg = {
            "provider_type": "anthropic",
            "api_key": "sk-test",
            "base_url": "https://api.anthropic.com/v1",
            "models": ["text-embedding-3-small"],
        }
        with patch("src.memory.embedder.get_embedding_config", return_value=cfg):
            with pytest.raises(RuntimeError, match="Embedding Provider 不能使用 Anthropic"):
                get_embedder()

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
# A-MEM 增强嵌入文本构建 (Phase 升级)
# ---------------------------------------------------------------------------

class TestEnhancedText:
    def test_content_only_when_no_metadata(self):
        """无元数据时仅返回 content（查询端用）。"""
        result = build_enhanced_text(content="这是一段回答")
        assert result == "这是一段回答"

    def test_includes_all_metadata_fields(self):
        """完整元数据时拼接所有字段。"""
        result = build_enhanced_text(
            content="我喜欢真实",
            stance_summary="更看重真实",
            topic_keywords=["真实", "意义"],
            core_preference=["人文关怀"],
        )
        assert "我喜欢真实" in result
        assert "stance: 更看重真实" in result
        assert "keywords: 真实, 意义" in result
        assert "preferences: 人文关怀" in result

    def test_skips_empty_metadata(self):
        """空字段不拼接。"""
        result = build_enhanced_text(
            content="回答",
            stance_summary="",
            topic_keywords=[],
            core_preference=["理想主义"],
        )
        assert "stance" not in result
        assert "keywords" not in result
        assert "preferences: 理想主义" in result

    def test_query_and_storage_format_share_content(self):
        """查询端（仅 content）与存储端（content + 元数据）共享 content 部分，
        保证两侧格式一致、余弦相似度可比。"""
        content = "AI 深度学习改变一切"
        storage_text = build_enhanced_text(
            content=content,
            stance_summary="AI 改变世界",
            topic_keywords=["AI", "深度学习"],
        )
        query_text = build_enhanced_text(content=content)
        # content 必须在两侧出现
        assert content in storage_text
        assert content in query_text
        # 存储侧多了元数据标记
        assert len(storage_text) > len(query_text)


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

    def test_rebuild_embeddings_migrates_null_vectors(self):
        """rebuild_embeddings 应为无 embedding 的旧记录补充增强向量。"""
        from src.memory.store import rebuild_embeddings
        from src.storage.db import _get_conn

        # 正常保存两条（带 embedding）
        save_memory(1, "A", "AI 改变世界", _PROFILE_AI)
        # 第二条先把 embedding 清空模拟旧数据
        mid2 = save_memory(2, "B", "食品与健康", _PROFILE_FOOD)
        with _get_conn() as conn:
            conn.execute(
                "UPDATE memories SET embedding = NULL WHERE id = ?", (mid2,)
            )

        count = rebuild_embeddings()
        assert count == 1

        with _get_conn() as conn:
            row = conn.execute(
                "SELECT embedding FROM memories WHERE id = ?", (mid2,)
            ).fetchone()
        assert row["embedding"] is not None


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
        """达到 window 阈值时触发结晶（结构化输出）。"""
        from src.memory.crystallize import crystallize_if_needed

        mock_chat.return_value = json.dumps({
            "type": "principle",
            "content": "你倾向于用技术决定论看待问题。",
            "confidence": 0.8,
            "reasoning": "多次发言都体现技术乐观",
            "tags": ["技术", "乐观"],
        })

        for i in range(12):
            save_memory(i, f"Article {i}", f"Response about AI topic {i}", _PROFILE_AI)

        with patch("src.memory.crystallize._get_last_anchor", return_value=0), \
             patch("src.memory.crystallize._save_crystal", return_value=1) as mock_save, \
             patch("src.memory.crystallize._append_to_user_profile") as mock_append:
            result = crystallize_if_needed(window=10)

        assert result is not None
        assert result["type"] == "principle"
        assert "技术决定论" in result["content"]
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["status"] == "candidate"
        assert isinstance(result["sources"], list)
        mock_save.assert_called_once()
        mock_append.assert_called_once()

    @patch("src.memory.crystallize.chat")
    def test_fallback_when_llm_returns_plain_text(self, mock_chat):
        """LLM 返回非 JSON 文本时降级为 observation。"""
        from src.memory.crystallize import crystallize_if_needed

        mock_chat.return_value = "你更看重实用价值。"  # 非 JSON

        for i in range(12):
            save_memory(i, f"Article {i}", f"Response {i}", _PROFILE_AI)

        with patch("src.memory.crystallize._get_last_anchor", return_value=0), \
             patch("src.memory.crystallize._save_crystal", return_value=1), \
             patch("src.memory.crystallize._append_to_user_profile"):
            result = crystallize_if_needed(window=10)

        assert result is not None
        assert result["type"] == "observation"
        assert "实用价值" in result["content"]
        assert result["confidence"] < 0.5  # 降级置信度低


class TestCrystalParsing:
    """直接测试 _parse_crystal 与渲染逻辑。"""

    def test_parse_valid_json(self):
        from src.memory.crystallize import _parse_crystal
        raw = json.dumps({
            "type": "axiom",
            "content": "我相信努力比天赋更重要",
            "confidence": 0.9,
            "reasoning": "反复提及",
            "tags": ["成长", "信念"],
        })
        result = _parse_crystal(raw, [1, 2])
        assert result["type"] == "axiom"
        assert result["confidence"] == 0.9
        assert result["sources"] == [1, 2]
        assert result["tags"] == ["成长", "信念"]

    def test_parse_invalid_type_normalized(self):
        from src.memory.crystallize import _parse_crystal
        raw = json.dumps({"type": "weird", "content": "x", "confidence": 2.0})
        result = _parse_crystal(raw, [])
        assert result["type"] == "observation"
        assert result["confidence"] == 1.0  # 被 clamp

    def test_parse_strips_code_fence(self):
        from src.memory.crystallize import _parse_crystal
        raw = "```json\n" + json.dumps({"type": "principle", "content": "y"}) + "\n```"
        result = _parse_crystal(raw, [3])
        assert result["type"] == "principle"
        assert result["content"] == "y"


# ---------------------------------------------------------------------------
# Brain export (Axiomind 结构化档案导出)
# ---------------------------------------------------------------------------

class TestBrainExport:
    def test_export_writes_markdown_files_by_type(self, tmp_path):
        """export_brain 应按 type 分目录导出带 frontmatter 的 Markdown。"""
        from src.memory.brain_export import export_brain
        from src.storage.db import init_db

        init_db()
        # 直接插入两条不同 type 的 crystal
        from src.storage.db import _get_conn
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO profile_crystals "
                "(content, anchor_memory_id, window, created_at, type, status, confidence, sources, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("信念陈述", 1, 10, "2026-06-14T00:00:00", "axiom", "candidate", 0.8,
                 json.dumps([1, 2]), json.dumps(["成长"])),
            )
            conn.execute(
                "INSERT INTO profile_crystals "
                "(content, anchor_memory_id, window, created_at, type, status, confidence, sources, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("观察模式", 2, 10, "2026-06-14T00:00:00", "observation", "candidate", 0.5,
                 json.dumps([3]), json.dumps([])),
            )

        count = export_brain(export_dir=tmp_path / "brain")

        assert count == 2
        axioms = list((tmp_path / "brain" / "axioms").glob("*.md"))
        observations = list((tmp_path / "brain" / "observations").glob("*.md"))
        assert len(axioms) == 1
        assert len(observations) == 1

        axiom_content = axioms[0].read_text(encoding="utf-8")
        assert "type: axiom" in axiom_content
        assert "status: candidate" in axiom_content
        assert "confidence: 0.8" in axiom_content
        assert "信念陈述" in axiom_content


# ---------------------------------------------------------------------------
# Crystal listing + terminal rendering（UX 入口支撑）
# ---------------------------------------------------------------------------

class TestCrystalListingAndRendering:
    def test_list_crystals_returns_parsed_dicts(self):
        """list_crystals 应返回解析过 JSON 字段的 dict 列表。"""
        from src.storage.db import init_db, list_crystals, _get_conn

        init_db()  # 确保 profile_crystals 表存在
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO profile_crystals "
                "(content, anchor_memory_id, window, created_at, type, status, confidence, sources, tags) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("洞察 A", 1, 10, "2026-06-14T00:00:00", "principle", "candidate", 0.7,
                 json.dumps([1, 2]), json.dumps(["成长"])),
            )

        crystals = list_crystals()
        assert len(crystals) >= 1
        cr = crystals[0]
        assert cr["type"] == "principle"
        assert cr["sources"] == [1, 2]
        assert cr["tags"] == ["成长"]
        assert cr["status"] == "candidate"

    def test_render_crystal_terminal_includes_type_and_content(self):
        """终端渲染应包含类型标签、置信度和内容。"""
        from src.memory.crystallize import render_crystal_terminal
        crystal = {
            "type": "axiom",
            "content": "我相信努力比天赋更重要",
            "confidence": 0.9,
            "reasoning": "反复提及",
            "tags": ["信念"],
            "sources": [1],
            "status": "candidate",
        }
        out = render_crystal_terminal(crystal)
        assert "Axiom" in out
        assert "0.90" in out
        assert "我相信努力比天赋更重要" in out
        assert "信念" in out  # tag

    def test_render_crystal_terminal_empty_metadata(self):
        """无元数据时也应正常渲染。"""
        from src.memory.crystallize import render_crystal_terminal
        crystal = {
            "type": "observation",
            "content": "简单观察",
            "confidence": 0.3,
            "reasoning": "",
            "tags": [],
            "sources": [],
            "status": "candidate",
        }
        out = render_crystal_terminal(crystal)
        assert "简单观察" in out
        assert "Observation" in out
