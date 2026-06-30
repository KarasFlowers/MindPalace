"""Scout 流水线测试。使用 mock 避免真实 LLM 调用。"""

import json
from unittest.mock import patch, MagicMock
import pytest

from src.scout.fetch import RawArticle
from src.scout.normalize import normalize, NormalizedArticle, _strip_html, _truncate
from src.scout.score import score_article, _compute_total, ScoredArticle


# === Normalize 测试 ===


class TestStripHtml:
    def test_removes_tags(self):
        assert _strip_html("<p>hello <b>world</b></p>") == "hello world"

    def test_removes_entities(self):
        assert _strip_html("foo&amp;bar") == "foo bar"

    def test_collapses_whitespace(self):
        assert _strip_html("  a   b  \n  c  ") == "a b c"

    def test_empty_string(self):
        assert _strip_html("") == ""


class TestTruncate:
    def test_short_text_unchanged(self):
        text = "This is short."
        assert _truncate(text, max_len=100) == text

    def test_long_text_truncated(self):
        text = "A" * 5000
        result = _truncate(text, max_len=100)
        assert len(result) <= 104  # 100 + "..."

    def test_truncates_at_sentence(self):
        text = "First sentence. Second sentence. " + "A" * 4000
        result = _truncate(text, max_len=50)
        assert result.endswith(".")


class TestNormalize:
    def test_basic_normalization(self):
        raw = RawArticle(
            url="https://example.com/1",
            title="  Test Article  ",
            content="<p>Hello <b>world</b></p>",
            source="Test Source",
            published_at="2026-01-01",
        )
        result = normalize(raw)
        assert isinstance(result, NormalizedArticle)
        assert result.title == "Test Article"
        assert result.clean_content == "Hello world"
        assert result.content == raw.content  # 原始内容保留


# === Score 测试 ===


class TestComputeTotal:
    def test_perfect_scores(self):
        scores = {
            "information_density": 10,
            "principle_depth": 10,
            "causal_chain": 10,
        }
        assert _compute_total(scores) == 10.0

    def test_weighted_average(self):
        scores = {
            "information_density": 10,  # weight 0.3 -> 3.0
            "principle_depth": 10,       # weight 0.4 -> 4.0
            "causal_chain": 0,           # weight 0.3 -> 0.0
        }
        assert _compute_total(scores) == 7.0

    def test_principle_depth_weighted_higher(self):
        """原理深度权重最高，相同分数下应对总分影响更大。"""
        # 只有原理深度高
        scores_a = {"information_density": 5, "principle_depth": 10, "causal_chain": 5}
        # 只有信息密度高
        scores_b = {"information_density": 10, "principle_depth": 5, "causal_chain": 5}
        assert _compute_total(scores_a) > _compute_total(scores_b)


class TestScoreArticle:
    @patch("src.scout.score.chat_json")
    def test_returns_scored_article(self, mock_chat_json):
        mock_chat_json.return_value = {
            "information_density": 8,
            "principle_depth": 9,
            "causal_chain": 7,
            "summary": "一篇关于测试的文章",
            "reasoning": "逻辑清晰，原理深入",
        }

        article = NormalizedArticle(
            url="https://example.com/1",
            title="Test",
            content="raw content",
            clean_content="clean content",
            source="Test Source",
            published_at="2026-01-01",
        )

        result = score_article(article)

        assert isinstance(result, ScoredArticle)
        assert result.scores["information_density"] == 8
        assert result.scores["principle_depth"] == 9
        assert result.scores["causal_chain"] == 7
        assert result.summary == "一篇关于测试的文章"
        assert result.total_score > 0
        assert result.source_lang == "unknown"
        assert result.translated is False

    @patch("src.scout.score.chat_json")
    def test_carries_translation_metadata(self, mock_chat_json):
        mock_chat_json.return_value = {
            "information_density": 8,
            "principle_depth": 9,
            "causal_chain": 7,
            "summary": "一篇关于测试的文章",
            "reasoning": "逻辑清晰，原理深入",
        }

        article = NormalizedArticle(
            url="https://example.com/1",
            title="Test",
            content="raw content",
            clean_content="clean content",
            source="Test Source",
            published_at="2026-01-01",
        )
        article.source_lang = "other"
        article.translated = True

        result = score_article(article)

        assert result.source_lang == "other"
        assert result.translated is True

    @patch("src.scout.score.chat_json")
    def test_sorting_by_total_score(self, mock_chat_json):
        """测试评分排序：高分在前。"""
        from src.scout.score import score_all

        # 第一篇高分，第二篇低分
        mock_chat_json.side_effect = [
            {
                "information_density": 3,
                "principle_depth": 2,
                "causal_chain": 3,
                "summary": "低分文章",
                "reasoning": "内容平庸",
            },
            {
                "information_density": 9,
                "principle_depth": 9,
                "causal_chain": 8,
                "summary": "高分文章",
                "reasoning": "深度好文",
            },
        ]

        articles = [
            NormalizedArticle(
                url=f"https://example.com/{i}",
                title=f"Article {i}",
                content="",
                clean_content="content",
                source="Test",
                published_at="2026-01-01",
            )
            for i in range(2)
        ]

        results = score_all(articles)

        assert len(results) == 2
        assert results[0].summary == "高分文章"
        assert results[1].summary == "低分文章"
        assert results[0].total_score > results[1].total_score


class TestPipelineTranslation:
    @patch("src.scout.pipeline.score_all", return_value=[])
    @patch("src.scout.pipeline.maybe_translate_article")
    def test_translate_all_updates_content_before_scoring(self, mock_translate, mock_score_all):
        from src.scout.pipeline import translate_all

        article = NormalizedArticle(
            url="https://example.com/1",
            title="Test",
            content="raw",
            clean_content="English content",
            source="Test",
            published_at="2026-01-01",
        )
        mock_translate.return_value = ("other", "中文内容", True)

        translated = translate_all([article], provider_config={"models": ["m"]})

        assert translated[0].clean_content == "中文内容"
        assert translated[0].source_lang == "other"
        assert translated[0].translated is True
        mock_score_all.assert_not_called()
