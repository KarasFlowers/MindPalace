"""Scout 翻译模块测试。"""

from unittest.mock import patch

from src.scout.normalize import NormalizedArticle
from src.scout.translate import detect_language, maybe_translate_article, translate_content


def _article(clean_content: str = "This is a test article.") -> NormalizedArticle:
    return NormalizedArticle(
        url="https://example.com/1",
        title="Test",
        content=clean_content,
        clean_content=clean_content,
        source="Example",
        published_at="2026-01-01",
    )


def test_detect_language_identifies_chinese_text():
    assert detect_language("这是一个中文段落，用来验证语言检测。") == "zh"


def test_detect_language_identifies_non_chinese_text():
    assert detect_language("This is an English paragraph for language detection.") == "other"


@patch("src.scout.translate.chat_json")
def test_translate_content_returns_translated_payload(mock_chat_json):
    mock_chat_json.return_value = {"translated": "这是一篇测试文章。"}

    result = translate_content("This is a test article.", "Test", provider_config={"models": ["m"]})

    assert result == "这是一篇测试文章。"


@patch("src.scout.translate.chat_json", side_effect=RuntimeError("boom"))
def test_translate_content_falls_back_to_original_on_failure(_mock_chat_json):
    text = "This is a test article."

    assert translate_content(text, "Test", provider_config={"models": ["m"]}) == text


@patch("src.scout.translate.translate_content", return_value="这是一篇测试文章。")
def test_maybe_translate_article_translates_non_chinese_when_enabled(mock_translate):
    source_lang, content, translated = maybe_translate_article(
        _article(), provider_config={"models": ["m"]}, enabled=True
    )

    assert source_lang == "other"
    assert content == "这是一篇测试文章。"
    assert translated is True
    mock_translate.assert_called_once()


@patch("src.scout.translate.translate_content")
def test_maybe_translate_article_skips_chinese_article(mock_translate):
    source_lang, content, translated = maybe_translate_article(
        _article("这是一篇中文文章。"), enabled=True
    )

    assert source_lang == "zh"
    assert content == "这是一篇中文文章。"
    assert translated is False
    mock_translate.assert_not_called()


@patch("src.scout.translate.translate_content", return_value="This is a test article.")
def test_maybe_translate_article_does_not_mark_fallback_as_translated(_mock_translate):
    source_lang, content, translated = maybe_translate_article(
        _article(), provider_config={"models": ["m"]}, enabled=True
    )

    assert source_lang == "other"
    assert content == "This is a test article."
    assert translated is False
