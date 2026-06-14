"""心智漫游模块测试。"""

import os
import tempfile
from unittest.mock import patch

import pytest

from src.inquiry.analysis import analyze_response
from src.inquiry.library import InquiryLibraryError, choose_random_card, get_card, load_cards
from src.inquiry.session import save_inquiry_memory
from src.inquiry.types import PromptCard
from src.memory.profiler import CognitiveProfile
from src.memory.store import get_memories_by_source


@pytest.fixture
def _isolated_memory_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    patcher = patch("src.storage.db.DB_PATH", tmp.name)
    patcher.start()
    yield
    patcher.stop()
    try:
        os.unlink(tmp.name)
    except OSError:
        pass


def test_load_cards_by_kind():
    cards = load_cards("self")
    assert cards
    assert all(card.kind == "self" for card in cards)


def test_load_all_cards_includes_thought_experiments():
    cards = load_cards()
    kinds = {card.kind for card in cards}
    assert {"self", "philosophy", "thought_experiment"}.issubset(kinds)
    assert any(card.context for card in cards if card.kind == "thought_experiment")


def test_get_card_finds_prompt_card():
    card = get_card("experience_machine")
    assert card is not None
    assert card.title == "体验机器"
    assert card.twists


def test_choose_random_card_uses_loaded_kind(monkeypatch):
    expected = load_cards("philosophy")[0]
    monkeypatch.setattr("src.inquiry.library.random.choice", lambda cards: cards[0])
    assert choose_random_card("philosophy") == expected


def test_prompt_card_from_dict_normalizes_optional_lists():
    card = PromptCard.from_dict(
        {
            "id": " x ",
            "kind": " self ",
            "title": " t ",
            "prompt": " p ",
            "tags": [" a ", ""],
        }
    )
    assert card.id == "x"
    assert card.kind == "self"
    assert card.tags == ["a"]
    assert card.followups == []


def test_load_cards_reports_bad_card_fields(tmp_path, monkeypatch):
    data_dir = tmp_path / "inquiry"
    data_dir.mkdir()
    (data_dir / "self.json").write_text('[{"id": "bad", "kind": "self"}]', encoding="utf-8")
    monkeypatch.setattr("src.inquiry.library.INQUIRY_DATA_DIR", data_dir)
    with pytest.raises(InquiryLibraryError):
        load_cards("self")


@patch("src.inquiry.analysis.chat_json")
def test_analyze_response_returns_normalized_json(mock_chat_json):
    mock_chat_json.return_value = {
        "core_stance": " 重视真实 ",
        "hidden_assumption": " 快乐不能完全替代真实 ",
        "reflection": " 你更在乎真实感。 ",
        "followup_question": " 如果真实带来痛苦呢？ ",
    }
    card = get_card("experience_machine")
    result = analyze_response(card, "我不愿意进入。")
    assert result["core_stance"] == "重视真实"
    assert result["followup_question"] == "如果真实带来痛苦呢？"


@patch("src.inquiry.analysis.chat_json", side_effect=RuntimeError("boom"))
def test_analyze_response_fallback_does_not_raise(_mock_chat_json):
    card = get_card("experience_machine")
    result = analyze_response(card, "我不愿意进入。")
    assert "error" in result
    assert result["followup_question"]


@pytest.mark.usefixtures("_isolated_memory_db")
def test_save_inquiry_memory_still_saves_when_profile_fails():
    card = get_card("happy_without_achievement")
    with patch("src.inquiry.session.profile_response", side_effect=RuntimeError("profile down")), \
         patch("src.inquiry.session.find_related_memories", return_value=[]), \
         patch("src.inquiry.session.generate_echo_report") as mock_echo, \
         patch("src.inquiry.session.format_echo_report", return_value="echo"):
        mock_echo.return_value = object()
        memory_id = save_inquiry_memory(card, "我觉得快乐更重要。")

    assert memory_id > 0
    memories = get_memories_by_source("philosophy")
    assert len(memories) == 1
    assert memories[0]["topic_keywords"] == card.tags


@pytest.mark.usefixtures("_isolated_memory_db")
def test_save_inquiry_memory_records_source_metadata():
    card = get_card("happy_without_achievement")
    profile = CognitiveProfile(
        core_preference=["体验优先"],
        reasoning_style="价值判断",
        emotional_tone="平静",
        topic_keywords=["快乐", "成就"],
        stance_summary="快乐比成就更重要",
    )
    with patch("src.inquiry.session.profile_response", return_value=profile), \
         patch("src.inquiry.session.find_related_memories", return_value=[]), \
         patch("src.inquiry.session.generate_echo_report") as mock_echo, \
         patch("src.inquiry.session.format_echo_report", return_value="echo"):
        mock_echo.return_value = object()
        memory_id = save_inquiry_memory(card, "我觉得快乐更重要。")

    assert memory_id > 0
    memories = get_memories_by_source("philosophy")
    assert len(memories) == 1
    assert memories[0]["source_type"] == "philosophy"
    assert memories[0]["source_id"] == "happy_without_achievement"
