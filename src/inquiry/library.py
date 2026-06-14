"""心智漫游问题卡加载器。"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import PROJECT_ROOT
from src.inquiry.types import PromptCard

INQUIRY_DATA_DIR = PROJECT_ROOT / "data" / "inquiry"
_KIND_FILES = {
    "self": "self.json",
    "philosophy": "philosophy.json",
    "thought_experiment": "thought_experiments.json",
}


class InquiryLibraryError(RuntimeError):
    """问题卡数据无法加载。"""


def _load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InquiryLibraryError(f"问题卡 JSON 格式错误: {path}") from exc
    if not isinstance(data, list):
        raise InquiryLibraryError(f"问题卡文件必须是数组: {path}")
    return data


def load_cards(kind: str | None = None) -> list[PromptCard]:
    """加载问题卡；kind 为空时加载全部。"""
    kinds = [kind] if kind else list(_KIND_FILES)
    cards: list[PromptCard] = []
    for item_kind in kinds:
        filename = _KIND_FILES.get(item_kind)
        if not filename:
            raise ValueError(f"未知心智漫游类型: {item_kind}")
        path = INQUIRY_DATA_DIR / filename
        for raw in _load_json(path):
            try:
                card = PromptCard.from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                raise InquiryLibraryError(f"问题卡字段错误: {path}") from exc
            if card.kind != item_kind:
                raise InquiryLibraryError(
                    f"问题卡 {card.id!r} 的 kind={card.kind!r} 与文件类型 {item_kind!r} 不一致"
                )
            cards.append(card)
    return cards


def get_card(card_id: str) -> PromptCard | None:
    """按 id 查找问题卡。"""
    for card in load_cards():
        if card.id == card_id:
            return card
    return None


def choose_random_card(kind: str | None = None) -> PromptCard:
    """随机选择一张问题卡。"""
    cards = load_cards(kind)
    if not cards:
        target = kind or "全部类型"
        raise InquiryLibraryError(f"没有可用的问题卡: {target}")
    return random.choice(cards)
