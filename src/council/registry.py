"""讨论范式注册表 — 借鉴 MALLM 的 utils/dicts.py 设计。

通过字符串 → 类的映射，新增范式只需子类化 DiscussionParadigm 并调用 register_paradigm()，
无需修改 run_council 的分支逻辑。
"""

from __future__ import annotations

import logging

from src.council.paradigms import (
    DiscussionParadigm,
    DebateParadigm,
    ReportParadigm,
)

logger = logging.getLogger(__name__)

# 字符串 → 范式类的注册表
PARADIGMS: dict[str, type[DiscussionParadigm]] = {
    "debate": DebateParadigm,
    "report": ReportParadigm,
}


def register_paradigm(name: str, paradigm_cls: type[DiscussionParadigm]) -> None:
    """注册一个新的讨论范式。

    Args:
        name: 范式标识（小写字符串）。
        paradigm_cls: DiscussionParadigm 的子类。
    """
    if not name or not name.strip():
        raise ValueError("Paradigm name must be non-empty")
    name = name.strip().lower()
    if not issubclass(paradigm_cls, DiscussionParadigm):
        raise TypeError(
            f"Paradigm class must subclass DiscussionParadigm, got {paradigm_cls!r}"
        )
    PARADIGMS[name] = paradigm_cls
    logger.info("Registered council paradigm: %s -> %s", name, paradigm_cls.__name__)


def get_paradigm(name: str) -> type[DiscussionParadigm]:
    """根据名称获取范式类；未知名称回退到 debate。"""
    name = (name or "").strip().lower()
    cls = PARADIGMS.get(name)
    if cls is None:
        logger.warning("Unknown paradigm %r, falling back to 'debate'", name)
        return PARADIGMS["debate"]
    return cls


def list_paradigms() -> list[str]:
    """返回所有已注册的范式名称。"""
    return sorted(PARADIGMS.keys())
