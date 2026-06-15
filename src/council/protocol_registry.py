"""收敛协议注册表 — 镜像 paradigm registry 的设计。

通过字符串 → 类的映射，新增协议只需子类化 ConvergenceProtocol 并调用
register_protocol()，无需修改 DebateParadigm 的分支逻辑。
"""

from __future__ import annotations

import logging

from src.council.protocols import (
    ConvergenceProtocol,
    MidcheckProtocol,
    ConsensusThresholdProtocol,
    VotingProtocol,
)

logger = logging.getLogger(__name__)

# 字符串 → 协议类的注册表
PROTOCOLS: dict[str, type[ConvergenceProtocol]] = {
    "midcheck": MidcheckProtocol,
    "consensus_threshold": ConsensusThresholdProtocol,
    "voting": VotingProtocol,
}


def register_protocol(name: str, protocol_cls: type[ConvergenceProtocol]) -> None:
    """注册一个新的收敛协议。

    Args:
        name: 协议标识（小写字符串）。
        protocol_cls: ConvergenceProtocol 的子类。
    """
    if not name or not name.strip():
        raise ValueError("Protocol name must be non-empty")
    name = name.strip().lower()
    if not issubclass(protocol_cls, ConvergenceProtocol):
        raise TypeError(
            f"Protocol class must subclass ConvergenceProtocol, got {protocol_cls!r}"
        )
    PROTOCOLS[name] = protocol_cls
    logger.info("Registered council convergence protocol: %s -> %s", name, protocol_cls.__name__)


def get_protocol(name: str) -> ConvergenceProtocol:
    """根据名称获取协议实例；未知名称回退到 midcheck。"""
    name = (name or "").strip().lower()
    cls = PROTOCOLS.get(name)
    if cls is None:
        logger.warning("Unknown convergence protocol %r, falling back to 'midcheck'", name)
        cls = PROTOCOLS["midcheck"]
    return cls()


def list_protocols() -> list[str]:
    """返回所有已注册的协议名称。"""
    return sorted(PROTOCOLS.keys())
