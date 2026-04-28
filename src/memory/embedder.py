"""Embedding 客户端 — 文本 → 向量。

提供抽象基类和 OpenAI 兼容实现。进程内单例，避免重复初始化。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
from openai import OpenAI

from src.config import get_embedding_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------

class Embedder(ABC):
    """Embedding 提供者的公共接口。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """将一组文本转为向量列表。每个向量为 np.float32 一维数组。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前使用的模型标识符（存入 embed_model 字段供日后追溯）。"""


# ---------------------------------------------------------------------------
# OpenAI 兼容实现
# ---------------------------------------------------------------------------

class OpenAIEmbedder(Embedder):
    """调用 OpenAI（或兼容 API）的 embeddings 接口。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    # -- public API --

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [
            np.array(d.embedding, dtype=np.float32)
            for d in sorted(resp.data, key=lambda d: d.index)
        ]

    @property
    def model_name(self) -> str:
        return self._model


# ---------------------------------------------------------------------------
# 单例工厂
# ---------------------------------------------------------------------------

_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """返回进程唯一的 Embedder 实例（懒初始化）。"""
    global _embedder
    if _embedder is not None:
        return _embedder

    cfg = get_embedding_config()
    api_key = cfg.get("api_key") or ""
    base_url = cfg.get("base_url") or "https://api.openai.com/v1"
    model = (cfg.get("models") or ["text-embedding-3-small"])[0]

    _embedder = OpenAIEmbedder(api_key=api_key, base_url=base_url, model=model)
    logger.info("[Embedder] Initialized: model=%s, base_url=%s", model, base_url)
    return _embedder


def reset_embedder() -> None:
    """重置单例（测试用）。"""
    global _embedder
    _embedder = None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度，返回 float。"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def vec_to_blob(vec: np.ndarray) -> bytes:
    """将 numpy float32 向量序列化为 bytes（存入 SQLite BLOB）。"""
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vec(blob: bytes) -> np.ndarray:
    """将 BLOB 反序列化为 numpy float32 向量。"""
    return np.frombuffer(blob, dtype=np.float32).copy()
