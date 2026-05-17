"""认知轨迹 — 按月聚合 embedding 质心，计算思维漂移。

对比相邻月份的质心余弦距离，用 fast 模型生成自然语言描述。
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from src.config import DB_PATH
from src.memory.embedder import blob_to_vec, cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_trajectory(months: int = 3) -> list[dict]:
    """按月聚合 embeddings 计算质心，返回相邻月份的漂移信息。

    Returns:
        列表，每项包含:
        - month_from / month_to: "YYYY-MM"
        - drift_score: 1 - cosine(centroid_from, centroid_to)，越大表示漂移越大
        - count_from / count_to: 各月份的 memory 条数
    """
    monthly = _load_monthly_embeddings(months)
    if len(monthly) < 2:
        logger.info("Not enough monthly data for trajectory (%d months)", len(monthly))
        return []

    sorted_months = sorted(monthly.keys())
    results = []
    for i in range(len(sorted_months) - 1):
        m_from = sorted_months[i]
        m_to = sorted_months[i + 1]
        centroid_from = _compute_centroid(monthly[m_from])
        centroid_to = _compute_centroid(monthly[m_to])
        drift = 1.0 - cosine_similarity(centroid_from, centroid_to)
        results.append({
            "month_from": m_from,
            "month_to": m_to,
            "drift_score": round(drift, 4),
            "count_from": len(monthly[m_from]),
            "count_to": len(monthly[m_to]),
        })

    return results


def describe_trajectory(
    trajectory: list[dict],
    provider_config: dict | None = None,
) -> str:
    """用 fast 模型为轨迹生成自然语言摘要。"""
    if not trajectory:
        return "暂无足够数据生成认知轨迹描述。"

    from src.llm.client import chat
    from src.config import get_fast_config

    cfg = provider_config or get_fast_config()
    prompt = _format_trajectory_prompt(trajectory)
    return chat(
        system_prompt=(
            "你是 MindPalace 的认知轨迹分析师。"
            "根据用户不同月份的思维漂移分数，用 2-3 句话描述他的认知变化趋势。"
            '用第二人称（"你"），简洁有力。'
        ),
        user_prompt=prompt,
        provider_config=cfg,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_monthly_embeddings(months: int) -> dict[str, list[np.ndarray]]:
    """从 DB 读取最近 N 个月有 embedding 的记忆，按 YYYY-MM 分组。"""
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT embedding, strftime('%%Y-%%m', created_at) AS ym
            FROM memories
            WHERE embedding IS NOT NULL
            ORDER BY created_at DESC
            """
        ).fetchall()
    finally:
        conn.close()

    monthly: dict[str, list[np.ndarray]] = defaultdict(list)
    seen_months: set[str] = set()
    for row in rows:
        ym = row["ym"]
        if ym is None:
            continue
        seen_months.add(ym)
        if len(seen_months) > months:
            break
        monthly[ym].append(blob_to_vec(row["embedding"]))

    return dict(monthly)


def _compute_centroid(vectors: list[np.ndarray]) -> np.ndarray:
    """计算向量列表的均值质心。"""
    stacked = np.stack(vectors)
    centroid = stacked.mean(axis=0)
    return centroid


def _format_trajectory_prompt(trajectory: list[dict]) -> str:
    """将轨迹数据格式化为 LLM prompt。"""
    lines = []
    for t in trajectory:
        lines.append(
            f"{t['month_from']} → {t['month_to']}: "
            f"drift={t['drift_score']:.4f} "
            f"({t['count_from']}条 → {t['count_to']}条)"
        )
    return "\n".join(lines)
