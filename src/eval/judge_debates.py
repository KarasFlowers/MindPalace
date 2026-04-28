"""LLM-as-a-Judge 周度评估 — 对历史 debates 打分并聚合弱点。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT, get_judge_config
from src.llm.client import chat_json
from src.storage.db import _get_conn, init_db

logger = logging.getLogger(__name__)


EVAL_PROMPT = """\
你是 MindPalace 的元评估器。对下面这次 Council 讨论打分（每项 1-10）：

评分维度:
- logical_rigor: 论证严密度（是否有逻辑漏洞、飞跃）
- inspiration: 启发性（能否引发新思考、视角转换）
- coverage: 角度覆盖度（多学科、多立场是否都涉及）
- groundedness: 事实扎实度（有无 citations，是否编造数据）

输出 JSON:
{
  "scores": {
    "logical_rigor": 8,
    "inspiration": 7,
    "coverage": 6,
    "groundedness": 5
  },
  "weaknesses": ["弱点1", "弱点2"],
  "prompt_improvement_hint": "一句话建议如何改进角色 prompt"
}
"""


def judge_recent_debates(days: int = 7) -> list[dict]:
    """评估最近 N 天的所有 debates，返回评分列表。"""
    init_db()

    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, article_title, turns, consensus, terminated_by, total_rounds
            FROM debates
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
            """,
            (f"-{days} days",),
        ).fetchall()

    if not rows:
        logger.info("No debates found in the last %d days", days)
        return []

    cfg = get_judge_config()
    reports = []

    for row in rows:
        debate_id = row["id"]
        title = row["article_title"]
        logger.info("Judging debate #%d: %s", debate_id, title[:40])

        user_prompt = _format_debate_for_eval(row)
        try:
            result = chat_json(EVAL_PROMPT, user_prompt, provider_config=cfg)
            result["debate_id"] = debate_id
            result["article_title"] = title
            reports.append(result)
        except Exception as exc:
            logger.warning("Failed to judge debate #%d: %s", debate_id, exc)
            reports.append({
                "debate_id": debate_id,
                "article_title": title,
                "error": str(exc),
            })

    return reports


def generate_weekly_report(reports: list[dict], days: int = 7) -> str:
    """从评分报告生成周度 markdown 报告。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    valid = [r for r in reports if "scores" in r]
    if not valid:
        return f"# Weekly Eval Report ({now})\n\n本周无有效评估数据。"

    # 聚合分数
    dims = ["logical_rigor", "inspiration", "coverage", "groundedness"]
    avg_scores = {}
    for d in dims:
        scores = [r["scores"].get(d, 0) for r in valid]
        avg_scores[d] = round(sum(scores) / len(scores), 1) if scores else 0

    # 聚合弱点
    all_weaknesses: list[str] = []
    all_hints: list[str] = []
    for r in valid:
        all_weaknesses.extend(r.get("weaknesses", []))
        hint = r.get("prompt_improvement_hint")
        if hint:
            all_hints.append(hint)

    # 输出
    lines = [
        f"# Weekly Eval Report ({now})",
        f"\n评估周期: 最近 {days} 天 | 评估数量: {len(valid)} / {len(reports)}",
        "\n## 平均分数\n",
        "| 维度 | 分数 |",
        "|---|---|",
    ]
    for d in dims:
        lines.append(f"| {d} | {avg_scores[d]} |")

    lines.append("\n## Top Weaknesses\n")
    for i, w in enumerate(all_weaknesses[:5], 1):
        lines.append(f"{i}. {w}")

    if all_hints:
        lines.append("\n## Prompt Improvement Hints\n")
        for i, h in enumerate(all_hints[:3], 1):
            lines.append(f"{i}. {h}")

    return "\n".join(lines)


def save_weekly_report(report: str) -> Path:
    """将周度报告保存到 eval/ 目录。"""
    eval_dir = PROJECT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = eval_dir / f"weekly_report_{now}.md"
    path.write_text(report, encoding="utf-8")
    logger.info("Saved weekly report to %s", path)
    return path


def _format_debate_for_eval(row) -> str:
    """将 debate 行格式化为 LLM 评估输入。"""
    title = row["article_title"]
    turns_raw = row["turns"]
    consensus_raw = row["consensus"]
    terminated_by = row["terminated_by"]
    total_rounds = row["total_rounds"]

    try:
        turns = json.loads(turns_raw)
    except (json.JSONDecodeError, TypeError):
        turns = []

    try:
        consensus = json.loads(consensus_raw) if consensus_raw else None
    except (json.JSONDecodeError, TypeError):
        consensus = None

    parts = [
        f"标题: {title}",
        f"轮数: {total_rounds} | 终止原因: {terminated_by}",
        "\n--- 发言记录 ---",
    ]

    for i, t in enumerate(turns):
        role = t.get("role_key", "?")
        phase = t.get("phase", "?")
        content = t.get("content", {})
        tc = t.get("tool_calls_used", 0)
        parts.append(f"\n[{phase}] {role} (tools={tc}):")
        parts.append(json.dumps(content, ensure_ascii=False)[:500])

    if consensus:
        parts.append("\n--- 共识 ---")
        parts.append(json.dumps(consensus, ensure_ascii=False)[:500])

    return "\n".join(parts)
