"""认知档案导出 — 将 profile_crystals 导出为 Markdown brain 目录。

借鉴 Axiomind 的 brain/ 目录设计：所有结构化洞察按 type 分目录导出为带
YAML frontmatter 的 Markdown 文件，可被 Obsidian 或未来 agent 直接读取。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from src.config import PROJECT_ROOT
from src.storage.db import _get_conn, init_db

logger = logging.getLogger(__name__)

_TYPE_DIR = {
    "axiom": "axioms",
    "principle": "principles",
    "observation": "observations",
}


def export_brain(export_dir=None) -> int:
    """将所有 profile_crystals 导出为 Markdown brain 目录。

    Args:
        export_dir: 导出根目录；默认为 PROJECT_ROOT/data/brain。

    Returns:
        导出的文件数量。
    """
    init_db()
    root = export_dir or (PROJECT_ROOT / "data" / "brain")
    root = _ensure_path(root)

    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM profile_crystals ORDER BY id ASC"
        ).fetchall()

    count = 0
    for row in rows:
        crystal_type = (row["type"] or "observation").lower()
        if crystal_type not in _TYPE_DIR:
            crystal_type = "observation"
        subdir = root / _TYPE_DIR[crystal_type]
        subdir.mkdir(parents=True, exist_ok=True)

        frontmatter = {
            "type": crystal_type,
            "status": row["status"] or "candidate",
            "confidence": row["confidence"] if row["confidence"] is not None else 0.0,
            "sources": _safe_json_loads(row["sources"], []),
            "tags": _safe_json_loads(row["tags"], []),
            "created_at": row["created_at"],
            "anchor_memory_id": row["anchor_memory_id"],
        }

        filename = f"{row['id']:04d}_{crystal_type}.md"
        filepath = subdir / filename

        content = _render_brain_file(frontmatter, row["content"])
        filepath.write_text(content, encoding="utf-8")
        count += 1

    logger.info("Exported %d crystals to %s", count, root)
    return count


def _ensure_path(path) -> "Path":
    """确保 path 是 Path 对象。"""
    from pathlib import Path
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_json_loads(text, default):
    try:
        return json.loads(text) if text else default
    except (json.JSONDecodeError, TypeError):
        return default


def _render_brain_file(frontmatter: dict, content: str) -> str:
    """渲染带 YAML frontmatter 的 Markdown 文件。"""
    import re
    fm_lines = ["---"]
    # 简单标识符（字母/数字/下划线/连字符）可不加引号，其他字符串加引号转义
    simple_re = re.compile(r"^[A-Za-z0-9_\-]+$")
    for key, value in frontmatter.items():
        if isinstance(value, list):
            if value:
                fm_lines.append(f"{key}:")
                for item in value:
                    fm_lines.append(f"  - {item}")
            else:
                fm_lines.append(f"{key}: []")
        elif isinstance(value, (int, float)):
            fm_lines.append(f"{key}: {value}")
        elif isinstance(value, str) and simple_re.match(value):
            fm_lines.append(f"{key}: {value}")
        else:
            safe = str(value).replace('"', '\\"')
            fm_lines.append(f'{key}: "{safe}"')
    fm_lines.append("---")
    fm_lines.append("")
    fm_lines.append(content or "")
    return "\n".join(fm_lines)
