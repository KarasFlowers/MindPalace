"""内容清洗与标准化。"""

import re
import logging
from dataclasses import dataclass
from src.scout.fetch import RawArticle

logger = logging.getLogger(__name__)

# 正文截断阈值（字符数），避免向 LLM 发送过长内容
MAX_CONTENT_LENGTH = 4000


@dataclass
class NormalizedArticle:
    """清洗后的文章。"""

    url: str
    title: str
    content: str        # 原始 content（保留备用）
    clean_content: str  # 清洗后的纯文本
    source: str
    published_at: str


def _strip_html(text: str) -> str:
    """移除 HTML 标签，保留纯文本。"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)  # HTML 实体
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _truncate(text: str, max_len: int = MAX_CONTENT_LENGTH) -> str:
    """截断过长文本，保留开头内容。"""
    if len(text) <= max_len:
        return text
    # 在最后一个完整句末截断
    truncated = text[:max_len]
    last_period = max(truncated.rfind("."), truncated.rfind("。"), truncated.rfind("!"))
    if last_period > max_len * 0.5:
        return truncated[: last_period + 1]
    return truncated + "..."


def normalize(raw: RawArticle) -> NormalizedArticle:
    """将原始文章清洗为标准化格式。"""
    clean = _strip_html(raw.content)
    clean = _truncate(clean)

    return NormalizedArticle(
        url=raw.url,
        title=raw.title.strip(),
        content=raw.content,
        clean_content=clean,
        source=raw.source,
        published_at=raw.published_at,
    )


def normalize_all(raws: list[RawArticle]) -> list[NormalizedArticle]:
    """批量标准化。"""
    results = []
    for raw in raws:
        try:
            results.append(normalize(raw))
        except Exception:
            logger.exception("Failed to normalize: %s", raw.url)
    return results
