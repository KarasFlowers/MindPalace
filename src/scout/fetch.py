"""RSS 抓取模块。"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

import feedparser
from src.config import MAX_WORKERS

logger = logging.getLogger(__name__)


@dataclass
class RawArticle:
    """从 RSS feed 抓取的原始文章。"""

    url: str
    title: str
    content: str
    source: str
    published_at: str  # ISO 格式或 feed 原始格式


def fetch_rss(feed_url: str) -> list[RawArticle]:
    """抓取一个 RSS feed，返回文章列表。

    Args:
        feed_url: RSS feed 的 URL。

    Returns:
        RawArticle 列表。
    """
    logger.info("Fetching RSS: %s", feed_url)
    feed = feedparser.parse(feed_url)

    if feed.bozo and not feed.entries:
        logger.warning("Feed 解析异常: %s — %s", feed_url, feed.bozo_exception)
        return []

    source_name = feed.feed.get("title", feed_url)
    articles = []

    for entry in feed.entries:
        # 优先用 content，其次 summary，最后 title
        content = ""
        if hasattr(entry, "content") and entry.content:
            content = entry.content[0].get("value", "")
        elif hasattr(entry, "summary"):
            content = entry.summary or ""

        # 发布时间
        published = ""
        if hasattr(entry, "published"):
            published = entry.published
        elif hasattr(entry, "updated"):
            published = entry.updated

        articles.append(
            RawArticle(
                url=entry.get("link", ""),
                title=entry.get("title", "Untitled"),
                content=content,
                source=source_name,
                published_at=published,
            )
        )

    logger.info("Fetched %d articles from %s", len(articles), source_name)
    return articles


def fetch_all(feed_urls: list[str]) -> list[RawArticle]:
    """抓取多个 RSS feed，合并去重。使用线程池并行抓取。"""
    all_articles = []
    seen_urls = set()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(fetch_rss, url): url for url in feed_urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                articles = future.result()
                for a in articles:
                    if a.url and a.url not in seen_urls:
                        seen_urls.add(a.url)
                        all_articles.append(a)
            except Exception:
                logger.exception("Failed to fetch %s", url)

    logger.info("Total unique articles: %d", len(all_articles))
    return all_articles
