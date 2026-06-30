"""Scout 流水线：fetch -> normalize -> translate -> score -> rank。"""

import logging
from dataclasses import replace
from src.scout.fetch import fetch_all
from src.scout.normalize import normalize_all
from src.scout.score import score_all, ScoredArticle
from src.scout.translate import maybe_translate_article
from src.storage.db import cleanup_old_articles, save_articles, get_existing_urls, mark_as_scanned
from src.config import ARTICLE_AUTO_CLEANUP, ARTICLE_RETENTION_DAYS, get_default_feeds, SCOUT_TOP_K
from src.obs import span

logger = logging.getLogger(__name__)


def _maybe_cleanup_old_articles(save: bool) -> None:
    if not save or not ARTICLE_AUTO_CLEANUP:
        return
    try:
        cleanup = cleanup_old_articles(ARTICLE_RETENTION_DAYS, keep_discussed=True)
        if cleanup["deleted_count"]:
            logger.info(
                "Auto-cleaned %d old non-favorite articles older than %d days.",
                cleanup["deleted_count"],
                ARTICLE_RETENTION_DAYS,
            )
    except Exception:
        logger.exception("Article auto-cleanup failed")


def translate_all(articles, provider_config: dict | None = None):
    """按需翻译文章正文，并把语言元数据附加到后续评分对象。"""
    translated_articles = []
    for article in articles:
        try:
            source_lang, clean_content, translated = maybe_translate_article(
                article, provider_config=provider_config
            )
            next_article = replace(article, clean_content=clean_content)
            setattr(next_article, "source_lang", source_lang)
            setattr(next_article, "translated", translated)
            translated_articles.append(next_article)
        except Exception:
            logger.exception("Failed to translate: %s", article.url)
            setattr(article, "source_lang", "unknown")
            setattr(article, "translated", False)
            translated_articles.append(article)
    return translated_articles


def run_scout(
    feed_urls: list[str] | None = None,
    top_k: int = SCOUT_TOP_K,
    save: bool = True,
    provider_config: dict | None = None,
) -> list[ScoredArticle]:
    """运行 Scout 流水线。

    Args:
        feed_urls: RSS 源 URL 列表，默认使用配置中的 DEFAULT_FEEDS。
        top_k: 返回前 K 条高分内容。
        save: 是否将结果保存到数据库。

    Returns:
        按启发性评分降序排列的 top_k 篇文章。
    """
    urls = feed_urls or get_default_feeds()
    logger.info("=== Scout Pipeline Start ===")

    with span("scout.pipeline", feed_count=len(urls), top_k=top_k):

        # 1. Fetch
        logger.info("[1/3] Fetching from %d feeds...", len(urls))
        raw_articles = fetch_all(urls)
        if not raw_articles:
            logger.warning("No articles fetched. Check feed URLs.")
            _maybe_cleanup_old_articles(save)
            return []

        # 2. Normalize
        logger.info("[2/3] Normalizing %d articles...", len(raw_articles))
        normalized = normalize_all(raw_articles)

        # Check existing
        existing_urls = get_existing_urls()
        new_articles = [a for a in normalized if a.url not in existing_urls]

        if not new_articles:
            logger.info("No new articles to score. Returning []")
            _maybe_cleanup_old_articles(save)
            return []

        # 3. Translate
        logger.info("[3/4] Translating %d new articles when needed...", len(new_articles))
        translated_articles = translate_all(new_articles, provider_config=provider_config)

        # 4. Score & Rank
        logger.info("[4/4] Scoring %d new articles (out of %d total)...", len(translated_articles), len(normalized))
        scored = score_all(translated_articles, provider_config=provider_config)

        # Persistent marking of scanned articles to avoid re-scoring
        mark_as_scanned([a.url for a in scored])

        # Take top_k
        result = scored[:top_k]

        # Save
        if save and result:
            save_articles(result)
            logger.info("Saved %d articles to database.", len(result))

        _maybe_cleanup_old_articles(save)

        logger.info("=== Scout Pipeline Done — Top %d selected ===", len(result))
        return result
