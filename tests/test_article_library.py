"""文章收藏与清理测试。"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch


def _article(url: str, title: str):
    return SimpleNamespace(
        url=url,
        title=title,
        source="Test Source",
        summary="summary",
        clean_content="content",
        scores={"information_density": 8, "principle_depth": 8, "causal_chain": 8},
        total_score=8.0,
        reasoning="reasoning",
    )


class TestArticleLibrary:
    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._patcher = patch("src.storage.db.DB_PATH", self._tmp.name)
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def _make_old(self, article_id: int, days: int = 60):
        from src.storage.db import _get_conn

        old = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with _get_conn() as conn:
            conn.execute("UPDATE articles SET created_at = ? WHERE id = ?", (old, article_id))

    def test_favorite_article_is_listed_and_protected_from_cleanup(self):
        from src.storage.db import (
            cleanup_old_articles,
            get_article,
            list_articles,
            save_articles,
            set_article_favorite,
        )

        save_articles([_article("https://example.com/a", "A")])
        article_id = list_articles()[0]["id"]
        self._make_old(article_id)

        assert set_article_favorite(article_id, favorite=True, note="keeper") is True
        favorites = list_articles(favorites_only=True)
        assert favorites[0]["id"] == article_id
        assert favorites[0]["favorite_note"] == "keeper"

        result = cleanup_old_articles(retention_days=30)
        assert result["deleted_count"] == 0
        assert get_article(article_id) is not None

    def test_cleanup_deletes_old_non_favorite_article(self):
        from src.storage.db import cleanup_old_articles, get_article, list_articles, save_articles

        save_articles([_article("https://example.com/b", "B")])
        article_id = list_articles()[0]["id"]
        self._make_old(article_id)

        result = cleanup_old_articles(retention_days=30)
        assert result["deleted_count"] == 1
        assert get_article(article_id) is None

    def test_cleanup_keeps_article_with_memory_by_default(self):
        from src.storage.db import _get_conn, cleanup_old_articles, get_article, list_articles, save_articles

        save_articles([_article("https://example.com/c", "C")])
        article_id = list_articles()[0]["id"]
        self._make_old(article_id)

        with _get_conn() as conn:
            conn.execute("CREATE TABLE memories (article_id INTEGER)")
            conn.execute("INSERT INTO memories (article_id) VALUES (?)", (article_id,))

        result = cleanup_old_articles(retention_days=30)
        assert result["deleted_count"] == 0
        assert get_article(article_id) is not None
