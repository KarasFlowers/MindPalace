"""文章档案库测试。"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch


def _article(url: str, title: str, source: str = "Test Source", score: float = 8.0):
    return SimpleNamespace(
        url=url,
        title=title,
        source=source,
        summary="summary",
        clean_content="content",
        source_lang="zh",
        translated=False,
        scores={"information_density": 8, "principle_depth": 8, "causal_chain": 8},
        total_score=score,
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

    def test_existing_favorites_survive_archive_migration(self):
        from src.storage.db import _get_conn, list_articles

        conn = sqlite3.connect(self._tmp.name)
        conn.execute(
            """
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                summary TEXT,
                clean_content TEXT,
                scores_json TEXT,
                total_score REAL,
                reasoning TEXT,
                is_favorite INTEGER NOT NULL DEFAULT 0,
                favorited_at TEXT,
                favorite_note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO articles
            (url, title, source, summary, clean_content, scores_json, total_score, reasoning,
             is_favorite, favorited_at, favorite_note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "https://example.com/legacy",
                "Legacy Favorite",
                "Legacy Source",
                "summary",
                "content",
                "{}",
                9.0,
                "reasoning",
                1,
                datetime.now(timezone.utc).isoformat(),
                "keeper",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()

        favorites = list_articles(favorites_only=True)
        assert favorites[0]["title"] == "Legacy Favorite"
        assert favorites[0]["favorite_note"] == "keeper"

        with _get_conn() as conn2:
            tables = {
                row["name"]
                for row in conn2.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        assert "article_tags" in tables

    def test_legacy_articles_table_gets_article_columns_migrated_idempotently(self):
        from src.storage.db import _get_conn, init_db

        conn = sqlite3.connect(self._tmp.name)
        conn.execute(
            """
            CREATE TABLE articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                source TEXT NOT NULL,
                summary TEXT,
                scores_json TEXT,
                total_score REAL,
                reasoning TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

        init_db()
        init_db()

        with _get_conn() as conn2:
            cols = {
                row["name"]
                for row in conn2.execute("PRAGMA table_info(articles)").fetchall()
            }
        assert "clean_content" in cols
        assert "source_lang" in cols
        assert "translated" in cols

    def test_save_articles_persists_translation_metadata(self):
        from src.storage.db import list_articles, save_articles

        article = _article("https://example.com/translated", "Translated")
        article.source_lang = "other"
        article.translated = True

        save_articles([article])

        saved = list_articles()[0]
        assert saved["source_lang"] == "other"
        assert saved["translated"] is True

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

    def test_note_survives_unfavorite(self):
        from src.storage.db import get_article, list_articles, save_articles, set_article_favorite

        save_articles([_article("https://example.com/a-note", "A Note")])
        article_id = list_articles()[0]["id"]

        set_article_favorite(article_id, favorite=True, note="keep this")
        set_article_favorite(article_id, favorite=False)
        article = get_article(article_id)

        assert article["is_favorite"] is False
        assert article["favorite_note"] == "keep this"

    def test_list_articles_filters_by_query_tag_source_and_days(self):
        from src.storage.db import add_article_tags, list_articles, save_articles, set_article_note

        save_articles(
            [
                _article("https://example.com/1", "Modernity and Care", source="Aeon", score=9.0),
                _article("https://example.com/2", "Physics Update", source="Quanta", score=7.0),
                _article("https://example.com/3", "History of Attention", source="Aeon", score=8.0),
            ]
        )

        articles = {article["title"]: article for article in list_articles(limit=10)}
        modernity_id = articles["Modernity and Care"]["id"]
        physics_id = articles["Physics Update"]["id"]
        history_id = articles["History of Attention"]["id"]

        add_article_tags(modernity_id, ["history", "archive"])
        add_article_tags(history_id, ["history", "ethics"])
        set_article_note(history_id, "适合回看")
        self._make_old(physics_id, days=45)

        history_matches = {article["id"] for article in list_articles(limit=10, tags=["history"])}
        assert history_matches == {modernity_id, history_id}

        ethics_matches = list_articles(limit=10, tags=["history", "ethics"])
        assert [article["id"] for article in ethics_matches] == [history_id]

        tag_query_matches = [article["id"] for article in list_articles(limit=10, query="archive")]
        assert tag_query_matches == [modernity_id]

        note_query_matches = [article["id"] for article in list_articles(limit=10, query="回看")]
        assert note_query_matches == [history_id]

        source_matches = {article["id"] for article in list_articles(limit=10, source="aeon")}
        assert source_matches == {modernity_id, history_id}

        recent_matches = {article["id"] for article in list_articles(limit=10, days=30)}
        assert recent_matches == {modernity_id, history_id}

    def test_cleanup_deletes_old_non_favorite_article(self):
        from src.storage.db import cleanup_old_articles, get_article, list_articles, save_articles

        save_articles([_article("https://example.com/b", "B")])
        article_id = list_articles()[0]["id"]
        self._make_old(article_id)

        result = cleanup_old_articles(retention_days=30)
        assert result["deleted_count"] == 1
        assert get_article(article_id) is None

    def test_cleanup_keeps_tagged_article_by_default(self):
        from src.storage.db import add_article_tags, cleanup_old_articles, get_article, list_articles, save_articles

        save_articles([_article("https://example.com/tagged", "Tagged")])
        article_id = list_articles()[0]["id"]
        self._make_old(article_id)
        add_article_tags(article_id, ["history"])

        result = cleanup_old_articles(retention_days=30)
        assert result["deleted_count"] == 0
        assert get_article(article_id) is not None

        result = cleanup_old_articles(retention_days=30, keep_tagged=False)
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
