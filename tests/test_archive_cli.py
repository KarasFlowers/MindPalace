"""档案库 CLI 与交互 smoke tests。"""

import subprocess
import sys
from unittest.mock import MagicMock, patch


def _prompt_result(value):
    prompt = MagicMock()
    prompt.ask.return_value = value
    return prompt


def test_list_help_mentions_archive_filters():
    result = subprocess.run(
        [sys.executable, "-m", "src", "list", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    out = result.stdout
    assert "--query" in out
    assert "--tag" in out
    assert "--source" in out
    assert "--days" in out


def test_interactive_list_smoke():
    from src import app

    article = {
        "id": 1,
        "title": "Archive Article",
        "source": "Aeon",
        "summary": "summary",
        "url": "https://example.com/archive",
        "total_score": 8.6,
        "created_at": "2026-06-09T00:00:00+00:00",
        "is_favorite": True,
        "favorite_note": "keeper",
        "tags": ["history"],
    }

    with patch("src.app.list_articles", return_value=[article]), \
         patch("src.app.get_article", return_value=article), \
         patch("src.app.list_recent_debates_for_article", return_value=[]), \
         patch(
             "src.app.questionary.select",
             side_effect=[
                 _prompt_result("[ID:1] [收藏] Archive Article ████████░░ 8.6/10 📝 #history"),
                 _prompt_result("🔙 返回文章列表"),
                 _prompt_result("🔙 返回主菜单"),
             ],
         ):
        app._interactive_list(favorites_only=True, filters={"tags": ["history"]})
