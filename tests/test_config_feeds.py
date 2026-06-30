"""Scout feed 配置测试。"""

import importlib
import os
from unittest.mock import patch

import src.config as config_module


def _reload_config():
    with patch("dotenv.load_dotenv", return_value=True):
        return importlib.reload(config_module)


class TestScoutFeedConfig:
    def test_defaults_to_humanities_preset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCOUT_FEED_PRESET", None)
            os.environ.pop("SCOUT_FEEDS", None)
            os.environ.pop("FEEDS", None)
            config = _reload_config()
            assert config.get_default_feeds() == config.FEED_PRESETS["humanities"]
        _reload_config()

    def test_humanities_preset_matches_curated_humanities_sources(self):
        config = _reload_config()
        assert config.FEED_PRESETS["humanities"] == [
            "https://aeon.co/feed.rss",
            "https://psyche.co/feed.rss",
            "https://daily.jstor.org/feed/",
            "https://thepointmag.com/feed/",
            "https://www.noemamag.com/feed/",
            "https://crookedtimber.org/feed/",
        ]
        _reload_config()

    def test_scout_feed_preset_can_switch_to_tech(self):
        with patch.dict(os.environ, {"SCOUT_FEED_PRESET": "tech"}, clear=False):
            os.environ.pop("SCOUT_FEEDS", None)
            os.environ.pop("FEEDS", None)
            config = _reload_config()
            assert config.get_default_feeds() == config.FEED_PRESETS["tech"]
        _reload_config()

    def test_chinese_presets_are_available(self):
        config = _reload_config()

        assert "humanities_zh" in config.FEED_PRESETS
        assert "mixed_zh" in config.FEED_PRESETS
        assert config.get_default_feeds("humanities_zh") == config.FEED_PRESETS["humanities_zh"]
        assert config.get_default_feeds("mixed_zh") == config.FEED_PRESETS["mixed_zh"]
        _reload_config()

    def test_scout_translate_defaults_to_enabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SCOUT_TRANSLATE", None)
            config = _reload_config()
            assert config.SCOUT_TRANSLATE is True
        _reload_config()

    def test_scout_translate_can_be_disabled(self):
        with patch.dict(os.environ, {"SCOUT_TRANSLATE": "false"}, clear=False):
            config = _reload_config()
            assert config.SCOUT_TRANSLATE is False
        _reload_config()

    def test_custom_feed_list_overrides_env_preset(self):
        with patch.dict(
            os.environ,
            {
                "SCOUT_FEED_PRESET": "tech",
                "SCOUT_FEEDS": "https://example.com/a.xml, https://example.com/b.xml",
            },
            clear=False,
        ):
            config = _reload_config()
            assert config.get_default_feeds() == [
                "https://example.com/a.xml",
                "https://example.com/b.xml",
            ]
        _reload_config()

    def test_explicit_preset_argument_beats_custom_env_list(self):
        with patch.dict(
            os.environ,
            {"SCOUT_FEEDS": "https://example.com/a.xml https://example.com/b.xml"},
            clear=False,
        ):
            os.environ.pop("SCOUT_FEED_PRESET", None)
            os.environ.pop("FEEDS", None)
            config = _reload_config()
            assert config.get_default_feeds("mixed") == config.FEED_PRESETS["mixed"]
        _reload_config()
