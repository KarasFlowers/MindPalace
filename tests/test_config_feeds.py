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

    def test_scout_feed_preset_can_switch_to_tech(self):
        with patch.dict(os.environ, {"SCOUT_FEED_PRESET": "tech"}, clear=False):
            os.environ.pop("SCOUT_FEEDS", None)
            os.environ.pop("FEEDS", None)
            config = _reload_config()
            assert config.get_default_feeds() == config.FEED_PRESETS["tech"]
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
