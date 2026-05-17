"""配置检测相关测试。"""

import importlib
import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import src.config as config_module


def _reload_config():
    with patch("dotenv.load_dotenv", return_value=True):
        return importlib.reload(config_module)


def test_embedding_config_defaults_to_embedding_model():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-test",
            "MODEL_NAMES": "gpt-4o-mini,gpt-4o",
        },
        clear=False,
    ):
        os.environ.pop("EMBEDDING_MODEL_NAMES", None)
        os.environ.pop("EMBEDDING_MODEL_NAME", None)
        config = _reload_config()
        assert config.get_embedding_config()["models"] == ["text-embedding-3-small"]
    _reload_config()


def test_cmd_config_provider_all_uses_batch_testing():
    from src import app

    with patch.object(Path, "exists", return_value=True), \
         patch("src.app._test_all_provider_configs") as mock_test_all:
        app.cmd_config(Namespace(test=True, provider="all"))

    mock_test_all.assert_called_once()


def test_test_provider_config_failure_prints_reconfigure_hint(capsys):
    from src import app

    cfg = {
        "api_key": "sk-test",
        "base_url": "https://api.openai.com/v1",
        "models": ["bad-model"],
    }

    with patch("src.llm.client.chat", side_effect=RuntimeError("boom")):
        ok = app._test_provider_config("SCOUT", cfg)

    out = capsys.readouterr().out
    assert ok is False
    assert "建议重新配置 Scout" in out
    assert "python -m src config --test --provider scout" in out
