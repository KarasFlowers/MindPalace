"""配置检测相关测试。"""

import importlib
import os
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import textwrap

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


def test_non_openai_provider_can_fallback_to_openai_model_names():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL_NAMES": "gpt-4o-mini",
        },
        clear=True,
    ):
        config = _reload_config()
        assert config.get_provider_config("SCOUT")["models"] == ["gpt-4o-mini"]
    _reload_config()


def test_provider_type_can_be_explicit_or_inferred_from_anthropic_base_url():
    with patch.dict(
        os.environ,
        {
            "COUNCIL_API_KEY": "sk-ant",
            "COUNCIL_BASE_URL": "https://api.anthropic.com/v1",
            "COUNCIL_MODEL_NAMES": "claude-test",
        },
        clear=True,
    ):
        config = _reload_config()
        assert config.get_provider_config("COUNCIL")["provider_type"] == "anthropic"

    with patch.dict(
        os.environ,
        {
            "COUNCIL_PROVIDER_TYPE": "anthropic",
            "COUNCIL_BASE_URL": "https://proxy.example/v1",
            "COUNCIL_MODEL_NAMES": "claude-test",
        },
        clear=True,
    ):
        config = _reload_config()
        assert config.get_provider_config("COUNCIL")["provider_type"] == "anthropic"
    _reload_config()


def test_task_base_url_inference_is_not_overridden_by_global_provider_type():
    with patch.dict(
        os.environ,
        {
            "OPENAI_PROVIDER_TYPE": "openai",
            "OPENAI_API_KEY": "sk-global",
            "OPENAI_BASE_URL": "https://proxy.example/v1",
            "OPENAI_MODEL_NAMES": "gpt-4o-mini",
            "COUNCIL_BASE_URL": "https://api.anthropic.com/v1",
            "COUNCIL_MODEL_NAMES": "claude-test",
        },
        clear=True,
    ):
        config = _reload_config()
        assert config.get_provider_config("COUNCIL")["provider_type"] == "anthropic"
    _reload_config()


def test_provider_config_can_use_named_api_profile():
    with patch.dict(
        os.environ,
        {
            "API_PROFILE_CLAUDE_PROVIDER_TYPE": "anthropic",
            "API_PROFILE_CLAUDE_API_KEY": "sk-ant",
            "API_PROFILE_CLAUDE_BASE_URL": "https://api.anthropic.com/v1",
            "API_PROFILE_CLAUDE_MODEL_NAMES": "claude-test",
            "COUNCIL_PROVIDER_PROFILE": "claude",
        },
        clear=True,
    ):
        config = _reload_config()
        cfg = config.get_provider_config("COUNCIL")
        assert cfg["provider_profile"] == "claude"
        assert cfg["provider_type"] == "anthropic"
        assert cfg["api_key"] == "sk-ant"
        assert cfg["base_url"] == "https://api.anthropic.com/v1"
        assert cfg["models"] == ["claude-test"]
    _reload_config()


def test_provider_direct_fields_override_named_api_profile():
    with patch.dict(
        os.environ,
        {
            "API_PROFILE_FAST_API_KEY": "sk-profile",
            "API_PROFILE_FAST_BASE_URL": "https://profile.example/v1",
            "API_PROFILE_FAST_MODEL_NAMES": "profile-model",
            "SCOUT_PROVIDER_PROFILE": "fast",
            "SCOUT_MODEL_NAMES": "direct-model",
        },
        clear=True,
    ):
        config = _reload_config()
        cfg = config.get_provider_config("SCOUT")
        assert cfg["api_key"] == "sk-profile"
        assert cfg["models"] == ["direct-model"]
    _reload_config()


def test_openai_model_names_takes_precedence_over_legacy_model_names():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL_NAMES": "claude-opus-4-6-thinking",
            "MODEL_NAMES": "deepseek-reasoner",
        },
        clear=True,
    ):
        config = _reload_config()
        assert config.get_provider_config("OPENAI")["models"] == ["claude-opus-4-6-thinking"]
    _reload_config()


def test_render_all_provider_overview_shows_current_values(tmp_path, capsys):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        textwrap.dedent(
            """
            OPENAI_API_KEY=sk-global
            OPENAI_BASE_URL=https://global.example/v1
            OPENAI_MODEL_NAMES=gpt-4o-mini
            SCOUT_BASE_URL=https://scout.example/v1
            """
        ).strip(),
        encoding="utf-8",
    )

    app._render_all_provider_overview(env_path)
    out = capsys.readouterr().out

    assert "Provider Overview" in out
    assert "Global Default" in out
    assert "Scout" in out
    assert "https://scout.example/v1" in out
    assert "继承自全局默认:" in out


def test_provider_overview_shows_profile_binding(tmp_path, capsys):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        textwrap.dedent(
            """
            API_PROFILE_CLAUDE_PROVIDER_TYPE=anthropic
            API_PROFILE_CLAUDE_API_KEY=sk-ant
            API_PROFILE_CLAUDE_BASE_URL=https://api.anthropic.com/v1
            API_PROFILE_CLAUDE_MODEL_NAMES=claude-test
            COUNCIL_PROVIDER_PROFILE=claude
            """
        ).strip(),
        encoding="utf-8",
    )

    cfg = app._read_provider_config_from_env_file(env_path, "COUNCIL")
    app._render_all_provider_overview(env_path)
    out = capsys.readouterr().out

    assert cfg["provider_profile"] == "claude"
    assert cfg["provider_type"] == "anthropic"
    assert "profile:" in out
    assert "claude" in out


def test_provider_overview_does_not_warn_for_missing_optional_keys(tmp_path, caplog):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-global\nOPENAI_MODEL_NAMES=gpt-4o-mini\n",
        encoding="utf-8",
    )

    app._render_all_provider_overview(env_path)

    assert "EMBEDDING_API_KEY" not in caplog.text
    assert "EMBEDDING_MODEL_NAMES" not in caplog.text


def test_embedding_overview_uses_embedding_default_model(tmp_path, capsys):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-global\nOPENAI_MODEL_NAMES=gpt-4o-mini\n",
        encoding="utf-8",
    )

    cfg = app._read_provider_config_from_env_file(env_path, "EMBEDDING")
    app._render_provider_config_summary("EMBEDDING", env_path)
    out = capsys.readouterr().out

    assert cfg["models"] == ["text-embedding-3-small"]
    assert "text-embedding-3-small" in out
    assert "API_KEY / BASE_URL / MODEL_NAMES" not in out


def test_model_list_parses_openai_compatible_shapes():
    from src import app

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": ["claude-opus-4-1", {"id": "claude-sonnet-4-5"}]}

    with patch("httpx.get", return_value=Response()) as mock_get:
        models = app._list_remote_models("sk-test", "https://api.anthropic.com/v1/")

    assert models == ["claude-opus-4-1", "claude-sonnet-4-5"]
    assert mock_get.call_args.args[0] == "https://api.anthropic.com/v1/models"
    assert mock_get.call_args.kwargs["headers"]["x-api-key"] == "sk-test"


def test_model_list_adds_v1_for_anthropic_compatible_proxy_root():
    from src import app

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "claude-sonnet-4-6"}]}

    with patch("httpx.get", return_value=Response()) as mock_get:
        models = app._list_remote_models("sk-test", "https://yundu.lat", "anthropic")

    assert models == ["claude-sonnet-4-6"]
    assert mock_get.call_args.args[0] == "https://yundu.lat/v1/models"


def test_model_list_non_json_response_has_clear_error():
    from src import app

    class Response:
        text = "<html>not json</html>"

        def raise_for_status(self):
            return None

        def json(self):
            import json
            raise json.JSONDecodeError("Expecting value", "", 0)

    with patch("httpx.get", return_value=Response()):
        try:
            app._list_remote_models("sk-test", "https://proxy.example/v1", "anthropic")
        except RuntimeError as exc:
            assert "模型列表接口没有返回 JSON" in str(exc)
            assert "响应预览=<html>not json</html>" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError")


def test_model_list_failure_hint_mentions_provider_specific_base_urls():
    from src import app

    gemini_hint = app._format_model_list_failure_hint("https://generativelanguage.googleapis.com")
    claude_hint = app._format_model_list_failure_hint("https://api.anthropic.com/v1/")
    anthropic_proxy_hint = app._format_model_list_failure_hint("https://proxy.example/v1", "anthropic")
    generic_hint = app._format_model_list_failure_hint("https://proxy.example/v1")

    assert "v1beta/openai" in gemini_hint
    assert "Claude" in claude_hint
    assert "Anthropic 协议" in anthropic_proxy_hint
    assert "PROVIDER_TYPE 改为 openai" in anthropic_proxy_hint
    assert "OpenAI 官方" in generic_hint


def test_configure_provider_prompts_base_url_before_api_key_and_models(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-old\nOPENAI_BASE_URL=https://old.example/v1\nOPENAI_MODEL_NAMES=old-model\n",
        encoding="utf-8",
    )

    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        if "PROVIDER_TYPE" in prompt:
            return ""
        if "BASE_URL" in prompt:
            return "https://new.example/v1"
        if "API_KEY" in prompt:
            return "sk-new"
        if "MODEL_NAMES" in prompt:
            return "new-model"
        raise AssertionError(prompt)

    with patch("builtins.input", side_effect=fake_input), \
         patch("src.app._render_provider_config_summary"), \
         patch("src.app.questionary.confirm") as mock_confirm:
        mock_confirm.return_value.ask.side_effect = [False, False]
        cfg = app._configure_provider(env_path, "OPENAI")

    assert prompts[:4] == [
        "    PROVIDER_TYPE [openai] (openai/anthropic): ",
        "    BASE_URL [https://old.example/v1]: ",
        "    API_KEY [***]: ",
        "    MODEL_NAMES [old-model]: ",
    ]
    assert cfg["base_url"] == "https://new.example/v1"
    assert cfg["api_key"] == "sk-new"
    assert cfg["models"] == ["new-model"]


def test_configure_provider_uses_new_base_url_to_discover_provider_type(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        textwrap.dedent(
            """
            OPENAI_PROVIDER_TYPE=openai
            OPENAI_API_KEY=sk-global
            OPENAI_BASE_URL=https://proxy.example/v1
            OPENAI_MODEL_NAMES=gpt-4o-mini
            COUNCIL_MODEL_NAMES=old-model
            """
        ).strip(),
        encoding="utf-8",
    )

    prompts = []

    def fake_input(prompt):
        prompts.append(prompt)
        if "PROVIDER_TYPE" in prompt:
            return ""
        if "BASE_URL" in prompt:
            return "https://api.anthropic.com/v1"
        if "API_KEY" in prompt:
            return ""
        if "MODEL_NAMES" in prompt:
            return ""
        raise AssertionError(prompt)

    with patch("builtins.input", side_effect=fake_input), \
         patch("src.app._render_provider_config_summary"), \
         patch("src.app.questionary.confirm") as mock_confirm, \
         patch("src.app._list_remote_models", return_value=["claude-test"]) as mock_list, \
         patch("src.app._choose_models_interactively", return_value=["claude-test"]):
        mock_confirm.return_value.ask.side_effect = [True, False]
        cfg = app._configure_provider(env_path, "COUNCIL")

    assert cfg["provider_type"] == "anthropic"
    assert cfg["models"] == ["claude-test"]
    assert mock_list.call_args.args[2] == "anthropic"


def test_configure_provider_restores_previous_values_when_test_fails_and_user_reverts(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        textwrap.dedent(
            """
            SCOUT_API_KEY=sk-old
            SCOUT_BASE_URL=https://old.example/v1
            SCOUT_MODEL_NAMES=old-model
            """
        ).strip(),
        encoding="utf-8",
    )

    answers = iter([
        "",
        "https://new.example/v1",
        "sk-new",
        "new-model",
    ])

    def fake_input(_prompt):
        return next(answers)

    with patch("builtins.input", side_effect=fake_input), \
         patch("src.app._render_provider_config_summary"), \
         patch("src.app.questionary.confirm") as mock_confirm, \
         patch("src.app.questionary.select") as mock_select, \
         patch("src.app._test_provider_config", return_value=False):
        mock_confirm.return_value.ask.side_effect = [False, True]
        mock_select.return_value.ask.return_value = "回退到修改前配置（推荐）"
        cfg = app._configure_provider(env_path, "SCOUT")

    assert app.get_key(env_path, "SCOUT_API_KEY") == "sk-old"
    assert app.get_key(env_path, "SCOUT_BASE_URL") == "https://old.example/v1"
    assert app.get_key(env_path, "SCOUT_MODEL_NAMES") == "old-model"
    assert cfg["api_key"] == "sk-old"
    assert cfg["base_url"] == "https://old.example/v1"
    assert cfg["models"] == ["old-model"]


def test_assign_profile_to_provider_clears_direct_overrides(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        textwrap.dedent(
            """
            API_PROFILE_DEEPSEEK_PROVIDER_TYPE=openai
            API_PROFILE_DEEPSEEK_API_KEY=sk-profile
            API_PROFILE_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
            API_PROFILE_DEEPSEEK_MODEL_NAMES=deepseek-chat
            SCOUT_API_KEY=sk-old
            SCOUT_BASE_URL=https://old.example/v1
            SCOUT_MODEL_NAMES=old-model
            """
        ).strip(),
        encoding="utf-8",
    )

    cfg = app._assign_profile_to_provider(env_path, "SCOUT", "deepseek")

    assert app.get_key(env_path, "SCOUT_PROVIDER_PROFILE") == "DEEPSEEK"
    assert app.get_key(env_path, "SCOUT_API_KEY") is None
    assert app.get_key(env_path, "SCOUT_BASE_URL") is None
    assert app.get_key(env_path, "SCOUT_MODEL_NAMES") is None
    assert cfg["provider_profile"] == "deepseek"
    assert cfg["api_key"] == "sk-profile"
    assert cfg["models"] == ["deepseek-chat"]


def test_configure_api_profile_writes_reusable_profile(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    answers = iter([
        "deepseek",
        "openai",
        "https://api.deepseek.com/v1",
        "sk-new",
        "deepseek-chat",
    ])

    with patch("builtins.input", side_effect=lambda _prompt: next(answers)), \
         patch("src.app.questionary.confirm") as mock_confirm:
        mock_confirm.return_value.ask.return_value = False
        profile = app._configure_api_profile(env_path)

    assert profile["display_name"] == "deepseek"
    assert app.get_key(env_path, "API_PROFILE_DEEPSEEK_API_KEY") == "sk-new"
    assert app.get_key(env_path, "API_PROFILE_DEEPSEEK_BASE_URL") == "https://api.deepseek.com/v1"
    assert app.get_key(env_path, "API_PROFILE_DEEPSEEK_MODEL_NAMES") == "deepseek-chat"


def test_interactive_profile_create_does_not_reopen_editor(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    with patch("src.app.questionary.select") as mock_select, \
         patch("src.app._configure_api_profile", return_value={"display_name": "deepseek"}) as mock_configure:
        mock_select.return_value.ask.return_value = "➕ 新增 API 档案"
        selected = app._select_api_profile(env_path)

    assert selected == app.API_PROFILE_HANDLED
    mock_configure.assert_called_once()

    with patch.object(app, "PROJECT_ROOT", tmp_path), \
         patch("src.app._render_api_profile_overview"), \
         patch("src.app.questionary.select") as mock_select, \
         patch("src.app._select_api_profile", return_value=app.API_PROFILE_HANDLED), \
         patch("src.app._configure_api_profile") as mock_configure:
        mock_select.return_value.ask.return_value = "📚 管理 API 档案"
        app._interactive_config()

    mock_configure.assert_not_called()


def test_select_api_profile_returns_profile_from_choice_mapping(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        textwrap.dedent(
            """
            API_PROFILE_DEEPSEEK_API_KEY=sk-test
            API_PROFILE_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
            API_PROFILE_DEEPSEEK_MODEL_NAMES=deepseek-chat
            """
        ).strip(),
        encoding="utf-8",
    )

    with patch("src.app.questionary.select") as mock_select:
        mock_select.return_value.ask.return_value = "deepseek (openai / deepseek-chat)"
        selected = app._select_api_profile(env_path)

    assert selected == "deepseek"


def test_read_provider_config_from_env_file_includes_provider_type(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        "COUNCIL_PROVIDER_TYPE=anthropic\nCOUNCIL_BASE_URL=https://api.anthropic.com/v1\nCOUNCIL_MODEL_NAMES=claude-test\n",
        encoding="utf-8",
    )

    cfg = app._read_provider_config_from_env_file(env_path, "COUNCIL")

    assert cfg["provider_type"] == "anthropic"


def test_test_provider_config_passes_provider_type_to_chat():
    from src import app

    cfg = {
        "provider_type": "anthropic",
        "api_key": "sk-test",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-test"],
    }

    with patch("src.llm.client.chat", return_value="OK") as mock_chat:
        assert app._test_provider_config("COUNCIL", cfg) is True

    assert mock_chat.call_args.kwargs["provider_config"]["provider_type"] == "anthropic"


def test_embedding_provider_rejects_anthropic_with_clear_hint(capsys):
    from src import app

    cfg = {
        "provider_type": "anthropic",
        "api_key": "sk-test",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["text-embedding-3-small"],
    }

    ok = app._test_provider_config("EMBEDDING", cfg)

    out = capsys.readouterr().out
    assert ok is False
    assert "Embedding 不能使用 Anthropic 官方 Claude 接口" in out
    assert "OpenAI 兼容的 embeddings 接口" in out


def test_check_first_run_silent_accepts_openai_model_names(tmp_path):
    from src import app

    env_path = tmp_path / ".env"
    env_path.write_text(
        "OPENAI_API_KEY=sk-test\nOPENAI_MODEL_NAMES=claude-opus-4-6-thinking\n",
        encoding="utf-8",
    )

    with patch.object(app, "PROJECT_ROOT", tmp_path):
        assert app._check_first_run_silent() is True
