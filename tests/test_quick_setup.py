"""快速配置流程（_quick_setup）的测试。

验证：
1. 只配对话型档案 → 全部对话节点绑定该档案，模型可解析。
2. 检索型节点在"不单独配置"时回落到全局默认（OPENAI 档案）。
3. 询问顺序正确：先配对话型，再询问检索型。
"""

from unittest.mock import patch

import src.app as app


class _FakeConfirm:
    """模拟 questionary.confirm().ask() 的返回序列。"""

    def __init__(self, returns):
        self._iter = iter(returns)

    def __call__(self, *_args, **_kwargs):
        return self

    def ask(self):
        return next(self._iter)


def _make_input_handler(base_url, api_key, models):
    """生成一个按字段内容分流的 fake input。"""
    state = {"models_returned": False}

    def fake_input(prompt):
        text = prompt
        if "BASE_URL" in text:
            return base_url
        if "API_KEY" in text:
            return api_key
        raise AssertionError(f"unexpected input prompt: {prompt}")

    return fake_input


def test_quick_setup_chat_profile_binds_to_all_chat_providers(tmp_path):
    """配置对话型档案后，全部对话节点应解析到该档案。"""
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    with patch.object(app, "PROJECT_ROOT", tmp_path), \
         patch("builtins.input", side_effect=_make_input_handler(
             "https://api.deepseek.com/v1", "sk-deep", "deepseek-chat"
         )), \
         patch("src.app._list_remote_models", return_value=["deepseek-chat"]), \
         patch("src.app._choose_models_interactively", return_value=["deepseek-chat"]), \
         patch("src.app.questionary.confirm", side_effect=_FakeConfirm([False])):
        ok = app._quick_setup()

    assert ok is True

    # 全部对话节点都应绑定到 main 档案
    for prefix in app._CHAT_PROVIDER_PREFIXES:
        binding = app.get_key(env_path, f"{prefix}_PROVIDER_PROFILE")
        assert binding == "MAIN", f"{prefix} 未绑定到 MAIN 档案（实际: {binding}）"

        # 直填字段应被清空（档案优先）
        assert app.get_key(env_path, f"{prefix}_API_KEY") is None
        assert app.get_key(env_path, f"{prefix}_MODEL_NAMES") is None

    # 档案本体应写入正确字段
    assert app.get_key(env_path, "API_PROFILE_MAIN_API_KEY") == "sk-deep"
    assert app.get_key(env_path, "API_PROFILE_MAIN_BASE_URL") == "https://api.deepseek.com/v1"
    assert app.get_key(env_path, "API_PROFILE_MAIN_MODEL_NAMES") == "deepseek-chat"


def test_quick_setup_retrieval_declined_falls_back_to_global(tmp_path):
    """拒绝单独配置检索型档案时，EMBEDDING/SCOUT 应回落到全局默认。"""
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    with patch.object(app, "PROJECT_ROOT", tmp_path), \
         patch("builtins.input", side_effect=_make_input_handler(
             "https://api.deepseek.com/v1", "sk-deep", "deepseek-chat"
         )), \
         patch("src.app._list_remote_models", return_value=["deepseek-chat"]), \
         patch("src.app._choose_models_interactively", return_value=["deepseek-chat"]), \
         patch("src.app.questionary.confirm", side_effect=_FakeConfirm([False])):
        app._quick_setup()

    # 检索节点不应有独立档案绑定（回落到全局 OPENAI）
    for prefix in app._RETRIEVAL_PROVIDER_PREFIXES:
        binding = app.get_key(env_path, f"{prefix}_PROVIDER_PROFILE")
        assert binding is None, f"{prefix} 不应有档案绑定（实际: {binding}）"
        assert app.get_key(env_path, f"{prefix}_API_KEY") is None


def test_quick_setup_retrieval_accepted_binds_separate_profile(tmp_path):
    """同意配置检索型档案时，EMBEDDING/SCOUT 应绑定到独立检索档案。"""
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    # input 会按顺序被调用两次（对话型 + 检索型），用迭代器
    inputs = iter([
        "https://api.deepseek.com/v1",  # 对话 base_url
        "sk-deep",                       # 对话 api_key
        "https://cheap.example/v1",      # 检索 base_url
        "sk-cheap",                      # 检索 api_key
    ])

    def fake_input(_prompt):
        return next(inputs)

    with patch.object(app, "PROJECT_ROOT", tmp_path), \
         patch("builtins.input", side_effect=fake_input), \
         patch("src.app._list_remote_models",
               return_value=["deepseek-chat"]), \
         patch("src.app._choose_models_interactively",
               return_value=["deepseek-chat"]), \
         patch("src.app.questionary.confirm",
               side_effect=_FakeConfirm([True])):  # 同意配检索型
        app._quick_setup()

    # 检索节点应绑定到 retrieval 档案
    for prefix in app._RETRIEVAL_PROVIDER_PREFIXES:
        binding = app.get_key(env_path, f"{prefix}_PROVIDER_PROFILE")
        assert binding == "RETRIEVAL", f"{prefix} 应绑定到 RETRIEVAL（实际: {binding}）"

    assert app.get_key(env_path, "API_PROFILE_RETRIEVAL_API_KEY") == "sk-cheap"
    assert app.get_key(env_path, "API_PROFILE_RETRIEVAL_BASE_URL") == "https://cheap.example/v1"


def test_quick_setup_chat_cancelled_returns_false(tmp_path):
    """对话型档案配置中途取消（输入清空哨兵）时返回 False。"""
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    def fake_input(prompt):
        if "BASE_URL" in prompt:
            return app.CLEAR_CONFIG_SENTINEL  # 取消
        raise AssertionError(f"unexpected prompt: {prompt}")

    with patch.object(app, "PROJECT_ROOT", tmp_path), \
         patch("builtins.input", side_effect=fake_input):
        ok = app._quick_setup()

    assert ok is False


def test_quick_setup_configure_profile_handles_fetch_failure(tmp_path, capsys):
    """模型抓取失败时，档案仍应保存，且给出友好提示。"""
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")

    with patch.object(app, "PROJECT_ROOT", tmp_path), \
         patch("builtins.input", side_effect=_make_input_handler(
             "https://broken.example/v1", "sk-test", ""
         )), \
         patch("src.app._list_remote_models",
               side_effect=RuntimeError("connection refused")), \
         patch("src.app.questionary.confirm", side_effect=_FakeConfirm([False])):
        result = app._quick_setup_configure_profile(
            env_path, "main", "测试", "测试提示"
        )

    # 模型抓取失败，但 base_url + api_key 仍应保存
    assert result == "MAIN"
    assert app.get_key(env_path, "API_PROFILE_MAIN_API_KEY") == "sk-test"
    assert app.get_key(env_path, "API_PROFILE_MAIN_BASE_URL") == "https://broken.example/v1"
    # 模型未写入（抓取失败）
    assert app.get_key(env_path, "API_PROFILE_MAIN_MODEL_NAMES") is None

    out = capsys.readouterr().out
    assert "无法抓取模型列表" in out
