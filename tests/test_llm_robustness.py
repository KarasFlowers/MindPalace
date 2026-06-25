"""LLM 鲁棒性测试。验证多模型轮询、重试与 JSON 解析恢复逻辑。"""

import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError, APIConnectionError
from src.llm.client import chat, chat_json, chat_with_tools, _extract_json_text, _extract_message_content, _anthropic_request, _loads_json_lenient


def test_model_rotation_on_failure():
    """验证当第一个模型失败时，会自动尝试第二个模型。"""
    # 模拟配置了两个模型
    with patch("src.llm.client.PRIMARY_CONFIG", {"models": ["model-fail", "model-pass"], "api_key": "sk-123", "base_url": "url"}):
        with patch("src.llm.client._call_llm") as mock_call:
            # 第一次调用抛出异常，第二次调用成功
            def side_effect(sys, user, json, model, api_key, base_url, provider_type="openai", history=None):
                if model == "model-fail":
                    raise APIConnectionError(message="Connection failed", request=MagicMock())
                return "Success"

            mock_call.side_effect = side_effect

            # 我们不随机化以便测试
            with patch("random.shuffle", lambda x: x.sort()): # model-fail 会排在 model-pass 前面
                 result = chat("sys", "user")
                 assert result == "Success"
                 assert mock_call.call_count == 2


def test_retry_on_rate_limit():
    """验证遇到 RateLimitError 时会进行重试。"""
    with patch("src.llm.client.PRIMARY_CONFIG", {"models": ["model-1"], "api_key": "sk-123", "base_url": "url"}):
        with patch("src.llm.client._call_llm") as mock_call:
            with patch("time.sleep"): # 避免测试过慢
                # 模拟前两次 RateLimit，第三次成功
                mock_call.side_effect = [
                    RateLimitError(message="Rate limit", response=MagicMock(), body=None),
                    RateLimitError(message="Rate limit", response=MagicMock(), body=None),
                    "Final Success"
                ]
                
                result = chat("sys", "user", max_retries=3)
                assert result == "Final Success"
                assert mock_call.call_count == 3


def test_all_models_fail():
    """验证所有模型都失败后会抛出异常。"""
    with patch("src.llm.client.PRIMARY_CONFIG", {"models": ["m1", "m2"], "api_key": "sk-123", "base_url": "url"}):
        with patch("src.llm.client._call_llm") as mock_call:
            mock_call.side_effect = Exception("Total disaster")
            
            with pytest.raises(RuntimeError, match="所有 LLM 尝试均已失败"):
                chat("sys", "user", max_retries=1)


def test_chat_routes_anthropic_provider_to_messages_api():
    cfg = {
        "provider_type": "anthropic",
        "api_key": "sk-ant",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-test"],
    }

    with patch(
        "src.llm.client._anthropic_request",
        return_value={"content": [{"type": "text", "text": "hello"}]},
    ) as mock_request, patch("src.llm.client._get_user_profile", return_value=""):
        result = chat("sys", "user", provider_config=cfg, max_retries=1)

    assert result == "hello"
    payload = mock_request.call_args.args[2]
    assert payload["model"] == "claude-test"
    assert payload["system"] == "sys"
    assert payload["messages"] == [{"role": "user", "content": "user"}]


def test_chat_autodetects_anthropic_from_base_url():
    cfg = {
        "api_key": "sk-ant",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-test"],
    }

    with patch(
        "src.llm.client._anthropic_request",
        return_value={"content": [{"type": "text", "text": "auto"}]},
    ) as mock_request, patch("src.llm.client._get_user_profile", return_value=""):
        result = chat("sys", "user", provider_config=cfg, max_retries=1)

    assert result == "auto"
    mock_request.assert_called_once()


def test_anthropic_non_json_response_has_actionable_error():
    class Response:
        text = "<html>gateway error</html>"

        def raise_for_status(self):
            return None

        def json(self):
            import json
            raise json.JSONDecodeError("Expecting value", "", 0)

    with patch("httpx.post", return_value=Response()):
        with pytest.raises(RuntimeError, match="Anthropic 响应不是 JSON") as exc:
            _anthropic_request("sk-ant", "https://proxy.example/v1", {"model": "claude-test", "messages": []})

    assert "https://proxy.example/v1/messages" in str(exc.value)
    assert "不支持 Anthropic /v1/messages 协议" in str(exc.value)


def test_anthropic_request_adds_v1_for_compatible_proxy_root():
    class Response:
        text = '{"content":[{"type":"text","text":"ok"}]}'

        def raise_for_status(self):
            return None

        def json(self):
            return {"content": [{"type": "text", "text": "ok"}]}

    with patch("httpx.post", return_value=Response()) as mock_post:
        result = _anthropic_request("sk-ant", "https://yundu.lat", {"model": "claude-test", "messages": []})

    assert result["content"][0]["text"] == "ok"
    assert mock_post.call_args.args[0] == "https://yundu.lat/v1/messages"


def test_anthropic_non_json_chat_failure_suggests_provider_type_switch():
    cfg = {
        "provider_type": "anthropic",
        "api_key": "sk-ant",
        "base_url": "https://proxy.example/v1",
        "models": ["claude-test"],
    }

    with patch(
        "src.llm.client._anthropic_request",
        side_effect=RuntimeError("Anthropic 响应不是 JSON。URL=https://proxy.example/v1/messages"),
    ), patch("src.llm.client._get_user_profile", return_value=""):
        with pytest.raises(RuntimeError, match="PROVIDER_TYPE 改为 openai") as exc:
            chat("sys", "user", provider_config=cfg, max_retries=1)

    assert "Anthropic-compatible 中转站" in str(exc.value)


def test_chat_json_parses_anthropic_text_response():
    cfg = {
        "provider_type": "anthropic",
        "api_key": "sk-ant",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-test"],
    }

    with patch(
        "src.llm.client._anthropic_request",
        return_value={"content": [{"type": "text", "text": '{"ok": true}'}]},
    ), patch("src.llm.client._get_user_profile", return_value=""):
        result = chat_json("sys", "user", provider_config=cfg, max_retries=1)

    assert result == {"ok": True}


def test_chat_with_tools_supports_anthropic_tool_use_loop():
    cfg = {
        "provider_type": "anthropic",
        "api_key": "sk-ant",
        "base_url": "https://api.anthropic.com/v1",
        "models": ["claude-test"],
    }
    tool = MagicMock()
    tool.run.return_value = '{"results": []}'
    responses = [
        {
            "content": [
                {"type": "text", "text": "Searching"},
                {"type": "tool_use", "id": "toolu_1", "name": "web_search", "input": {"query": "test"}},
            ]
        },
        {"content": [{"type": "text", "text": '{"done": true}'}]},
    ]

    with patch("src.llm.client._anthropic_request", side_effect=responses) as mock_request, \
         patch("src.llm.client._get_user_profile", return_value=""):
        result = chat_with_tools(
            system_prompt="sys",
            user_prompt="user",
            tools_schema=[{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}],
            tool_executor={"web_search": tool},
            provider_config=cfg,
        )

    assert result["content"] == '{"done": true}'
    assert result["tool_calls_used"] == 1
    tool.run.assert_called_once_with(query="test")
    second_payload = mock_request.call_args_list[1].args[2]
    assert second_payload["messages"][1]["role"] == "assistant"
    assert second_payload["messages"][2]["content"][0]["type"] == "tool_result"


def test_extract_json_text_handles_code_fence():
    raw = "```json\n{\"ok\": true}\n```"
    assert _extract_json_text(raw) == '{"ok": true}'


def test_chat_json_retries_after_empty_response():
    with patch("src.llm.client.chat") as mock_chat:
        mock_chat.side_effect = ["", '{"summary": "ok"}']

        result = chat_json("sys", "user", max_retries=2)

        assert result == {"summary": "ok"}
        assert mock_chat.call_count == 2


def test_chat_json_repairs_unescaped_quotes_inside_string_values():
    raw = '''```json
{
  "core_stance": "用户将"家"定义为快乐的核心场域",
  "hidden_assumption": "隐含着"快乐需要一个稳定的物理容器"这一前提",
  "reflection": "家是一个很诚实的答案。"
}
```'''

    result = _loads_json_lenient(raw)

    assert result["core_stance"] == '用户将"家"定义为快乐的核心场域'
    assert result["hidden_assumption"] == '隐含着"快乐需要一个稳定的物理容器"这一前提'


def test_chat_json_uses_lenient_repair_without_retrying():
    raw = '{"core_stance": "用户将"家"定义为快乐的核心场域"}'

    with patch("src.llm.client.chat", return_value=raw) as mock_chat:
        result = chat_json("sys", "user", max_retries=3)

    assert result["core_stance"] == '用户将"家"定义为快乐的核心场域'
    assert mock_chat.call_count == 1


def test_chat_json_raises_clear_error_after_invalid_json():
    with patch("src.llm.client.chat", return_value=""):
        with pytest.raises(RuntimeError, match="LLM 返回了空响应或非 JSON 内容"):
            chat_json("sys", "user", max_retries=1)


def test_extract_message_content_raises_on_nonstandard_response():
    response = MagicMock()
    response.choices = "unexpected-string"

    with pytest.raises(RuntimeError, match="缺少 choices"):
        _extract_message_content(response)
