"""LLM 鲁棒性测试。验证多模型轮询、重试与 JSON 解析恢复逻辑。"""

import pytest
from unittest.mock import patch, MagicMock
from openai import RateLimitError, APIConnectionError
from src.llm.client import chat, chat_json, _extract_json_text


def test_model_rotation_on_failure():
    """验证当第一个模型失败时，会自动尝试第二个模型。"""
    # 模拟配置了两个模型
    with patch("src.llm.client.PRIMARY_CONFIG", {"models": ["model-fail", "model-pass"], "api_key": "sk-123", "base_url": "url"}):
        with patch("src.llm.client._call_llm") as mock_call:
            # 第一次调用抛出异常，第二次调用成功
            def side_effect(sys, user, json, model, api_key, base_url, history=None):
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


def test_extract_json_text_handles_code_fence():
    raw = "```json\n{\"ok\": true}\n```"
    assert _extract_json_text(raw) == '{"ok": true}'


def test_chat_json_retries_after_empty_response():
    with patch("src.llm.client.chat") as mock_chat:
        mock_chat.side_effect = ["", '{"summary": "ok"}']

        result = chat_json("sys", "user", max_retries=2)

        assert result == {"summary": "ok"}
        assert mock_chat.call_count == 2


def test_chat_json_raises_clear_error_after_invalid_json():
    with patch("src.llm.client.chat", return_value=""):
        with pytest.raises(RuntimeError, match="LLM 返回了空响应或非 JSON 内容"):
            chat_json("sys", "user", max_retries=1)
