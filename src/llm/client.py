"""LLM 调用封装。支持 OpenAI 兼容接口与 Anthropic 官方接口。"""

import json
import logging
import random
import threading
import time
from typing import Any

import httpx
from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
from src.config import PRIMARY_CONFIG, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 全局缓存用户画像
_user_profile_cache = None

def _get_user_profile() -> str:
    """读取并缓存全局用户画像。"""
    global _user_profile_cache
    if _user_profile_cache is not None:
        return _user_profile_cache

    profile_path = PROJECT_ROOT / "data" / "user_profile.md"
    if profile_path.exists():
        try:
            _user_profile_cache = profile_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to read user_profile.md: %s", e)
            _user_profile_cache = ""
    else:
        _user_profile_cache = ""

    return _user_profile_cache


def reset_user_profile_cache() -> None:
    """清除全局用户画像缓存，下次 LLM 调用时重新读取 data/user_profile.md。"""
    global _user_profile_cache
    _user_profile_cache = None


# 客户端连接池，以 (api_key, base_url) 为键
_client_pool = {}
_pool_lock = threading.Lock()
ANTHROPIC_API_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MAX_TOKENS = 2048


def _provider_type(cfg: dict | None) -> str:
    """推断 provider 协议类型。"""
    if not cfg:
        return "openai"
    explicit = (cfg.get("provider_type") or "").strip().lower()
    if explicit in {"anthropic", "claude"}:
        return "anthropic"
    base_url = (cfg.get("base_url") or "").lower()
    if "anthropic.com" in base_url:
        return "anthropic"
    return "openai"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if text:
            return str(text)
    return str(content)


def _openai_system_and_messages(system_prompt: str, history: list[dict] | None, user_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _anthropic_system_and_messages(system_prompt: str, history: list[dict] | None, user_prompt: str) -> tuple[str, list[dict]]:
    system_parts = []
    if system_prompt.strip():
        system_parts.append(system_prompt.strip())

    messages: list[dict] = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        text = _content_to_text(content)
        if role in {"system", "developer"}:
            if text:
                system_parts.append(text)
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        messages.append({"role": role, "content": content if content is not None else text})

    messages.append({"role": "user", "content": user_prompt})
    return "\n\n".join(system_parts), messages


def _anthropic_tools_from_openai_schema(tools_schema: list[dict]) -> list[dict]:
    tools = []
    for tool in tools_schema or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        parameters = function.get("parameters") or {"type": "object", "properties": {}}
        tools.append(
            {
                "name": function.get("name", "tool"),
                "description": function.get("description", ""),
                "input_schema": parameters,
            }
        )
    return tools


def _anthropic_api_url(base_url: str, endpoint: str) -> str:
    """Build Anthropic API URLs while accepting either root or versioned base URLs."""
    base = (base_url or "https://api.anthropic.com/v1").rstrip("/")
    path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    if not base.lower().endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}{path}"


def _anthropic_request(api_key: str, base_url: str, payload: dict) -> dict:
    url = _anthropic_api_url(base_url, "/messages")
    resp = httpx.post(
        url,
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
            "accept": "application/json",
        },
        json=payload,
        timeout=30.0,
        follow_redirects=True,
    )
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Anthropic 请求失败 ({resp.status_code}): {resp.text[:300]}"
        ) from exc
    try:
        return resp.json()
    except json.JSONDecodeError as exc:
        preview = (resp.text or "").replace("\n", " ").strip()[:300]
        raise RuntimeError(
            "Anthropic 响应不是 JSON。"
            f"URL={url}，响应预览={preview or '<empty>'}。"
            "这通常表示 BASE_URL 指向了网页、被中转站拦截，或该服务并不支持 Anthropic /v1/messages 协议。"
        ) from exc


def _anthropic_extract_text_and_tools(response: dict) -> tuple[str, list[dict]]:
    blocks = response.get("content") or []
    text_parts = []
    tool_uses = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if text:
                text_parts.append(str(text))
        elif block_type == "tool_use":
            tool_uses.append(block)
    return "\n".join(text_parts).strip(), tool_uses


def _format_model_failure_hint(
    model: str,
    error: Exception,
    base_url: str,
    provider_type: str = "openai",
) -> str:
    """把底层异常整理成更可执行的用户提示。"""
    error_text = str(error)
    if provider_type == "anthropic" and "响应不是 JSON" in error_text:
        return (
            f"模型 {model} 使用 Anthropic 协议调用失败：{error_text} "
            "如果你使用的是 Anthropic 官方 API，请确认 BASE_URL 为 https://api.anthropic.com/v1；"
            "如果你使用的是 Anthropic-compatible 中转站，BASE_URL 可填中转根地址或 /v1 地址，"
            "程序会请求 /v1/messages；如果该中转站实际暴露的是 OpenAI 兼容协议，"
            "请把 PROVIDER_TYPE 改为 openai，并使用网关提供的兼容地址和模型名。"
        )
    if isinstance(error, APIStatusError) and error.status_code == 404:
        return (
            f"模型 {model} 在 {base_url} 不可用（404）。"
            " 请检查这个 provider 的 BASE_URL 和 MODEL_NAMES 是否与服务端实际支持的模型一致。"
        )
    if "choices" in error_text and "attribute" in error_text:
        return (
            f"模型 {model} 返回了非标准 OpenAI chat.completions 响应。"
            " 当前客户端期望拿到带 choices 的兼容格式，请检查所填 BASE_URL 是否真的是 OpenAI 兼容接口。"
        )
    return f"模型 {model} 调用失败：{error_text}"


def _extract_message_content(response) -> str:
    """兼容不同 SDK/代理返回形状，安全提取 message.content。"""
    choices = getattr(response, "choices", None)
    if not choices or not isinstance(choices, (list, tuple)):
        raise RuntimeError(
            "LLM 返回了非标准响应：缺少 choices。请检查 BASE_URL 是否为兼容的 chat.completions 接口。"
        )

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise RuntimeError("LLM 返回了非标准响应：choices[0] 缺少 message。")

    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    text_parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    text_parts.append(str(text))
        return "\n".join(text_parts).strip()
    return str(content)


def _get_client_from_pool(api_key: str, base_url: str) -> OpenAI:
    key = (api_key, base_url)
    with _pool_lock:
        if key not in _client_pool:
            if not api_key:
                raise RuntimeError(
                    "API Key 未设置。请通过 'python -m src config' 进行配置。"
                )
            _client_pool[key] = OpenAI(api_key=api_key, base_url=base_url)
        return _client_pool[key]


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool,
    model: str,
    api_key: str,
    base_url: str,
    provider_type: str = "openai",
    history: list[dict] | None = None,
) -> str:
    """内部直接调用 LLM，不含重试逻辑。"""
    if provider_type == "anthropic":
        system_text, messages = _anthropic_system_and_messages(system_prompt, history, user_prompt)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": DEFAULT_ANTHROPIC_MAX_TOKENS,
        }
        if system_text:
            payload["system"] = system_text
        if json_mode:
            payload["temperature"] = 0.0
            payload["system"] = (payload.get("system", "") + "\n\nReturn valid JSON text only.").strip()
        response = _anthropic_request(api_key, base_url, payload)
        text, _tool_uses = _anthropic_extract_text_and_tools(response)
        return text

    client = _get_client_from_pool(api_key, base_url)
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    messages = _openai_system_and_messages(system_prompt, history, user_prompt)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        **kwargs,
    )
    return _extract_message_content(response)


def chat(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    model: str | None = None,
    max_retries: int = 3,
    provider_config: dict | None = None,
    history: list[dict] | None = None,
) -> str:
    """调用 LLM，支持故障转移与自动重试。"""
    cfg = provider_config or PRIMARY_CONFIG
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "")
    provider = _provider_type(cfg)
    
    # 注入用户全局画像
    profile = _get_user_profile()
    if profile:
        system_prompt = f"{system_prompt.strip()}\n\n=== User Profile ===\n{profile}"
    
    # 确定要尝试的模型列表
    if model:
        models_to_try = [model]
    elif cfg.get("models"):
        models_to_try = cfg["models"].copy()
        random.shuffle(models_to_try)
    else:
        raise RuntimeError(
            f"未配置任何模型。请在 .env 中设置相应的 MODEL_NAMES。"
        )

    last_error = None
    blocked_models = []  # 记录被安全过滤的模型

    for m in models_to_try:
        logger.debug("尝试使用模型: %s (Provider: %s)", m, base_url)
        for attempt in range(max_retries):
            try:
                return _call_llm(
                    system_prompt,
                    user_prompt,
                    json_mode,
                    m,
                    api_key,
                    base_url,
                    provider_type=provider,
                    history=history,
                )

            except (RateLimitError, APITimeoutError) as e:
                last_error = e
                wait_time = (2**attempt) + random.random()
                logger.warning(
                    "模型 %s 频率限制或超时，%.1fs 后重试 (%d/%d)...",
                    m, wait_time, attempt + 1, max_retries,
                )
                time.sleep(wait_time)

            except APIStatusError as e:
                last_error = e
                # 403 通常是内容安全过滤，记录后跳过该模型
                if e.status_code == 403:
                    logger.warning("模型 %s 内容被安全过滤阻止，尝试下一个模型...", m)
                    blocked_models.append(m)
                    break
                elif e.status_code in (404, 503):
                    logger.error("模型 %s 通道不可用 (状态码: %d): %s", m, e.status_code, e.message)
                    break 
                else:
                    logger.error("模型 %s 返回状态错误 (%d): %s", m, e.status_code, e.message)
                    break

            except APIConnectionError as e:
                last_error = e
                logger.error("模型 %s 连接失败: %s", m, e)
                break

            except Exception as e:
                last_error = e
                logger.error("模型 %s 发生未知错误: %s", m, e)
                break

    # 如果所有模型都被安全过滤阻止，给出更友好的错误信息
    if len(blocked_models) == len(models_to_try):
        raise RuntimeError(
            f"所有模型均被内容安全过滤阻止。这通常是因为内容触发了 API 的安全策略。"
        )
    
    model_name = models_to_try[-1]
    hint = _format_model_failure_hint(model_name, last_error, base_url, provider) if last_error else "未知错误"
    raise RuntimeError(f"所有 LLM 尝试均已失败。最后一次错误 (模型:{model_name}): {hint}")


def chat_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools_schema: list[dict],
    tool_executor: dict,
    max_tool_calls: int = 3,
    provider_config: dict | None = None,
    history: list[dict] | None = None,
) -> dict:
    """带 tool-use 循环的 LLM 调用。

    Args:
        system_prompt: 系统提示。
        user_prompt: 用户提示。
        tools_schema: OpenAI function calling 格式的 tools 列表。
        tool_executor: 工具名 -> Tool 实例的映射（需有 .run(**kwargs) 方法）。
        max_tool_calls: 最大工具调用轮数。
        provider_config: LLM 配置。
        history: 对话历史。

    Returns:
        {"content": str, "tool_calls_used": int, "tool_log": list[dict]}
    """
    cfg = provider_config or PRIMARY_CONFIG
    api_key = cfg.get("api_key", "")
    base_url = cfg.get("base_url", "")
    models = cfg.get("models", [])
    provider = _provider_type(cfg)
    if not models:
        raise RuntimeError("未配置任何模型。")
    model = random.choice(models)

    profile = _get_user_profile()
    if profile:
        system_prompt = f"{system_prompt.strip()}\n\n=== User Profile ===\n{profile}"

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    tool_log: list[dict] = []
    calls_used = 0

    for i in range(max_tool_calls + 1):
        final_round = i == max_tool_calls

        if final_round:
            messages.append({
                "role": "system",
                "content": "⚠️ 你必须立即给出最终答复，不得再调用任何工具。",
            })

        if provider == "anthropic":
            anth_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if role in {"system", "developer"}:
                    continue
                if role == "tool":
                    anth_messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id"), "content": content}],
                    })
                    continue
                if role not in {"user", "assistant"}:
                    role = "user"
                anth_messages.append({"role": role, "content": content})

            payload = {
                "model": model,
                "messages": anth_messages,
                "max_tokens": DEFAULT_ANTHROPIC_MAX_TOKENS,
            }
            system_text = ""
            system_items = []
            for msg in messages:
                if msg.get("role") in {"system", "developer"} and msg.get("content"):
                    system_items.append(_content_to_text(msg.get("content")))
            if system_items:
                system_text = "\n\n".join(system_items)
                payload["system"] = system_text
            if not final_round and tools_schema:
                payload["tools"] = _anthropic_tools_from_openai_schema(tools_schema)

            resp = None
            last_exc = None
            for attempt in range(3):
                try:
                    resp = _anthropic_request(api_key, base_url, payload)
                    break
                except Exception as exc:
                    last_exc = exc
                    logger.warning("chat_with_tools anthropic error on attempt %d: %s", attempt + 1, exc)
                    time.sleep(1 + attempt)

            if resp is None:
                err_msg = f"[LLM 调用失败: {last_exc}]" if last_exc else "[LLM 调用失败]"
                logger.error("chat_with_tools exhausted retries: %s", err_msg)
                return {"content": err_msg, "tool_calls_used": calls_used, "tool_log": tool_log}

            content, tool_uses = _anthropic_extract_text_and_tools(resp)
            if tool_uses and not final_round:
                messages.append({"role": "assistant", "content": resp.get("content") or []})
                tool_result_blocks = []
                for block in tool_uses:
                    fn_name = block.get("name")
                    fn_args = block.get("input") or {}
                    logger.info("[ToolUse] calling %s(%s)", fn_name, fn_args)
                    if fn_name in tool_executor:
                        try:
                            result = tool_executor[fn_name].run(**fn_args)
                        except Exception as exc:
                            result = json.dumps({"error": str(exc)}, ensure_ascii=False)
                            logger.warning("[ToolUse] %s failed: %s", fn_name, exc)
                    else:
                        result = json.dumps({"error": f"Unknown tool: {fn_name}"}, ensure_ascii=False)
                    result_text = str(result)
                    tool_result_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": block.get("id"),
                        "content": result_text,
                    })
                    tool_log.append({"tool": fn_name, "args": fn_args, "result_preview": result_text[:200]})
                messages.append({"role": "user", "content": tool_result_blocks})
                calls_used += 1
                continue

            return {"content": content, "tool_calls_used": calls_used, "tool_log": tool_log}

        client = _get_client_from_pool(api_key, base_url)
        kwargs: dict = {"model": model, "messages": messages, "temperature": 0.4}
        if not final_round and tools_schema:
            kwargs["tools"] = tools_schema
            kwargs["tool_choice"] = "auto"

        # 重试机制（与 chat() 保持一致）
        resp = None
        last_exc = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except RateLimitError:
                wait = 2 * (attempt + 1)
                logger.warning("chat_with_tools rate limited, waiting %ss...", wait)
                time.sleep(wait)
            except APITimeoutError:
                logger.warning("chat_with_tools timeout on attempt %d", attempt + 1)
            except APIStatusError as e:
                if e.status_code in (403, 401):
                    logger.error("chat_with_tools auth error (%s), aborting.", e.status_code)
                    break
                elif e.status_code in (404, 503):
                    logger.warning("chat_with_tools server error (%s), retrying...", e.status_code)
                else:
                    logger.warning("chat_with_tools API status %s on attempt %d", e.status_code, attempt + 1)
            except Exception as exc:
                logger.warning("chat_with_tools unexpected error on attempt %d: %s", attempt + 1, exc)
                last_exc = exc

        if resp is None:
            err_msg = f"[LLM 调用失败: {last_exc}]" if last_exc else "[LLM 调用失败]"
            logger.error("chat_with_tools exhausted retries: %s", err_msg)
            return {"content": err_msg, "tool_calls_used": calls_used, "tool_log": tool_log}

        msg = resp.choices[0].message

        if msg.tool_calls and not final_round:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                logger.info("[ToolUse] calling %s(%s)", fn_name, fn_args)

                if fn_name in tool_executor:
                    try:
                        result = tool_executor[fn_name].run(**fn_args)
                    except Exception as exc:
                        result = json.dumps({"error": str(exc)}, ensure_ascii=False)
                        logger.warning("[ToolUse] %s failed: %s", fn_name, exc)
                else:
                    result = json.dumps({"error": f"Unknown tool: {fn_name}"}, ensure_ascii=False)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                tool_log.append({"tool": fn_name, "args": fn_args, "result_preview": result[:200]})
            calls_used += 1
            continue

        content = msg.content or ""
        return {"content": content, "tool_calls_used": calls_used, "tool_log": tool_log}

    return {"content": "", "tool_calls_used": calls_used, "tool_log": tool_log}


def _preview_response(raw: str | None, limit: int = 80) -> str:
    text = (raw or "").strip()
    if not text:
        return "<empty response>"
    return text.replace("\n", " ")[:limit]


def _extract_json_text(raw: str) -> str:
    text = raw.strip()
    if not text:
        raise ValueError("LLM returned an empty response")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if "\n" in text:
        first_line, rest = text.split("\n", 1)
        if first_line.strip().lower() == "json":
            text = rest.strip()

    decoder = json.JSONDecoder()

    try:
        _, end = decoder.raw_decode(text)
        return text[:end]
    except json.JSONDecodeError:
        pass

    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
            return text[index:index + end]
        except json.JSONDecodeError:
            continue

    raise ValueError("LLM response did not contain JSON")


def _repair_unescaped_quotes_in_json_strings(text: str) -> str:
    """Repair common LLM JSON slips like `"用户将"家"定义为..."`.

    The repair is intentionally conservative: it only runs after normal JSON
    parsing fails, and only escapes quote characters inside a JSON string when
    the next non-space character is not a JSON structural delimiter.
    """
    out: list[str] = []
    in_string = False
    escape = False
    length = len(text)

    for index, char in enumerate(text):
        if escape:
            out.append(char)
            escape = False
            continue

        if char == "\\" and in_string:
            out.append(char)
            escape = True
            continue

        if char != '"':
            out.append(char)
            continue

        if not in_string:
            in_string = True
            out.append(char)
            continue

        next_index = index + 1
        while next_index < length and text[next_index].isspace():
            next_index += 1
        next_char = text[next_index] if next_index < length else ""
        if next_char in {":", ",", "}", "]", ""}:
            in_string = False
            out.append(char)
        else:
            out.append('\\"')

    return "".join(out)


def _extract_probable_json_text(raw: str) -> str:
    """Return the broadest likely JSON block without requiring valid JSON."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if "\n" in text:
        first_line, rest = text.split("\n", 1)
        if first_line.strip().lower() == "json":
            text = rest.strip()

    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("LLM response did not contain JSON")
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    end = text.rfind(closer)
    if end < start:
        raise ValueError("LLM response did not contain a complete JSON block")
    return text[start:end + 1]


def _loads_json_lenient(raw: str) -> dict:
    """Parse JSON with a small recovery pass for common LLM formatting slips."""
    try:
        text = _extract_json_text(raw)
    except ValueError:
        text = _extract_probable_json_text(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _repair_unescaped_quotes_in_json_strings(text)
        if repaired != text:
            return json.loads(repaired)
        raise


def chat_json(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    max_retries: int = 3,
    provider_config: dict | None = None,
) -> dict:
    """调用 LLM 并返回解析后的 JSON dict。"""
    attempts = max(1, max_retries)
    last_error = None

    for attempt in range(attempts):
        raw = chat(
            system_prompt,
            user_prompt,
            json_mode=True,
            model=model,
            max_retries=max_retries if attempt == 0 else 1,
            provider_config=provider_config,
        )

        try:
            return _loads_json_lenient(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            preview = _preview_response(raw)
            if attempt == attempts - 1:
                logger.error("LLM 返回了无法解析的 JSON: %s", preview)
                raise RuntimeError("LLM 返回了空响应或非 JSON 内容。") from exc
            logger.warning(
                "LLM 返回了无效 JSON，准备重试 (%d/%d)。预览: %s",
                attempt + 1,
                attempts,
                preview,
            )

    raise RuntimeError("LLM 返回了空响应或非 JSON 内容。") from last_error
