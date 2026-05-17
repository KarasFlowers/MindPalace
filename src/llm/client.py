"""LLM 调用封装。统一使用 OpenAI 兼容 API。"""

import json
import logging
import random
import threading
import time
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
        except Exception as e:
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
    history: list[dict] | None = None,
) -> str:
    """内部直接调用 LLM，不含重试逻辑。"""
    client = _get_client_from_pool(api_key, base_url)
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        **kwargs,
    )
    return response.choices[0].message.content or ""


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
                return _call_llm(system_prompt, user_prompt, json_mode, m, api_key, base_url, history=history)

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
    
    raise RuntimeError(f"所有 LLM 尝试均已失败。最后一次错误 (模型:{models_to_try[-1]}): {last_error}")


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

    client = _get_client_from_pool(api_key, base_url)
    tool_log: list[dict] = []
    calls_used = 0

    for i in range(max_tool_calls + 1):
        final_round = i == max_tool_calls

        if final_round:
            messages.append({
                "role": "system",
                "content": "⚠️ 你必须立即给出最终答复，不得再调用任何工具。",
            })

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


def _preview_response(raw: str | None, limit: int = 200) -> str:
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
            return json.loads(_extract_json_text(raw))
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            preview = _preview_response(raw)
            if attempt == attempts - 1:
                logger.error("LLM 返回了无法解析的 JSON: %s", preview)
                raise RuntimeError("LLM 返回了空响应或非 JSON 内容。") from exc
            logger.warning(
                "LLM 返回了无效 JSON，准备重试 (%d/%d): %s",
                attempt + 1,
                attempts,
                preview,
            )

    raise RuntimeError("LLM 返回了空响应或非 JSON 内容。") from last_error
