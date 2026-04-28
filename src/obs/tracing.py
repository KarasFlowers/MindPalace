"""OpenTelemetry + Arize Phoenix 链路追踪初始化。

设计原则：
1. 默认关闭（TRACING_ENABLED=false），零开销：span() 返回 OTel no-op context
2. 仅在 TRACING_ENABLED=true 时尝试加载 Phoenix + OpenInference
3. 依赖缺失时只打 warning，不中断程序
4. OpenAI SDK 自动埋点：所有经过 openai.Client 的调用自动生成 span
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace

logger = logging.getLogger(__name__)

_tracer: trace.Tracer = trace.get_tracer("mindpalace")
_enabled: bool = False
_phoenix_session: Any = None


def is_enabled() -> bool:
    """返回追踪是否已启用。"""
    return _enabled


def init_tracing(force: bool | None = None) -> bool:
    """初始化 OTel + Phoenix。

    Args:
        force: 显式开关。None 时读取 TRACING_ENABLED 环境变量。

    Returns:
        True 如果成功初始化，否则 False。
    """
    global _enabled, _phoenix_session

    if _enabled:
        return True

    enabled = force
    if enabled is None:
        enabled = os.environ.get("TRACING_ENABLED", "false").lower() in ("1", "true", "yes")

    if not enabled:
        return False

    # ── 尝试加载 Phoenix 全家桶 ──
    try:
        import phoenix as px
        from phoenix.otel import register as phoenix_register
        from openinference.instrumentation.openai import OpenAIInstrumentor
    except ImportError as e:
        logger.warning(
            "Tracing requested but optional deps missing (%s). "
            "Install with: pip install -e '.[obs]'",
            e,
        )
        return False

    try:
        # 启动本地 Phoenix UI (http://localhost:6006)
        _phoenix_session = px.launch_app()
        logger.info("Phoenix UI started at %s", _phoenix_session.url)

        # 注册 OTel tracer provider → 数据发往 Phoenix
        tracer_provider = phoenix_register(
            project_name="mindpalace",
            endpoint=f"http://localhost:6006/v1/traces",
        )

        # 自动埋点 openai SDK（覆盖所有 chat/embedding 调用）
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        _enabled = True
        logger.info("MindPalace tracing enabled (Phoenix + OpenTelemetry)")
        return True

    except Exception:
        logger.exception("Failed to initialize tracing")
        return False


def shutdown_tracing() -> None:
    """关闭 Phoenix 进程和 OTel provider。"""
    global _enabled, _phoenix_session
    if not _enabled:
        return

    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception:
        logger.debug("OTel provider shutdown error", exc_info=True)

    try:
        if _phoenix_session is not None:
            import phoenix as px
            px.close_app()
            _phoenix_session = None
    except Exception:
        logger.debug("Phoenix shutdown error", exc_info=True)

    _enabled = False
    logger.info("Tracing shut down")


def span(name: str, **attributes: Any) -> trace.Span:
    """创建一个 OTel span 作为 context manager。

    未启用时退化为 OTel 内置的 no-op span（零开销）。

    用法::

        with span("council.debate", title="AI Ethics", difficulty="hard") as s:
            ...
            s.set_attribute("tool_calls", 3)  # 可追加属性
    """
    # 过滤掉 None 值，OTel 不接受 None 属性
    clean_attrs = {k: v for k, v in attributes.items() if v is not None}
    return _tracer.start_as_current_span(name, attributes=clean_attrs)
