"""Observability 模块测试。

验证：
1. 默认关闭时 span 是 OTel no-op，零开销
2. init_tracing(force=False) 不启用
3. span 上下文管理器正常工作
4. 缺少 Phoenix 时 init_tracing 优雅降级
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from opentelemetry import trace


class TestSpanNoOp:
    """追踪关闭时 span 必须是 no-op。"""

    def test_span_returns_context_manager(self):
        from src.obs import span
        ctx = span("test.noop", key="value")
        # 应该可以用 with 语句
        with ctx as s:
            assert s is not None  # OTel no-op span 对象

    def test_span_noop_has_no_side_effects(self):
        from src.obs import span
        # 不应抛出异常
        with span("test.noop2", x=1, y="hello"):
            pass

    def test_span_filters_none_attributes(self):
        from src.obs import span
        # None 值应被过滤，不应报错
        with span("test.none_attr", good="ok", bad=None):
            pass


class TestInitTracing:
    def test_disabled_by_default(self):
        from src.obs.tracing import init_tracing, is_enabled
        result = init_tracing(force=False)
        assert result is False
        assert is_enabled() is False

    def test_enabled_but_missing_deps(self):
        """启用但缺少 Phoenix 时应优雅降级。"""
        from src.obs import tracing
        # 重置状态
        tracing._enabled = False

        with patch.dict("sys.modules", {"phoenix": None}):
            result = tracing.init_tracing(force=True)
            # 应该返回 False（导入失败）
            assert result is False
            assert tracing._enabled is False


class TestShutdown:
    def test_shutdown_noop_when_disabled(self):
        """关闭状态下 shutdown 不应报错。"""
        from src.obs import shutdown_tracing
        shutdown_tracing()  # 不应抛异常


class TestSpanIntegration:
    """验证 span 在 OTel no-op 模式下的属性设置。"""

    def test_set_attribute_on_span(self):
        from src.obs import span
        with span("test.attrs", initial="val") as s:
            # no-op span 的 set_attribute 不应报错
            s.set_attribute("dynamic_key", 42)
            s.set_attribute("another", "str_value")
