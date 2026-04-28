"""MindPalace 可观测性 (Observability) 模块。

提供 OpenTelemetry 兼容的链路追踪 (Tracing) 能力。默认关闭，零开销。
启用时会将所有 LLM 调用、Council 辩论、Daily Session 的完整执行链路
可视化到本地 Arize Phoenix UI (http://localhost:6006)。

使用：
    from src.obs import span, init_tracing

    init_tracing()  # 在进程启动时调用一次

    with span("council.debate", title=title, difficulty=difficulty):
        ...  # 内部所有 LLM 调用会自动归属到此 span 下
"""

from src.obs.tracing import init_tracing, span, shutdown_tracing, is_enabled

__all__ = ["init_tracing", "span", "shutdown_tracing", "is_enabled"]
