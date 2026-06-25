"""用户体验 (UX) 工具模块。

提供终端进度指示与统一的多行输入收集器，统一全 CLI 的交互观感。

设计原则：
1. 零依赖：仅使用 sys.stdout，不引入第三方库
2. 跨平台：使用 CR + erase line (`\\r\\x1b[2K`) 做行覆写，Windows 10+ 与 Unix 均支持
3. 非阻塞：进度通过回调驱动，不使用线程，避免与同步 LLM 调用竞争
4. 可关闭：所有进度/动画在非 TTY（如管道/重定向）下自动降级为无输出
"""

from src.ux.input import collect_multiline
from src.ux.progress import PhaseIndicator, Spinner

__all__ = ["PhaseIndicator", "Spinner", "collect_multiline"]
