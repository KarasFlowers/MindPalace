"""终端进度指示器。

提供两类轻量进度反馈：

- :class:`PhaseIndicator`：阶段式进度（``[2/5] Critic 正在分析...``），
  通过回调驱动，适合已知总阶段数的流程（Council 辩论、Daily Session）。
- :class:`Spinner`：行内旋转动画，适合未知时长的单一阻塞操作。

两者都基于 ``\\r\\x1b[2K``（回车 + 清行）做行内覆写，不依赖 ANSI 光标控制序列，
在 Windows 10+ 的 conhost / Windows Terminal 与 Unix 终端均能正常工作。

非 TTY（输出被重定向/管道）时会自动降级为无输出，避免日志文件里出现大量回车。
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Callable

# ANSI 颜色（与项目其它模块保持一致）
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

# 盲文旋转帧（Unicode Braille patterns），顺时针视觉效果较好。
# 其它预设保持克制，用于区分“抓取 / 生成 / 评估”等不同等待气质。
_SPINNER_PRESETS = {
    "braille": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    "dots": "⠁⠂⠄⡀⢀⠠⠐⠈",
    "pulse": "·∙●∙",
    "bar": "▏▎▍▌▋▊▉█▉▊▋▌▍▎",
    "moon": "◐◓◑◒",
}


def _is_tty() -> bool:
    """判断 stdout 是否为交互式终端。重定向/管道时返回 False。"""
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _clear_line() -> None:
    """擦除当前行并将光标回到行首。"""
    sys.stdout.write("\r\x1b[2K")
    sys.stdout.flush()


class PhaseIndicator:
    """阶段式进度指示器，通过回调驱动，不使用线程。

    用法::

        indicator = PhaseIndicator(total=5)
        run_council(on_phase=indicator.advance)
        indicator.done()

    每次 :meth:`advance` 调用会在终端同一行覆写为 ``[i/total] label``，
    适用于在多次同步 LLM 调用之间显示进度。

    在非 TTY 环境下 :meth:`advance` 为空操作，:meth:`done` 也不会输出任何字符，
    因此可以安全地嵌入到任何流程中而不污染日志文件。
    """

    def __init__(self, total: int) -> None:
        if total < 1:
            raise ValueError("total 必须大于等于 1")
        self._total = total
        self._current = 0
        self._active = _is_tty()

    def advance(self, label: str) -> None:
        """推进到下一阶段并刷新终端行。

        Args:
            label: 当前阶段的简短描述（如 ``"Critic 正在分析..."``）。

        计数（逻辑进度）始终推进，与是否为 TTY 无关；只有终端写入受 TTY 控制，
        这样在重定向/管道环境下也能通过 :attr:`current` 查询进度。
        """
        self._current += 1
        if not self._active:
            return
        # 限制显示的 current 不超过 total，避免显示成 [6/5]
        shown = min(self._current, self._total)
        text = f"  {DIM}[{shown}/{self._total}]{RESET} {label}"
        sys.stdout.write(f"\r\x1b[2K{text}")
        sys.stdout.flush()

    def done(self, final_message: str | None = None) -> None:
        """结束进度显示。

        Args:
            final_message: 若提供，会打印一行完成消息；否则仅清除进度行。
        """
        if not self._active:
            if final_message:
                print(final_message)
            return
        _clear_line()
        if final_message:
            print(final_message)

    @property
    def current(self) -> int:
        """当前已推进的阶段数。"""
        return self._current


class Spinner:
    """行内旋转动画，用于未知时长的单一阻塞操作。

    用法::

        with Spinner("正在抓取 RSS 源..."):
            articles = fetch_all(urls)

    在后台线程中以 ~10fps 刷新旋转字符；退出 ``with`` 块时自动清除该行。
    非 TTY 环境下为空操作，不会启动线程。

    注意：Spinner 假设阻塞期间没有其它代码向 stdout 写入，否则进度行会被
    中间输出打断。对于会产生自身输出的操作（如 Council 多阶段），应改用
    :class:`PhaseIndicator`。
    """

    def __init__(
        self,
        text: str,
        interval: float = 0.1,
        style: str = "braille",
        success_text: str | None = None,
        failure_text: str | None = None,
        show_elapsed: bool = True,
    ) -> None:
        self._text = text
        self._interval = interval
        self._frames = self._resolve_frames(style)
        self._success_text = success_text
        self._failure_text = failure_text
        self._show_elapsed = show_elapsed
        self._started_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active = _is_tty()

    @staticmethod
    def _resolve_frames(style: str) -> str:
        frames = _SPINNER_PRESETS.get(style)
        if not frames:
            raise ValueError(f"未知 Spinner 样式: {style}")
        return frames

    def _elapsed_text(self) -> str:
        if not self._show_elapsed or self._started_at is None:
            return ""
        elapsed = max(0.0, time.perf_counter() - self._started_at)
        return f" {DIM}· {elapsed:.1f}s{RESET}"

    def _print_final(self, ok: bool, message: str) -> None:
        symbol = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        sys.stdout.write(f"  {symbol} {message}{self._elapsed_text()}\n")
        sys.stdout.flush()

    def _spin(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = self._frames[i % len(self._frames)]
            sys.stdout.write(f"\r\x1b[2K {GREEN}{frame}{RESET} {self._text}")
            sys.stdout.flush()
            i += 1
            time.sleep(self._interval)

    def start(self) -> None:
        if not self._active:
            return
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, clear: bool = True, final_text: str | None = None, ok: bool = True) -> None:
        if not self._active:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if clear:
            _clear_line()
        if final_text:
            self._print_final(ok=ok, message=final_text)
        self._stop.clear()

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self.stop(clear=True, final_text=self._success_text, ok=True)
            return None
        self.stop(clear=True, final_text=self._failure_text, ok=False)
        return None


def make_phase_callback(indicator: PhaseIndicator | None) -> Callable[[str], None] | None:
    """把 PhaseIndicator 包装成 run_council 接受的 on_phase 回调。

    传 None 时返回 None，方便调用方按需启用::

        on_phase = make_phase_callback(indicator)
        run_council(..., on_phase=on_phase)
    """
    if indicator is None:
        return None
    return indicator.advance
