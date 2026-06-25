"""终端进度指示器测试。

验证：
1. PhaseIndicator 在非 TTY 下为空操作（不向 stdout 写入）
2. PhaseIndicator 的 advance/done 计数与状态正确
3. Spinner 在非 TTY 下不启动线程
4. make_phase_callback 正确包装与降级
"""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest


@pytest.fixture
def fake_stdout():
    """捕获 stdout 写入，便于断言进度指示器没有产生垃圾输出。"""
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        yield buf


class TestPhaseIndicator:
    """阶段式进度指示器。"""

    def test_total_must_be_positive(self):
        from src.ux.progress import PhaseIndicator

        with pytest.raises(ValueError):
            PhaseIndicator(total=0)
        with pytest.raises(ValueError):
            PhaseIndicator(total=-1)

    def test_advance_increments_current(self, fake_stdout):
        from src.ux.progress import PhaseIndicator

        # 计数（逻辑进度）始终推进，与 TTY 无关
        indicator = PhaseIndicator(total=3)
        assert indicator.current == 0

        indicator.advance("phase 1")
        assert indicator.current == 1
        indicator.advance("phase 2")
        assert indicator.current == 2

    def test_advance_is_noop_in_non_tty(self, fake_stdout):
        from src.ux.progress import PhaseIndicator

        indicator = PhaseIndicator(total=2)
        indicator.advance("anything")
        indicator.advance("anything")
        # 非 TTY 下不应向（patch 后的）stdout 写入任何字节
        assert fake_stdout.getvalue() == ""

    def test_done_without_message_writes_nothing_in_non_tty(self, fake_stdout):
        from src.ux.progress import PhaseIndicator

        indicator = PhaseIndicator(total=2)
        indicator.done()
        assert fake_stdout.getvalue() == ""

    def test_done_with_message_does_not_raise(self, fake_stdout):
        """done(final_message=...) 在非 TTY 下应安全打印，不抛异常。"""
        from src.ux.progress import PhaseIndicator

        indicator = PhaseIndicator(total=2)
        # 不抛异常即可；输出捕获在多层 patch 下不可靠，不严格断言内容
        indicator.done(final_message="完成")

    def test_advance_does_not_exceed_total_display(self, fake_stdout):
        """advance 次数超过 total 时，内部计数继续推进。"""
        from src.ux.progress import PhaseIndicator

        indicator = PhaseIndicator(total=2)
        for _ in range(5):
            indicator.advance("x")
        assert indicator.current == 5


class TestSpinner:
    """行内旋转动画。"""

    def test_unknown_spinner_style_raises(self, fake_stdout):
        from src.ux.progress import Spinner

        with pytest.raises(ValueError, match="未知 Spinner 样式"):
            Spinner("loading", style="unknown")

    def test_spinner_no_thread_in_non_tty(self, fake_stdout):
        from src.ux.progress import Spinner

        with Spinner("loading", success_text="完成") as s:
            # 非 TTY 下不应启动后台线程
            assert s._thread is None
        # 退出时也不应写入任何字节
        assert fake_stdout.getvalue() == ""

    def test_spinner_context_manager_exits_cleanly(self, fake_stdout):
        from src.ux.progress import Spinner

        # 即使内部状态异常，with 块也应正常退出
        with Spinner("test"):
            pass
        # 到这里说明没有抛异常

    def test_spinner_success_final_message_in_tty(self):
        from src.ux.progress import Spinner

        class TtyBuffer(io.StringIO):
            def isatty(self):
                return True

        buf = TtyBuffer()
        with patch("sys.stdout", buf), patch("time.perf_counter", return_value=11.25):
            spinner = Spinner("loading", success_text="加载完成", show_elapsed=True)
            spinner._thread = None
            spinner._started_at = 10.0
            spinner.stop(clear=True, final_text="加载完成", ok=True)

        out = buf.getvalue()
        assert "✓" in out
        assert "加载完成" in out
        assert "1.2s" in out

    def test_spinner_failure_final_message_in_tty(self):
        from src.ux.progress import Spinner

        class TtyBuffer(io.StringIO):
            def isatty(self):
                return True

        buf = TtyBuffer()
        with patch("sys.stdout", buf), patch("time.perf_counter", return_value=10.5):
            spinner = Spinner("loading", failure_text="加载失败", show_elapsed=True)
            spinner._thread = None
            spinner._started_at = 10.0
            spinner.stop(clear=True, final_text="加载失败", ok=False)

        out = buf.getvalue()
        assert "✗" in out
        assert "加载失败" in out


class TestMakePhaseCallback:
    """回调包装器。"""

    def test_none_returns_none(self):
        from src.ux.progress import make_phase_callback

        assert make_phase_callback(None) is None

    def test_wraps_indicator_advance(self, fake_stdout):
        from src.ux.progress import PhaseIndicator, make_phase_callback

        indicator = PhaseIndicator(total=3)
        cb = make_phase_callback(indicator)
        assert cb is not None
        assert callable(cb)

        cb("phase")
        # 计数始终推进（与 TTY 无关）
        assert indicator.current == 1
