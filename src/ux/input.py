"""统一的多行输入收集器。

项目中 council / daily / inquiry 多处需要收集用户的多行自由输入，原先各自
实现了几乎相同的 while 循环（连续两次空行提交、skip 跳过），但提示措辞、
提示符风格、skip 行为细节不一致。本模块统一这些规则。

统一规则：
- 提示符：统一的 ``>`` （绿色），不带 ``You`` 前缀
- 提交：连续两次空行（Enter）
- 跳过：输入 ``skip``
- 中断：Ctrl+C / EOF 返回 ``None``

调用方负责在调用前打印自己的上下文标题（如 ``[Your Turn]``）和可选的起手句提示。
"""

from __future__ import annotations

GREEN = "\033[32m"
RESET = "\033[0m"

DEFAULT_PROMPT = f"  {GREEN}>{RESET} "


def collect_multiline(
    prompt: str = DEFAULT_PROMPT,
    allow_skip: bool = True,
    hint: str | None = None,
) -> str | None:
    """收集多行用户输入。

    Args:
        prompt: 每行输入前显示的提示符，默认为绿色的 ``>``。
        allow_skip: 是否允许输入 ``skip`` 跳过（返回 ``None``）。
        hint: 若提供，会在开始前打印一行操作提示（如"连续两次空行提交"）。

    Returns:
        用户输入的文本（已 strip）；用户跳过或中断时返回 ``None``。

    提交规则：
    - 连续两次空行（即两个空字符串）触发提交。
    - 首行输入 ``skip`` 跳过（当 ``allow_skip=True``）。
    - Ctrl+C / EOF（如管道关闭）安全返回 ``None``。
    """
    if hint:
        print(hint)

    lines: list[str] = []
    blank_count = 0
    first_input = True

    while True:
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if first_input:
            if allow_skip and line.strip().lower() == "skip":
                return None
            first_input = False

        if line == "":
            blank_count += 1
            # 连续两次空行：有内容则提交，无内容则视为放弃返回 None
            if blank_count >= 2:
                return "\n".join(lines).strip() or None
            continue

        blank_count = 0
        lines.append(line)

    return "\n".join(lines).strip() or None
