"""文章翻译模块。

将非中文文章译成中文，使 Council 讨论对中文用户不再有语言断裂。
翻译发生在 normalize 之后、score 之前；翻译结果写回 clean_content。

设计要点：
- 语言检测基于 CJK 字符占比（轻量、零依赖、离线）。
- 翻译通过现有的 chat_json 调用 SCOUT 档（性价比模型），与评分共用配置。
- 翻译失败时降级返回原文，不阻断流水线。
"""

import logging
import os
import re
from src.llm.client import chat_json
from src.config import get_scout_config, SCOUT_TRANSLATE
from src.scout.normalize import NormalizedArticle

logger = logging.getLogger(__name__)

# 判定为中文所需的 CJK 字符占比阈值（占总字符数）。
# 取 0.3：标题/正文里只要三成是汉字即视为中文源，容错混合内容。
_CJK_RATIO_THRESHOLD = 0.3

# CJK 统一表意符号 + 常见中文标点范围
_CJK_CHAR_RE = re.compile(
    r"[\u4e00-\u9fff"      # CJK 统一表意符号（常用汉字）
    r"\u3400-\u4dbf"       # CJK 扩展 A
    r"\uf900-\ufaff"       # CJK 兼容表意符号
    r"]"
)

TRANSLATE_SYSTEM_PROMPT = """\
你是 MindPalace 的翻译引擎。请把用户提供的文章正文翻译成流畅、准确的简体中文。

要求：
1. **忠实原文**：完整保留作者的观点、论证和例证，不得增删、改写或发表评论。
2. **保持结构**：保留段落划分与逻辑层次，不要合并或拆分段落。
3. **术语处理**：专有名词、人名、机构名首次出现时采用"中文（原文）"形式，之后用中文。
4. **只输出译文**：不要输出任何说明、注释、前后缀（如"以下是翻译："）。
5. **若原文已是中文**：直接原样返回，不要改写。

以 JSON 格式输出：{"translated": "<中文译文全文>"}
"""


def _translation_enabled() -> bool:
    raw = os.getenv("SCOUT_TRANSLATE")
    if raw is None:
        return SCOUT_TRANSLATE
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def detect_language(text: str) -> str:
    """检测文本主语言。

    Returns:
        "zh" 表示中文，"other" 表示非中文（含空文本）。
    """
    if not text:
        return "other"
    total = len(text)
    cjk_count = len(_CJK_CHAR_RE.findall(text))
    ratio = cjk_count / total if total else 0
    return "zh" if ratio >= _CJK_RATIO_THRESHOLD else "other"


def translate_content(
    text: str,
    title: str,
    provider_config: dict | None = None,
) -> str:
    """调用 LLM 将非中文文本译成中文。

    翻译失败（异常或返回为空）时降级返回原文，确保流水线不中断。
    """
    if not text:
        return text
    cfg = provider_config or get_scout_config()
    user_prompt = f"标题: {title}\n\n正文:\n{text}"
    try:
        result = chat_json(TRANSLATE_SYSTEM_PROMPT, user_prompt, provider_config=cfg)
        translated = (result.get("translated") or "").strip()
        if translated:
            return translated
        logger.warning("Translate returned empty, falling back to original text.")
        return text
    except Exception:
        logger.exception("Translation failed, falling back to original text.")
        return text


def maybe_translate_article(
    article: NormalizedArticle,
    provider_config: dict | None = None,
    enabled: bool | None = None,
) -> tuple[str, str, bool]:
    """按需翻译单篇文章。

    Args:
        article: 已标准化的文章。
        provider_config: LLM 配置，默认用 SCOUT 档。
        enabled: 是否启用翻译；None 时读取全局 SCOUT_TRANSLATE 开关。

    Returns:
        (source_lang, clean_content, translated)
        - source_lang: "zh" / "other"
        - clean_content: 进入后续流程的正文（必要时已翻译）
        - translated: 是否实际发生了翻译
    """
    translate_on = _translation_enabled() if enabled is None else enabled
    source_lang = detect_language(article.clean_content)

    if source_lang == "zh" or not translate_on:
        return source_lang, article.clean_content, False

    translated_text = translate_content(
        article.clean_content, article.title, provider_config
    )
    return source_lang, translated_text, translated_text != article.clean_content
