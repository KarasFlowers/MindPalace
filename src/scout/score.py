"""启发性评分模块。调用 LLM 对文章进行多维度评估。"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from src.scout.normalize import NormalizedArticle
from src.llm.client import chat_json
from src.config import get_scout_config, MAX_WORKERS

logger = logging.getLogger(__name__)

SCORING_SYSTEM_PROMPT = """\
你是 MindPalace 的"启发性评分引擎"。你的任务是评估一篇文章是否值得深度思考。

请从以下三个维度打分（每项 1-10 分）：

1. **信息密度 (information_density)**
   单位篇幅中新概念、新逻辑、新数据的密集程度。
   - 1-3: 大量废话、情绪化表达、重复已知常识
   - 4-6: 有一定信息量，但混杂较多冗余
   - 7-10: 几乎每段都有实质性新信息或新视角

2. **原理深度 (principle_depth)**
   是否从第一性原理解释机制，而非仅描述表象。
   - 1-3: 纯粹描述现象或转述他人观点
   - 4-6: 有一定分析，但停留在中间层
   - 7-10: 深入到底层机制、因果关系或数学模型

3. **因果链长度 (causal_chain)**
   逻辑推演的层级深度，思维链有多长。
   - 1-3: 只有单步结论，无推理过程
   - 4-6: 有 2-3 步推理，但跳跃较多
   - 7-10: 完整的多步推演，每步都有依据

请同时输出：
- **summary**: 一句话概括文章核心观点（中文，不超过100字）
- **reasoning**: 评分理由（中文，2-3 句话，解释为什么值得或不值得深度阅读）

以 JSON 格式输出：
{
  "information_density": <int>,
  "principle_depth": <int>,
  "causal_chain": <int>,
  "summary": "<string>",
  "reasoning": "<string>"
}
"""


@dataclass
class ScoredArticle:
    """带评分的文章。"""

    url: str
    title: str
    source: str
    published_at: str
    clean_content: str
    summary: str
    scores: dict[str, int]  # {"information_density": x, "principle_depth": y, "causal_chain": z}
    total_score: float
    reasoning: str


def _compute_total(scores: dict[str, int]) -> float:
    """加权平均分。原理深度权重略高。"""
    weights = {
        "information_density": 0.3,
        "principle_depth": 0.4,
        "causal_chain": 0.3,
    }
    total = sum(scores.get(k, 0) * w for k, w in weights.items())
    return round(total, 2)


def score_article(
    article: NormalizedArticle, 
    provider_config: dict | None = None
) -> ScoredArticle:
    """调用 LLM 对单篇文章进行启发性评分。"""
    user_prompt = f"标题: {article.title}\n来源: {article.source}\n\n正文:\n{article.clean_content}"

    cfg = provider_config or get_scout_config()
    result = chat_json(SCORING_SYSTEM_PROMPT, user_prompt, provider_config=cfg)

    scores = {
        "information_density": int(result.get("information_density", 1)),
        "principle_depth": int(result.get("principle_depth", 1)),
        "causal_chain": int(result.get("causal_chain", 1)),
    }

    return ScoredArticle(
        url=article.url,
        title=article.title,
        source=article.source,
        published_at=article.published_at,
        clean_content=article.clean_content,
        summary=result.get("summary", ""),
        scores=scores,
        total_score=_compute_total(scores),
        reasoning=result.get("reasoning", ""),
    )


def score_all(
    articles: list[NormalizedArticle], 
    provider_config: dict | None = None
) -> list[ScoredArticle]:
    """批量评分并按总分降序排列。使用线程池并行评分。"""
    scored = []
    cfg = provider_config or get_scout_config()
    total = len(articles)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_article = {
            executor.submit(score_article, article, cfg): article
            for article in articles
        }
        for i, future in enumerate(as_completed(future_to_article)):
            article = future_to_article[future]
            logger.info("Scoring result [%d/%d]: %s", i + 1, total, article.title[:60])
            try:
                scored.append(future.result())
            except Exception:
                logger.exception("Failed to score: %s", article.url)

    scored.sort(key=lambda a: a.total_score, reverse=True)
    return scored
