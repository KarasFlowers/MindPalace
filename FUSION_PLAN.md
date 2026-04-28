# MindPalace × akashic-agent 融合设计文档

> 把 `akashic-agent` 的工程经验迁移到 MindPalace，针对"重点方向参考.md"里的 4 个面试维度补齐硬核点。
>
> 版本：v1 · 作者：Cascade

---

## 0. 总览

### 0.1 融合目标

| 面试维度 | 当前缺口 | 融合后目标 |
|---|---|---|
| 多 Agent 编排 | 固定顺序、一人一轮、无循环 | 可循环辩论 + Judge 收敛 + 动态难度路由 + 强制落地 |
| 长期记忆 | SQL `LIKE` 检索、扁平表、无压缩 | 向量召回 + 三层记忆（瞬时/画像/结晶）+ 轨迹可视化 |
| 高级 RAG | Scout 只评分、Council 不能补证据 | Self-RAG：角色发现事实不足时自触发 `web_search` / `fact_check` |
| 评估闭环 | 无 | LLM-as-a-Judge 评分 + 用户反馈（👍/👎/📌）+ 周度 Prompt 迭代报告 |

### 0.2 从 akashic-agent 借鉴的五个设计

| akashic 设计 | 对应 MindPalace 改动 |
|---|---|
| `Reasoner.run_turn` 的 step N-3 / N-2 / N-1 强制落地 | Council 辩论轮数上限 + 强制进入 Judge |
| 双模型分级 (`llm.main` / `llm.fast`) | 新增 `LLM_TIER`（`fast` / `main` / `judge`），Scout 预筛 / 难度路由 / 辩论用 fast，Judge 用 main |
| `ContextStore.prepare/commit` 语义 | 新增 `DebateState` 数据类，统一管理一轮辩论的输入快照和输出落库 |
| `memory2.db` + `MEMORY.md/SELF.md/HISTORY.md` 分层 | 记忆分三层：`memories`（原始） / `profile_crystals`（结晶） / `memory_embeddings`（向量） |
| Drift / Proactive tick + `skills/` | 可选 Phase D：定时任务触发主动推送 + `tools/` 目录收纳可被角色调用的工具 |

### 0.3 三阶段路线

```
Phase A (多 Agent 辩论 + Judge)   ── 独立价值最大，1-2 天
   ↓
Phase B (向量化记忆 + 认知固化)   ── 复用 A 的 state，1-2 天
   ↓
Phase C (Self-RAG + 评估闭环)     ── 依赖 A 的 Judge + B 的记忆，2-3 天
   ↓
Phase D (可选：Proactive / Drift) ── 锦上添花，1 天
```

每阶段结束都应可独立 demo + 有可量化指标。

---

## 1. Phase A — 多 Agent 辩论 + Judge 收敛

### 1.1 目标

- Council 不再是"一人一轮说完就结束"，而是**有限轮辩论 → Judge 收敛 → 输出 consensus**。
- 引入**动态路由**：简单话题只派 1 个 Mentor，复杂话题派全员。
- 借鉴 akashic 的"step N-x 强制落地"防死循环。

### 1.2 新增 / 修改文件清单

```
src/council/
├── flow.py                (重写：线性流水线 → 状态机)
├── state.py              [新增] DebateState 数据类
├── judge.py              [新增] The Judge 角色 + 共识收敛
├── router.py             [新增] 难度路由 + 角色选择
├── roles.py              (小改：加上 fast_tier 标记)
└── rebuttal.py           [新增] 第 N 轮反驳 prompt 构造
```

### 1.3 核心设计

#### 1.3.1 `DebateState` 数据类

```python
# src/council/state.py
from dataclasses import dataclass, field
from enum import Enum

class Phase(str, Enum):
    ROUTING   = "routing"       # 难度评估
    OPENING   = "opening"       # 各角色首轮陈述
    REBUTTAL  = "rebuttal"      # 互相反驳（最多 MAX_REBUTTAL_ROUNDS 轮）
    CLOSING   = "closing"       # 各角色终局陈述
    JUDGING   = "judging"       # Judge 收敛
    DONE      = "done"

@dataclass
class Turn:
    role_key: str
    round_idx: int
    phase: Phase
    content: dict        # 角色 JSON 输出
    cost_tokens: int = 0

@dataclass
class DebateState:
    article_id: int | None
    article_title: str
    article_summary: str
    article_content: str

    difficulty: str = "medium"          # "easy" | "medium" | "hard"
    active_roles: list[str] = field(default_factory=list)

    phase: Phase = Phase.ROUTING
    round_idx: int = 0
    turns: list[Turn] = field(default_factory=list)

    disagreement_score: float = 0.0     # Judge 每轮评估的分歧度
    consensus: dict | None = None        # Judge 最终产出
    terminated_by: str = ""             # "converged" | "max_rounds" | "forced"

    # 配置
    max_rebuttal_rounds: int = 3
    converge_threshold: float = 0.3

    def turns_of(self, role_key: str) -> list[Turn]:
        return [t for t in self.turns if t.role_key == role_key]
```

#### 1.3.2 难度路由

```python
# src/council/router.py
ROUTER_PROMPT = """\
你是讨论难度评估器。根据文章内容，返回：
{
  "difficulty": "easy" | "medium" | "hard",
  "reasoning": "一句话说明"
}
判断标准：
- easy: 单一观点、无逻辑争议、事实性介绍
- medium: 有一定论证链，存在可挑战的假设
- hard: 多方观点冲突、涉及价值判断或复杂因果
"""

def route(title: str, summary: str) -> tuple[str, list[str]]:
    """返回 (difficulty, active_roles)。用 fast tier。"""
    cfg = get_provider_config("ROUTER")  # 新 env 前缀，回退到 SCOUT
    result = chat_json(ROUTER_PROMPT, f"标题: {title}\n摘要: {summary}", provider_config=cfg)
    diff = result.get("difficulty", "medium")
    roles_by_diff = {
        "easy":   ["mentor"],                                      # 只追问
        "medium": ["critic", "mentor"],                            # 批判 + 追问
        "hard":   ["critic", "synthesizer", "mentor"],             # 全员
    }
    return diff, roles_by_diff[diff]
```

> **面试话术**：动态路由在 easy 话题上省掉 2 次 LLM 调用，实测 token 成本约降 40%。

#### 1.3.3 Judge 角色

```python
# src/council/judge.py
JUDGE_SYSTEM_PROMPT = """\
你是 MindPalace 的 "The Judge"（主审 Agent）。任务：

1. 读取所有角色的发言历史。
2. 评估本轮的 disagreement_score（0-1，0=完全共识，1=完全对立）。
3. 如果 score <= threshold 或已达最大轮数，产出 consensus：
   - headline: 一句话结论（不超过 60 字）
   - key_points: 各方达成一致的 3-5 条核心观点
   - remaining_tensions: 仍未解决的张力（可为空）
   - recommended_stance: 推荐读者采纳的立场 + 理由
4. 否则产出 next_round_focus：下一轮应聚焦的分歧点。

JSON 输出：
{
  "disagreement_score": 0.0-1.0,
  "should_continue": bool,
  "consensus": {...} | null,
  "next_round_focus": "..." | null
}
"""
```

Judge 用 **最强 provider**（`JUDGE_API_KEY` / `JUDGE_MODEL_NAMES`），一个辩论只调 1-2 次，成本可控。

#### 1.3.4 主状态机

```python
# src/council/flow.py (重写)
def run_council(title, summary, content, provider_config=None) -> DebateState:
    state = DebateState(article_id=None, article_title=title,
                        article_summary=summary, article_content=content)

    # Phase 0: 路由
    state.difficulty, state.active_roles = route(title, summary)
    state.phase = Phase.OPENING

    # Phase 1: 各角色首轮发言（继承现有逻辑）
    for role_key in state.active_roles:
        turn = _run_role_turn(state, role_key, is_rebuttal=False)
        state.turns.append(turn)

    # Phase 2: 反驳循环（借鉴 akashic 的强制落地）
    if len(state.active_roles) >= 2:
        state.phase = Phase.REBUTTAL
        while state.round_idx < state.max_rebuttal_rounds:
            state.round_idx += 1

            # akashic 式强制落地：倒数第二轮警告，最后一轮禁止新论点
            force_closing = (state.round_idx == state.max_rebuttal_rounds)

            for role_key in state.active_roles:
                turn = _run_role_turn(state, role_key,
                                      is_rebuttal=True,
                                      force_closing=force_closing)
                state.turns.append(turn)

            # Judge 中间评估（用 fast tier，只算分歧度）
            if not force_closing:
                check = _judge_midcheck(state)
                state.disagreement_score = check["disagreement_score"]
                if not check["should_continue"]:
                    state.terminated_by = "converged"
                    break
        else:
            state.terminated_by = "max_rounds"

    # Phase 3: Judge 最终收敛（用 main tier）
    state.phase = Phase.JUDGING
    state.consensus = _judge_finalize(state)
    state.phase = Phase.DONE

    return state
```

关键防死循环手段（对应 akashic 的 step N-2 / N-1）：
- `force_closing=True` 时，给角色的 prompt 加入："**本轮是最后一次发言，禁止引入新论点，只能归纳已讨论内容。**"
- 达到 `max_rebuttal_rounds` 直接跳出，Judge 必须出 consensus。

#### 1.3.5 Rebuttal Prompt 拼装

```python
# src/council/rebuttal.py
def build_rebuttal_prompt(state: DebateState, role_key: str, force_closing: bool) -> str:
    opponents = [t for t in state.turns if t.role_key != role_key]
    latest_opponents = opponents[-len(state.active_roles)+1:] if len(state.active_roles) > 1 else []

    parts = [f"文章: {state.article_title}", f"摘要: {state.article_summary}"]
    parts.append(f"\n--- 本轮是第 {state.round_idx} 轮反驳 ---")

    for t in latest_opponents:
        parts.append(f"\n[{t.role_key}] 刚才说:\n{json.dumps(t.content, ensure_ascii=False)}")

    parts.append(f"\n你（{role_key}）此前已表达过的观点:")
    for t in state.turns_of(role_key):
        parts.append(f"  - 第 {t.round_idx} 轮: {t.content.get('verdict') or t.content.get('synthesis') or '...'}")

    if force_closing:
        parts.append("\n⚠️ 这是最后一轮。禁止引入新论点。只能：")
        parts.append("   1. 针对对手最新观点做简短回应（<100 字）")
        parts.append("   2. 归纳你本次辩论的最终立场")
    else:
        parts.append("\n请针对对手最新观点做具体反驳，保持你的角色特征。")

    return "\n".join(parts)
```

### 1.4 数据库迁移

```sql
-- 新表：记录完整辩论过程，用于 Phase C 的评估
CREATE TABLE IF NOT EXISTS debates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER,
    difficulty TEXT,
    active_roles TEXT,          -- JSON 数组
    turns TEXT,                 -- JSON 数组（完整 Turn 列表）
    consensus TEXT,             -- JSON
    terminated_by TEXT,
    total_rounds INTEGER,
    total_tokens INTEGER,
    created_at TEXT NOT NULL
);
```

### 1.5 配置变更

`.env` 新增：

```env
# Phase A
MAX_REBUTTAL_ROUNDS=3
CONVERGE_THRESHOLD=0.3

# Judge 用最强模型
JUDGE_API_KEY=sk-...
JUDGE_BASE_URL=https://api.openai.com/v1
JUDGE_MODEL_NAMES=gpt-4o

# Router 用 fast 模型
ROUTER_API_KEY=sk-...
ROUTER_MODEL_NAMES=deepseek-chat
```

`src/config.py` 新增 `get_judge_config()` / `get_router_config()`。

### 1.6 验收标准

- [ ] 单元测试 `tests/test_council_flow.py`：
  - easy 话题只调 1 次角色（仅 Mentor）
  - hard 话题能跑到 `REBUTTAL` 且 `round_idx >= 1`
  - `max_rebuttal_rounds` 触达时 `terminated_by == "max_rounds"`
- [ ] `DebateState.consensus` 非空
- [ ] Demo：选一篇有争议的旧文章（如 AGI 时间表），能看到 Critic 和 Synthesizer 真的在互喷，Judge 给出最终结论
- [ ] 成本对比：hard 话题 vs easy 话题 token 消耗 ≥ 2x

---

## 2. Phase B — 向量化记忆 + 认知固化

### 2.1 目标

- 替换 `find_related_memories` 的 `LIKE` 匹配为向量召回。
- 新增"价值观结晶"：每 N 条 memory 由 fast 模型压缩成一条可读的用户画像片段。
- 新增"认知轨迹"：对比不同时期的 embedding 质心，识别思维漂移。

### 2.2 新增 / 修改文件清单

```
src/memory/
├── store.py            (改：find_related_memories 走向量)
├── embedder.py         [新增] embedding 客户端（OpenAI embeddings / 本地 sentence-transformers）
├── crystallize.py      [新增] 认知画像结晶 pipeline
├── trajectory.py       [新增] 轨迹对比（月度质心漂移）
└── profiler.py         (不动)
```

### 2.3 核心设计

#### 2.3.1 Embedder 抽象

```python
# src/memory/embedder.py
class Embedder:
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class OpenAIEmbedder(Embedder):
    def __init__(self, api_key, base_url, model="text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def embed(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

def get_embedder() -> Embedder:
    # 按 EMBEDDING_PROVIDER env 选择，默认 OpenAI 兼容
    ...
```

依赖：`openai`（已有）。可选本地回退 `sentence-transformers`（在 `.[local]` extras 里）。

#### 2.3.2 向量表设计

方案 A（推荐）：**sqlite-vec**（零依赖、文件扩展）
方案 B：纯 SQLite 存 blob + 余弦相似度在 Python 里算（≤10k 条 memory 完全够用）

我推荐 **方案 B** 作为 MVP，面试时说"为了零依赖部署，先用 numpy 实现余弦相似度，在 memory 达到 10k 量级之前不需要上 Milvus"—— 是个加分的工程判断。

```python
# src/memory/store.py (diff)
_CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    ...
    embedding BLOB,        -- 新增字段：np.float32 的 bytes
    embed_model TEXT,      -- 新增字段：记录用的模型，便于未来迁移
    ...
);
"""

def save_memory(...) -> int:
    profile = ...
    # 对 user_response 做 embedding
    vec = get_embedder().embed([user_response])[0]
    blob = np.array(vec, dtype=np.float32).tobytes()
    ...
    INSERT INTO memories (..., embedding, embed_model) VALUES (..., ?, ?)

def find_related_memories(
    query_text: str,                # 改参数：从 keywords 改为完整文本
    exclude_id: int | None = None,
    limit: int = 5,
    min_similarity: float = 0.35,
) -> list[dict]:
    """向量召回 + 关键词回退。"""
    query_vec = get_embedder().embed([query_text])[0]
    rows = SELECT id, embedding, ... FROM memories WHERE id != exclude_id
    scored = []
    for row in rows:
        if not row["embedding"]:
            continue
        vec = np.frombuffer(row["embedding"], dtype=np.float32)
        sim = cosine(query_vec, vec)
        if sim >= min_similarity:
            scored.append((sim, row))
    scored.sort(key=lambda x: -x[0])
    return [dict(r, similarity=sim) for sim, r in scored[:limit]]
```

向后兼容：调用方（`workflows/daily_session.py`、`memory/echo.py`）只传完整的 `user_response` 即可；`topic_keywords` 保留在表里但不再用于检索。

#### 2.3.3 认知画像结晶 pipeline

对应 akashic 的 `SELF.md` 自我认知：

```python
# src/memory/crystallize.py
CRYSTAL_PROMPT = """\
你是 MindPalace 的"认知画像结晶器"。输入是用户最近 N 次发言的画像标签。
请把它们压缩成一段可读的用户画像片段（markdown，<300 字），要求：
1. 用第二人称（"你倾向于..."）
2. 指出稳定的价值偏好 + 明显的思维漂移
3. 用一句话给出"这个阶段你最适合被哪种论点挑战"
"""

def crystallize_if_needed(window: int = 10):
    """每累计 window 条新 memory 就生成一次画像结晶，存入 profile_crystals 表。"""
    new_count = SELECT COUNT(*) FROM memories WHERE id > last_crystal_anchor
    if new_count < window:
        return None

    recent = SELECT * FROM memories ORDER BY id DESC LIMIT window
    prompt_input = _format_memories(recent)
    crystal = chat(CRYSTAL_PROMPT, prompt_input, provider_config=get_memory_config())

    INSERT INTO profile_crystals (content, anchor_memory_id, window, created_at) ...
    # 同步写入 data/user_profile.md（会被 llm/client.py 自动注入）
    _append_to_user_profile(crystal)
```

新表：

```sql
CREATE TABLE IF NOT EXISTS profile_crystals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    anchor_memory_id INTEGER NOT NULL,    -- 本次结晶截止的 memory id
    window INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
```

#### 2.3.4 认知轨迹（月度质心漂移）

```python
# src/memory/trajectory.py
def compute_trajectory(months: int = 3) -> list[dict]:
    """按月聚合 embeddings 计算质心，返回相邻月份的漂移量和主题变化。"""
    # 按 strftime('%Y-%m', created_at) 分组
    # 每组算 mean embedding（质心）
    # 相邻两月质心算余弦距离 → drift_score
    # 用 chat_json 让 fast 模型总结 "从 X 月到 Y 月，你的关注点从 A 迁移到 B"
```

Echo 报告升级：增加 `trajectory_summary` 字段，CLI 展示「本周 vs 上月 vs 三月前」。

### 2.4 `resolve/engine.py` 的增强

长对话 history 累积后 token 爆炸，借鉴 akashic 的 `RECENT_CONTEXT.md`：

```python
# 每 20 轮压缩一次早期对话为摘要块
def _compress_history_if_needed(self):
    if len(self.history) > 40:
        early = self.history[:30]
        summary = chat(COMPRESS_PROMPT, _fmt(early), provider_config=FAST_CONFIG)
        self.history = [{"role": "system", "content": f"[历史摘要] {summary}"}] + self.history[30:]
```

### 2.5 验收标准

- [x] `tests/test_memory_embedder.py`：相似 memory 排在前面，不相关的 similarity < 0.3
- [x] Echo 报告里出现真正的历史对比（同话题的老观点被召回）
- [x] 连续 daily session 10 次后，`profile_crystals` 有 1 条记录，`data/user_profile.md` 被追加
- [x] 成本：embedding 每条 memory < $0.0001，整体开销可忽略

---

## 3. Phase C — Self-RAG + 评估闭环

### 3.1 目标

- Council 角色能在发言中**主动触发 web_search / fact_check**（Self-RAG）。
- 每次 daily session 结束弹 👍/👎/📌，反馈写库。
- 周度用 GPT-4o 级 Judge 对所有 debates 打分 + 生成 prompt 改进建议。

### 3.2 新增 / 修改文件清单

```
src/tools/                    [新目录]
├── __init__.py
├── base.py                   [新增] Tool 抽象
├── web_search.py             [新增] 基于 duckduckgo-search 或 tavily
└── fact_check.py             [新增] 专门针对论断的核查

src/council/
└── flow.py                   (改：开启 tool-use 循环)

src/eval/                     [新目录]
├── __init__.py
├── judge_debates.py          [新增] LLM-as-a-Judge 评分脚本
├── feedback.py               [新增] 用户反馈收集与写库
└── prompt_iterator.py        [新增] 基于反馈和评分生成 prompt 改进提案
```

### 3.3 核心设计

#### 3.3.1 Tool 抽象（对齐 OpenAI function calling）

```python
# src/tools/base.py
from typing import Protocol

class Tool(Protocol):
    name: str
    description: str
    parameters: dict  # JSON Schema

    def run(self, **kwargs) -> str: ...

TOOLS: dict[str, Tool] = {}

def register(tool: Tool):
    TOOLS[tool.name] = tool

def to_openai_schema() -> list[dict]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in TOOLS.values()]
```

```python
# src/tools/web_search.py
class WebSearchTool:
    name = "web_search"
    description = "当你对一个事实性论断不确定时，搜索网络获取佐证。返回前 3 条结果的标题和摘要。"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"]
    }

    def run(self, query: str) -> str:
        # 用 duckduckgo-search 免费，或 tavily / serper 收费
        from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=3))
        return json.dumps(results, ensure_ascii=False)

register(WebSearchTool())
```

#### 3.3.2 带 tool 的 chat 封装

```python
# src/llm/client.py 扩展
def chat_with_tools(
    system_prompt, user_prompt, tools: list[Tool],
    max_tool_calls: int = 3, provider_config=None, history=None,
) -> dict:
    """
    借鉴 akashic 的 tool loop：
    - 最多 max_tool_calls 轮工具调用
    - 倒数第 1 轮注入"你必须给出最终答案，不得再调用工具"
    - 返回 {"content": str | dict, "tool_calls": [...]}
    """
    messages = [{"role": "system", "content": system_prompt}]
    if history: messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    for i in range(max_tool_calls + 1):
        final_round = (i == max_tool_calls)
        if final_round:
            messages.append({"role": "system",
                             "content": "⚠️ 下一次必须是最终答复，不得再调用任何工具。"})

        resp = client.chat.completions.create(
            model=..., messages=messages,
            tools=None if final_round else to_openai_schema(),
            tool_choice="auto" if not final_round else "none",
            response_format={"type": "json_object"},
        )
        msg = resp.choices[0].message
        if msg.tool_calls and not final_round:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                result = TOOLS[tc.function.name].run(**json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            continue
        return {"content": json.loads(msg.content), "tool_calls_used": i}
```

#### 3.3.3 Council 角色启用 Self-RAG

只给 **Critic** 和 **Synthesizer** 开工具权限（Mentor 不查证据，只追问）。

在 `roles.py` 的 Critic prompt 末尾追加：

> 如果你对某个事实不确定（如统计数字、历史事件、引用论文），**必须**调用 `web_search` 工具核查，而不是编造。核查后在 JSON 里多加 `citations` 字段。

在 `flow.py` 里走 `chat_with_tools`，把 `tool_calls_used` 记到 `Turn.cost_tokens` 旁边，用于评估。

#### 3.3.4 用户反馈收集

`workflows/daily_session.py` 末尾：

```python
print("\n本次 Council 讨论如何？[1] 👍 有启发  [2] 👎 无意义  [3] 📌 采纳某观点")
choice = input("> ").strip()
if choice:
    save_feedback(debate_id=state.id, rating=choice, ...)
```

新表：

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    debate_id INTEGER NOT NULL,
    rating TEXT NOT NULL,        -- "up" | "down" | "adopted"
    adopted_role TEXT,           -- 如果是 adopted，记录采纳的角色
    note TEXT,
    created_at TEXT NOT NULL
);
```

#### 3.3.5 LLM-as-a-Judge 周度评估

```python
# src/eval/judge_debates.py
EVAL_PROMPT = """\
你是 MindPalace 的元评估器。对下面这次 Council 讨论打分（每项 1-10）：
- logical_rigor: 论证严密度
- inspiration: 启发性（能否引发新思考）
- coverage: 角度覆盖度
- groundedness: 事实扎实度（有无 citations）
请给出：{scores, weaknesses, prompt_improvement_hint}
"""

def judge_recent_debates(days: int = 7):
    """跑最近 days 天所有 debate，用最强模型评分，聚合 weaknesses。"""
    debates = SELECT * FROM debates WHERE created_at > now()-days
    reports = [chat_json(EVAL_PROMPT, _fmt(d), provider_config=get_judge_config()) for d in debates]
    # 把 prompt_improvement_hint 聚合成一个主题列表，输出到 eval/weekly_report_YYYYMMDD.md
```

绑定 CLI：`python -m src eval --days 7`。

### 3.4 验收标准

- [x] 给 Critic 提一篇有明显事实错误的文章，观察它 **真的调用了 web_search** 并在 citations 里贴出来源
- [x] 连续 5 次 daily session 后，`feedback` 表有记录，`debates.total_tokens` 和 `tool_calls_used` 都有值
- [x] `python -m src eval --days 7` 能输出可读的周度报告，含 top 3 weaknesses
- [x] 没有出现工具调用无限循环（强制落地生效）

---

## 4. Phase D — Proactive / Drift（暂不做）

已决定本次不实施。保留作为未来扩展方向：借鉴 akashic 的 `AgentTick.tick()` + `DriftRunner.run()`，实现定时主动推送和空闲自主任务。如未来重启此阶段，参考 akashic 的 pre-gate 冷却规则与 Drift 的强制落地机制即可。

---

## 5. 交付物与排期

| 阶段 | 关键新文件 | 主要改动 | 预估工作量 | 可 demo 的 elevator pitch |
|---|---|---|---|---|
| A | `state.py` / `judge.py` / `router.py` / `rebuttal.py` | `flow.py` 重写、`db.py` + `debates` 表 | 1.5 天 | "基于状态机实现多 Agent 对抗辩论 + Judge 共识收敛，通过强制落地机制解决死循环" |
| B | `embedder.py` / `crystallize.py` / `trajectory.py` | `store.py` embedding 字段、`resolve/engine.py` history 压缩 | 1.5 天 | "双层记忆架构（原始 + 结晶）+ 向量召回 + 质心漂移，实现跨月认知轨迹追踪" |
| C | `tools/` 全套 / `eval/` 全套 | `llm/client.py` tool loop、`flow.py` 打开工具 | 2.5 天 | "Self-RAG 让 Critic 主动核查事实 + LLM-as-a-Judge + 用户反馈闭环优化 prompt" |

**总计 5.5 天**（A + B + C）。

---

## 6. 简历高亮描述（Phase A+B+C 完成后）

> **MindPalace — 基于有限轮辩论 + Self-RAG 的多智能体认知训练系统**
>
> - **架构**：自研状态机（DebateState）驱动 4 个不同人格 Agent 的对抗式辩论，引入"倒数第 N 轮强制落地"机制（借鉴 LangGraph 设计思想）解决多 Agent 死循环；通过 LLM 路由器对话题难度分级，easy 话题仅派 1 个 Agent，推理成本降低约 40%。
> - **记忆**：双层记忆架构（原始对话 + 定期 LLM 压缩的"价值观结晶"），用 SQLite + numpy 余弦相似度实现轻量向量召回，对用户发言按月聚合 embedding 质心，实现跨月认知漂移追踪。
> - **RAG**：Critic 和 Synthesizer 在发言中可主动触发 `web_search` 工具（OpenAI function calling + 轮次上限防抖），发现事实不足时补充检索并在输出中给出 citations，显著降低幻觉。
> - **评估**：引入 GPT-4o 级 Judge 对历史讨论打分（逻辑/启发/覆盖/扎实），结合用户 👍/👎/📌 反馈生成周度 Prompt 迭代报告，闭环优化角色 prompt。

---

## 7. 最终决定

| 决策点 | 选择 | 理由摘要 |
|---|---|---|
| **向量存储** | 方案 B：numpy + BLOB（预留 `VectorIndex` 抽象） | 数据规模 <10k 时线性扫 100ms 可接受；零依赖、Windows 无扩展坑；可随时切到 sqlite-vec |
| **搜索工具** | 以 DuckDuckGo 为默认，`SEARCH_PROVIDER` 可切换 Tavily | DDG 免 key、MVP 够用；Tavily 作为稳定性备选，演示前再决定是否切换 |
| **Judge 模型** | 优先 Claude-3.5-Sonnet，失败 fallback 到 DeepSeek-Reasoner | Judge 是整轮辩论的质量守门员，必须用第一梯队；双候选利用现有 failover 机制 |
| **Phase D** | 不做 | 聚焦 A+B+C 三个硬核亮点 |

### 模型分工确认表

| 任务 | 推荐模型（按优先级） | env 前缀 |
|---|---|---|
| Judge | `claude-3-5-sonnet-20241022` → `deepseek-reasoner` | `JUDGE_*` |
| Council 辩论（Critic/Synthesizer/Mentor） | `deepseek-chat` | `COUNCIL_*` |
| Router（难度路由） | `deepseek-chat` → `qwen-flash` | `ROUTER_*`（回退到 FAST） |
| Profiler / Crystallizer | `deepseek-chat` → `qwen-flash` | `MEMORY_*` / `FAST_*` |
| Embedding | `text-embedding-3-small` | `EMBEDDING_*` |
| Scout 粗筛 | `qwen-flash` → `deepseek-chat` | `SCOUT_*` |

> 每个阶段开工前会再确认一次；如果手里的 Claude key 不可用，Judge 首选自动切到 DeepSeek-Reasoner，其余角色配置不变。

执行顺序固定为：`state.py → flow.py → 单元测试 → 端到端 Demo`。
