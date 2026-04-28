# MindPalace Agent 技术深度解析

> 你的私人认知进化实验室与多智能体议事厅

---

## 目录

1. [项目定位与设计哲学](#1-项目定位与设计哲学)
2. [系统架构总览](#2-系统架构总览)
3. [核心模块深度解析](#3-核心模块深度解析)
   - 3.1 [智能猎手 (The Scout)](#31-智能猎手-the-scout)
   - 3.2 [智库议事厅 (The Council)](#32-智库议事厅-the-council)
   - 3.3 [认知账本 (Memory)](#33-认知账本-memory)
   - 3.4 [Self-RAG 工具系统](#34-self-rag-工具系统)
   - 3.5 [评估闭环](#35-评估闭环)
4. [工程亮点与面试话术](#4-工程亮点与面试话术)
5. [技术选型与权衡决策](#5-技术选型与权衡决策)
6. [扩展性与未来规划](#6-扩展性与未来规划)

---

## 1. 项目定位与设计哲学

### 1.1 从"工具"到"教练"的范式转变

大多数 AI 项目是**搜索增强 (RAG)**——用户问什么，AI 就答什么，目标是"让用户更快找到答案"。

MindPalace 是**思考增强 (Thinking Augmented)**——它不顺从用户，而是**挑战用户**。通过苏格拉底式提问、多角度辩论、历史观点对比，逼迫用户触及自己思维的边界。

**核心价值主张**：
> 不是帮你"知道更多"，而是帮你"想得更深"。

### 1.2 主动式交互设计

区别于传统的"用户问 → AI 答"被动模式，MindPalace 由 Agent **主动发起**对话：

1. **Scout** 主动抓取高质量内容，而非等待用户输入关键词
2. **Council** 主动发起辩论，而非等待用户提问
3. **Echo** 主动对比历史观点，而非等待用户请求分析

这种设计涉及对**定时任务、异步处理、推送机制**的综合运用，展示了超越简单 Q&A 系统的工程能力。

### 1.3 闭环系统思维

MindPalace 不是一次性工具，而是一个**持续进化的闭环**：

```
信息抓取 → 多Agent辩论 → 用户观点记录 → 认知画像更新 → 历史对比 → 反馈收集 → Prompt优化
    ↑                                                                                    ↓
    └────────────────────────────────────────────────────────────────────────────────────┘
```

每次使用都会让系统"更懂你"，形成正向飞轮。

---

## 2. 系统架构总览

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MindPalace Agent                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Scout      │    │   Council    │    │   Memory     │                   │
│  │  (信息采编)   │───▶│  (多Agent辩论) │───▶│  (认知账本)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ 启发性评分器  │    │  Self-RAG    │    │  向量召回    │                   │
│  │ (三维度评估)  │    │  (工具调用)   │    │  (Embedding) │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             ▼                                               │
│                    ┌──────────────┐                                         │
│                    │  Eval Loop   │                                         │
│                    │  (评估闭环)   │                                         │
│                    └──────────────┘                                         │
│                             │                                               │
│                             ▼                                               │
│                    ┌──────────────┐                                         │
│                    │  LLM-as-Judge│                                         │
│                    │  (周度评分)   │                                         │
│                    └──────────────┘                                         │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  LLM Client  │    │   Storage    │    │    Obs       │                   │
│  │  (分层调度)   │    │   (SQLite)   │    │  (OTel追踪)  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
RSS/网页 ──▶ Scout抓取 ──▶ 启发性评分 ──▶ 入库存储
                                              │
                                              ▼
用户选择文章 ──▶ Router难度路由 ──▶ 角色分配 ──▶ Opening陈述
                                              │
                                              ▼
                    ┌─────────────────────────────────────────┐
                    │            Rebuttal 循环                 │
                    │  ┌─────────────────────────────────┐    │
                    │  │  Critic ←→ Synthesizer ←→ Mentor │    │
                    │  │         ↓                        │    │
                    │  │    Judge Midcheck (分歧度评估)    │    │
                    │  │         ↓                        │    │
                    │  │  [分歧度 < 阈值?] ──▶ 提前收敛    │    │
                    │  │  [达到最大轮数?] ──▶ 强制落地    │    │
                    │  └─────────────────────────────────┘    │
                    └─────────────────────────────────────────┘
                                              │
                                              ▼
                    Judge Finalize ──▶ 共识输出 ──▶ 用户回应
                                              │
                                              ▼
                    认知画像分析 ──▶ 向量存储 ──▶ Echo对比报告
                                              │
                                              ▼
                    结晶压缩 ──▶ 用户画像更新 ──▶ 注入后续LLM上下文
```

---

## 3. 核心模块深度解析

### 3.1 智能猎手 (The Scout)

#### 3.1.1 设计目标

区别于传统 RSS 订阅的"全量推送"，Scout 实现了**精细化信息采编**——只推送真正具有启发性的内容。

#### 3.1.2 启发性评分机制

Scout 从三个维度评估文章价值：

| 维度 | 含义 | 评估标准 |
|------|------|----------|
| **信息密度** | 单位篇幅中新概念、新逻辑、新数据的密集程度 | 是否有具体数据、案例、方法论？还是情绪化煽动？ |
| **原理深度** | 是否从第一性原理解释机制 | 是解释"为什么"，还是只描述"是什么"？ |
| **因果链长度** | 逻辑推演的层级深度 | 是简单归因，还是多层级因果分析？ |

**评分 Prompt 示例**：
```python
SCORING_PROMPT = """
你是 MindPalace 的内容评估专家。请从三个维度对文章打分（1-10）：

1. information_density: 新概念/数据/方法的密集程度
2. principle_depth: 是否从底层原理解释，而非表象描述
3. causal_chain: 逻辑推演的层级深度

输出 JSON: {"scores": {...}, "reasoning": "..."}
"""
```

#### 3.1.3 工程亮点

- **并行抓取**：使用 `concurrent.futures.ThreadPoolExecutor` 并行处理多个 RSS 源
- **内容清洗**：自动提取正文、去除广告和导航栏
- **去重机制**：基于 URL 和标题的幂等性保证

---

### 3.2 智库议事厅 (The Council)

Council 是整个项目的**核心亮点**，展示了复杂的多智能体编排能力。

#### 3.2.1 角色矩阵

| 角色 | 职责 | 行为特征 |
|------|------|----------|
| **The Critic** (理性批判者) | 寻找逻辑漏洞，挑战直觉思维 | 冷静、精确、不留情面，必须列出至少 3 个漏洞 |
| **The Synthesizer** (跨界连接者) | 将话题与其他学科进行关联 | 充满好奇心，必须找到至少 2 个跨学科类比 |
| **The Mentor** (苏格拉底导师) | 通过连续追问引导深度思考 | 绝不给答案，只提问，3 个递进式问题 |
| **The Judge** (主审 Agent) | 中期评估分歧度，最终收敛产出共识 | 中立、综合、给出可执行的结论 |

#### 3.2.2 状态机设计

Council 不是简单的顺序流水线，而是基于 `DebateState` 的**有限状态机**：

```python
class Phase(str, Enum):
    ROUTING   = "routing"       # 难度评估
    OPENING   = "opening"       # 各角色首轮陈述
    REBUTTAL  = "rebuttal"      # 互相反驳（最多 MAX_REBUTTAL_ROUNDS 轮）
    JUDGING   = "judging"       # Judge 收敛
    DONE      = "done"
```

**状态转移图**：

```
ROUTING ──(router.route)──▶ OPENING
                               │
                               ▼
                    ┌──────────────────────┐
                    │   OPENING (各角色)    │
                    └──────────────────────┘
                               │
                               ▼
         ┌─────────────────────────────────────────┐
         │         REBUTTAL 循环 (≤N轮)             │
         │                                          │
         │   for round in 1..MAX_REBUTTAL_ROUNDS:  │
         │       for role in active_roles:         │
         │           role.speak()                  │
         │                                          │
         │       if round == MAX:                  │
         │           force_closing = True          │
         │           break                         │
         │                                          │
         │       check = Judge.midcheck()          │
         │       if check.disagreement < threshold:│
         │           terminated_by = "converged"   │
         │           break                         │
         │                                          │
         └─────────────────────────────────────────┘
                               │
                               ▼
                         JUDGING
                               │
                               ▼
                      Judge.finalize()
                               │
                               ▼
                           DONE
```

#### 3.2.3 动态难度路由

**问题**：简单话题派 3 个 Agent 是浪费，复杂话题派 1 个 Agent 是敷衍。

**解决方案**：Router 用轻量模型评估话题难度，动态决定派几个角色：

```python
DIFFICULTY_ROLES = {
    "easy":   ["mentor"],                           # 只追问，省 2 次 LLM 调用
    "medium": ["critic", "mentor"],                  # 批判 + 追问
    "hard":   ["critic", "synthesizer", "mentor"],  # 全员辩论
}
```

**面试话术**：
> 动态路由在 easy 话题上省掉 2 次 LLM 调用，实测 token 成本约降 40%。

#### 3.2.4 强制落地机制（防死循环）

**问题**：多 Agent 辩论容易陷入无限循环——A 反驳 B，B 反驳 A，永无止境。

**解决方案**（借鉴 akashic-agent 设计）：

1. **硬性轮数上限**：`MAX_REBUTTAL_ROUNDS=3`，达到即停止
2. **倒数轮警告**：最后一轮注入系统级约束

```python
if force_closing:
    prompt += """
    ⚠️ 这是最后一轮。禁止引入新论点。只能：
       1. 针对对手最新观点做简短回应（<100 字）
       2. 归纳你本次辩论的最终立场
    """
```

3. **分歧度早停**：每轮结束后 Judge 评估分歧度，低于阈值提前收敛

```python
check = judge_midcheck(state)
if check["disagreement_score"] < CONVERGE_THRESHOLD:
    state.terminated_by = "converged"
    break
```

**面试话术**：
> 借鉴 LangGraph 的设计思想，通过"倒数第 N 轮强制落地"机制，彻底解决多 Agent 死循环问题。

#### 3.2.5 分层模型调度

不同任务对模型能力的要求不同，MindPalace 实现了**精细化成本控制**：

| 任务 | 模型要求 | 推荐配置 | 原因 |
|------|----------|----------|------|
| Router (难度路由) | 低 | DeepSeek-Chat / Qwen-Flash | 只需判断"简单/中等/复杂" |
| Midcheck (分歧度) | 低 | DeepSeek-Chat | 只需输出 0-1 的分数 |
| Critic/Synthesizer/Mentor | 中 | DeepSeek-Chat / GPT-4o-mini | 需要逻辑推理，但不需要顶级 |
| Judge (最终共识) | **高** | Claude-3.5-Sonnet / DeepSeek-Reasoner | 质量守门员，必须一流 |

**配置示例**：
```env
# Judge 用最强模型（辩论的质量守门员）
JUDGE_MODEL_NAMES=claude-3-5-sonnet-20241022,deepseek-reasoner

# Router 用便宜模型（只判断"要不要再来一轮"）
ROUTER_MODEL_NAMES=deepseek-chat
```

**面试话术**：
> Judge 是整场辩论的质量守门员，必须用第一梯队模型；而 Router/midcheck 只判断"要不要再来一轮"，用便宜小模型即可。这样整场辩论成本可控，同时最终结论质量不打折。

#### 3.2.6 完整状态落库

每次辩论的完整状态都会写入 `debates` 表：

```sql
CREATE TABLE debates (
    id INTEGER PRIMARY KEY,
    article_id INTEGER,
    difficulty TEXT,           -- "easy" | "medium" | "hard"
    active_roles TEXT,         -- JSON 数组
    turns TEXT,                -- JSON 数组（完整发言记录）
    consensus TEXT,            -- JSON（Judge 最终产出）
    terminated_by TEXT,        -- "converged" | "max_rounds" | "single_role"
    total_rounds INTEGER,
    total_tokens INTEGER,
    created_at TEXT
);
```

这为后续的 **LLM-as-a-Judge 评估闭环** 和 **Prompt 迭代优化** 提供了原始数据。

---

### 3.3 认知账本 (Memory)

Memory 模块实现了**长期记忆与心智评估**，底层基于向量检索。

#### 3.3.1 三层记忆架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Layer 1: 原始记忆 (memories)                                │
│  ├── 用户每次发言的完整记录                                   │
│  ├── 认知画像标签 (core_preference, reasoning_style, ...)   │
│  └── 向量 embedding (BLOB)                                   │
│                                                              │
│  Layer 2: 画像结晶 (profile_crystals)                        │
│  ├── 每 N 条记忆压缩成一段用户画像片段                        │
│  ├── 自动追加到 data/user_profile.md                        │
│  └── 被所有 LLM 调用自动注入为上下文                          │
│                                                              │
│  Layer 3: 认知轨迹 (trajectory)                              │
│  ├── 按月聚合 embedding 质心                                  │
│  ├── 计算相邻月份的思维漂移分数                               │
│  └── 识别核心关注点和认知模式演化                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 3.3.2 向量化记忆实现

**技术选型**：numpy + SQLite BLOB

**为什么不用 Milvus/Pinecone？**

> 数据规模 <10k 时，线性扫描 100ms 完全可接受；零依赖、Windows 无扩展坑；可随时切换到 sqlite-vec。

**核心代码**：

```python
def find_related_memories(
    query_text: str,
    exclude_id: int | None = None,
    limit: int = 5,
    min_similarity: float = 0.35,
) -> list[dict]:
    """向量召回 + 关键词回退"""
    
    # 1. 对 query 做 embedding
    query_vec = embedder.embed([query_text])[0]
    
    # 2. 遍历所有有 embedding 的记忆，计算余弦相似度
    scored = []
    for row in rows_with_embeddings:
        vec = blob_to_vec(row["embedding"])
        sim = cosine_similarity(query_vec, vec)
        if sim >= min_similarity:
            scored.append((sim, row))
    
    # 3. 按相似度排序返回
    scored.sort(key=lambda x: -x[0])
    return [row for _, row in scored[:limit]]
```

**关键词回退**：当向量召回为空（如旧记录无 embedding），自动回退到 `LIKE` 匹配。

#### 3.3.3 认知固化 (Crystallization)

每累计 N 条发言，系统会自动将散点认知压缩成一段**用户画像片段**：

```python
CRYSTAL_PROMPT = """
你是 MindPalace 的"认知画像结晶器"。

输入是用户最近 N 次发言的画像标签。
请压缩成一段可读的用户画像片段（<300 字），要求：
1. 用第二人称（"你倾向于..."）
2. 指出稳定的价值偏好 + 明显的思维漂移
3. 用一句话给出"这个阶段你最适合被哪种论点挑战"
"""
```

生成的画像会追加到 `data/user_profile.md`，并被 `llm/client.py` 自动注入所有 LLM 调用：

```python
def chat(system_prompt, user_prompt, ...):
    profile = _get_user_profile()  # 读取 data/user_profile.md
    if profile:
        system_prompt = f"{system_prompt}\n\n=== User Profile ===\n{profile}"
    # ... 调用 LLM
```

**效果**：Agent 会随着使用变得越来越"懂你"。

#### 3.3.4 回声定位 (Echo Location)

当用户对某个话题发表看法时，系统会调取历史相关观点进行跨期对比：

```python
# 1. 向量召回相关历史记忆
related = find_related_memories(user_response, exclude_id=memory_id)

# 2. 生成对比报告
echo = generate_echo_report(
    current_response=user_response,
    current_tags=current_tags,
    historical_memories=related,
)
```

**Echo 报告示例**：

```
=== [Echo Location] -- Cognitive Reflection ===

[Stance Shift]
你现在的观点比以前更务实了。三个月前你认为"技术能解决一切"，
现在你开始关注"技术的社会成本"。

[Reasoning Shift]
从直觉判断变成了系统性分析。你开始引用具体数据和案例，
而不是依赖抽象概念。

[Tone Drift]
从乐观激进变成了审慎平衡。

[!! Bias Alert !!]
你似乎对"效率"有强烈的偏好，可能忽略了"公平"维度。
建议阅读一些关于技术伦理的文章。

* 你的认知正在从"技术乐观主义"向"技术现实主义"演化。
```

#### 3.3.5 长对话压缩

Resolve 模块支持无限轮次对话，通过**自动压缩历史**解决 Token 爆炸：

```python
def _compress_history_if_needed(self):
    if len(self.history) > 40:
        early = self.history[:30]
        summary = chat(COMPRESS_PROMPT, _fmt(early), provider_config=FAST_CONFIG)
        self.history = [
            {"role": "system", "content": f"[历史摘要] {summary}"}
        ] + self.history[30:]
```

---

### 3.4 Self-RAG 工具系统

Critic 和 Synthesizer 在辩论中可**主动触发工具调用**，对不确定的事实进行核查。

#### 3.4.1 Tool 协议

```python
class Tool(Protocol):
    name: str
    description: str
    parameters: dict  # JSON Schema
    
    def run(self, **kwargs) -> str: ...
```

#### 3.4.2 已实现工具

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| `web_search` | DuckDuckGo 搜索 | 查证统计数字、历史事件、引用论文 |
| `fact_check` | 搜索 + LLM 判定 | 对模糊论断进行综合核查 |

#### 3.4.3 工具调用流程

```python
def chat_with_tools(system_prompt, user_prompt, tools_schema, tool_executor, max_tool_calls=3):
    messages = [{"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_prompt}]
    
    for i in range(max_tool_calls + 1):
        final_round = (i == max_tool_calls)
        
        if final_round:
            messages.append({
                "role": "system",
                "content": "⚠️ 你必须立即给出最终答复，不得再调用任何工具。"
            })
        
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_schema if not final_round else None,
            tool_choice="auto" if not final_round else "none",
        )
        
        if resp.message.tool_calls and not final_round:
            # 执行工具调用
            for tc in resp.message.tool_calls:
                result = tool_executor[tc.function.name].run(**tc.args)
                messages.append({"role": "tool", "content": result})
            continue
        
        return resp.message.content
```

**防抖机制**：
- 最多 3 轮工具调用
- 最后一轮强制禁止工具调用
- 工具调用记录写入 `Turn.tool_log`

#### 3.4.4 角色工具权限

```python
# 只有 Critic 和 Synthesizer 有工具权限
# Mentor 不查证据，只追问
TOOL_ENABLED_ROLES = {"critic", "synthesizer"}
```

**Prompt 注入**：
```
如果你对某个事实不确定（如统计数字、历史事件、引用论文），
**必须**调用 `web_search` 或 `fact_check` 工具核查，而不是编造。
核查后在 JSON 里多加 `citations` 字段。
```

---

### 3.5 评估闭环

MindPalace 实现了完整的"讨论 → 反馈 → 评估 → 优化"闭环。

#### 3.5.1 用户反馈收集

每次 Council 讨论后收集反馈：

```python
print("本次 Council 讨论如何？")
print("[1] 👍 有启发  [2] 👎 无意义  [3] 📌 采纳某观点")
choice = input("> ")
save_feedback(debate_id=state.id, rating=choice)
```

存储到 `feedback` 表：

```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    debate_id INTEGER,
    rating TEXT,           -- "up" | "down" | "adopted"
    adopted_role TEXT,     -- 如果是 adopted，记录采纳的角色
    note TEXT,
    created_at TEXT
);
```

#### 3.5.2 LLM-as-a-Judge 周度评估

用最强档模型对历史 debates 打分：

```python
EVAL_PROMPT = """
你是 MindPalace 的元评估器。对这次 Council 讨论打分（每项 1-10）：

- logical_rigor: 论证严密度
- inspiration: 启发性
- coverage: 角度覆盖度
- groundedness: 事实扎实度（有无 citations）

输出 JSON: {"scores": {...}, "weaknesses": [...], "prompt_improvement_hint": "..."}
"""
```

**周度报告示例**：

```markdown
# Weekly Eval Report (2024-01-15)

评估周期: 最近 7 天 | 评估数量: 12

## 平均分数
| 维度 | 分数 |
|------|------|
| logical_rigor | 7.2 |
| inspiration | 6.8 |
| coverage | 5.5 |
| groundedness | 6.1 |

## Top Weaknesses
1. Critic 经常遗漏"假设崩塌条件"的分析
2. Synthesizer 的跨学科类比有时牵强
3. Judge 的 recommended_stance 过于笼统

## Prompt Improvement Hints
1. 在 Critic prompt 中强调"必须说明假设在什么条件下会失效"
2. 给 Synthesizer 增加"类比成立的前提条件"检查
```

#### 3.5.3 Prompt 迭代建议

基于评估报告 + 用户反馈分布，自动生成针对具体角色的 prompt 改进方案：

```python
def generate_iteration_suggestions(days: int = 7) -> str:
    reports = judge_recent_debates(days)
    feedbacks = get_feedback_distribution(days)
    
    # 聚合弱点
    weaknesses = aggregate_weaknesses(reports)
    
    # 结合用户反馈
    low_rated = filter_low_rated(reports, feedbacks)
    
    # 生成改进建议
    return chat_json(ITERATION_PROMPT, {
        "weaknesses": weaknesses,
        "low_rated_debates": low_rated,
    })
```

---

## 4. 工程亮点与面试话术

### 4.1 多 Agent 编排

**亮点**：
- 基于有限状态机的辩论流程，而非简单的顺序调用
- 动态难度路由，成本优化 40%
- 强制落地机制，彻底解决死循环

**面试话术**：
> 我设计了一个基于状态机的多 Agent 辩论系统。通过动态路由，简单话题只派 1 个 Agent，复杂话题派全员，token 成本降低约 40%。借鉴 LangGraph 的设计思想，通过"倒数第 N 轮强制落地"机制，彻底解决多 Agent 死循环问题。

### 4.2 长期记忆

**亮点**：
- 向量召回 + 关键词回退的双层检索
- 三层记忆架构（原始/结晶/轨迹）
- 自动画像注入，Agent 越用越懂你

**面试话术**：
> 我实现了双层记忆架构：原始对话 + 定期 LLM 压缩的"价值观结晶"。用 SQLite + numpy 余弦相似度实现轻量向量召回，对用户发言按月聚合 embedding 质心，实现跨月认知漂移追踪。数据规模 <10k 时线性扫描 100ms 可接受，零依赖部署。

### 4.3 Self-RAG

**亮点**：
- 角色主动触发工具调用，而非被动等待
- 轮次上限 + 强制落地，杜绝工具无限循环
- 输出带 citations，显著降低幻觉

**面试话术**：
> Critic 和 Synthesizer 在发言中可主动触发 web_search 工具，发现事实不足时补充检索并在输出中给出 citations，显著降低幻觉。通过 OpenAI function calling + 轮次上限防抖，杜绝工具无限循环。

### 4.4 评估闭环

**亮点**：
- 用户反馈 + LLM-as-a-Judge 双重评估
- 自动生成 Prompt 改进建议
- 完整的"讨论 → 反馈 → 评估 → 优化"闭环

**面试话术**：
> 引入 GPT-4o 级 Judge 对历史讨论打分（逻辑/启发/覆盖/扎实），结合用户反馈生成周度 Prompt 迭代报告，闭环优化角色 prompt。

### 4.5 可观测性

**亮点**：
- OpenTelemetry + Arize Phoenix 纯本地链路追踪
- OpenAI SDK 自动埋点 + 关键管道手动埋点
- 零开销默认关闭，一键开启

**面试话术**：
> 基于 OpenTelemetry + Arize Phoenix 实现纯本地链路追踪，所有 OpenAI SDK 调用自动捕获为 OTel Span，包含 token 用量和延迟。通过环境变量一键开启，关闭时 span 为 OTel no-op，零开销。

---

## 5. 技术选型与权衡决策

### 5.1 向量存储：numpy + BLOB vs Milvus

| 方案 | 优点 | 缺点 | 决策 |
|------|------|------|------|
| numpy + BLOB | 零依赖、Windows 无坑、<10k 性能足够 | 无索引、扩展性差 | **MVP 选择** |
| sqlite-vec | 有索引、仍零依赖 | 需要加载扩展 | 预留接口 |
| Milvus/Pinecone | 高性能、可扩展 | 重依赖、部署复杂 | 未来考虑 |

**决策**：MVP 阶段用 numpy + BLOB，预留 `VectorIndex` 抽象，可随时切换。

### 5.2 搜索工具：DuckDuckGo vs Tavily

| 方案 | 优点 | 缺点 | 决策 |
|------|------|------|------|
| DuckDuckGo | 免费、无 key | 稳定性一般 | **默认选择** |
| Tavily | 稳定、专为 AI 设计 | 收费 | 备选 |

**决策**：MVP 用 DDG，通过 `SEARCH_PROVIDER` 环境变量可切换。

### 5.3 Judge 模型：Claude vs DeepSeek

| 模型 | 优点 | 缺点 | 决策 |
|------|------|------|------|
| Claude-3.5-Sonnet | 逻辑最强、文学性好 | 贵、可能被墙 | **首选** |
| DeepSeek-Reasoner | 性价比高、推理强 | 文学性稍弱 | **备选** |

**决策**：Judge 是质量守门员，必须用第一梯队；双候选利用现有 failover 机制。

---

## 6. 扩展性与未来规划

### 6.1 已完成

- [x] Phase A: Council 状态机重构 — 难度路由 + Judge 共识收敛 + 强制落地
- [x] Phase B: 向量化记忆 + 认知固化 — Embedding 语义召回 + 自动画像结晶
- [x] Phase C: Self-RAG + 评估闭环 — 工具调用 + LLM-as-a-Judge
- [x] LLMOps 可观测性 — OpenTelemetry + Phoenix 本地追踪

### 6.2 规划中

- [ ] Phase 5: 更多信息源（微信公众号、播客等）
- [ ] Phase 7: Web 前端

### 6.3 架构扩展点

1. **向量存储**：预留 `VectorIndex` 抽象，可切换到 sqlite-vec / Milvus
2. **搜索工具**：通过 `SEARCH_PROVIDER` 环境变量切换
3. **LLM Provider**：支持所有 OpenAI 兼容 API
4. **自定义角色**：在 `data/personas/` 目录添加 `.md` 文件即可

---

## 附录：项目结构

```
mindpalace/
├── src/
│   ├── app.py              # CLI 入口 + 交互式菜单
│   ├── config.py           # 配置管理（分层 Provider）
│   ├── scout/              # 信息抓取与评分
│   │   ├── fetch.py        # RSS 抓取
│   │   ├── normalize.py    # 内容清洗
│   │   ├── score.py        # 启发性评分
│   │   └── pipeline.py     # 流水线编排
│   ├── council/            # 多智能体辩论（状态机 + Self-RAG）
│   │   ├── state.py        # DebateState / Turn / Phase 数据模型
│   │   ├── router.py       # 难度路由
│   │   ├── rebuttal.py     # opening / rebuttal prompt 构造
│   │   ├── judge.py        # The Judge：midcheck + finalize
│   │   ├── flow.py         # 状态机主循环
│   │   ├── roles.py        # 角色定义 + 工具权限
│   │   └── output.py       # 结果格式化
│   ├── memory/             # 认知记忆（向量 + 结晶 + 轨迹）
│   │   ├── embedder.py     # Embedding 抽象 + OpenAI 实现
│   │   ├── profiler.py     # 认知画像
│   │   ├── store.py        # 记忆存储（向量召回 + 关键词回退）
│   │   ├── echo.py         # 回声定位
│   │   ├── crystallize.py  # 认知固化 pipeline
│   │   └── trajectory.py   # 月度质心漂移分析
│   ├── tools/              # Council 可调用的外部工具 (Self-RAG)
│   │   ├── base.py         # Tool Protocol + 注册表
│   │   ├── web_search.py   # DuckDuckGo 搜索
│   │   └── fact_check.py   # 事实核查
│   ├── eval/               # 评估闭环
│   │   ├── feedback.py     # 用户反馈收集
│   │   ├── judge_debates.py # LLM-as-a-Judge 周度评分
│   │   └── prompt_iterator.py # Prompt 改进建议生成
│   ├── resolve/            # 交互式对话（自动压缩）
│   │   └── engine.py       # REPL 引擎 + history 压缩
│   ├── workflows/          # 端到端流程
│   │   └── daily_session.py # 抓取→讨论→记忆→回声→结晶→反馈
│   ├── llm/                # LLM 调用封装
│   │   └── client.py       # chat / chat_json / chat_with_tools
│   ├── obs/                # 可观测性 (LLMOps)
│   │   └── tracing.py      # OTel + Phoenix 初始化
│   └── storage/            # 数据持久化
│       └── db.py           # SQLite DDL + CRUD
├── data/
│   ├── personas/           # 自定义角色定义
│   ├── user_profile.md     # 结晶累计的用户画像
│   └── library/            # 本地知识库
├── eval/                   # 周度评估报告输出
├── tests/                  # 测试用例
├── .env                    # 本地配置
└── pyproject.toml
```

---

**MindPalace Agent** — 让每一次思考都有迹可循，让每一次辩论都推动认知进化。
