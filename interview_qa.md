# MindPalace Agent 面试问答模拟

> 模拟面试官对项目的深度提问与标准回答

---

## 目录

1. [项目概述类问题](#1-项目概述类问题)
2. [多 Agent 编排类问题](#2-多-agent-编排类问题)
3. [记忆系统类问题](#3-记忆系统类问题)
4. [RAG 与工具调用类问题](#4-rag-与工具调用类问题)
5. [评估与优化类问题](#5-评估与优化类问题)
6. [工程实践类问题](#6-工程实践类问题)
7. [系统设计类问题](#7-系统设计类问题)
8. [扩展与反思类问题](#8-扩展与反思类问题)

---

## 1. 项目概述类问题

### Q1.1: 请用一分钟介绍这个项目

**回答**：

MindPalace 是一个**多智能体认知训练系统**，核心理念是"思考增强"而非"搜索增强"。

系统由三个核心模块组成：
- **Scout**：主动抓取高质量内容，通过三维度评分筛选启发性文章
- **Council**：多个不同人格的 Agent 进行对抗式辩论，通过苏格拉底式提问挑战用户思维
- **Memory**：向量化长期记忆，追踪用户认知演变

技术亮点包括：基于状态机的多 Agent 编排、动态难度路由降低 40% token 成本、强制落地机制解决死循环、Self-RAG 工具调用、LLM-as-a-Judge 评估闭环。

---

### Q1.2: 这个项目解决了什么痛点？和 ChatGPT 有什么本质区别？

**回答**：

**痛点**：传统 AI 助手是"顺从型"的——用户问什么就答什么，容易让用户陷入信息茧房和确认偏误。用户缺乏深度思考的动力和工具。

**本质区别**：

| 维度 | ChatGPT | MindPalace |
|------|---------|------------|
| 交互模式 | 用户问 → AI 答 | AI 主动发起辩论 |
| 目标 | 快速给答案 | 逼迫用户深度思考 |
| 记忆 | 无长期记忆 | 向量化长期记忆 + 认知轨迹追踪 |
| 多角度 | 单一视角 | 多个对立人格辩论 |
| 反馈 | 无闭环 | 用户反馈 + LLM 评估 + Prompt 迭代 |

MindPalace 的角色是"认知教练"而非"信息检索员"。

---

### Q1.3: 这个项目的应用场景是什么？目标用户是谁？

**回答**：

**目标用户**：
- 知识工作者：需要持续学习、保持思维敏锐度
- 研究者：需要多角度审视观点、发现盲点
- 决策者：需要在复杂问题上听取多方意见
- 终身学习者：希望追踪自己的认知演变

**应用场景**：
- 每日认知训练：抓取高质量文章 → 多 Agent 辩论 → 记录观点 → 追踪演变
- 观点审视：对某个议题发起辩论，听取 Critic、Synthesizer、Mentor 的不同视角
- 认知复盘：通过 Echo Location 对比历史观点，识别思维定式和偏见

---

## 2. 多 Agent 编排类问题

### Q2.1: 你提到了"状态机设计"，能详细说说吗？为什么不直接用顺序调用？

**回答**：

**为什么不用顺序调用**：

顺序调用无法处理以下场景：
1. **条件分支**：简单话题派 1 个 Agent，复杂话题派 3 个
2. **循环控制**：辩论可能需要多轮，但何时停止？
3. **状态追踪**：每轮发言、分歧度、终止原因都需要记录
4. **失败恢复**：某个阶段失败时如何降级？

**状态机设计**：

```python
class Phase(str, Enum):
    ROUTING   = "routing"       # 难度评估
    OPENING   = "opening"       # 各角色首轮陈述
    REBUTTAL  = "rebuttal"      # 互相反驳循环
    JUDGING   = "judging"       # Judge 收敛
    DONE      = "done"
```

每个阶段有明确的进入条件、执行逻辑、转移条件。`DebateState` 数据类贯穿全程，记录完整状态。

**好处**：
- 可复现：任何时刻的状态都可以序列化
- 可调试：明确知道卡在哪个阶段
- 可扩展：新增阶段只需定义转移规则

---

### Q2.2: 多 Agent 辩论很容易陷入死循环，你是怎么解决的？

**回答**：

这是多 Agent 系统的经典难题。我实现了**三层防护**：

**第一层：硬性轮数上限**

```python
MAX_REBUTTAL_ROUNDS = 3
while state.round_idx < state.max_rebuttal_rounds:
    # ... 辩论逻辑
```

达到上限必须停止，这是兜底机制。

**第二层：分歧度早停**

每轮结束后，Judge 评估各方观点的分歧度：

```python
check = judge_midcheck(state)
if check["disagreement_score"] < CONVERGE_THRESHOLD:  # 默认 0.3
    state.terminated_by = "converged"
    break
```

如果分歧度已经很低，说明各方观点趋同，没必要继续。

**第三层：强制落地（Force Closing）**

借鉴 akashic-agent 的设计，最后一轮注入系统级约束：

```python
if force_closing:
    prompt += """
    ⚠️ 这是最后一轮。禁止引入新论点。只能：
       1. 针对对手最新观点做简短回应（<100 字）
       2. 归纳你本次辩论的最终立场
    """
```

这确保即使辩论没有自然收敛，最后一轮也会产出结论。

---

### Q2.3: 动态难度路由是怎么实现的？为什么能降低 40% 成本？

**回答**：

**实现逻辑**：

Router 用轻量模型评估话题难度：

```python
ROUTER_PROMPT = """
判断标准：
- easy: 单一观点、无逻辑争议、事实性介绍
- medium: 有一定论证链，存在可挑战的假设
- hard: 多方观点冲突、涉及价值判断或复杂因果
"""
```

根据难度决定派几个角色：

```python
DIFFICULTY_ROLES = {
    "easy":   ["mentor"],                           # 1 个 Agent
    "medium": ["critic", "mentor"],                  # 2 个 Agent
    "hard":   ["critic", "synthesizer", "mentor"],  # 3 个 Agent
}
```

**成本计算**：

假设平均每轮辩论调用 3 次 LLM（每个 Agent 一次）：

| 难度 | Agent 数 | Opening 调用 | Rebuttal 调用 (3轮) | 总调用 |
|------|----------|--------------|---------------------|--------|
| easy | 1 | 1 | 0 (单角色不辩论) | **1** |
| medium | 2 | 2 | 2×3=6 | **8** |
| hard | 3 | 3 | 3×3=9 | **12** |

如果话题分布是 30% easy、40% medium、30% hard：

- 无路由（全部 hard）：12 次调用
- 有路由：0.3×1 + 0.4×8 + 0.3×12 = **6.7 次调用**

**节省约 44%**。

---

### Q2.4: 为什么 Critic、Synthesizer、Mentor 这三个角色是这样设计的？有考虑过其他角色吗？

**回答**：

**设计逻辑**：

这三个角色形成**互补的认知三角**：

```
        Critic (批判)
           △
          ╱ ╲
         ╱   ╲
        ╱     ╲
       ╱       ╲
      ▼─────────▼
Mentor (追问) ←── Synthesizer (连接)
```

- **Critic**：解构，找漏洞，挑战直觉
- **Synthesizer**：建构，跨学科连接，提供新视角
- **Mentor**：引导，通过追问逼迫深度思考

**为什么不是其他组合**：

- 两个 Critic？会变成纯粹的否定，没有建设性
- 两个 Mentor？会变成温和的聊天，缺乏冲击力
- 加一个"支持者"？会削弱 Critic 的挑战效果

**扩展性**：

系统支持自定义角色，只需在 `data/personas/` 添加 `.md` 文件。有用户添加过"历史学家"角色，专门从历史视角分析问题。

---

### Q2.5: Judge 角色的 midcheck 和 finalize 有什么区别？为什么要分开？

**回答**：

**职责不同**：

| 方法 | 职责 | 模型要求 | 调用频率 |
|------|------|----------|----------|
| midcheck | 评估分歧度，决定是否继续 | 低（便宜模型） | 每轮一次 |
| finalize | 产出最终共识 | 高（最强模型） | 仅最后一次 |

**为什么要分开**：

1. **成本优化**：midcheck 只需输出一个 0-1 的分数，用便宜模型即可；finalize 需要综合整场辩论产出高质量结论，必须用强模型。

2. **职责分离**：midcheck 是"裁判"，只判断"要不要继续"；finalize 是"总结者"，需要理解整场辩论并给出可执行的结论。

**代码体现**：

```python
# midcheck 用 router 档（便宜）
def midcheck(state, provider_config=None):
    cfg = provider_config or get_router_config()  # 便宜模型
    # ...

# finalize 用 judge 档（最强）
def finalize(state, provider_config=None):
    cfg = provider_config or get_judge_config()  # 最强模型
    # ...
```

---

## 3. 记忆系统类问题

### Q3.1: 为什么用 numpy + SQLite BLOB 而不是专业的向量数据库？

**回答**：

**决策依据**：

| 因素 | numpy + BLOB | Milvus/Pinecone |
|------|--------------|-----------------|
| 数据规模 | <10k 完全够用 | 适合百万级以上 |
| 部署复杂度 | 零依赖 | 需要独立部署 |
| Windows 兼容 | 完美 | 可能有坑 |
| 查询延迟 | ~100ms 线性扫描 | ~10ms 有索引 |
| 学习成本 | 低 | 高 |

**我的判断**：

- MVP 阶段用户记忆条数不会超过 10k
- 线性扫描 100ms 对用户体验影响可忽略
- 零依赖意味着更低的部署门槛和更少的故障点

**扩展性预留**：

```python
class Embedder(ABC):
    def embed(self, texts: list[str]) -> list[np.ndarray]: ...

# 未来可以无缝切换到 Milvus
class MilvusEmbedder(Embedder):
    def embed(self, texts): ...
```

---

### Q3.2: 向量召回和关键词回退是怎么结合的？为什么不只用一种？

**回答**：

**为什么需要双重机制**：

| 场景 | 向量召回 | 关键词回退 |
|------|----------|------------|
| 语义相似但用词不同 | ✅ 能找到 | ❌ 找不到 |
| 旧记录无 embedding | ❌ 无法召回 | ✅ 能找到 |
| API 不可用 | ❌ 无法计算 query 向量 | ✅ 仍可工作 |
| 精确匹配 | ❌ 可能漏掉 | ✅ 精确命中 |

**实现逻辑**：

```python
def find_related_memories(query_text, ...):
    # 1. 优先向量召回
    results = _vector_search(query_text, ...)
    if results:
        return results
    
    # 2. 向量召回为空时，回退到关键词
    results = _keyword_fallback(query_text, ...)
    return results
```

**关键词提取策略**：

```python
def _extract_keywords(text):
    # 英文：提取 >= 2 字母的完整单词
    en_tokens = re.findall(r'[a-zA-Z]{2,}', text)
    
    # 中文：生成 2-gram（双字切片）
    zh_bigrams = [seq[i:i+2] for seq in zh_seqs for i in range(len(seq)-1)]
    
    return en_tokens + zh_bigrams
```

---

### Q3.3: 认知固化（Crystallization）是什么？为什么要压缩记忆？

**回答**：

**问题**：

用户每次发言都存储为一条 memory，随着使用积累，会有几百条记录。问题是：
1. LLM 上下文窗口有限，不可能把所有历史都塞进去
2. 散点的记忆难以形成连贯的用户画像
3. 用户自己都忘了自己说过什么

**解决方案**：

每累计 N 条（默认 10 条）memory，用 LLM 压缩成一段**用户画像片段**：

```python
CRYSTAL_PROMPT = """
请把用户最近 N 次发言的画像标签压缩成一段可读的用户画像片段（<300 字）：
1. 用第二人称（"你倾向于..."）
2. 指出稳定的价值偏好 + 明显的思维漂移
3. 用一句话给出"这个阶段你最适合被哪种论点挑战"
"""
```

**效果**：

生成的画像追加到 `data/user_profile.md`，被所有 LLM 调用自动注入：

```python
def chat(system_prompt, user_prompt, ...):
    profile = _get_user_profile()  # 读取 user_profile.md
    if profile:
        system_prompt = f"{system_prompt}\n\n=== User Profile ===\n{profile}"
```

**价值**：

- Agent 越用越"懂你"
- 用户可以看到自己的认知演变
- 长期记忆不会无限膨胀

---

### Q3.4: Echo Location（回声定位）是怎么实现的？能举个例子吗？

**回答**：

**实现流程**：

```python
# 1. 用户发表观点后，向量召回相关历史记忆
related = find_related_memories(user_response, exclude_id=memory_id)

# 2. 构建对比 Prompt
user_prompt = f"""
=== 用户本次发言 ===
{current_response}

本次认知标签:
  核心偏好: {current_tags['core_preference']}
  推理模式: {current_tags['reasoning_style']}
  情感底色: {current_tags['emotional_tone']}

=== 历史画像 ===
{format_historical_memories(related)}
"""

# 3. LLM 生成对比报告
echo = chat_json(ECHO_SYSTEM_PROMPT, user_prompt)
```

**实际例子**：

用户三个月前对"AI 取代人类"的观点：
> "技术进步是好事，人类会找到新工作。"

用户现在对同一话题的观点：
> "我开始担心了，这次好像不一样，AI 能做创造性工作了。"

Echo 报告：
```
[Stance Shift]
你从"技术乐观主义"转向了"审慎担忧"。三个月前你认为"人类会找到新工作"，
现在你开始质疑这个假设，因为"AI 能做创造性工作"打破了你的预期。

[Reasoning Shift]
从"历史归纳"（过去每次技术革命都如此）变成了"差异分析"（这次有什么不同）。

[Tone Drift]
从"自信"变成了"不确定"。

[!! Bias Alert !!]
你似乎对"创造性工作"有特殊情结，可能是因为你自己的职业属于这一类。
建议阅读一些关于"AI 如何改变非创造性工作"的文章，避免自我中心偏差。

* 你的认知正在经历一次重要的范式转换，这是好事。
```

---

### Q3.5: 长对话压缩是怎么做的？不会丢失信息吗？

**回答**：

**问题**：

Resolve 模块支持无限轮次对话，但对话历史会无限增长，最终超出 LLM 上下文窗口。

**解决方案**：

当对话超过 40 条消息时，压缩早期的 30 条：

```python
def _compress_history_if_needed(self):
    if len(self.history) > 40:
        early = self.history[:30]
        summary = chat(COMPRESS_PROMPT, _fmt(early), provider_config=FAST_CONFIG)
        self.history = [
            {"role": "system", "content": f"[历史摘要] {summary}"}
        ] + self.history[30:]
```

**压缩 Prompt**：

```
请将以下对话历史压缩成一段紧凑的摘要（<500 字），保留：
1. 用户的核心观点和立场
2. 讨论过的主要话题
3. 未解决的问题或悬念
```

**信息丢失问题**：

确实会丢失细节，但：
1. 保留的是"核心观点"和"未解决问题"，这些是后续对话最需要的
2. 原始对话已存储在数据库，随时可以回溯
3. 这是"有损压缩"与"无限对话"之间的合理权衡

---

## 4. RAG 与工具调用类问题

### Q4.1: Self-RAG 和传统 RAG 有什么区别？

**回答**：

| 维度 | 传统 RAG | Self-RAG |
|------|----------|----------|
| 触发时机 | 用户查询时 | Agent 发言时主动判断 |
| 检索内容 | 预先索引的文档库 | 实时网络搜索 |
| 决策权 | 系统决定是否检索 | Agent 自己决定 |
| 目的 | 补充知识 | 核查事实、补充证据 |

**Self-RAG 的核心**：

Agent 在发言过程中，如果发现自己对某个事实不确定，**主动**调用工具核查：

```
Critic: "文章声称全球有 70% 的程序员在使用 AI 工具..."
        [内部判断：这个数据我确定吗？不确定。]
        [调用 web_search 工具]
        [获得结果：实际是 2024 年 Stack Overflow 调查显示 76%...]
        "文章声称 70%，但根据 Stack Overflow 2024 调查，实际是 76%..."
```

---

### Q4.2: 工具调用怎么防止无限循环？

**回答**：

**问题**：

Agent 可能不断调用工具，永远不给出最终答案。

**解决方案**：

```python
def chat_with_tools(..., max_tool_calls=3):
    for i in range(max_tool_calls + 1):
        final_round = (i == max_tool_calls)
        
        if final_round:
            # 最后一轮强制禁止工具调用
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
            # 执行工具调用，继续循环
            ...
            continue
        
        # 没有工具调用或最后一轮，返回结果
        return resp.message.content
```

**三层保障**：
1. `max_tool_calls=3` 硬性上限
2. 最后一轮 `tool_choice="none"` 强制禁止
3. 最后一轮注入系统提示"必须给出最终答复"

---

### Q4.3: 为什么只有 Critic 和 Synthesizer 有工具权限？Mentor 为什么没有？

**回答**：

**角色定位不同**：

| 角色 | 定位 | 是否需要工具 | 原因 |
|------|------|--------------|------|
| Critic | 找漏洞、挑战事实 | ✅ 需要 | 需要核查文章中的数据和论断 |
| Synthesizer | 跨学科连接 | ✅ 需要 | 需要验证引用的案例和理论 |
| Mentor | 苏格拉底式追问 | ❌ 不需要 | 追问不需要查证据，只需要提问 |

**设计原则**：

Mentor 的力量在于"提问"而非"回答"。如果 Mentor 也去查证据，就变成了另一个 Critic，失去了角色的独特性。

**代码体现**：

```python
TOOL_ENABLED_ROLES = {"critic", "synthesizer"}

if role_key in TOOL_ENABLED_ROLES and not force_closing:
    result = chat_with_tools(...)
else:
    result = chat_json(...)  # 普通路径
```

---

### Q4.4: DuckDuckGo 搜索的稳定性如何？有考虑过其他方案吗？

**回答**：

**DuckDuckGo 的优缺点**：

| 优点 | 缺点 |
|------|------|
| 免费、无需 API Key | 稳定性一般，偶尔超时 |
| 无使用限制 | 结果质量不如 Google |
| 隐私友好 | 中文搜索效果一般 |

**备选方案**：

```python
_SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo").lower()

# 可切换到 Tavily（专为 AI 设计的搜索 API）
if _SEARCH_PROVIDER == "tavily":
    return _search_tavily(query)
```

**决策**：

- MVP 阶段用 DDG，零成本、快速验证
- 生产环境可切换到 Tavily（$0.01/次调用）
- 通过环境变量一键切换，无需改代码

---

## 5. 评估与优化类问题

### Q5.1: LLM-as-a-Judge 是怎么实现的？怎么保证评估的客观性？

**回答**：

**实现流程**：

```python
def judge_recent_debates(days=7):
    # 1. 获取最近 N 天的辩论记录
    debates = get_debates_since(days)
    
    # 2. 对每场辩论打分
    reports = []
    for debate in debates:
        result = chat_json(EVAL_PROMPT, format_debate(debate), provider_config=JUDGE_CONFIG)
        reports.append(result)
    
    # 3. 聚合生成周度报告
    return generate_weekly_report(reports)
```

**评估维度**：

```python
EVAL_PROMPT = """
对这次 Council 讨论打分（每项 1-10）：
- logical_rigor: 论证严密度
- inspiration: 启发性
- coverage: 角度覆盖度
- groundedness: 事实扎实度（有无 citations）
"""
```

**客观性保障**：

1. **多维度评分**：不是单一分数，而是四个维度的综合
2. **具体弱点**：要求输出具体的 weakness，而非笼统评价
3. **改进建议**：要求输出 prompt_improvement_hint，可执行
4. **最强模型**：Judge 用 Claude-3.5-Sonnet 或 DeepSeek-Reasoner，保证评估质量

**局限性**：

LLM-as-a-Judge 仍然有主观性，但：
- 比没有评估好
- 结合用户反馈（👍/👎）形成双重验证
- 评估结果用于"发现趋势"而非"绝对判断"

---

### Q5.2: 用户反馈是怎么收集和利用的？

**回答**：

**收集时机**：

每次 Council 讨论结束后：

```
本次 Council 讨论如何？
[1] 👍 有启发  [2] 👎 无意义  [3] 📌 采纳某观点
```

**存储**：

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

**利用方式**：

1. **周度报告**：结合 LLM-as-a-Judge 评分，识别低分 + 低反馈的辩论模式
2. **Prompt 迭代**：针对低反馈的角色生成改进建议
3. **难度校准**：如果 easy 话题反馈普遍低，说明路由策略需要调整

---

### Q5.3: Prompt 迭代建议是怎么生成的？能举个例子吗？

**回答**：

**生成流程**：

```python
def generate_iteration_suggestions(days=7):
    # 1. 获取评估报告
    reports = judge_recent_debates(days)
    
    # 2. 获取用户反馈
    feedbacks = get_feedback_distribution(days)
    
    # 3. 聚合弱点
    weaknesses = aggregate_weaknesses(reports)
    
    # 4. 生成改进建议
    return chat_json(ITERATION_PROMPT, {
        "weaknesses": weaknesses,
        "low_feedback_topics": filter_low_feedback(feedbacks),
    })
```

**实际例子**：

输入（聚合的弱点）：
```
- Critic 经常遗漏"假设崩塌条件"的分析
- Synthesizer 的跨学科类比有时牵强
- Judge 的 recommended_stance 过于笼统
```

输出（改进建议）：
```markdown
## Prompt 改进建议

### Critic
在 prompt 中增加：
"对每个漏洞，必须说明'这个假设在什么具体条件下会失效'，
而不是只说'这个假设可能不成立'。"

### Synthesizer
在 prompt 中增加：
"每个跨学科类比必须说明'类比成立的前提条件'和'类比的局限性'。
如果类比只在特定条件下成立，必须明确指出。"

### Judge
在 prompt 中增加：
"recommended_stance 必须包含具体的行动建议，
而非笼统的'建议读者深入思考'。格式：'建议读者...，因为...'"
```

---

### Q5.4: 评估闭环的完整流程是怎样的？

**回答**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        评估闭环                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│  │ Council  │────▶│ 用户反馈 │────▶│ 数据库   │                │
│  │  辩论    │     │ 👍👎📌   │     │  存储    │                │
│  └──────────┘     └──────────┘     └──────────┘                │
│       │                                  │                      │
│       │                                  ▼                      │
│       │                          ┌──────────────┐              │
│       │                          │ LLM-as-Judge │              │
│       │                          │   周度评分    │              │
│       │                          └──────────────┘              │
│       │                                  │                      │
│       │                                  ▼                      │
│       │                          ┌──────────────┐              │
│       │                          │ Prompt 迭代  │              │
│       │                          │   建议生成    │              │
│       │                          └──────────────┘              │
│       │                                  │                      │
│       └──────────────────────────────────┘                      │
│                     优化后的 Prompt                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**时间维度**：
- 实时：用户反馈收集
- 周度：LLM-as-a-Judge 评估 + 报告生成
- 迭代：Prompt 改进建议 → 人工审核 → 更新角色 prompt

---

## 6. 工程实践类问题

### Q6.1: 这个项目的测试覆盖率如何？怎么保证代码质量？

**回答**：

**测试文件**：

```
tests/
├── test_council_flow.py      # 状态机流程测试
├── test_memory.py            # 记忆存储测试
├── test_memory_embedder.py   # 向量召回测试
├── test_llm_robustness.py    # LLM 调用容错测试
├── test_obs.py               # 可观测性测试
├── test_phase_c.py           # Self-RAG 测试
└── test_scout_pipeline.py    # Scout 流水线测试
```

**关键测试用例**：

```python
def test_easy_topic_single_agent():
    """easy 话题只派 1 个 Agent"""
    state = run_council(title="什么是 Python", summary="...", content="...")
    assert state.difficulty == "easy"
    assert state.active_roles == ["mentor"]
    assert len(state.turns) == 1

def test_force_closing():
    """达到最大轮数时强制落地"""
    state = run_council_with_max_rounds(max_rounds=2)
    assert state.terminated_by == "max_rounds"
    assert state.consensus is not None

def test_vector_search():
    """相似 memory 排在前面"""
    save_memory(..., user_response="我喜欢用 Python 做数据分析")
    save_memory(..., user_response="今天天气不错")
    
    results = find_related_memories("Python 数据科学工具")
    assert results[0]["similarity"] > 0.5
    assert "Python" in results[0]["user_response"]
```

**质量保障**：
- 单元测试覆盖核心逻辑
- 集成测试覆盖端到端流程
- Mock LLM 调用避免真实消耗

---

### Q6.2: LLM 调用的容错机制是怎么设计的？

**回答**：

**多层容错**：

```python
def chat(system_prompt, user_prompt, ...):
    # 1. 多模型轮换
    models_to_try = cfg["models"].copy()
    random.shuffle(models_to_try)  # 随机顺序，避免单点压力
    
    for model in models_to_try:
        for attempt in range(max_retries):
            try:
                return _call_llm(...)
            
            # 2. 频率限制：指数退避重试
            except RateLimitError:
                wait_time = (2 ** attempt) + random.random()
                time.sleep(wait_time)
            
            # 3. 内容过滤：跳过该模型
            except APIStatusError as e:
                if e.status_code == 403:
                    break  # 换下一个模型
            
            # 4. 连接错误：跳过该模型
            except APIConnectionError:
                break
    
    # 5. 全部失败：抛出友好错误
    raise RuntimeError("所有 LLM 尝试均已失败")
```

**降级策略**：

| 场景 | 降级方案 |
|------|----------|
| Router 失败 | 回退到 medium 档 |
| Judge finalize 失败 | 返回带 error 标记的降级 consensus |
| Embedding 失败 | 存储记忆但不带向量，后续用关键词回退 |
| 工具调用失败 | 返回错误信息，Agent 可选择忽略 |

---

### Q6.3: 可观测性是怎么实现的？为什么选择 OpenTelemetry？

**回答**：

**为什么选 OpenTelemetry**：

| 优点 | 说明 |
|------|------|
| 标准化 | 业界标准，可对接多种后端 |
| 零开销 | 关闭时 span 为 no-op，无性能损耗 |
| 本地优先 | Phoenix 进程内运行，数据不出本地 |
| 自动埋点 | OpenAI SDK 调用自动捕获 |

**实现**：

```python
# 自动埋点：OpenAI SDK 调用
@wraps(OpenAI.chat.completions.create)
def traced_create(*args, **kwargs):
    with span("llm.chat", model=kwargs.get("model")):
        return original_create(*args, **kwargs)

# 手动埋点：Council 辩论
with span("council.debate", article_title=title):
    state = run_council(...)
    span.set_attribute("difficulty", state.difficulty)
    span.set_attribute("terminated_by", state.terminated_by)
```

**可视化**：

启用后访问 `http://localhost:6006` 查看：
- 完整调用树
- Token 用量和成本
- 延迟分布
- 错误追踪

---

### Q6.4: 配置管理是怎么设计的？支持哪些 LLM Provider？

**回答**：

**分层配置**：

```python
def get_provider_config(prefix: str = "OPENAI") -> dict:
    # 1. 尝试特定配置
    api_key = os.getenv(f"{prefix}_API_KEY")
    if not api_key and prefix != "OPENAI":
        # 2. 回退到全局配置
        api_key = os.getenv("OPENAI_API_KEY")
    
    return {
        "api_key": api_key,
        "base_url": os.getenv(f"{prefix}_BASE_URL", "https://api.openai.com/v1"),
        "models": parse_models(os.getenv(f"{prefix}_MODEL_NAMES")),
    }
```

**支持的 Provider**：

| Provider | 配置示例 |
|----------|----------|
| OpenAI | `OPENAI_API_KEY=sk-...` |
| DeepSeek | `COUNCIL_API_KEY=sk-...`, `COUNCIL_BASE_URL=https://api.deepseek.com` |
| Claude | `JUDGE_API_KEY=sk-...`, `JUDGE_BASE_URL=https://api.anthropic.com/v1` |
| 本地 Ollama | `OPENAI_BASE_URL=http://localhost:11434/v1` |

**任务分档**：

```env
# Judge 用最强模型
JUDGE_MODEL_NAMES=claude-3-5-sonnet-20241022

# Council 用性价比模型
COUNCIL_MODEL_NAMES=deepseek-chat

# Router 用便宜模型
ROUTER_MODEL_NAMES=deepseek-chat
```

---

## 7. 系统设计类问题

### Q7.1: 如果用户量增长 100 倍，系统需要做哪些改进？

**回答**：

**瓶颈分析**：

| 组件 | 当前方案 | 瓶颈 | 扩展方案 |
|------|----------|------|----------|
| 数据库 | SQLite | 单机、写锁 | PostgreSQL |
| 向量存储 | numpy + BLOB | 线性扫描 O(n) | Milvus / Pinecone |
| LLM 调用 | 同步 | 阻塞 | 异步 + 队列 |
| 搜索 | DuckDuckGo | 限流 | Tavily / 自建索引 |

**扩展路径**：

```
Phase 1: 单机优化
├── SQLite → PostgreSQL
├── 同步 → 异步 (asyncio)
└── 添加缓存层 (Redis)

Phase 2: 水平扩展
├── 无状态 API 服务
├── 任务队列 (Celery / RQ)
└── 向量数据库 (Milvus)

Phase 3: 高可用
├── 多区域部署
├── LLM Provider 多活
└── 降级熔断机制
```

---

### Q7.2: 如果要给这个系统加一个 Web 前端，你会怎么设计？

**回答**：

**架构**：

```
┌─────────────────────────────────────────────────────────────┐
│                        Web Frontend                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   React/Vue  │───▶│   FastAPI    │───▶│  MindPalace  │   │
│  │   Frontend   │◀───│   Backend    │◀───│    Core      │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│         │                   │                   │            │
│         ▼                   ▼                   ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │   WebSocket  │    │  PostgreSQL  │    │   Milvus     │   │
│  │  (实时辩论)   │    │   (数据层)    │    │  (向量层)    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**关键功能**：

1. **实时辩论流**：WebSocket 推送每个 Agent 的发言
2. **认知仪表盘**：可视化用户画像和认知轨迹
3. **历史回放**：查看过去的辩论记录
4. **反馈收集**：一键 👍/👎/📌

**技术选型**：

| 层 | 技术 | 原因 |
|----|------|------|
| 前端 | React + Tailwind | 组件化、快速开发 |
| 后端 | FastAPI | 异步、自动文档 |
| 实时 | WebSocket | 双向通信 |
| 数据库 | PostgreSQL | 并发、扩展性 |

---

### Q7.3: 如果要支持多用户，需要做哪些改动？

{}
**回答**：

**当前架构**：单用户，所有数据存储在本地 SQLite。

**多用户改造**：

```python
# 1. 数据库增加 user_id
CREATE TABLE memories (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,  # 新增
    article_id INTEGER,
    ...
);

CREATE TABLE debates (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL,  # 新增
    ...
);

# 2. 所有查询增加 user_id 过滤
def find_related_memories(user_id, query_text, ...):
    rows = conn.execute(
        "SELECT * FROM memories WHERE user_id = ? AND ...",
        (user_id,)
    )

# 3. 用户画像按 user_id 隔离
def _get_user_profile(user_id):
    profile_path = PROJECT_ROOT / "data" / "profiles" / f"{user_id}.md"
    ...
```

**认证方案**：

| 方案 | 适用场景 | 复杂度 |
|------|----------|--------|
| API Key | 个人/小团队 | 低 |
| OAuth | 公开服务 | 中 |
| JWT | 企业内部 | 中 |

**隔离策略**：

- 数据隔离：每个用户的数据通过 `user_id` 隔离
- 配置隔离：每个用户可以有自己的模型偏好
- 画像隔离：每个用户有独立的 `user_profile.md`

---

### Q7.4: 系统的安全性如何保障？

**回答**：

**安全风险与对策**：

| 风险 | 对策 |
|------|------|
| API Key 泄露 | 存储在 `.env`，不提交到 Git |
| 用户输入注入 | Prompt 边界检查，限制特殊字符 |
| 工具调用滥用 | 轮次上限 + 域名白名单 |
| 数据泄露 | 本地存储，不上传云端 |

**敏感信息处理**：

```python
# 读取 .env 时不输出
api_key = os.getenv("OPENAI_API_KEY")
# 日志中脱敏
logger.info("Using API key: %s...", api_key[:8] + "****")

# 用户画像中不存储敏感信息
# Prompt 中明确禁止
MENTOR_PROMPT = """
...
禁止询问或存储用户的个人信息（姓名、电话、地址等）。
"""
```

**工具调用安全**：

```python
# 域名白名单
ALLOWED_DOMAINS = ["wikipedia.org", "arxiv.org", "github.com"]

def web_search(query):
    results = ddg_search(query)
    # 过滤危险域名
    return [r for r in results if is_safe_domain(r["url"])]
```

---

## 8. 扩展与反思类问题

### Q8.1: 这个项目最大的技术挑战是什么？你是怎么解决的？

**回答**：

**最大挑战：多 Agent 辩论的死循环问题**。

**问题本质**：

- Agent 之间互相反驳，没有自然的终止条件
- 每个 Agent 都想"说最后一句"
- LLM 的生成特性导致对话可能无限延伸

**解决过程**：

1. **第一版**：简单轮数上限
   - 问题：到上限时突然中断，没有结论
   
2. **第二版**：增加分歧度检测
   - 问题：分歧度计算不稳定，有时过早终止
   
3. **最终版**：三层防护
   - 轮数上限（兜底）
   - 分歧度早停（智能终止）
   - 强制落地（保证有结论）

**关键洞察**：

借鉴了 akashic-agent 的"倒数第 N 轮强制落地"设计。这不是纯技术问题，而是**工程与产品体验的平衡**——既要防止死循环，又要保证用户得到有价值的结论。

---

### Q8.2: 如果重新设计这个项目，你会做哪些不同的选择？

**回答**：

**会保持的选择**：

1. **状态机设计**：这是正确的抽象，让流程可控可调试
2. **分层模型调度**：成本优化的关键
3. **向量召回 + 关键词回退**：实用主义的平衡

**会改进的选择**：

| 原选择 | 改进方向 | 原因 |
|--------|----------|------|
| SQLite | PostgreSQL | 更早考虑并发和扩展 |
| 同步 LLM 调用 | 异步 | 提升响应速度 |
| 手动 Prompt 管理 | Prompt 版本控制 | 便于迭代和回滚 |
| 无缓存 | Redis 缓存 | 重复查询优化 |

**会新增的功能**：

1. **A/B 测试框架**：对比不同 Prompt 的效果
2. **用户画像可视化**：让用户看到自己的认知演变
3. **主动推送**：定时推送高质量内容，而非等待用户

---

### Q8.3: 这个项目让你学到了什么？

**回答**：

**技术层面**：

1. **多 Agent 编排**：理解了状态机、循环控制、角色设计
2. **LLM 工程化**：容错、降级、成本控制、可观测性
3. **向量检索**：从理论到实践，理解了相似度计算、索引、召回

**产品层面**：

1. **用户心理**：用户不想要"顺从的 AI"，而是"有挑战性的对话伙伴"
2. **闭环思维**：没有反馈的系统是盲目的，评估闭环是持续改进的基础
3. **成本意识**：不是所有任务都需要 GPT-4，分层调度是工程成熟度的体现

**工程层面**：

1. **简单优先**：numpy + BLOB 在 MVP 阶段足够，不要过早优化
2. **可观测性**：没有追踪的系统是黑盒，出问题无从下手
3. **渐进式架构**：先跑起来，再优化，不要一开始就设计完美架构

---

### Q8.4: 你认为这个项目还有哪些不足？

**回答**：

**功能不足**：

1. **信息源单一**：目前只支持 RSS，缺少微信公众号、播客等
2. **无 Web 前端**：CLI 门槛高，限制了用户群体
3. **无移动端**：无法随时随地使用

**技术不足**：

1. **同步架构**：LLM 调用阻塞，响应慢
2. **无缓存**：重复查询浪费 token
3. **测试覆盖**：核心逻辑有测试，但边界情况覆盖不足

**产品不足**：

1. **冷启动问题**：新用户没有历史记忆，Echo 功能无法发挥作用
2. **角色固定**：虽然有自定义机制，但默认角色可能不适合所有用户
3. **反馈机制简单**：只有 👍/👎，缺少细粒度反馈

**改进计划**：

| 不足 | 改进方案 | 优先级 |
|------|----------|--------|
| 信息源单一 | 接入更多爬虫 | 高 |
| 无 Web 前端 | FastAPI + React | 高 |
| 同步架构 | asyncio 重构 | 中 |
| 冷启动问题 | 预置通用画像模板 | 中 |
| 反馈机制简单 | 增加文字反馈 | 低 |

---

### Q8.5: 如果面试官问你"这个项目的亮点是什么"，你会怎么回答？

**回答**：

> MindPalace 是一个多智能体认知训练系统，有三个核心亮点：
>
> **第一，多 Agent 编排**。我设计了一个基于状态机的辩论系统，通过动态难度路由，简单话题只派 1 个 Agent，复杂话题派全员，token 成本降低约 40%。借鉴 LangGraph 的设计思想，通过"倒数第 N 轮强制落地"机制，彻底解决多 Agent 死循环问题。
>
> **第二，长期记忆系统**。我实现了双层记忆架构：原始对话 + 定期 LLM 压缩的"价值观结晶"。用 SQLite + numpy 余弦相似度实现轻量向量召回，对用户发言按月聚合 embedding 质心，实现跨月认知漂移追踪。数据规模 <10k 时线性扫描 100ms 可接受，零依赖部署。
>
> **第三，Self-RAG 与评估闭环**。Critic 和 Synthesizer 在发言中可主动触发 web_search 工具，发现事实不足时补充检索并在输出中给出 citations，显著降低幻觉。引入 GPT-4o 级 Judge 对历史讨论打分，结合用户反馈生成周度 Prompt 迭代报告，闭环优化角色 prompt。
>
> 这个项目的核心价值是"思考增强"而非"搜索增强"——它挑战用户，而不是顺从用户。

---

## 附录：快速记忆卡片

### 多 Agent 编排

| 问题 | 关键词 |
|------|--------|
| 为什么用状态机？ | 条件分支、循环控制、状态追踪、失败恢复 |
| 怎么防死循环？ | 轮数上限、分歧度早停、强制落地 |
| 动态路由降成本？ | easy=1 Agent, hard=3 Agent, 省 40% |
| 为什么分 midcheck/finalize？ | 成本优化、职责分离 |

### 记忆系统

| 问题 | 关键词 |
|------|--------|
| 为什么不用 Milvus？ | <10k 够用、零依赖、Windows 兼容 |
| 向量 + 关键词？ | 语义相似、无 embedding 回退、API 不可用兜底 |
| 认知固化？ | 压缩散点记忆、自动注入 LLM、越用越懂你 |
| Echo Location？ | 历史对比、偏见预警、认知漂移追踪 |

### Self-RAG

| 问题 | 关键词 |
|------|--------|
| 和传统 RAG 区别？ | Agent 主动触发、实时搜索、自己决定 |
| 防工具无限循环？ | max_tool_calls、最后一轮禁止、系统提示 |
| 为什么 Mentor 无工具？ | 追问不需要查证据、保持角色独特性 |

### 评估闭环

| 问题 | 关键词 |
|------|--------|
| LLM-as-Judge 客观性？ | 多维度、具体弱点、最强模型、结合用户反馈 |
| Prompt 迭代？ | 聚合弱点、生成可执行建议、人工审核 |

---

**祝面试顺利！** 🎯
