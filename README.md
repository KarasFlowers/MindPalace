# MindPalace Agent

> 你的私人认知进化实验室与多智能体议事厅

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MindPalace 不仅仅是一个信息聚合器，它是一个**"将信息转化为智慧"**的闭环系统。通过主动抓取高质量内容，驱动由多个不同人格组成的 Agent 专家团与用户进行深度对话、辩论和复盘，旨在通过"苏格拉底式提问"打破信息茧房，提升心智深度。

**🚀 [5分钟快速上手](QUICKSTART.md)** | **📖 [完整文档](#核心功能)**

## 核心功能

### 🎯 智能猎手 (The Scout)

精细化信息采编系统。区别于传统 RSS 订阅，Agent 根据认知图谱筛选具有启发性的内容。

**启发性评分机制**：从三个维度评估文章价值
- **信息密度**：单位篇幅中新概念、新逻辑、新数据的密集程度
- **原理深度**：是否从第一性原理解释机制，而非仅描述表象
- **因果链长度**：逻辑推演的层级深度

### 🏛️ 智库议事厅 (The Council)

基于有限状态机的多智能体**对抗式辩论**系统 + **LLM-as-a-Judge** 共识收敛。

**角色矩阵**

| 角色 | 职责 |
|------|------|
| **The Critic** (理性批判者) | 寻找逻辑漏洞，挑战直觉思维 |
| **The Synthesizer** (跨界连接者) | 将话题与其他学科进行关联 |
| **The Mentor** (苏格拉底导师) | 通过连续追问引导深度思考 |
| **The Judge** (主审 Agent) | 中期评估分歧度，最终收敛产出 consensus |

**状态机流程**

```
ROUTING  ──►  OPENING  ──►  REBUTTAL loop (≤ N 轮)  ──►  JUDGING  ──►  DONE
             │            │    ├─ 每轮各角色互相反驳
             │            │    ├─ midcheck 分歧度 < 阈值 → 提前收敛
             │            │    └─ 达到最大轮数 → 强制落地（force_closing）
```

**三项工程化设计**

- **动态难度路由**：Router 按话题难度分派角色数量。`easy` 只派 Mentor（省 2 次 LLM 调用），`hard` 派全员辩论。
- **强制落地 (force_closing)**：反驳循环最后一轮，在 prompt 层注入"禁止引入新论点，只能归纳已讨论内容"的系统级约束，配合硬性轮数上限，彻底防止多 Agent 死循环。
- **分层模型调度**：Router/Midcheck 用便宜小模型判分歧度；Finalize（Judge）用第一梯队模型保证结论质量。通过 env 前缀 `ROUTER_*` / `FAST_*` / `JUDGE_*` 独立配置。

每次辩论的完整状态（difficulty / turns / consensus / terminated_by）会落库到 `debates` 表，为后续 LLM-as-a-Judge 评估闭环和 prompt 迭代提供原始数据。

### 🧠 认知账本 (Memory)

长期记忆与心智评估系统。底层基于 **向量检索 (Vector Embedding)** 与精准语义匹配，确切追踪你的思维脉络。

**回声定位 (Echo Location)**：当你对某个话题发表看法时，Agent 会通过向量余弦相似度调取历史相关观点进行跨期对比：
> "你现在的观点比以前更务实了，是因为上次那个案例改变了你吗？"

**认知固化 (Crystallization)**：系统会定期（如每累计 10 条发言）将你的散点认知标签压缩成一段连贯的**用户画像片段 (User Profile)**，并自动注入到后续所有 LLM 的上下文中。Agent 会随着你的使用变得越来越"懂你"。

**认知轨迹 (Trajectory)**：按月聚合你发言的 Embedding 质心，计算相邻月份的**思维漂移分数 (Drift Score)**，识别你的核心关注点和认知模式的演化趋势。

### 🔍 Self-RAG (检索增强生成)

Critic 和 Synthesizer 在辩论中可**主动触发 `web_search` 和 `fact_check` 工具**（基于 OpenAI function calling），对不确定的事实论断进行网络核查，并在输出中附上 `citations` 来源。工具调用设有**轮次上限 + 强制落地**机制（借鉴 Council 的 akashic 式防抖），杜绝工具无限循环。

### 📊 评估闭环 (Eval Loop)

**用户反馈**：每次 Council 讨论后收集 👍(有启发) / 👎(无意义) / 📌(采纳某观点) 反馈，存入 `feedback` 表。

**LLM-as-a-Judge 周度评估**：用最强档模型（Judge 档）对历史 debates 打分（论证严密度 / 启发性 / 角度覆盖 / 事实扎实度），生成周度报告并输出 Top Weaknesses。

**Prompt 迭代建议**：基于评分报告 + 用户反馈分布，自动生成针对具体角色的 prompt 改进方案，形成"讨论 → 反馈 → 评估 → 优化"的闭环。

### 📡 可观测性 (Observability / LLMOps)

基于 **OpenTelemetry + Arize Phoenix** 的纯本地链路追踪：

- **自动埋点**：所有 OpenAI SDK 调用（chat / embedding）自动捕获为 OTel Span，包含 token 用量和延迟
- **手动埋点**：Council 辩论、Scout 抓取、Daily Session 的关键阶段标注为层级 Span
- **可视化面板**：启用后访问 `http://localhost:6006` 查看完整调用树、成本分析、延迟分布
- **零开销默认关闭**：通过 `TRACING_ENABLED=true` 环境变量一键开启，关闭时 span 为 OTel no-op
- **数据不出本地**：Phoenix 以进程内方式运行，所有 trace 数据存储在本地，保护隐私

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repo-url>
cd mindpalace

# 安装依赖
pip install -e .

# 或安装开发依赖
pip install -e ".[dev]"

# 可选：安装可观测性（LLMOps）依赖
pip install -e ".[obs]"
```

### 启动

**推荐：交互式菜单模式** 🎯

```bash
# 直接运行，进入交互式菜单
python -m src

# 或显式指定交互模式
python -m src -i
```

使用方向键 ↑↓ 选择功能，按 Enter 确认。无需记忆命令！

![Interactive Menu](https://via.placeholder.com/600x400?text=Interactive+Menu+Screenshot)

**传统命令行模式**

如果你更喜欢传统 CLI，所有命令仍然可用：

```bash
python -m src scout
python -m src list
python -m src council --item 1
# ... 等等
```

### 配置

**方式1：交互式配置（推荐）**

```bash
python -m src
# 选择 "⚙️ Config - 配置 API"
```

**方式2：命令行配置**

```bash
python -m src config
```

**方式3：手动编辑 .env**

```env
# 全局默认配置（所有任务的兜底）
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAMES=gpt-4o-mini,gpt-4o

# 可选：为不同任务分档配置
SCOUT_MODEL_NAMES=deepseek-chat           # Scout 粗筛
COUNCIL_MODEL_NAMES=deepseek-chat         # Council 辩论主力
FAST_MODEL_NAMES=deepseek-chat            # 路由/画像/压缩等轻量任务
JUDGE_MODEL_NAMES=claude-3-5-sonnet-20241022,deepseek-reasoner  # Judge，最强档
EMBEDDING_MODEL_NAMES=text-embedding-3-small # 文本向量化模型

# 控制参数
MAX_REBUTTAL_ROUNDS=3                     # 最大反驳轮数（含强制落地）
CONVERGE_THRESHOLD=0.3                    # 分歧度低于此值提前结束
CRYSTAL_WINDOW=10                         # 每累计 10 条发言做一次画像结晶
```

**模型分档策略**

每档都有独立的 `*_API_KEY` / `*_BASE_URL` / `*_MODEL_NAMES`，未配置则按下表回落：

| 档位 | 职责 | 推荐模型 | 回落顺序 |
|---|---|---|---|
| `JUDGE_*` | 辩论最终共识（关键质量守门） | Claude-3.5-Sonnet / DeepSeek-Reasoner | → `COUNCIL_*` → `OPENAI_*` |
| `COUNCIL_*` | Critic / Synthesizer / Mentor 发言 | DeepSeek-Chat / GPT-4o-mini | → `OPENAI_*` |
| `ROUTER_*` | 难度路由、midcheck 分歧度 | DeepSeek-Chat / Qwen-Flash | → `FAST_*` → `OPENAI_*` |
| `FAST_*` | 记忆压缩、轨迹分析总结 | DeepSeek-Chat / Qwen-Flash | → `OPENAI_*` |
| `SCOUT_*` | 文章启发性评分 | DeepSeek-Chat / Qwen-Flash | → `OPENAI_*` |
| `MEMORY_*` | Echo 对比 / Profiler 画像 | DeepSeek-Chat | → `OPENAI_*` |
| `EMBEDDING_*`| 文本向量化 | text-embedding-3-small | → `OPENAI_*` |

> **设计目的**：Judge 必须用一流模型（辩论的质量守门员），而 Router/midcheck 可以用便宜小模型（只判断"要不要再来一轮"）。这样整场辩论成本可控，同时最终结论质量不打折。

### 使用

```bash
# 抓取并评分高质量内容
python -m src scout

# 查看已保存的文章列表
python -m src list

# 查看文章完整内容
python -m src view --item 1

# 生成文章导读精炼版（快速了解核心内容）
python -m src brief --item 1

# 对某篇文章发起议事厅讨论
python -m src council --item 1

# 查看你的认知进化历史
python -m src reflect

# 一键运行完整流程：抓取 → 讨论 → 记录观点 → 回声定位
python -m src daily

# 进入交互式对话空间
python -m src resolve              # 与整个议事厅对话
python -m src resolve --role mentor # 与特定角色对话
python -m src resolve --list       # 查看历史会话并恢复
python -m src resolve --session <id> # 恢复特定会话

# 周度评估报告（LLM-as-a-Judge）
python -m src eval                 # 评估最近 7 天的讨论
python -m src eval --days 14       # 自定义评估周期
python -m src eval --iterate       # 同时生成 Prompt 改进建议

# 启用 LLMOps 可观测性（启用后打开 http://localhost:6006 查看）
TRACING_ENABLED=true python -m src daily
```

## 项目结构

```
mindpalace/
├── src/
│   ├── app.py              # CLI 入口
│   ├── config.py           # 配置管理
│   ├── scout/              # 信息抓取与评分
│   │   ├── fetch.py        # RSS 抓取
│   │   ├── normalize.py    # 内容清洗
│   │   ├── score.py        # 启发性评分
│   │   └── pipeline.py     # 流水线编排
│   ├── council/            # 多智能体辩论（状态机 + Self-RAG）
│   │   ├── state.py        # DebateState / Turn / Phase 数据模型
│   │   ├── router.py       # 难度路由（派 1/2/3 个角色）
│   │   ├── rebuttal.py     # opening / rebuttal prompt 构造
│   │   ├── judge.py        # The Judge：midcheck + finalize
│   │   ├── flow.py         # 状态机主循环（run_council + tool-use）
│   │   ├── roles.py        # 角色定义 + 工具权限声明
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
│   │   └── fact_check.py   # 事实核查（搜索 + LLM 判定）
│   ├── eval/               # 评估闭环
│   │   ├── feedback.py     # 用户反馈收集与存储
│   │   ├── judge_debates.py # LLM-as-a-Judge 周度评分
│   │   └── prompt_iterator.py # Prompt 改进建议生成
│   ├── resolve/            # 交互式对话（自动压缩）
│   │   └── engine.py       # REPL 引擎 + history 压缩
│   ├── workflows/          # 端到端流程
│   │   └── daily_session.py # 拓取→讨论→记忆→回声→结晶→反馈
│   ├── llm/                # LLM 调用封装
│   │   └── client.py       # chat / chat_json / chat_with_tools
│   ├── obs/                # 可观测性 (LLMOps)
│   │   └── tracing.py      # OTel + Phoenix 初始化 + span 封装
│   └── storage/            # 数据持久化
│       └── db.py           # SQLite DDL + CRUD
├── data/
│   ├── personas/           # 自定义角色定义
│   ├── user_profile.md     # 结晶累计的用户画像
│   └── library/            # 本地知识库
├── eval/                   # 周度评估报告输出目录
├── tests/                  # 测试用例 (77+)
├── .env                    # 本地配置
└── pyproject.toml
```

## 使用示例

### 方式1：交互式菜单（推荐新手）

```bash
# 启动交互式菜单
python -m src
```

然后使用方向键选择：
1. 🎯 **Scout** - 抓取内容
2. 📚 **Browse** - 浏览文章
   - 从列表选择文章
   - 生成导读/查看完整内容/发起讨论
3. 🧠 **Memory** - 查看认知历史
4. 💬 **Resolve** - 交互式对话

### 方式2：命令行模式（适合脚本化）

```bash
# 典型工作流
python -m src scout
python -m src list
python -m src brief --item 1
python -m src council --item 1
python -m src reflect
```

### 方式3：一键体验

```bash
# 运行完整流程
python -m src daily
```

## 会话管理 (Resolve Engine)

Resolve 模块提供一个支持长程对话的交互式 REPL，所有会话都会自动保存到数据库。

**历史自动压缩**：当对话轮数过多时（默认 > 40 条消息），引擎会自动使用 Fast 模型将早期的对话压缩成紧凑的摘要上下文，彻底解决长对话 Token 爆炸的问题，实现无限轮次交流。

### 查看历史会话

```bash
python -m src resolve --list
```

显示所有历史会话，可以选择恢复继续讨论。

### 恢复特定会话

```bash
# 通过会话ID恢复
python -m src resolve --session abc12345

# 或使用 --list 交互式选择
python -m src resolve --list
```

### 删除会话

```bash
python -m src resolve --delete abc12345
```

### 会话模式

- **议事厅模式**（默认）：与 Critic、Synthesizer、Mentor 三个角色同时对话
- **单角色模式**：与特定角色深入对话

```bash
# 与苏格拉底导师深入探讨
python -m src resolve --role mentor

# 与理性批判者辩论
python -m src resolve --role critic
```

## 自定义角色

在 `data/personas/` 目录下创建 `.md` 文件即可添加自定义角色：

```markdown
# The Historian (历史学家)

你是一个专注于历史视角的分析者...

你的核心使命是：从历史中寻找相似的模式和教训...

输出格式（JSON）：
{
  "historical_parallels": [...],
  "lessons": "..."
}
```

## 技术栈

- **语言**：Python 3.11+
- **LLM**：OpenAI 兼容 API（支持 OpenAI、DeepSeek、Claude、Gemini 等）
- **存储**：SQLite（轻量零配置）
- **RSS 解析**：feedparser

## 设计理念

### 从"工具"到"教练"的转变

大多数项目是"搜索增强 (RAG)"，本项目是**"思考增强 (Thinking Augmented)"**。它挑战用户，而不是顺从用户。

### 主动式交互

区别于"用户问-AI答"的传统模式，MindPalace 由 Agent 根据信息源主动发起对话，引导用户深入思考。

### 复杂状态管理

Council 不是简单的顺序流水线，而是基于 `DebateState` 的有限状态机：

- **阶段显式化**：`ROUTING / OPENING / REBUTTAL / JUDGING / DONE` 五阶段转移，每个 `Turn` 携带 phase / round_idx / force_closing，完整可复现。
- **双重收敛保证**：midcheck 看分歧度决定早停，硬性 `max_rebuttal_rounds` 兜底；最后一轮注入"禁止新论点"的 system 级约束，杜绝死循环。
- **失败降级**：Router 失败 → 回退 medium 档；Judge finalize 失败 → 返回带 error 的降级 consensus。主流程永远能跑完。
- **完整落库**：每次辩论写入 `debates` 表（turns / consensus / terminated_by / disagreement_score），为评估闭环和 Prompt 迭代提供原始数据。

## 开发路线

**已完成**

- [x] Phase 1: Scout MVP - 信息筛选闭环
- [x] Phase 2: Council MVP - 三角色顺序讨论
- [x] Phase 3: Memory MVP - 回声定位
- [x] Phase 4: 端到端演示闭环
- [x] **Phase A: Council 状态机重构** - 难度路由 + Judge 共识收敛 + 强制落地防死循环
- [x] **Phase B: 向量化记忆 + 认知固化** - Numpy Embedding 语义召回 + 自动画像结晶 + 月度质心轨迹 + 长对话压缩
- [x] **Phase C: Self-RAG + 评估闭环** - Critic/Synthesizer 主动 web_search/fact_check + 用户反馈收集 + LLM-as-a-Judge 周度评分 + Prompt 迭代建议
- [x] **LLMOps 可观测性** - OpenTelemetry + Arize Phoenix 纯本地链路追踪，OpenAI SDK 自动埋点 + 关键管道手动埋点

**规划中**

- [ ] Phase 5: 更多信息源（微信公众号、播客等）
- [ ] Phase 7: Web 前端

> 详细设计见 [`FUSION_PLAN.md`](FUSION_PLAN.md)。

## 许可证

MIT
