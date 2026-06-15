# MindPalace Agent 项目结构

> 完整的目录结构与模块说明

---

## 目录树

```
mindpalace/
├── .claude/                        # Claude 配置
│   └── settings.local.json
├── .pytest_cache/                  # Pytest 缓存
├── .vscode/                        # VSCode 配置
├── data/                           # 数据目录
│   ├── learn/                      # 学习资料
│   ├── library/                    # 本地知识库
│   ├── personas/                   # 自定义角色定义
│   │   └── .keep
│   └── user_profile.md.example     # 用户画像模板
├── eval/                           # 评估报告输出目录
├── src/                            # 源代码
│   ├── __init__.py
│   ├── __main__.py                 # 程序入口
│   ├── app.py                      # CLI 主程序 + 交互式菜单
│   ├── config.py                   # 配置管理（分层 Provider）
│   │
│   ├── council/                    # 多智能体辩论系统
│   │   ├── __init__.py
│   │   ├── flow.py                 # 状态机主循环（范式调度入口）
│   │   ├── state.py                # DebateState / Turn / Phase 数据模型
│   │   ├── router.py               # 难度路由（动态派单）
│   │   ├── judge.py                # Judge 角色（midcheck + finalize）
│   │   ├── rebuttal.py             # Prompt 构造（opening / rebuttal）
│   │   ├── roles.py                # 角色定义 + 工具权限
│   │   ├── paradigms.py            # 讨论范式（Debate / Report）— 借鉴 MALLM
│   │   ├── registry.py             # 范式注册表（字符串 → 类映射）
│   │   ├── protocols.py            # 收敛协议（midcheck/consensus/voting）— 借鉴 MALLM
│   │   ├── protocol_registry.py    # 协议注册表（字符串 → 类映射）
│   │   └── output.py               # 结果格式化
│   │
│   ├── eval/                       # 评估闭环
│   │   ├── __init__.py
│   │   ├── feedback.py             # 用户反馈收集
│   │   ├── judge_debates.py        # LLM-as-a-Judge 周度评分
│   │   └── prompt_iterator.py      # Prompt 改进建议生成
│   │
│   ├── llm/                        # LLM 调用封装
│   │   ├── __init__.py
│   │   └── client.py               # chat / chat_json / chat_with_tools
│   │
│   ├── memory/                     # 认知记忆系统
│   │   ├── __init__.py
│   │   ├── store.py                # 记忆存储（A-MEM 增强嵌入 + 向量召回 + agentic 链接遍历）
│   │   ├── embedder.py             # Embedding 抽象 + OpenAI 实现 + 增强文本构建
│   │   ├── evolution.py            # 记忆演化引擎（A-MEM link_memories）
│   │   ├── profiler.py             # 认知画像分析
│   │   ├── echo.py                 # 回声定位（历史对比）
│   │   ├── crystallize.py          # 结构化认知结晶（Axiomind 知识金字塔 + 晋升启发式）
│   │   ├── brain_export.py         # 认知档案 Markdown 导出（brain/ 目录）
│   │   └── trajectory.py           # 月度质心漂移分析
│   │
│   ├── obs/                        # 可观测性（LLMOps）
│   │   ├── __init__.py
│   │   └── tracing.py              # OpenTelemetry + Phoenix 追踪
│   │
│   ├── resolve/                    # 交互式对话引擎
│   │   └── engine.py               # REPL + 历史压缩
│   │
│   ├── scout/                      # 信息抓取与评分
│   │   ├── fetch.py                # RSS 抓取
│   │   ├── normalize.py            # 内容清洗
│   │   ├── score.py                # 启发性评分
│   │   └── pipeline.py             # 流水线编排
│   │
│   ├── storage/                    # 数据持久化
│   │   ├── __init__.py
│   │   └── db.py                   # SQLite DDL + CRUD
│   │
│   ├── tools/                      # Self-RAG 工具系统
│   │   ├── __init__.py
│   │   ├── base.py                 # Tool Protocol + 注册表
│   │   ├── web_search.py           # DuckDuckGo 搜索
│   │   └── fact_check.py           # 事实核查
│
│   ├── inquiry/                    # 心智漫游（内省式问题卡）
│   │   ├── __init__.py
│   │   ├── types.py                # PromptCard 数据模型
│   │   ├── library.py              # 卡组加载（data/inquiry/*.json）
│   │   ├── analysis.py             # 回答分析（LLM 提炼）
│   │   ├── diff.py                 # 基线 diff（Axiomind 暂存区 + 相似历史检测）
│   │   ├── session.py              # 单次会话流程（含相似提示 + 演化链接）
│   │   └── cli.py                  # 交互式子菜单（含演化轨迹展示）
│   │
│   └── workflows/                  # 端到端流程
│       ├── __init__.py
│       └── daily_session.py        # 完整流程编排
│
├── tests/                          # 测试用例
│   ├── __init__.py
│   ├── test_council_flow.py        # 状态机流程测试
│   ├── test_llm_robustness.py      # LLM 容错测试
│   ├── test_memory.py              # 记忆存储测试
│   ├── test_memory_embedder.py     # 向量召回测试
│   ├── test_obs.py                 # 可观测性测试
│   ├── test_phase_c.py             # Self-RAG 测试
│   └── test_scout_pipeline.py      # Scout 流水线测试
│
├── .env                            # 环境变量配置（本地）
├── .env.example                    # 环境变量模板
├── .gitignore                      # Git 忽略规则
├── akashic-agent.code-workspace    # VSCode 工作区配置
├── CLAUDE.md                       # Claude 编码规范
├── FUSION_PLAN.md                  # 融合设计文档
├── introduction.md                 # 技术深度解析
├── interview_qa.md                 # 面试问答模拟
├── mindpalace.db                   # SQLite 数据库
├── PROJECT_STRUCTURE.md            # 本文件
├── pyproject.toml                  # Python 项目配置
├── QUICKSTART.md                   # 快速上手指南
├── README.md                       # 项目说明
├── test_pipeline.py                # 管道测试脚本
├── 重点方向参考.md                  # 面试维度参考
├── 项目简述.txt                     # 项目简介
└── src/mindpalace.egg-info/        # 包信息
    ├── dependency_links.txt
    ├── PKG-INFO
    ├── requires.txt
    ├── SOURCES.txt
    └── top_level.txt
```

---

## 核心模块说明

### 1. Council（多智能体辩论）

**职责**：管理多个 Agent 的对抗式辩论流程

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `state.py` | 数据模型 | `DebateState`, `Turn`, `Phase` |
| `flow.py` | 状态机主循环（范式调度） | `run_council()` |
| `router.py` | 难度路由 | `route()` |
| `judge.py` | Judge 角色 | `midcheck()`, `finalize()` |
| `rebuttal.py` | Prompt 构造 | `build_opening_prompt()`, `build_rebuttal_prompt()` |
| `roles.py` | 角色定义 | `get_role()`, `TOOL_ENABLED_ROLES` |
| `paradigms.py` | 讨论范式 | `DiscussionParadigm`, `DebateParadigm`, `ReportParadigm` |
| `registry.py` | 范式注册表 | `PARADIGMS`, `register_paradigm()`, `get_paradigm()` |
| `output.py` | 结果格式化 | `format_council_result()` |

**核心流程**（默认 Debate 范式）：
```
ROUTING → OPENING → REBUTTAL(循环) → JUDGING → DONE
```

**讨论范式**（借鉴 MALLM，通过注册表可插拔）：
- `debate`（默认）：对抗式多轮反驳，midcheck 收敛
- `report`：中心化起草（主起草人生成报告）+ 其他人单轮审阅

**收敛协议**（借鉴 MALLM decision_protocols，仅 debate 范式生效）：
- `midcheck`（默认）：LLM 判断 should_continue
- `consensus_threshold`：分歧度 < CONVERGE_THRESHOLD 即收敛
- `voting`：评估各方立场一致程度

```bash
python -m src council --item 1                                  # 默认 debate + midcheck
python -m src council --item 1 --paradigm report                # Report 范式
python -m src council --item 1 --protocol voting                # 投票收敛协议
python -m src council --item 1 --protocol consensus_threshold   # 共识阈值
```

---

### 2. Memory（认知记忆）

**职责**：长期记忆存储、向量召回、认知画像

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `store.py` | 记忆存储（A-MEM 增强嵌入 + 链接/访问统计） | `save_memory(link_after_save=)`, `find_related_memories(agentic=)`, `get_memory()`, `update_memory_links()` |
| `embedder.py` | 向量化 + 增强文本构建 | `Embedder`, `OpenAIEmbedder`, `build_enhanced_text()`, `cosine_similarity()` |
| `evolution.py` | 记忆演化引擎（A-MEM link_memories） | `link_memories()` — 找邻居→LLM决策→strengthen/update_neighbor |
| `profiler.py` | 认知画像 | `profile_response()`, `CognitiveProfile` |
| `echo.py` | 回声定位 | `generate_echo_report()`, `EchoReport` |
| `crystallize.py` | 结构化认知结晶（Axiomind 知识金字塔 + 晋升启发式） | `crystallize_if_needed()`, `render_crystal_terminal()` |
| `brain_export.py` | 认知档案 Markdown 导出 | `export_brain()` |
| `trajectory.py` | 轨迹分析 | `compute_trajectory()` |

**A-MEM 增强嵌入**：存储与查询都嵌入拼接后的 `content + stance + keywords + preferences`，
而非原始内容，显著提升召回质量。`rebuild_embeddings()` 可迁移存量记录。

**A-MEM 记忆演化**：每次保存新记忆可触发 `link_memories()`——找 5 个最近邻居 →
LLM 决定 `should_evolve` → `strengthen`（建链接）或 `update_neighbor`（更新邻居标签）。
`find_related_memories(agentic=True)` 支持沿 `links` 遍历扩展召回。`retrieval_count` 记录被召回次数。

**Axiomind 知识金字塔**（结构化结晶输出）：
```
Layer 1: 原始记忆 (memories 表)
Layer 2: 结构化洞察 (profile_crystals 表 + user_profile.md)
         ├─ observation  日常观察模式（candidate 状态）
         ├─ principle    可复用行动规则（candidate 状态）
         └─ axiom        身份级深层信念（candidate 状态，需人类激活）
Layer 3: 认知轨迹 (月度质心漂移)
```

`export_brain()` 将结构化洞察导出为 `data/brain/{axioms,principles,observations}/*.md`，
带 YAML frontmatter，可被 Obsidian 或未来 agent 直接读取。

**三层架构**：
```
Layer 1: 原始记忆 (memories 表)
Layer 2: 画像结晶 (profile_crystals 表 + user_profile.md)
Layer 3: 认知轨迹 (月度质心漂移)
```

---

### 3. Tools（Self-RAG 工具）

**职责**：Agent 可调用的外部工具

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `base.py` | 工具协议 | `Tool`, `register()`, `to_openai_schema()` |
| `web_search.py` | 网络搜索 | `WebSearchTool` |
| `fact_check.py` | 事实核查 | `FactCheckTool` |

**工具权限**：
- Critic: ✅ 可调用工具
- Synthesizer: ✅ 可调用工具
- Mentor: ❌ 不调用工具（只追问）

---

### 4. Eval（评估闭环）

**职责**：用户反馈收集、LLM-as-a-Judge 评分、Prompt 迭代

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `feedback.py` | 反馈收集 | `save_feedback()`, `get_feedback_distribution()` |
| `judge_debates.py` | 周度评分 | `judge_recent_debates()`, `generate_weekly_report()` |
| `prompt_iterator.py` | Prompt 迭代 | `generate_iteration_suggestions()` |

**闭环流程**：
```
辩论 → 用户反馈 → LLM 评分 → 聚合弱点 → 生成改进建议 → 更新 Prompt
```

---

### 5. Scout（信息抓取）

**职责**：RSS 抓取、内容清洗、启发性评分

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `fetch.py` | RSS 抓取 | `fetch_rss()` |
| `normalize.py` | 内容清洗 | `normalize_content()` |
| `score.py` | 启发性评分 | `score_article()` |
| `pipeline.py` | 流水线编排 | `run_scout()` |

**评分维度**：
- 信息密度（information_density）
- 原理深度（principle_depth）
- 因果链长度（causal_chain）

---

### 6. LLM（调用封装）

**职责**：统一 LLM 调用接口、容错、降级

| 文件 | 功能 | 关键函数 |
|------|------|----------|
| `client.py` | LLM 调用 | `chat()`, `chat_json()`, `chat_with_tools()` |

**容错机制**：
- 多模型轮换
- 指数退避重试
- 内容过滤跳过
- 连接错误降级

---

### 7. Storage（数据持久化）

**职责**：SQLite 数据库管理

| 文件 | 功能 | 关键函数 |
|------|------|----------|
| `db.py` | 数据库操作 | `init_db()`, `save_debate()`, `get_article()` |

**核心表**：
```sql
articles          -- 文章存储
debates           -- 辩论记录
memories          -- 用户记忆
profile_crystals  -- 画像结晶
feedback          -- 用户反馈
sessions          -- 对话会话
```

---

### 8. Obs（可观测性）

**职责**：OpenTelemetry + Phoenix 链路追踪

| 文件 | 功能 | 关键函数 |
|------|------|----------|
| `tracing.py` | 追踪初始化 | `init_tracing()`, `span()` |

**追踪内容**：
- OpenAI SDK 自动埋点
- Council 辩论手动埋点
- Token 用量和延迟

---

### 9. Workflows（端到端流程）

**职责**：编排完整业务流程

| 文件 | 功能 | 关键函数 |
|------|------|----------|
| `daily_session.py` | 每日流程 | `run_daily_session()` |

**流程**：
```
Scout 抓取 → 选择文章 → Council 辩论 → 用户回应 → 
认知画像 → 保存记忆 → Echo 对比 → 认知固化 → 反馈收集
```

---

### 10. Resolve（交互式对话）

**职责**：REPL 对话引擎、历史压缩

| 文件 | 功能 | 关键函数 |
|------|------|----------|
| `engine.py` | 对话引擎 | `run_repl()`, `list_sessions()` |

**特性**：
- 无限轮次对话
- 自动历史压缩（>40 条消息）
- 会话持久化

---

### 11. Inquiry（心智漫游）

**职责**：内省式问题卡驱动自我反思，回答写入长期记忆

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `types.py` | 卡片数据模型 | `PromptCard`, `PromptCard.from_dict()` |
| `library.py` | 卡组加载 | `load_cards()`, `get_card()`, `choose_random_card()` |
| `analysis.py` | 回答分析 | `analyze_response()` |
| `session.py` | 单次会话流程 | `run_inquiry_session()`, `save_inquiry_memory()` |
| `cli.py` | 交互式子菜单 | `run_inquiry_menu()` |

**卡组**（`data/inquiry/*.json`，源文本归档于 `data/inquiry/sources/`）：
- `self.json` — 认识自己（36 张卡）
- `philosophy.json` — 哲思问题（12 张卡）
- `thought_experiments.json` — 思想实验（7 张卡，含 context/followups/twists）

**流程**：
```
选卡 → 展示问题 → 用户多行回答 → LLM 提炼心智镜像 → 写入 memories 表（带 source_type/source_id）
```

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                         数据流向                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  RSS/网页                                                        │
│     │                                                            │
│     ▼                                                            │
│  Scout (fetch → normalize → score)                              │
│     │                                                            │
│     ▼                                                            │
│  articles 表                                                     │
│     │                                                            │
│     ▼                                                            │
│  Council (router → opening → rebuttal → judge)                  │
│     │                                                            │
│     ├──▶ debates 表                                             │
│     │                                                            │
│     ▼                                                            │
│  用户回应                                                         │
│     │                                                            │
│     ▼                                                            │
│  Memory (profiler → embedder → store)                           │
│     │                                                            │
│     ├──▶ memories 表                                            │
│     │                                                            │
│     ├──▶ Echo (向量召回 → 对比报告)                              │
│     │                                                            │
│     └──▶ Crystallize (压缩 → profile_crystals 表 → user_profile.md) │
│                                                                  │
│  用户反馈                                                         │
│     │                                                            │
│     ▼                                                            │
│  feedback 表                                                     │
│     │                                                            │
│     ▼                                                            │
│  Eval (judge_debates → weekly_report → prompt_iterator)         │
│     │                                                            │
│     └──▶ 优化后的 Prompt                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 配置文件说明

### `.env` 环境变量

```env
# 全局默认配置
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAMES=gpt-4o-mini,gpt-4o

# 任务分档配置
SCOUT_MODEL_NAMES=deepseek-chat           # Scout 评分
COUNCIL_MODEL_NAMES=deepseek-chat         # Council 辩论
ROUTER_MODEL_NAMES=deepseek-chat          # 难度路由
JUDGE_MODEL_NAMES=claude-3-5-sonnet-20241022  # Judge 收敛
FAST_MODEL_NAMES=deepseek-chat            # 轻量任务
MEMORY_MODEL_NAMES=deepseek-chat          # 认知画像
EMBEDDING_MODEL_NAMES=text-embedding-3-small  # 向量化

# 控制参数
MAX_REBUTTAL_ROUNDS=3                     # 最大反驳轮数
CONVERGE_THRESHOLD=0.3                    # 分歧度阈值 / 共识置信度阈值
COUNCIL_CONVERGENCE_PROTOCOL=midcheck     # 收敛协议：midcheck | consensus_threshold | voting
CRYSTAL_WINDOW=10                         # 结晶窗口
MAX_WORKERS=10                            # 并行抓取数

# 可观测性
TRACING_ENABLED=false                     # 是否启用追踪
```

### `pyproject.toml` 项目配置

```toml
[project]
name = "mindpalace"
version = "0.1.0"
dependencies = [
    "openai>=1.0.0",
    "feedparser",
    "python-dotenv",
    "numpy",
    "questionary",
    "duckduckgo-search",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]
obs = ["opentelemetry-api", "arize-phoenix"]
```

---

## 命令行接口

### 主命令

```bash
python -m src                    # 交互式菜单（推荐）
python -m src -i                 # 显式启动交互模式
python -m src -v                 # 详细日志模式
```

### 子命令

```bash
# Scout
python -m src scout              # 抓取并评分
python -m src scout --top 10     # 返回前 10 篇

# 浏览
python -m src list               # 列出文章
python -m src view --item 1      # 查看完整内容
python -m src brief --item 1     # 生成导读

# Council
python -m src council --item 1   # 发起辩论

# Memory
python -m src reflect            # 查看认知历史

# Resolve
python -m src resolve            # 进入对话
python -m src resolve --role mentor  # 单角色对话
python -m src resolve --list     # 查看历史会话

# Eval
python -m src eval               # 周度评估
python -m src eval --days 14     # 自定义周期
python -m src eval --iterate     # 生成 Prompt 改进

# Daily
python -m src daily              # 一键完整流程

# Config
python -m src config             # 配置 API
```

---

## 测试运行

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_council_flow.py

# 查看覆盖率
pytest --cov=src tests/

# 详细输出
pytest -v -s
```

---

## 开发工作流

### 1. 新增角色

```bash
# 在 data/personas/ 创建 .md 文件
echo "# The Historian\n你是历史学家..." > data/personas/historian.md

# 系统自动加载，无需重启
```

### 2. 新增工具

```python
# src/tools/my_tool.py
class MyTool:
    name = "my_tool"
    description = "..."
    parameters = {...}
    
    def run(self, **kwargs):
        return "result"

register(MyTool())
```

### 3. 调整 Prompt

```python
# 直接修改 src/council/roles.py 中的 PROMPT
CRITIC_SYSTEM_PROMPT = """
你是理性批判者...
[新增内容]
"""
```

### 4. 查看追踪

```bash
# 启用追踪
TRACING_ENABLED=true python -m src daily

# 访问 http://localhost:6006 查看
```

---

## 部署建议

### 本地开发

```bash
pip install -e ".[dev,obs]"
cp .env.example .env
# 编辑 .env 配置 API Key
python -m src
```

### 生产部署

```bash
# 使用 PostgreSQL 替代 SQLite
# 使用 Milvus 替代 numpy 向量
# 使用 Tavily 替代 DuckDuckGo
# 增加 Redis 缓存层
```

---

**项目结构清晰，模块职责明确，易于扩展和维护。**
