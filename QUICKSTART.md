# MindPalace 快速上手指南

## 5 分钟快速体验

### 1. 安装依赖

```bash
pip install -e .
```

### 2. 配置 API（首次使用）

```bash
python -m src
```

选择 `⚙️ 设置` → `配置 API / 模型`，输入你的 API Key。保存后可以立即检测是否可用。

### 3. 开始使用

```bash
python -m src
```

默认 Scout 内容源现在偏向人文社科。如果你想临时切到混合或科技预设，可以直接运行：

```bash
python -m src scout --preset mixed
python -m src scout --preset tech
```

## 交互式菜单说明

启动后你会看到：

```
  __  __ _           _ _____      _                 
 |  \/  (_)         | |  __ \    | |                
 | \  / |_ _ __   __| | |__) |_ _| | __ _  ___ ___  
 | |\/| | | '_ \ / _` |  ___/ _` | |/ _` |/ __/ _ \ 
 | |  | | | | | | (_| | |  | (_| | | (_| | (_|  __/ 
 |_|  |_|_|_| |_|\__,_|_|   \__,_|_|\__,_|\___\___| 
                                                     
  你的私人认知进化实验室

? 请选择功能： (Use arrow keys)
 » 🚀 今日练习
   📚 文章库
   💬 深度对话
   🧠 认知回顾
   ⚙️  设置
   ───────────────────────────────
   ❌ 退出
```

### 操作方式

- **↑↓ 方向键**：上下移动选择
- **Enter**：确认选择
- **Ctrl+C**：取消当前操作，返回菜单

## 推荐使用流程

### 第一次使用

1. **配置 API** → 选择 `⚙️ 设置`
2. **抓取内容** → 选择 `📚 文章库` → `🎯 发现新文章`
3. **浏览文章** → 选择 `📚 文章库` → `📚 浏览文章`
   - 从列表中选择感兴趣的文章
   - 选择操作：
     - 📖 生成导读精炼版（快速了解）
     - 🌐 查看原文（浏览器打开）
     - 🏛️ 发起议事厅讨论（深度思考）
     - ⭐ 收藏高价值文章
4. **查看历史** → 选择 `🧠 认知回顾`

### 日常使用

**方式1：浏览模式（推荐）**

选择 `📚 文章库` → `📚 浏览文章`：
1. 从文章列表中选择感兴趣的文章
2. 先看导读了解核心内容
3. 需要时在浏览器中查看原文
4. 准备好后发起讨论

**方式2：一键流程**

直接选择 `🚀 今日练习`，一键完成：
- 抓取最新高质量内容
- 自动选择最佳文章
- 发起议事厅讨论
- 记录你的观点
- 生成认知对比报告

## 常见问题

### Q: 如何退出交互式菜单？
A: 选择 `❌ 退出`，或按 `Ctrl+C`

### Q: 如何使用传统命令行？
A: 直接运行命令，例如 `python -m src scout`

### Q: 如何检测 API 配置是否可用？
A: 在菜单里进入 `设置` → `检测 API 是否可用`。命令行也可以用：
`python -m src config --test --provider global`
检测 Scout 档：
`python -m src config --test --provider scout`
全部检测：
`python -m src config --test --provider all`

### Q: 想多看人文社科，少看科技文章怎么办？
A: 默认预设已经是 `humanities`。你也可以在 `.env` 里设置
`SCOUT_FEED_PRESET=humanities`
如果想临时切换，使用
`python -m src scout --preset humanities|mixed|tech`

### Q: 如何收藏高价值文章？
A: 在菜单里进入 `Browse`，选中文章后选择收藏。命令行也可以用：
`python -m src favorite --item 1`
查看收藏夹：
`python -m src favorites`

### Q: 旧文章会怎么清理？
A: 默认每次 Scout 后会清理 30 天前的普通文章；收藏文章、讨论过的文章、留下记忆的文章会保留。你可以先预览：
`python -m src cleanup --dry-run`

### Q: 如何恢复之前的对话？
A: 选择 `💬 Resolve` → `📜 查看并恢复历史会话`

### Q: 支持哪些 LLM？
A: 所有 OpenAI 兼容 API，包括：
- OpenAI (GPT-4, GPT-4o)
- DeepSeek
- Claude (通过兼容层)
- Gemini (通过兼容层)
- 本地 Ollama

### Q: 如何为不同任务配置不同模型？
A: 在配置时选择：
- **Global Default**：全局默认
- **Scout**：评分任务（推荐 DeepSeek，性价比高）
- **Council**：讨论任务（推荐 Claude/Gemini，逻辑强）
- **Memory**：认知分析（默认跟随 Council）

### Q: 遇到 "Your request was blocked" 错误怎么办？
A: 这是 API 的内容安全过滤机制触发。解决方案：
1. **切换模型**：某些模型对内容更敏感，尝试使用其他模型
2. **检查内容**：文章可能包含敏感话题
3. **使用 DeepSeek/OpenAI**：这些模型的内容策略相对宽松
4. **直接查看原文**：跳过导读，直接在浏览器中阅读

## 下一步

- 阅读 [README.md](README.md) 了解完整功能
- 查看 [data/personas/](data/personas/) 自定义角色
- 探索 `python -m src --help` 查看所有命令

祝你在 MindPalace 中享受认知进化之旅！🧠✨
