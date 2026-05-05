# 智能客服系统

基于 AI 的智能客服系统，采用前后端分离架构，支持知识库检索（RAG）、工具调用和多轮对话。

## 特色功能

- **智能知识库**: 基于向量数据库的语义检索，支持 .txt、.pdf 等文档格式
- **RAG 增强检索**: 检索增强生成技术，提升 AI 回答的准确性和相关性
- **上下文管理**: 独立会话管理，支持多轮对话和上下文记忆
- **工具调用**: AI 自动判断并调用外部工具（天气查询、天气推荐、表单提交），支持链式调用
- **前后端分离**: React + Vite 前端 + Flask 后端
- **模块化设计**: 清晰的后端架构，易于扩展

## 项目结构

```
LangchainChatFlow/
├── backend/                    # Python 后端 (Flask)
│   ├── app.py                 # Flask 主应用入口
│   ├── config.json            # 系统配置
│   ├── requirements.txt       # Python 依赖
│   ├── modules/               # 核心功能模块
│   │   ├── ai_client.py       # AI 客户端（兼容 OpenAI SDK）
│   │   ├── assistant.py       # AI 助手/Agent
│   │   ├── store/             # 存储模块
│   │   │   └── vector_store.py # 向量数据库管理
│   │   ├── prompt/            # Prompt 模板管理
│   │   └── tools/              # 工具插件
│   │       ├── weather_plugin.py         # 天气查询
│   │       ├── weather_recommend_plugin.py # 天气推荐
│   │       └── submit_form_plugin.py     # 表单提交
│   ├── knowledge_base/        # 知识库文档目录
│   └── db/                    # 向量数据库存储目录
│
└── frontend/                   # React 前端 (Vite)
    ├── src/
    │   ├── components/         # UI 组件
    │   │   ├── ChatArea.jsx    # 聊天区域
    │   │   ├── Header.jsx      # 顶部标题栏
    │   │   └── InputArea.jsx   # 输入区域
    │   └── api/
    │       └── chat.js         # API 调用
    └── package.json
```

## 快速开始

### 环境要求

- Python >= 3.8
- Node.js >= 16
- npm 或 yarn
- 阿里云百炼 API 密钥（或 OpenAI API）

### 后端启动

```bash
# 进入后端目录
cd backend

# 安装依赖
pip install -r requirements.txt

# 配置 API 密钥
# 编辑 config.json 文件，填入你的 API Key

# 启动服务
python app.py
```

后端运行在: `http://localhost:5000`

### 前端启动

```bash
# 新开终端，进入前端目录
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端运行在: `http://localhost:5173`

### 访问应用

浏览器打开: `http://localhost:5173`

## 配置说明

### config.json

```json
{
  "api_key": "your_api_key",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "model": "qwen-plus",
  "embedding_model": "text-embedding-v3"
}
```

| 配置项 | 说明 |
|--------|------|
| api_key | 阿里云百炼或 OpenAI API 密钥 |
| base_url | API 基础地址 |
| model | 对话模型名称 |
| embedding_model | 向量化模型名称 |

## API 接口

### GET /start

检查服务状态。

**响应示例:**
```json
{
  "status": "ready",
  "message": "客服系统已就绪 (LangChain)",
  "model": "qwen-plus",
  "knowledge_base": true
}
```

### POST /chat

处理对话请求。

**请求:**
```json
{
  "message": "用户输入",
  "session_id": "可选会话ID"
}
```

**响应:**
```json
{
  "reply": "AI回复内容",
  "tool_calls": [],
  "session_id": "会话ID",
  "finished": false
}
```

## 使用流程

1. **准备知识库**: 在 `backend/knowledge_base/` 目录添加文档（支持 .txt、.pdf）
2. **启动服务**: 按上述步骤启动前后端
3. **开始对话**: 访问前端地址，与智能客服对话
4. **RAG 增强**: 系统自动从知识库检索相关内容，增强 AI 回答
5. **工具调用**: AI 自动调用天气查询、推荐或表单提交等工具

## 自定义扩展

### 修改系统提示词

编辑 `backend/modules/prompt/__init__.py` 中的 `CUSTOMER_SERVICE_PROMPT_TEMPLATE` 变量

### 添加新工具

1. 在 `backend/modules/tools/` 创建插件文件，继承 `BaseTool` 基类
2. 定义工具的名称、描述和执行逻辑
3. 在插件文件末尾导出工具实例
4. 在 `backend/modules/tools/__init__.py` 中注册工具
5. 重启服务

### 更新知识库

1. 在 `backend/knowledge_base/` 添加或修改文档
2. 重启服务，系统自动重新向量化

## 技术栈

**后端:**
- Flask 3.0.0 - Web 框架
- OpenAI SDK 1.12.0 - AI API 客户端（兼容阿里云百炼）
- Flask-CORS - 跨域支持
- numpy 2.4.4 - 数值计算

**前端:**
- React 18 - UI 框架
- Vite - 构建工具
- Axios - HTTP 请求

**AI 服务:**
- 阿里云百炼平台 / OpenAI
- qwen-plus 模型
- text-embedding-v3 向量化模型

## 后续优化

- [x] 缓存机制
- [x] 多轮对话优化
- [ ] 支持更多向量数据库
- [ ] 数据库替代 JSON 存储
- [ ] API 限流和安全验证
- [ ] Docker 容器化部署
