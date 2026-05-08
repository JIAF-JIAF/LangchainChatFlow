# 智能客服系统

基于 AI 的智能客服系统，采用前后端分离架构，支持知识库检索（RAG）、工具调用和多轮对话。系统已迁移至 **MCP（Model Context Protocol）** 架构，支持工具的独立部署和多 Agent 共享调用。

## 特色功能

- **智能知识库**: 基于向量数据库的语义检索，支持 .txt、.pdf 等文档格式
- **RAG 增强检索**: 检索增强生成技术，提升 AI 回答的准确性和相关性
- **上下文管理**: 独立会话管理，支持多轮对话和上下文记忆
- **工具调用**: AI 自动判断并调用外部工具（天气查询、天气推荐、表单提交），支持链式调用
- **MCP 架构**: 工具独立部署，支持多个 Agent 共享调用
- **Streamable HTTP**: 基于 HTTP 的流式传输协议，支持实时响应
- **FewShot 示例学习**: 基于 LengthBasedExampleSelector 的动态示例选择
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
│   │   ├── __init__.py       # 模块包初始化
│   │   ├── ai_client.py       # AI 客户端（兼容 OpenAI SDK）
│   │   ├── assistant.py       # AI 助手/Agent
│   │   ├── document_loaders/  # 文档加载器
│   │   │   ├── __init__.py
│   │   │   ├── loader_factory.py
│   │   │   ├── text_loader.py
│   │   │   ├── pdf_loader.py
│   │   │   └── docx_loader.py
│   │   ├── vector_stores/     # 向量存储
│   │   │   ├── __init__.py
│   │   │   ├── base_vector_store.py  # 向量存储基类（含工具方法）
│   │   │   ├── chroma_store.py       # Chroma 实现
│   │   │   ├── milvus_store.py       # Milvus 实现
│   │   │   └── store_factory.py      # 存储工厂
│   │   ├── prompt/            # Prompt 模板管理
│   │   │   └── __init__.py   # 包含 FewShot 和 LengthBasedExampleSelector
│   │   └── rate_limit.py      # 限流模块
│   ├── mcp_module/            # MCP 模块（工具服务）
│   │   ├── __init__.py        # MCP 模块初始化
│   │   ├── config.py          # MCP 配置常量
│   │   ├── logger.py          # 统一日志模块
│   │   ├── mcp_server.py      # MCP 服务器核心
│   │   ├── mcp_client.py      # MCP 客户端
│   │   ├── mcp_service.py     # MCP 服务封装
│   │   ├── start.py           # MCP 服务器启动脚本
│   │   └── tools/             # 工具插件目录
│   │       ├── __init__.py
│   │       ├── registry.py     # 工具注册中心
│   │       ├── weather_plugin.py
│   │       ├── weather_recommend_plugin.py
│   │       └── submit_form_plugin.py
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

- Python >= 3.10
- Node.js >= 16
- npm 或 yarn
- 阿里云百炼 API 密钥（或 OpenAI API）

### 后端启动

**第一步：启动 MCP 服务**

```bash
# 进入后端目录
cd backend

# 安装依赖（首次运行）
pip install -r requirements.txt

# 启动 MCP 服务器（独立部署）
python mcp_module/start.py
```

MCP 服务器运行在: `http://localhost:8080/mcp`

**第二步：启动应用服务**

```bash
# 新开终端，进入后端目录
cd backend

# 配置 API 密钥
# 编辑 config.json 文件，填入你的 API Key
# 支持阿里云百炼 API（推荐）或 OpenAI API

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

### MCP 配置

MCP 服务器配置位于 `backend/mcp_module/config.py`:

```python
# MCP 服务器配置
MCP_HOST = "0.0.0.0"
MCP_PORT = 8080
MCP_PATH = "/mcp"
MCP_URL = f"http://127.0.0.1:{MCP_PORT}{MCP_PATH}"

# 应用服务器配置
APP_HOST = "0.0.0.0"
APP_PORT = 5000
```

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

## MCP 架构说明

### MCP 服务器

MCP（Model Context Protocol）服务器负责管理和提供工具服务：

```
┌─────────────────────────────────────────────────────┐
│                MCP 服务器 (8080端口)                 │
│  ┌─────────────────────────────────────────────┐   │
│  │           工具注册表 (_TOOL_REGISTRY)          │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────────┐   │   │
│  │  │get_weather│ │get_weather_forecast│ │submit_form│ │   │
│  │  └────┬────┘ └────┬────┘ └──────┬──────┘   │   │
│  └───────┼───────────┼─────────────┼───────────┘   │
│          │           │             │               │
│          └───────────┼─────────────┘               │
│                      ↓                             │
│            ┌─────────────────┐                     │
│            │  FastMCP Server │ ← Streamable HTTP   │
│            └─────────────────┘                     │
└─────────────────────────────────────────────────────┘
                          ↑
                          │ HTTP 请求
                          ↓
┌─────────────────────────────────────────────────────┐
│              应用服务器 (5000端口)                    │
│  ┌─────────────┐    ┌─────────────┐                │
│  │   Agent     │←───│MCPToolService│                │
│  │             │    │   (客户端)   │                │
│  └─────────────┘    └─────────────┘                │
└─────────────────────────────────────────────────────┘
```

### 工具注册机制

工具通过装饰器注册到全局注册表：

```python
from mcp_module.tools.registry import register_tool

@register_tool(
    name="get_weather",
    description="查询指定城市的实时天气",
    parameters=[
        {
            "name": "city",
            "type": "string",
            "description": "要查询天气的城市名称",
            "required": True
        }
    ],
    return_type="string"
)
def get_weather(city: str) -> str:
    # 工具实现...
```

## 使用流程

1. **准备知识库**: 在 `backend/knowledge_base/` 目录添加文档（支持 .txt、.pdf）
2. **启动 MCP 服务器**: `python backend/mcp_module/start.py`
3. **启动应用服务器**: `python backend/app.py`
4. **开始对话**: 访问前端地址，与智能客服对话
5. **RAG 增强**: 系统自动从知识库检索相关内容，增强 AI 回答
6. **工具调用**: AI 通过 MCP 服务器调用天气查询、推荐或表单提交等工具
7. **FewShot 学习**: 系统使用默认示例对话，也可自定义示例

## 自定义扩展

### 修改系统提示词

编辑 `backend/modules/prompt/__init__.py` 中的 `CUSTOMER_SERVICE_PROMPT_TEMPLATE` 变量

### 自定义 FewShot 示例

在 `backend/modules/prompt/__init__.py` 中修改 `DEFAULT_FEW_SHOT_EXAMPLES` 列表：

```python
DEFAULT_FEW_SHOT_EXAMPLES = [
    {"user_query": "你好", "assistant_response": "您好！请问有什么可以帮助您的？"},
    {"user_query": "你们有什么产品?", "assistant_response": "我们提供多种优质产品..."},
]
```

### 添加新工具

1. 在 `backend/mcp_module/tools/` 创建插件文件
2. 使用 `@register_tool` 装饰器注册工具
3. 在 `start.py` 中导入工具模块
4. 重启 MCP 服务

```python
# 示例工具插件
from mcp_module.tools.registry import register_tool

@register_tool(
    name="my_tool",
    description="我的自定义工具",
    parameters=[...],
    return_type="string"
)
def my_tool(param1: str) -> str:
    return "工具执行结果"
```

### 更新知识库

1. 在 `backend/knowledge_base/` 添加或修改文档
2. 重启服务，系统自动重新向量化

## 技术栈

**后端:**
- Flask 3.0.0 - Web 框架
- OpenAI SDK 1.12.0 - AI API 客户端（兼容阿里云百炼）
- Flask-CORS - 跨域支持
- numpy 2.4.4 - 数值计算
- LangChain >= 0.3.0 - Agent 和工具框架
- LangChain Core >= 0.3.0 - 核心组件
- LangChain Community >= 0.3.0 - 社区组件
- MCP - Model Context Protocol（工具服务协议）

**前端:**
- React 18 - UI 框架
- Vite - 构建工具
- Axios - HTTP 请求

**AI 服务:**
- 阿里云百炼平台 / OpenAI
- qwen-plus 模型
- text-embedding-v3 向量化模型

**向量数据库:**
- Chroma - 默认向量存储
- Milvus - 可选向量存储

## 后续优化

- [x] MCP 架构迁移
- [x] 工具独立部署
- [x] Streamable HTTP 支持
- [x] 统一日志模块
- [x] 集中配置管理
- [ ] 数据库替代 JSON 存储
- [ ] API 限流和安全验证
- [ ] Docker 容器化部署