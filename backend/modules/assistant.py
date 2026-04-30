"""
LangChain Agent 模块
结合 LLM + Tools + Memory 的 Agent 实现
"""

from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from .ai_client import LLMClient
from .tools import get_all_tools
from .store.vector_store import VectorStore


CUSTOMER_SERVICE_PROMPT = """你是一个专业的智能客服助手。

## 你的职责:
1. 热情友好地回答客户问题
2. 提供准确的产品和服务信息
3. 收集客户信息并记录咨询内容
4. 根据客户需求提供合适的解决方案

## 可用工具:
- get_weather: 查询城市天气
- get_weather_forecast: 查询天气预报
- submit_form: 提交客户咨询表单

## 工作流程:
1. 首先了解客户的具体需求和问题
2. 如果需要查询天气，使用天气工具
3. 如果需要收集客户信息，使用表单提交工具
4. 根据客户需求提供合适的解决方案

## 注意事项:
- 保持专业、友好的语气
- 不要编造不实信息
- 遇到无法回答的问题,引导客户留下联系方式,安排专人跟进
- 收集客户信息时要礼貌说明用途"""


class Agent:
    """LangChain Agent 封装"""

    def __init__(
        self,
        options: Optional[Dict] = None,
        config_path: str = "config.json"
    ):
        if options is None:
            options = {}

        self.llm_client = options.get('aiClient')
        self.vector_store = options.get('vectorStore')
        self.rag_module = options.get('ragModule')
        self.prompt_template = options.get('prompt')
        self.memory = options.get('memory')
        self._tools = options.get('tools', [])

        self.system_prompt = CUSTOMER_SERVICE_PROMPT
        if self.prompt_template:
            if isinstance(self.prompt_template, dict):
                self.system_prompt = self.prompt_template.get("content", CUSTOMER_SERVICE_PROMPT)
            else:
                self.system_prompt = str(self.prompt_template)

        self.verbose = True
        self._chat_history: Dict[str, List] = {}

        if not self._tools:
            self._tools = get_all_tools()

        self._build_agent()

    def _build_agent(self):
        """构建 Agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
        ])

        self._chain = prompt | self.llm_client.chat | StrOutputParser()
        print(f"Agent 构建成功，共 {len(self._tools)} 个工具")

    def set_rag_module(self, ragModule, vectorStore=None):
        """设置 RAG 模块"""
        self.rag_module = ragModule
        self.vector_store = vectorStore

    def set_tools(self, tools):
        """设置工具"""
        self._tools = tools

    def enhance_query(self, user_message):
        """使用 RAG 增强查询"""
        if self.rag_module and self.vector_store:
            try:
                return self.rag_module.enhance_query(
                    user_message,
                    self.vector_store.create_embeddings,
                    top_k=3
                )
            except Exception as e:
                print("RAG 检索失败: {}".format(e))
                return user_message
        return user_message

    def get_session(self, session_id):
        """获取会话"""
        return self._chat_history.get(session_id)

    def create_session(self, session_id, system_content=""):
        """创建会话"""
        self._chat_history[session_id] = []
        return self._chat_history[session_id]

    def invoke(self, input: str, session_id: str = "default") -> Dict[str, Any]:
        """执行 Agent"""
        chat_history = self.get_session(session_id)
        if chat_history is None:
            chat_history = self.create_session(session_id)

        result = self._chain.invoke({
            "input": input,
            "chat_history": chat_history
        })

        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        self._chat_history[session_id].append(HumanMessage(content=input))
        self._chat_history[session_id].append(AIMessage(content=result))

        return {
            "answer": result,
            "intermediate_steps": [],
            "tool_messages": []
        }

    def chat(self, session_id, user_message):
        """发送对话（兼容原有接口）"""
        result = self.invoke(user_message, session_id)
        return {
            "content": result["answer"],
            "tool_calls": []
        }

    def submit_tool_result(self, session_id, tool_call_id, tool_result):
        """提交工具结果"""
        return self.chat(session_id, f"工具 {tool_call_id} 返回: {tool_result}")["content"]

    def process_message(self, session_id, user_message):
        """处理完整对话流程"""
        return self.chat(session_id, user_message)

    def add_message(self, session_id, message):
        """添加消息到历史"""
        if session_id not in self._chat_history:
            self._chat_history[session_id] = []
        self._chat_history[session_id].append(message)

    def prune_session_history(self, session_id, max_messages=50):
        """修剪会话历史"""
        if session_id in self._chat_history:
            history = self._chat_history[session_id]
            system_msgs = [m for m in history if isinstance(m, SystemMessage)]
            other_msgs = [m for m in history if not isinstance(m, SystemMessage)]
            if len(system_msgs) + len(other_msgs) > max_messages:
                self._chat_history[session_id] = system_msgs + other_msgs[-(max_messages - len(system_msgs)):]


__all__ = ['Agent', 'CUSTOMER_SERVICE_PROMPT']
