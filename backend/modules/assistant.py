"""
LangChain Agent 模块
结合 LLM + Tools 的 Agent 实现
"""

from typing import Optional, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from .ai_client import LLMClient
from .tools import get_all_tools
from .store.vector_store import VectorStore
from .prompt import CUSTOMER_SERVICE_PROMPT_TEMPLATE, create_chat_prompt


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
        self._tools = options.get('tools', [])
        self.prompt = options.get('prompt')

        self.verbose = True
        self._chat_history_store: Dict[str, InMemoryChatMessageHistory] = {}

        if not self._tools:
            self._tools = get_all_tools()

        self._build_rag_chain()

    def _get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """获取或创建会话历史（供 RunnableWithMessageHistory 使用）"""
        if session_id not in self._chat_history_store:
            self._chat_history_store[session_id] = InMemoryChatMessageHistory()
        return self._chat_history_store[session_id]

    def _build_rag_chain(self):
        """构建 RAG Chain（纯 LCEL 方式）"""
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            self._rag_chain = (
                {
                    "input": lambda x: x["input"],
                    "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs]))
                }
                | self.prompt
                | self.llm_client.chat
            )
            print("RAG Chain 构建成功，检索 top3 相关块")
        except Exception as e:
            print(f"警告：RAG 初始化失败 ({e})，使用纯对话模式")
            self._rag_chain = (
                {
                    "input": lambda x: x["input"],
                    "context": lambda x: "（当前无知识库数据）"
                }
                | self.prompt
                | self.llm_client.chat
            )

        self._rag_chain = RunnableWithMessageHistory(
            self._rag_chain,
            self._get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def invoke(self, input: str, session_id: str = "default") -> Dict[str, Any]:
        """执行 Agent（RAG 模式）"""
        result = self._rag_chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": session_id}}
        )

        return {
            "answer": result.content if hasattr(result, 'content') else str(result),
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

    def process_message(self, session_id, user_message):
        """处理完整对话流程"""
        return self.chat(session_id, user_message)


__all__ = ['Agent']