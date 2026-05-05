"""
LangChain Agent 模块
结合 LLM + Tools 的 Agent 实现
"""

from typing import Optional, Dict, Any, List
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor


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

        self._build_agent()

    def _get_chat_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """获取或创建会话历史（供 RunnableWithMessageHistory 使用）"""
        if session_id not in self._chat_history_store:
            self._chat_history_store[session_id] = InMemoryChatMessageHistory()
        return self._chat_history_store[session_id]

    def _build_agent(self):
        """构建 Agent（使用 create_tool_calling_agent）"""
        tools = [self.vector_store.retrieve_knowledge] + self._tools

        self._agent = create_tool_calling_agent(
            llm=self.llm_client.chat,
            tools=tools,
            prompt=self.prompt
        )

        self._agent_executor = AgentExecutor(
            agent=self._agent,
            tools=tools,
            verbose=self.verbose,
            handle_parsing_errors=True
        )

        self._agent_chain = RunnableWithMessageHistory(
            self._agent_executor,
            self._get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def invoke(self, input: str, session_id: str = "default") -> Dict[str, Any]:
        """执行 Agent"""
        result = self._agent_chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": session_id}}
        )

        return {
            "answer": result["output"] if hasattr(result, 'output') else str(result),
            "intermediate_steps": result.get("intermediate_steps", []),
            "tool_messages": []
        }

    def process_message(self, session_id, user_message):
        """发送对话（兼容原有接口）"""
        result = self.invoke(user_message, session_id)
        return {
            "content": result["answer"],
            "tool_calls": []
        }


__all__ = ['Agent']