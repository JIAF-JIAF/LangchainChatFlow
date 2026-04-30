"""
LangChain Memory 模块
使用 LangChain 内置的 Memory 组件管理会话历史
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory


class Memory:
    """
    会话记忆管理器
    基于 LangChain Memory 封装，支持多会话管理
    """

    def __init__(self):
        self._chat_histories: Dict[str, ChatMessageHistory] = {}

    def get_chat_history(self, session_id: str = "default") -> ChatMessageHistory:
        """获取或创建聊天历史"""
        if session_id not in self._chat_histories:
            self._chat_histories[session_id] = ChatMessageHistory()
        return self._chat_histories[session_id]

    def get_session(self, session_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取会话（兼容原有接口）"""
        if session_id not in self._chat_histories:
            return None
        history = self._chat_histories[session_id]
        return [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in history.messages if isinstance(m, (HumanMessage, AIMessage))
        ]

    def create_session(self, session_id: str, system_instructions: str = "") -> List[Dict[str, Any]]:
        """创建新会话（兼容原有接口）"""
        self._chat_histories[session_id] = ChatMessageHistory()
        if system_instructions:
            self._chat_histories[session_id].add_system_message(system_instructions)
        return self.get_session(session_id) or []

    def add_message(self, session_id: str, message: Dict[str, Any]) -> None:
        """添加消息到会话历史"""
        chat_history = self.get_chat_history(session_id)
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            chat_history.add_user_message(content)
        elif role == "assistant":
            chat_history.add_ai_message(content)
        elif role == "system":
            chat_history.add_system_message(content)
        elif role == "tool":
            tool_call_id = message.get("tool_call_id", "")
            chat_history.add_message(
                SystemMessage(content=f"[Tool {tool_call_id}]: {content}")
            )

    def get_session_history(self, session_id: str) -> List[BaseMessage]:
        """获取会话历史"""
        return self.get_chat_history(session_id).messages

    def clear_session(self, session_id: str) -> None:
        """清除会话历史"""
        if session_id in self._chat_histories:
            self._chat_histories[session_id].clear()

    def get_all_sessions(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有会话"""
        result = {}
        for sid, history in self._chat_histories.items():
            result[sid] = [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                for m in history.messages if isinstance(m, (HumanMessage, AIMessage))
            ]
        return result

    def update_session(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """更新会话历史"""
        self._chat_histories[session_id] = ChatMessageHistory()
        for msg in messages:
            self.add_message(session_id, msg)

    def prune_session_history(self, session_id: str, max_messages: int = 50) -> None:
        """修剪会话历史，保持在指定长度内"""
        if session_id not in self._chat_histories:
            return

        history = self._chat_histories[session_id]
        messages = history.messages

        if len(messages) > max_messages:
            system_messages = [m for m in messages if isinstance(m, SystemMessage)]
            other_messages = [m for m in messages if not isinstance(m, SystemMessage)]

            if len(system_messages) + len(other_messages) > max_messages:
                keep_messages = system_messages + other_messages[-(max_messages - len(system_messages)):]
                history.messages = keep_messages


__all__ = ['Memory']
