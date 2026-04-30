"""
LangChain RAG 模块
简单的 RAG 检索实现
"""

from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from .ai_client import LLMClient
from .store.vector_store import VectorStore


CUSTOMER_SERVICE_SYSTEM_PROMPT = """你是一个专业的智能客服助手,负责解答客户咨询、提供产品和服务信息。

## 你的职责:
1. 热情友好地回答客户问题
2. 提供准确的产品和服务信息
3. 收集客户信息并记录咨询内容
4. 根据客户需求提供合适的解决方案

## 工作流程:
1. 首先了解客户的具体需求和问题
2. 根据知识库内容提供准确回答
3. 如果需要进一步跟进,收集客户联系方式(姓名、电话、微信)
4. 记录客户咨询意图和摘要
5. 提交表单以便后续跟进

## 知识库上下文:
{context}

## 注意事项:
- 保持专业、友好的语气
- 不要编造不实信息
- 遇到无法回答的问题,引导客户留下联系方式,安排专人跟进
- 收集客户信息时要礼貌说明用途"""


class RAGChain:
    """RAG Chain 封装"""

    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: Optional[VectorStore] = None,
        system_prompt: Optional[str] = None,
        return_source_documents: bool = True,
        verbose: bool = False
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.system_prompt = system_prompt or CUSTOMER_SERVICE_SYSTEM_PROMPT
        self.return_source_documents = return_source_documents
        self.verbose = verbose
        self.chat_history: Dict[str, List[BaseMessage]] = {}

    def set_vector_store(self, vector_store: VectorStore):
        """设置向量存储"""
        self.vector_store = vector_store

    def get_chat_history(self, session_id: str) -> List[BaseMessage]:
        """获取会话历史"""
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        return self.chat_history[session_id]

    def add_to_chat_history(self, session_id: str, human: str, ai: str):
        """添加对话到历史"""
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        self.chat_history[session_id].append(HumanMessage(content=human))
        self.chat_history[session_id].append(AIMessage(content=ai))

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """直接检索相关文档"""
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(query, k=k)

    def build_context(self, query: str, top_k: int = 3) -> str:
        """构建检索上下文"""
        docs = self.retrieve(query, k=top_k)
        if not docs:
            return ""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[{i}] {doc.page_content}")
        return "\n\n".join(context_parts)

    def enhance_query(self, query: str, embed_func=None, top_k: int = 3) -> str:
        """增强用户查询"""
        docs = self.retrieve(query, k=top_k)
        if not docs:
            return query
        context = "\n\n".join([doc.page_content for doc in docs])
        return "【相关知识】\n{}\n\n【用户问题】\n{}".format(context, query)

    def clear_history(self, session_id: str):
        """清除会话历史"""
        if session_id in self.chat_history:
            del self.chat_history[session_id]


class RAG:
    """兼容原有 RAG 接口"""

    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base

    def set_knowledge_base(self, knowledge_base: Dict) -> None:
        self.knowledge_base = knowledge_base


__all__ = ['RAGChain', 'RAG', 'CUSTOMER_SERVICE_SYSTEM_PROMPT']
