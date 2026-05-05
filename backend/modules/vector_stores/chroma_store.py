"""
Chroma 向量存储实现
继承 BaseVectorStore 抽象基类
"""

import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma

from .base_vector_store import BaseVectorStore
from .store_factory import VectorStoreFactory


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma 向量存储实现
    
    基于 Chroma 的向量存储，支持文档加载、分词、向量化存储和相似度检索。
    """

    def __init__(self, ai_client=None, config: Optional[Dict] = None):
        """
        初始化 Chroma 向量存储
        
        Args:
            ai_client: AI 客户端实例，用于生成向量嵌入
            config: 配置字典，包含:
                - persist_directory: 持久化目录，默认 "db/chroma"
                - collection_name: 集合名称，默认 "knowledge_base"
        """
        # 调用父类初始化（包含文本分割器）
        super().__init__(ai_client=ai_client, config=config)
        
        self.persist_directory = self.config.get("persist_directory", "db/chroma")
        self.collection_name = self.config.get("collection_name", "knowledge_base")
        self.vector_store = None

    def init_knowledge_base(self, source_dir: Optional[str] = "knowledge_base") -> Dict[str, Any]:
        """
        初始化知识库
        
        尝试加载已存在的向量存储，如果不存在则从源目录创建新的知识库
        
        Args:
            source_dir: 源文档目录路径，默认 "knowledge_base"
            
        Returns:
            包含 status、message、count 等信息的字典
        """
        # 尝试加载已存在的向量存储
        if self.load_vector_store():
            stats = self.get_collection_stats()
            return {
                "status": "loaded",
                "message": f"成功加载已存在的知识库",
                "count": stats.get("vector_count", 0)
            }
        
        # 如果不存在，尝试从源目录创建
        if source_dir and os.path.exists(source_dir):
            if self._load_and_embed_documents(source_dir):
                stats = self.get_collection_stats()
                return {
                    "status": "created",
                    "message": "成功创建新知识库",
                    "count": stats.get("vector_count", 0)
                }
            else:
                return {
                    "status": "error",
                    "message": "创建知识库失败"
                }
        else:
            return {
                "status": "empty",
                "message": "源目录不存在，知识库为空"
            }

    def load_vector_store(self) -> bool:
        """
        加载已存在的向量存储
        
        Returns:
            加载成功返回 True，失败返回 False
        """
        if not os.path.exists(self.persist_directory):
            return False

        if not self.ai_client:
            raise ValueError("需要提供 AI 客户端来加载向量存储")

        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.ai_client.embeddings,
                collection_name=self.collection_name
            )
            count = self.vector_store._collection.count()
            print(f"Chroma 向量存储加载成功，共 {count} 个向量")
            return True
        except Exception as e:
            print(f"加载 Chroma 向量存储失败: {e}")
            return False

    def _load_and_embed_documents(self, source_dir: str) -> bool:
        """
        从目录加载文档并向量化（使用基类的通用文档加载方法）
        
        Args:
            source_dir: 源文件目录路径
            
        Returns:
            向量化成功返回 True，失败返回 False
        """
        # 使用基类的通用方法加载和分割文档
        split_docs = self._load_documents_from_directory(source_dir)
        if split_docs is None:
            return False

        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.ai_client.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            self.persist()
            print(f"向量化完成，共生成 {self.vector_store._collection.count()} 个向量")
            return True
        except Exception as e:
            print(f"向量化失败: {e}")
            return False

    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文档到向量存储
        
        Args:
            documents: Document 对象列表
            
        Returns:
            添加成功返回 True，失败返回 False
        """
        if not self.vector_store:
            print("向量存储未初始化")
            return False

        try:
            split_docs = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(split_docs)
            self.persist()
            return True
        except Exception as e:
            print(f"添加文档失败: {e}")
            return False

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回的文档数量，默认 3
            
        Returns:
            匹配的 Document 对象列表
        """
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        return self.vector_store.similarity_search(query, k=k)

    def as_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        转换为 LangChain 检索器
        
        Args:
            search_kwargs: 检索参数，如 {"k": 3}
            
        Returns:
            LangChain Retriever 实例
        """
        if not self.vector_store:
            raise ValueError("向量存储未初始化")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs or {"k": 3}
        )

    @tool("retrieve_knowledge")
    def retrieve_knowledge(self, query: str) -> str:
        """
        从知识库检索相关知识（作为工具供 Agent 调用）
        
        Args:
            query: 查询文本
            
        Returns:
            检索到的知识内容
        """
        if not self.vector_store:
            return "知识库未初始化"

        results = self.similarity_search(query, k=3)
        if not results:
            return "未找到相关知识"

        knowledge = "\n\n".join([doc.page_content for doc in results])
        return f"检索到的相关知识:\n{knowledge}"

    def persist(self) -> None:
        """
        持久化向量存储到磁盘
        """
        if self.vector_store:
            self.vector_store.persist()

    def delete_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            删除成功返回 True，失败返回 False
        """
        try:
            if self.vector_store:
                self.vector_store.delete_collection()
                self.vector_store = None
            return True
        except Exception as e:
            print(f"删除集合失败: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            包含向量数量等统计信息的字典
        """
        if not self.vector_store:
            return {"vector_count": 0}
        
        try:
            count = self.vector_store._collection.count()
            return {
                "vector_count": count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"获取统计信息失败: {e}")
            return {"vector_count": 0}


VectorStoreFactory.register_store("chroma", ChromaVectorStore)

__all__ = ['ChromaVectorStore']