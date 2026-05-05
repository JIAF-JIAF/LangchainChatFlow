"""
向量存储抽象基类
定义通用的向量库 API 接口，所有向量库实现都必须继承此类并实现抽象方法
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.document_loaders import DocumentLoaderFactory


class BaseVectorStore(ABC):
    """
    向量存储抽象基类
    
    定义了向量库的核心操作接口，包括：
    - 初始化和加载
    - 文档向量化和存储
    - 相似度检索
    - 作为 LangChain Retriever 使用
    
    同时提供通用的文档加载和分割逻辑。
    """

    def __init__(self, ai_client=None, config: Optional[Dict] = None):
        """
        初始化向量存储
        
        Args:
            ai_client: AI 客户端实例，用于生成向量嵌入
            config: 配置字典，包含存储路径、集合名称等参数
        """
        self.ai_client = ai_client
        self.config = config or {}
        
        # 初始化文本分割器（通用配置）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            length_function=len,
            add_start_index=True
        )

    def _load_documents_from_directory(self, source_dir: str) -> Optional[List[Document]]:
        """
        从目录加载文档并分割（通用逻辑）

        Args:
            source_dir: 源文件目录路径

        Returns:
            分割后的 Document 对象列表，失败返回 None
        """
        if not os.path.exists(source_dir):
            print(f"源文件目录 {source_dir} 不存在")
            return None

        if not self.ai_client:
            raise ValueError("需要提供 AI 客户端来进行向量化")

        documents = []
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            if os.path.isfile(file_path):
                docs = DocumentLoaderFactory.load(file_path)
                if docs:
                    documents.extend(docs)
                    print(f"加载文档: {filename}")

        if not documents:
            print("未找到可加载的文档")
            return None

        split_docs = self.text_splitter.split_documents(documents)
        print(f"文档分割完成，共 {len(split_docs)} 个片段")
        
        return split_docs

    @abstractmethod
    def init_knowledge_base(self, source_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        初始化知识库
        
        尝试加载已存在的向量存储，如果不存在则从源目录创建新的知识库
        
        Args:
            source_dir: 源文档目录路径
            
        Returns:
            包含 status、message、count 等信息的字典
        """
        pass

    @abstractmethod
    def load_vector_store(self) -> bool:
        """
        加载已存在的向量存储
        
        Returns:
            加载成功返回 True，失败返回 False
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文档到向量存储
        
        Args:
            documents: Document 对象列表
            
        Returns:
            添加成功返回 True，失败返回 False
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回的文档数量，默认 3
            
        Returns:
            匹配的 Document 对象列表
        """
        pass

    @abstractmethod
    def as_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        转换为 LangChain 检索器
        
        Args:
            search_kwargs: 检索参数，如 {"k": 3}
            
        Returns:
            LangChain Retriever 实例
        """
        pass

    @abstractmethod
    def persist(self) -> None:
        """
        持久化向量存储到磁盘
        """
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            删除成功返回 True，失败返回 False
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            包含向量数量等统计信息的字典
        """
        pass


__all__ = ['BaseVectorStore']