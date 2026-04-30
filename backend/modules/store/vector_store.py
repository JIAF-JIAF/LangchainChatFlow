"""
LangChain 向量存储模块
基于 Chroma 和 LangChain 的向量存储实现
"""

import os
from typing import List, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

try:
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
except ImportError:
    TextLoader = None
    PyPDFLoader = None

from ..ai_client import LLMClient


class DocumentLoader:
    """文档加载器"""

    _loader_map = {}

    @classmethod
    def _init_loaders(cls):
        if not cls._loader_map:
            if TextLoader:
                cls._loader_map['.txt'] = TextLoader
            if PyPDFLoader:
                cls._loader_map['.pdf'] = PyPDFLoader
            try:
                from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                cls._loader_map['.docx'] = UnstructuredWordDocumentLoader
            except ImportError:
                pass

    @classmethod
    def get_loader(cls, file_path: str):
        cls._init_loaders()
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = cls._loader_map.get(ext)
        if loader_class:
            return loader_class(file_path, encoding='utf-8')
        return None

    @classmethod
    def load(cls, file_path: str) -> List[Document]:
        loader = cls.get_loader(file_path)
        if loader:
            return loader.load()
        return []


class VectorStore:
    """LangChain VectorStore 封装"""

    def __init__(
        self,
        ai_client: Optional[LLMClient] = None,
        persist_directory: str = "db",
        collection_name: str = "knowledge_base"
    ):
        self.ai_client = ai_client
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store: Optional[Chroma] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True
        )

    def load_documents(self, directory: str = "knowledge_base") -> List[Document]:
        """加载目录下所有支持的文档"""
        documents = []
        if not os.path.exists(directory):
            print(f"知识库目录不存在: {directory}")
            return documents

        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                docs = DocumentLoader.load(file_path)
                if docs:
                    documents.extend(docs)
                    print(f"加载文档: {file_path}")

        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档"""
        if not documents:
            return []
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """创建向量存储"""
        if not documents:
            print("没有文档需要处理")
            return None

        if not self.ai_client:
            raise ValueError("需要提供 AI 客户端来创建向量存储")

        splits = self.split_documents(documents)
        print(f"文档分割完成，共 {len(splits)} 个 chunks")

        os.makedirs(self.persist_directory, exist_ok=True)

        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.ai_client.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        self.vector_store.persist()
        print(f"向量存储创建成功，共 {len(splits)} 个向量")

        return self.vector_store

    def load_vector_store(self) -> Optional[Chroma]:
        """加载已存在的向量存储"""
        if not os.path.exists(self.persist_directory):
            print("向量存储目录不存在")
            return None

        if not self.ai_client:
            raise ValueError("需要提供 AI 客户端来加载向量存储")

        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.ai_client.embeddings,
                collection_name=self.collection_name
            )
            count = self.vector_store._collection.count()
            print(f"向量存储加载成功，共 {count} 个向量")
            return self.vector_store
        except Exception as e:
            print(f"加载向量存储失败: {e}")
            return None

    def as_retriever(self, search_kwargs: Optional[Dict] = None) -> Any:
        """转换为检索器"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化")

        if search_kwargs is None:
            search_kwargs = {"k": 3}

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """相似度搜索"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
        """带分数的相似度搜索"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化")
        return self.vector_store.similarity_search_with_score(query, k=k)

    def create_embeddings(self, text: str) -> list:
        """创建嵌入（兼容原有接口）"""
        if not self.ai_client:
            raise ValueError("AI 客户端未初始化")
        return self.ai_client.create_embedding(text)

    def init_knowledge_base(self) -> Optional[Dict]:
        """初始化知识库（兼容原有接口）"""
        kb_data = self.load_vector_store()
        if kb_data:
            return {"entries": []}
        return None

    def get_vector_store(self) -> Optional[Chroma]:
        """获取向量存储实例"""
        return self.vector_store


__all__ = ['VectorStore', 'DocumentLoader']
