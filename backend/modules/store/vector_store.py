"""
LangChain 向量存储模块
基于 Chroma 和 LangChain 的向量存储实现
"""

import os
from typing import List, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from ..ai_client import LLMClient


class DocumentLoader:
    """文档加载器。

    支持多种文档格式的加载，包括 txt、pdf 和 docx。

    Attributes:
        _loader_map: 文件扩展名到加载器类的映射
    """

    _loader_map = {}

    @classmethod
    def _init_loaders(cls):
        """初始化加载器映射。

        将支持的文件扩展名与对应的加载器类关联。
        """
        try:
            cls._loader_map['.txt'] = TextLoader
            cls._loader_map['.pdf'] = PyPDFLoader
            cls._loader_map['.docx'] = UnstructuredWordDocumentLoader
        except ImportError:
            pass

    @classmethod
    def get_loader(cls, file_path: str):
        """获取文件对应的加载器。

        Args:
            file_path: 文件路径

        Returns:
            对应文件类型的加载器实例，不支持则返回 None
        """
        cls._init_loaders()
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = cls._loader_map.get(ext)
        if loader_class:
            if ext == '.txt':
                return loader_class(file_path, encoding='utf-8')
            else:
                return loader_class(file_path)
        return None

    @classmethod
    def load(cls, file_path: str) -> List[Document]:
        """加载文档。

        Args:
            file_path: 文件路径

        Returns:
            Document 对象列表，加载失败返回空列表
        """
        loader = cls.get_loader(file_path)
        if loader:
            return loader.load()
        return []


class VectorStore:
    """LangChain VectorStore 封装。

    基于 Chroma 实现的向量存储，支持文档加载、分词、向量化存储和相似度检索。

    Attributes:
        ai_client: AI 客户端实例，用于生成文本嵌入
        persist_directory: 向量数据库持久化目录
        collection_name: 集合名称
        vector_store: Chroma 向量存储实例
    """

    def __init__(
        self,
        ai_client: Optional[LLMClient] = None,
        persist_directory: str = "db",
        collection_name: str = "knowledge_base"
    ):
        """初始化 VectorStore。

        Args:
            ai_client: AI 客户端实例
            persist_directory: 向量数据库持久化目录，默认 "db"
            collection_name: 集合名称，默认 "knowledge_base"
        """
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

    def load_vector_store(self) -> Optional[Chroma]:
        """加载已存在的向量存储。

        从 persist_directory 加载已持久化的 Chroma 向量数据库。

        Returns:
            Chroma 向量存储实例，失败时返回 None
        """
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

    def _load_and_embed_documents(self, source_dir: str) -> bool:
        """从目录加载文档并向量化。

        遍历目录下的所有支持的文件类型，加载文档、
        分词后进行向量化存储。

        Args:
            source_dir: 源文件目录路径

        Returns:
            向量化成功返回 Chroma 实例，失败返回 False
        """
        if not os.path.exists(source_dir):
            print(f"源文件目录 {source_dir} 不存在")
            return False

        if not self.ai_client:
            raise ValueError("需要提供 AI 客户端来进行向量化")

        documents = []
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            if os.path.isfile(file_path):
                docs = DocumentLoader.load(file_path)
                if docs:
                    documents.extend(docs)
                    print(f"加载文档: {filename}")

        if not documents:
            print("未找到可加载的文档")
            return False

        split_docs = self.text_splitter.split_documents(documents)
        print(f"文档分割完成，共 {len(split_docs)} 个片段")

        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.ai_client.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            self.vector_store.persist()
            count = self.vector_store._collection.count()
            print(f"向量化完成，共生成 {count} 个向量")
            return self.vector_store
        except Exception as e:
            print(f"向量化失败: {e}")
            return False


    def as_retriever(self, search_kwargs: Optional[Dict] = None) -> Any:
        """转换为检索器。

        将向量存储转换为 LangChain 检索器接口。

        Args:
            search_kwargs: 检索参数，如 {"k": 3}（必传）

        Returns:
            LangChain Retriever 实例
        """
        if not self.vector_store:
            raise ValueError("向量存储未初始化")

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

    # def similarity_search(self, query: str, k: int = 3) -> List[Document]:
    #     """相似度搜索"""
    #     if not self.vector_store:
    #         raise ValueError("向量存储未初始化")
    #     return self.vector_store.similarity_search(query, k=k)

    # def similarity_search_with_score(self, query: str, k: int = 3) -> List[tuple]:
    #     """带分数的相似度搜索"""
    #     if not self.vector_store:
    #         raise ValueError("向量存储未初始化")
    #     return self.vector_store.similarity_search_with_score(query, k=k)

    def init_knowledge_base(self) -> Optional[Dict]:
        """初始化知识库（兼容原有接口）。

        尝试加载已存在的向量存储，如不存在则从 collection_name 目录创建新的。

        Returns:
            包含 status、count 和 message 的字典
        """
        kb_data = self.load_vector_store() or self._load_and_embed_documents(self.collection_name)
        if not kb_data: 
            return {"status": "error", "count": 0, "message": "知识库加载失败"}

        count = kb_data._collection.count()
        if count > 0:
            print(f"知识库已有 {count} 个向量，直接加载")
            return {"status": "loaded", "count": count}
        
        return {"status": "empty", "count": 0}

    def get_vector_store(self) -> Optional[Chroma]:
        """获取向量存储实例。

        Returns:
            Chroma 向量存储实例，可能为 None
        """
        return self.vector_store

    @tool
    def retrieve_knowledge(self, query: str) -> str:
        """从知识库中检索相关内容。

        Args:
            query: 检索查询文本

        Returns:
            相关文档内容拼接的字符串，无结果时返回提示信息
        """
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            if docs:
                return "\n\n".join([d.page_content for d in docs])
            return "（当前无知识库数据）"
        except Exception as e:
            return "（当前无知识库数据）"


__all__ = ['VectorStore', 'DocumentLoader']
