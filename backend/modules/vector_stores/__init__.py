"""
向量存储模块
提供插件化的向量数据库支持

使用方式:
    from modules.vector_stores import VectorStoreFactory

    store = VectorStoreFactory.from_config("config.json", ai_client)
    result = store.init_knowledge_base()
    docs = store.similarity_search("查询文本")
"""

from .base_vector_store import BaseVectorStore
from .chroma_store import ChromaVectorStore
from .milvus_store import MilvusVectorStore
from .store_factory import VectorStoreFactory

__all__ = [
    'BaseVectorStore',
    'ChromaVectorStore',
    'MilvusVectorStore',
    'VectorStoreFactory'
]