"""
重排序检索器（预留接口）
"""

from typing import List, Optional, Dict
from langchain_core.documents import Document

from .base import BaseRetriever


class RerankingRetriever(BaseRetriever):
    """
    带重排序的检索器（预留接口）
    先检索再用 Cross-Encoder 重排序
    """

    def __init__(self, base_retriever: BaseRetriever, config: Optional[Dict] = None):
        super().__init__(indexer=None, config=config)
        self.base_retriever = base_retriever
        self.rerank_top_k = self.config.get("rerank_top_k", 3)

    def retrieve(self, query: str) -> List[Document]:
        """
        先检索后重排序（预留实现）
        
        Args:
            query: 查询文本
            
        Returns:
            重排序后的文档列表
        """
        # 暂未实现，直接返回基础检索结果
        return self.base_retriever.retrieve(query)[:self.rerank_top_k]

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        获取相关文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        return self.retrieve(query)
