"""
检索模块
负责从索引中检索相关文档，支持多种检索策略

已实现：
- SimpleVectorRetriever：基于向量相似度的简单检索

预留接口：
- RerankingRetriever：重排序检索
- FilteredRetriever：过滤检索
"""

from .base import BaseRetriever
from .simple import SimpleVectorRetriever

__all__ = [
    'BaseRetriever',
    'SimpleVectorRetriever'
]
