"""
路由模块
负责决定是否需要检索、使用哪个知识库、使用哪种检索策略
"""

from .base import BaseRouter
from .simple import SimpleRouter

__all__ = [
    'BaseRouter',
    'SimpleRouter'
]
