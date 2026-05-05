"""
向量存储工厂类
根据配置文件动态创建和返回对应的向量存储实例
"""

import json
from typing import Optional, Dict, Any

from .base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """
    向量存储工厂类
    
    根据配置文件创建对应的向量存储实例。
    新的向量存储类型通过 VectorStoreFactory.register_store() 注册。
    """

    _store_types: Dict[str, type] = {}

    @classmethod
    def register_store(cls, store_type: str, store_class: type) -> None:
        """
        注册向量存储类型
        
        Args:
            store_type: 存储类型名称，如 "chroma"、"milvus"
            store_class: 继承自 BaseVectorStore 的向量存储类
        """
        if not issubclass(store_class, BaseVectorStore):
            raise TypeError(f"{store_class.__name__} 必须继承自 BaseVectorStore")
        cls._store_types[store_type.lower()] = store_class

    @staticmethod
    def from_config(config_path: str = "config.json", ai_client=None) -> BaseVectorStore:
        """
        从配置文件加载并创建向量存储实例
        
        配置文件格式示例:
        {
            "vector_store": {
                "type": "chroma",
                "config": {
                    "persist_directory": "db/chroma",
                    "collection_name": "knowledge_base"
                }
            }
        }
        
        Args:
            config_path: 配置文件路径，默认 "config.json"
            ai_client: AI 客户端实例
            
        Returns:
            配置文件中指定的向量存储实例
            
        Raises:
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置文件格式不正确或缺少必要字段
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")

        vector_store_config = config.get("vector_store", {})
        store_type = vector_store_config.get("type", "chroma")
        store_config = vector_store_config.get("config", {})

        if store_type not in VectorStoreFactory._store_types:
            raise ValueError(f"不支持的向量存储类型: {store_type}。支持的类型: {list(VectorStoreFactory._store_types.keys())}")

        store_class = VectorStoreFactory._store_types[store_type]
        return store_class(ai_client=ai_client, config=store_config)


__all__ = ['VectorStoreFactory']