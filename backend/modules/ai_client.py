"""
LangChain LLM 模块
封装 LangChain 的 ChatOpenAI 客户端
"""

import json
from typing import Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from openai import OpenAI


class LangChainEmbeddings(Embeddings):
    """LangChain Embeddings 封装"""

    def __init__(self, ai_client):
        self.ai_client = ai_client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.ai_client.create_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.ai_client.create_embedding(text)


class LLMClient:
    """LangChain LLM 客户端封装"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        temperature: float = 0.7
    ):
        self.temperature = temperature
        self.config_path = config_path
        self._client: Optional[ChatOpenAI] = None
        self._embedding_client: Optional[LangChainEmbeddings] = None

        if config_path:
            self._load_from_config()
        else:
            raise ValueError("请提供 API 密钥和 base_url 或配置文件路径")

        self._init_clients()

    def _load_from_config(self):
        """从配置文件加载 API 配置。

        从指定的配置文件路径读取 API 密钥、base_url 和 model 等配置。

        Raises:
            Exception: 读取或解析配置文件失败时抛出
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.api_key = config.get('api_key')
            self.base_url = config.get('base_url')
            self.model = config.get('model')
            print(f"从配置文件 {self.config_path} 读取 API 配置")
        except Exception as e:
            print(f"读取配置文件失败: {e}")
            raise

    def _init_clients(self):
        """初始化 LangChain ChatOpenAI 客户端。

        使用配置文件中的 model、api_key 和 base_url 创建 ChatOpenAI 实例。
        """
        self._client = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature
        )
        print(f"LangChain LLM 客户端初始化成功 (model={self.model})")

    @property
    def chat(self) -> ChatOpenAI:
        """获取 ChatOpenAI 客户端"""
        return self._client

    @property
    def embeddings(self) -> LangChainEmbeddings:
        """获取 Embeddings 客户端"""
        if self._embedding_client is None:
            self._embedding_client = LangChainEmbeddings(self)
        return self._embedding_client

    def create_embedding(self, text: str) -> list[float]:
        """创建文本嵌入向量。

        Args:
            text: 要嵌入的文本内容

        Returns:
            文本的嵌入向量表示，失败时返回空列表
        """
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.embeddings.create(
                input=text,
                model="text-embedding-v3"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"生成嵌入失败: {e}")
            return []


class AIClient(LLMClient):
    """兼容原有 AIClient 接口"""
    pass


__all__ = ['LLMClient', 'AIClient', 'LangChainEmbeddings']
