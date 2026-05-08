"""
RAG 链核心模块

负责组合索引、检索、生成、记忆和路由模块，实现完整的 RAG 流程。

RAG（Retrieval-Augmented Generation）是检索增强生成的缩写，
通过检索知识库中的相关文档来增强语言模型的回答能力。

核心工作流程：
1. 用户提问 → 路由判断是否需要检索 → 检索相关文档 → 生成回答 → 返回结果

模块组合方式：
- 索引器（Indexer）：负责文档的向量化存储
- 检索器（Retriever）：负责从索引中查询相关文档
- 生成器（Generator）：负责将查询和文档结合生成回答
- 记忆（Memory）：负责管理对话历史
- 路由（Router）：负责决策是否检索及使用哪种策略

提供的核心方法：
- init_default_modules：初始化默认模块组合
- build_index：构建知识库索引
- retrieve：执行检索
- generate：生成回答
- run：执行完整的 RAG 流程
- get_retrieve_knowledge_tool：获取供 Agent 调用的检索工具
"""

from typing import Optional, Dict, Any, List
from langchain_core.documents import Document
from langchain.tools import StructuredTool

from .indexer import BaseIndexer, ChromaIndexer
from .retriever import BaseRetriever, SimpleVectorRetriever
from .generator import BaseGenerator, StuffGenerator
from .memory import BaseMemory, ConversationMemory
from .router import BaseRouter, SimpleRouter


class RAGChain:
    """
    RAG 链核心类
    
    组合索引、检索、生成、记忆和路由模块，实现完整的 RAG 流程。
    
    属性：
        config: 配置字典
        indexer: 索引器实例
        retriever: 检索器实例
        generator: 生成器实例
        memory: 记忆模块实例
        router: 路由器实例
        initialized: 是否已初始化
    
    方法：
        set_indexer/set_retriever/set_generator/set_memory/set_router: 设置各模块
        init_default_modules: 使用默认模块初始化
        build_index: 构建索引
        retrieve: 执行检索
        generate: 生成回答
        run: 完整 RAG 流程
        get_retrieve_knowledge_tool: 获取检索工具
        get_tool: 获取完整 RAG 工具
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化 RAG 链
        
        Args:
            config: 配置字典，包含各模块的配置
        """
        self.config = config or {}
        
        # 模块实例（始终存在，使用默认实现或基类）
        self.indexer: BaseIndexer = BaseIndexer()  # 默认使用基类，init_default_modules 中可替换
        self.retriever: BaseRetriever = SimpleVectorRetriever(config=self.config.get("retriever", {}))
        self.generator: BaseGenerator = BaseGenerator()  # 默认使用基类，init_default_modules 中可替换
        self.memory: BaseMemory = ConversationMemory(config=self.config.get("memory", {}))
        self.router: BaseRouter = SimpleRouter(config=self.config.get("router", {}))
        
        # 初始化状态
        self.initialized = False

    def set_indexer(self, indexer: BaseIndexer):
        """
        设置索引器
        
        Args:
            indexer: 索引器实例
        """
        self.indexer = indexer

    def set_retriever(self, retriever: BaseRetriever):
        """
        设置检索器
        
        Args:
            retriever: 检索器实例
        """
        self.retriever = retriever

    def set_generator(self, generator: BaseGenerator):
        """
        设置生成器
        
        Args:
            generator: 生成器实例
        """
        self.generator = generator

    def set_memory(self, memory: BaseMemory):
        """
        设置记忆模块
        
        Args:
            memory: 记忆模块实例
        """
        self.memory = memory

    def set_router(self, router: BaseRouter):
        """
        设置路由器
        
        Args:
            router: 路由器实例
        """
        self.router = router

    def init_default_modules(self, ai_client):
        """
        使用默认模块初始化 RAG 链
        
        默认配置：
        - 索引器：ChromaIndexer
        - 检索器：SimpleVectorRetriever（已在初始化时创建）
        - 生成器：StuffGenerator
        - 记忆：ConversationMemory（已在初始化时创建）
        - 路由：SimpleRouter（已在初始化时创建）
        
        Args:
            ai_client: AI 客户端实例（用于嵌入和生成）
        """
        # 索引器：Chroma（需要 ai_client）
        self.indexer = ChromaIndexer(
            ai_client=ai_client,
            config=self.config.get("indexer", {})
        )
        
        # 生成器：默认使用基类（需要 ai_client 时可在 build_index 后替换为 StuffGenerator）
        # self.generator = StuffGenerator(
        #     llm_client=ai_client,
        #     config=self.config.get("generator", {})
        # )

        self.initialized = True

    def build_index(self, source_dir: str = "knowledge_base") -> Dict[str, Any]:
        """
        构建知识库索引
        
        Args:
            source_dir: 源文档目录
            
        Returns:
            索引构建结果字典
        """
        result = self.indexer.build_index(source_dir)
        
        # 索引构建成功后，关联索引器到检索器
        if result.get("status") in ("loaded", "created"):
            self.retriever.indexer = self.indexer
            self.retriever._init_retriever()
        
        return result

    def retrieve(self, query: str) -> List[Document]:
        """
        执行检索
        
        Args:
            query: 用户查询
            
        Returns:
            检索到的文档列表
        """

        print(f"[RAG] ---------- 检索阶段开始 ----------")
        print(f"[RAG] 查询内容: {query}")

        # 使用路由器判断是否需要检索
        if not self.router.should_retrieve(query):
            return []
        
        return self.retriever.retrieve(query)

    def generate(self, query: str, documents: List[Document], 
                session_id: str = "default") -> str:
        """
        生成回答
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            session_id: 会话 ID
            
        Returns:
            生成的回答文本
        """
        # 添加用户查询到记忆
        print(f"[RAG] 添加用户查询到记忆...")
        self.memory.add_message(session_id, "human", query)
        
        # 生成回答
        print(f"[RAG] 调用生成器生成回答...")
        answer = self.generator.generate(query, documents)
        
        # 添加回答到记忆
        print(f"[RAG] 添加回答到记忆...")
        self.memory.add_message(session_id, "assistant", answer)

        preview = answer[:150].replace('\n', ' ')
        print(f"[RAG] 生成结果: {preview}...")
        
        return answer

    def run(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        执行完整的 RAG 流程
        
        Args:
            query: 用户查询
            session_id: 会话 ID
            
        Returns:
            包含回答和中间结果的字典
        """
        # 1. 检索
        documents = self.retrieve(query)
        
        # 2. 生成
        answer = self.generate(query, documents, session_id)
        
        return {
            "answer": answer,
            "retrieved_documents": len(documents),
            "session_id": session_id
        }

    def get_retrieve_knowledge_tool(self):
        """
        获取检索知识工具（用于 Agent 调用）
        
        Returns:
            LangChain StructuredTool 实例，如果未初始化则返回 None
        """
        # 如果未初始化，返回 None
        if not self.initialized:
            return None
        
        def retrieve_knowledge(query: str) -> str:
            """
            从知识库检索相关知识
            
            Args:
                query: 查询文本
                
            Returns:
                检索到的知识内容
            """
            documents = self.retrieve(query)
            if not documents:
                return "未找到相关知识"
            
            knowledge = "\n\n".join([doc.page_content for doc in documents])
            return f"检索到的相关知识:\n{knowledge}"
        
        return StructuredTool.from_function(
            func=retrieve_knowledge,
            name="retrieve_knowledge",
            description="从知识库检索相关知识"
        )

    def get_tool(self):
        """
        获取完整的 RAG 工具（检索+生成）
        
        Returns:
            LangChain StructuredTool 实例
        """
        def rag_tool(query: str, session_id: str = "default") -> str:
            """
            从知识库检索并生成回答
            
            Args:
                query: 用户查询
                session_id: 会话 ID
                
            Returns:
                生成的回答
            """
            result = self.run(query, session_id)
            return result["answer"]
        
        return StructuredTool.from_function(
            func=rag_tool,
            name="rag_answer",
            description="从知识库检索相关知识并生成回答"
        )
