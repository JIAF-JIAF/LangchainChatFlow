"""
Prompt 模块
定义客服系统的提示模板
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


CUSTOMER_SERVICE_PROMPT_TEMPLATE = """你是一个专业的智能客服助手。

## 你的职责:
1. 热情友好地回答客户问题
2. 提供准确的产品和服务信息
3. 收集客户信息并记录咨询内容
4. 根据客户需求提供合适的解决方案

## 参考知识:
（如果需要，使用 retrieve_knowledge 工具从知识库检索相关信息）

## 工作流程:
1. 首先了解客户的具体需求和问题
2. 如果需要，使用工具来帮助客户（如查询天气、提交表单等）
3. 根据需要收集客户信息
4. 提供合适的解决方案

## 注意事项:
- 保持专业、友好的语气
- 不要编造不实信息
- 遇到无法回答的问题,引导客户留下联系方式,安排专人跟进
- 收集客户信息时要礼貌说明用途"""


def create_chat_prompt(system_prompt: str = CUSTOMER_SERVICE_PROMPT_TEMPLATE):
    """
    创建聊天提示模板（兼容 create_tool_calling_agent）

    参数:
        system_prompt: 系统提示文本

    返回:
        ChatPromptTemplate 实例
    """
    
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


__all__ = [
    'CUSTOMER_SERVICE_PROMPT_TEMPLATE',
    'create_chat_prompt',
]
