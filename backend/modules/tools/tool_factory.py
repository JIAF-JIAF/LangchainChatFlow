"""
工具工厂类
管理所有可用的工具，支持注册和获取
"""

from typing import List, Callable


class ToolFactory:
    """
    工具工厂类
    
    管理所有可用的工具，新的工具通过 ToolFactory.register_tool() 注册。
    """

    _tools: List[Callable] = []

    @classmethod
    def register_tool(cls, tool: Callable) -> None:
        """
        注册工具
        
        Args:
            tool: 工具函数（通常是 @tool 装饰的函数）
        """
        if tool not in cls._tools:
            cls._tools.append(tool)

    @classmethod
    def get_all_tools(cls) -> List[Callable]:
        """
        获取所有已注册的工具列表
        
        Returns:
            所有已注册的工具列表
        """
        return cls._tools.copy()


__all__ = ['ToolFactory']