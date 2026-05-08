"""
MCP 服务模块
封装 MCP 工具获取逻辑，统一从远程 MCP 服务器获取工具
"""

import asyncio
from typing import List, Any

from langchain.tools import StructuredTool
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

import mcp_module.mcp_client as mcp_client
import mcp_module.logger as logger
import mcp_module.config as config


class MCPToolService:
    """MCP 工具服务类"""

    @staticmethod
    def get_tools(server_url: str = config.MCP_URL) -> List[Any]:
        """
        从 MCP 服务器获取工具列表
        
        Args:
            server_url: MCP 服务器地址
        """
        logger.logger.info(f"连接 MCP 服务器: {server_url}")
        
        mcp_tools = asyncio.run(mcp_client.get_tools_from_server(server_url))
        logger.logger.info(f"从 MCP 服务器获取到 {len(mcp_tools)} 个工具")
        
        callable_tools = MCPToolService._create_callable_tools(mcp_tools, server_url)
        
        return callable_tools

    @staticmethod
    def _create_callable_tools(mcp_tools: List[Any], server_url: str) -> List[StructuredTool]:
        """将 MCP 工具转换为 LangChain 可调用的工具对象"""
        tools = []
        
        for mcp_tool in mcp_tools:
            name = getattr(mcp_tool, 'name', 'unknown')
            description = getattr(mcp_tool, 'description', '').strip()
            
            # 使用默认参数捕获当前迭代的值，避免闭包变量问题
            def create_tool_call(tool_name: str = name, url: str = server_url):
                def call_tool(**kwargs):
                    # 处理参数格式：可能是 {'kwargs': {...}} 或直接 {...}
                    args = kwargs.get('kwargs', kwargs)
                    
                    async def _call():
                        async with streamable_http_client(url) as (read_stream, write_stream, get_session_id):
                            async with ClientSession(
                                read_stream=read_stream,
                                write_stream=write_stream,
                                client_info={"name": "chartflow-client", "version": "1.0.0"}
                            ) as session:
                                await session.initialize()
                                result = await session.call_tool(tool_name, args)
                                return result
                    
                    return asyncio.run(_call())
                
                return call_tool
            
            tool_func = create_tool_call()
            tool_func.__name__ = name
            tool_func.__doc__ = description
            
            tool = StructuredTool.from_function(
                func=tool_func,
                name=name,
                description=description
            )
            
            tools.append(tool)
            
        return tools


__all__ = ['MCPToolService']