"""
配置模块
集中管理项目常量配置
"""

# MCP 服务器配置
MCP_HOST = "0.0.0.0"
MCP_PORT = 8080
MCP_PATH = "/mcp"
MCP_URL = f"http://127.0.0.1:{MCP_PORT}{MCP_PATH}"

# 应用服务器配置
APP_HOST = "0.0.0.0"
APP_PORT = 5000

# 日志配置
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


__all__ = [
    'MCP_HOST',
    'MCP_PORT',
    'MCP_PATH',
    'MCP_URL',
    'APP_HOST',
    'APP_PORT',
    'LOG_LEVEL',
    'LOG_FORMAT',
    'LOG_DATE_FORMAT'
]