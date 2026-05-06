"""
Tools package
每个工具单独一个文件，便于扩展
"""

from .tool_factory import ToolFactory

from . import weather_plugin
from . import weather_recommend_plugin
from . import submit_form_plugin

__all__ = [
    'ToolFactory',
]