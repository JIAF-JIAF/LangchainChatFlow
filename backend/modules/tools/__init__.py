"""
Tools package
每个工具单独一个文件，便于扩展
"""

from .weather_plugin import get_weather
from .weather_recommend_plugin import get_weather_forecast
from .submit_form_plugin import submit_form


def get_all_tools():
    """获取所有工具列表"""
    return [get_weather, get_weather_forecast, submit_form]

__all__ = [
    'get_weather',
    'get_weather_forecast',
    'submit_form',
    'get_all_tools',
]
