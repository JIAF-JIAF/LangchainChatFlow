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


def get_tool_by_name(name: str):
    """根据名称获取工具"""
    tools_map = {
        "get_weather": get_weather,
        "get_weather_forecast": get_weather_forecast,
        "submit_form": submit_form,
    }
    return tools_map.get(name)


__all__ = [
    'get_weather',
    'get_weather_forecast',
    'submit_form',
    'get_all_tools',
    'get_tool_by_name',
]
