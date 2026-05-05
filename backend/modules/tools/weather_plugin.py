"""
天气查询工具插件
用于查询指定城市的实时天气和预报信息
"""

from langchain_core.tools import tool
import requests


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气。

    当用户询问天气情况但未指定城市时，先使用此工具并提示用户提供城市名称。

    参数:
        city: 要查询天气的城市名称，例如：杭州、北京、上海等

    返回:
        格式化的天气报告字符串
    """
    if not city or not city.strip():
        return "请提供要查询天气的城市名称"

    city = city.strip()

    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()

        current = weather_data.get('current_condition', [{}])[0]
        location = weather_data.get('nearest_area', [{}])[0]

        city_name = location.get('areaName', [{}])[0].get('value', city)
        region = location.get('region', [{}])[0].get('value', '')
        country = location.get('country', [{}])[0].get('value', '')

        temp_c = current.get('temp_C', 'N/A')
        feels_like_c = current.get('FeelsLikeC', 'N/A')
        weather_desc = current.get('weatherDesc', [{}])[0].get('value', 'N/A')
        humidity = current.get('humidity', 'N/A')
        wind_speed = current.get('windspeedKmph', 'N/A')
        wind_dir = current.get('winddir16Point', 'N/A')
        uv_index = current.get('uvIndex', 'N/A')
        visibility = current.get('visibility', 'N/A')

        weather_report = f"📍 {city_name} ({region}, {country}) 实时天气:\n"
        weather_report += f"🌡 温度: {temp_c}°C (体感 {feels_like_c}°C)\n"
        weather_report += f"☁ 天气: {weather_desc}\n"
        weather_report += f"💧 湿度: {humidity}%\n"
        weather_report += f"🌬 风速: {wind_speed} km/h, 风向: {wind_dir}\n"
        weather_report += f"☀ 紫外线指数: {uv_index}\n"
        weather_report += f"👁 能见度: {visibility} km"

        return weather_report

    except Exception as e:
        return f"查询天气失败: {str(e)}"


from .tool_factory import ToolFactory
ToolFactory.register_tool(get_weather)
