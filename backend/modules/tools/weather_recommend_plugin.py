"""
天气预报推荐工具插件
用于查询指定城市未来几天的天气预报
"""

from langchain_core.tools import tool
import requests


@tool
def get_weather_forecast(city: str, days: int = 3) -> str:
    """
    查询指定城市未来几天的天气预报。

    当用户询问未来天气或需要出行建议时使用此工具。

    参数:
        city: 要查询天气的城市名称
        days: 预报天数，可选值 1-3（默认为3天）

    返回:
        格式化的天气预报字符串
    """
    if not city or not city.strip():
        return "请提供要查询天气的城市名称"

    city = city.strip()
    days = max(1, min(3, days))

    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        weather_data = response.json()

        weather_desc = weather_data.get('weather', [])

        if not weather_desc:
            return f"无法获取 {city} 的天气预报"

        location = weather_data.get('nearest_area', [{}])[0]
        city_name = location.get('areaName', [{}])[0].get('value', city)

        forecast_text = f"📍 {city_name} 天气预报 ({days}天):\n\n"

        for i, day in enumerate(weather_desc[:days]):
            date = day.get('date', '')
            max_temp = day.get('maxtempC', 'N/A')
            min_temp = day.get('mintempC', 'N/A')
            avg_temp = day.get('avgtempC', 'N/A')
            desc = day.get('hourly', [{}])[4].get('weatherDesc', [{}])[0].get('value', 'N/A')
            chance_of_rain = day.get('hourly', [{}])[4].get('chanceofrain', 'N/A')

            day_name = "今天" if i == 0 else f"第{i+1}天"
            forecast_text += f"{day_name} ({date}):\n"
            forecast_text += f"  🌡 温度: {min_temp}°C ~ {max_temp}°C (平均 {avg_temp}°C)\n"
            forecast_text += f"  ☁ 天气: {desc}\n"
            forecast_text += f"  🌧 降雨概率: {chance_of_rain}%\n\n"

        return forecast_text.strip()

    except Exception as e:
        return f"查询天气预报失败: {str(e)}"
