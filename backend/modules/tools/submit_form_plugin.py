"""
表单提交工具插件
用于提交客户咨询表单，记录客户信息和咨询内容
"""

from typing import Optional
from langchain_core.tools import tool


@tool
def submit_form(
    name: str,
    phone: str,
    intention: str,
    wechat: Optional[str] = None,
    address: Optional[str] = None
) -> str:
    """
    提交客户咨询表单，记录客户信息和咨询内容。

    当需要收集客户联系方式以便后续跟进时使用此工具。

    参数:
        name: 客户姓名
        phone: 客户联系电话
        intention: 客户咨询意图或需求概述
        wechat: 客户微信号（可选）
        address: 客户地址（可选）

    返回:
        表单提交结果的描述
    """
    if not name or not phone or not intention:
        return "表单提交失败：姓名、电话和咨询意图为必填项"

    summary = f"客户 {name} 咨询: {intention}"

    print(f"表单提交成功: {summary}")
    print(f"客户信息: 姓名={name}, 电话={phone}, 微信={wechat}, 地址={address}")

    return f"✅ 表单提交成功\n客户姓名: {name}\n联系电话: {phone}\n咨询意图: {intention}\n微信号: {wechat or '未提供'}\n地址: {address or '未提供'}"
