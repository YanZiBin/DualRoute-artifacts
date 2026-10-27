# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
"""Fast Path v1 demo app registry for local launch commands."""

from __future__ import annotations

from typing import Final


# WeChat is present in the product whitelist, but the runtime guard keeps an
# unspecified instance fail-closed because this device exposes two instances.
# The final experiment catalog intentionally excludes WeChat.
FAST_PATH_APP_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "微信": ("微信", "WeChat", "wechat"),
    "QQ": ("QQ", "qq"),
    "QQ邮箱": ("QQ邮箱",),
    "QQ音乐": ("QQ音乐", "QQ 音乐"),
    "七猫免费小说": ("七猫免费小说", "七猫", "七猫小说"),
    "京东": ("京东", "JD", "jd"),
    "京东秒送": ("京东秒送",),
    "今日头条": ("今日头条", "头条"),
    "优酷视频": ("优酷视频", "优酷", "YOUKU"),
    "去哪儿": ("去哪儿", "Qunar"),
    "去哪儿旅行": ("去哪儿旅行",),
    "同花顺": ("同花顺",),
    "喜马拉雅": ("喜马拉雅",),
    "大众点评": ("大众点评", "点评"),
    "安居客": ("安居客",),
    "小红书": ("小红书", "XHS", "xhs"),
    "崩坏：星穹铁道": ("崩坏：星穹铁道", "崩坏星穹铁道", "星穹铁道"),
    "微博": ("微博", "WEIBO"),
    "快手": ("快手", "KUAISHOU"),
    "恋与深空": ("恋与深空",),
    "抖音": ("抖音", "Douyin", "douyin"),
    "拼多多": ("拼多多", "PDD", "pdd"),
    "支付宝": ("支付宝", "Alipay", "alipay"),
    "携程": ("携程", "携程旅行", "Ctrip"),
    "汽水音乐": ("汽水音乐", "汽水"),
    "淘宝": ("淘宝", "手机淘宝", "Taobao", "taobao"),
    "淘宝闪购": ("淘宝闪购",),
    "滴滴出行": ("滴滴出行", "滴滴", "DiDi"),
    "爱奇艺": ("爱奇艺", "iQIYI"),
    "番茄免费小说": ("番茄免费小说",),
    "番茄小说": ("番茄小说",),
    "百度地图": ("百度地图", "Baidu地图"),
    "知乎": ("知乎", "ZHIHU"),
    "红果短剧": ("红果短剧", "红果"),
    "网易云音乐": ("网易云音乐", "网易云", "NetEase云音乐"),
    "美团": ("美团", "Meituan", "meituan"),
    "美柚": ("美柚",),
    "肯德基": ("肯德基", "KFC"),
    "腾讯新闻": ("腾讯新闻", "Tencent新闻"),
    "腾讯视频": ("腾讯视频", "Tencent Video"),
    "芒果TV": ("芒果TV", "Mango TV"),
    "豆包": ("豆包", "DOUBAO"),
    "豆瓣": ("豆瓣", "Douban", "douban"),
    "贝壳找房": ("贝壳找房", "贝壳"),
    "铁路12306": ("铁路12306", "铁路 12306", "12306"),
    "飞书": ("飞书", "Feishu", "feishu"),
    "饿了么": ("饿了么", "ELEME"),
    "高德地图": ("高德地图", "高德", "Amap", "amap"),
    "bilibili": ("bilibili", "B站", "b站", "哔哩哔哩"),
}


FAST_PATH_APPS: Final[tuple[str, ...]] = tuple(FAST_PATH_APP_ALIASES.keys())


def _build_alias_indices() -> tuple[dict[str, str], dict[str, str]]:
    exact: dict[str, str] = {}
    casefolded: dict[str, str] = {}
    for canonical, aliases in FAST_PATH_APP_ALIASES.items():
        for alias in aliases:
            folded = alias.casefold()
            existing = casefolded.get(folded)
            if existing is not None and existing != canonical:
                raise ValueError(
                    f"Fast Path alias collision: {alias!r} maps to both "
                    f"{existing!r} and {canonical!r}"
                )
            exact[alias] = canonical
            casefolded[folded] = canonical
    return exact, casefolded


_EXACT_ALIAS_TO_CANONICAL, _CASEFOLDED_ALIAS_TO_CANONICAL = _build_alias_indices()


def normalize_fast_path_app_name(app_name: str | None) -> str | None:
    """
    Normalize a user-provided app name to the Fast Path v1 canonical name.

    Args:
        app_name: Raw app name candidate from the parsed command.

    Returns:
        Canonical app name if it belongs to the Fast Path v1 demo set,
        otherwise None.
    """
    if not app_name:
        return None

    candidate = app_name.strip().strip("，。！？；：,.!?;:")
    if not candidate:
        return None

    if candidate in _EXACT_ALIAS_TO_CANONICAL:
        return _EXACT_ALIAS_TO_CANONICAL[candidate]

    return _CASEFOLDED_ALIAS_TO_CANONICAL.get(candidate.casefold())


def is_fast_path_app(app_name: str | None) -> bool:
    """Return True when the app belongs to the Fast Path v1 demo set."""
    return normalize_fast_path_app_name(app_name) is not None


def list_fast_path_apps() -> list[str]:
    """Return the canonical Fast Path v1 demo app list."""
    return list(FAST_PATH_APPS)
