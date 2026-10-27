# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors: routing/runtime integration and/or release packaging; upstream attribution retained.
"""Configuration exports for Phone Agent."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path


def _read_prompt_file(variable: str) -> str:
    filename = os.environ.get(variable)
    if not filename:
        raise RuntimeError(
            f"Research prompts are not distributed. Set {variable} to your own UTF-8 prompt file. "
            "Saved-output analysis requires no prompt or model."
        )
    value = Path(filename).expanduser().read_text(encoding="utf-8")
    if not value.strip():
        raise ValueError(f"{variable} must refer to a nonempty prompt file.")
    return value


def get_system_prompt(lang: str = "cn") -> str:
    variable = "DUALROUTE_EXECUTION_PROMPT_EN_FILE" if lang == "en" else "DUALROUTE_EXECUTION_PROMPT_ZH_FILE"
    return _read_prompt_file(variable)


def get_messages(*args, **kwargs):
    """Proxy to the i18n message registry."""
    return import_module("phone_agent.config.i18n").get_messages(*args, **kwargs)


def get_message(*args, **kwargs):
    """Proxy to the i18n message registry."""
    return import_module("phone_agent.config.i18n").get_message(*args, **kwargs)


_LAZY_EXPORTS = {
    "APP_PACKAGES": ("phone_agent.config.apps", "APP_PACKAGES"),
    "APP_PACKAGES_IOS": ("phone_agent.config.apps_ios", "APP_PACKAGES_IOS"),
    "FAST_PATH_APPS": ("phone_agent.config.fast_path_apps", "FAST_PATH_APPS"),
    "normalize_fast_path_app_name": (
        "phone_agent.config.fast_path_apps",
        "normalize_fast_path_app_name",
    ),
    "list_fast_path_apps": ("phone_agent.config.fast_path_apps", "list_fast_path_apps"),
    "is_fast_path_app": ("phone_agent.config.fast_path_apps", "is_fast_path_app"),
    "TIMING_CONFIG": ("phone_agent.config.timing", "TIMING_CONFIG"),
    "TimingConfig": ("phone_agent.config.timing", "TimingConfig"),
    "ActionTimingConfig": ("phone_agent.config.timing", "ActionTimingConfig"),
    "DeviceTimingConfig": ("phone_agent.config.timing", "DeviceTimingConfig"),
    "ConnectionTimingConfig": ("phone_agent.config.timing", "ConnectionTimingConfig"),
    "get_timing_config": ("phone_agent.config.timing", "get_timing_config"),
    "update_timing_config": ("phone_agent.config.timing", "update_timing_config"),
}


def __getattr__(name: str):
    if name in {"SYSTEM_PROMPT", "SYSTEM_PROMPT_ZH", "SYSTEM_PROMPT_EN"}:
        return get_system_prompt("en" if name == "SYSTEM_PROMPT_EN" else "cn")
    if name in _LAZY_EXPORTS:
        module_name, attribute_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "APP_PACKAGES",
    "APP_PACKAGES_IOS",
    "FAST_PATH_APPS",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_ZH",
    "SYSTEM_PROMPT_EN",
    "get_system_prompt",
    "get_messages",
    "get_message",
    "normalize_fast_path_app_name",
    "list_fast_path_apps",
    "is_fast_path_app",
    "TIMING_CONFIG",
    "TimingConfig",
    "ActionTimingConfig",
    "DeviceTimingConfig",
    "ConnectionTimingConfig",
    "get_timing_config",
    "update_timing_config",
]
