# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors: routing/runtime integration and/or release packaging; upstream attribution retained.
"""Compatibility export; no research execution prompt is distributed."""
from phone_agent.config import get_system_prompt


def __getattr__(name):
    if name == "SYSTEM_PROMPT":
        return get_system_prompt("cn")
    raise AttributeError(name)
