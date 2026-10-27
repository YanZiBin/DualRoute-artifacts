# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors: routing/runtime integration and/or release packaging; upstream attribution retained.
"""
Phone Agent package exports.

Keep imports lazy so small utility modules such as Fast Path routing can be
loaded without pulling in the full agent stack.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"
__all__ = ["PhoneAgent", "IOSPhoneAgent"]


def __getattr__(name: str):
    if name == "PhoneAgent":
        return import_module("phone_agent.agent").PhoneAgent
    if name == "IOSPhoneAgent":
        return import_module("phone_agent.agent_ios").IOSPhoneAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
