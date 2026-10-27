# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
"""Small, deterministic helpers for the final DualRoute outcome contract."""

from __future__ import annotations

from enum import Enum


class ExpectedOutcome(str, Enum):
    EXECUTE = "execute"
    REFUSE = "refuse"


class CanonicalRoute(str, Enum):
    FAST_PATH = "fast_path"
    SLOW_PATH = "slow_path"
    REJECT = "reject"


ROUTE_FAST_PATH = CanonicalRoute.FAST_PATH.value
ROUTE_SLOW_PATH = CanonicalRoute.SLOW_PATH.value
ROUTE_REJECT = CanonicalRoute.REJECT.value
CANONICAL_ROUTES = frozenset({ROUTE_FAST_PATH, ROUTE_SLOW_PATH, ROUTE_REJECT})


def route_success(*, expected_outcome: str | None, actual_route: str | None,
                  expected_route: str | None = None) -> bool | None:
    """Evaluate route responsibility independently from task completion."""
    if actual_route is None:
        return None
    if expected_route is not None:
        return actual_route == expected_route
    if expected_outcome == ExpectedOutcome.REFUSE:
        return actual_route == ROUTE_REJECT
    if expected_outcome == ExpectedOutcome.EXECUTE:
        return actual_route in {ROUTE_FAST_PATH, ROUTE_SLOW_PATH}
    return None


def semantic_success(*, expected_outcome: str | None, actual_route: str | None,
                     machine_success: bool | None,
                     user_action_required: bool = False) -> bool | None:
    """Evaluate the user-visible terminal state without hiding route failures."""
    if user_action_required or machine_success is None:
        return None
    if expected_outcome == ExpectedOutcome.EXECUTE:
        return bool(machine_success) and actual_route != ROUTE_REJECT
    if expected_outcome == ExpectedOutcome.REFUSE:
        return bool(machine_success) and actual_route == ROUTE_REJECT
    return bool(machine_success)
