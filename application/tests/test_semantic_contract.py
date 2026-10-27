# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import unittest

from phone_agent.actions.handler import ActionHandler
from phone_agent.experiment_logger import ROUTE_FAST, ROUTE_REJECT, ROUTE_SLOW
from phone_agent.semantic_contract import (
    CANONICAL_ROUTES,
    ROUTE_FAST_PATH,
    ROUTE_REJECT as CONTRACT_REJECT,
    ROUTE_SLOW_PATH,
    route_success,
    semantic_success,
)


class SemanticContractTests(unittest.TestCase):
    def test_logger_aliases_share_canonical_route_values(self):
        self.assertEqual({ROUTE_FAST, ROUTE_SLOW, ROUTE_REJECT}, CANONICAL_ROUTES)
        self.assertEqual((ROUTE_FAST, ROUTE_SLOW, ROUTE_REJECT),
                         (ROUTE_FAST_PATH, ROUTE_SLOW_PATH, CONTRACT_REJECT))

    def test_execute_allows_fast_or_slow_but_not_reject(self):
        self.assertTrue(
            route_success(expected_outcome="execute", actual_route="fast_path")
        )
        self.assertTrue(
            route_success(expected_outcome="execute", actual_route="slow_path")
        )
        self.assertFalse(
            route_success(expected_outcome="execute", actual_route="reject")
        )

    def test_refuse_is_semantically_successful_only_when_nothing_runs(self):
        self.assertTrue(
            semantic_success(
                expected_outcome="refuse",
                actual_route="reject",
                machine_success=True,
            )
        )
        self.assertFalse(
            semantic_success(
                expected_outcome="refuse",
                actual_route="slow_path",
                machine_success=True,
            )
        )

    def test_user_action_required_is_an_intermediate_state(self):
        self.assertIsNone(
            semantic_success(
                expected_outcome="execute",
                actual_route="slow_path",
                machine_success=False,
                user_action_required=True,
            )
        )

    def test_takeover_action_is_a_pause_signal(self):
        messages: list[str] = []
        handler = ActionHandler(takeover_callback=messages.append)
        result = handler.execute(
            {"_metadata": "do", "action": "Take_over", "message": "请登录"},
            100,
            100,
        )
        self.assertTrue(result.should_finish)
        self.assertTrue(result.user_action_required)
        self.assertEqual(messages, ["请登录"])


if __name__ == "__main__":
    unittest.main()
