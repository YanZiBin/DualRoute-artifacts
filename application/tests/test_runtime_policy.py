# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import unittest

from phone_agent.experiment_logger import ROUTE_FAST, ROUTE_REJECT, ROUTE_SLOW
from phone_agent.runtime_policy import (
    ADMISSION_ADMIT,
    ADMISSION_REJECT,
    RUNTIME_DEGRADED,
    RUNTIME_NORMAL,
    RuntimePolicyConfig,
    decide_runtime_admission,
)


class RuntimePolicyTests(unittest.TestCase):
    def test_degraded_mode_rejects_slow_path_but_keeps_fast_path_admissible(self):
        config = RuntimePolicyConfig(
            enabled=True,
            runtime_degraded_mode="on",
            degraded_slow_path_admission="reject",
            fast_path_deadline_ms=800.0,
            slow_path_deadline_ms=5000.0,
        )

        slow_decision = decide_runtime_admission(ROUTE_SLOW, config)
        self.assertEqual(slow_decision.runtime_mode, RUNTIME_DEGRADED)
        self.assertEqual(slow_decision.admission_decision, ADMISSION_REJECT)
        self.assertEqual(slow_decision.deadline_ms, 5000.0)

        fast_decision = decide_runtime_admission(ROUTE_FAST, config)
        self.assertEqual(fast_decision.runtime_mode, RUNTIME_DEGRADED)
        self.assertEqual(fast_decision.admission_decision, ADMISSION_ADMIT)
        self.assertEqual(fast_decision.deadline_ms, 800.0)

    def test_auto_mode_uses_degraded_runtime_when_network_profile_is_not_normal(self):
        config = RuntimePolicyConfig(
            enabled=True,
            runtime_degraded_mode="auto",
            network_profile="jitter",
            degraded_slow_path_admission="reject",
        )

        decision = decide_runtime_admission(ROUTE_SLOW, config)

        self.assertEqual(decision.runtime_mode, RUNTIME_DEGRADED)
        self.assertEqual(decision.admission_decision, ADMISSION_REJECT)

    def test_auto_mode_prefers_runtime_monitor_state_when_monitor_is_enabled(self):
        config = RuntimePolicyConfig(
            enabled=True,
            runtime_degraded_mode="auto",
            runtime_monitor_enabled=True,
            runtime_monitor_degraded=False,
            network_profile="jitter",
            cloud_delay_ms=20000.0,
            degraded_slow_path_admission="reject",
        )

        decision = decide_runtime_admission(ROUTE_SLOW, config)

        self.assertEqual(decision.runtime_mode, RUNTIME_NORMAL)
        self.assertEqual(decision.admission_decision, ADMISSION_ADMIT)

        degraded_config = RuntimePolicyConfig(
            enabled=True,
            runtime_degraded_mode="auto",
            runtime_monitor_enabled=True,
            runtime_monitor_degraded=True,
            degraded_slow_path_admission="reject",
        )

        degraded_decision = decide_runtime_admission(ROUTE_SLOW, degraded_config)

        self.assertEqual(degraded_decision.runtime_mode, RUNTIME_DEGRADED)
        self.assertEqual(degraded_decision.admission_decision, ADMISSION_REJECT)

    def test_half_open_probe_admits_one_slow_path_in_degraded_reject_mode(self):
        config = RuntimePolicyConfig(
            enabled=True,
            runtime_degraded_mode="auto",
            runtime_monitor_enabled=True,
            runtime_monitor_degraded=True,
            degraded_slow_path_admission="reject",
            half_open_probe_allowed=True,
        )

        decision = decide_runtime_admission(ROUTE_SLOW, config)

        self.assertEqual(decision.runtime_mode, RUNTIME_DEGRADED)
        self.assertEqual(decision.admission_decision, ADMISSION_ADMIT)
        self.assertTrue(decision.half_open_probe)
        self.assertTrue(decision.to_log_payload()["half_open_probe"])

    def test_disabled_policy_only_reports_normal_admission_metadata(self):
        config = RuntimePolicyConfig(enabled=False, runtime_degraded_mode="on")

        decision = decide_runtime_admission(ROUTE_REJECT, config)

        self.assertEqual(decision.runtime_mode, RUNTIME_NORMAL)
        self.assertEqual(decision.admission_decision, ADMISSION_ADMIT)
        self.assertIsNone(decision.deadline_ms)


if __name__ == "__main__":
    unittest.main()
