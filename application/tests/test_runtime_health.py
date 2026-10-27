# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from phone_agent.runtime_health import RuntimeHealthConfig, RuntimeHealthMonitor


class RuntimeHealthMonitorTests(unittest.TestCase):
    def test_slow_path_deadline_miss_persists_degraded_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            health_path = Path(temp_dir) / "runtime_health.json"
            config = RuntimeHealthConfig(
                enabled=True,
                health_file=health_path,
                degraded_cooldown_seconds=300.0,
            )

            monitor = RuntimeHealthMonitor(config)
            self.assertFalse(monitor.is_degraded(now=1000.0))

            monitor.record_slow_path_result(
                success=True,
                e2e_latency_ms=65000.0,
                deadline_ms=60000.0,
                now=1000.0,
            )

            reloaded = RuntimeHealthMonitor(config)
            self.assertTrue(reloaded.is_degraded(now=1001.0))
            self.assertFalse(reloaded.is_degraded(now=1401.0))

    def test_cloud_probe_failure_triggers_degraded_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            health_path = Path(temp_dir) / "runtime_health.json"
            monitor = RuntimeHealthMonitor(
                RuntimeHealthConfig(
                    enabled=True,
                    health_file=health_path,
                    cloud_probe_deadline_ms=8000.0,
                    degraded_cooldown_seconds=60.0,
                )
            )

            monitor.record_cloud_probe(
                available=False,
                latency_ms=12000.0,
                now=2000.0,
            )

            self.assertTrue(RuntimeHealthMonitor(monitor.config).is_degraded(now=2001.0))


class SlidingWindowDetectorTests(unittest.TestCase):
    """Per-step sliding-window detector (issue #6).

    Samples are per-step cloud round-trip times (model_total_ms), fed via
    record_step_latency. Default thresholds enter=8000ms / exit=5000ms reflect
    the X2 Pro baseline (per-step P50~3.3s / P95~6.5s).
    """

    def _make_monitor(
        self, health_path: Path, *, half_open_interval: float = 15.0
    ) -> RuntimeHealthMonitor:
        return RuntimeHealthMonitor(
            RuntimeHealthConfig(
                enabled=True,
                health_file=health_path,
                detector="sliding_window",
                window_size=10,
                window_min_samples=5,
                window_percentile=75.0,
                window_enter_ms=8000.0,
                window_exit_ms=5000.0,
                half_open_probe_interval_seconds=half_open_interval,
            )
        )

    def _feed_steps(self, monitor: RuntimeHealthMonitor, latencies):
        for latency in latencies:
            monitor.record_step_latency(model_total_ms=latency)

    def test_enters_degraded_when_percentile_exceeds_threshold(self):
        # Six steps all at 10s (injected slow network) -> P90 > 8000 -> degraded.
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self._make_monitor(Path(temp_dir) / "rh.json")
            self._feed_steps(monitor, [10000.0] * 6)
            self.assertTrue(monitor.is_degraded())

    def test_tolerates_single_spike(self):
        # Nine normal per-step samples (~3.3s) + one big spike: P90 stays below
        # the 8000 enter threshold, so a single transient does NOT trip degraded.
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self._make_monitor(Path(temp_dir) / "rh.json")
            self._feed_steps(monitor, [3300.0] * 9 + [20000.0])
            self.assertFalse(monitor.is_degraded())

    def test_recovers_when_bad_samples_slide_out(self):
        # Sustained slow steps enter degraded; then a run of normal per-step
        # samples pushes the bad ones out and it recovers on its own (no fixed
        # cooldown timer).
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self._make_monitor(Path(temp_dir) / "rh.json")
            self._feed_steps(monitor, [11000.0] * 10)
            self.assertTrue(monitor.is_degraded())
            self._feed_steps(monitor, [3300.0] * 10)
            self.assertFalse(monitor.is_degraded())

    def test_holds_state_with_insufficient_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self._make_monitor(Path(temp_dir) / "rh.json")
            self._feed_steps(monitor, [99999.0] * 3)
            self.assertFalse(monitor.is_degraded())

    def test_state_persists_across_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            health_path = Path(temp_dir) / "rh.json"
            monitor = self._make_monitor(health_path)
            self._feed_steps(monitor, [11000.0] * 6)
            self.assertTrue(monitor.is_degraded())
            reloaded = self._make_monitor(health_path)
            self.assertTrue(reloaded.is_degraded())

    def test_cloud_probe_miss_triggers_sliding_window_degraded_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self._make_monitor(
                Path(temp_dir) / "rh.json", half_open_interval=30.0
            )
            self._feed_steps(monitor, [3300.0] * 10)
            self.assertFalse(monitor.is_degraded())

            monitor.record_cloud_probe(
                available=False,
                latency_ms=12000.0,
                now=2000.0,
            )

            self.assertTrue(monitor.is_degraded())
            self.assertFalse(monitor.claim_half_open_probe(now=2020.0))
            self.assertTrue(monitor.claim_half_open_probe(now=2031.0))

    def test_half_open_probe_is_rate_limited_while_degraded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self._make_monitor(
                Path(temp_dir) / "rh.json", half_open_interval=30.0
            )
            for offset in range(10):
                monitor.record_step_latency(
                    model_total_ms=11000.0,
                    now=1000.0 + offset,
                )

            self.assertTrue(monitor.is_degraded())
            self.assertFalse(monitor.claim_half_open_probe(now=1020.0))
            self.assertTrue(monitor.claim_half_open_probe(now=1035.0))
            self.assertFalse(monitor.claim_half_open_probe(now=1040.0))
            self.assertTrue(monitor.claim_half_open_probe(now=1066.0))

    def test_half_open_probe_step_samples_can_recover_degraded_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            health_path = Path(temp_dir) / "rh.json"
            monitor = self._make_monitor(health_path, half_open_interval=0.0)
            for offset in range(10):
                monitor.record_step_latency(
                    model_total_ms=11000.0,
                    now=1000.0 + offset,
                )
            self.assertTrue(monitor.is_degraded())
            self.assertTrue(monitor.claim_half_open_probe(now=1010.0))

            for offset in range(10):
                monitor.record_step_latency(
                    model_total_ms=3300.0,
                    now=1011.0 + offset,
                )
            monitor.record_half_open_probe_result(
                success=True,
                e2e_latency_ms=30000.0,
                deadline_ms=50000.0,
                now=1022.0,
            )

            self.assertFalse(monitor.is_degraded())
            self.assertFalse(self._make_monitor(health_path).is_degraded())

    def test_half_open_probe_e2e_miss_does_not_override_window_recovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            health_path = Path(temp_dir) / "rh.json"
            monitor = self._make_monitor(health_path, half_open_interval=30.0)
            for offset in range(10):
                monitor.record_step_latency(
                    model_total_ms=11000.0,
                    now=1000.0 + offset,
                )
            self.assertTrue(monitor.claim_half_open_probe(now=1040.0))

            for offset in range(10):
                monitor.record_step_latency(
                    model_total_ms=3300.0,
                    now=1041.0 + offset,
                )
            monitor.record_half_open_probe_result(
                success=True,
                e2e_latency_ms=65000.0,
                deadline_ms=50000.0,
                now=1052.0,
            )

            self.assertFalse(monitor.is_degraded())
            self.assertFalse(self._make_monitor(health_path).is_degraded())

    def test_half_open_probe_keeps_degraded_when_window_is_still_slow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = self._make_monitor(
                Path(temp_dir) / "rh.json", half_open_interval=30.0
            )
            for offset in range(10):
                monitor.record_step_latency(
                    model_total_ms=11000.0,
                    now=1000.0 + offset,
                )
            self.assertTrue(monitor.claim_half_open_probe(now=1040.0))

            monitor.record_half_open_probe_result(
                success=True,
                e2e_latency_ms=30000.0,
                deadline_ms=50000.0,
                now=1050.0,
            )

            self.assertTrue(monitor.is_degraded())
            self.assertFalse(monitor.claim_half_open_probe(now=1079.0))
            self.assertTrue(monitor.claim_half_open_probe(now=1081.0))


if __name__ == "__main__":
    unittest.main()
