# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

"""Persistent runtime health state shared by per-task launcher processes."""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeHealthConfig:
    enabled: bool = False
    health_file: Path = Path("experiment_logs/runtime_health.json")
    cloud_probe_deadline_ms: float = 8000.0
    degraded_cooldown_seconds: float = 300.0
    max_events: int = 20
    # Degraded-mode detector (reviewer issue #6).
    # "single_miss" keeps the original behavior: any single probe/slow-path
    # miss enters degraded mode for a fixed cooldown.
    # "sliding_window" makes a finer-grained decision from the latency
    # percentile over the most recent PER-STEP cloud round-trip times
    # (model_total_ms of each Slow Path step), with hysteresis and adaptive
    # (non-fixed) recovery. Per-step latency is the clean network-health signal:
    # it is unaffected by task complexity / step count (an error-recovering task
    # that takes many steps stays per-step normal), unlike task-level e2e which
    # conflates "task is long" with "network is slow".
    detector: str = "single_miss"
    window_size: int = 10
    window_min_samples: int = 5
    # Symmetric hysteresis on the P75 of the per-step window. P75 (not P90) so a
    # single transient spike does not trip it (at 9 good + 1 bad, P75 is still a
    # good sample) yet recovery is not so sticky that a few residual slow samples
    # keep the monitor degraded forever (the failure mode of P90 on a size-10
    # window). enter_ms > exit_ms gives the hysteresis band that prevents
    # chattering around a single threshold. Tuned on a spike-recover trace to a
    # clean single enter / single exit; see handoff issue-6 notes.
    window_percentile: float = 75.0
    # Per-step thresholds (ms). Baseline per-step cloud round-trip on the
    # X2 Pro is P50~3.3s / P90~5.5s / P95~6.5s (from history steps.jsonl,
    # n=355). enter=8000 sits above normal P95 so only injected network delay
    # trips it; exit=5000 gives a ~3s hysteresis band for adaptive recovery.
    window_enter_ms: float = 8000.0
    window_exit_ms: float = 5000.0
    # While sliding_window is degraded and admission is configured to reject
    # slow-path work, periodically admit one Slow Path request as a half-open
    # probe. Its per-step latencies feed the same window and can drive recovery.
    half_open_probe_interval_seconds: float = 15.0


class RuntimeHealthMonitor:
    """Small JSON-backed health monitor for Slow Path and cloud availability."""

    def __init__(self, config: RuntimeHealthConfig, *, event_sink: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.config = config
        self.health_file = config.health_file.expanduser()
        self._event_sink = event_sink
        self._state = self._load_state()
        self._emit_snapshot("initial_load", self.is_degraded(), time.time())

    def record_snapshot(self, cause: str = "final") -> None:
        self._emit_snapshot(cause, self.is_degraded(), time.time())

    def _emit_snapshot(self, cause: str, before: bool, now: float, **extras: Any) -> None:
        if self._event_sink is None:
            return
        values = list(self._state.get("step_latencies", []))
        self._event_sink("health_snapshot", {
            "state_before": "degraded" if before else "normal",
            "state_after": "degraded" if self.is_degraded(now=now) else "normal",
            "cause": cause, "window_values_ms": values, "sample_count": len(values),
            "window_p75_ms": _percentile(values, 75.0) if values else None,
            "window_percentile_ms": _percentile(values, self.config.window_percentile) if values else None,
            "next_halfopen_eligible_at_unix": self._state.get("half_open_next_probe_after", 0.0),
            "state_timestamp_unix": now, **extras,
        })

    def is_degraded(self, now: float | None = None) -> bool:
        if not self.config.enabled:
            return False
        if self.config.detector == "sliding_window":
            return bool(self._state.get("window_degraded", False))
        now = time.time() if now is None else now
        degraded_until = _as_float(self._state.get("degraded_until"), 0.0)
        return degraded_until > now

    def degraded_until(self) -> float | None:
        if self.config.detector == "sliding_window":
            return None
        value = _as_float(self._state.get("degraded_until"), 0.0)
        return value or None

    def record_cloud_probe(
        self,
        *,
        available: bool,
        latency_ms: float,
        now: float | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        now = time.time() if now is None else now
        before = self.is_degraded(now=now)
        event = {
            "type": "cloud_probe",
            "timestamp": now,
            "available": available,
            "latency_ms": round(latency_ms, 3),
            "deadline_ms": self.config.cloud_probe_deadline_ms,
            "miss": (not available) or latency_ms > self.config.cloud_probe_deadline_ms,
        }
        self._append_event(event)
        self._update_degraded_state(event, now)
        self._save_state()
        self._emit_snapshot("availability_unavailable" if event["miss"] else "cloud_probe", before, now, availability_available=available,
                            probe_elapsed_ms=latency_ms, probe_deadline_miss=event["miss"])

    def record_slow_path_result(
        self,
        *,
        success: bool,
        e2e_latency_ms: float,
        deadline_ms: float | None,
        now: float | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        now = time.time() if now is None else now
        before = self.is_degraded(now=now)
        miss = False
        if deadline_ms is not None:
            miss = e2e_latency_ms > deadline_ms
        event = {
            "type": "slow_path_result",
            "timestamp": now,
            "success": success,
            "e2e_latency_ms": round(e2e_latency_ms, 3),
            "deadline_ms": deadline_ms,
            "miss": miss,
        }
        self._append_event(event)
        # Per-step detector is driven by record_step_latency, not by task-level
        # results; only the single_miss detector reacts to slow_path_result.
        if self.config.detector != "sliding_window":
            self._update_degraded_state(event, now)
        self._save_state()
        self._emit_snapshot("slow_path_result", before, now, success=success, e2e_latency_ms=e2e_latency_ms)

    def claim_half_open_probe(self, now: float | None = None) -> bool:
        """Return True when one rejected Slow Path request should be probed."""
        event_now = time.time() if now is None else now
        before = self.is_degraded(now=event_now)
        if (
            not self.config.enabled
            or self.config.detector != "sliding_window"
            or not self.is_degraded(now=now)
        ):
            self._emit_halfopen_claim(False, "not_degraded_or_disabled", before, event_now)
            return False
        now = time.time() if now is None else now
        next_probe_after = _as_float(
            self._state.get("half_open_next_probe_after"), 0.0
        )
        if now < next_probe_after:
            self._emit_halfopen_claim(False, "not_eligible_yet", before, now, next_probe_after)
            return False
        self._state["half_open_next_probe_after"] = (
            now + max(0.0, self.config.half_open_probe_interval_seconds)
        )
        self._append_event(
            {
                "type": "half_open_probe_claimed",
                "timestamp": now,
                "next_probe_after": round(
                    self._state["half_open_next_probe_after"], 3
                ),
            }
        )
        self._save_state()
        self._emit_halfopen_claim(True, "claimed", before, now, next_probe_after)
        return True

    def _emit_halfopen_claim(self, claimed: bool, cause: str, before: bool, now: float, eligible_at: float = 0.0) -> None:
        self._emit_snapshot("halfopen_claim:" + cause, before, now)
        if self._event_sink is not None:
            self._event_sink("halfopen_claim", {
                "claimed": claimed, "cause": cause, "state_timestamp_unix": now,
                "eligible_at_unix": eligible_at,
                "next_halfopen_eligible_at_unix": self._state.get("half_open_next_probe_after", 0.0),
            })

    def record_half_open_probe_result(
        self,
        *,
        success: bool,
        e2e_latency_ms: float,
        deadline_ms: float | None,
        now: float | None = None,
    ) -> None:
        """Record the result of a half-open Slow Path probe.

        The latency window itself is still driven by record_step_latency(); this
        result event records task-level success and deadline outcome without
        treating task complexity as a network-health signal.
        """
        if not self.config.enabled or self.config.detector != "sliding_window":
            return
        now = time.time() if now is None else now
        before = self.is_degraded(now=now)
        miss = False
        if deadline_ms is not None:
            miss = e2e_latency_ms > deadline_ms
        event = {
            "type": "half_open_probe_result",
            "timestamp": now,
            "success": success,
            "e2e_latency_ms": round(e2e_latency_ms, 3),
            "deadline_ms": deadline_ms,
            "miss": miss,
        }
        self._append_event(event)
        self._evaluate_window(now)
        if self._state.get("window_degraded", False):
            self._schedule_half_open_probe(now)
        else:
            self._state["half_open_next_probe_after"] = 0.0
        self._save_state()
        self._emit_snapshot("halfopen_result", before, now, success=success, e2e_latency_ms=e2e_latency_ms)
        if self._event_sink is not None:
            self._event_sink("halfopen_result", {
                "success": success, "e2e_latency_ms": e2e_latency_ms, "deadline_ms": deadline_ms,
                "miss": miss, "state_timestamp_unix": now,
                "next_halfopen_eligible_at_unix": self._state.get("half_open_next_probe_after", 0.0),
            })

    def record_step_latency(
        self,
        *,
        model_total_ms: float,
        now: float | None = None,
    ) -> None:
        """Feed one Slow Path step's cloud round-trip time to the window.

        ``model_total_ms`` is the per-step model request time (screenshot ->
        cloud inference -> action returned), the clean network-health signal.
        Only the sliding_window detector consumes it; other detectors ignore it.
        """
        if not self.config.enabled or self.config.detector != "sliding_window":
            return
        now = time.time() if now is None else now
        before = self.is_degraded(now=now)
        samples = self._state.setdefault("step_latencies", [])
        if not isinstance(samples, list):
            samples = []
            self._state["step_latencies"] = samples
        samples.append(round(float(model_total_ms), 3))
        del samples[: max(0, len(samples) - self.config.window_size)]
        self._evaluate_window(now)
        self._save_state()
        self._emit_snapshot("step_latency", before, now, model_total_ms=model_total_ms)

    def _update_degraded_state(self, event: dict[str, Any], now: float) -> None:
        if self.config.detector == "sliding_window":
            self._evaluate_window(now)
            if event.get("type") == "cloud_probe" and event.get("miss"):
                self._state["window_degraded"] = True
                self._schedule_half_open_probe(now)
            return
        # Original single-miss behavior.
        should_degrade = bool(event.get("miss"))
        if event.get("type") == "slow_path_result" and not event.get("success", True):
            should_degrade = True
        if should_degrade:
            self._enter_degraded(now)

    def _evaluate_window(self, now: float | None = None) -> None:
        raw = self._state.get("step_latencies", [])
        if not isinstance(raw, list):
            return
        samples = [float(x) for x in raw][-self.config.window_size :]
        if len(samples) < self.config.window_min_samples:
            return
        now = time.time() if now is None else now
        degraded = bool(self._state.get("window_degraded", False))
        percentile = _percentile(samples, self.config.window_percentile)
        if not degraded and percentile > self.config.window_enter_ms:
            self._state["window_degraded"] = True
            self._schedule_half_open_probe(now)
        elif degraded and percentile < self.config.window_exit_ms:
            self._state["window_degraded"] = False
            self._state["half_open_next_probe_after"] = 0.0
        self._state["window_percentile_ms"] = round(percentile, 3)

    def _schedule_half_open_probe(self, now: float) -> None:
        self._state["half_open_next_probe_after"] = (
            now + max(0.0, self.config.half_open_probe_interval_seconds)
        )

    def _enter_degraded(self, now: float) -> None:
        new_until = now + self.config.degraded_cooldown_seconds
        current_until = _as_float(self._state.get("degraded_until"), 0.0)
        self._state["degraded_until"] = max(current_until, new_until)

    def _append_event(self, event: dict[str, Any]) -> None:
        events = self._state.setdefault("events", [])
        if not isinstance(events, list):
            events = []
            self._state["events"] = events
        events.append(event)
        del events[: max(0, len(events) - self.config.max_events)]

    def _load_state(self) -> dict[str, Any]:
        if not self.config.enabled or not self.health_file.exists():
            return _empty_state()
        try:
            payload = json.loads(self.health_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if self._event_sink is not None:
                raise
            return _empty_state()
        if not isinstance(payload, dict):
            if self._event_sink is not None:
                raise ValueError("Invalid health state object")
            return _empty_state()
        payload.setdefault("degraded_until", 0.0)
        payload.setdefault("events", [])
        payload.setdefault("window_degraded", False)
        payload.setdefault("step_latencies", [])
        payload.setdefault("half_open_next_probe_after", 0.0)
        return payload

    def _save_state(self) -> None:
        self.health_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.health_file.with_suffix(self.health_file.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, self.health_file)


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (no numpy dependency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _empty_state() -> dict[str, Any]:
    return {
        "degraded_until": 0.0,
        "events": [],
        "window_degraded": False,
        "step_latencies": [],
        "half_open_next_probe_after": 0.0,
    }


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
