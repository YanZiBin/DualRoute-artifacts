# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

"""Deadline and admission-control policy for edge-cloud task routing."""

from dataclasses import dataclass
from typing import Any

from phone_agent.experiment_logger import ROUTE_FAST, ROUTE_REJECT, ROUTE_SLOW


RUNTIME_NORMAL = "normal"
RUNTIME_DEGRADED = "degraded"
ADMISSION_ADMIT = "admit"
ADMISSION_REJECT = "reject"


@dataclass(frozen=True)
class RuntimePolicyConfig:
    enabled: bool = False
    fast_path_deadline_ms: float = 800.0
    slow_path_deadline_ms: float = 5000.0
    reject_deadline_ms: float = 500.0
    runtime_degraded_mode: str = "off"
    degraded_slow_path_admission: str = "reject"
    network_profile: str = "normal"
    cloud_delay_ms: float = 0.0
    cloud_fail_mode: str = "none"
    runtime_monitor_enabled: bool = False
    runtime_monitor_degraded: bool = False
    runtime_degraded_until: float | None = None
    half_open_probe_allowed: bool = False


@dataclass(frozen=True)
class RuntimeAdmissionDecision:
    predicted_route: str
    runtime_mode: str
    admission_decision: str
    admission_reason: str
    deadline_ms: float | None
    deadline_metric: str | None
    network_profile: str
    cloud_delay_ms: float
    cloud_fail_mode: str
    runtime_monitor_enabled: bool
    runtime_monitor_degraded: bool
    runtime_degraded_until: float | None
    half_open_probe: bool = False

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "runtime_mode": self.runtime_mode,
            "admission_decision": self.admission_decision,
            "admission_reason": self.admission_reason,
            "deadline_ms": self.deadline_ms,
            "deadline_metric": self.deadline_metric,
            "network_profile": self.network_profile,
            "cloud_delay_ms": self.cloud_delay_ms,
            "cloud_fail_mode": self.cloud_fail_mode,
            "runtime_monitor_enabled": self.runtime_monitor_enabled,
            "runtime_monitor_degraded": self.runtime_monitor_degraded,
            "runtime_degraded_until": self.runtime_degraded_until,
            "half_open_probe": self.half_open_probe,
        }


def decide_runtime_admission(
    predicted_route: str,
    config: RuntimePolicyConfig,
) -> RuntimeAdmissionDecision:
    if not config.enabled:
        return RuntimeAdmissionDecision(
            predicted_route=predicted_route,
            runtime_mode=RUNTIME_NORMAL,
            admission_decision=ADMISSION_ADMIT,
            admission_reason="deadline policy disabled",
            deadline_ms=None,
            deadline_metric=None,
            network_profile=config.network_profile,
            cloud_delay_ms=config.cloud_delay_ms,
            cloud_fail_mode=config.cloud_fail_mode,
            runtime_monitor_enabled=config.runtime_monitor_enabled,
            runtime_monitor_degraded=config.runtime_monitor_degraded,
            runtime_degraded_until=config.runtime_degraded_until,
            half_open_probe=False,
        )

    runtime_mode = _runtime_mode(config)
    deadline_ms = _deadline_for_route(predicted_route, config)
    deadline_metric = _deadline_metric_for_route(predicted_route)

    if (
        predicted_route == ROUTE_SLOW
        and runtime_mode == RUNTIME_DEGRADED
        and config.degraded_slow_path_admission == ADMISSION_REJECT
    ):
        if config.half_open_probe_allowed:
            return RuntimeAdmissionDecision(
                predicted_route=predicted_route,
                runtime_mode=runtime_mode,
                admission_decision=ADMISSION_ADMIT,
                admission_reason="slow path admitted as half-open degraded-mode probe",
                deadline_ms=deadline_ms,
                deadline_metric=deadline_metric,
                network_profile=config.network_profile,
                cloud_delay_ms=config.cloud_delay_ms,
                cloud_fail_mode=config.cloud_fail_mode,
                runtime_monitor_enabled=config.runtime_monitor_enabled,
                runtime_monitor_degraded=config.runtime_monitor_degraded,
                runtime_degraded_until=config.runtime_degraded_until,
                half_open_probe=True,
            )
        return RuntimeAdmissionDecision(
            predicted_route=predicted_route,
            runtime_mode=runtime_mode,
            admission_decision=ADMISSION_REJECT,
            admission_reason="slow path rejected by degraded-mode admission control",
            deadline_ms=deadline_ms,
            deadline_metric=deadline_metric,
            network_profile=config.network_profile,
            cloud_delay_ms=config.cloud_delay_ms,
            cloud_fail_mode=config.cloud_fail_mode,
            runtime_monitor_enabled=config.runtime_monitor_enabled,
            runtime_monitor_degraded=config.runtime_monitor_degraded,
            runtime_degraded_until=config.runtime_degraded_until,
            half_open_probe=False,
        )

    return RuntimeAdmissionDecision(
        predicted_route=predicted_route,
        runtime_mode=runtime_mode,
        admission_decision=ADMISSION_ADMIT,
        admission_reason="route admitted",
        deadline_ms=deadline_ms,
        deadline_metric=deadline_metric,
        network_profile=config.network_profile,
        cloud_delay_ms=config.cloud_delay_ms,
        cloud_fail_mode=config.cloud_fail_mode,
        runtime_monitor_enabled=config.runtime_monitor_enabled,
        runtime_monitor_degraded=config.runtime_monitor_degraded,
        runtime_degraded_until=config.runtime_degraded_until,
        half_open_probe=False,
    )


def _runtime_mode(config: RuntimePolicyConfig) -> str:
    if config.runtime_degraded_mode == "on":
        return RUNTIME_DEGRADED
    if config.runtime_degraded_mode == "auto" and config.runtime_monitor_enabled:
        return RUNTIME_DEGRADED if config.runtime_monitor_degraded else RUNTIME_NORMAL
    if config.runtime_degraded_mode == "auto" and _has_network_degradation(config):
        return RUNTIME_DEGRADED
    return RUNTIME_NORMAL


def _has_network_degradation(config: RuntimePolicyConfig) -> bool:
    return (
        config.network_profile != "normal"
        or config.cloud_delay_ms > 0
        or config.cloud_fail_mode != "none"
    )


def _deadline_for_route(
    predicted_route: str,
    config: RuntimePolicyConfig,
) -> float | None:
    if predicted_route == ROUTE_FAST:
        return config.fast_path_deadline_ms
    if predicted_route == ROUTE_SLOW:
        return config.slow_path_deadline_ms
    if predicted_route == ROUTE_REJECT:
        return config.reject_deadline_ms
    return None


def _deadline_metric_for_route(predicted_route: str) -> str | None:
    if predicted_route == ROUTE_FAST:
        return "ttfa_ms"
    if predicted_route in {ROUTE_SLOW, ROUTE_REJECT}:
        return "e2e_latency_ms"
    return None
