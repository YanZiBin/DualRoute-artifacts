# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

"""Lightweight structured experiment logging for AutoGLM runs."""

import csv
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from phone_agent.semantic_contract import (
    ROUTE_FAST_PATH,
    ROUTE_REJECT,
    ROUTE_SLOW_PATH,
    route_success,
    semantic_success,
)


# Backward-compatible names used by main.py and existing experiment scripts.
ROUTE_FAST = ROUTE_FAST_PATH
ROUTE_SLOW = ROUTE_SLOW_PATH


def now_iso() -> str:
    """Return a local ISO-8601 timestamp with seconds precision."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def default_run_id() -> str:
    """Build a stable enough run id for ad-hoc phone-side experiments."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class ExperimentConfig:
    enabled: bool = False
    log_dir: Path = Path("experiment_logs")
    run_id: str = field(default_factory=default_run_id)
    experiment_group: str | None = None
    variant: str | None = None
    expected_route: str | None = None
    expected_success: bool | None = None
    repeat_index: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str | None = None


class ExperimentLogger:
    """Append-only JSONL/CSV logger scoped to one task process."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.run_dir = config.log_dir.expanduser() / config.run_id
        self._lock = Lock()
        self._task: dict[str, Any] | None = None
        self._task_start_perf: float | None = None
        self._first_action_ms: float | None = None
        self._first_action_name: str | None = None
        self._route_payload: dict[str, Any] | None = None

        if self.config.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._write_config()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _write_config(self) -> None:
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            return
        payload = {
            "run_id": self.config.run_id,
            "created_at": now_iso(),
            "log_dir": str(self.run_dir),
            "experiment_group": self.config.experiment_group,
            "variant": self.config.variant,
            "expected_route": self.config.expected_route,
            "expected_success": self.config.expected_success,
            "expected_outcome": self.config.expected_outcome,
            "repeat_index": self.config.repeat_index,
            "extra": self.config.extra,
        }
        config_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def start_task(self, task_text: str, task_id: str | None = None) -> str | None:
        if not self.enabled:
            return None

        resolved_task_id = task_id or f"task_{uuid.uuid4().hex[:10]}"
        with self._lock:
            self._task_start_perf = time.perf_counter()
            self._first_action_ms = None
            self._first_action_name = None
            self._route_payload = None
            self._task = {
                "run_id": self.config.run_id,
                "task_id": resolved_task_id,
                "task_text": task_text,
                "experiment_group": self.config.experiment_group,
                "variant": self.config.variant,
                "repeat_index": self.config.repeat_index,
                "expected_route": self.config.expected_route,
                "expected_success": self.config.expected_success,
                "expected_outcome": self.config.expected_outcome,
                "started_at": now_iso(),
            }
        return resolved_task_id

    def mark_first_action(self, action_name: str | None) -> None:
        if not self.enabled or not action_name:
            return
        with self._lock:
            if self._first_action_ms is not None or self._task_start_perf is None:
                return
            self._first_action_name = action_name
            self._first_action_ms = (time.perf_counter() - self._task_start_perf) * 1000

    def record_route(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        enriched = self._base_payload(payload)
        with self._lock:
            self._route_payload = enriched
        self._append_jsonl("routes.jsonl", enriched)

    def record_step(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._append_jsonl("steps.jsonl", self._base_payload(payload))

    def finish_task(
        self,
        *,
        success: bool | None,
        result_text: str | None,
        return_code: int | None = None,
        step_count: int | None = None,
        user_action_required: bool = False,
        semantic_reason: str | None = None,
        safety_violation: bool = False,
        candidate_set: list[str] | None = None,
        completed_candidate: str | None = None,
        extra_non_sensitive_action: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return

        with self._lock:
            if self._task is None or self._task_start_perf is None:
                return
            finished_at = now_iso()
            e2e_latency_ms = (time.perf_counter() - self._task_start_perf) * 1000
            route_payload = self._route_payload or {}
            ttfa_ms = self._first_action_ms
            deadline_metric = route_payload.get("deadline_metric")
            deadline_ms = route_payload.get("deadline_ms")
            deadline_latency_ms = self._deadline_latency_ms(
                deadline_metric=deadline_metric,
                e2e_latency_ms=e2e_latency_ms,
                ttfa_ms=ttfa_ms,
            )
            deadline_miss = self._deadline_miss(deadline_latency_ms, deadline_ms)
            admission_decision = route_payload.get("admission_decision")
            final_route = route_payload.get("final_route") or route_payload.get(
                "predicted_route"
            )
            semantic_result = semantic_success(
                expected_outcome=self.config.expected_outcome,
                actual_route=final_route,
                machine_success=success,
                user_action_required=user_action_required,
            )
            route_result = route_success(
                expected_outcome=self.config.expected_outcome,
                actual_route=final_route,
                expected_route=self.config.expected_route,
            )
            task_payload = {
                **self._task,
                "finished_at": finished_at,
                "return_code": return_code,
                "success": success,
                "result_text": result_text,
                "e2e_latency_ms": round(e2e_latency_ms, 3),
                "ttfa_ms": round(ttfa_ms, 3) if ttfa_ms is not None else None,
                "first_action_name": self._first_action_name,
                "step_count": step_count,
                "predicted_route": route_payload.get("predicted_route"),
                "route_label": route_payload.get("route_label"),
                "classifier_label": route_payload.get("classifier_label"),
                "classifier_outcome": route_payload.get("classifier_outcome"),
                "parser_status": route_payload.get("parser_status"),
                "guard_outcome": route_payload.get("guard_outcome"),
                "final_route": final_route,
                "reason": route_payload.get("reason"),
                "app_name": route_payload.get("app_name"),
                "raw_prediction": route_payload.get("raw_prediction"),
                "route_correct": self._route_correct(route_payload.get("predicted_route")),
                "route_success": route_result,
                "semantic_success": semantic_result,
                "semantic_reason": semantic_reason,
                "user_action_required": user_action_required,
                "safety_violation": safety_violation,
                "candidate_set": list(candidate_set or []),
                "completed_candidate": completed_candidate,
                "extra_non_sensitive_action": list(extra_non_sensitive_action or []),
                "route_latency_ms": route_payload.get("route_latency_ms"),
                "runtime_mode": route_payload.get("runtime_mode"),
                "admission_decision": admission_decision,
                "admission_reason": route_payload.get("admission_reason"),
                "deadline_ms": deadline_ms,
                "deadline_metric": deadline_metric,
                "deadline_latency_ms": (
                    round(deadline_latency_ms, 3)
                    if deadline_latency_ms is not None
                    else None
                ),
                "deadline_miss": deadline_miss,
                "network_profile": route_payload.get("network_profile"),
                "cloud_delay_ms": route_payload.get("cloud_delay_ms"),
                "cloud_fail_mode": route_payload.get("cloud_fail_mode"),
                "runtime_monitor_enabled": route_payload.get("runtime_monitor_enabled"),
                "runtime_monitor_degraded": route_payload.get("runtime_monitor_degraded"),
                "runtime_degraded_until": route_payload.get("runtime_degraded_until"),
                "half_open_probe": route_payload.get("half_open_probe"),
                "slow_path_used": (
                    route_payload.get("predicted_route") == ROUTE_SLOW
                    and admission_decision != "reject"
                ),
                "reject_used": (
                    route_payload.get("predicted_route") == ROUTE_REJECT
                    or admission_decision == "reject"
                ),
            }

        self._append_jsonl("tasks.jsonl", task_payload)
        self._append_csv("task_metrics.csv", task_payload)

    def _deadline_latency_ms(
        self,
        *,
        deadline_metric: Any,
        e2e_latency_ms: float,
        ttfa_ms: float | None,
    ) -> float | None:
        if deadline_metric == "ttfa_ms":
            return ttfa_ms
        if deadline_metric == "e2e_latency_ms":
            return e2e_latency_ms
        return None

    def _deadline_miss(
        self,
        deadline_latency_ms: float | None,
        deadline_ms: Any,
    ) -> bool | None:
        if deadline_latency_ms is None or deadline_ms is None:
            return None
        try:
            return deadline_latency_ms > float(deadline_ms)
        except (TypeError, ValueError):
            return None

    def _base_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        task = self._task or {}
        return {
            "run_id": self.config.run_id,
            "task_id": task.get("task_id"),
            "task_text": task.get("task_text"),
            "experiment_group": self.config.experiment_group,
            "variant": self.config.variant,
            "repeat_index": self.config.repeat_index,
            "expected_route": self.config.expected_route,
            "expected_outcome": self.config.expected_outcome,
            "timestamp": now_iso(),
            **payload,
        }

    def _route_correct(self, predicted_route: Any) -> bool | None:
        expected = self.config.expected_route
        if expected is None or predicted_route is None:
            return None
        return expected == predicted_route

    def _append_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.run_dir / filename
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _append_csv(self, filename: str, payload: dict[str, Any]) -> None:
        path = self.run_dir / filename
        fieldnames = [
            "run_id",
            "task_id",
            "task_text",
            "experiment_group",
            "variant",
            "repeat_index",
            "expected_route",
            "expected_outcome",
            "predicted_route",
            "route_label",
            "classifier_label",
            "classifier_outcome",
            "parser_status",
            "guard_outcome",
            "final_route",
            "reason",
            "app_name",
            "raw_prediction",
            "route_correct",
            "route_success",
            "semantic_success",
            "semantic_reason",
            "user_action_required",
            "safety_violation",
            "candidate_set",
            "completed_candidate",
            "extra_non_sensitive_action",
            "route_latency_ms",
            "runtime_mode",
            "admission_decision",
            "admission_reason",
            "deadline_ms",
            "deadline_metric",
            "deadline_latency_ms",
            "deadline_miss",
            "network_profile",
            "cloud_delay_ms",
            "cloud_fail_mode",
            "runtime_monitor_enabled",
            "runtime_monitor_degraded",
            "runtime_degraded_until",
            "half_open_probe",
            "success",
            "expected_success",
            "e2e_latency_ms",
            "ttfa_ms",
            "first_action_name",
            "step_count",
            "slow_path_used",
            "reject_used",
            "started_at",
            "finished_at",
            "return_code",
        ]
        row = {key: payload.get(key) for key in fieldnames}
        with self._lock:
            file_exists = path.exists()
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)


_LOGGER = ExperimentLogger(ExperimentConfig(enabled=False))


def configure_experiment_logger(config: ExperimentConfig) -> ExperimentLogger:
    global _LOGGER
    _LOGGER = ExperimentLogger(config)
    return _LOGGER


def get_experiment_logger() -> ExperimentLogger:
    return _LOGGER


def parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    parsed = parse_optional_bool(value)
    return default if parsed is None else parsed
