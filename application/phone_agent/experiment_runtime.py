# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
"""Opt-in wall-clock exposure and append-only evidence; no admission policy.

A context is supplied by the external controller for exactly one request. Loading
it does not start an experiment or establish device/cloud authorization.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentContextError(ValueError):
    """Invalid or inconsistent acquisition context; never silently ignored."""


class InjectedAvailabilityError(RuntimeError):
    """Failure from the existing unavailable seam, without invoking a model."""


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentContextError(f"Expected JSON object: {path}")
    return payload


def _absolute(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ExperimentContextError(f"{field} must be an absolute local path")
    return Path(value).resolve()


def _number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentContextError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ExperimentContextError(f"Invalid {field}: {value!r}")
    return result


class ExperimentRuntime:
    """Wall-clock call binding plus evidence, using the existing runtime seams."""

    def __init__(self, context_file: str, args: Any) -> None:
        self.context_path = _absolute(context_file, "experiment_context_file")
        self.context = _object(self.context_path)
        c = self.context
        required = {
            "protocol_path", "episode_id", "request_id", "task_id", "batch",
            "replicate", "tau_s", "episode_start_monotonic_s", "events_path",
            "exposure_state_path", "source_manifest_sha256", "config_sha256",
            "simulation",
        }
        if set(c) != required:
            raise ExperimentContextError(
                f"Context fields missing={sorted(required - set(c))}, "
                f"unknown={sorted(set(c) - required)}"
            )
        if c["simulation"] is not False:
            raise ExperimentContextError("Runtime context requires simulation:false")
        for key in ("episode_id", "request_id", "task_id"):
            if not isinstance(c[key], str) or not c[key].strip():
                raise ExperimentContextError(f"{key} must be a nonempty string")
        for key in ("source_manifest_sha256", "config_sha256"):
            if not isinstance(c[key], str) or re.fullmatch(r"[0-9a-fA-F]{64}", c[key]) is None:
                raise ExperimentContextError(f"{key} must be a SHA-256 hex digest")
        for key in ("batch", "replicate"):
            if type(c[key]) is not int or c[key] <= 0:
                raise ExperimentContextError(f"{key} must be a positive integer")
        _number(c["tau_s"], "tau_s", minimum=1.0)
        self.episode_start = _number(c["episode_start_monotonic_s"], "episode_start_monotonic_s")
        if self.episode_start > time.monotonic():
            raise ExperimentContextError("episode_start_monotonic_s is in the future")
        self.protocol_path = _absolute(c["protocol_path"], "protocol_path")
        self.events_path = _absolute(c["events_path"], "events_path")
        self.exposure_path = _absolute(c["exposure_state_path"], "exposure_state_path")
        paths = {self.context_path, self.protocol_path, self.events_path, self.exposure_path}
        if len(paths) != 4:
            raise ExperimentContextError("Context/protocol/event/exposure paths must differ")
        self.protocol = _object(self.protocol_path)
        self.protocol_sha256 = hashlib.sha256(self.protocol_path.read_bytes()).hexdigest()
        self.protocol_id = self.protocol["protocol_id"]
        if not isinstance(self.protocol_id, str) or not self.protocol_id:
            raise ExperimentContextError("Invalid protocol_id")
        batches = self.protocol["batch_order"]
        if not any(
            b.get("batch") == c["batch"] and b.get("replicate") == c["replicate"]
            and b.get("tau_s") == c["tau_s"] for b in batches
        ):
            raise ExperimentContextError("Context batch/replicate/tau does not match protocol")
        if c["task_id"] not in self.protocol["tasks"]:
            raise ExperimentContextError("Unknown protocol task_id")
        self.phases = self.protocol["episode"]["phases"]
        previous_end = 0.0
        names: set[str] = set()
        for phase in self.phases:
            if not isinstance(phase, dict) or not isinstance(phase.get("name"), str):
                raise ExperimentContextError("Invalid phase")
            if phase["name"] in names:
                raise ExperimentContextError("Duplicate phase name")
            names.add(phase["name"])
            start = _number(phase["start_s"], "phase.start_s")
            end = _number(phase["end_s"], "phase.end_s")
            if start != previous_end or end <= start:
                raise ExperimentContextError("Phases must be contiguous and start at zero")
            previous_end = end
            _number(phase["extra_step_delay_ms"], "phase.extra_step_delay_ms")
            fault = phase["availability_fault"]
            if fault != "none" and not fault.startswith("unavailable at "):
                raise ExperimentContextError("Unsupported availability fault")
            shot = phase.get("one_shot")
            if shot is not None:
                left = _number(shot["eligible_start_s"], "one_shot.eligible_start_s")
                right = _number(shot["eligible_end_s"], "one_shot.eligible_end_s")
                if not (start <= left < right <= end) or shot["max_cloud_step_calls"] != 1:
                    raise ExperimentContextError("Invalid one-shot eligibility")
                _number(shot["extra_step_delay_ms"], "one_shot.extra_step_delay_ms")
        if not self.phases or previous_end != self.protocol["episode"]["arrival_horizon_s"]:
            raise ExperimentContextError("Phase horizon mismatch")
        last = self.phases[-1]
        if last["extra_step_delay_ms"] != 0 or last["availability_fault"] != "none":
            raise ExperimentContextError("Drain must retain the recovered condition")
        self._validate_args(args)
        self.native_start: float | None = None
        self.terminal_recorded = False
        self.route_recorded = False
        self.active_step: dict[str, Any] | None = None
        self._call_count = 0

    def _validate_args(self, args: Any) -> None:
        c = self.context
        if not getattr(args, "task", None) or args.task != self.protocol["tasks"][c["task_id"]]["text"]:
            raise ExperimentContextError("Context requires exactly the registered task text")
        if args.device_type != "adb" or args.disable_fast_path or args.routing_mode != "hybrid":
            raise ExperimentContextError("Context requires the protocol Android hybrid path")
        for flag in ("start_fast_path_router_service", "list_apps", "list_fast_path_apps",
                     "list_devices", "connect", "disconnect", "enable_tcpip", "pair", "wda_status"):
            if getattr(args, flag, False):
                raise ExperimentContextError(f"Context cannot be used with {flag}")
        if args.cloud_delay_ms != 0 or args.cloud_fail_mode != "none" or args.network_profile != "normal":
            raise ExperimentContextError("Context cannot be mixed with legacy delay/failure settings")
        if args.cloud_delay_profile or args.cloud_delay_cursor_file:
            raise ExperimentContextError("Context cannot use a legacy delay profile/cursor")
        frozen = self.protocol["frozen_runtime"]
        for key in (
            "enable_deadline_policy", "enable_runtime_monitor", "runtime_degraded_mode",
            "degraded_slow_path_admission", "runtime_degraded_detector", "runtime_window_size",
            "runtime_window_min_samples", "runtime_window_percentile", "runtime_window_enter_ms",
            "runtime_window_exit_ms", "cloud_probe_deadline_ms", "runtime_degraded_cooldown_seconds",
            "fast_path_deadline_ms", "slow_path_deadline_ms", "reject_deadline_ms",
        ):
            if getattr(args, key) != frozen[key]:
                raise ExperimentContextError(f"CLI {key} differs from frozen_runtime")
        if args.runtime_half_open_probe_interval_seconds != c["tau_s"]:
            raise ExperimentContextError("CLI half-open interval differs from context tau_s")
        health_path = Path(args.runtime_health_file).expanduser().resolve()
        if health_path in {self.context_path, self.protocol_path, self.events_path, self.exposure_path}:
            raise ExperimentContextError("Health file must not overwrite context/protocol/events/exposure")
        if args.experiment_task_id and args.experiment_task_id != c["request_id"]:
            raise ExperimentContextError("experiment_task_id must equal context request_id")
        args.experiment_task_id = c["request_id"]

    def emit(self, event_type: str, **extras: Any) -> None:
        now = time.monotonic()
        c = self.context
        common = {
            "schema": "unified-admission-events-v1", "event_type": event_type,
            "event_id": uuid.uuid4().hex,
            **{k: c[k] for k in ("episode_id", "request_id", "task_id", "batch", "replicate", "tau_s",
                                  "source_manifest_sha256", "config_sha256")},
            "protocol_id": self.protocol_id, "protocol_sha256": self.protocol_sha256,
            "utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": now,
            "elapsed_s": now - self.episode_start, "simulation": False,
        }
        overlap = set(common) & set(extras)
        if overlap:
            raise ExperimentContextError(f"Event extras overwrite common fields: {sorted(overlap)}")
        data = (json.dumps({**common, **extras}, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        # Lock across complete line write(s), including a rare short os.write.
        with self.events_path.open("ab", buffering=0) as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            view = memoryview(data)
            while view:
                written = os.write(handle.fileno(), view)
                if written <= 0:
                    raise OSError("Unable to append runtime event")
                view = view[written:]
            os.fsync(handle.fileno())

    def _claim_one_shot(self, phase: dict[str, Any], call_id: str, elapsed: float) -> bool:
        self.exposure_path.parent.mkdir(parents=True, exist_ok=True)
        with self.exposure_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            raw = handle.read()
            state = json.loads(raw) if raw else {
                "schema": "unified-admission-exposure-v1", "episode_id": self.context["episode_id"],
                "protocol_sha256": self.protocol_sha256, "claims": {},
            }
            if (not isinstance(state, dict) or state.get("schema") != "unified-admission-exposure-v1"
                    or state.get("episode_id") != self.context["episode_id"]
                    or state.get("protocol_sha256") != self.protocol_sha256
                    or not isinstance(state.get("claims"), dict)):
                raise ExperimentContextError("Exposure state belongs to another episode/protocol or is corrupt")
            claims = state["claims"]
            if phase["name"] in claims:
                return False
            claims[phase["name"]] = {"call_id": call_id, "request_id": self.context["request_id"],
                                     "call_start_elapsed_s": elapsed}
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(state, ensure_ascii=False, allow_nan=False))
            handle.flush()
            os.fsync(handle.fileno())
            return True

    def begin_call(self, call_kind: str, *, step_index: int | None = None) -> dict[str, Any]:
        if call_kind not in {"cloud_step", "availability_probe"}:
            raise ExperimentContextError(f"Unknown call_kind: {call_kind}")
        now = time.monotonic()
        elapsed = now - self.episode_start
        if elapsed < 0:
            raise ExperimentContextError("Monotonic clock precedes episode start")
        phase = next((p for p in self.phases if p["start_s"] <= elapsed < p["end_s"]), self.phases[-1])
        self._call_count += 1
        call_id = f"{self.context['request_id']}:c{self._call_count:04d}:{uuid.uuid4().hex}"
        # The protocol adds delay to cloud steps only; a probe can receive the
        # unavailable fault. This preserves extra_step_delay_ms literally.
        delay = float(phase["extra_step_delay_ms"]) if call_kind == "cloud_step" else 0.0
        shot = phase.get("one_shot")
        claimed = False
        if (call_kind == "cloud_step" and shot is not None
                and shot["eligible_start_s"] <= elapsed < shot["eligible_end_s"]):
            claimed = self._claim_one_shot(phase, call_id, elapsed)
            if claimed:
                delay += float(shot["extra_step_delay_ms"])
        call = {
            "call_id": call_id, "call_kind": call_kind, "step_index": step_index,
            "call_start_monotonic_s": now, "call_start_elapsed_s": elapsed,
            "call_end_elapsed_s": None, "phase_applied": phase["name"],
            "injected_delay_ms": delay, "actual_delay_wait_ms": 0.0,
            "fault_applied": "none" if phase["availability_fault"] == "none" else "unavailable",
            "one_shot_claimed": claimed, "model_invoked": False,
        }
        if call_kind == "cloud_step":
            if self.active_step is not None:
                raise ExperimentContextError("Overlapping cloud steps in single worker")
            self.active_step = call
        self.emit("cloud_call_start", **call, success=None, error=None, measured_ms=None,
                  injection_status="bound_at_call_start")
        return call

    def active_step_delay_ms(self) -> float:
        if self.active_step is None:
            raise ExperimentContextError("Delay requested outside a bound cloud step")
        return self.active_step["injected_delay_ms"]

    def end_call(self, call: dict[str, Any], *, success: bool, error: str | None = None) -> float:
        now = time.monotonic()
        measured = (now - call["call_start_monotonic_s"]) * 1000.0
        self.emit("cloud_call_end", **{**call, "call_end_elapsed_s": now - self.episode_start},
                  success=success, error=error, measured_ms=measured)
        if self.active_step is call:
            self.active_step = None
        return measured

    def health_event(self, event_type: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        next_unix = payload.pop("next_halfopen_eligible_at_unix", 0.0)
        state_now = payload.pop("state_timestamp_unix")
        eligible_unix = payload.pop("eligible_at_unix", 0.0)
        if event_type == "halfopen_claim":
            payload["eligible_at_elapsed_s"] = (
                time.monotonic() - self.episode_start + eligible_unix - state_now if eligible_unix else None
            )
        payload["next_halfopen_eligible_at"] = (
            time.monotonic() - self.episode_start + next_unix - state_now if next_unix else None
        )
        self.emit(event_type, **payload)

    def record_route(self, payload: dict[str, Any]) -> None:
        self.route_recorded = True
        admitted = payload.get("admission_decision") == "admit"
        self.emit(
            "route_decision", raw=payload.get("raw_prediction"),
            effective_route=payload.get("final_route", payload.get("predicted_route")),
            guard_reason=payload.get("reason"), guard_outcome=payload.get("guard_outcome"),
            classifier_label=payload.get("classifier_label"), parser_status=payload.get("parser_status"),
            admission_decision=payload.get("admission_decision"),
            admission_reason=payload.get("admission_reason"),
            half_open=bool(payload.get("half_open_probe", False)),
            admitted_elapsed_s=time.monotonic() - self.episode_start if admitted else None,
        )

    def start_task(self) -> None:
        if self.native_start is not None:
            raise ExperimentContextError("A context may execute only one request")
        self.native_start = time.monotonic()
        self.emit("task_start", native_start_elapsed_s=self.native_start - self.episode_start)

    def terminal(self, *, success: bool, status: str, error: str | None = None,
                 step_count: int | None = None, user_action_required: bool = False,
                 last_step_success: bool | None = None) -> None:
        if self.terminal_recorded:
            return
        if not self.route_recorded:
            self.emit("route_decision", raw=None, effective_route=None, guard_reason="not_reached",
                      admission_decision=None, half_open=False, admitted_elapsed_s=None, error=error)
        self.emit("task_terminal", machine_success=success, status=status, error=error,
                  native_e2e_ms=(time.monotonic() - self.native_start) * 1000 if self.native_start is not None else None,
                  step_count=step_count, user_action_required=user_action_required,
                  last_step_success=last_step_success)
        self.terminal_recorded = True
