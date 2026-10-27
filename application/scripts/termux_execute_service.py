# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import formatdate
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MAIN_PATH = REPO_ROOT / "main.py"
WEB_ROOT = REPO_ROOT / "web_console"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _default_runtime_health_file() -> str:
    return str(REPO_ROOT / "experiment_logs" / "runtime_health.json")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _guess_content_type(path: Path) -> str:
    if path.suffix == ".html":
        return "text/html; charset=utf-8"
    if path.suffix == ".css":
        return "text/css; charset=utf-8"
    if path.suffix == ".js":
        return "application/javascript; charset=utf-8"
    if path.suffix == ".json":
        return "application/json; charset=utf-8"
    return "text/plain; charset=utf-8"


def extract_result_text(log_text: str) -> str | None:
    result_line_pattern = re.compile(r"^\s*Result[：:]\s*(.*)$", re.IGNORECASE)
    lines = log_text.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        match = result_line_pattern.match(lines[index])
        if not match:
            continue

        collected = [match.group(1)]
        for follow_line in lines[index + 1 :]:
            if follow_line.startswith("[service]") or follow_line.startswith("[stderr]"):
                break
            collected.append(follow_line)

        result_text = "\n".join(collected).strip()
        return result_text or None

    non_empty_lines = [line.strip() for line in lines if line.strip()]
    if not non_empty_lines:
        return None

    for line in reversed(non_empty_lines):
        if "[stderr]" in line:
            return line
        if line.lower().startswith("error:"):
            return line
    return non_empty_lines[-1]


@dataclass
class TaskState:
    task_id: int = 0
    experiment_task_id: str | None = None
    expected_outcome: str | None = None
    candidate_set: list[str] = field(default_factory=list)
    completed_candidate: str | None = None
    extra_non_sensitive_action: list[str] = field(default_factory=list)
    status: str = "idle"
    current_text: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    result: str | None = None
    pid: int | None = None
    command: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


class ExecuteTaskController:
    def __init__(
        self,
        *,
        python_executable: str | None = None,
        main_script_path: Path | str = MAIN_PATH,
        working_directory: Path | str = REPO_ROOT,
        main_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        command_builder: Callable[[str], list[str]] | None = None,
        process_factory: Callable[..., subprocess.Popen[str]] | None = None,
    ) -> None:
        self.python_executable = python_executable or sys.executable
        self.main_script_path = Path(main_script_path)
        self.working_directory = Path(working_directory)
        self.main_args = list(main_args or [])
        self._base_env = os.environ.copy()
        self._base_env.setdefault("PYTHONIOENCODING", "utf-8")
        self._base_env.setdefault("PYTHONUTF8", "1")
        if env:
            self._base_env.update(env)
        self._command_builder = command_builder or self._build_command
        self._process_factory = process_factory or subprocess.Popen
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._worker_thread: threading.Thread | None = None
        self._state = TaskState()
        self._next_task_id = 1

    def _build_command(self, text: str) -> list[str]:
        return [
            self.python_executable,
            "-u",
            str(self.main_script_path),
            *self.main_args,
            text,
        ]

    def _build_task_command(
        self,
        text: str,
        experiment_task_id: str | None,
        expected_outcome: str | None,
        candidate_set: list[str] | None = None,
        completed_candidate: str | None = None,
        extra_non_sensitive_action: list[str] | None = None,
    ) -> list[str]:
        command = self._command_builder(text)
        if not (experiment_task_id or expected_outcome):
            return command
        # Keep the task text last so existing runners and test stubs remain compatible.
        task_text = command[-1:]
        command = command[:-1]
        if experiment_task_id:
            command.extend(["--experiment-task-id", experiment_task_id])
        if expected_outcome:
            command.extend(["--experiment-expected-outcome", expected_outcome])
        if candidate_set:
            command.extend(["--experiment-candidate-set-json", json.dumps(candidate_set, ensure_ascii=False)])
        if completed_candidate:
            command.extend(["--experiment-completed-candidate", completed_candidate])
        if extra_non_sensitive_action:
            command.extend(
                [
                    "--experiment-extra-non-sensitive-action-json",
                    json.dumps(extra_non_sensitive_action, ensure_ascii=False),
                ]
            )
        command.extend(task_text)
        return command

    def is_busy(self) -> bool:
        with self._lock:
            return self._state.status in {"starting", "running"}

    def get_status(self) -> dict[str, object]:
        with self._lock:
            return self._serialize_state(self._state)

    def get_logs(self) -> str:
        with self._lock:
            return "".join(self._state.logs)

    def start_task(
        self,
        text: str,
        *,
        experiment_task_id: str | None = None,
        expected_outcome: str | None = None,
        candidate_set: list[str] | None = None,
        completed_candidate: str | None = None,
        extra_non_sensitive_action: list[str] | None = None,
    ) -> tuple[bool, dict[str, object]]:
        stripped = text.strip()
        if not stripped:
            raise ValueError("Task text cannot be empty.")

        with self._lock:
            if self._state.status in {"starting", "running"}:
                return False, self._serialize_state(self._state)

            task_id = self._next_task_id
            self._next_task_id += 1
            self._state = TaskState(
                task_id=task_id,
                experiment_task_id=(experiment_task_id or f"task_{task_id}"),
                expected_outcome=expected_outcome,
                candidate_set=list(candidate_set or []),
                completed_candidate=completed_candidate,
                extra_non_sensitive_action=list(extra_non_sensitive_action or []),
                status="starting",
                current_text=stripped,
                started_at=_timestamp(),
            )
            thread = threading.Thread(
                target=self._run_task,
                args=(
                    task_id,
                    stripped,
                    self._state.experiment_task_id,
                    self._state.expected_outcome,
                    self._state.candidate_set,
                    self._state.completed_candidate,
                    self._state.extra_non_sensitive_action,
                ),
                daemon=True,
                name=f"execute-task-{task_id}",
            )
            self._worker_thread = thread
            snapshot = self._serialize_state(self._state)

        thread.start()
        return True, snapshot

    def _serialize_state(self, state: TaskState) -> dict[str, object]:
        return {
            "task_id": state.task_id,
            "experiment_task_id": state.experiment_task_id,
            "expected_outcome": state.expected_outcome,
            "candidate_set": list(state.candidate_set),
            "completed_candidate": state.completed_candidate,
            "extra_non_sensitive_action": list(state.extra_non_sensitive_action),
            "status": state.status,
            "current_text": state.current_text,
            "started_at": state.started_at,
            "finished_at": state.finished_at,
            "return_code": state.return_code,
            "result": state.result,
            "pid": state.pid,
            "is_busy": state.status in {"starting", "running"},
            "log_length": sum(len(chunk) for chunk in state.logs),
        }

    def _append_log(self, task_id: int, text: str) -> None:
        with self._lock:
            if self._state.task_id != task_id:
                return
            self._state.logs.append(text)

    def _set_running(self, task_id: int, process: subprocess.Popen[str], command: list[str]) -> None:
        with self._lock:
            if self._state.task_id != task_id:
                return
            self._process = process
            self._state.status = "running"
            self._state.pid = process.pid
            self._state.command = list(command)

    def _finish_task(self, task_id: int, return_code: int, result: str | None) -> None:
        with self._lock:
            if self._state.task_id != task_id:
                return
            self._process = None
            self._state.return_code = return_code
            self._state.finished_at = _timestamp()
            self._state.result = result
            self._state.status = "succeeded" if return_code == 0 else "failed"

    def _fail_to_start(self, task_id: int, message: str) -> None:
        self._append_log(task_id, f"[service] {message}\n")
        self._finish_task(task_id, 1, message)

    def _consume_stream(self, task_id: int, stream_name: str, stream) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                if stream_name == "stderr":
                    self._append_log(task_id, f"[stderr] {line}")
                else:
                    self._append_log(task_id, line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _run_task(
        self,
        task_id: int,
        text: str,
        experiment_task_id: str | None,
        expected_outcome: str | None,
        candidate_set: list[str],
        completed_candidate: str | None,
        extra_non_sensitive_action: list[str],
    ) -> None:
        command = self._build_task_command(
            text,
            experiment_task_id,
            expected_outcome,
            candidate_set,
            completed_candidate,
            extra_non_sensitive_action,
        )
        env = self._base_env.copy()
        process: subprocess.Popen[str] | None = None

        try:
            process = self._process_factory(
                command,
                cwd=str(self.working_directory),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
            )
        except Exception as exc:
            self._fail_to_start(task_id, f"Failed to start task process: {exc}")
            return

        if process.stdout is None or process.stderr is None:
            self._fail_to_start(task_id, "Task process pipes are unavailable.")
            return

        self._set_running(task_id, process, command)
        self._append_log(task_id, f"[service] Task process started at {_timestamp()}.\n")

        stdout_thread = threading.Thread(
            target=self._consume_stream,
            args=(task_id, "stdout", process.stdout),
            daemon=True,
            name=f"execute-task-{task_id}-stdout",
        )
        stderr_thread = threading.Thread(
            target=self._consume_stream,
            args=(task_id, "stderr", process.stderr),
            daemon=True,
            name=f"execute-task-{task_id}-stderr",
        )
        stdout_thread.start()
        stderr_thread.start()

        return_code = process.wait()
        stdout_thread.join(timeout=5.0)
        stderr_thread.join(timeout=5.0)

        log_text = self.get_logs()
        result = extract_result_text(log_text)
        if result is None:
            if return_code == 0:
                result = "Task finished successfully. Check logs for details."
            else:
                result = "Task failed. Check logs for details."

        self._append_log(
            task_id,
            f"[service] Task process finished with return code {return_code} at {_timestamp()}.\n",
        )
        self._finish_task(task_id, return_code, result)


class ExecuteServiceHandler(BaseHTTPRequestHandler):
    server_version = "AutoGLMExecuteService/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            self._serve_static_file("index.html")
            return
        if path in {"/app.js", "/styles.css"}:
            self._serve_static_file(path.lstrip("/"))
            return
        if path == "/health":
            controller = self.server.controller  # type: ignore[attr-defined]
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "task_status": controller.get_status()["status"],
                    "timestamp": _timestamp(),
                },
            )
            return
        if path == "/status":
            controller = self.server.controller  # type: ignore[attr-defined]
            self._write_json(HTTPStatus.OK, controller.get_status())
            return
        if path == "/logs":
            controller = self.server.controller  # type: ignore[attr-defined]
            self._write_json(
                HTTPStatus.OK,
                {
                    "logs": controller.get_logs(),
                    "status": controller.get_status()["status"],
                },
            )
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path != "/execute":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
            return

        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_text"})
            return

        experiment_task_id = payload.get("experiment_task_id", payload.get("task_id"))
        if experiment_task_id is not None and (
            not isinstance(experiment_task_id, str) or not experiment_task_id.strip()
        ):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_experiment_task_id"})
            return
        expected_outcome = payload.get("expected_outcome")
        if expected_outcome is not None and expected_outcome not in {"execute", "refuse"}:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_expected_outcome"})
            return
        metadata_lists: dict[str, list[str]] = {}
        for field_name in ("candidate_set", "extra_non_sensitive_action"):
            value = payload.get(field_name, [])
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                self._write_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid_{field_name}"})
                return
            metadata_lists[field_name] = [item.strip() for item in value if item.strip()]
        completed_candidate = payload.get("completed_candidate")
        if completed_candidate is not None and (
            not isinstance(completed_candidate, str) or not completed_candidate.strip()
        ):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_completed_candidate"})
            return

        controller = self.server.controller  # type: ignore[attr-defined]
        try:
            accepted, snapshot = controller.start_task(
                text,
                experiment_task_id=experiment_task_id.strip() if experiment_task_id else None,
                expected_outcome=expected_outcome,
                candidate_set=metadata_lists["candidate_set"],
                completed_candidate=(completed_candidate.strip() if completed_candidate else None),
                extra_non_sensitive_action=metadata_lists["extra_non_sensitive_action"],
            )
        except ValueError:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_text"})
            return

        if not accepted:
            self._write_json(
                HTTPStatus.CONFLICT,
                {
                    "status": "busy",
                    "message": "A task is already running.",
                    "current_task": snapshot,
                },
            )
            return

        self._write_json(
            HTTPStatus.ACCEPTED,
            {
                "status": "accepted",
                "message": "Task accepted.",
                "task": snapshot,
            },
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _serve_static_file(self, relative_path: str) -> None:
        web_root = Path(self.server.web_root)  # type: ignore[attr-defined]
        file_path = (web_root / relative_path).resolve()
        if not file_path.exists() or web_root.resolve() not in file_path.parents and file_path != web_root.resolve():
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _guess_content_type(file_path))
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Last-Modified", formatdate(file_path.stat().st_mtime, usegmt=True))
        self.end_headers()
        self.wfile.write(content)

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def create_execute_service(
    *,
    controller: ExecuteTaskController,
    host: str = "127.0.0.1",
    port: int = 8001,
    web_root: Path | str = WEB_ROOT,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), ExecuteServiceHandler)
    server.controller = controller  # type: ignore[attr-defined]
    server.web_root = str(Path(web_root))  # type: ignore[attr-defined]
    return server


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a local AutoGLM browser console and /execute service."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Service host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8001, help="Service port. Default: 8001")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to spawn main.py. Default: current Python",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1"),
        help="Model API base URL passed to main.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b"),
        help="Model name passed to main.py",
    )
    parser.add_argument(
        "--apikey",
        type=str,
        default=os.getenv("PHONE_AGENT_API_KEY", "EMPTY"),
        help="API key passed to main.py",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("PHONE_AGENT_MAX_STEPS", "100")),
        help="Maximum steps passed to main.py",
    )
    parser.add_argument(
        "--fast-path-router-service-url",
        type=str,
        default=os.getenv("PHONE_AGENT_FAST_PATH_ROUTER_SERVICE_URL"),
        help="Existing Fast Path /route service URL reused by main.py",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default=os.getenv("PHONE_AGENT_DEVICE_ID"),
        help="ADB device ID passed to main.py",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=("cn", "en"),
        default=os.getenv("PHONE_AGENT_LANG", "cn"),
        help="Language passed to main.py",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        choices=("adb", "hdc", "ios"),
        default=os.getenv("PHONE_AGENT_DEVICE_TYPE", "adb"),
        help="Device type passed to main.py",
    )
    parser.add_argument(
        "--wda-url",
        type=str,
        default=os.getenv("PHONE_AGENT_WDA_URL", "http://localhost:8100"),
        help="WebDriverAgent URL passed to main.py for iOS",
    )
    parser.add_argument(
        "--routing-mode",
        choices=("hybrid", "cloud_only", "device_only"),
        default=os.getenv("PHONE_AGENT_ROUTING_MODE", "hybrid"),
        help=(
            "Routing ablation mode passed to main.py: hybrid (DualRoute), "
            "cloud_only (always Slow Path), device_only (Fast Path only, "
            "complex tasks unsupported with no cloud fallback)"
        ),
    )
    parser.add_argument(
        "--disable-fast-path",
        action="store_true",
        help="Deprecated alias for --routing-mode cloud_only",
    )
    parser.add_argument(
        "--disable-screenshot-prewarm",
        action="store_true",
        help="Disable Android slow-path screenshot pre-warm in spawned main.py tasks",
    )
    parser.add_argument(
        "--enable-experiment-logging",
        action="store_true",
        help="Enable structured experiment logging in spawned main.py tasks",
    )
    parser.add_argument(
        "--experiment-log-dir",
        default=os.getenv(
            "PHONE_AGENT_EXPERIMENT_LOG_DIR",
            str(REPO_ROOT / "experiment_logs"),
        ),
        help="Directory used for structured experiment logs",
    )
    parser.add_argument(
        "--experiment-run-id",
        default=os.getenv("PHONE_AGENT_EXPERIMENT_RUN_ID"),
        help="Run id shared by all tasks submitted to this execute service",
    )
    parser.add_argument(
        "--experiment-group",
        default=os.getenv("PHONE_AGENT_EXPERIMENT_GROUP"),
        help="Experiment group passed to main.py",
    )
    parser.add_argument(
        "--experiment-variant",
        default=os.getenv("PHONE_AGENT_EXPERIMENT_VARIANT"),
        help="Experiment variant passed to main.py",
    )
    parser.add_argument(
        "--experiment-expected-route",
        choices=("fast_path", "slow_path", "reject"),
        default=os.getenv("PHONE_AGENT_EXPERIMENT_EXPECTED_ROUTE"),
        help="Expected route passed to main.py for route-accuracy logging",
    )
    parser.add_argument(
        "--experiment-expected-success",
        default=os.getenv("PHONE_AGENT_EXPERIMENT_EXPECTED_SUCCESS"),
        help="Expected success passed to main.py: true or false",
    )
    parser.add_argument(
        "--experiment-expected-outcome",
        choices=("execute", "refuse"),
        default=os.getenv("PHONE_AGENT_EXPERIMENT_EXPECTED_OUTCOME"),
        help="Default semantic outcome passed to main.py; /execute payload may override it",
    )
    parser.add_argument(
        "--experiment-repeat-index",
        type=int,
        default=(
            int(os.getenv("PHONE_AGENT_EXPERIMENT_REPEAT_INDEX"))
            if os.getenv("PHONE_AGENT_EXPERIMENT_REPEAT_INDEX")
            else None
        ),
        help="Repeat index passed to main.py",
    )
    parser.add_argument(
        "--enable-deadline-policy",
        action="store_true",
        default=os.getenv("PHONE_AGENT_ENABLE_DEADLINE_POLICY", "").lower()
        in {"1", "true", "yes", "y"},
        help="Enable deadline-aware runtime admission control in spawned main.py tasks",
    )
    parser.add_argument(
        "--fast-path-deadline-ms",
        type=float,
        default=_env_float("PHONE_AGENT_FAST_PATH_DEADLINE_MS", 800.0),
        help="Deadline budget for Fast Path interactions passed to main.py",
    )
    parser.add_argument(
        "--slow-path-deadline-ms",
        type=float,
        default=_env_float("PHONE_AGENT_SLOW_PATH_DEADLINE_MS", 5000.0),
        help="Deadline budget for Slow Path tasks passed to main.py",
    )
    parser.add_argument(
        "--reject-deadline-ms",
        type=float,
        default=_env_float("PHONE_AGENT_REJECT_DEADLINE_MS", 500.0),
        help="Deadline budget for local rejection passed to main.py",
    )
    parser.add_argument(
        "--runtime-degraded-mode",
        choices=("off", "on", "auto"),
        default=os.getenv("PHONE_AGENT_RUNTIME_DEGRADED_MODE", "off"),
        help="Runtime degraded mode passed to main.py",
    )
    parser.add_argument(
        "--degraded-slow-path-admission",
        choices=("admit", "reject"),
        default=os.getenv("PHONE_AGENT_DEGRADED_SLOW_PATH_ADMISSION", "reject"),
        help="Slow Path admission behavior in degraded mode passed to main.py",
    )
    parser.add_argument(
        "--network-profile",
        default=os.getenv("PHONE_AGENT_NETWORK_PROFILE", "normal"),
        help="Network profile label passed to main.py",
    )
    parser.add_argument(
        "--cloud-delay-ms",
        type=float,
        default=_env_float("PHONE_AGENT_CLOUD_DELAY_MS", 0.0),
        help="Injected Slow Path cloud delay passed to main.py",
    )
    parser.add_argument(
        "--cloud-fail-mode",
        choices=("none", "timeout", "unavailable"),
        default=os.getenv("PHONE_AGENT_CLOUD_FAIL_MODE", "none"),
        help="Injected Slow Path cloud failure mode passed to main.py",
    )
    parser.add_argument(
        "--enable-runtime-monitor",
        action="store_true",
        default=os.getenv("PHONE_AGENT_ENABLE_RUNTIME_MONITOR", "").lower()
        in {"1", "true", "yes", "y"},
        help="Enable runtime health monitoring in spawned main.py tasks",
    )
    parser.add_argument(
        "--runtime-health-file",
        default=os.getenv("PHONE_AGENT_RUNTIME_HEALTH_FILE", _default_runtime_health_file()),
        help="Runtime health JSON file passed to main.py",
    )
    parser.add_argument(
        "--cloud-probe-deadline-ms",
        type=float,
        default=_env_float("PHONE_AGENT_CLOUD_PROBE_DEADLINE_MS", 8000.0),
        help="Cloud probe deadline passed to main.py",
    )
    parser.add_argument(
        "--runtime-degraded-cooldown-seconds",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_DEGRADED_COOLDOWN_SECONDS", 300.0),
        help="Degraded-mode cooldown passed to main.py",
    )
    parser.add_argument(
        "--runtime-degraded-detector",
        choices=("single_miss", "sliding_window"),
        default=os.getenv("PHONE_AGENT_RUNTIME_DEGRADED_DETECTOR", "single_miss"),
        help="Degraded-mode detector passed to main.py",
    )
    parser.add_argument(
        "--runtime-window-size",
        type=int,
        default=_env_int("PHONE_AGENT_RUNTIME_WINDOW_SIZE", 10),
        help="Sliding-window size passed to main.py",
    )
    parser.add_argument(
        "--runtime-window-min-samples",
        type=int,
        default=_env_int("PHONE_AGENT_RUNTIME_WINDOW_MIN_SAMPLES", 5),
        help="Sliding-window minimum samples passed to main.py",
    )
    parser.add_argument(
        "--runtime-window-percentile",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_WINDOW_PERCENTILE", 90.0),
        help="Sliding-window percentile passed to main.py",
    )
    parser.add_argument(
        "--runtime-window-enter-ms",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_WINDOW_ENTER_MS", 5000.0),
        help="Sliding-window enter threshold passed to main.py",
    )
    parser.add_argument(
        "--runtime-window-exit-ms",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_WINDOW_EXIT_MS", 3000.0),
        help="Sliding-window exit threshold passed to main.py",
    )
    parser.add_argument(
        "--runtime-half-open-probe-interval-seconds",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_HALF_OPEN_PROBE_INTERVAL_SECONDS", 15.0),
        help="Half-open Slow Path probe interval passed to main.py",
    )
    parser.add_argument(
        "--cloud-delay-profile",
        default=os.getenv("PHONE_AGENT_CLOUD_DELAY_PROFILE", ""),
        help="Time-varying cloud-delay profile path passed to main.py",
    )
    parser.add_argument(
        "--cloud-delay-cursor-file",
        default=os.getenv("PHONE_AGENT_CLOUD_DELAY_CURSOR_FILE", ""),
        help="Cloud-delay profile cursor file passed to main.py",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Pass --quiet to main.py to suppress extra logs",
    )
    parser.add_argument(
        "--main-script-path",
        default=str(MAIN_PATH),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--web-root",
        default=str(WEB_ROOT),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def build_main_args(args: argparse.Namespace) -> list[str]:
    main_args = [
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--apikey",
        args.apikey,
        "--max-steps",
        str(args.max_steps),
        "--lang",
        args.lang,
        "--device-type",
        args.device_type,
        "--wda-url",
        args.wda_url,
    ]

    if args.fast_path_router_service_url:
        main_args.extend(
            ["--fast-path-router-service-url", args.fast_path_router_service_url]
        )
    if args.device_id:
        main_args.extend(["--device-id", args.device_id])
    main_args.extend(["--routing-mode", args.routing_mode])
    if args.disable_fast_path:
        main_args.append("--disable-fast-path")
    if args.disable_screenshot_prewarm:
        main_args.append("--disable-screenshot-prewarm")
    if args.enable_experiment_logging:
        main_args.extend(
            [
                "--enable-experiment-logging",
                "--experiment-log-dir",
                args.experiment_log_dir,
                "--experiment-run-id",
                args.experiment_run_id,
            ]
        )
        if args.experiment_group:
            main_args.extend(["--experiment-group", args.experiment_group])
        if args.experiment_variant:
            main_args.extend(["--experiment-variant", args.experiment_variant])
        if args.experiment_expected_route:
            main_args.extend(
                ["--experiment-expected-route", args.experiment_expected_route]
            )
        if args.experiment_expected_success:
            main_args.extend(
                ["--experiment-expected-success", args.experiment_expected_success]
            )
        if args.experiment_expected_outcome:
            main_args.extend(
                ["--experiment-expected-outcome", args.experiment_expected_outcome]
            )
        if args.experiment_repeat_index is not None:
            main_args.extend(
                ["--experiment-repeat-index", str(args.experiment_repeat_index)]
            )
    if args.enable_deadline_policy:
        main_args.append("--enable-deadline-policy")
    if args.enable_runtime_monitor:
        main_args.append("--enable-runtime-monitor")
    main_args.extend(
        [
            "--fast-path-deadline-ms",
            str(args.fast_path_deadline_ms),
            "--slow-path-deadline-ms",
            str(args.slow_path_deadline_ms),
            "--reject-deadline-ms",
            str(args.reject_deadline_ms),
            "--runtime-degraded-mode",
            args.runtime_degraded_mode,
            "--degraded-slow-path-admission",
            args.degraded_slow_path_admission,
            "--network-profile",
            args.network_profile,
            "--cloud-delay-ms",
            str(args.cloud_delay_ms),
            "--cloud-fail-mode",
            args.cloud_fail_mode,
            "--runtime-health-file",
            args.runtime_health_file,
            "--cloud-probe-deadline-ms",
            str(args.cloud_probe_deadline_ms),
            "--runtime-degraded-cooldown-seconds",
            str(args.runtime_degraded_cooldown_seconds),
            "--runtime-degraded-detector",
            args.runtime_degraded_detector,
            "--runtime-window-size",
            str(args.runtime_window_size),
            "--runtime-window-min-samples",
            str(args.runtime_window_min_samples),
            "--runtime-window-percentile",
            str(args.runtime_window_percentile),
            "--runtime-window-enter-ms",
            str(args.runtime_window_enter_ms),
            "--runtime-window-exit-ms",
            str(args.runtime_window_exit_ms),
            "--runtime-half-open-probe-interval-seconds",
            str(args.runtime_half_open_probe_interval_seconds),
        ]
    )
    if args.cloud_delay_profile:
        main_args.extend(["--cloud-delay-profile", args.cloud_delay_profile])
    if args.cloud_delay_cursor_file:
        main_args.extend(["--cloud-delay-cursor-file", args.cloud_delay_cursor_file])
    if args.quiet:
        main_args.append("--quiet")

    return main_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.enable_experiment_logging and not args.experiment_run_id:
        args.experiment_run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    controller = ExecuteTaskController(
        python_executable=args.python_executable,
        main_script_path=Path(args.main_script_path),
        working_directory=REPO_ROOT,
        main_args=build_main_args(args),
    )
    service = create_execute_service(
        controller=controller,
        host=args.host,
        port=args.port,
        web_root=Path(args.web_root),
    )

    host = service.server_address[0]
    port = service.server_address[1]
    print("AutoGLM execute service is ready.")
    print(f"  Console: http://{host}:{port}/")
    print(f"  Health:  http://{host}:{port}/health")
    print(f"  Execute: http://{host}:{port}/execute")
    if args.enable_experiment_logging:
        print(
            "  Experiment logs: "
            f"{Path(args.experiment_log_dir).expanduser() / args.experiment_run_id}"
        )
    print("Press Ctrl+C to stop.")

    try:
        service.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AutoGLM execute service...")
    finally:
        service.shutdown()
        service.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
