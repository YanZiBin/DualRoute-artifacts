# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phone_agent.fast_path import FastPathRouteService  # noqa: E402
from termux_llama_route import (  # noqa: E402
    DEFAULT_GGUF_MODEL_PATH,
    DEFAULT_LLAMA_SIMPLE_PATH,
    extract_label,
    run_route,
)

DEFAULT_WORKER_PATH = Path.home() / "AutoGLM" / "scripts" / "bin" / "termux_llama_route_worker"


class LlamaCppRoutePredictor:
    """Wrap the verified llama.cpp one-shot script into the existing route-service contract."""

    def __init__(
        self,
        llama_simple_path: Path,
        model_path: Path,
        max_new_tokens: int = 2,
        timeout_seconds: float = 60.0,
        runner: Callable[..., tuple[str, object]] | None = None,
    ) -> None:
        self.llama_simple_path = Path(llama_simple_path)
        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self._runner = runner or run_route

    def predict(self, text: str) -> dict[str, str]:
        label, result = self._runner(
            text=text,
            llama_simple_path=self.llama_simple_path,
            model_path=self.model_path,
            max_new_tokens=self.max_new_tokens,
            timeout_seconds=self.timeout_seconds,
        )
        raw_stdout = getattr(result, "stdout", "")
        raw_prediction = extract_label(raw_stdout)
        return {
            "label": label,
            "raw_prediction": raw_prediction,
        }


class PersistentWorkerRoutePredictor:
    """Keep a worker process alive so the GGUF model can stay loaded across requests."""

    def __init__(
        self,
        worker_path: Path,
        model_path: Path,
        max_new_tokens: int = 2,
        process_factory: Callable[..., subprocess.Popen[str]] | None = None,
    ) -> None:
        self.worker_path = Path(worker_path)
        self.model_path = Path(model_path)
        self.max_new_tokens = max_new_tokens
        self._process_factory = process_factory or subprocess.Popen
        self._process: subprocess.Popen[str] | None = None

    def _start_process(self) -> subprocess.Popen[str]:
        command = [
            str(self.worker_path),
            "--model-path",
            str(self.model_path),
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        return self._process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._process = self._start_process()
        return self._process

    def predict(self, text: str) -> dict[str, str]:
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        process = self._ensure_process()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Persistent worker pipes are not available.")

        process.stdin.write(text + "\n")
        process.stdin.flush()
        response_line = process.stdout.readline()
        if not response_line:
            raise RuntimeError("Persistent worker exited before returning a response.")

        payload = json.loads(response_line)
        error_message = payload.get("error")
        if isinstance(error_message, str) and error_message:
            raise RuntimeError(error_message)

        label = payload.get("label")
        raw_prediction = payload.get("raw_prediction")
        if label in {"0", "1", "2"} and isinstance(raw_prediction, str):
            return {
                "label": str(label),
                "raw_prediction": raw_prediction,
            }
        raise RuntimeError(f"Persistent worker returned invalid payload: {payload!r}")

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3.0)
        self._process = None


def create_route_service(
    predictor: object,
    host: str,
    port: int,
) -> FastPathRouteService:
    """Create a Fast Path route service compatible with the existing HTTP contract."""
    return FastPathRouteService(predictor.predict, host=host, port=port)  # type: ignore[arg-type]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a local Fast Path /route service backed by llama.cpp GGUF inference."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the local route service. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the local route service. Default: 8765",
    )
    parser.add_argument(
        "--backend",
        choices=("oneshot", "worker"),
        default="oneshot",
        help="Route backend kind. Default: oneshot",
    )
    parser.add_argument(
        "--llama-simple-path",
        default=str(DEFAULT_LLAMA_SIMPLE_PATH),
        help=f"Path to llama-simple (default: {DEFAULT_LLAMA_SIMPLE_PATH})",
    )
    parser.add_argument(
        "--worker-path",
        default=str(DEFAULT_WORKER_PATH),
        help=f"Path to persistent worker binary (default: {DEFAULT_WORKER_PATH})",
    )
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_GGUF_MODEL_PATH),
        help=f"Path to GGUF model (default: {DEFAULT_GGUF_MODEL_PATH})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2,
        help="Maximum new tokens to generate. Default: 2",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for each llama.cpp subprocess call. Default: 60",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    model_path = Path(args.model_path).expanduser()
    predictor: object
    if args.backend == "worker":
        predictor = PersistentWorkerRoutePredictor(
            worker_path=Path(args.worker_path).expanduser(),
            model_path=model_path,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        predictor = LlamaCppRoutePredictor(
            llama_simple_path=Path(args.llama_simple_path).expanduser(),
            model_path=model_path,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
        )

    service = create_route_service(
        predictor=predictor,
        host=args.host,
        port=args.port,
    )

    print("Fast Path llama.cpp route service is ready.")
    print(f"  Backend:  {args.backend}")
    print(f"  Base URL: {service.base_url}")
    print(f"  Health:   {service.base_url}/health")
    print(f"  Route:    {service.base_url}/route")
    print("Press Ctrl+C to stop.")

    try:
        service.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Fast Path llama.cpp route service...")
    finally:
        service.shutdown()
        if hasattr(predictor, "close"):
            predictor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
