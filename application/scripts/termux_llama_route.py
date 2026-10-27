# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phone_agent.config.route_prompt import get_route_prompt  # noqa: E402



DEFAULT_LLAMA_SIMPLE_PATH = Path.home() / "llama.cpp" / "build" / "bin" / "llama-simple"
DEFAULT_GGUF_MODEL_PATH = (
    Path.home()
    / "AutoGLM"
    / "models-gguf"
    / "modelA_v18_4_q8_0_20260901.gguf"
)
ASSISTANT_MARKER = "<|im_start|>assistant"


class RouteOutputParseError(RuntimeError):
    """Raised when llama-simple output does not contain a route label."""


def build_prompt(text: str) -> str:
    """Build the Qwen chat-template prompt used for phone-side routing."""
    return (
        f"<|im_start|>system\n{get_route_prompt()}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def extract_label(raw_stdout: str) -> str:
    """Extract the first route label generated after the assistant marker."""
    assistant_index = raw_stdout.rfind(ASSISTANT_MARKER)
    if assistant_index >= 0:
        candidate = raw_stdout[assistant_index + len(ASSISTANT_MARKER) :]
    else:
        candidate = raw_stdout

    match = re.search(r"\b([012])\b", candidate)
    if match:
        return match.group(1)

    stripped = candidate.strip()
    if stripped in {"0", "1", "2"}:
        return stripped

    raise RouteOutputParseError(
        "Could not extract route label from llama-simple output."
    )


def build_command(
    llama_simple_path: Path,
    model_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    """Build the llama-simple command line."""
    return [
        str(llama_simple_path),
        "-m",
        str(model_path),
        "-ngl",
        "0",
        "-n",
        str(max_new_tokens),
        prompt,
    ]


def run_route(
    text: str,
    llama_simple_path: Path,
    model_path: Path,
    max_new_tokens: int = 2,
    timeout_seconds: float = 60.0,
) -> tuple[str, subprocess.CompletedProcess[str]]:
    """Run llama-simple and return the parsed route label plus raw process output."""
    if not text.strip():
        raise ValueError("Input text cannot be empty.")
    if not llama_simple_path.exists():
        raise FileNotFoundError(f"llama-simple not found: {llama_simple_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    prompt = build_prompt(text)
    command = build_command(
        llama_simple_path=llama_simple_path,
        model_path=model_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )

    if result.returncode != 0:
        stderr_tail = result.stderr.strip()[-800:]
        raise RuntimeError(
            f"llama-simple failed with exit code {result.returncode}. "
            f"stderr tail: {stderr_tail}"
        )

    label = extract_label(result.stdout)
    return label, result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run phone-side Fast Path route classification via llama.cpp."
    )
    parser.add_argument("text", help="User instruction to classify.")
    parser.add_argument(
        "--llama-simple-path",
        default=str(DEFAULT_LLAMA_SIMPLE_PATH),
        help=f"Path to llama-simple (default: {DEFAULT_LLAMA_SIMPLE_PATH})",
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
        help="Subprocess timeout in seconds. Default: 60",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print raw stdout and stderr to stderr for debugging.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    llama_simple_path = Path(args.llama_simple_path).expanduser()
    model_path = Path(args.model_path).expanduser()

    try:
        label, result = run_route(
            text=args.text,
            llama_simple_path=llama_simple_path,
            model_path=model_path,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(f"[termux_llama_route] {exc}", file=sys.stderr)
        return 1

    if args.show_raw:
        print("=== STDOUT ===", file=sys.stderr)
        print(result.stdout, file=sys.stderr, end="")
        print("\n=== STDERR ===", file=sys.stderr)
        print(result.stderr, file=sys.stderr, end="")
        print("\n=== END ===", file=sys.stderr)

    print(label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
