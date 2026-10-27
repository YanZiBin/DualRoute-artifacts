# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from termux_llama_route import (  # noqa: E402
    DEFAULT_GGUF_MODEL_PATH,
    DEFAULT_LLAMA_SIMPLE_PATH,
    run_route,
)


DEFAULT_CASES_PATH = SCRIPT_DIR / "termux_llama_regression_cases.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
CRITICAL_CASE_RULES = {
    "打开美团": lambda prediction: prediction == "0",
    "打开铁路12306": lambda prediction: prediction == "0",
    "打开美团点奶茶": lambda prediction: prediction == "1",
    "12306": lambda prediction: prediction != "0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch llama.cpp phone-side regression cases."
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"Regression cases JSON file (default: {DEFAULT_CASES_PATH})",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to scripts/outputs/termux_llama_regression_<timestamp>.json",
    )
    parser.add_argument(
        "--llama-simple-path",
        type=Path,
        default=None,
        help="Optional override for llama-simple path.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional override for GGUF model path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2,
        help="Maximum new tokens for each prediction. Default: 2",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for each subprocess call. Default: 60",
    )
    return parser.parse_args()


def load_cases(cases_file: Path) -> list[dict[str, str]]:
    payload = json.loads(cases_file.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Cases file must contain a JSON list.")

    cases: list[dict[str, str]] = []
    for index, item in enumerate(payload, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Case #{index} must be a JSON object.")
        text = str(item.get("text", "")).strip()
        expected_prediction = str(item.get("expected_prediction", "")).strip()
        if not text:
            raise ValueError(f"Case #{index} is missing a non-empty text field.")
        if expected_prediction not in {"0", "1", "2"}:
            raise ValueError(
                f"Case #{index} has invalid expected_prediction: {expected_prediction!r}"
            )
        case = {
            "text": text,
            "expected_prediction": expected_prediction,
            "experiment_task_id": str(
                item.get("experiment_task_id", item.get("task_id", f"case_{index:03d}"))
            ).strip(),
            "expected_outcome": str(item.get("expected_outcome", "execute")).strip(),
        }
        if not case["experiment_task_id"]:
            raise ValueError(f"Case #{index} has an empty experiment_task_id.")
        if case["expected_outcome"] not in {"execute", "refuse"}:
            raise ValueError(
                f"Case #{index} has invalid expected_outcome: {case['expected_outcome']!r}"
            )
        note = str(item.get("note", "")).strip()
        if note:
            case["note"] = note
        cases.append(case)
    return cases


def run_cases(
    cases: list[dict[str, str]],
    predictor: Callable[[str], str],
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    pass_count = 0

    for index, case in enumerate(cases, 1):
        text = case["text"]
        expected_prediction = case["expected_prediction"]
        note = case.get("note")
        try:
            prediction = predictor(text)
            error_message = None
        except Exception as exc:  # noqa: BLE001
            prediction = "ERROR"
            error_message = str(exc)

        passed = prediction == expected_prediction
        if passed:
            pass_count += 1

        row: dict[str, object] = {
            "text": text,
            "expected_prediction": expected_prediction,
            "experiment_task_id": case.get("experiment_task_id") or f"case_{index:03d}",
            "expected_outcome": case.get("expected_outcome", "execute"),
            "prediction": prediction,
            "pass": passed,
        }
        if note:
            row["note"] = note
        if error_message:
            row["error"] = error_message

        results.append(row)
        if not passed:
            failures.append(row)

    critical_input = [
        {"text": str(item["text"]), "prediction": str(item["prediction"])}
        for item in results
    ]
    critical_summary = evaluate_critical_cases(critical_input)

    return {
        "pass_count": pass_count,
        "total": len(cases),
        "accuracy": pass_count / len(cases) if cases else 0.0,
        "critical_summary": critical_summary,
        "failures": failures,
        "results": results,
    }


def evaluate_critical_cases(results: list[dict[str, str]]) -> dict[str, object]:
    """Evaluate the four phone-side critical route cases."""
    prediction_by_text = {str(item["text"]): str(item["prediction"]) for item in results}
    checks: dict[str, dict[str, object]] = {}
    all_passed = True

    for text, rule in CRITICAL_CASE_RULES.items():
        prediction = prediction_by_text.get(text)
        passed = prediction is not None and rule(prediction)
        if not passed:
            all_passed = False
        expected_rule = {
            "打开美团": "prediction == 0",
            "打开铁路12306": "prediction == 0",
            "打开美团点奶茶": "prediction == 1",
            "12306": "prediction != 0",
        }[text]
        checks[text] = {
            "expected_rule": expected_rule,
            "prediction": prediction,
            "passed": passed,
        }

    return {
        "passed": all_passed,
        "checks": checks,
    }


def default_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"termux_llama_regression_{timestamp}.json"


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases_file)
    output_path = args.output_path or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    llama_simple_path = args.llama_simple_path or DEFAULT_LLAMA_SIMPLE_PATH
    model_path = args.model_path or DEFAULT_GGUF_MODEL_PATH

    def predictor(text: str) -> str:
        label, _result = run_route(
            text=text,
            llama_simple_path=llama_simple_path,
            model_path=model_path,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout_seconds,
        )
        return label

    payload = run_cases(cases, predictor=predictor)
    payload.update(
        {
            "cases_file": str(args.cases_file),
            "output_path": str(output_path),
            "llama_simple_path": str(llama_simple_path),
            "model_path": str(model_path),
        }
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("Termux llama.cpp regression")
    print("=" * 60)
    print(f"Cases file        : {args.cases_file}")
    print(f"Output path       : {output_path}")
    print(f"llama-simple path : {llama_simple_path}")
    print(f"Model path        : {model_path}")
    print("-" * 60)
    for index, item in enumerate(payload["results"], 1):
        print(
            f"[{index}/{payload['total']}] {item['text']} -> "
            f"预测 {item['prediction']} / 期望 {item['expected_prediction']} / "
            f"{'PASS' if item['pass'] else 'FAIL'}"
        )
    print("-" * 60)
    print(f"通过: {payload['pass_count']}/{payload['total']}")
    print(f"关键样例通过: {payload['critical_summary']['passed']}")
    print(f"失败数: {len(payload['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
