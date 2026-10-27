# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

from pathlib import Path


def _has_full_model_weights(artifact_dir: Path) -> bool:
    candidates = (
        artifact_dir / "model.safetensors",
        artifact_dir / "pytorch_model.bin",
        artifact_dir / "model.safetensors.index.json",
        artifact_dir / "pytorch_model.bin.index.json",
    )
    if any(path.exists() for path in candidates):
        return True

    if list(artifact_dir.glob("model-*.safetensors")):
        return True
    if list(artifact_dir.glob("pytorch_model-*.bin")):
        return True

    return False


def detect_model_artifact_kind(model_path: str | Path) -> str:
    artifact_dir = Path(model_path)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Model artifact path not found: {artifact_dir}")
    if not artifact_dir.is_dir():
        raise ValueError(f"Model artifact path must be a directory: {artifact_dir}")

    has_adapter_config = (artifact_dir / "adapter_config.json").exists()
    has_adapter_weights = (artifact_dir / "adapter_model.safetensors").exists()
    has_full_config = (artifact_dir / "config.json").exists()
    has_full_weights = _has_full_model_weights(artifact_dir)

    if has_adapter_config and has_adapter_weights:
        return "adapter"
    if has_full_config and has_full_weights:
        return "merged_hf"

    raise ValueError(
        "Unable to determine model artifact kind from directory contents: "
        f"{artifact_dir}"
    )


def build_result_index(results: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(item["text"]): item
        for item in results
        if isinstance(item, dict) and "text" in item
    }


def evaluate_critical_cases(results: list[dict[str, object]]) -> dict[str, object]:
    result_index = build_result_index(results)

    checks = {
        "打开美团": {
            "expected_rule": "prediction == 0",
            "prediction": str(result_index.get("打开美团", {}).get("prediction")),
            "passed": str(result_index.get("打开美团", {}).get("prediction")) == "0",
        },
        "打开铁路12306": {
            "expected_rule": "prediction == 0",
            "prediction": str(result_index.get("打开铁路12306", {}).get("prediction")),
            "passed": str(result_index.get("打开铁路12306", {}).get("prediction")) == "0",
        },
        "打开美团点奶茶": {
            "expected_rule": "prediction == 1",
            "prediction": str(result_index.get("打开美团点奶茶", {}).get("prediction")),
            "passed": str(result_index.get("打开美团点奶茶", {}).get("prediction")) == "1",
        },
        "12306": {
            "expected_rule": "prediction != 0",
            "prediction": str(result_index.get("12306", {}).get("prediction")),
            "passed": str(result_index.get("12306", {}).get("prediction")) != "0",
        },
    }

    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def compare_prediction_sets(
    baseline_results: list[dict[str, object]],
    candidate_results: list[dict[str, object]],
) -> dict[str, object]:
    baseline_index = build_result_index(baseline_results)
    candidate_index = build_result_index(candidate_results)

    changes: list[dict[str, object]] = []
    for text, baseline_item in baseline_index.items():
        candidate_item = candidate_index.get(text)
        if candidate_item is None:
            continue

        baseline_prediction = str(baseline_item.get("prediction"))
        candidate_prediction = str(candidate_item.get("prediction"))
        if baseline_prediction == candidate_prediction:
            continue

        changes.append(
            {
                "text": text,
                "baseline_prediction": baseline_prediction,
                "candidate_prediction": candidate_prediction,
                "baseline_expected": baseline_item.get("expected"),
                "candidate_expected": candidate_item.get("expected"),
            }
        )

    return {
        "changed_count": len(changes),
        "changes": changes,
    }
