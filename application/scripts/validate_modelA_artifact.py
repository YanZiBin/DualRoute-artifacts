# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelA_artifact_utils import (
    detect_model_artifact_kind,
    evaluate_critical_cases,
)
from phone_agent.config.route_prompt import get_route_prompt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAINING_WORKSPACE = REPO_ROOT.parent / "模型训练专用工作间"
DEFAULT_BASE_MODEL_PATH = TRAINING_WORKSPACE / "models" / "QwenQwen2.5-0.5B-Instruct"
DEFAULT_MODEL_PATH = TRAINING_WORKSPACE / "training" / "modelA"
DEFAULT_CASES_PATH = (
    TRAINING_WORKSPACE
    / "training"
    / "evaluations"
    / "cases"
    / "targeted_validation_cases_round5.json"
)
DEFAULT_OUTPUT_PATH = (
    TRAINING_WORKSPACE
    / "training"
    / "evaluations"
    / "results"
    / f"modelA_validation_{date.today().strftime('%Y%m%d')}.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Model A adapter or merged artifact")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Adapter or merged Hugging Face model directory",
    )
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Base model directory used when --model-path points to a LoRA adapter",
    )
    parser.add_argument(
        "--cases-file",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Validation cases JSON file",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Validation results JSON path",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device",
    )
    return parser.parse_args()


def resolve_device_mode(requested: str) -> tuple[str | None, torch.dtype]:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "auto", torch.float16

    if requested == "cpu":
        return None, torch.float32

    if torch.cuda.is_available():
        return "auto", torch.float16
    return None, torch.float32


def load_model_and_tokenizer(
    model_path: Path,
    base_model_path: Path,
    device_map: str | None,
    torch_dtype: torch.dtype,
) -> tuple[object, object, str]:
    artifact_kind = detect_model_artifact_kind(model_path)

    if artifact_kind == "adapter":
        tokenizer = AutoTokenizer.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(model, str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if device_map is None:
        model.to("cpu")

    return model, tokenizer, artifact_kind


def normalize_prediction(raw_prediction: str) -> str:
    raw_prediction = raw_prediction.strip()
    if raw_prediction in {"0", "1", "2"}:
        return raw_prediction

    match = re.search(r"\b([012])\b", raw_prediction)
    if match:
        return match.group(1)

    return raw_prediction


def predict(model, tokenizer, text: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": get_route_prompt()},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0, inputs.input_ids.shape[1] :]
    raw_prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return normalize_prediction(raw_prediction), raw_prediction


def main() -> None:
    args = parse_args()
    get_route_prompt()
    device_map, torch_dtype = resolve_device_mode(args.device)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = json.loads(args.cases_file.read_text(encoding="utf-8-sig"))
    model, tokenizer, artifact_kind = load_model_and_tokenizer(
        args.model_path,
        args.base_model_path,
        device_map,
        torch_dtype,
    )

    results: list[dict[str, object]] = []
    pass_count = 0

    print("=" * 60)
    print("Validate Model A artifact")
    print("=" * 60)
    print(f"Model path    : {args.model_path}")
    print(f"Artifact kind : {artifact_kind}")
    print(f"Cases file    : {args.cases_file}")
    print(f"Output path   : {args.output_path}")

    for index, case in enumerate(cases, 1):
        text = str(case["text"])
        expected = str(case["expected"])
        prediction, raw_prediction = predict(model, tokenizer, text)
        passed = prediction == expected
        if passed:
            pass_count += 1

        print(
            f"[{index}/{len(cases)}] {text} -> 预测 {prediction} / 期望 {expected} / "
            f"{'PASS' if passed else 'FAIL'}"
        )
        results.append(
            {
                "text": text,
                "expected": expected,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "pass": passed,
            }
        )

    critical_summary = evaluate_critical_cases(results)
    payload = {
        "model_path": str(args.model_path),
        "artifact_kind": artifact_kind,
        "cases_file": str(args.cases_file),
        "pass_count": pass_count,
        "total": len(cases),
        "accuracy": pass_count / len(cases) if cases else 0.0,
        "critical_summary": critical_summary,
        "results": results,
    }
    args.output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("-" * 60)
    print(f"通过: {pass_count}/{len(cases)}")
    print(f"关键样例通过: {critical_summary['passed']}")


if __name__ == "__main__":
    main()
