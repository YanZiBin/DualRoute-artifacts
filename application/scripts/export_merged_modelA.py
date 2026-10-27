# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import argparse
import shutil
from datetime import date
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TRAINING_WORKSPACE = REPO_ROOT.parent / "模型训练专用工作间"
DEFAULT_BASE_MODEL_PATH = TRAINING_WORKSPACE / "models" / "QwenQwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER_PATH = TRAINING_WORKSPACE / "training" / "modelA"
DEFAULT_OUTPUT_DIR = (
    TRAINING_WORKSPACE
    / "training"
    / "exports"
    / f"modelA_v5_merged_hf_{date.today().strftime('%Y%m%d')}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge Qwen base model with Model A LoRA")
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Base model directory",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help="LoRA adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Merged Hugging Face model output directory",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device for merge",
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


def main() -> None:
    args = parse_args()
    device_map, torch_dtype = resolve_device_mode(args.device)

    if not args.base_model_path.exists():
        raise FileNotFoundError(f"Base model path not found: {args.base_model_path}")
    if not args.adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Export merged Model A artifact")
    print("=" * 60)
    print(f"Base model : {args.base_model_path}")
    print(f"Adapter    : {args.adapter_path}")
    print(f"Output dir : {args.output_dir}")
    print(f"Device mode: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model_path),
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    merged_model = model.merge_and_unload()

    if device_map is None:
        merged_model.to("cpu")

    merged_model.save_pretrained(
        str(args.output_dir),
        safe_serialization=True,
    )
    tokenizer.save_pretrained(str(args.output_dir))

    adapter_chat_template = args.adapter_path / "chat_template.jinja"
    if adapter_chat_template.exists():
        shutil.copy2(adapter_chat_template, args.output_dir / "chat_template.jinja")

    print("Merged model export finished.")


if __name__ == "__main__":
    main()
