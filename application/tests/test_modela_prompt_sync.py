# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from phone_agent.config import get_system_prompt
from phone_agent.config.route_prompt import get_route_prompt
from phone_agent.fast_path import FastPathRouter, LocalRouteModel


ROOT = Path(__file__).resolve().parents[1]


class ReleasePromptBoundaryTests(unittest.TestCase):
    def test_saved_output_replay_needs_no_prompt(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            decision = FastPathRouter(lambda text: " 0").decide("打开淘宝")
        self.assertEqual(decision.final_route, "fast_path")

    def test_missing_prompts_fail_before_loading_model(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DUALROUTE_ROUTE_PROMPT_FILE"):
                get_route_prompt()
            with self.assertRaisesRegex(RuntimeError, "DUALROUTE_EXECUTION_PROMPT_ZH_FILE"):
                get_system_prompt("cn")
            model = LocalRouteModel("/unused/base", "/unused/model")
            with mock.patch.object(model, "_ensure_loaded") as load_model:
                for operation in [lambda: model.predict("test"), model.load]:
                    with self.assertRaisesRegex(RuntimeError, "DUALROUTE_ROUTE_PROMPT_FILE"):
                        operation()
                load_model.assert_not_called()

    def test_validation_main_rejects_missing_prompt_before_model_or_case_io(self):
        # Execute the exact main() body without importing unbundled ML dependencies.
        source = (ROOT / "scripts/validate_modelA_artifact.py").read_text(encoding="utf-8")
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "main")
        module = ast.Module(body=[function], type_ignores=[])
        args = mock.Mock()
        namespace = {
            "parse_args": mock.Mock(return_value=args),
            "get_route_prompt": get_route_prompt,
            "resolve_device_mode": mock.Mock(),
            "load_model_and_tokenizer": mock.Mock(),
            "json": json,
        }
        exec(compile(module, "validate_modelA_artifact.main", "exec"), namespace)
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DUALROUTE_ROUTE_PROMPT_FILE"):
                namespace["main"]()
        namespace["resolve_device_mode"].assert_not_called()
        namespace["load_model_and_tokenizer"].assert_not_called()
        args.output_path.parent.mkdir.assert_not_called()
        args.cases_file.read_text.assert_not_called()

    def test_supplied_prompt_is_read_verbatim_and_empty_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "operator-prompt.txt"
            value = "operator-supplied test fixture\n"
            path.write_text(value, encoding="utf-8")
            with mock.patch.dict(os.environ, {
                "DUALROUTE_ROUTE_PROMPT_FILE": str(path),
                "DUALROUTE_EXECUTION_PROMPT_ZH_FILE": str(path),
                "DUALROUTE_EXECUTION_PROMPT_EN_FILE": str(path),
            }):
                self.assertEqual(get_route_prompt(), value)
                self.assertEqual(get_system_prompt("cn"), value)
                self.assertEqual(get_system_prompt("en"), value)
                self.assertEqual(LocalRouteModel("/unused/base", "/unused/model").SYSTEM_PROMPT, value)
                spec = importlib.util.spec_from_file_location("release_termux_prompt", ROOT / "scripts/termux_llama_route.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.assertEqual(module.build_prompt("test"), "<|im_start|>system\n" + value + "<|im_end|>\n<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n")
                path.write_text(" \n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    get_route_prompt()


if __name__ == "__main__":
    unittest.main()
