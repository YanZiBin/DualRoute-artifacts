# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import ast
import importlib.util
import sys
from types import SimpleNamespace
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_SPEC = importlib.util.spec_from_file_location("autoglm_main_metadata_test", PROJECT_ROOT / "main.py")
assert _SPEC is not None and _SPEC.loader is not None
main_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = main_module
_SPEC.loader.exec_module(main_module)


class SlowPathKeyboardPrecheckTests(unittest.TestCase):
    def test_completion_metadata_is_normalized_for_finish_task(self):
        args = SimpleNamespace(
            experiment_candidate_set_json='["淘宝", "京东"]',
            experiment_completed_candidate="京东",
            experiment_extra_non_sensitive_action_json='["return_home"]',
        )
        self.assertEqual(
            main_module.build_experiment_completion_metadata(args),
            {
                "candidate_set": ["淘宝", "京东"],
                "completed_candidate": "京东",
                "extra_non_sensitive_action": ["return_home"],
            },
        )

    def test_slow_path_skips_keyboard_precheck(self):
        module = ast.parse((PROJECT_ROOT / "main.py").read_text(encoding="utf-8"))
        slow_path_environment = next(
            node
            for node in ast.walk(module)
            if isinstance(node, ast.FunctionDef)
            and node.name == "ensure_slow_path_environment"
        )
        system_check = next(
            node
            for node in ast.walk(slow_path_environment)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "check_system_requirements"
        )
        require_keyboard = next(
            keyword
            for keyword in system_check.keywords
            if keyword.arg == "require_keyboard"
        )

        self.assertIsInstance(require_keyboard.value, ast.Constant)
        self.assertIs(require_keyboard.value.value, False)


if __name__ == "__main__":
    unittest.main()
