# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "autoglm_main_under_test", MAIN_PATH
)
main_module = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
MODULE_SPEC.loader.exec_module(main_module)


class MainFastPathBackendTests(unittest.TestCase):
    def test_real_router_factory_preserves_injected_multi_launch_evidence(self):
        for raw in "012":
            with self.subTest(raw=raw), mock.patch.object(main_module, "RemoteRoutePredictor") as remote:
                remote.return_value.predict_label.return_value = raw
                router = main_module.create_fast_path_router(
                    True, main_module.DeviceType.ADB, "/unused/base", "/unused/model",
                    router_service_url="http://unused.invalid",
                )
                text = "依次打开淘宝和京东。"
                result = router.decide(text)
                self.assertEqual(result.final_route, "reject" if raw == "2" else "slow_path")
                self.assertEqual(result.raw_prediction, raw)
                remote.return_value.predict_label.assert_called_once_with(text)

    def _make_args(self, **overrides):
        defaults = {
            "fast_path_router_backend": "auto",
            "fast_path_base_model_path": "/tmp/base",
            "fast_path_model_a_path": "/tmp/modelA",
            "fast_path_router_service_host": "127.0.0.1",
            "fast_path_router_service_port": 8765,
            "fast_path_gguf_model_path": "/tmp/model.gguf",
            "fast_path_llama_simple_path": "/tmp/llama-simple",
            "fast_path_llama_worker_path": "/tmp/termux-worker",
            "fast_path_router_max_new_tokens": 2,
            "fast_path_router_timeout_seconds": 60.0,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_resolve_backend_auto_prefers_llama_worker_in_termux(self):
        with mock.patch.dict(os.environ, {"TERMUX_VERSION": "0.119"}, clear=False):
            self.assertEqual(
                main_module.resolve_fast_path_router_backend("auto"),
                "llama-worker",
            )

    def test_resolve_backend_auto_uses_pytorch_outside_termux(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                main_module.resolve_fast_path_router_backend("auto"),
                "pytorch",
            )

    def test_build_llama_router_service_argv_uses_worker_backend(self):
        args = self._make_args()

        argv = main_module.build_llama_router_service_argv(args, "llama-worker")

        self.assertEqual(
            argv,
            [
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--backend",
                "worker",
                "--max-new-tokens",
                "2",
                "--timeout-seconds",
                "60.0",
                "--model-path",
                "/tmp/model.gguf",
                "--llama-simple-path",
                "/tmp/llama-simple",
                "--worker-path",
                "/tmp/termux-worker",
            ],
        )

    def test_start_fast_path_router_service_dispatches_to_pytorch_backend(self):
        args = self._make_args(fast_path_router_backend="pytorch")

        with mock.patch.object(main_module, "serve_fast_path_route_service") as serve_mock:
            main_module.start_fast_path_router_service(args)

        serve_mock.assert_called_once_with(
            base_model_path="/tmp/base",
            model_a_path="/tmp/modelA",
            host="127.0.0.1",
            port=8765,
        )

    def test_start_fast_path_router_service_dispatches_to_llama_module(self):
        args = self._make_args(fast_path_router_backend="llama-worker")
        fake_module = mock.Mock()

        with mock.patch.object(
            main_module,
            "_load_termux_llama_route_service_module",
            return_value=fake_module,
        ):
            main_module.start_fast_path_router_service(args)

        fake_module.main.assert_called_once_with(
            [
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--backend",
                "worker",
                "--max-new-tokens",
                "2",
                "--timeout-seconds",
                "60.0",
                "--model-path",
                "/tmp/model.gguf",
                "--llama-simple-path",
                "/tmp/llama-simple",
                "--worker-path",
                "/tmp/termux-worker",
            ]
        )


if __name__ == "__main__":
    unittest.main()
