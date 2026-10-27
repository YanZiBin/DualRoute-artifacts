# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from pathlib import Path
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phone_agent.fast_path import FastPathRouteService, FastPathRouter, RemoteRoutePredictor


class FastPathRouteServiceTests(unittest.TestCase):
    def test_route_endpoint_exposes_debug_fields(self):
        service = FastPathRouteService(
            lambda _text: {"label": "0", "raw_prediction": "0"},
            host="127.0.0.1",
            port=0,
        )
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.1)

        try:
            request = urllib_request.Request(
                f"{service.base_url}/route",
                data=json.dumps({"text": "打开美团"}).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urllib_request.urlopen(request, timeout=5.0) as response:
                parsed = json.loads(response.read().decode("utf-8"))

            self.assertEqual(parsed["label"], "0")
            self.assertEqual(parsed["raw_prediction"], "0")
            self.assertIn("pid", parsed)
            self.assertIn("memory", parsed)
            self.assertIn("vmrss_kb", parsed["memory"])
            self.assertIn("vmsize_kb", parsed["memory"])
            self.assertIn("threads", parsed["memory"])
        finally:
            service.shutdown()
            thread.join(timeout=2.0)

    def test_remote_predictor_reads_label_from_service(self):
        service = FastPathRouteService(lambda _text: "1", host="127.0.0.1", port=0)
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.1)

        try:
            predictor = RemoteRoutePredictor(service.base_url, timeout_seconds=5.0)
            self.assertEqual(predictor.predict_label("打开飞书"), "1")
        finally:
            service.shutdown()
            thread.join(timeout=2.0)

    def test_router_can_use_remote_predictor_for_fallback(self):
        service = FastPathRouteService(lambda _text: "0", host="127.0.0.1", port=0)
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.1)

        try:
            predictor = RemoteRoutePredictor(service.base_url, timeout_seconds=5.0)
            router = FastPathRouter(predictor.predict_label)
            decision = router.decide("打开Chrome")

            self.assertEqual(decision.reason, "unknown_target")
        finally:
            service.shutdown()
            thread.join(timeout=2.0)

    def test_remote_predictor_reports_connection_failure(self):
        predictor = RemoteRoutePredictor("http://127.0.0.1:1", timeout_seconds=0.2)

        with self.assertRaises(RuntimeError):
            predictor.predict_label("打开QQ")


if __name__ == "__main__":
    unittest.main()
