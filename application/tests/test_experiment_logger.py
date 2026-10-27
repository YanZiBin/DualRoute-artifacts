# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import csv
import json
import time
import tempfile
import unittest
from pathlib import Path

from phone_agent.experiment_logger import (
    ExperimentConfig,
    ROUTE_FAST,
    ROUTE_REJECT,
    ROUTE_SLOW,
    configure_experiment_logger,
)


class ExperimentLoggerTests(unittest.TestCase):
    def test_writes_task_route_step_and_csv_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = configure_experiment_logger(
                ExperimentConfig(
                    enabled=True,
                    log_dir=Path(temp_dir),
                    run_id="run-1",
                    experiment_group="final",
                    variant="hybrid",
                    expected_route=ROUTE_FAST,
                    expected_success=True,
                    repeat_index=2,
                )
            )

            task_id = logger.start_task("打开美团", "task-1")
            logger.record_route(
                {
                    "predicted_route": ROUTE_FAST,
                    "route_label": "0",
                    "route_latency_ms": 123.4,
                }
            )
            logger.mark_first_action("Launch")
            logger.record_step(
                {
                    "step_index": 1,
                    "action_name": "Launch",
                    "image_payload_bytes": 100,
                }
            )
            logger.finish_task(
                success=True,
                result_text="ok",
                step_count=1,
                candidate_set=["淘宝", "京东"],
                completed_candidate="京东",
                extra_non_sensitive_action=["return_home"],
            )

            run_dir = Path(temp_dir) / "run-1"
            self.assertEqual(task_id, "task-1")
            self.assertTrue((run_dir / "config.json").exists())

            task_payload = json.loads(
                (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(task_payload["task_id"], "task-1")
            self.assertEqual(task_payload["predicted_route"], ROUTE_FAST)
            self.assertTrue(task_payload["route_correct"])
            self.assertTrue(task_payload["route_success"])
            self.assertEqual(task_payload["candidate_set"], ["淘宝", "京东"])
            self.assertEqual(task_payload["completed_candidate"], "京东")
            self.assertEqual(task_payload["extra_non_sensitive_action"], ["return_home"])
            self.assertTrue(task_payload["success"])
            self.assertIsNotNone(task_payload["ttfa_ms"])

            route_payload = json.loads(
                (run_dir / "routes.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(route_payload["task_text"], "打开美团")

            step_payload = json.loads(
                (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(step_payload["image_payload_bytes"], 100)

            with (run_dir / "task_metrics.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["task_id"], "task-1")
            self.assertEqual(rows[0]["predicted_route"], ROUTE_FAST)

    def test_records_deadline_miss_and_runtime_policy_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = configure_experiment_logger(
                ExperimentConfig(
                    enabled=True,
                    log_dir=Path(temp_dir),
                    run_id="deadline-run",
                )
            )

            logger.start_task("打开微信", "deadline-task")
            logger.record_route(
                {
                    "predicted_route": ROUTE_FAST,
                    "route_label": "0",
                    "deadline_ms": 1.0,
                    "deadline_metric": "ttfa_ms",
                    "runtime_mode": "normal",
                    "admission_decision": "admit",
                    "admission_reason": "route admitted",
                    "network_profile": "normal",
                }
            )
            time.sleep(0.005)
            logger.mark_first_action("Launch")
            logger.finish_task(success=True, result_text="ok", step_count=1,
                               candidate_set=["淘宝", "京东"],
                               completed_candidate="京东",
                               extra_non_sensitive_action=["return_home"])

            run_dir = Path(temp_dir) / "deadline-run"
            task_payload = json.loads(
                (run_dir / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(task_payload["deadline_metric"], "ttfa_ms")
            self.assertEqual(task_payload["runtime_mode"], "normal")
            self.assertEqual(task_payload["admission_decision"], "admit")
            self.assertTrue(task_payload["deadline_miss"])

            with (run_dir / "task_metrics.csv").open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["deadline_metric"], "ttfa_ms")
            self.assertEqual(row["deadline_miss"], "True")

    def test_keeps_semantic_and_route_success_separate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = configure_experiment_logger(
                ExperimentConfig(
                    enabled=True,
                    log_dir=Path(temp_dir),
                    run_id="semantic-run",
                    expected_outcome="execute",
                )
            )
            logger.start_task("打开美团并搜索火锅", "semantic-task")
            logger.record_route(
                {"predicted_route": ROUTE_REJECT, "final_route": ROUTE_REJECT}
            )
            logger.finish_task(success=True, result_text="已拒绝", step_count=0)

            payload = json.loads(
                (Path(temp_dir) / "semantic-run" / "tasks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertFalse(payload["semantic_success"])
            self.assertFalse(payload["route_success"])
            self.assertTrue(payload["success"])

    def test_user_action_required_is_not_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = configure_experiment_logger(
                ExperimentConfig(
                    enabled=True,
                    log_dir=Path(temp_dir),
                    run_id="blocked-run",
                    expected_outcome="execute",
                )
            )
            logger.start_task("打开美团", "blocked-task")
            logger.record_route(
                {"predicted_route": ROUTE_SLOW, "final_route": ROUTE_SLOW}
            )
            logger.finish_task(
                success=False,
                result_text="需要登录",
                user_action_required=True,
            )
            payload = json.loads(
                (Path(temp_dir) / "blocked-run" / "tasks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertIsNone(payload["semantic_success"])
            self.assertTrue(payload["user_action_required"])


if __name__ == "__main__":
    unittest.main()
