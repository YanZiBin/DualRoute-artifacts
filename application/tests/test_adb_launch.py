# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# DualRoute analysis/runtime source; original experimental scope is documented in the release provenance.
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE_MODULE_PATH = PROJECT_ROOT / "phone_agent" / "adb" / "device.py"
DEVICE_SPEC = importlib.util.spec_from_file_location(
    "phone_agent_adb_device_under_test", DEVICE_MODULE_PATH
)
adb_device = importlib.util.module_from_spec(DEVICE_SPEC)
assert DEVICE_SPEC.loader is not None
DEVICE_SPEC.loader.exec_module(adb_device)

from phone_agent.config.apps import get_package_name


class ADBLaunchTests(unittest.TestCase):
    def test_taobao_instant_retail_uses_rebranded_eleme_package(self):
        self.assertEqual(get_package_name("淘宝闪购"), "me.ele")
        self.assertEqual(get_package_name("饿了么"), "me.ele")
        self.assertEqual(get_package_name("淘宝"), "com.taobao.taobao")

    def test_extract_foreground_package(self):
        dump = (
            "mCurrentFocus=Window{f5cbb7 u0 "
            "com.tencent.mobileqq/com.tencent.mobileqq.activity.SplashActivity}"
        )
        self.assertEqual(
            adb_device._extract_foreground_package(dump),
            "com.tencent.mobileqq",
        )

    def test_launch_app_returns_false_for_unknown_app(self):
        self.assertFalse(adb_device.launch_app("不存在的App"))

    @patch.object(adb_device.time, "sleep", return_value=None)
    @patch.object(adb_device.subprocess, "run")
    def test_launch_app_succeeds_when_monkey_reaches_foreground(
        self, mock_run, _mock_sleep
    ):
        mock_run.side_effect = [
            CompletedProcess(
                args=[],
                returncode=0,
                stdout="Events injected: 1",
                stderr="",
            ),
            CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "mCurrentFocus=Window{f5cbb7 u0 "
                    "com.tencent.mobileqq/com.tencent.mobileqq.activity.SplashActivity}"
                ),
                stderr="",
            ),
        ]

        self.assertTrue(adb_device.launch_app("QQ"))
        self.assertEqual(mock_run.call_count, 2)

    @patch.object(adb_device.time, "sleep", return_value=None)
    @patch.object(adb_device.subprocess, "run")
    def test_launch_app_falls_back_to_explicit_activity(
        self, mock_run, _mock_sleep
    ):
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout="Events injected: 1", stderr=""),
            CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "mCurrentFocus=Window{c63e68b u0 "
                    "com.android.launcher/com.android.launcher.Launcher}"
                ),
                stderr="",
            ),
            CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "priority=0 preferredOrder=0\n"
                    "com.tencent.mobileqq/.activity.SplashActivity\n"
                ),
                stderr="",
            ),
            CompletedProcess(args=[], returncode=0, stdout="Status: ok", stderr=""),
            CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "mCurrentFocus=Window{f5cbb7 u0 "
                    "com.tencent.mobileqq/com.tencent.mobileqq.activity.SplashActivity}"
                ),
                stderr="",
            ),
        ]

        self.assertTrue(adb_device.launch_app("QQ"))
        self.assertEqual(mock_run.call_count, 5)

    @patch.object(adb_device.time, "sleep", return_value=None)
    @patch.object(adb_device.subprocess, "run")
    def test_launch_app_returns_false_when_chooser_stays_in_front(
        self, mock_run, _mock_sleep
    ):
        mock_run.side_effect = [
            CompletedProcess(args=[], returncode=0, stdout="Events injected: 1", stderr=""),
            CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "mCurrentFocus=Window{c63e68b u0 "
                    "com.android.launcher/com.android.launcher.Launcher}"
                ),
                stderr="",
            ),
            CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "priority=0 preferredOrder=0\n"
                    "com.tencent.mm/.ui.LauncherUI\n"
                ),
                stderr="",
            ),
            CompletedProcess(args=[], returncode=0, stdout="Activity: android/com.android.internal.app.ChooserActivity", stderr=""),
            CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "mCurrentFocus=Window{abcd123 u0 "
                    "android/com.android.internal.app.ChooserActivity}"
                ),
                stderr="",
            ),
        ]

        self.assertFalse(adb_device.launch_app("WeChat"))
        self.assertEqual(mock_run.call_count, 5)


if __name__ == "__main__":
    unittest.main()
