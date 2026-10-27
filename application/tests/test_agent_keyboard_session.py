# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors for the prompt-excluded release; numerical/parser/guard scope is documented in docs/REPRODUCTION_SCOPE.md.
from __future__ import annotations

import unittest
from unittest import mock

from phone_agent.actions.handler import ActionHandler
from phone_agent.agent import AgentConfig, PhoneAgent, StepResult
from phone_agent.device_factory import DeviceType


class ActionHandlerKeyboardSessionTests(unittest.TestCase):
    @mock.patch("phone_agent.actions.handler.time.sleep", return_value=None)
    def test_type_uses_existing_keyboard_session_without_repeated_switch(
        self, _sleep_mock
    ):
        device_factory = mock.Mock()
        device_factory.device_type = DeviceType.ADB
        device_factory.detect_and_set_adb_keyboard.return_value = "ime.original/.Ime"

        handler = ActionHandler(device_id="device-1")
        with mock.patch(
            "phone_agent.actions.handler.get_device_factory", return_value=device_factory
        ):
            self.assertTrue(handler.activate_adb_keyboard_session())

            result = handler.execute(
                {"_metadata": "do", "action": "Type", "text": "PPT"},
                1080,
                2400,
            )

            self.assertTrue(result.success)
            device_factory.detect_and_set_adb_keyboard.assert_called_once_with("device-1")
            device_factory.clear_text.assert_called_once_with("device-1")
            device_factory.type_text.assert_called_once_with("PPT", "device-1")
            device_factory.restore_keyboard.assert_not_called()

            self.assertTrue(handler.restore_adb_keyboard_session())
            device_factory.restore_keyboard.assert_called_once_with(
                "ime.original/.Ime", "device-1"
            )


class PhoneAgentKeyboardSessionTests(unittest.TestCase):
    @mock.patch("phone_agent.agent.time.sleep", return_value=None)
    @mock.patch("builtins.print")
    def test_run_switches_to_adb_keyboard_once_and_restores_after_finish(
        self, _print_mock, _sleep_mock
    ):
        device_factory = mock.Mock()
        device_factory.device_type = DeviceType.ADB
        device_factory.get_screenshot.return_value = mock.Mock()

        action_handler = mock.Mock()
        action_handler.activate_adb_keyboard_session.return_value = True
        action_handler.restore_adb_keyboard_session.side_effect = [False, True]

        with mock.patch(
            "phone_agent.agent.get_device_factory", return_value=device_factory
        ), mock.patch(
            "phone_agent.agent.ActionHandler", return_value=action_handler
        ), mock.patch("phone_agent.agent.ModelClient"):
            agent = PhoneAgent(agent_config=AgentConfig(system_prompt="test-only supplied prompt"))
            with mock.patch.object(
                agent,
                "_execute_step",
                return_value=StepResult(
                    success=True,
                    finished=True,
                    action=None,
                    thinking="",
                    message="done",
                ),
            ):
                result = agent.run("打开小红书搜索PPT")

        self.assertEqual(result, "done")
        action_handler.activate_adb_keyboard_session.assert_called_once_with()
        self.assertEqual(action_handler.restore_adb_keyboard_session.call_count, 2)


if __name__ == "__main__":
    unittest.main()
