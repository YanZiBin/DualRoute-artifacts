# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors: routing/runtime integration and/or release packaging; upstream attribution retained.
# -*- coding: utf-8 -*-
"""Main PhoneAgent class for orchestrating phone automation."""

import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

from phone_agent.actions import ActionHandler
from phone_agent.actions.handler import do, finish, parse_action
from phone_agent.config import get_messages, get_system_prompt
from phone_agent.config.timing import TIMING_CONFIG
from phone_agent.device_factory import get_device_factory
from phone_agent.experiment_logger import get_experiment_logger
from phone_agent.model import ModelClient, ModelConfig
from phone_agent.model.client import MessageBuilder


@dataclass
class AgentConfig:
    """Configuration for the PhoneAgent."""

    max_steps: int = 100
    device_id: str | None = None
    lang: str = "cn"
    system_prompt: str | None = None
    verbose: bool = True
    screenshot_prewarm_enabled: bool = True
    # Per-step hooks for the degraded-mode sliding-window experiment (issue #6).
    # step_cloud_delay_provider() -> ms to sleep before each cloud call (network
    # degradation injection at the per-step granularity). step_latency_sink(ms)
    # receives each step's model_total_ms so the runtime health monitor can keep
    # a per-step latency window. Both default to None (no behavior change).
    step_cloud_delay_provider: Callable[[], float] | None = None
    step_latency_sink: Callable[[float], None] | None = None
    experiment_runtime: Any | None = None
    step_availability_failure_sink: Callable[[float], None] | None = None
    step_cloud_failure_provider: Callable[[str], str | None] | None = None

    def __post_init__(self):
        if self.system_prompt is None:
            self.system_prompt = get_system_prompt(self.lang)


@dataclass
class StepResult:
    """Result of a single agent step."""

    success: bool
    finished: bool
    action: dict[str, Any] | None
    thinking: str
    message: str | None = None
    user_action_required: bool = False


class PhoneAgent:
    """
    AI-powered agent for automating Android phone interactions.

    The agent uses a vision-language model to understand screen content
    and decide on actions to complete user tasks.

    Args:
        model_config: Configuration for the AI model.
        agent_config: Configuration for the agent behavior.
        confirmation_callback: Optional callback for sensitive action confirmation.
        takeover_callback: Optional callback for takeover requests.

    Example:
        >>> from phone_agent import PhoneAgent
        >>> from phone_agent.model import ModelConfig
        >>>
        >>> model_config = ModelConfig(base_url="http://localhost:8000/v1")
        >>> agent = PhoneAgent(model_config)
        >>> agent.run("Open WeChat and send a message to John")
    """

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        agent_config: AgentConfig | None = None,
        confirmation_callback: Callable[[str], bool] | None = None,
        takeover_callback: Callable[[str], None] | None = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.agent_config = agent_config or AgentConfig()

        self.model_client = ModelClient(self.model_config)
        self.action_handler = ActionHandler(
            device_id=self.agent_config.device_id,
            confirmation_callback=confirmation_callback,
            takeover_callback=takeover_callback,
        )

        self._context: list[dict[str, Any]] = []
        self._step_count = 0
        self._user_action_required = False

        if self.agent_config.screenshot_prewarm_enabled:
            try:
                if self.agent_config.verbose:
                    print("Pre-warming screenshot pipeline...")
                get_device_factory().get_screenshot(self.agent_config.device_id)
            except Exception as e:
                if self.agent_config.verbose:
                    print(f"Pre-warm failed: {e}")

    def _begin_text_input_session(self) -> None:
        """Switch to ADB Keyboard once for the whole Android slow-path task."""
        if self.action_handler.activate_adb_keyboard_session():
            time.sleep(TIMING_CONFIG.action.keyboard_switch_delay)
            if self.agent_config.verbose:
                print("⌨️ Slow Path session switched to ADB Keyboard.")

    def _end_text_input_session(self) -> None:
        """Restore the original Android keyboard after the slow-path task ends."""
        if self.action_handler.restore_adb_keyboard_session():
            time.sleep(TIMING_CONFIG.action.keyboard_restore_delay)
            if self.agent_config.verbose:
                print("⌨️ Slow Path session restored the original keyboard.")

    def run(self, task: str) -> str:
        """
        Run the agent to complete a task.

        Args:
            task: Natural language description of the task.

        Returns:
            Final message from the agent.
        """
        self.reset()
        self._begin_text_input_session()
        try:
            # First step with user prompt
            result = self._execute_step(task, is_first=True)

            if result.finished:
                return result.message or "Task completed"

            # Continue until finished or max steps reached
            while self._step_count < self.agent_config.max_steps:
                result = self._execute_step(is_first=False)

                if result.finished:
                    return result.message or "Task completed"

            return "Max steps reached"
        finally:
            self._end_text_input_session()

    def step(self, task: str | None = None) -> StepResult:
        """
        Execute a single step of the agent.

        Useful for manual control or debugging.

        Args:
            task: Task description (only needed for first step).

        Returns:
            StepResult with step details.
        """
        is_first = len(self._context) == 0

        if is_first and not task:
            raise ValueError("Task is required for the first step")

        if is_first:
            self._begin_text_input_session()

        result = self._execute_step(task, is_first)
        if result.finished:
            self._end_text_input_session()
        return result

    def reset(self) -> None:
        """Reset the agent state for a new task."""
        self._end_text_input_session()
        self._context = []
        self._step_count = 0
        self._user_action_required = False

    def _execute_step(
        self, user_prompt: str | None = None, is_first: bool = False
    ) -> StepResult:
        """Execute a single step of the agent loop."""
        step_start = time.perf_counter()
        self._step_count += 1
        if self.agent_config.experiment_runtime is not None:
            self.last_step_success = False

        # Capture current screen state
        device_factory = get_device_factory()
        screenshot_start = time.perf_counter()
        screenshot = device_factory.get_screenshot(self.agent_config.device_id)
        screenshot_total_ms = (time.perf_counter() - screenshot_start) * 1000
        current_app_start = time.perf_counter()
        current_app = device_factory.get_current_app(self.agent_config.device_id)
        current_app_ms = (time.perf_counter() - current_app_start) * 1000

        # Build messages
        message_build_start = time.perf_counter()
        if is_first:
            self._context.append(
                MessageBuilder.create_system_message(self.agent_config.system_prompt)
            )

            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"{user_prompt}\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content,
                    image_base64=screenshot.base64_data,
                    image_mime=screenshot.image_mime,
                )
            )
        else:
            screen_info = MessageBuilder.build_screen_info(current_app)
            text_content = f"** Screen Info **\n\n{screen_info}"

            self._context.append(
                MessageBuilder.create_user_message(
                    text=text_content,
                    image_base64=screenshot.base64_data,
                    image_mime=screenshot.image_mime,
                )
            )

        message_build_ms = (time.perf_counter() - message_build_start) * 1000

        runtime = self.agent_config.experiment_runtime
        call = None
        if runtime is None:
            # Per-step network degradation injection (issue #6 sliding-window exp).
            # The injected delay models a slower network round-trip, so it is added
            # to the per-step latency reported to the health window below.
            injected_delay_ms = 0.0
            provider = self.agent_config.step_cloud_delay_provider
            if provider is not None:
                try:
                    injected_delay_ms = float(provider() or 0.0)
                except Exception:
                    injected_delay_ms = 0.0
                if injected_delay_ms > 0:
                    if self.agent_config.verbose:
                        print(f"Injecting per-step cloud delay: {injected_delay_ms:.1f} ms")
                    time.sleep(injected_delay_ms / 1000.0)

            # Get model response
            try:
                msgs = get_messages(self.agent_config.lang)
                print("\n" + "=" * 50)
                print(f"💭 {msgs['thinking']}:")
                print("-" * 50)
                response = self.model_client.request(self._context)
            except Exception as e:
                if self.agent_config.verbose:
                    traceback.print_exc()
                return StepResult(
                    success=False,
                    finished=True,
                    action=None,
                    thinking="",
                    message=f"Model error: {e}",
                )

        else:
            call = runtime.begin_call("cloud_step", step_index=self._step_count) if runtime is not None else None
            injected_delay_ms = 0.0
            model_error = None
            request_started = False
            response = None
            try:
                provider = self.agent_config.step_cloud_delay_provider
                if provider is not None:
                    try:
                        injected_delay_ms = float(provider() or 0.0)
                    except Exception:
                        raise
                    if injected_delay_ms > 0:
                        if self.agent_config.verbose:
                            print(f"Injecting per-step cloud delay: {injected_delay_ms:.1f} ms")
                        delay_start = time.monotonic()
                        try:
                            time.sleep(injected_delay_ms / 1000.0)
                        finally:
                            if call is not None:
                                call["actual_delay_wait_ms"] = (time.monotonic() - delay_start) * 1000
                if call is not None and call["fault_applied"] != "none":
                    from phone_agent.experiment_runtime import InjectedAvailabilityError

                    failure = self.agent_config.step_cloud_failure_provider(call["fault_applied"])
                    if failure:
                        raise InjectedAvailabilityError(failure)
                msgs = get_messages(self.agent_config.lang)
                print("\n" + "=" * 50)
                print(f"💭 {msgs['thinking']}:")
                print("-" * 50)
                if call is not None:
                    call["model_invoked"] = True
                request_started = True
                response = self.model_client.request(self._context)
            except BaseException as exc:
                model_error = exc
                if runtime is None and not isinstance(exc, Exception):
                    raise
            finally:
                if call is not None:
                    call["model_total_ms"] = (
                        response.total_time * 1000 + call["actual_delay_wait_ms"]
                        if response is not None and response.total_time is not None else None
                    )
                    measured_model_ms = runtime.end_call(
                        call, success=model_error is None,
                        error=type(model_error).__name__ + ": " + str(model_error) if model_error is not None else None,
                    )
            if runtime is not None:
                # Failed calls remain call evidence. Only injected unavailability
                # uses the existing availability-failure seam; never infer it from
                # an arbitrary model exception.
                from phone_agent.experiment_runtime import ExperimentContextError, InjectedAvailabilityError

                if model_error is not None and not isinstance(model_error, Exception):
                    raise model_error
                if model_error is not None and not request_started and not isinstance(model_error, InjectedAvailabilityError):
                    raise model_error
                if isinstance(model_error, InjectedAvailabilityError):
                    # Unavailable is injected at the existing failure seam, before
                    # invoking a model. It is not a completed workflow sample.
                    if self.agent_config.step_availability_failure_sink is not None:
                        self.agent_config.step_availability_failure_sink(measured_model_ms)
            if model_error is not None:
                if self.agent_config.verbose:
                    traceback.print_exception(type(model_error), model_error, model_error.__traceback__)
                return StepResult(
                    success=False, finished=True, action=None, thinking="",
                    message=f"Model error: {model_error}",
                )

        # Parse action from response
        try:
            action = parse_action(response.action)
        except ValueError:
            if self.agent_config.verbose:
                traceback.print_exc()
            action = finish(message=response.action)

        if self.agent_config.verbose:
            # Print thinking process
            print("-" * 50)
            print(f"🎯 {msgs['action']}:")
            print(json.dumps(action, ensure_ascii=False, indent=2))
            print("=" * 50 + "\n")

        # Remove image from context to save space
        self._context[-1] = MessageBuilder.remove_images_from_message(self._context[-1])

        # Execute action
        try:
            if action.get("_metadata") == "do":
                get_experiment_logger().mark_first_action(action.get("action"))
            action_start = time.perf_counter()
            result = self.action_handler.execute(
                action, screenshot.width, screenshot.height
            )
            action_execute_ms = (time.perf_counter() - action_start) * 1000
            self._user_action_required = (
                self._user_action_required or result.user_action_required
            )
        except Exception as e:
            if self.agent_config.verbose:
                traceback.print_exc()
            action_start = time.perf_counter()
            result = self.action_handler.execute(
                finish(message=str(e)), screenshot.width, screenshot.height
            )
            action_execute_ms = (time.perf_counter() - action_start) * 1000

        # Add assistant response to context
        self._context.append(
            MessageBuilder.create_assistant_message(
                f"<think>{response.thinking}</think><answer>{response.action}</answer>"
            )
        )

        # Check if finished
        finished = action.get("_metadata") == "finish" or result.should_finish
        step_total_ms = (time.perf_counter() - step_start) * 1000
        get_experiment_logger().record_step(
            {
                "step_index": self._step_count,
                "is_first_step": is_first,
                "current_app": current_app,
                "screenshot_total_ms": round(screenshot_total_ms, 3),
                "screenshot_capture_ms": getattr(screenshot, "capture_ms", None),
                "image_encode_ms": getattr(screenshot, "image_encode_ms", None),
                "image_base64_ms": getattr(screenshot, "base64_ms", None),
                "image_payload_bytes": getattr(screenshot, "payload_bytes", None),
                "image_mime": getattr(screenshot, "image_mime", None),
                "image_width": getattr(screenshot, "width", None),
                "image_height": getattr(screenshot, "height", None),
                "jpeg_quality": getattr(screenshot, "jpeg_quality", None),
                "current_app_ms": round(current_app_ms, 3),
                "message_build_ms": round(message_build_ms, 3),
                "perception_latency_ms": round(
                    screenshot_total_ms + current_app_ms + message_build_ms, 3
                ),
                "model_ttft_ms": round(response.time_to_first_token * 1000, 3)
                if response.time_to_first_token is not None
                else None,
                "model_thinking_end_ms": round(response.time_to_thinking_end * 1000, 3)
                if response.time_to_thinking_end is not None
                else None,
                "model_total_ms": (
                    round(response.total_time * 1000 + call["actual_delay_wait_ms"], 3)
                    if runtime is not None and response.total_time is not None else (
                        round(response.total_time * 1000, 3) if response.total_time is not None else None
                    )
                ),
                **({"phase_applied": call["phase_applied"], "call_id": call["call_id"],
                    "fault_applied": call["fault_applied"]} if call is not None else {}),
                "action_name": action.get("action"),
                "action_metadata": action.get("_metadata"),
                "action_execute_ms": round(action_execute_ms, 3),
                "step_total_ms": round(step_total_ms, 3),
                "injected_cloud_delay_ms": round(injected_delay_ms, 3),
                "finished": finished,
                "success": result.success,
                "user_action_required": result.user_action_required,
                "message": result.message or action.get("message"),
            }
        )

        # Report this step's effective cloud round-trip to the health window.
        # Effective = real model time + injected network delay (issue #6).
        sink = self.agent_config.step_latency_sink
        if sink is not None and response.total_time is not None:
            if runtime is not None:
                sink(response.total_time * 1000.0 + call["actual_delay_wait_ms"])
            else:
                try:
                    sink(response.total_time * 1000.0 + injected_delay_ms)
                except Exception:
                    pass

        if finished and self.agent_config.verbose:
            msgs = get_messages(self.agent_config.lang)
            print("\n" + "🎉 " + "=" * 48)
            print(
                f"✅ {msgs['task_completed']}: {result.message or action.get('message', msgs['done'])}"
            )
            print("=" * 50 + "\n")

        if runtime is not None:
            self.last_step_success = result.success
        return StepResult(
            success=result.success,
            finished=finished,
            action=action,
            thinking=response.thinking,
            message=result.message or action.get("message"),
            user_action_required=result.user_action_required,
        )

    @property
    def context(self) -> list[dict[str, Any]]:
        """Get the current conversation context."""
        return self._context.copy()

    @property
    def step_count(self) -> int:
        """Get the current step count."""
        return self._step_count

    @property
    def user_action_required(self) -> bool:
        """Whether the last run paused for login, CAPTCHA, permission, or choice."""
        return self._user_action_required
