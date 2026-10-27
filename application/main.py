#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors: routing/runtime integration and/or release packaging; upstream attribution retained.
"""
Phone Agent CLI - AI-powered phone automation.

Usage:
    python main.py [OPTIONS]

Environment Variables:
    PHONE_AGENT_BASE_URL: Model API base URL (default: http://localhost:8000/v1)
    PHONE_AGENT_MODEL: Model name (default: autoglm-phone-9b)
    PHONE_AGENT_API_KEY: API key for model authentication (default: EMPTY)
    PHONE_AGENT_MAX_STEPS: Maximum steps per task (default: 100)
    PHONE_AGENT_DEVICE_ID: ADB device ID for multi-device setups
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from phone_agent.config.fast_path_apps import list_fast_path_apps
from phone_agent.device_factory import DeviceType, get_device_factory, set_device_type
from phone_agent.experiment_logger import (
    ExperimentConfig,
    ROUTE_FAST,
    ROUTE_REJECT,
    ROUTE_SLOW,
    configure_experiment_logger,
    default_run_id,
    parse_optional_bool,
)
from phone_agent.fast_path import (
    FastPathDecisionKind,
    FastPathRouter,
    LocalRouteModel,
    RemoteRoutePredictor,
    execute_fast_path_decision,
    format_fast_path_decision,
    serve_fast_path_route_service,
)
from phone_agent.runtime_health import RuntimeHealthConfig, RuntimeHealthMonitor
from phone_agent.runtime_policy import (
    ADMISSION_REJECT,
    RuntimePolicyConfig,
    decide_runtime_admission,
)


FAST_PATH_ROUTER_BACKENDS = (
    "auto",
    "pytorch",
    "llama-oneshot",
    "llama-worker",
)


def _default_fast_path_base_model_path() -> str:
    """Return the default local base-model path for Fast Path routing."""
    repo_root = Path(__file__).resolve().parent
    return str(
        repo_root.parent / "模型训练专用工作间" / "models" / "QwenQwen2.5-0.5B-Instruct"
    )


def _default_fast_path_model_a_path() -> str:
    """Return the default local Model A path for Fast Path routing."""
    repo_root = Path(__file__).resolve().parent
    return str(repo_root.parent / "模型训练专用工作间" / "training" / "modelA")


def _default_experiment_log_dir() -> str:
    """Return the default structured experiment log directory."""
    return str(Path(__file__).resolve().parent / "experiment_logs")


def _default_runtime_health_file() -> str:
    return str(Path(_default_experiment_log_dir()) / "runtime_health.json")


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def is_termux_environment() -> bool:
    """Whether the current Python process is running inside Termux."""
    try:
        home = str(Path.home()).lower()
    except RuntimeError:
        home = ""
    prefix = os.getenv("PREFIX", "").lower()
    return bool(os.getenv("TERMUX_VERSION")) or "com.termux" in home or "termux" in prefix


def resolve_fast_path_router_backend(requested_backend: str) -> str:
    """Resolve `auto` into a concrete Fast Path router backend."""
    if requested_backend != "auto":
        return requested_backend
    return "llama-worker" if is_termux_environment() else "pytorch"


def _load_termux_llama_route_service_module():
    """Load the standalone llama.cpp route-service script as a module."""
    module_path = Path(__file__).resolve().parent / "scripts" / "termux_llama_route_service.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Fast Path llama.cpp route service script not found: {module_path}")

    module_name = "autoglm_termux_llama_route_service"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Fast Path llama.cpp route service module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_llama_router_service_argv(args: argparse.Namespace, backend: str) -> list[str]:
    """Build argv for the standalone llama.cpp route-service script."""
    argv = [
        "--host",
        args.fast_path_router_service_host,
        "--port",
        str(args.fast_path_router_service_port),
        "--backend",
        "worker" if backend == "llama-worker" else "oneshot",
        "--max-new-tokens",
        str(args.fast_path_router_max_new_tokens),
        "--timeout-seconds",
        str(args.fast_path_router_timeout_seconds),
    ]

    if args.fast_path_gguf_model_path:
        argv.extend(["--model-path", args.fast_path_gguf_model_path])
    if args.fast_path_llama_simple_path:
        argv.extend(["--llama-simple-path", args.fast_path_llama_simple_path])
    if args.fast_path_llama_worker_path:
        argv.extend(["--worker-path", args.fast_path_llama_worker_path])

    return argv


def start_fast_path_router_service(args: argparse.Namespace) -> None:
    """Start the selected Fast Path router service backend."""
    backend = resolve_fast_path_router_backend(args.fast_path_router_backend)
    if backend == "pytorch":
        serve_fast_path_route_service(
            base_model_path=args.fast_path_base_model_path,
            model_a_path=args.fast_path_model_a_path,
            host=args.fast_path_router_service_host,
            port=args.fast_path_router_service_port,
        )
        return

    route_service_module = _load_termux_llama_route_service_module()
    route_service_module.main(build_llama_router_service_argv(args, backend))


def check_system_requirements(
    device_type: DeviceType = DeviceType.ADB,
    wda_url: str = "http://localhost:8100",
    require_keyboard: bool = True,
    device_id: str | None = None,
) -> bool:
    """
    Check system requirements before running the agent.

    Checks:
    1. ADB/HDC/iOS tools installed
    2. At least one device connected
    3. ADB Keyboard installed on the device (for ADB only, when required)
    4. WebDriverAgent running (for iOS only)

    Args:
        device_type: Type of device tool (ADB, HDC, or IOS).
        wda_url: WebDriverAgent URL (for iOS only).
        require_keyboard: Whether the current execution path requires ADB Keyboard.

    Returns:
        True if all checks pass, False otherwise.
    """
    print("🔍 Checking system requirements...")
    print("-" * 50)

    all_passed = True

    # Determine tool name and command
    if device_type == DeviceType.IOS:
        tool_name = "libimobiledevice"
        tool_cmd = "idevice_id"
    else:
        tool_name = "ADB" if device_type == DeviceType.ADB else "HDC"
        tool_cmd = "adb" if device_type == DeviceType.ADB else "hdc"

    # Check 1: Tool installed
    print(f"1. Checking {tool_name} installation...", end=" ")
    if shutil.which(tool_cmd) is None:
        print("❌ FAILED")
        print(f"   Error: {tool_name} is not installed or not in PATH.")
        print(f"   Solution: Install {tool_name}:")
        if device_type == DeviceType.ADB:
            print("     - macOS: brew install android-platform-tools")
            print("     - Linux: sudo apt install android-tools-adb")
            print(
                "     - Windows: Download from https://developer.android.com/studio/releases/platform-tools"
            )
        elif device_type == DeviceType.HDC:
            print(
                "     - Download from HarmonyOS SDK or https://gitee.com/openharmony/docs"
            )
            print("     - Add to PATH environment variable")
        else:  # IOS
            print("     - macOS: brew install libimobiledevice")
            print("     - Linux: sudo apt-get install libimobiledevice-utils")
        all_passed = False
    else:
        # Double check by running version command
        try:
            if device_type == DeviceType.ADB:
                version_cmd = [tool_cmd, "version"]
            elif device_type == DeviceType.HDC:
                version_cmd = [tool_cmd, "-v"]
            else:  # IOS
                version_cmd = [tool_cmd, "-ln"]

            result = subprocess.run(
                version_cmd, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.strip().split("\n")[0]
                print(f"✅ OK ({version_line if version_line else 'installed'})")
            else:
                print("❌ FAILED")
                print(f"   Error: {tool_name} command failed to run.")
                all_passed = False
        except FileNotFoundError:
            print("❌ FAILED")
            print(f"   Error: {tool_name} command not found.")
            all_passed = False
        except subprocess.TimeoutExpired:
            print("❌ FAILED")
            print(f"   Error: {tool_name} command timed out.")
            all_passed = False

    # If ADB is not installed, skip remaining checks
    if not all_passed:
        print("-" * 50)
        print("❌ System check failed. Please fix the issues above.")
        return False

    # Check 2: Device connected
    print("2. Checking connected devices...", end=" ")
    try:
        if device_type == DeviceType.ADB:
            adb_cmd = ["adb", "-s", device_id] if device_id else ["adb"]
            result = subprocess.run(
                adb_cmd + ["devices"], capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            # Filter out header and empty lines, look for 'device' status
            devices = [
                line for line in lines[1:] if line.strip() and "\tdevice" in line
            ]
        elif device_type == DeviceType.HDC:
            result = subprocess.run(
                ["hdc", "list", "targets"], capture_output=True, text=True, timeout=10
            )
            lines = result.stdout.strip().split("\n")
            devices = [line for line in lines if line.strip()]
        else:  # IOS
            from phone_agent.xctest import list_devices as list_ios_devices

            ios_devices = list_ios_devices()
            devices = [d.device_id for d in ios_devices]

        if not devices:
            print("❌ FAILED")
            print("   Error: No devices connected.")
            print("   Solution:")
            if device_type == DeviceType.ADB:
                print("     1. Enable USB debugging on your Android device")
                print("     2. Connect via USB and authorize the connection")
                print(
                    "     3. Or connect remotely: python main.py --connect <ip>:<port>"
                )
            elif device_type == DeviceType.HDC:
                print("     1. Enable USB debugging on your HarmonyOS device")
                print("     2. Connect via USB and authorize the connection")
                print(
                    "     3. Or connect remotely: python main.py --device-type hdc --connect <ip>:<port>"
                )
            else:  # IOS
                print("     1. Connect your iOS device via USB")
                print("     2. Unlock device and tap 'Trust This Computer'")
                print("     3. Verify: idevice_id -l")
                print("     4. Or connect via WiFi using device IP")
            all_passed = False
        else:
            if device_type == DeviceType.ADB:
                device_ids = [d.split("\t")[0] for d in devices]
            elif device_type == DeviceType.HDC:
                device_ids = [d.strip() for d in devices]
            else:  # IOS
                device_ids = devices
            print(
                f"✅ OK ({len(devices)} device(s): {', '.join(device_ids[:2])}{'...' if len(device_ids) > 2 else ''})"
            )
    except subprocess.TimeoutExpired:
        print("❌ FAILED")
        print(f"   Error: {tool_name} command timed out.")
        all_passed = False
    except Exception as e:
        print("❌ FAILED")
        print(f"   Error: {e}")
        all_passed = False

    # If no device connected, skip ADB Keyboard check
    if not all_passed:
        print("-" * 50)
        print("❌ System check failed. Please fix the issues above.")
        return False

    # Check 3: ADB Keyboard installed (only for ADB when needed) or WebDriverAgent
    if device_type == DeviceType.ADB and require_keyboard:
        print("3. Checking ADB Keyboard...", end=" ")
        try:
            adb_cmd = ["adb", "-s", device_id] if device_id else ["adb"]
            result = subprocess.run(
                adb_cmd + ["shell", "ime", "list", "-s"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            ime_list = result.stdout.strip()

            if "com.android.adbkeyboard/.AdbIME" in ime_list:
                print("✅ OK")
            else:
                print("❌ FAILED")
                print("   Error: ADB Keyboard is not installed on the device.")
                print("   Solution:")
                print("     1. Download ADB Keyboard APK from:")
                print(
                    "        https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk"
                )
                print("     2. Install it on your device: adb install ADBKeyboard.apk")
                print(
                    "     3. Enable it in Settings > System > Languages & Input > Virtual Keyboard"
                )
                all_passed = False
        except subprocess.TimeoutExpired:
            print("❌ FAILED")
            print("   Error: ADB command timed out.")
            all_passed = False
        except Exception as e:
            print("❌ FAILED")
            print(f"   Error: {e}")
            all_passed = False
    elif device_type == DeviceType.ADB:
        print("3. Skipping ADB Keyboard check...", end=" ")
        print("✅ OK (not required for current Fast Path action)")
    elif device_type == DeviceType.HDC:
        # For HDC, skip keyboard check as it uses different input method
        print("3. Skipping keyboard check for HarmonyOS...", end=" ")
        print("✅ OK (using native input)")
    else:  # IOS
        # Check WebDriverAgent
        print(f"3. Checking WebDriverAgent ({wda_url})...", end=" ")
        try:
            conn = XCTestConnection(wda_url=wda_url)

            if conn.is_wda_ready():
                print("✅ OK")
                # Get WDA status for additional info
                status = conn.get_wda_status()
                if status:
                    session_id = status.get("sessionId", "N/A")
                    print(f"   Session ID: {session_id}")
            else:
                print("❌ FAILED")
                print("   Error: WebDriverAgent is not running or not accessible.")
                print("   Solution:")
                print("     1. Run WebDriverAgent on your iOS device via Xcode")
                print("     2. For USB: Set up port forwarding: iproxy 8100 8100")
                print(
                    "     3. For WiFi: Use device IP, e.g., --wda-url http://192.168.1.100:8100"
                )
                print("     4. Verify in browser: open http://localhost:8100/status")
                all_passed = False
        except Exception as e:
            print("❌ FAILED")
            print(f"   Error: {e}")
            all_passed = False

    print("-" * 50)

    if all_passed:
        print("✅ All system checks passed!\n")
    else:
        print("❌ System check failed. Please fix the issues above.")

    return all_passed


def check_model_api(base_url: str, model_name: str, api_key: str = "EMPTY") -> bool:
    """
    Check if the model API is accessible and the specified model exists.

    Checks:
    1. Network connectivity to the API endpoint
    2. Model exists in the available models list

    Args:
        base_url: The API base URL
        model_name: The model name to check
        api_key: The API key for authentication

    Returns:
        True if all checks pass, False otherwise.
    """
    print("🔍 Checking model API...")
    print("-" * 50)

    all_passed = True

    # Check 1: Network connectivity using chat API
    print(f"1. Checking API connectivity ({base_url})...", end=" ")
    try:
        from openai import OpenAI

        # Create OpenAI client
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)

        # Use chat completion to test connectivity (more universally supported than /models)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
            temperature=0.0,
            stream=False,
        )

        # Check if we got a valid response
        if response.choices and len(response.choices) > 0:
            print("✅ OK")
        else:
            print("❌ FAILED")
            print("   Error: Received empty response from API")
            all_passed = False

    except Exception as e:
        print("❌ FAILED")
        error_msg = str(e)

        # Provide more specific error messages
        if "Connection refused" in error_msg or "Connection error" in error_msg:
            print(f"   Error: Cannot connect to {base_url}")
            print("   Solution:")
            print("     1. Check if the model server is running")
            print("     2. Verify the base URL is correct")
            print(f"     3. Try: curl {base_url}/chat/completions")
        elif "timed out" in error_msg.lower() or "timeout" in error_msg.lower():
            print(f"   Error: Connection to {base_url} timed out")
            print("   Solution:")
            print("     1. Check your network connection")
            print("     2. Verify the server is responding")
        elif (
            "Name or service not known" in error_msg
            or "nodename nor servname" in error_msg
        ):
            print(f"   Error: Cannot resolve hostname")
            print("   Solution:")
            print("     1. Check the URL is correct")
            print("     2. Verify DNS settings")
        else:
            print(f"   Error: {error_msg}")

        all_passed = False

    print("-" * 50)

    if all_passed:
        print("✅ Model API checks passed!\n")
    else:
        print("❌ Model API check failed. Please fix the issues above.")

    return all_passed


def create_fast_path_router(
    enable_fast_path: bool,
    device_type: DeviceType,
    base_model_path: str,
    model_a_path: str,
    router_service_url: str | None = None,
) -> FastPathRouter | None:
    """Create the Fast Path router for Android tasks when enabled."""
    if not enable_fast_path or device_type != DeviceType.ADB:
        return None

    if router_service_url:
        remote_predictor = RemoteRoutePredictor(router_service_url)
        return FastPathRouter(remote_predictor.predict_label)

    local_route_model = LocalRouteModel(
        base_model_path=base_model_path,
        model_a_path=model_a_path,
    )
    return FastPathRouter(local_route_model.predict_label)


def create_agent(args, device_type: DeviceType, runtime_health_monitor=None):
    """Create the slow-path agent only when it is actually needed."""
    from phone_agent import IOSPhoneAgent, PhoneAgent
    from phone_agent.agent import AgentConfig
    from phone_agent.agent_ios import IOSAgentConfig
    from phone_agent.model import ModelConfig

    model_config = ModelConfig(
        base_url=args.base_url,
        model_name=args.model,
        api_key=args.apikey,
        lang=args.lang,
    )

    # Per-step hooks for the degraded-mode sliding-window experiment (issue #6).
    # Only active under the sliding_window detector (Android slow path), so the
    # single_miss path keeps its original task-level injection unchanged and we
    # never double-inject. Both default off.
    step_cloud_delay_provider = None
    step_latency_sink = None
    experiment_runtime = getattr(args, "_experiment_runtime", None)
    per_step_mode = (
        device_type == DeviceType.ADB
        and getattr(args, "runtime_degraded_detector", "single_miss") == "sliding_window"
    )
    if per_step_mode:
        if experiment_runtime is not None:
            step_cloud_delay_provider = experiment_runtime.active_step_delay_ms
        elif (getattr(args, "cloud_delay_profile", "") or "") or args.cloud_delay_ms > 0:
            step_cloud_delay_provider = lambda: resolve_cloud_delay_ms(args)
        if runtime_health_monitor is not None:
            step_latency_sink = lambda ms: runtime_health_monitor.record_step_latency(
                model_total_ms=ms
            )

    if device_type == DeviceType.IOS:
        agent_config = IOSAgentConfig(
            max_steps=args.max_steps,
            wda_url=args.wda_url,
            device_id=args.device_id,
            verbose=not args.quiet,
            lang=args.lang,
        )
        agent = IOSPhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
        )
    else:
        agent_config = AgentConfig(
            max_steps=args.max_steps,
            device_id=args.device_id,
            verbose=not args.quiet,
            lang=args.lang,
            screenshot_prewarm_enabled=not args.disable_screenshot_prewarm,
            step_cloud_delay_provider=step_cloud_delay_provider,
            step_latency_sink=step_latency_sink,
            experiment_runtime=experiment_runtime,
            step_availability_failure_sink=(
                lambda ms: runtime_health_monitor.record_cloud_probe(available=False, latency_ms=ms)
            ) if experiment_runtime is not None and runtime_health_monitor is not None else None,
            step_cloud_failure_provider=(
                lambda fault: apply_slow_path_network_degradation(
                    RuntimePolicyConfig(cloud_fail_mode=fault), delay_ms=0.0
                )
            ) if experiment_runtime is not None else None,
        )
        agent = PhoneAgent(
            model_config=model_config,
            agent_config=agent_config,
        )

    return agent, agent_config, model_config


def apply_slow_path_network_degradation(
    config: RuntimePolicyConfig, delay_ms: float | None = None
) -> str | None:
    """Apply configured Slow Path network degradation before cloud execution.

    ``delay_ms`` overrides the constant ``config.cloud_delay_ms`` to support
    time-varying delay injection. When omitted, behavior is identical to the
    original constant-delay path.
    """
    effective_delay_ms = config.cloud_delay_ms if delay_ms is None else delay_ms
    if effective_delay_ms > 0:
        print(f"Simulating Slow Path cloud delay: {effective_delay_ms:.1f} ms")
        time.sleep(effective_delay_ms / 1000.0)

    if config.cloud_fail_mode == "timeout":
        return "Slow Path cloud request timed out under simulated network degradation."
    if config.cloud_fail_mode == "unavailable":
        return "Slow Path cloud endpoint unavailable under simulated network degradation."
    return None


def _cloud_delay_cursor_file(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "cloud_delay_cursor_file", "") or ""
    if explicit:
        return Path(explicit).expanduser()
    return Path(args.runtime_health_file).expanduser().parent / "cloud_delay_cursor.json"


def resolve_cloud_delay_ms(args: argparse.Namespace) -> float:
    """Return the injected cloud delay for the current Slow Path request.

    Without ``--cloud-delay-profile`` this returns the constant
    ``--cloud-delay-ms`` so default behavior is unchanged. With a profile file
    (one delay in ms per non-comment line), successive Slow Path requests
    consume successive values via a persistent cursor, enabling the
    time-varying delay injection used by the degraded-mode sliding-window
    experiment.
    """
    profile_path = getattr(args, "cloud_delay_profile", "") or ""
    if not profile_path:
        return args.cloud_delay_ms
    try:
        lines = Path(profile_path).expanduser().read_text(encoding="utf-8").splitlines()
    except OSError:
        return args.cloud_delay_ms
    values: list[float] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            values.append(float(stripped))
        except ValueError:
            continue
    if not values:
        return args.cloud_delay_ms
    cursor_path = _cloud_delay_cursor_file(args)
    cursor = 0
    try:
        payload = json.loads(cursor_path.read_text(encoding="utf-8"))
        cursor = int(payload.get("cursor", 0))
    except (OSError, json.JSONDecodeError, ValueError, AttributeError):
        cursor = 0
    delay = values[cursor % len(values)]
    try:
        cursor_path.parent.mkdir(parents=True, exist_ok=True)
        cursor_path.write_text(json.dumps({"cursor": cursor + 1}), encoding="utf-8")
    except OSError:
        pass
    return delay


def print_agent_header(args, device_type: DeviceType, agent_config, model_config) -> None:
    """Print the standard slow-path agent header."""
    print("=" * 50)
    if device_type == DeviceType.IOS:
        print("Phone Agent iOS - AI-powered iOS automation")
    else:
        print("Phone Agent - AI-powered phone automation")
    print("=" * 50)
    print(f"Model: {model_config.model_name}")
    print(f"Base URL: {model_config.base_url}")
    print(f"Max Steps: {agent_config.max_steps}")
    print(f"Language: {agent_config.lang}")
    print(f"Device Type: {args.device_type.upper()}")
    if hasattr(agent_config, "screenshot_prewarm_enabled"):
        print(f"Screenshot Prewarm: {agent_config.screenshot_prewarm_enabled}")

    if device_type == DeviceType.IOS:
        from phone_agent.xctest import list_devices as list_ios_devices

        print(f"WDA URL: {args.wda_url}")
        devices = list_ios_devices()
        if agent_config.device_id:
            print(f"Device: {agent_config.device_id}")
        elif devices:
            device = devices[0]
            print(f"Device: {device.device_name or device.device_id[:16]}")
            if device.model and device.ios_version:
                print(f"        {device.model}, iOS {device.ios_version}")
    else:
        device_factory = get_device_factory()
        devices = device_factory.list_devices()
        if agent_config.device_id:
            print(f"Device: {agent_config.device_id}")
        elif devices:
            print(f"Device: {devices[0].device_id} (auto-detected)")

    print("=" * 50)

def _parse_string_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in value.split(",")]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("metadata list must be JSON string array or comma-separated text")
    return [item.strip() for item in parsed if item.strip()]


def build_experiment_completion_metadata(args: argparse.Namespace) -> dict[str, object]:
    """Normalize task-level completion evidence before every experiment finish record."""
    return {
        "candidate_set": _parse_string_list(args.experiment_candidate_set_json),
        "completed_candidate": args.experiment_completed_candidate,
        "extra_non_sensitive_action": _parse_string_list(
            args.experiment_extra_non_sensitive_action_json
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phone Agent - AI-powered phone automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (Android)
    python main.py

    # Run a local Fast Path atomic command
    python main.py "打开微信"

    # Specify model endpoint
    python main.py --base-url http://localhost:8000/v1

    # Use API key for authentication
    python main.py --apikey sk-xxxxx

    # Run with specific device
    python main.py --device-id emulator-5554

    # Connect to remote device
    python main.py --connect 192.168.1.100:5555

    # List connected devices
    python main.py --list-devices

    # Enable TCP/IP on USB device and get connection info
    python main.py --enable-tcpip

    # List supported apps
    python main.py --list-apps

    # iOS specific examples
    # Run with iOS device
    python main.py --device-type ios "Open Safari and search for iPhone tips"

    # Use WiFi connection for iOS
    python main.py --device-type ios --wda-url http://192.168.1.100:8100

    # List connected iOS devices
    python main.py --device-type ios --list-devices

    # Check WebDriverAgent status
    python main.py --device-type ios --wda-status

    # Pair with iOS device
    python main.py --device-type ios --pair
        """,
    )

    # Model options
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("PHONE_AGENT_BASE_URL", "http://localhost:8000/v1"),
        help="Model API base URL",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("PHONE_AGENT_MODEL", "autoglm-phone-9b"),
        help="Model name",
    )

    parser.add_argument(
        "--apikey",
        type=str,
        default=os.getenv("PHONE_AGENT_API_KEY", "EMPTY"),
        help="API key for model authentication",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.getenv("PHONE_AGENT_MAX_STEPS", "100")),
        help="Maximum steps per task",
    )

    parser.add_argument(
        "--enable-experiment-logging",
        action="store_true",
        default=os.getenv("PHONE_AGENT_EXPERIMENT_LOGGING", "").lower()
        in {"1", "true", "yes", "y"},
        help="Enable structured experiment logging for TSR/RA/E2E/TTFA/PL/IPS metrics",
    )

    parser.add_argument(
        "--experiment-log-dir",
        type=str,
        default=os.getenv(
            "PHONE_AGENT_EXPERIMENT_LOG_DIR",
            _default_experiment_log_dir(),
        ),
        help="Directory used for structured experiment logs",
    )

    parser.add_argument(
        "--experiment-run-id",
        type=str,
        default=os.getenv("PHONE_AGENT_EXPERIMENT_RUN_ID", default_run_id()),
        help="Run id subdirectory for experiment logs",
    )

    parser.add_argument(
        "--experiment-group",
        type=str,
        default=os.getenv("PHONE_AGENT_EXPERIMENT_GROUP"),
        help="Experiment group name, e.g. final_hybrid, routing, perception",
    )

    parser.add_argument(
        "--experiment-variant",
        type=str,
        default=os.getenv("PHONE_AGENT_EXPERIMENT_VARIANT"),
        help="Experiment variant name, e.g. final, slow_only, raw_jpeg70_prewarm",
    )

    parser.add_argument(
        "--experiment-expected-route",
        type=str,
        choices=[ROUTE_FAST, ROUTE_SLOW, ROUTE_REJECT],
        default=os.getenv("PHONE_AGENT_EXPERIMENT_EXPECTED_ROUTE"),
        help="Expected route for route-accuracy logging",
    )

    parser.add_argument(
        "--experiment-expected-success",
        type=str,
        default=os.getenv("PHONE_AGENT_EXPERIMENT_EXPECTED_SUCCESS"),
        help="Expected task success for TSR logging: true or false",
    )

    parser.add_argument(
        "--experiment-expected-outcome",
        choices=("execute", "refuse"),
        default=os.getenv("PHONE_AGENT_EXPERIMENT_EXPECTED_OUTCOME"),
        help="Pre-registered semantic contract: execute or refuse",
    )

    parser.add_argument(
        "--experiment-repeat-index",
        type=int,
        default=(
            int(os.getenv("PHONE_AGENT_EXPERIMENT_REPEAT_INDEX"))
            if os.getenv("PHONE_AGENT_EXPERIMENT_REPEAT_INDEX")
            else None
        ),
        help="Repeat index for repeated experiment samples",
    )

    parser.add_argument(
        "--experiment-task-id",
        type=str,
        default=os.getenv("PHONE_AGENT_EXPERIMENT_TASK_ID"),
        help="Optional stable task id for experiment logs",
    )

    parser.add_argument(
        "--experiment-candidate-set-json",
        default=os.getenv("PHONE_AGENT_EXPERIMENT_CANDIDATE_SET_JSON"),
        help="Task candidate set as a JSON string array (or comma-separated text)",
    )
    parser.add_argument(
        "--experiment-completed-candidate",
        default=os.getenv("PHONE_AGENT_EXPERIMENT_COMPLETED_CANDIDATE"),
        help="Candidate actually completed for a one-of task",
    )
    parser.add_argument(
        "--experiment-extra-non-sensitive-action-json",
        default=os.getenv("PHONE_AGENT_EXPERIMENT_EXTRA_NON_SENSITIVE_ACTION_JSON"),
        help="Non-sensitive extra actions as a JSON string array (or comma-separated text)",
    )

    parser.add_argument(
        "--enable-deadline-policy",
        action="store_true",
        default=os.getenv("PHONE_AGENT_ENABLE_DEADLINE_POLICY", "").lower()
        in {"1", "true", "yes", "y"},
        help="Enable deadline-aware runtime admission control and deadline-miss logging",
    )

    parser.add_argument(
        "--fast-path-deadline-ms",
        type=float,
        default=_env_float("PHONE_AGENT_FAST_PATH_DEADLINE_MS", 800.0),
        help="Deadline budget for latency-critical Fast Path interactions",
    )

    parser.add_argument(
        "--slow-path-deadline-ms",
        type=float,
        default=_env_float("PHONE_AGENT_SLOW_PATH_DEADLINE_MS", 5000.0),
        help="Deadline budget for admitted cloud-assisted Slow Path tasks",
    )

    parser.add_argument(
        "--reject-deadline-ms",
        type=float,
        default=_env_float("PHONE_AGENT_REJECT_DEADLINE_MS", 500.0),
        help="Deadline budget for local unsupported-task rejection",
    )

    parser.add_argument(
        "--runtime-degraded-mode",
        choices=("off", "on", "auto"),
        default=os.getenv("PHONE_AGENT_RUNTIME_DEGRADED_MODE", "off"),
        help="Runtime degraded mode: off, on, or auto from configured network degradation",
    )

    parser.add_argument(
        "--degraded-slow-path-admission",
        choices=("admit", "reject"),
        default=os.getenv("PHONE_AGENT_DEGRADED_SLOW_PATH_ADMISSION", "reject"),
        help="Slow Path admission behavior when deadline policy is enabled in degraded mode",
    )

    parser.add_argument(
        "--network-profile",
        type=str,
        default=os.getenv("PHONE_AGENT_NETWORK_PROFILE", "normal"),
        help="Experiment label for network condition, e.g. normal, jitter, timeout, offline",
    )

    parser.add_argument(
        "--cloud-delay-ms",
        type=float,
        default=_env_float("PHONE_AGENT_CLOUD_DELAY_MS", 0.0),
        help="Injected delay before admitted Slow Path cloud execution",
    )

    parser.add_argument(
        "--cloud-fail-mode",
        choices=("none", "timeout", "unavailable"),
        default=os.getenv("PHONE_AGENT_CLOUD_FAIL_MODE", "none"),
        help="Injected Slow Path cloud failure mode for network-degradation experiments",
    )

    parser.add_argument(
        "--enable-runtime-monitor",
        action="store_true",
        default=os.getenv("PHONE_AGENT_ENABLE_RUNTIME_MONITOR", "").lower()
        in {"1", "true", "yes", "y"},
        help="Enable JSON-backed Slow Path and cloud availability runtime health monitoring",
    )

    parser.add_argument(
        "--runtime-health-file",
        type=str,
        default=os.getenv("PHONE_AGENT_RUNTIME_HEALTH_FILE", _default_runtime_health_file()),
        help="Path to the JSON runtime health state shared across per-task processes",
    )

    parser.add_argument(
        "--cloud-probe-deadline-ms",
        type=float,
        default=_env_float("PHONE_AGENT_CLOUD_PROBE_DEADLINE_MS", 8000.0),
        help="Cloud availability probe deadline for runtime health monitoring",
    )

    parser.add_argument(
        "--runtime-degraded-cooldown-seconds",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_DEGRADED_COOLDOWN_SECONDS", 300.0),
        help="Seconds to keep degraded mode active after a runtime health miss",
    )

    parser.add_argument(
        "--runtime-degraded-detector",
        choices=("single_miss", "sliding_window"),
        default=os.getenv("PHONE_AGENT_RUNTIME_DEGRADED_DETECTOR", "single_miss"),
        help=(
            "Degraded-mode trigger: single_miss (default, original behavior) or "
            "sliding_window (latency percentile over recent Slow Path samples)"
        ),
    )

    parser.add_argument(
        "--runtime-window-size",
        type=int,
        default=_env_int("PHONE_AGENT_RUNTIME_WINDOW_SIZE", 10),
        help="Sliding-window size (recent Slow Path latency samples) for degraded detection",
    )

    parser.add_argument(
        "--runtime-window-min-samples",
        type=int,
        default=_env_int("PHONE_AGENT_RUNTIME_WINDOW_MIN_SAMPLES", 5),
        help="Minimum samples before the sliding-window detector changes state",
    )

    parser.add_argument(
        "--runtime-window-percentile",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_WINDOW_PERCENTILE", 90.0),
        help="Latency percentile used by the sliding-window degraded detector",
    )

    parser.add_argument(
        "--runtime-window-enter-ms",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_WINDOW_ENTER_MS", 5000.0),
        help="Enter degraded mode when the windowed latency percentile exceeds this (ms)",
    )

    parser.add_argument(
        "--runtime-window-exit-ms",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_WINDOW_EXIT_MS", 3000.0),
        help="Exit degraded mode when the windowed latency percentile drops below this (ms)",
    )

    parser.add_argument(
        "--runtime-half-open-probe-interval-seconds",
        type=float,
        default=_env_float("PHONE_AGENT_RUNTIME_HALF_OPEN_PROBE_INTERVAL_SECONDS", 15.0),
        help=(
            "Seconds between half-open Slow Path probes while sliding-window "
            "degraded mode is rejecting Slow Path tasks"
        ),
    )

    parser.add_argument(
        "--cloud-delay-profile",
        type=str,
        default=os.getenv("PHONE_AGENT_CLOUD_DELAY_PROFILE", ""),
        help=(
            "Path to a time-varying cloud-delay sequence (one delay in ms per "
            "line); empty uses the constant --cloud-delay-ms"
        ),
    )

    parser.add_argument(
        "--cloud-delay-cursor-file",
        type=str,
        default=os.getenv("PHONE_AGENT_CLOUD_DELAY_CURSOR_FILE", ""),
        help="State file tracking the --cloud-delay-profile cursor across per-task processes",
    )

    parser.add_argument(
        "--routing-mode",
        choices=("hybrid", "cloud_only", "device_only"),
        default=os.getenv("PHONE_AGENT_ROUTING_MODE", "hybrid"),
        help=(
            "Routing ablation mode: hybrid (DualRoute), cloud_only "
            "(always Slow Path), device_only (Fast Path only, complex tasks "
            "are unsupported with no cloud fallback)"
        ),
    )

    parser.add_argument(
        "--disable-fast-path",
        action="store_true",
        help=(
            "Deprecated alias for --routing-mode cloud_only: disable the "
            "local Fast Path router and always use Slow Path"
        ),
    )

    parser.add_argument(
        "--disable-screenshot-prewarm",
        action="store_true",
        default=os.getenv("PHONE_AGENT_DISABLE_SCREENSHOT_PREWARM", "").lower()
        in {"1", "true", "yes", "y"},
        help="Disable the Android slow-path screenshot pre-warm for ablation experiments",
    )

    parser.add_argument(
        "--fast-path-base-model-path",
        type=str,
        default=os.getenv(
            "PHONE_AGENT_FAST_PATH_BASE_MODEL_PATH",
            _default_fast_path_base_model_path(),
        ),
        help="Local base-model path for the Fast Path route classifier",
    )

    parser.add_argument(
        "--fast-path-model-a-path",
        type=str,
        default=os.getenv(
            "PHONE_AGENT_FAST_PATH_MODEL_A_PATH",
            _default_fast_path_model_a_path(),
        ),
        help="Local Model A path for the Fast Path route classifier",
    )

    parser.add_argument(
        "--fast-path-router-service-url",
        type=str,
        default=os.getenv("PHONE_AGENT_FAST_PATH_ROUTER_SERVICE_URL"),
        help="Reuse a running local Fast Path route service instead of loading Model A in each main.py process",
    )

    parser.add_argument(
        "--start-fast-path-router-service",
        action="store_true",
        help="Start a long-lived local Fast Path route service and keep Model A loaded in memory",
    )

    parser.add_argument(
        "--fast-path-router-backend",
        type=str,
        choices=FAST_PATH_ROUTER_BACKENDS,
        default=os.getenv("PHONE_AGENT_FAST_PATH_ROUTER_BACKEND", "auto"),
        help="Fast Path route service backend: auto, pytorch, llama-oneshot, or llama-worker",
    )

    parser.add_argument(
        "--fast-path-router-service-host",
        type=str,
        default=os.getenv("PHONE_AGENT_FAST_PATH_ROUTER_SERVICE_HOST", "127.0.0.1"),
        help="Host for the Fast Path route service (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--fast-path-router-service-port",
        type=int,
        default=int(os.getenv("PHONE_AGENT_FAST_PATH_ROUTER_SERVICE_PORT", "8765")),
        help="Port for the Fast Path route service (default: 8765)",
    )

    parser.add_argument(
        "--fast-path-gguf-model-path",
        type=str,
        default=os.getenv("PHONE_AGENT_FAST_PATH_GGUF_MODEL_PATH"),
        help="GGUF model path for the llama.cpp Fast Path route service",
    )

    parser.add_argument(
        "--fast-path-llama-simple-path",
        type=str,
        default=os.getenv("PHONE_AGENT_FAST_PATH_LLAMA_SIMPLE_PATH"),
        help="llama-simple path for the llama.cpp oneshot Fast Path route service",
    )

    parser.add_argument(
        "--fast-path-llama-worker-path",
        type=str,
        default=os.getenv("PHONE_AGENT_FAST_PATH_LLAMA_WORKER_PATH"),
        help="Persistent worker binary path for the llama.cpp Fast Path route service",
    )

    parser.add_argument(
        "--fast-path-router-max-new-tokens",
        type=int,
        default=int(os.getenv("PHONE_AGENT_FAST_PATH_ROUTER_MAX_NEW_TOKENS", "2")),
        help="Maximum new tokens for the llama.cpp Fast Path route service",
    )

    parser.add_argument(
        "--fast-path-router-timeout-seconds",
        type=float,
        default=float(os.getenv("PHONE_AGENT_FAST_PATH_ROUTER_TIMEOUT_SECONDS", "60")),
        help="Timeout seconds for llama.cpp oneshot Fast Path route requests",
    )

    # Device options
    parser.add_argument(
        "--device-id",
        "-d",
        type=str,
        default=os.getenv("PHONE_AGENT_DEVICE_ID"),
        help="ADB device ID",
    )

    parser.add_argument(
        "--connect",
        "-c",
        type=str,
        metavar="ADDRESS",
        help="Connect to remote device (e.g., 192.168.1.100:5555)",
    )

    parser.add_argument(
        "--disconnect",
        type=str,
        nargs="?",
        const="all",
        metavar="ADDRESS",
        help="Disconnect from remote device (or 'all' to disconnect all)",
    )

    parser.add_argument(
        "--list-devices", action="store_true", help="List connected devices and exit"
    )

    parser.add_argument(
        "--enable-tcpip",
        type=int,
        nargs="?",
        const=5555,
        metavar="PORT",
        help="Enable TCP/IP debugging on USB device (default port: 5555)",
    )

    # iOS specific options
    parser.add_argument(
        "--wda-url",
        type=str,
        default=os.getenv("PHONE_AGENT_WDA_URL", "http://localhost:8100"),
        help="WebDriverAgent URL for iOS (default: http://localhost:8100)",
    )

    parser.add_argument(
        "--pair",
        action="store_true",
        help="Pair with iOS device (required for some operations)",
    )

    parser.add_argument(
        "--wda-status",
        action="store_true",
        help="Show WebDriverAgent status and exit (iOS only)",
    )

    # Other options
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress verbose output"
    )

    parser.add_argument(
        "--list-apps", action="store_true", help="List supported apps and exit"
    )

    parser.add_argument(
        "--list-fast-path-apps",
        action="store_true",
        help="List Fast Path v1 demo apps and exit",
    )

    parser.add_argument(
        "--lang",
        type=str,
        choices=["cn", "en"],
        default=os.getenv("PHONE_AGENT_LANG", "cn"),
        help="Language for system prompt (cn or en, default: cn)",
    )

    parser.add_argument(
        "--device-type",
        type=str,
        choices=["adb", "hdc", "ios"],
        default=os.getenv("PHONE_AGENT_DEVICE_TYPE", "adb"),
        help="Device type: adb for Android, hdc for HarmonyOS, ios for iPhone (default: adb)",
    )

    parser.add_argument(
        "task",
        nargs="?",
        type=str,
        help="Task to execute (interactive mode if not provided)",
    )

    parser.add_argument(
        "--experiment-context-file",
        type=str,
        default="",
        help="Opt-in single-request wall-clock acquisition context JSON; never starts a controller",
    )
    return parser.parse_args()


def handle_ios_device_commands(args) -> bool:
    """
    Handle iOS device-related commands.

    Returns:
        True if a device command was handled (should exit), False otherwise.
    """
    from phone_agent.xctest import XCTestConnection
    from phone_agent.xctest import list_devices as list_ios_devices

    conn = XCTestConnection(wda_url=args.wda_url)

    # Handle --list-devices
    if args.list_devices:
        devices = list_ios_devices()
        if not devices:
            print("No iOS devices connected.")
            print("\nTroubleshooting:")
            print("  1. Connect device via USB")
            print("  2. Unlock device and trust this computer")
            print("  3. Run: idevice_id -l")
        else:
            print("Connected iOS devices:")
            print("-" * 70)
            for device in devices:
                conn_type = device.connection_type.value
                model_info = f"{device.model}" if device.model else "Unknown"
                ios_info = f"iOS {device.ios_version}" if device.ios_version else ""
                name_info = device.device_name or "Unnamed"

                print(f"  ✓ {name_info}")
                print(f"    UUID: {device.device_id}")
                print(f"    Model: {model_info}")
                print(f"    OS: {ios_info}")
                print(f"    Connection: {conn_type}")
                print("-" * 70)
        return True

    # Handle --pair
    if args.pair:
        print("Pairing with iOS device...")
        success, message = conn.pair_device(args.device_id)
        print(f"{'✓' if success else '✗'} {message}")
        return True

    # Handle --wda-status
    if args.wda_status:
        print(f"Checking WebDriverAgent status at {args.wda_url}...")
        print("-" * 50)

        if conn.is_wda_ready():
            print("✓ WebDriverAgent is running")

            status = conn.get_wda_status()
            if status:
                print(f"\nStatus details:")
                value = status.get("value", {})
                print(f"  Session ID: {status.get('sessionId', 'N/A')}")
                print(f"  Build: {value.get('build', {}).get('time', 'N/A')}")

                current_app = value.get("currentApp", {})
                if current_app:
                    print(f"\nCurrent App:")
                    print(f"  Bundle ID: {current_app.get('bundleId', 'N/A')}")
                    print(f"  Process ID: {current_app.get('pid', 'N/A')}")
        else:
            print("✗ WebDriverAgent is not running")
            print("\nPlease start WebDriverAgent on your iOS device:")
            print("  1. Open WebDriverAgent.xcodeproj in Xcode")
            print("  2. Select your device")
            print("  3. Run WebDriverAgentRunner (Product > Test or Cmd+U)")
            print(f"  4. For USB: Run port forwarding: iproxy 8100 8100")

        return True

    return False


def handle_device_commands(args) -> bool:
    """
    Handle device-related commands.

    Returns:
        True if a device command was handled (should exit), False otherwise.
    """
    device_type = (
        DeviceType.ADB
        if args.device_type == "adb"
        else (DeviceType.HDC if args.device_type == "hdc" else DeviceType.IOS)
    )

    # Handle iOS-specific commands
    if device_type == DeviceType.IOS:
        return handle_ios_device_commands(args)

    device_factory = get_device_factory()
    ConnectionClass = device_factory.get_connection_class()
    conn = ConnectionClass()

    # Handle --list-devices
    if args.list_devices:
        devices = device_factory.list_devices()
        if not devices:
            print("No devices connected.")
        else:
            print("Connected devices:")
            print("-" * 60)
            for device in devices:
                status_icon = "✓" if device.status == "device" else "✗"
                conn_type = device.connection_type.value
                model_info = f" ({device.model})" if device.model else ""
                print(
                    f"  {status_icon} {device.device_id:<30} [{conn_type}]{model_info}"
                )
        return True

    # Handle --connect
    if args.connect:
        print(f"Connecting to {args.connect}...")
        success, message = conn.connect(args.connect)
        print(f"{'✓' if success else '✗'} {message}")
        if success:
            # Set as default device
            args.device_id = args.connect
        return not success  # Continue if connection succeeded

    # Handle --disconnect
    if args.disconnect:
        if args.disconnect == "all":
            print("Disconnecting all remote devices...")
            success, message = conn.disconnect()
        else:
            print(f"Disconnecting from {args.disconnect}...")
            success, message = conn.disconnect(args.disconnect)
        print(f"{'✓' if success else '✗'} {message}")
        return True

    # Handle --enable-tcpip
    if args.enable_tcpip:
        port = args.enable_tcpip
        print(f"Enabling TCP/IP debugging on port {port}...")

        success, message = conn.enable_tcpip(port, args.device_id)
        print(f"{'✓' if success else '✗'} {message}")

        if success:
            # Try to get device IP
            ip = conn.get_device_ip(args.device_id)
            if ip:
                print(f"\nYou can now connect remotely using:")
                print(f"  python main.py --connect {ip}:{port}")
                print(f"\nOr via ADB directly:")
                print(f"  adb connect {ip}:{port}")
            else:
                print("\nCould not determine device IP. Check device WiFi settings.")
        return True

    return False


def _main(args, experiment_runtime=None):
    """Run the existing CLI; optional evidence is supplied by main()."""
    try:
        expected_success = parse_optional_bool(args.experiment_expected_success)
    except ValueError as exc:
        print(f"Invalid --experiment-expected-success: {exc}")
        sys.exit(2)
    try:
        completion_metadata = build_experiment_completion_metadata(args)
    except ValueError as exc:
        print(f"Invalid experiment completion metadata: {exc}")
        sys.exit(2)

    routing_mode = args.routing_mode
    if args.disable_fast_path:
        routing_mode = "cloud_only"

    experiment_logger = configure_experiment_logger(
        ExperimentConfig(
            enabled=args.enable_experiment_logging,
            log_dir=Path(args.experiment_log_dir),
            run_id=args.experiment_run_id,
            experiment_group=args.experiment_group,
            variant=args.experiment_variant,
            expected_route=args.experiment_expected_route,
            expected_success=expected_success,
            expected_outcome=args.experiment_expected_outcome,
            repeat_index=args.experiment_repeat_index,
            extra={
                "device_type": args.device_type,
                "routing_mode": routing_mode,
                "fast_path_disabled": args.disable_fast_path,
                "screenshot_prewarm_enabled": not args.disable_screenshot_prewarm,
                "fast_path_router_service_url": args.fast_path_router_service_url,
                "base_url": args.base_url,
                "model": args.model,
                "deadline_policy_enabled": args.enable_deadline_policy,
                "fast_path_deadline_ms": args.fast_path_deadline_ms,
                "slow_path_deadline_ms": args.slow_path_deadline_ms,
                "reject_deadline_ms": args.reject_deadline_ms,
                "runtime_degraded_mode": args.runtime_degraded_mode,
                "degraded_slow_path_admission": args.degraded_slow_path_admission,
                "network_profile": args.network_profile,
                "cloud_delay_ms": args.cloud_delay_ms,
                "cloud_fail_mode": args.cloud_fail_mode,
                "runtime_monitor_enabled": args.enable_runtime_monitor,
                "runtime_health_file": args.runtime_health_file,
                "cloud_probe_deadline_ms": args.cloud_probe_deadline_ms,
                "runtime_degraded_cooldown_seconds": args.runtime_degraded_cooldown_seconds,
                "runtime_degraded_detector": args.runtime_degraded_detector,
                "runtime_window_size": args.runtime_window_size,
                "runtime_window_min_samples": args.runtime_window_min_samples,
                "runtime_window_percentile": args.runtime_window_percentile,
                "runtime_window_enter_ms": args.runtime_window_enter_ms,
                "runtime_window_exit_ms": args.runtime_window_exit_ms,
                "runtime_half_open_probe_interval_seconds": args.runtime_half_open_probe_interval_seconds,
                "cloud_delay_profile": args.cloud_delay_profile,
            },
        )
    )
    runtime_health_monitor = RuntimeHealthMonitor(
        RuntimeHealthConfig(
            enabled=args.enable_runtime_monitor,
            health_file=Path(args.runtime_health_file),
            cloud_probe_deadline_ms=args.cloud_probe_deadline_ms,
            degraded_cooldown_seconds=args.runtime_degraded_cooldown_seconds,
            detector=args.runtime_degraded_detector,
            window_size=args.runtime_window_size,
            window_min_samples=args.runtime_window_min_samples,
            window_percentile=args.runtime_window_percentile,
            window_enter_ms=args.runtime_window_enter_ms,
            window_exit_ms=args.runtime_window_exit_ms,
            half_open_probe_interval_seconds=args.runtime_half_open_probe_interval_seconds,
        ),
        event_sink=experiment_runtime.health_event if experiment_runtime is not None else None,
    )

    half_open_pending = False

    def record_route(payload):
        nonlocal half_open_pending
        if experiment_runtime is not None:
            half_open_pending = half_open_pending or bool(payload.get("half_open_probe"))
            experiment_runtime.record_route(payload)
        experiment_logger.record_route(payload)

    def make_runtime_policy_config(
        *, half_open_probe_allowed: bool = False
    ) -> RuntimePolicyConfig:
        return RuntimePolicyConfig(
            enabled=args.enable_deadline_policy,
            fast_path_deadline_ms=args.fast_path_deadline_ms,
            slow_path_deadline_ms=args.slow_path_deadline_ms,
            reject_deadline_ms=args.reject_deadline_ms,
            runtime_degraded_mode=args.runtime_degraded_mode,
            degraded_slow_path_admission=args.degraded_slow_path_admission,
            network_profile=args.network_profile,
            cloud_delay_ms=args.cloud_delay_ms,
            cloud_fail_mode=args.cloud_fail_mode,
            runtime_monitor_enabled=args.enable_runtime_monitor,
            runtime_monitor_degraded=runtime_health_monitor.is_degraded(),
            runtime_degraded_until=runtime_health_monitor.degraded_until(),
            half_open_probe_allowed=half_open_probe_allowed,
        )

    def claim_half_open_probe_for_route(
        predicted_route: str, *, will_execute_slow_path: bool
    ) -> bool:
        if (
            predicted_route != ROUTE_SLOW
            or not will_execute_slow_path
            or args.degraded_slow_path_admission != ADMISSION_REJECT
        ):
            return False
        return runtime_health_monitor.claim_half_open_probe()

    def record_half_open_probe_result(
        admission, *, success: bool, e2e_latency_ms: float
    ) -> None:
        nonlocal half_open_pending
        if getattr(admission, "half_open_probe", False):
            runtime_health_monitor.record_half_open_probe_result(
                success=success,
                e2e_latency_ms=e2e_latency_ms,
                deadline_ms=args.slow_path_deadline_ms,
            )
            half_open_pending = False

    if args.enable_experiment_logging:
        print(f"Experiment logging enabled: {experiment_logger.run_dir}")

    if args.start_fast_path_router_service:
        start_fast_path_router_service(args)
        return

    # Set device type globally based on args
    if args.device_type == "adb":
        device_type = DeviceType.ADB
    elif args.device_type == "hdc":
        device_type = DeviceType.HDC
    else:  # ios
        device_type = DeviceType.IOS

    # Set device type globally for non-iOS devices
    if device_type != DeviceType.IOS:
        set_device_type(device_type)

    # Enable HDC verbose mode if using HDC
    if device_type == DeviceType.HDC:
        from phone_agent.hdc import set_hdc_verbose

        set_hdc_verbose(True)

    # Handle --list-apps (no system check needed)
    if args.list_apps:
        if device_type == DeviceType.HDC:
            from phone_agent.config.apps_harmonyos import (
                list_supported_apps as list_harmonyos_apps,
            )

            print("Supported HarmonyOS apps:")
            apps = list_harmonyos_apps()
        elif device_type == DeviceType.IOS:
            from phone_agent.config.apps_ios import list_supported_apps as list_ios_apps

            print("Supported iOS apps:")
            print("\nNote: For iOS apps, Bundle IDs are configured in:")
            print("  phone_agent/config/apps_ios.py")
            print("\nCurrently configured apps:")
            apps = list_ios_apps()
        else:
            from phone_agent.config.apps import list_supported_apps

            print("Supported Android apps:")
            apps = list_supported_apps()

        for app in sorted(apps):
            print(f"  - {app}")

        if device_type == DeviceType.IOS:
            print(
                "\nTo add iOS apps, find the Bundle ID and add to APP_PACKAGES_IOS dictionary."
            )
        return

    if args.list_fast_path_apps:
        if device_type != DeviceType.ADB:
            print("Fast Path v1 demo apps are currently defined for Android/ADB.")
        print("Fast Path v1 demo apps:")
        for app in list_fast_path_apps():
            print(f"  - {app}")
        return

    # Handle device commands (these may need partial system checks)
    if handle_device_commands(args):
        return

    fast_path_router = create_fast_path_router(
        enable_fast_path=routing_mode != "cloud_only",
        device_type=device_type,
        base_model_path=args.fast_path_base_model_path,
        model_a_path=args.fast_path_model_a_path,
        router_service_url=args.fast_path_router_service_url,
    )

    fast_path_env_ready = False
    slow_path_env_ready = False
    slow_agent = None
    agent_config = None
    model_config = None
    slow_header_printed = False

    def ensure_fast_path_environment() -> bool:
        nonlocal fast_path_env_ready
        if fast_path_env_ready:
            return True
        fast_path_env_ready = check_system_requirements(
            device_type,
            wda_url=args.wda_url if device_type == DeviceType.IOS else "http://localhost:8100",
            require_keyboard=False,
            device_id=args.device_id,
        )
        return fast_path_env_ready

    def ensure_slow_path_environment() -> bool:
        nonlocal slow_path_env_ready
        if slow_path_env_ready:
            return True
        if not check_system_requirements(
            device_type,
            wda_url=args.wda_url if device_type == DeviceType.IOS else "http://localhost:8100",
            require_keyboard=False,
            device_id=args.device_id,
        ):
            return False
        cloud_probe_start = time.perf_counter()
        if experiment_runtime is None:
            model_api_ok = check_model_api(args.base_url, args.model, args.apikey)
            cloud_probe_ms = (time.perf_counter() - cloud_probe_start) * 1000
        else:
            call = experiment_runtime.begin_call("availability_probe")
            model_api_ok = False
            error = None
            try:
                # Reuse the existing unavailable seam, with no model invocation
                # on an injected failure. The 8s deadline remains scoring only.
                fault_config = RuntimePolicyConfig(cloud_fail_mode=call["fault_applied"])
                failure = apply_slow_path_network_degradation(fault_config, delay_ms=0.0)
                if failure:
                    error = failure
                else:
                    call["model_invoked"] = True
                    model_api_ok = check_model_api(args.base_url, args.model, args.apikey)
                    if not model_api_ok:
                        error = "check_model_api_returned_false; details in native stdout"
            except BaseException as exc:
                error = type(exc).__name__ + ": " + str(exc)
                raise
            finally:
                cloud_probe_ms = experiment_runtime.end_call(call, success=model_api_ok, error=error)
        runtime_health_monitor.record_cloud_probe(
            available=model_api_ok,
            latency_ms=cloud_probe_ms,
        )
        if not model_api_ok:
            return False
        slow_path_env_ready = True
        return True

    def ensure_slow_agent():
        nonlocal slow_agent, agent_config, model_config, slow_header_printed
        if slow_agent is not None:
            return slow_agent
        if not ensure_slow_path_environment():
            return None
        slow_agent, agent_config, model_config = create_agent(
            args, device_type, runtime_health_monitor=runtime_health_monitor
        )
        if not slow_header_printed:
            print_agent_header(args, device_type, agent_config, model_config)
            slow_header_printed = True
        return slow_agent

    def finish_task(**kwargs: object) -> None:
        if experiment_runtime is not None:
            runtime_health_monitor.record_snapshot("task_terminal")
            experiment_runtime.pending_terminal = dict(
                success=bool(kwargs.get("success")), status="finished",
                step_count=kwargs.get("step_count"),
                user_action_required=bool(kwargs.get("user_action_required", False)),
                last_step_success=getattr(slow_agent, "last_step_success", None),
            )
        experiment_logger.finish_task(
            **kwargs,
            **completion_metadata,
        )

    def run_task_impl(task: str) -> str | None:
        task_start_perf = time.perf_counter()
        experiment_logger.start_task(task, args.experiment_task_id)
        slow_path_admission = None
        fast_path_decision_recorded = False
        if fast_path_router is not None and device_type == DeviceType.ADB:
            route_start = time.perf_counter()
            decision = fast_path_router.decide(task)
            route_latency_ms = (time.perf_counter() - route_start) * 1000
            print(format_fast_path_decision(decision))

            if decision.kind == FastPathDecisionKind.FAST_LAUNCH:
                predicted_route = ROUTE_FAST
            elif decision.kind == FastPathDecisionKind.UNSUPPORTED_OR_CLARIFY:
                predicted_route = ROUTE_REJECT
            else:
                predicted_route = ROUTE_SLOW

            half_open_probe_allowed = claim_half_open_probe_for_route(
                predicted_route,
                will_execute_slow_path=(
                    decision.kind == FastPathDecisionKind.FALLBACK_SLOW
                    and routing_mode != "device_only"
                ),
            )
            admission = decide_runtime_admission(
                predicted_route,
                make_runtime_policy_config(
                    half_open_probe_allowed=half_open_probe_allowed
                ),
            )
            record_route(
                {
                    "predicted_route": predicted_route,
                    "routing_mode": routing_mode,
                    "route_label": decision.route_label,
                    "route_correct": (
                        args.experiment_expected_route == predicted_route
                        if args.experiment_expected_route
                        else None
                    ),
                    "route_latency_ms": round(route_latency_ms, 3),
                    "decision_kind": decision.kind.value,
                    "reason": decision.reason,
                    "app_name": decision.app_name,
                    "raw_prediction": decision.raw_prediction,
                    "classifier_label": decision.classifier_label,
                    "classifier_outcome": decision.classifier_outcome,
                    "parser_status": decision.parser_status,
                    "guard_outcome": decision.guard_outcome,
                    "final_route": decision.final_route,
                    "matched_app_count": decision.matched_app_count,
                    "launch_intent_count": decision.launch_intent_count,
                    "message": decision.message,
                    **admission.to_log_payload(),
                }
            )
            fast_path_decision_recorded = True

            if admission.admission_decision == ADMISSION_REJECT:
                result_message = (
                    "Slow Path task rejected by degraded-mode admission control."
                )
                finish_task(
                    success=False,
                    result_text=result_message,
                    step_count=0,
                )
                return result_message

            if decision.kind == FastPathDecisionKind.FALLBACK_SLOW:
                if routing_mode == "device_only":
                    result_message = (
                        "Device-only mode: task requires Slow Path and is "
                        "unsupported on device (no cloud fallback)."
                    )
                    record_route(
                        {
                            "predicted_route": predicted_route,
                            "routing_mode": routing_mode,
                            "route_label": decision.route_label,
                            "route_latency_ms": round(route_latency_ms, 3),
                            "decision_kind": "device_unsupported",
                            "reason": "device_only_no_cloud_fallback",
                            "raw_prediction": decision.raw_prediction,
                            **admission.to_log_payload(),
                        }
                    )
                    finish_task(
                        success=False,
                        result_text=result_message,
                        step_count=0,
                    )
                    return result_message
                if decision.message and not args.quiet:
                    print(decision.message)
                slow_path_admission = admission
            elif decision.kind == FastPathDecisionKind.UNSUPPORTED_OR_CLARIFY:
                result_message = decision.message or "Task is unsupported or needs clarification."
                finish_task(
                    success=True,
                    result_text=result_message,
                    step_count=0,
                )
                return result_message
            else:
                if not ensure_fast_path_environment():
                    result_message = "Fast Path system check failed."
                    finish_task(
                        success=False,
                        result_text=result_message,
                        step_count=0,
                    )
                    return result_message
                experiment_logger.mark_first_action("Launch")
                fast_action_start = time.perf_counter()
                success, message = execute_fast_path_decision(decision, args.device_id)
                fast_action_ms = (time.perf_counter() - fast_action_start) * 1000
                experiment_logger.record_step(
                    {
                        "step_index": 1,
                        "is_fast_path_step": True,
                        "action_name": "Launch",
                        "app_name": decision.app_name,
                        "action_execute_ms": round(fast_action_ms, 3),
                        "step_total_ms": round(route_latency_ms + fast_action_ms, 3),
                        "finished": True,
                        "success": success,
                        "message": message,
                    }
                )
                finish_task(
                    success=success,
                    result_text=message,
                    step_count=1,
                )
                return message

        if slow_path_admission is None:
            half_open_probe_allowed = claim_half_open_probe_for_route(
                ROUTE_SLOW, will_execute_slow_path=True
            )
            admission = decide_runtime_admission(
                ROUTE_SLOW,
                make_runtime_policy_config(
                    half_open_probe_allowed=half_open_probe_allowed
                ),
            )
        else:
            admission = slow_path_admission
        if not fast_path_decision_recorded:
            record_route(
                {
                    "predicted_route": ROUTE_SLOW,
                    "routing_mode": routing_mode,
                    "route_label": None,
                    "route_correct": (
                        args.experiment_expected_route == ROUTE_SLOW
                        if args.experiment_expected_route
                        else None
                    ),
                    "route_latency_ms": 0.0,
                    "decision_kind": "slow_path_only",
                    "reason": "fast_path_disabled_or_unavailable",
                    "classifier_label": None,
                    "classifier_outcome": "router_disabled",
                    "parser_status": "not_run",
                    "guard_outcome": "not_evaluated",
                    "final_route": ROUTE_SLOW,
                    "app_name": None,
                    "raw_prediction": None,
                    **admission.to_log_payload(),
                }
            )
        if admission.admission_decision == ADMISSION_REJECT:
            result_message = "Slow Path task rejected by degraded-mode admission control."
            finish_task(
                success=False,
                result_text=result_message,
                step_count=0,
            )
            return result_message
        runtime_policy_config = make_runtime_policy_config(
            half_open_probe_allowed=admission.half_open_probe
        )
        # In sliding_window mode the delay is injected per-step inside the agent
        # (see create_agent step_cloud_delay_provider), so skip the task-level
        # injection here to avoid double-injecting and double-advancing cursor.
        if args.runtime_degraded_detector == "sliding_window":
            current_cloud_delay_ms = 0.0
            cloud_failure = apply_slow_path_network_degradation(
                runtime_policy_config, delay_ms=0.0
            )
        else:
            current_cloud_delay_ms = resolve_cloud_delay_ms(args)
            cloud_failure = apply_slow_path_network_degradation(
                runtime_policy_config, delay_ms=current_cloud_delay_ms
            )
        if cloud_failure:
            e2e_latency_ms = (time.perf_counter() - task_start_perf) * 1000
            runtime_health_monitor.record_cloud_probe(
                available=False,
                latency_ms=current_cloud_delay_ms,
            )
            runtime_health_monitor.record_slow_path_result(
                success=False,
                e2e_latency_ms=e2e_latency_ms,
                deadline_ms=args.slow_path_deadline_ms,
            )
            record_half_open_probe_result(
                admission, success=False, e2e_latency_ms=e2e_latency_ms
            )
            finish_task(
                success=False,
                result_text=cloud_failure,
                step_count=0,
            )
            return cloud_failure
        agent = ensure_slow_agent()
        if agent is None:
            e2e_latency_ms = (time.perf_counter() - task_start_perf) * 1000
            runtime_health_monitor.record_slow_path_result(
                success=False,
                e2e_latency_ms=e2e_latency_ms,
                deadline_ms=args.slow_path_deadline_ms,
            )
            record_half_open_probe_result(
                admission, success=False, e2e_latency_ms=e2e_latency_ms
            )
            finish_task(
                success=False,
                result_text="Failed to initialize the required execution path.",
                step_count=0,
            )
            return None
        result = agent.run(task)
        slow_success = (
            not str(result).startswith("Model error:") and result != "Max steps reached"
        )
        user_action_required = bool(getattr(agent, "user_action_required", False))
        e2e_latency_ms = (time.perf_counter() - task_start_perf) * 1000
        runtime_health_monitor.record_slow_path_result(
            success=slow_success,
            e2e_latency_ms=e2e_latency_ms,
            deadline_ms=args.slow_path_deadline_ms,
        )
        record_half_open_probe_result(
            admission, success=slow_success, e2e_latency_ms=e2e_latency_ms
        )
        finish_task(
            success=slow_success and not user_action_required,
            result_text=result,
            step_count=getattr(agent, "step_count", None),
            user_action_required=user_action_required,
        )
        agent.reset()
        return result

    def run_task(task: str) -> str | None:
        if experiment_runtime is None:
            return run_task_impl(task)
        experiment_runtime.start_task()
        try:
            result = run_task_impl(task)
            runtime_health_monitor.record_snapshot("native_return")
            experiment_runtime.terminal(**experiment_runtime.pending_terminal)
            return result
        except BaseException as exc:
            # Native Python failures and SIGINT are recorded and propagated.
            # SIGKILL cannot run this block; the controller owns hard-cap logs.
            elapsed_ms = (time.monotonic() - experiment_runtime.native_start) * 1000
            if half_open_pending:
                experiment_runtime.emit(
                    "halfopen_result", success=False, e2e_latency_ms=elapsed_ms,
                    error=type(exc).__name__ + ": " + str(exc),
                    policy_result_recorded=False, cause="exception_before_native_result",
                )
            runtime_health_monitor.record_snapshot("task_exception")
            experiment_runtime.terminal(
                success=False, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "exception",
                error=type(exc).__name__ + ": " + str(exc),
                step_count=getattr(slow_agent, "step_count", None),
            )
            raise

    # Run with provided task or enter interactive mode
    if args.task:
        print(f"\nTask: {args.task}\n")
        result = run_task(args.task)
        if result is None:
            sys.exit(1)
        print(f"\nResult: {result}")
        return

    print("\nEntering interactive mode. Type 'quit' to exit.\n")

    while True:
        try:
            task = input("Enter your task: ").strip()

            if task.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if not task:
                continue

            print()
            result = run_task(task)
            if result is None:
                print("\nResult: Failed to initialize the required execution path.\n")
                continue
            print(f"\nResult: {result}\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Validate opt-in context before any device/router/model initialization."""
    args = parse_args()
    runtime = None
    if args.experiment_context_file:
        # Keep platform-specific file locking an opt-in dependency.
        from phone_agent.experiment_runtime import ExperimentRuntime

        runtime = ExperimentRuntime(args.experiment_context_file, args)
        args._experiment_runtime = runtime
        runtime.emit("context_validated", preparation_only=False,
                     execution_authorization="external_controller_responsibility")
    try:
        return _main(args, runtime)
    except BaseException as exc:
        if runtime is not None:
            runtime.terminal(success=False, status="bootstrap_or_runtime_exception",
                             error=type(exc).__name__ + ": " + str(exc))
        raise


if __name__ == "__main__":
    main()
