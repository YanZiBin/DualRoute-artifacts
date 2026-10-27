# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DualRoute authors.
# Modified by the DualRoute authors: routing/runtime integration and/or release packaging; upstream attribution retained.
# -*- coding: utf-8 -*-
"""Screenshot utilities for capturing Android device screen."""

import base64
import os
import struct
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

from PIL import Image


@dataclass
class Screenshot:
    """Represents a captured screenshot."""

    base64_data: str
    width: int
    height: int
    image_mime: str = "image/png"
    is_sensitive: bool = False
    capture_ms: float | None = None
    image_encode_ms: float | None = None
    base64_ms: float | None = None
    payload_bytes: int | None = None
    jpeg_quality: int | None = None


def get_screenshot(device_id: str | None = None, timeout: int = 10) -> Screenshot:
    """
    Capture a screenshot from the device using raw byte stream.
    
    This method uses 'adb shell screencap' (raw mode) to avoid:
    1. On-device PNG compression overhead.
    2. File I/O on the device (saving to /sdcard).
    3. File transfer overhead (adb pull).
    """
    adb_prefix = _get_adb_prefix(device_id)
    capture_start = time.perf_counter()

    try:
        # Execute adb shell screencap and pipe stdout directly
        # text=False ensures we get raw bytes, not decoded text
        cmd = adb_prefix + ["shell", "screencap"]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Read the entire output
        # NOTE: screencap raw output format:
        # Header (12 bytes): Width (4B), Height (4B), Format (4B) - all Little Endian
        # Data: RGBA pixels
        stdout, stderr = process.communicate(timeout=timeout)
        capture_ms = (time.perf_counter() - capture_start) * 1000

        if process.returncode != 0 or not stdout:
            print(f"Screenshot failed. Stderr: {stderr.decode('utf-8', errors='ignore') if stderr else 'None'}")
            return _create_fallback_screenshot(is_sensitive=True)

        # Basic validation of data length
        if len(stdout) < 12:
             print("Screenshot data too short (no header)")
             return _create_fallback_screenshot(is_sensitive=True)

        # Parse header
        width, height, pixel_format = struct.unpack("<III", stdout[:12])
        
        # The raw data starts after the 12-byte header
        raw_data = stdout[12:]
        
        # Calculate expected size: width * height * 4 (RGBA is 4 bytes per pixel)
        expected_len = width * height * 4
        
        # Some devices might send extra data or line padding, but standard 'screencap' is usually packed RGBA.
        # If data is slightly larger, we just take what we need. 
        # If smaller, something is wrong.
        if len(raw_data) < expected_len:
             # Fallback logic for some older devices that might output PNG even without -p 
             # (though rare for 'screencap', common for 'screencap -p')
             # Or if the output is corrupted.
             # Try to see if it's a PNG by checking magic bytes just in case
             if stdout.startswith(b'\x89PNG\r\n\x1a\n'):
                 img = Image.open(BytesIO(stdout))
             else:
                 print(f"Screenshot incomplete. Expected {expected_len} bytes, got {len(raw_data)}")
                 return _create_fallback_screenshot(is_sensitive=True)
        else:
             # Construct image from raw bytes
             # 'screencap' typically outputs RGBA
             try:
                img = Image.frombytes("RGBA", (width, height), raw_data)
             except Exception as e:
                print(f"Failed to decode raw image bytes: {e}")
                return _create_fallback_screenshot(is_sensitive=True)

        # --- Post-processing (Same as original) ---
        
        # Convert to RGB (JPEG doesn't support alpha channel)
        img = img.convert("RGB")
        
        # Optimize and resize logic
        # Current logic: compress quality=70
        buffered = BytesIO()
        encode_start = time.perf_counter()
        img.save(buffered, format="JPEG", quality=70, optimize=True)
        payload = buffered.getvalue()
        image_encode_ms = (time.perf_counter() - encode_start) * 1000
        base64_start = time.perf_counter()
        base64_data = base64.b64encode(payload).decode("utf-8")
        base64_ms = (time.perf_counter() - base64_start) * 1000

        return Screenshot(
            base64_data=base64_data,
            width=width,
            height=height,
            image_mime="image/jpeg",
            is_sensitive=False,
            capture_ms=round(capture_ms, 3),
            image_encode_ms=round(image_encode_ms, 3),
            base64_ms=round(base64_ms, 3),
            payload_bytes=len(payload),
            jpeg_quality=70,
        )

    except subprocess.TimeoutExpired:
        print("Screenshot timed out")
        return _create_fallback_screenshot(is_sensitive=False)
    except Exception as e:
        print(f"Screenshot error: {e}")
        return _create_fallback_screenshot(is_sensitive=False)


def _get_adb_prefix(device_id: str | None) -> list:
    """Get ADB command prefix with optional device specifier."""
    if device_id:
        return ["adb", "-s", device_id]
    return ["adb"]


def _create_fallback_screenshot(is_sensitive: bool) -> Screenshot:
    """Create a black fallback image when screenshot fails."""
    default_width, default_height = 1080, 2400

    black_img = Image.new("RGB", (default_width, default_height), color="black")
    buffered = BytesIO()
    black_img.save(buffered, format="PNG")
    base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return Screenshot(
        base64_data=base64_data,
        width=default_width,
        height=default_height,
        image_mime="image/png",
        is_sensitive=is_sensitive,
    )
