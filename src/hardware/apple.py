from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from data.gpu import GPU_BANDWIDTH
from hardware.types import BackendCapability, GPUInfo

logger = logging.getLogger(__name__)


def lookup_bandwidth(chip_name: str) -> float | None:
    chip_upper = chip_name.upper()
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in chip_upper:
            return GPU_BANDWIDTH[key]
    return None


def run_system_profiler(data_type: str) -> dict | None:
    try:
        result = subprocess.run(
            ["system_profiler", data_type, "-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def find_metal_value(obj: object) -> str | None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if "metal" in str(key).lower() and isinstance(value, str):
                return value
            found = find_metal_value(value)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_metal_value(item)
            if found:
                return found
    return None


def metal_capability(display_data: dict | None) -> BackendCapability:
    metal_value = find_metal_value(display_data) if display_data else None
    if not metal_value:
        return BackendCapability(
            "metal",
            True,
            details="Assumed available for Apple Silicon; display metadata unavailable",
        )
    value_lower = metal_value.lower()
    if "unsupported" in value_lower or value_lower in {"no", "none"}:
        return BackendCapability("metal", False, details=metal_value)
    return BackendCapability("metal", True, version=metal_value, details=metal_value)


def apple_backend_capabilities(display_data: dict | None) -> list[BackendCapability]:
    return [
        metal_capability(display_data),
        BackendCapability("mps", True, details="PyTorch MPS-compatible Apple Silicon"),
        BackendCapability("mlx", True, details="MLX-ready Apple Silicon"),
    ]


def detect_apple_gpu() -> list[GPUInfo]:
    data = run_system_profiler("SPHardwareDataType")
    if data is None:
        logger.debug("system_profiler not available (not macOS)")
        return []
    display_data = run_system_profiler("SPDisplaysDataType")

    try:
        hw_items = data["SPHardwareDataType"]
        hw = hw_items[0]
        chip_name = hw.get("chip_type", "")
        if not chip_name:
            return []

        memory_str = hw.get("physical_memory", "0 GB")
        parts = memory_str.split()
        mem_value = int(parts[0])
        mem_unit = parts[1].upper() if len(parts) > 1 else "GB"
        multiplier = {"GB": 1024**3, "TB": 1024**4, "MB": 1024**2}.get(
            mem_unit, 1024**3
        )
        unified_memory = mem_value * multiplier

        return [
            GPUInfo(
                name=chip_name,
                vendor="apple",
                vram_bytes=unified_memory,
                memory_bandwidth_gbps=lookup_bandwidth(chip_name),
                shared_memory=True,
                backend_capabilities=apple_backend_capabilities(display_data),
                neural_engine_available=True,
            )
        ]
    except (KeyError, IndexError, ValueError) as e:
        logger.debug(f"Failed to parse Apple hardware info: {e}")
        return []


ASAHI_DRIVER_NAMES = ("asahi", "apple")


def chip_name_from_devicetree() -> str | None:
    try:
        raw = Path("/sys/firmware/devicetree/base/model").read_bytes()
        model = raw.decode("utf-8", errors="replace").strip().rstrip("\x00")
        if not model:
            return None
        m = re.search(r"\b(M\d+(?:\s+(?:Pro|Max|Ultra))?)\b", model)
        if m:
            return f"Apple {m.group(1)}"
        return model
    except OSError:
        return None


def detect_apple_gpu_linux(
    drm_path: Path = Path("/sys/class/drm"),
) -> list[GPUInfo]:
    try:
        cards = sorted(drm_path.glob("card[0-9]*"))
    except OSError:
        return []

    for card in cards:
        driver = card / "device" / "driver"
        try:
            driver_name = driver.resolve().name
        except OSError:
            continue
        if driver_name not in ASAHI_DRIVER_NAMES:
            continue

        chip_name = chip_name_from_devicetree() or "Apple Silicon"

        import psutil

        unified_memory = psutil.virtual_memory().total

        return [
            GPUInfo(
                name=chip_name,
                vendor="apple",
                vram_bytes=unified_memory,
                memory_bandwidth_gbps=lookup_bandwidth(chip_name),
                shared_memory=True,
                backend_capabilities=[
                    BackendCapability("vulkan", True, details="Asahi Linux driver"),
                    BackendCapability("metal", False, details="Metal requires macOS"),
                    BackendCapability("mlx", False, details="MLX requires macOS"),
                ],
            )
        ]

    return []
