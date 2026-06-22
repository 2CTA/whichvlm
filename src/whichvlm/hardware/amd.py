from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path

from whichvlm.constants import AMD_SHARED_MEMORY_APU_MARKERS, BYTES_PER_GIB
from whichvlm.hardware.gpu_db import static_bandwidth, resolve_detected_bandwidth
from whichvlm.hardware.types import GPUInfo

# AMD probe. Reads rocm-smi first, then falls back to Linux device probes.
logger = logging.getLogger(__name__)

DISPLAY_CLASSES = (
    "vga compatible controller",
    "3d controller",
    "display controller",
)


def lookup_bandwidth(name: str) -> float | None:

    return static_bandwidth(name)


def is_shared_memory_apu(name: str) -> bool:
    name_upper = name.upper()
    return any(marker in name_upper for marker in AMD_SHARED_MEMORY_APU_MARKERS)


def normalize_apu_vram(name: str, vram_bytes: int) -> int:
    if is_shared_memory_apu(name) and vram_bytes < 2 * BYTES_PER_GIB:
        return 0
    return vram_bytes


def make_gpu(
    name: str,
    *,
    vram_bytes: int = 0,
    rocm_version: str | None = None,
) -> GPUInfo:
    shared_memory = is_shared_memory_apu(name)
    return GPUInfo(
        name=name,
        vendor="amd",
        vram_bytes=normalize_apu_vram(name, vram_bytes),
        rocm_version=rocm_version,
        memory_bandwidth_gbps=resolve_detected_bandwidth(name, vram_bytes),
        shared_memory=shared_memory,
    )


AMD_VENDOR_MARKERS = (
    "advanced micro devices",
    "amd/ati",
    "[amd]",
    "[ati]",
    "ati technologies",
)


def vendor_is_amd(vendor: str) -> bool:
    vendor_lower = vendor.lower()
    return any(marker in vendor_lower for marker in AMD_VENDOR_MARKERS)


def detect_from_lspci() -> list[str]:
    try:
        result = subprocess.run(
            ["lspci", "-mm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        logger.debug("lspci not available or timed out")
        return []

    if result.returncode != 0:
        return []

    names: list[str] = []
    seen: set[str] = set()
    for line in result.stdout.splitlines():


        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if len(tokens) < 4:
            continue
        device_class, vendor, device = tokens[1], tokens[2], tokens[3]
        if device_class.lower() not in DISPLAY_CLASSES:
            continue
        if not vendor_is_amd(vendor):
            continue
        name = device.strip() or "AMD Graphics"
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def read_int(path: Path) -> int:
    try:
        text = path.read_text().strip()
    except OSError:
        return 0
    try:
        return int(text, 0)
    except ValueError:
        return 0


def detect_from_sysfs(drm_path: Path = Path("/sys/class/drm")) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    seen: set[str] = set()
    try:
        cards = sorted(drm_path.glob("card[0-9]*"))
    except OSError:
        return []

    for card in cards:
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if vendor != "0x1002":
            continue

        name = "AMD Graphics"
        try:
            product_name = (device / "product_name").read_text().strip()
            if product_name:
                name = product_name
        except OSError:
            pass

        vram_bytes = read_int(device / "mem_info_vram_total")
        key = f"{name}:{vram_bytes}"
        if key in seen:
            continue
        seen.add(key)
        gpus.append(make_gpu(name, vram_bytes=vram_bytes))
    return gpus


def read_sysfs_amd_vram(drm_path: Path = Path("/sys/class/drm")) -> list[int]:
    result: list[int] = []
    try:
        cards = sorted(drm_path.glob("card[0-9]*"))
    except OSError:
        return []
    for card in cards:
        device = card / "device"
        try:
            vendor = (device / "vendor").read_text().strip().lower()
        except OSError:
            continue
        if vendor != "0x1002":
            continue
        result.append(read_int(device / "mem_info_vram_total"))
    return result


def detect_amd_gpus_fallback() -> list[GPUInfo]:
    # Linux fallback. Combines lspci names with sysfs VRAM when ROCm is absent.
    sysfs_gpus = detect_from_sysfs()

    if sysfs_gpus:

        has_generic = any(g.name == "AMD Graphics" for g in sysfs_gpus)
        if has_generic:
            lspci_names = detect_from_lspci()
            if lspci_names and len(lspci_names) == len(sysfs_gpus):
                return [
                    make_gpu(
                        lspci_names[i] if gpu.name == "AMD Graphics" else gpu.name,
                        vram_bytes=gpu.vram_bytes,
                    )
                    for i, gpu in enumerate(sysfs_gpus)
                ]
        return sysfs_gpus


    names = detect_from_lspci()
    if names:
        vram_list = read_sysfs_amd_vram()
        return [
            make_gpu(name, vram_bytes=vram_list[i] if i < len(vram_list) else 0)
            for i, name in enumerate(names)
        ]
    return []


def detect_amd_gpus() -> list[GPUInfo]:
    # Main AMD probe. Builds normalized GPU records from rocm-smi json.
    gpus: list[GPUInfo] = []

    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return detect_amd_gpus_fallback()
        product_data = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("rocm-smi not available or failed")
        return detect_amd_gpus_fallback()

    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return detect_amd_gpus_fallback()
        mem_data = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        logger.debug("Failed to get AMD VRAM info")
        return detect_amd_gpus_fallback()

    rocm_version = None
    try:
        result = subprocess.run(
            ["rocm-smi", "--showdriverversion", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            driver_data = json.loads(result.stdout)
            for key, val in driver_data.items():
                if isinstance(val, dict) and "Driver version" in val:
                    rocm_version = val["Driver version"]
                    break
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        pass

    for key in sorted(product_data.keys()):
        if not key.startswith("card"):
            continue
        card_info = product_data[key]


        name = (
            card_info.get("Card Series")
            or card_info.get("Card series")
            or card_info.get("Card SKU")
            or "Unknown AMD GPU"
        )

        vram_total = 0
        if key in mem_data:
            vram_str = mem_data[key].get("VRAM Total Memory (B)", "0")
            try:
                vram_total = int(vram_str)
            except (ValueError, TypeError):
                pass

        gpus.append(make_gpu(name, vram_bytes=vram_total, rocm_version=rocm_version))

    return gpus
