"""Unified hardware detection orchestrator."""

from __future__ import annotations

import logging
import platform

from whichvlm.hardware.amd import detect_amd_gpus
from whichvlm.hardware.apple import detect_apple_gpu, detect_apple_gpu_linux
from whichvlm.hardware.cpu import detect_avx_support, detect_cpu_cores, detect_cpu_name
from whichvlm.hardware.intel import detect_intel_gpus
from whichvlm.hardware.memory import detect_disk_free_bytes, detect_ram_bytes
from whichvlm.hardware.nvidia import detect_nvidia_gpus
from whichvlm.hardware.types import BackendCapability, HardwareInfo, ensure_backend_capabilities
from whichvlm.hardware.windows import detect_windows_gpus

logger = logging.getLogger(__name__)


def detect_hardware() -> HardwareInfo:
    """Detect all hardware. Each detector is fail-safe (returns empty on error)."""
    os_name = platform.system().lower()
    if os_name not in ("linux", "darwin", "windows"):
        os_name = "linux"

    # GPU detection
    gpus = []
    gpus.extend(detect_nvidia_gpus())
    if os_name == "linux":
        gpus.extend(detect_amd_gpus())
        gpus.extend(detect_intel_gpus())
        gpus.extend(detect_apple_gpu_linux())
    if os_name == "darwin":
        gpus.extend(detect_apple_gpu())
    if os_name == "windows":
        gpus.extend(detect_windows_gpus())
    for gpu in gpus:
        ensure_backend_capabilities(gpu, os_name)

    # CPU
    cpu_name = detect_cpu_name()
    cpu_cores = detect_cpu_cores()
    has_avx2, has_avx512 = detect_avx_support()

    # Memory
    ram_bytes = detect_ram_bytes()
    disk_free = detect_disk_free_bytes()

    return HardwareInfo(
        gpus=gpus,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        has_avx2=has_avx2,
        has_avx512=has_avx512,
        ram_bytes=ram_bytes,
        disk_free_bytes=disk_free,
        os=os_name,
        backend_capabilities=[BackendCapability("cpu", True)],
    )
