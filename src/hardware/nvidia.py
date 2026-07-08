from __future__ import annotations

import logging
import re
import subprocess

from data.gpu import BYTES_PER_GIB, NVIDIA_COMPUTE_CAPABILITY
from hardware.gpu_db import static_bandwidth, resolve_detected_bandwidth
from hardware.types import GPUInfo

logger = logging.getLogger(__name__)

NVIDIA_UNIFIED_MEMORY_MARKERS = ("GB10", "DGX SPARK")


def lookup_compute_capability(name: str) -> tuple[int, int] | None:
    name_upper = name.upper()
    for key, cc in NVIDIA_COMPUTE_CAPABILITY.items():
        if key.upper() in name_upper:
            return cc
    return None


def lookup_bandwidth(name: str) -> float | None:
    return static_bandwidth(name)


def is_unified_memory_nvidia_gpu(name: str) -> bool:
    name_upper = name.upper()
    return any(marker in name_upper for marker in NVIDIA_UNIFIED_MEMORY_MARKERS)


def system_memory_bytes() -> int:
    from hardware.memory import detect_ram_bytes

    ram_bytes = detect_ram_bytes()
    if ram_bytes > 0:
        return ram_bytes
    return 128 * BYTES_PER_GIB


def make_nvidia_gpu(
    name: str,
    vram_bytes: int | None,
    cuda_version: str | None = None,
    mem_clock_mhz: float | None = None,
) -> GPUInfo:
    shared_memory = is_unified_memory_nvidia_gpu(name)
    if shared_memory and (vram_bytes is None or vram_bytes <= 0):
        vram_bytes = system_memory_bytes()
    elif vram_bytes is None:
        vram_bytes = 0

    return GPUInfo(
        name=name,
        vendor="nvidia",
        vram_bytes=vram_bytes,
        compute_capability=lookup_compute_capability(name),
        cuda_version=cuda_version,
        memory_bandwidth_gbps=resolve_detected_bandwidth(
            name, vram_bytes, mem_clock_mhz
        ),
        shared_memory=shared_memory,
    )


def run_smi_query(fields: str) -> str:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    )
    return result.stdout


def detect_nvidia_gpus_via_smi() -> list[GPUInfo]:
    try:
        stdout = run_smi_query("name,memory.total,clocks.max.memory")
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug(f"nvidia-smi 3-field query failed ({e}); retrying without clock")
        try:
            stdout = run_smi_query("name,memory.total")
        except (subprocess.SubprocessError, OSError) as e2:
            logger.debug(f"nvidia-smi fallback failed: {e2}")
            return []

    gpus: list[GPUInfo] = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) < 2 or not parts[0]:
            continue

        name, memory_mib_text = parts[0], parts[1]

        mem_clock_mhz: float | None = None
        if len(parts) == 3:
            clock_match = re.search(r"\d+", parts[2])
            if clock_match:
                mem_clock_mhz = float(clock_match.group(0))

        match = re.search(r"\d+", memory_mib_text)
        if not match:
            if not is_unified_memory_nvidia_gpu(name):
                logger.debug(f"Could not parse nvidia-smi memory value: {line!r}")
                continue
            gpus.append(make_nvidia_gpu(name, None, mem_clock_mhz=mem_clock_mhz))
            continue

        memory_mib = int(match.group(0))
        gpus.append(
            make_nvidia_gpu(name, memory_mib * 1024**2, mem_clock_mhz=mem_clock_mhz)
        )

    return gpus


def detect_nvidia_gpus() -> list[GPUInfo]:
    try:
        import pynvml
    except ImportError:
        logger.debug("pynvml not installed, trying nvidia-smi fallback")
        return detect_nvidia_gpus_via_smi()

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        logger.debug("NVML init failed, trying nvidia-smi fallback")
        return detect_nvidia_gpus_via_smi()

    gpus: list[GPUInfo] = []
    try:
        count = pynvml.nvmlDeviceGetCount()
        try:
            pynvml.nvmlSystemGetDriverVersion()
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        except pynvml.NVMLError:
            cuda_str = None

        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_bytes = mem_info.total
            except pynvml.NVMLError:
                if not is_unified_memory_nvidia_gpu(name):
                    raise
                logger.debug(f"NVML did not report dedicated memory for {name}")
                vram_bytes = None

            try:
                mem_clock_mhz: float | None = float(
                    pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                )
            except (pynvml.NVMLError, AttributeError) as clock_err:
                logger.debug(
                    f"max mem clock unavailable for {name} "
                    f"({clock_err}); bandwidth from name only"
                )
                mem_clock_mhz = None

            gpus.append(make_nvidia_gpu(name, vram_bytes, cuda_str, mem_clock_mhz))
    except pynvml.NVMLError as e:
        logger.debug(f"Error enumerating NVIDIA GPUs: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass

    if gpus:
        return gpus

    logger.debug("NVML returned no NVIDIA GPUs, trying nvidia-smi fallback")
    return detect_nvidia_gpus_via_smi()
