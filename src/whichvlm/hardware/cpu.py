from __future__ import annotations

import logging
import platform
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def cpu_name_from_lscpu() -> str | None:
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.strip().startswith("Model name"):
                    name = line.split(":", 1)[1].strip()
                    if name and name != "-":
                        return name
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def cpu_name_from_devicetree() -> str | None:
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


def clean_cpu_name(name: str | None) -> str | None:
    if name is None:
        return None
    cleaned = name.strip()
    if not cleaned or cleaned == "-" or cleaned.lower() == "name":
        return None
    return cleaned


def cpu_name_from_wmic() -> str | None:
    try:
        result = subprocess.run(
            ["wmic", "cpu", "get", "name"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None

    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        name = clean_cpu_name(line)
        if name:
            return name
    return None


def cpu_name_from_windows_cim() -> str | None:
    script = (
        "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name"
    )
    for executable in ("powershell", "pwsh"):
        try:
            result = subprocess.run(
                [executable, "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError):
            continue

        if result.returncode != 0:
            continue

        for line in result.stdout.splitlines():
            name = clean_cpu_name(line)
            if name:
                return name
    return None


def detect_cpu_name() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError as e:
            logger.debug(f"Failed to read /proc/cpuinfo: {e}")
        name = cpu_name_from_lscpu() or cpu_name_from_devicetree()
        if name:
            return name
    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Failed to run sysctl for CPU name: {e}")
        else:
            if result.returncode == 0:
                name = clean_cpu_name(result.stdout)
                if name:
                    return name
    elif system == "Windows":
        name = cpu_name_from_wmic() or cpu_name_from_windows_cim()
        if name:
            return name
    return "Unknown CPU"


def count_physical_cores_linux() -> int | None:
    try:
        physical_ids: set[tuple[str, str]] = set()
        current_physical = ""
        current_core = ""
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("physical id"):
                    current_physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    current_core = line.split(":", 1)[1].strip()
                    physical_ids.add((current_physical, current_core))
        if physical_ids:
            return len(physical_ids)
    except OSError:
        pass
    return None


def detect_cpu_cores() -> int:
    import psutil

    cores = psutil.cpu_count(logical=False)
    if cores:
        return cores


    if platform.system() == "Linux":
        linux_cores = count_physical_cores_linux()
        if linux_cores:
            return linux_cores

    return psutil.cpu_count(logical=True) or 1


def detect_avx_linux() -> tuple[bool, bool]:
    has_avx2 = False
    has_avx512 = False
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
            flags_line = ""
            for line in content.split("\n"):
                if line.startswith("flags"):
                    flags_line = line
                    break
            has_avx2 = "avx2" in flags_line
            has_avx512 = "avx512f" in flags_line
    except OSError:
        pass
    return has_avx2, has_avx512


def detect_avx_darwin() -> tuple[bool, bool]:
    has_avx2 = False
    has_avx512 = False
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.avx2_0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_avx2 = result.stdout.strip() == "1"
    except (subprocess.SubprocessError, OSError):
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.avx512f"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_avx512 = result.stdout.strip() == "1"
    except (subprocess.SubprocessError, OSError):
        pass
    return has_avx2, has_avx512


def detect_avx_support() -> tuple[bool, bool]:
    system = platform.system()
    if system == "Linux":
        return detect_avx_linux()
    elif system == "Darwin":
        return detect_avx_darwin()

    return True, False
