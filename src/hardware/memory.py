from __future__ import annotations

import os
import shutil

import psutil


def detect_ram_bytes() -> int:
    return psutil.virtual_memory().total


def detect_available_ram_bytes() -> int:
    return psutil.virtual_memory().available


def estimate_usable_ram(total: int) -> int:
    BYTES_PER_GIB = 1024**3
    reserve = int(total * 0.15)
    reserve = max(4 * BYTES_PER_GIB, min(reserve, 32 * BYTES_PER_GIB))
    return max(0, total - reserve)


def effective_usable_ram(total: int, budget: int | None = None) -> int:
    usable = estimate_usable_ram(total)
    if budget is None:
        return usable
    return max(0, min(usable, budget))


def detect_disk_free_bytes(path: str | None = None) -> int:
    if path is None:
        path = os.path.expanduser("~")
    try:
        usage = shutil.disk_usage(path)
        return usage.free
    except OSError:
        return 0
