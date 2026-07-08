from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbgpu import GPUSpecification

from data.gpu import (
    AMD_SHARED_MEMORY_APU_MARKERS,
    BYTES_PER_GIB,
    GPU_BANDWIDTH,
)
from hardware.catalog import lookup_catalog_entry
from hardware.types import BackendCapability, GPUInfo

MANUFACTURER_TO_VENDOR: dict[str, str] = {
    "NVIDIA": "nvidia",
    "AMD": "amd",
    "ATI": "amd",
    "Intel": "intel",
    "Apple": "apple",
}

MANUFACTURER_PREFIXES = ["GeForce ", "Radeon ", "Arc ", "NVIDIA ", "AMD "]
COMMON_GPU_ALIASES: dict[str, list[str]] = {
    "a10080gb": [
        "NVIDIA A100 PCIe 80 GB",
        "NVIDIA A100 SXM4 80 GB",
    ],
    "h10080gb": [
        "NVIDIA H100 PCIe 80 GB",
        "NVIDIA H100 SXM5 80 GB",
    ],
}


APPLE_SILICON_CHIPS: dict[str, tuple[str, float]] = {
    "M1": ("Apple M1", 8.0),
    "M1 Pro": ("Apple M1 Pro", 16.0),
    "M1 Max": ("Apple M1 Max", 32.0),
    "M1 Ultra": ("Apple M1 Ultra", 64.0),
    "M2": ("Apple M2", 16.0),
    "M2 Pro": ("Apple M2 Pro", 16.0),
    "M2 Max": ("Apple M2 Max", 32.0),
    "M2 Ultra": ("Apple M2 Ultra", 64.0),
    "M3": ("Apple M3", 16.0),
    "M3 Pro": ("Apple M3 Pro", 18.0),
    "M3 Max": ("Apple M3 Max", 36.0),
    "M3 Ultra": ("Apple M3 Ultra", 96.0),
    "M4": ("Apple M4", 16.0),
    "M4 Pro": ("Apple M4 Pro", 24.0),
    "M4 Max": ("Apple M4 Max", 36.0),
    "M4 Ultra": ("Apple M4 Ultra", 64.0),
    "M5": ("Apple M5", 16.0),
    "M5 Pro": ("Apple M5 Pro", 24.0),
    "M5 Max": ("Apple M5 Max", 36.0),
}


def lookup_apple_silicon(
    name: str,
) -> tuple[str, str, float, float] | None:
    compact = re.sub(r"\s+", "", name).lower()
    if compact.startswith("apple"):
        compact = compact.removeprefix("apple")

    for key in sorted(APPLE_SILICON_CHIPS, key=len, reverse=True):
        key_compact = re.sub(r"\s+", "", key).lower()
        if compact == key_compact:
            canonical, default_vram = APPLE_SILICON_CHIPS[key]
            bandwidth = GPU_BANDWIDTH.get(key, 100.0)
            return canonical, "apple", default_vram, bandwidth
    return None


def is_amd_shared_memory_apu(name: str) -> bool:
    name_upper = name.upper()
    return any(marker in name_upper for marker in AMD_SHARED_MEMORY_APU_MARKERS)


def lookup_static_bandwidth(name: str) -> float | None:
    name_upper = name.upper()
    for key in sorted(GPU_BANDWIDTH, key=len, reverse=True):
        if key.upper() in name_upper:
            return GPU_BANDWIDTH[key]
    return None


def normalize_gpu_name(name: str) -> str:
    name = re.sub(r"([A-Za-z])(\d)", r"\1 \2", name)

    name = re.sub(r"(\d)([A-Za-z])", r"\1 \2", name)

    return re.sub(r"\s+", " ", name).strip()


def substring_search(db, name: str):
    name_upper = name.upper()
    candidates = []
    for db_name in db.names:
        idx = db_name.upper().find(name_upper)
        if idx < 0:
            continue
        after = db_name[idx + len(name) :]

        if not after or re.match(r"^(\s+(\d|GA\d|PCIe|SXM|NVL|CNX))", after):
            candidates.append(db_name)
    if candidates:
        candidates.sort(key=len)
        return db[candidates[0]]
    return None


def lookup_dbgpu(name: str) -> GPUSpecification | None:
    from dbgpu import GPUDatabase

    db = GPUDatabase.default()

    normalized = normalize_gpu_name(name)
    compact = re.sub(r"\s+", "", normalized.lower())
    names_to_try = [name] if normalized == name else [name, normalized]
    alias_hits = COMMON_GPU_ALIASES.get(compact)
    if alias_hits:
        names_to_try.extend(alias_hits)

    for n in names_to_try:
        try:
            return db[n]
        except KeyError:
            pass

        for prefix in MANUFACTURER_PREFIXES:
            try:
                return db[prefix + n]
            except KeyError:
                pass

        result = substring_search(db, n)
        if result is not None:
            return result

    try:
        from thefuzz import fuzz, process

        results = process.extract(
            normalized, db.names, limit=3, scorer=fuzz.token_set_ratio
        )
        if results and results[0][1] >= 90:
            return db[results[0][0]]

        if results:
            last_suggestions[:] = [
                (name, score) for name, score in results if score >= 70
            ]
    except ImportError:
        pass
    return None


last_suggestions: list[tuple[str, int]] = []


def parse_synthetic_gpu_specs(values: Sequence[str] | str) -> list[str]:
    raw_values = [values] if isinstance(values, str) else list(values)
    gpu_names: list[str] = []

    for raw in raw_values:
        for part in raw.split(","):
            spec = part.strip()
            if not spec:
                raise ValueError("Empty GPU entry in --gpu.")

            count_match = re.match(r"^(\d+)\s*x\s+(.+)$", spec, re.IGNORECASE)
            if count_match:
                count = int(count_match.group(1))
                name = count_match.group(2).strip()
                if count < 1:
                    raise ValueError("GPU count must be at least 1.")
                if not name:
                    raise ValueError("GPU count shorthand requires a GPU name.")
                gpu_names.extend([name] * count)
            else:
                gpu_names.append(spec)

    if not gpu_names:
        raise ValueError("At least one GPU must be specified.")
    return gpu_names


def create_synthetic_gpus(
    values: Sequence[str] | str,
    vram_override_gb: float | None = None,
) -> list[GPUInfo]:
    names = parse_synthetic_gpu_specs(values)
    if vram_override_gb is not None and len(names) != 1:
        raise ValueError(
            "--vram currently supports exactly one simulated GPU. "
            "For multi-GPU simulation, specify known GPU names and omit --vram."
        )
    return [create_synthetic_gpu(name, vram_override_gb) for name in names]


def create_synthetic_gpu(name: str, vram_override_gb: float | None = None) -> GPUInfo:
    last_suggestions.clear()

    amd_shared_memory_apu = is_amd_shared_memory_apu(name)

    apple_hit = lookup_apple_silicon(name)
    if apple_hit is not None:
        canonical, vendor, default_vram_gb, apple_bandwidth = apple_hit
        vram_gb = vram_override_gb if vram_override_gb is not None else default_vram_gb
        return GPUInfo(
            name=f"{canonical} (simulated)",
            vendor=vendor,
            vram_bytes=int(vram_gb * BYTES_PER_GIB),
            memory_bandwidth_gbps=apple_bandwidth,
            shared_memory=True,
            backend_capabilities=[
                BackendCapability("metal", True, details="Simulated Apple Silicon"),
                BackendCapability("mps", True, details="Simulated Apple Silicon"),
                BackendCapability("mlx", True, details="Simulated Apple Silicon"),
            ],
            neural_engine_available=True,
        )

    catalog_hit = lookup_catalog_entry(name)
    if catalog_hit is not None:
        vram_gb = (
            vram_override_gb if vram_override_gb is not None else catalog_hit.vram_gb
        )
        return GPUInfo(
            name=f"{catalog_hit.name} (simulated)",
            vendor=catalog_hit.vendor,
            vram_bytes=int(vram_gb * BYTES_PER_GIB),
            compute_capability=catalog_hit.compute_capability,
            memory_bandwidth_gbps=catalog_hit.memory_bandwidth_gbps,
            shared_memory=catalog_hit.shared_memory,
            backend_capabilities=[
                BackendCapability(backend, True)
                for backend in catalog_hit.supported_backends
            ],
        )

    spec = lookup_dbgpu(name)

    if vram_override_gb is not None:
        vram_bytes = int(vram_override_gb * BYTES_PER_GIB)
    elif spec is not None and spec.memory_size_gb:
        vram_bytes = int(spec.memory_size_gb * BYTES_PER_GIB)
    else:
        msg = f"Unknown GPU '{name}'."
        if last_suggestions:
            candidates = ", ".join(name for name, score in last_suggestions)
            msg += f" Did you mean: {candidates}?"
        msg += " Use --vram to specify VRAM in GB."
        raise ValueError(msg)

    bandwidth: float | None = None
    if spec is not None and spec.memory_bandwidth_gb_s:
        bandwidth = spec.memory_bandwidth_gb_s
    if bandwidth is None:
        bandwidth = lookup_static_bandwidth(name)

    compute_cap: tuple[int, int] | None = None
    if spec is not None and spec.cuda_major_version is not None:
        compute_cap = (spec.cuda_major_version, spec.cuda_minor_version or 0)

    vendor = "nvidia"
    if spec is not None:
        vendor = MANUFACTURER_TO_VENDOR.get(spec.manufacturer, "nvidia")
    elif amd_shared_memory_apu:
        vendor = "amd"

    display_name = spec.name if spec is not None else name

    return GPUInfo(
        name=f"{display_name} (simulated)",
        vendor=vendor,
        vram_bytes=vram_bytes,
        compute_capability=compute_cap,
        memory_bandwidth_gbps=bandwidth,
        shared_memory=amd_shared_memory_apu,
    )
