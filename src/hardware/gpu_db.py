from __future__ import annotations

import functools
import logging
import re
from typing import Any, cast

from data.gpu import BYTES_PER_GIB, GPU_BANDWIDTH, GPU_MEMORY_CLOCK_VARIANTS

logger = logging.getLogger(__name__)

TRADEMARK_RE = re.compile(r"\((?:tm|r)\)", re.IGNORECASE)
VENDOR_WORD_RE = re.compile(r"\b(?:nvidia|amd|ati|intel|corporation)\b", re.IGNORECASE)
LAPTOP_GPU_RE = re.compile(r"\blaptop gpu\b", re.IGNORECASE)
TRAILING_GRAPHICS_RE = re.compile(r"\bgraphics\s*$", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
MOBILE_MARKER_RE = re.compile(r"\b(?:laptop|mobile|max-?q)\b", re.IGNORECASE)


VRAM_NOSPACE_RE = re.compile(r"\b(\d+)GB\b", re.IGNORECASE)
BRACKET_RE = re.compile(r"\[(.+)]")


VRAM_SUFFIX_RE = re.compile(r"^\s+\d+\s*gb\b", re.IGNORECASE)
VRAM_GB_RE = re.compile(r"(\d+)\s*gb", re.IGNORECASE)


def normalize_detected_gpu_name(name: str) -> str:
    text = TRADEMARK_RE.sub("", name)
    text = VENDOR_WORD_RE.sub("", text)
    text = LAPTOP_GPU_RE.sub("Mobile", text)
    text = TRAILING_GRAPHICS_RE.sub("", text)
    text = VRAM_NOSPACE_RE.sub(r"\1 GB", text)
    return WHITESPACE_RE.sub(" ", text).strip()


SORTED_BW_KEYS = sorted(GPU_BANDWIDTH, key=len, reverse=True)


def substring_bandwidth(name: str) -> float | None:
    if not name:
        return None
    name_upper = name.upper()
    name_is_mobile = bool(MOBILE_MARKER_RE.search(name))
    for key in SORTED_BW_KEYS:
        if key.upper() in name_upper:
            if name_is_mobile and not MOBILE_MARKER_RE.search(key):
                continue
            return GPU_BANDWIDTH[key]
    return None


def static_bandwidth(name: str) -> float | None:
    if not name:
        return None
    if "/" not in name:
        return substring_bandwidth(name)
    bracket = BRACKET_RE.search(name)
    raw = bracket.group(1) if bracket else name
    for seg in raw.split("/"):
        seg = seg.strip()
        if not seg:
            continue
        bandwidth = substring_bandwidth(seg) or substring_bandwidth(f"RX {seg}")
        if bandwidth is not None:
            return bandwidth
    return None


def vram_gb(canonical_name: str) -> int | None:
    match = VRAM_GB_RE.search(canonical_name)
    return int(match.group(1)) if match else None


@functools.lru_cache(maxsize=1)
def dbgpu_index() -> tuple[object | None, dict[str, str] | None]:
    try:
        from dbgpu import GPUDatabase

        db = GPUDatabase.default()
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        logger.debug("dbgpu unavailable, using static bandwidth only: %s", exc)
        return None, None
    index: dict[str, str] = {}
    for canonical in db.names:
        index.setdefault(normalize_detected_gpu_name(canonical).lower(), canonical)
    return db, index


def dbgpu_bandwidth(name: str, vram_bytes: int | None) -> float | None:
    db, index = dbgpu_index()
    if db is None or index is None:
        return None
    query = normalize_detected_gpu_name(name).lower()
    if not query:
        return None

    if query in index:
        candidates = [index[query]]
    else:
        candidates = [
            original
            for normalized, original in index.items()
            if normalized.startswith(query + " ")
            and VRAM_SUFFIX_RE.match(normalized[len(query) :])
        ]
        if not candidates:
            return None
        if vram_bytes and len(candidates) > 1:
            target_gb = round(vram_bytes / BYTES_PER_GIB)
            same_vram = [c for c in candidates if vram_gb(c) == target_gb]
            if same_vram:
                candidates = same_vram

    bandwidths: list[float] = []
    db_lookup = cast(Any, db)
    for canonical in candidates:
        try:
            spec = db_lookup[canonical]
        except KeyError:
            continue
        bandwidth = getattr(spec, "memory_bandwidth_gb_s", None)
        if bandwidth:
            bandwidths.append(float(bandwidth))
    return min(bandwidths) if bandwidths else None


def memory_clock_variant_bandwidth(
    name: str, mem_clock_mhz: float | None
) -> float | None:
    if not name or not mem_clock_mhz or mem_clock_mhz <= 0:
        return None
    name_upper = name.upper()
    for key in sorted(GPU_MEMORY_CLOCK_VARIANTS, key=len, reverse=True):
        if key.upper() in name_upper:
            for min_clock, bandwidth in GPU_MEMORY_CLOCK_VARIANTS[key]:
                if mem_clock_mhz >= min_clock:
                    return bandwidth
    return None


def resolve_detected_bandwidth(
    name: str,
    vram_bytes: int | None = None,
    mem_clock_mhz: float | None = None,
) -> float | None:
    if not name:
        return None
    variant = memory_clock_variant_bandwidth(name, mem_clock_mhz)
    if variant is not None:
        return variant
    return static_bandwidth(name) or dbgpu_bandwidth(name, vram_bytes)
