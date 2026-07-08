import importlib

import pytest

from data.gpu import GPU_BANDWIDTH, NVIDIA_COMPUTE_CAPABILITY, VULKAN_ONLY_GPUS
from engine.compatibility import check_compatibility
from hardware.nvidia import (
    lookup_bandwidth,
    lookup_compute_capability,
)
from hardware.types import GPUInfo, HardwareInfo
from models.types import GGUFVariant, ModelInfo


KEPLER_GPUS: list[tuple[str, float, tuple[int, int]]] = [
    ("Quadro K6000", 288.0, (3, 5)),
    ("Quadro K5200", 192.3, (3, 5)),
    ("Quadro K4200", 173.0, (3, 0)),
    ("Quadro K2200", 80.0, (3, 0)),
    ("Quadro K620", 29.0, (3, 0)),
    ("Quadro K420", 14.4, (3, 0)),
    ("GTX 780", 288.4, (3, 5)),
    ("GTX 770", 224.3, (3, 0)),
    ("GTX 760", 192.2, (3, 0)),
]


def test_kepler_tables_cover_vulkan_only_cards():
    expected = {name for name, _, _ in KEPLER_GPUS}

    assert set(VULKAN_ONLY_GPUS) == expected
    assert isinstance(VULKAN_ONLY_GPUS, frozenset)
    for name, bandwidth, cc in KEPLER_GPUS:
        assert GPU_BANDWIDTH[name] == bandwidth
        assert NVIDIA_COMPUTE_CAPABILITY[name] == cc


def test_constants_shim_reexports_gpu_tables():
    legacy_constants = importlib.import_module("whichvlm.constants")

    assert legacy_constants.GPU_BANDWIDTH is GPU_BANDWIDTH
    assert legacy_constants.VULKAN_ONLY_GPUS is VULKAN_ONLY_GPUS


@pytest.mark.parametrize("name, bandwidth, cc", KEPLER_GPUS)
def test_nvidia_name_lookup_resolves_kepler_specs(name, bandwidth, cc):
    full_name = f"NVIDIA {name}"
    assert lookup_bandwidth(full_name) == bandwidth
    assert lookup_compute_capability(full_name) == cc


def test_k4200_lookup_not_shadowed_by_k420():
    assert lookup_bandwidth("NVIDIA Quadro K4200") == 173.0
    assert lookup_bandwidth("NVIDIA Quadro K420") == 14.4


def kepler_hardware(name: str, cc: tuple[int, int], bandwidth: float) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name=f"NVIDIA {name}",
                vendor="nvidia",
                vram_bytes=4 * 1024**3,
                compute_capability=cc,
                memory_bandwidth_gbps=bandwidth,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=8,
        ram_bytes=16 * 1024**3,
        disk_free_bytes=200 * 1024**3,
        os="linux",
    )


@pytest.mark.parametrize("name, bandwidth, cc", KEPLER_GPUS)
def test_compatibility_flags_kepler_as_vulkan_only(name, bandwidth, cc):
    model = ModelInfo(
        id="test/model",
        family_id="test/model",
        name="model",
        parameter_count=3_000_000_000,
    )
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=2_000_000_000,
    )
    hw = kepler_hardware(name, cc, bandwidth)

    result = check_compatibility(model, variant, hw)

    assert any("vulkan" in w.lower() for w in result.warnings), (
        f"{name} should be flagged as Vulkan-only"
    )


def test_compatibility_does_not_flag_modern_nvidia_as_vulkan_only():
    model = ModelInfo(
        id="test/model",
        family_id="test/model",
        name="model",
        parameter_count=3_000_000_000,
    )
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=2_000_000_000,
    )
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="NVIDIA GeForce RTX 4090",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=1008.0,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=8,
        ram_bytes=32 * 1024**3,
        disk_free_bytes=200 * 1024**3,
        os="linux",
    )

    result = check_compatibility(model, variant, hw)

    assert not any("vulkan" in w.lower() for w in result.warnings)
