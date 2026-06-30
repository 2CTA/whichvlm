import json
from io import StringIO

from rich.console import Console

import whichvlm.output.console as console_mod
from whichvlm.constants import BYTES_PER_GIB
from whichvlm.hardware.catalog import (
    HARDWARE_CATALOG,
    PLAN_SYSTEM_RAM_BYTES,
    lookup_catalog_entry,
)
from whichvlm.hardware.gpu_simulator import create_synthetic_gpu
from whichvlm.hardware.types import BackendCapability, GPUInfo, HardwareInfo
from whichvlm.models.types import ModelInfo
from whichvlm.output.display import display_plan_json
from whichvlm.output.plan import (
    plan_row_for_hardware,
    plan_gpu_compatibility,
    plan_multi_gpu_compatibility,
    plan_recommendations,
    plan_target_vram,
    plan_vision_workload,
    plan_vram_by_quant,
)


def planning_model(
    params: int = 70_000_000_000, context_length: int = 32768
) -> ModelInfo:
    return ModelInfo(
        id="org/Test-VL-GGUF",
        family_id="test-vl",
        name="Test VL",
        parameter_count=params,
        architecture="qwen2_vl",
        context_length=context_length,
        hf_pipeline_tag="image-text-to-text",
    )


def test_plan_partial_offload_uses_ram_not_vram_ratio():
    model = planning_model()
    vram_by_quant = plan_vram_by_quant(model, 4096)
    target_vram = plan_target_vram(model, 4096, "Q4_K_M", vram_by_quant)

    rows = plan_gpu_compatibility(model, "Q4_K_M")
    rtx4060 = next(row for row in rows if row["name"] == "RTX 4060")

    assert rtx4060["fit_type"] == "partial_offload"
    assert rtx4060["usable_vram_bytes"] < target_vram * 0.4
    assert rtx4060["practical_partial_offload"] is False
    assert rtx4060["system_ram_bytes"] == PLAN_SYSTEM_RAM_BYTES
    assert rtx4060["binding_constraint"] == "VRAM"


def test_plan_target_vram_respects_workload_without_cached_rows():
    model = planning_model(params=7_000_000_000)

    no_images = plan_target_vram(
        model, 4096, "Q4_K_M", image_count=0, video_frames=0
    )
    with_images = plan_target_vram(
        model, 4096, "Q4_K_M", image_count=2, video_frames=4
    )

    assert with_images > no_images


def test_plan_reverse_lookup_returns_full_partial_and_multi_gpu():
    model = planning_model(params=120_000_000_000)
    single_gpu_rows = plan_gpu_compatibility(
        model,
        "Q4_K_M",
        system_ram_bytes=64 * BYTES_PER_GIB,
    )
    multi_gpu_rows = plan_multi_gpu_compatibility(
        model,
        "Q4_K_M",
        4096,
        1,
        448,
        0,
        64 * BYTES_PER_GIB,
        None,
    )

    recommendations = plan_recommendations(single_gpu_rows, multi_gpu_rows)

    assert recommendations["smallest_full_gpu"]["name"] == "H200"
    assert recommendations["smallest_partial_offload"]["name"] == "A100 40GB"
    assert recommendations["multi_gpu_alternatives"][0]["name"] == "2x MI210"
    assert recommendations["multi_gpu_alternatives"][0]["uses_multi_gpu"] is True
    assert recommendations["multi_gpu_alternatives"][0][
        "multi_gpu_support"
    ].startswith("practical ")


def test_plan_reverse_lookup_rejects_context_mismatch():
    model = planning_model(params=7_000_000_000, context_length=4096)
    rows = plan_gpu_compatibility(model, "Q4_K_M", context_length=32768)

    recommendations = plan_recommendations(rows, [])

    assert recommendations["smallest_full_gpu"] is None
    assert recommendations["smallest_partial_offload"] is None
    assert next(row for row in rows if row["fit_type"] == "full_gpu")[
        "binding_constraint"
    ] == "context length"


def test_plan_json_includes_workload_and_reverse_lookup():
    model = planning_model(params=7_000_000_000)
    buf = StringIO()
    original_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_plan_json(
            model,
            context_length=8192,
            target_quant="Q4_K_M",
            image_count=2,
            image_size=896,
            video_frames=4,
            system_ram_bytes=32 * BYTES_PER_GIB,
            min_speed=1.0,
        )
    finally:
        console_mod.console = original_console

    data = json.loads(buf.getvalue())
    assert data["workload"]["image_count"] == 2
    assert data["workload"]["video_frames"] == 4
    assert "reverse_lookup" in data
    assert data["gpu_compatibility"][0]["supported_backends"]
    assert data["workload"]["os"] == "linux"


def test_hardware_catalog_carries_normalized_metadata():
    rtx4070 = lookup_catalog_entry("RTX 4070")

    assert rtx4070 in HARDWARE_CATALOG
    assert rtx4070.vram_gb == 12
    assert rtx4070.memory_bandwidth_gbps is not None
    assert rtx4070.compute_capability is not None
    assert "cuda" in rtx4070.supported_backends
    assert "windows" in rtx4070.os_names


def test_plan_respects_target_os_for_backends():
    model = planning_model(params=7_000_000_000)
    rows = plan_gpu_compatibility(model, "Q4_K_M", os_name="darwin")
    rtx4070 = next(row for row in rows if row["name"] == "RTX 4070")

    assert rtx4070["os_supported"] is False
    assert rtx4070["supported_backends"] == []
    assert rtx4070["binding_constraint"] == "OS support"
    assert plan_recommendations(rows, [])["smallest_full_gpu"]["name"] == "Apple M4 Max"


def test_catalog_exposes_availability_and_shared_memory_behavior():
    m4_max = lookup_catalog_entry("Apple M4 Max")
    h100 = lookup_catalog_entry("H100")

    assert m4_max.shared_memory is True
    assert m4_max.shared_memory_behavior == "unified system memory"
    assert m4_max.availability == "new systems"
    assert h100.multi_gpu_backends == ("cuda",)
    assert h100.interconnect is not None
    assert lookup_catalog_entry("A6000").name == "RTX A6000"


def test_synthetic_gpu_uses_catalog_backends_for_hardware_to_model_path():
    gpu = create_synthetic_gpu("RTX 4070")

    assert gpu.vram_bytes == 12 * BYTES_PER_GIB
    assert gpu.compute_capability == lookup_catalog_entry("RTX 4070").compute_capability
    assert {cap.name for cap in gpu.backend_capabilities} == {"cuda", "vulkan"}


def test_plan_row_exposes_uncertainty_when_bandwidth_is_missing():
    model = planning_model(params=7_000_000_000)
    hardware = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Unknown CUDA GPU",
                vendor="nvidia",
                vram_bytes=24 * BYTES_PER_GIB,
                backend_capabilities=[BackendCapability("cuda", True)],
            )
        ],
        ram_bytes=64 * BYTES_PER_GIB,
        disk_free_bytes=1_000 * BYTES_PER_GIB,
    )

    row = plan_row_for_hardware(
        model,
        "Q4_K_M",
        hardware,
        "Unknown CUDA GPU",
        4096,
        plan_vision_workload(4096, 1, 448, 0),
        None,
        ("linux",),
    )

    assert row["binding_constraint"] == "bandwidth"
    assert any("bandwidth is unknown" in warning for warning in row["warnings"])
