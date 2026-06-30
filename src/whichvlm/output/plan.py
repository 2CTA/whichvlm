from __future__ import annotations

from copy import copy

from rich.panel import Panel
from rich.table import Table

from whichvlm.constants import (
    QUANT_BYTES_PER_WEIGHT,
    QUANT_QUALITY_PENALTY,
    BYTES_PER_GIB,
)
from whichvlm.engine.compatibility import check_compatibility
from whichvlm.engine.performance import estimate_tok_per_sec
from whichvlm.engine.vram import estimate_vram, estimate_vram_details
from whichvlm.engine.workload import VisionWorkload
from whichvlm.hardware.catalog import (
  HARDWARE_CATALOG,
  GPUInfo,
  PLAN_SYSTEM_RAM_BYTES,
  PLAN_VRAM_HEADROOM_RATIO,
  HardwareCatalogEntry,
)
from whichvlm.hardware.types import GPUInfo, HardwareInfo
from whichvlm.models.types import GGUFVariant, ModelInfo
from whichvlm.output import console
from whichvlm.output.formatting import format_bytes, format_params

PLAN_QUANTS = ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16")
PRACTICAL_PARTIAL_MAX_OFFLOAD_RATIO = 0.65
PRACTICAL_PARTIAL_MIN_USABLE_VRAM_BYTES = 6 * BYTES_PER_GIB
PRACTICAL_PARTIAL_MIN_TOK_PER_SEC = 2.0
MULTI_GPU_SPEED_FACTOR = 0.70


def plan_variant_for_quant(model: ModelInfo, quant: str) -> GGUFVariant:
    bpw = QUANT_BYTES_PER_WEIGHT.get(quant.upper(), 0.5625)
    return GGUFVariant(
        filename="",
        quant_type=quant,
        file_size_bytes=int(model.parameter_count * bpw),
    )


def plan_vision_workload(
    context_length: int,
    image_count: int,
    image_size: int,
    video_frames: int,
) -> VisionWorkload | None:
    visual_inputs = image_count + video_frames
    if visual_inputs <= 0:
        return None
    return VisionWorkload(
        image_count=visual_inputs,
        image_size=image_size,
        context_length=context_length,
    )


def plan_vram_by_quant(
    model: ModelInfo,
    context_length: int,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
) -> dict[str, dict]:
    rows = {}
    vision_workload = plan_vision_workload(
        context_length, image_count, image_size, video_frames
    )
    for quant in PLAN_QUANTS:
        if quant not in QUANT_BYTES_PER_WEIGHT:
            continue
        vram = estimate_vram_details(
            model, plan_variant_for_quant(model, quant), context_length, vision_workload
        )
        rows[quant] = {
            "vram_bytes": vram.required_bytes,
            "vram_range_bytes": [vram.lower_bytes, vram.upper_bytes],
            "vram_confidence": vram.confidence,
            "quality_loss": QUANT_QUALITY_PENALTY.get(quant, 0.0),
        }
    return rows


def plan_target_vram(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
    vram_by_quant: dict[str, dict] | None = None,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
) -> int:
    vram_by_quant = vram_by_quant or plan_vram_by_quant(
        model, context_length, image_count, image_size, video_frames
    )
    existing = vram_by_quant.get(target_quant.upper())
    if existing:
        return int(existing["vram_bytes"])
    vision_workload = plan_vision_workload(
        context_length, image_count, image_size, video_frames
    )
    return estimate_vram(
        model,
        plan_variant_for_quant(model, target_quant),
        context_length,
        vision_workload,
    )


def plan_binding_constraint(row: dict, min_speed: float | None) -> str:
    if not row["context_fits"]:
        return "context length"
    if not row["can_run"]:
        return "memory"
    if not row["os_supported"]:
        return "OS support"
    if not row["supported_backends"]:
        return "backend support"
    if min_speed is not None and not row["meets_speed"]:
        return "speed"
    if row["estimated_tok_per_sec"] is None:
        return "bandwidth"
    if row["fit_type"] == "partial_offload":
        return "VRAM"
    if row["uses_multi_gpu"]:
        return "multi-GPU split"
    return "none"


def gpu_backends(gpu: GPUInfo) -> list[str]:
    return [capability.name for capability in gpu.backend_capabilities if capability.available]


def plan_metadata_warnings(gpu: GPUInfo) -> list[str]:
    warnings = []
    if gpu.memory_bandwidth_gbps is None:
        warnings.append("Memory bandwidth is unknown; speed estimate is unavailable")
    if gpu.vendor == "nvidia" and gpu.compute_capability is None:
        warnings.append("NVIDIA compute capability is unknown")
    if not gpu.backend_capabilities:
        warnings.append("Backend support metadata is unavailable")
    return warnings


def is_practical_partial_offload(row: dict) -> bool:
    return (
        row["fit_type"] == "partial_offload"
        and row["context_fits"]
        and row["usable_vram_bytes"] >= PRACTICAL_PARTIAL_MIN_USABLE_VRAM_BYTES
        and row["offload_ratio"] <= PRACTICAL_PARTIAL_MAX_OFFLOAD_RATIO
        and row["estimated_tok_per_sec"] is not None
        and row["estimated_tok_per_sec"] >= PRACTICAL_PARTIAL_MIN_TOK_PER_SEC
        and row["os_supported"]
        and bool(row["supported_backends"])
        and row["meets_speed"]
    )


def plan_row_for_hardware(
    model: ModelInfo,
    target_quant: str,
    hardware: HardwareInfo,
    label: str,
    context_length: int,
    vision_workload: VisionWorkload | None,
    min_speed: float | None,
    os_constraints: tuple[str, ...],
    catalog_entry: HardwareCatalogEntry | None = None,
) -> dict:
    variant = plan_variant_for_quant(model, target_quant)
    result = check_compatibility(model, variant, hardware, context_length, vision_workload)
    fit_type = result.fit_type if result.can_run else "too_small"
    gpu = hardware.gpus[0]
    speed = None
    if result.can_run and gpu.memory_bandwidth_gbps:
        speed = estimate_tok_per_sec(model, variant, gpu, result.fit_type)
        if result.uses_multi_gpu:
            speed *= MULTI_GPU_SPEED_FACTOR
        speed = round(speed, 1)
    row = {
        "name": label,
        "vram_gb": round(sum(gpu.vram_bytes for gpu in hardware.gpus) / BYTES_PER_GIB),
        "usable_vram_bytes": (
            result.multi_gpu_effective_vram_bytes or result.vram_available_bytes
        ),
        "reserved_headroom_bytes": sum(
            gpu.vram_bytes - (gpu.usable_vram_bytes or gpu.vram_bytes)
            for gpu in hardware.gpus
        ),
        "system_ram_bytes": hardware.ram_bytes,
        "required_memory_bytes": result.vram_required_bytes,
        "fit_type": fit_type,
        "can_run": result.can_run,
        "context_fits": result.context_fits,
        "offload_ratio": result.offload_ratio,
        "uses_multi_gpu": result.uses_multi_gpu,
        "multi_gpu_effective_vram_bytes": result.multi_gpu_effective_vram_bytes,
        "multi_gpu_support": multi_gpu_support_label(result.uses_multi_gpu, catalog_entry),
        "estimated_tok_per_sec": speed,
        "meets_speed": min_speed is None or (speed is not None and speed >= min_speed),
        "supported_backends": gpu_backends(gpu),
        "os_constraints": list(os_constraints),
        "os_supported": hardware.os in os_constraints,
        "memory_bandwidth_gbps": gpu.memory_bandwidth_gbps,
        "compute_capability": gpu.compute_capability,
        "shared_memory": gpu.shared_memory,
        "shared_memory_behavior": (
            catalog_entry.shared_memory_behavior
            if catalog_entry
            else ("shared memory" if gpu.shared_memory else "dedicated VRAM")
        ),
        "price_usd": catalog_entry.price_usd if catalog_entry else None,
        "availability": catalog_entry.availability if catalog_entry else None,
        "interconnect": catalog_entry.interconnect if catalog_entry else None,
        "warnings": result.warnings + plan_metadata_warnings(gpu),
    }
    row["metadata_complete"] = not row["warnings"]
    row["binding_constraint"] = plan_binding_constraint(row, min_speed)
    row["practical_partial_offload"] = is_practical_partial_offload(row)
    return row


def multi_gpu_support_label(
    uses_multi_gpu: bool,
    entry: HardwareCatalogEntry | None,
) -> str:
    if not uses_multi_gpu:
        return "single GPU"
    if entry is None or not entry.multi_gpu_backends:
        return "theoretical split"
    backends = "/".join(entry.multi_gpu_backends)
    if entry.interconnect:
        return f"practical {backends} layer split over {entry.interconnect}"
    return f"practical {backends} layer split"


def hardware_size_key(row: dict) -> tuple[int, int, float]:
    price = row["price_usd"] if row["price_usd"] is not None else 10**9
    bandwidth = row["memory_bandwidth_gbps"] or 0.0
    return (row["usable_vram_bytes"], price, -bandwidth)


def plan_gpu_compatibility(
    model: ModelInfo,
    target_quant: str,
    context_length: int = 4096,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
    system_ram_bytes: int = PLAN_SYSTEM_RAM_BYTES,
    min_speed: float | None = None,
    os_name: str = "linux",
) -> list[dict]:
    vision_workload = plan_vision_workload(
        context_length, image_count, image_size, video_frames
    )
    rows = []
    for entry in HARDWARE_CATALOG:
        hardware = entry.to_hardware(system_ram_bytes, os_name)
        rows.append(
            plan_row_for_hardware(
                model,
                target_quant,
                hardware,
                entry.name,
                context_length,
                vision_workload,
                min_speed,
                entry.os_names,
                entry,
            )
        )
    return sorted(rows, key=hardware_size_key)


def multi_gpu_hardware(
    entry: HardwareCatalogEntry,
    count: int,
    system_ram_bytes: int,
    os_name: str,
) -> HardwareInfo:
    hardware = entry.to_hardware(system_ram_bytes, os_name)
    gpu = hardware.gpus[0]
    hardware.gpus = [copy(gpu) for _ in range(count)]
    return hardware


def plan_multi_gpu_compatibility(
    model: ModelInfo,
    target_quant: str,
    context_length: int,
    image_count: int,
    image_size: int,
    video_frames: int,
    system_ram_bytes: int,
    min_speed: float | None,
    os_name: str = "linux",
) -> list[dict]:
    vision_workload = plan_vision_workload(
        context_length, image_count, image_size, video_frames
    )
    rows = []
    for count in (2, 4):
        for entry in HARDWARE_CATALOG:
            if not entry.multi_gpu_backends:
                continue
            hardware = multi_gpu_hardware(entry, count, system_ram_bytes, os_name)
            rows.append(
                plan_row_for_hardware(
                    model,
                    target_quant,
                    hardware,
                    f"{count}x {entry.name}",
                    context_length,
                    vision_workload,
                    min_speed,
                    entry.os_names,
                    entry,
                )
            )
    return sorted(rows, key=hardware_size_key)


def first_runnable(rows: list[dict], fit_type: str) -> dict | None:
    for row in rows:
        if (
            row["fit_type"] == fit_type
            and row["can_run"]
            and row["context_fits"]
            and row["meets_speed"]
            and row["os_supported"]
            and row["supported_backends"]
        ):
            return row
    return None


def plan_recommendations(
    single_gpu_rows: list[dict],
    multi_gpu_rows: list[dict],
) -> dict:
    full_gpu = first_runnable(single_gpu_rows, "full_gpu")
    show_multi_gpu = full_gpu is None or full_gpu["vram_gb"] >= 80
    return {
        "smallest_full_gpu": full_gpu,
        "smallest_partial_offload": next(
            (row for row in single_gpu_rows if row["practical_partial_offload"]),
            None,
        ),
        "multi_gpu_alternatives": [
            row
            for row in multi_gpu_rows
            if show_multi_gpu
            and row["fit_type"] == "full_gpu"
            and row["can_run"]
            and row["context_fits"]
            and row["meets_speed"]
            and row["os_supported"]
            and row["supported_backends"]
            and row["multi_gpu_support"].startswith("practical ")
        ][:3],
    }


def display_plan(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
    system_ram_bytes: int = PLAN_SYSTEM_RAM_BYTES,
    min_speed: float | None = None,
    os_name: str = "linux",
) -> None:
    params = format_params(model.parameter_count)
    active = ""
    if model.is_moe and model.parameter_count_active:
        active = f" ({format_params(model.parameter_count_active)} active)"
    ctx = str(model.context_length) if model.context_length else "unknown"

    lines = [
        f"[bold cyan]Model:[/]  {model.id}",
        f"[bold cyan]Params:[/] {params}{active} | Arch: {model.architecture} | Context: {ctx}",
    ]
    if model.license:
        lines.append(f"[bold cyan]License:[/] {model.license}")
    panel = Panel("\n".join(lines), title="[bold]Model Info[/]", border_style="cyan")
    console.console.print(panel)

    vram_table = Table(
        title=f"VRAM Required (context: {context_length})", show_lines=True
    )
    vram_table.add_column("Quant", style="bold", width=8)
    vram_table.add_column("VRAM", justify="right", width=10)
    vram_table.add_column("Quality Loss", justify="right", width=12)

    vram_by_quant = plan_vram_by_quant(
        model, context_length, image_count, image_size, video_frames
    )
    target_vram = plan_target_vram(
        model,
        context_length,
        target_quant,
        vram_by_quant,
        image_count,
        image_size,
        video_frames,
    )
    for qt, row in vram_by_quant.items():
        vram_bytes = row["vram_bytes"]
        penalty = row["quality_loss"]
        penalty_str = f"-{penalty * 100:.0f}%" if penalty > 0 else "0%"
        marker = " ★" if qt.upper() == target_quant.upper() else ""
        style = "bold green" if qt.upper() == target_quant.upper() else ""
        vram_table.add_row(
            f"{qt}{marker}", format_bytes(vram_bytes), penalty_str, style=style
        )

    console.console.print(vram_table)

    gpu_rows = plan_gpu_compatibility(
        model,
        target_quant,
        context_length,
        image_count,
        image_size,
        video_frames,
        system_ram_bytes,
        min_speed,
        os_name,
    )
    multi_gpu_rows = plan_multi_gpu_compatibility(
        model,
        target_quant,
        context_length,
        image_count,
        image_size,
        video_frames,
        system_ram_bytes,
        min_speed,
        os_name,
    )
    recommendations = plan_recommendations(gpu_rows, multi_gpu_rows)

    gpu_table = Table(
        title=(
            f"GPU Compatibility ({target_quant}, {format_bytes(target_vram)} required, "
            f"{PLAN_VRAM_HEADROOM_RATIO:.0%} headroom reserved)"
        ),
        show_lines=True,
    )
    gpu_table.add_column("GPU", style="bold", min_width=14)
    gpu_table.add_column("VRAM", justify="right", width=8)
    gpu_table.add_column("Fit", justify="center", width=12)
    gpu_table.add_column("Backend", justify="left", width=12)
    gpu_table.add_column("Est. Speed", justify="right", width=10)
    gpu_table.add_column("Limit", justify="left", width=14)

    for row in gpu_rows:
        gpu_name = row["name"]
        vram_gb = row["vram_gb"]
        fit_type = row["fit_type"]
        if fit_type == "full_gpu":
            fit = "[green]✓ Full GPU[/]"
        elif fit_type == "partial_offload":
            fit = (
                "[yellow]~ Partial[/]"
                if row["practical_partial_offload"]
                else "[yellow]~ Partial (rough)[/]"
            )
        else:
            fit = "[red]✗ Too small[/]"
        speed = row["estimated_tok_per_sec"]
        speed_str = f"{speed:.1f} tok/s" if speed is not None else "—"
        backends = ", ".join(row["supported_backends"]) or "unknown"

        gpu_table.add_row(
            gpu_name,
            f"{vram_gb} GB",
            fit,
            backends,
            speed_str,
            row["binding_constraint"],
        )

    console.console.print(gpu_table)

    display_plan_recommendations(recommendations)


def recommendation_line(title: str, row: dict | None) -> str:
    if row is None:
        return f"[bold]{title}:[/] none found"
    backends = ", ".join(row["supported_backends"]) or "unknown"
    price = f", price ${row['price_usd']:,}" if row["price_usd"] else ""
    availability = f", {row['availability']}" if row["availability"] else ""
    uncertainty = "" if row["metadata_complete"] else ", uncertainty noted"
    return (
        f"[bold]{title}:[/] {row['name']} | {row['fit_type']} | "
        f"required {format_bytes(row['required_memory_bytes'])}, "
        f"usable VRAM {format_bytes(row['usable_vram_bytes'])}, "
        f"reserved {format_bytes(row['reserved_headroom_bytes'])}, "
        f"backend {backends}, OS {','.join(row['os_constraints'])}, "
        f"limit {row['binding_constraint']}{price}{availability}{uncertainty}"
    )


def display_plan_recommendations(recommendations: dict) -> None:
    lines = [
        recommendation_line(
            "Smallest full-GPU option", recommendations["smallest_full_gpu"]
        ),
        recommendation_line(
            "Smallest partial-offload option",
            recommendations["smallest_partial_offload"],
        ),
    ]
    multi = recommendations["multi_gpu_alternatives"]
    if multi:
        lines.append("[bold]Practical multi-GPU alternatives:[/]")
        lines.extend(recommendation_line("  option", row) for row in multi)
    else:
        lines.append("[bold]Practical multi-GPU alternatives:[/] none found")

    console.console.print(Panel("\n".join(lines), title="[bold]Reverse Lookup[/]"))
