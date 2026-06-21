"""Plan-command Rich output."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from whichvlm.constants import (
    GPU_BANDWIDTH,
    QUANT_BYTES_PER_WEIGHT,
    QUANT_QUALITY_PENALTY,
    BYTES_PER_GIB,
)
from whichvlm.engine.performance import estimate_tok_per_sec
from whichvlm.engine.vram import estimate_vram
from whichvlm.hardware.types import GPUInfo
from whichvlm.models.types import GGUFVariant, ModelInfo
from whichvlm.output import _console
from whichvlm.output.formatting import format_bytes, format_params

PLAN_GPUS: tuple[tuple[str, int], ...] = (
    ("RTX 4060", 8),
    ("RTX 3060", 12),
    ("RTX 4070", 12),
    ("RTX 4080", 16),
    ("RTX 4090", 24),
    ("RX 7900 XTX", 24),
    ("RTX 5090", 32),
    ("A100 40GB", 40),
    ("L40S", 48),
    ("A100 80GB", 80),
    ("H100", 80),
    ("H200", 141),
)

PLAN_QUANTS = ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16")


def plan_variant_for_quant(model: ModelInfo, quant: str) -> GGUFVariant:
    bpw = QUANT_BYTES_PER_WEIGHT.get(quant.upper(), 0.5625)
    return GGUFVariant(
        filename="",
        quant_type=quant,
        file_size_bytes=int(model.parameter_count * bpw),
    )


def plan_vram_by_quant(model: ModelInfo, context_length: int) -> dict[str, dict]:
    rows = {}
    for quant in PLAN_QUANTS:
        if quant not in QUANT_BYTES_PER_WEIGHT:
            continue
        vram_bytes = estimate_vram(
            model, plan_variant_for_quant(model, quant), context_length
        )
        rows[quant] = {
            "vram_bytes": vram_bytes,
            "quality_loss": QUANT_QUALITY_PENALTY.get(quant, 0.0),
        }
    return rows


def plan_target_vram(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
    vram_by_quant: dict[str, dict] | None = None,
) -> int:
    vram_by_quant = vram_by_quant or plan_vram_by_quant(model, context_length)
    existing = vram_by_quant.get(target_quant.upper())
    if existing:
        return int(existing["vram_bytes"])
    return estimate_vram(model, plan_variant_for_quant(model, target_quant), context_length)


def plan_gpu_compatibility(
    model: ModelInfo,
    target_quant: str,
    target_vram: int,
) -> list[dict]:
    variant = plan_variant_for_quant(model, target_quant)
    rows = []
    for gpu_name, vram_gb in PLAN_GPUS:
        vram_bytes = int(vram_gb * BYTES_PER_GIB)
        bandwidth = GPU_BANDWIDTH.get(gpu_name)
        gpu_info = GPUInfo(
            name=gpu_name,
            vendor="nvidia",
            vram_bytes=vram_bytes,
            memory_bandwidth_gbps=bandwidth,
        )
        if vram_bytes >= target_vram:
            fit_type = "full_gpu"
        elif vram_bytes >= target_vram * 0.4:
            fit_type = "partial_offload"
        else:
            fit_type = "too_small"

        speed = None
        if fit_type != "too_small" and bandwidth:
            speed = round(estimate_tok_per_sec(model, variant, gpu_info, fit_type), 1)

        rows.append(
            {
                "name": gpu_name,
                "vram_gb": vram_gb,
                "fit_type": fit_type,
                "estimated_tok_per_sec": speed,
            }
        )
    return rows


def display_plan(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
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
    _console.console.print(panel)

    vram_table = Table(
        title=f"VRAM Required (context: {context_length})", show_lines=True
    )
    vram_table.add_column("Quant", style="bold", width=8)
    vram_table.add_column("VRAM", justify="right", width=10)
    vram_table.add_column("Quality Loss", justify="right", width=12)

    vram_by_quant = plan_vram_by_quant(model, context_length)
    target_vram = plan_target_vram(
        model, context_length, target_quant, vram_by_quant
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

    _console.console.print(vram_table)

    gpu_table = Table(
        title=f"GPU Compatibility ({target_quant}, {format_bytes(target_vram)} required)",
        show_lines=True,
    )
    gpu_table.add_column("GPU", style="bold", min_width=14)
    gpu_table.add_column("VRAM", justify="right", width=8)
    gpu_table.add_column("Fit", justify="center", width=12)
    gpu_table.add_column("Est. Speed", justify="right", width=10)

    min_full_gpu = None
    for row in plan_gpu_compatibility(model, target_quant, target_vram):
        gpu_name = row["name"]
        vram_gb = row["vram_gb"]
        fit_type = row["fit_type"]
        if fit_type == "full_gpu":
            fit = "[green]✓ Full GPU[/]"
            if min_full_gpu is None:
                min_full_gpu = (gpu_name, vram_gb)
        elif fit_type == "partial_offload":
            fit = "[yellow]~ Partial[/]"
        else:
            fit = "[red]✗ Too small[/]"
        speed = row["estimated_tok_per_sec"]
        speed_str = f"{speed:.1f} tok/s" if speed is not None else "—"

        gpu_table.add_row(gpu_name, f"{vram_gb} GB", fit, speed_str)

    _console.console.print(gpu_table)

    if min_full_gpu:
        _console.console.print(
            f"  [green]★[/] Minimum GPU for full offload: "
            f"[bold]{min_full_gpu[0]}[/] ({min_full_gpu[1]} GB) at {target_quant}"
        )
    else:
        _console.console.print(
            f"  [yellow]Note:[/] No single GPU can fully load this model at {target_quant}. "
            "Consider a lower quantization or multi-GPU setup."
        )
