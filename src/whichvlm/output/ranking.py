from __future__ import annotations

import re
from math import log10

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from whichvlm.engine.quantization import effective_quant_type
from whichvlm.engine.types import CompatibilityResult
from whichvlm.hardware.types import HardwareInfo
from whichvlm.output import console
from whichvlm.output.formatting import (
    downloads_style,
    format_bytes,
    format_downloads,
    format_params,
    format_published_at,
    format_speed,
    parse_published_at,
    published_style,
)

ACCENT = "#f472b6"
CYAN = "#67e8f9"
VIOLET = "#a78bfa"
MINT = "#5eead4"
AMBER = "#fbbf24"


def detect_specializations(model_id: str) -> list[str]:

    lower = model_id.lower()
    tags: list[str] = []
    if re.search(r"(coder|codegen|starcoder|program|coding)", lower):
        tags.append("coding")
    if re.search(r"(^|[-_/])(vl|vision|multimodal|llava|image)([-_/]|$)", lower):
        tags.append("vision")
    if re.search(r"(^|[-_/])math([-_/]|$)", lower):
        tags.append("math")
    return tags


def top_pick_confidence(results: list[CompatibilityResult]) -> tuple[str, str]:

    top = results[0]
    gap = (top.quality_score - results[1].quality_score) if len(results) > 1 else 999.0
    notes: list[str] = []
    if top.fit_type == "partial_offload":
        notes.append("partial offload")
    elif top.fit_type == "cpu_only":
        notes.append("CPU-only")
    if top.speed_confidence == "low":
        notes.append("low-confidence speed")
    risk_note = f", {', '.join(notes)}" if notes else ""

    if top.benchmark_status == "none":
        return "Low", f"no benchmark data, gap +{gap:.1f}{risk_note}"
    if top.benchmark_status == "self_reported":
        return (
            "Low",
            f"uploader-reported benchmark only (unverified), gap +{gap:.1f}{risk_note}",
        )
    if top.benchmark_status == "estimated":
        if gap >= 2.0:
            confidence = "Medium"
        else:
            confidence = "Low"
        if top.speed_confidence == "low" and confidence == "Medium":
            confidence = "Low"
        return confidence, f"estimated benchmark, gap +{gap:.1f}{risk_note}"
    if gap >= 2.5:
        confidence = "High"
        reason = f"direct benchmark, gap +{gap:.1f}{risk_note}"
    elif gap >= 1.0:
        confidence = "Medium"
        reason = f"direct benchmark, gap +{gap:.1f}{risk_note}"
    else:
        confidence = "Low"
        reason = f"direct benchmark but very close (+{gap:.1f}){risk_note}"


    if top.fit_type != "full_gpu" or top.speed_confidence == "low":
        if confidence == "High":
            confidence = "Medium"
        elif confidence == "Medium":
            confidence = "Low"
    return confidence, reason


def display_hardware(hw: HardwareInfo) -> None:
    lines: list[str] = []

    if hw.gpus:
        for i, gpu in enumerate(hw.gpus):
            if gpu.shared_memory:
                vram = (
                    f"{format_bytes(gpu.vram_bytes)} shared"
                    if gpu.vram_bytes > 0
                    else "shared memory"
                )
            else:
                vram = format_bytes(gpu.vram_bytes)
            if (
                gpu.usable_vram_bytes is not None
                and gpu.usable_vram_bytes < gpu.vram_bytes
            ):
                vram += f" (budget {format_bytes(gpu.usable_vram_bytes)})"
            bw = (
                f"{gpu.memory_bandwidth_gbps:.0f} GB/s"
                if gpu.memory_bandwidth_gbps
                else "N/A"
            )
            cc = (
                f"CC {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                if gpu.compute_capability
                else ""
            )
            extra = []
            if cc:
                extra.append(cc)
            if gpu.cuda_version:
                extra.append(f"CUDA {gpu.cuda_version}")
            if gpu.rocm_version:
                extra.append(f"ROCm {gpu.rocm_version}")
            backends = [
                c.name.upper()
                for c in gpu.backend_capabilities
                if c.available and c.name != "cpu"
            ]
            if backends:
                extra.append("Backends " + "/".join(backends))
            if gpu.neural_engine_available:
                extra.append("ANE info")
            extra_str = f" ({', '.join(extra)})" if extra else ""
            lines.append(f"[bold {MINT}]GPU {i}[/] {gpu.name}")
            lines.append(f"  [dim]memory[/] {vram}   [dim]bandwidth[/] {bw}")
            if extra_str:
                lines.append(f"  [dim]{extra_str.strip(' ()')}[/]")
    else:
        lines.append(f"[{AMBER}]No GPU detected[/] - CPU/RAM fallback")

    avx_flags = []
    if hw.has_avx2:
        avx_flags.append("AVX2")
    if hw.has_avx512:
        avx_flags.append("AVX-512")
    avx_str = f" ({', '.join(avx_flags)})" if avx_flags else ""
    lines.append(f"[bold {CYAN}]CPU[/] {hw.cpu_name} - {hw.cpu_cores} cores{avx_str}")

    ram = format_bytes(hw.ram_bytes)
    if hw.ram_budget_bytes is not None and hw.ram_budget_bytes < hw.ram_bytes:
        ram += f" (budget {format_bytes(hw.ram_budget_bytes)})"
    lines.append(f"[bold {CYAN}]RAM[/] {ram}")
    lines.append(f"[bold {CYAN}]Disk[/] {format_bytes(hw.disk_free_bytes)} free")
    lines.append(f"[bold {CYAN}]OS[/] {hw.os}")
    for note in hw.budget_notes:
        lines.append(f"[dim]{note}[/dim]")

    panel = Panel(
        "\n".join(lines),
        title=f"[bold {ACCENT}]Vision Hardware[/]",
        border_style=ACCENT,
        box=box.ROUNDED,
    )
    console.console.print(panel)


def display_ranking(
    results: list[CompatibilityResult],
    *,
    has_gpu: bool = True,
    show_status: bool = False,
    empty_message: str | None = None,
) -> None:
    if not results:
        console.console.print(
            f"[yellow]{empty_message or 'No compatible models found for your hardware.'}[/]"
        )
        return

    mem_label = "VRAM" if has_gpu else "RAM"

    table = Table(
        title=f"[bold {ACCENT}]VLM Fit Board[/]",
        box=box.SIMPLE_HEAVY,
        border_style=VIOLET,
        header_style=f"bold {CYAN}",
        row_styles=["", "dim"],
        show_lines=False,
    )
    table.add_column("#", style=f"bold {ACCENT}", width=3, justify="right")
    table.add_column("Model", style=CYAN, min_width=14, overflow="fold")
    table.add_column("Artifact", style=VIOLET, justify="center", width=8)
    if show_status:
        table.add_column(f"Fit / {mem_label}", justify="center", width=8)
        table.add_column("Speed", justify="right", width=12)
        table.add_column("Published", justify="center", width=10)
    else:
        table.add_column("Params", justify="right", width=6)
        table.add_column("Published", justify="center", width=10)
        table.add_column("Downloads", justify="right", width=9)
    table.add_column("Score", justify="right", width=5)

    download_logs = [
        log10(max(r.model.downloads, 1)) for r in results if r.model.downloads > 0
    ]
    min_download_log = min(download_logs) if download_logs else 0.0
    max_download_log = max(download_logs) if download_logs else 1.0
    published_dates = [parse_published_at(r.model.published_at) for r in results]
    published_valid = [d for d in published_dates if d is not None]
    oldest_ts = min((d.timestamp() for d in published_valid), default=None)
    newest_ts = max((d.timestamp() for d in published_valid), default=None)

    for i, r in enumerate(results, 1):
        quant = effective_quant_type(r.model, r.gguf_variant)
        vram_str = format_bytes(r.vram_required_bytes)
        speed_str = format_speed(r)

        score_val = f"{r.quality_score:.1f}"
        if r.benchmark_status == "none":
            score_str = f"[red]{score_val} ?[/red]"
        elif r.benchmark_status == "self_reported":
            score_str = f"[{AMBER}]{score_val} !sr[/{AMBER}]"
        elif r.benchmark_status == "estimated":
            score_str = f"[{AMBER}]{score_val} ~[/{AMBER}]"
        else:
            score_str = f"[{MINT}]{score_val}[/{MINT}]"

        fit_style = {
            "full_gpu": f"[{MINT}]Full GPU[/]",
            "partial_offload": f"[{AMBER}]Partial[/]",
            "cpu_only": "[red]CPU only[/]",
        }
        fit_str = fit_style.get(r.fit_type, r.fit_type)
        published_dt = parse_published_at(r.model.published_at)
        published_str = Text(
            format_published_at(r.model.published_at),
            style=published_style(published_dt, oldest_ts, newest_ts),
        )
        downloads_str = Text(
            format_downloads(r.model.downloads),
            style=downloads_style(
                r.model.downloads, min_download_log, max_download_log
            ),
        )

        params_str = format_params(r.model.parameter_count)
        if r.model.is_moe and r.model.parameter_count_active:
            params_str += f" ({format_params(r.model.parameter_count_active)}a)"

        model_link = Text(r.model.id, style=CYAN)
        model_link.stylize(f"link https://huggingface.co/{r.model.id}")
        if show_status:
            model_link.append(f"\n{params_str}", style="dim")

        row_cells = [
            str(i),
            model_link,
            quant,
        ]
        if show_status:
            row_cells.extend(
                [f"{fit_str}\n[dim]{vram_str}[/dim]", speed_str, published_str]
            )
        else:
            row_cells.append(params_str)
            row_cells.extend([published_str, downloads_str])
        row_cells.append(score_str)
        table.add_row(*row_cells)

    console.console.print(table)

    has_estimated = any(r.benchmark_status == "estimated" for r in results)
    has_self = any(r.benchmark_status == "self_reported" for r in results)
    has_none = any(r.benchmark_status == "none" for r in results)
    if has_estimated or has_none or has_self:
        parts = []
        if has_self:
            parts.append(
                "[bright_yellow]!sr[/bright_yellow] = uploader-reported only (unverified)"
            )
        if has_estimated:
            parts.append("[yellow]Estimated / ~[/yellow] = inferred from model family")
        if has_none:
            parts.append("[red]None / ?[/red] = no benchmark data")
        console.console.print(f"  [dim]Score:[/dim]  {',  '.join(parts)}")

    if show_status:
        has_speed_medium = any(r.speed_confidence == "medium" for r in results)
        has_speed_low = any(r.speed_confidence == "low" for r in results)
        if has_speed_medium or has_speed_low:
            parts = []
            if has_speed_medium:
                parts.append("[yellow]~[/yellow] = estimated tok/s range")
            if has_speed_low:
                parts.append("[red]?[/red] = low-confidence/backend-sensitive tok/s")
            console.console.print(f"  [dim]Speed:[/dim]  {',  '.join(parts)}")

    has_direct = any(r.benchmark_status == "direct" for r in results)
    if not has_direct:
        console.console.print(
            "  [red]No confirmed winner:[/] direct benchmark data is missing for current candidates."
        )

    confidence, reason = top_pick_confidence(results)
    confidence_style = {
        "High": "green",
        "Medium": "yellow",
        "Low": "red",
    }[confidence]
    console.console.print(
        f"  Top pick confidence: [{confidence_style}]{confidence}[/{confidence_style}] ({reason})"
    )

    from whichvlm.models.benchmark_sources import BENCHMARK_SNAPSHOT

    console.console.print(
        f"  [dim]Benchmark reference: {BENCHMARK_SNAPSHOT} curated snapshot; "
        "vision scores lead VLMs, text scores are fallback evidence.[/dim]"
    )


    if len(results) >= 2:
        gap = results[0].quality_score - results[1].quality_score
        if gap < 1.5:
            console.console.print(
                f"  [yellow]Note:[/] Top candidates are very close (#{1} vs #{2}: {gap:.1f} pts)."
            )


    weak_top = [
        idx + 1 for idx, r in enumerate(results[:3]) if r.benchmark_status != "direct"
    ]
    if weak_top:
        joined = ", ".join(f"#{i}" for i in weak_top)
        console.console.print(
            f"  [yellow]Caution:[/] Weaker benchmark evidence in top ranks: {joined}"
        )

    weak_speed_top = [
        idx + 1 for idx, r in enumerate(results[:3]) if r.speed_confidence == "low"
    ]
    if weak_speed_top:
        joined = ", ".join(f"#{i}" for i in weak_speed_top)
        console.console.print(
            f"  [yellow]Speed caution:[/] Low-confidence speed estimates in top ranks: {joined}"
        )

    specialized: list[str] = []
    for idx, r in enumerate(results[:10], 1):
        tags = detect_specializations(r.model.id)
        if tags:
            joined_tags = "/".join(tags)
            specialized.append(f"#{idx} {joined_tags}")
    if specialized:
        console.console.print(
            "  [yellow]Task hint:[/] Specialized models detected in ranking: "
            + ", ".join(specialized)
        )

    for i, r in enumerate(results[:3], 1):
        if r.warnings:
            for w in r.warnings:
                console.console.print(f"  [yellow]Warning #{i} {r.model.name}:[/] {w}")
