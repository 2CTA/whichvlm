from __future__ import annotations

import re

import typer

import output.console as console_mod
from data.gpu import BYTES_PER_GIB
from hardware.types import HardwareInfo

MEMORY_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>gib|gb|g|mib|mb|m)?$",
    re.IGNORECASE,
)


def parse_memory_amount(
    value: str, *, option_name: str, total_bytes: int | None = None
) -> int:
    raw = value.strip()
    if not raw:
        console_mod.console.print(f"[red]Error:[/] {option_name} cannot be empty.")
        raise typer.Exit(code=1)

    if raw.endswith("%"):
        if total_bytes is None:
            console_mod.console.print(
                f"[red]Error:[/] {option_name} percentage needs a base size."
            )
            raise typer.Exit(code=1)
        try:
            pct = float(raw[:-1])
        except ValueError:
            console_mod.console.print(
                f"[red]Error:[/] Invalid {option_name}: {value!r}."
            )
            raise typer.Exit(code=1)
        if pct < 0:
            console_mod.console.print(
                f"[red]Error:[/] {option_name} must be non-negative."
            )
            raise typer.Exit(code=1)
        return int(total_bytes * pct / 100.0)

    match = MEMORY_RE.match(raw)
    if not match:
        console_mod.console.print(
            f"[red]Error:[/] Invalid {option_name}: {value!r}. "
            "Use values like 1.5GB, 512MB, 10%, or 8."
        )
        raise typer.Exit(code=1)

    number = float(match.group("number"))
    unit = (match.group("unit") or "gb").lower()
    if number < 0:
        console_mod.console.print(f"[red]Error:[/] {option_name} must be non-negative.")
        raise typer.Exit(code=1)

    if unit in {"gib", "gb", "g"}:
        return int(number * BYTES_PER_GIB)
    return int(number * 1024**2)


def auto_vram_headroom(vram_bytes: int) -> int:
    if vram_bytes <= 0:
        return 0
    return int(max(512 * 1024**2, min(vram_bytes * 0.05, 2 * BYTES_PER_GIB)))


def parse_vram_headroom(value: str, vram_bytes: int) -> int:
    mode = value.strip().lower()
    if mode == "auto":
        return auto_vram_headroom(vram_bytes)
    if mode in {"none", "off", "0"}:
        return 0
    return parse_memory_amount(
        value,
        option_name="--vram-headroom",
        total_bytes=vram_bytes,
    )


def format_budget_bytes(value: int) -> str:
    if value >= BYTES_PER_GIB:
        return f"{value / BYTES_PER_GIB:.1f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.0f} MB"
    return f"{value / 1024:.0f} KB"


def apply_memory_budgets(
    hardware: HardwareInfo,
    *,
    vram_headroom: str,
    perf_vram: str = "none",
    ram_budget: str | None,
) -> HardwareInfo:
    headroom_mode = vram_headroom.strip().lower()
    if not hardware.gpus and headroom_mode not in {"auto", "none", "off", "0"}:
        parse_memory_amount(
            vram_headroom,
            option_name="--vram-headroom",
            total_bytes=BYTES_PER_GIB,
        )
    perf_mode = perf_vram.strip().lower()
    if not hardware.gpus and perf_mode not in {"none", "off", "0"}:
        parse_memory_amount(
            perf_vram,
            option_name="--perf-vram",
            total_bytes=BYTES_PER_GIB,
        )

    reserved_values: list[int] = []
    perf_reserved_values: list[int] = []
    for gpu in hardware.gpus:
        reserved = parse_vram_headroom(vram_headroom, gpu.vram_bytes)
        perf_reserved = 0
        if perf_mode not in {"none", "off", "0"}:
            perf_reserved = parse_memory_amount(
                perf_vram,
                option_name="--perf-vram",
                total_bytes=gpu.vram_bytes,
            )
        gpu.usable_vram_bytes = max(0, gpu.vram_bytes - reserved - perf_reserved)
        if reserved > 0:
            reserved_values.append(reserved)
        if perf_reserved > 0:
            perf_reserved_values.append(perf_reserved)

    if reserved_values:
        unique_reserved = sorted(set(reserved_values))
        if len(unique_reserved) == 1:
            note = f"VRAM headroom: {format_budget_bytes(unique_reserved[0])} reserved per GPU"
        else:
            note = "VRAM headroom: auto reserve applied per GPU"
        hardware.budget_notes.append(note)
    if perf_reserved_values:
        unique_reserved = sorted(set(perf_reserved_values))
        if len(unique_reserved) == 1:
            note = f"Performance reserve: {format_budget_bytes(unique_reserved[0])} per GPU"
        else:
            note = "Performance reserve: applied per GPU"
        hardware.budget_notes.append(note)

    if ram_budget:
        mode = ram_budget.strip().lower()
        if mode == "available":
            from hardware.memory import detect_available_ram_bytes

            hardware.ram_budget_bytes = detect_available_ram_bytes()
            hardware.budget_notes.append(
                f"RAM budget: current available {format_budget_bytes(hardware.ram_budget_bytes)}"
            )
        elif mode not in {"auto", "none", "off"}:
            hardware.ram_budget_bytes = parse_memory_amount(
                ram_budget, option_name="--ram-budget", total_bytes=hardware.ram_bytes
            )
            hardware.budget_notes.append(
                f"RAM budget: {format_budget_bytes(hardware.ram_budget_bytes)}"
            )
    return hardware
