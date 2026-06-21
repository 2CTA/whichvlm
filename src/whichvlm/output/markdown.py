"""GitHub-Flavored Markdown output for ranking results."""

from __future__ import annotations

from whichvlm.engine.quantization import effective_quant_type
from whichvlm.engine.types import CompatibilityResult
from whichvlm.hardware.types import HardwareInfo
from whichvlm.output import _console
from whichvlm.output.formatting import (
    format_bytes,
    format_downloads,
    format_params,
    format_published_at,
)


def _escape_markdown_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _format_markdown_speed(result: CompatibilityResult) -> str:
    speed = result.estimated_tok_per_sec
    if speed is None:
        return "N/A"
    marker = ""
    if result.speed_confidence == "low":
        marker = " ?"
    elif result.speed_confidence == "medium":
        marker = " ~"
    return f"{speed:.1f} tok/s{marker}"


def _format_markdown_score(result: CompatibilityResult) -> str:
    score = f"{result.quality_score:.1f}"
    if result.benchmark_status == "none":
        return f"{score} ?"
    if result.benchmark_status == "self_reported":
        return f"{score} !sr"
    if result.benchmark_status == "estimated":
        return f"{score} ~"
    return score


def _format_markdown_fit(fit_type: str) -> str:
    labels = {
        "full_gpu": "Full GPU",
        "partial_offload": "Partial",
        "cpu_only": "CPU only",
    }
    return labels.get(fit_type, fit_type)


def _format_markdown_params(result: CompatibilityResult) -> str:
    params = format_params(result.model.parameter_count)
    if result.model.is_moe and result.model.parameter_count_active:
        params += f" ({format_params(result.model.parameter_count_active)}a)"
    return params


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |"
        )
    return "\n".join(lines)


def _write_markdown(text: str) -> None:
    _console.console.file.write(text + "\n")
    _console.console.file.flush()


def display_markdown(
    results: list[CompatibilityResult],
    hardware: HardwareInfo,
    *,
    show_status: bool = False,
    empty_message: str | None = None,
) -> None:
    """Emit ranking results as a pasteable GitHub-Flavored Markdown table."""
    lines = ["## Recommended Models", ""]

    if not results:
        lines.append(empty_message or "No compatible models found for your hardware.")
        _write_markdown("\n".join(lines))
        return

    if show_status:
        mem_label = "VRAM" if hardware.gpus else "RAM"
        headers = [
            "#",
            "Model",
            "Params",
            "Quant",
            "Fit",
            mem_label,
            "Speed",
            "Published",
            "Score",
            "License",
        ]
        rows = [
            [
                str(index),
                result.model.id,
                _format_markdown_params(result),
                effective_quant_type(result.model, result.gguf_variant),
                _format_markdown_fit(result.fit_type),
                format_bytes(result.vram_required_bytes),
                _format_markdown_speed(result),
                format_published_at(result.model.published_at),
                _format_markdown_score(result),
                result.model.license or "-",
            ]
            for index, result in enumerate(results, 1)
        ]
    else:
        headers = [
            "#",
            "Model",
            "Params",
            "Quant",
            "Published",
            "Downloads",
            "Score",
            "License",
        ]
        rows = [
            [
                str(index),
                result.model.id,
                _format_markdown_params(result),
                effective_quant_type(result.model, result.gguf_variant),
                format_published_at(result.model.published_at),
                format_downloads(result.model.downloads),
                _format_markdown_score(result),
                result.model.license or "-",
            ]
            for index, result in enumerate(results, 1)
        ]

    lines.append(_markdown_table(headers, rows))
    _write_markdown("\n".join(lines))
