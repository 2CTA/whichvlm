"""Tests for output formatting helpers."""

from whichvlm.engine.types import CompatibilityResult
from whichvlm.models.types import ModelInfo
from whichvlm.output.formatting import _format_speed


def speed_result(speed: float, confidence: str) -> CompatibilityResult:
    return CompatibilityResult(
        model=ModelInfo(
            id="org/model",
            family_id="org/model",
            name="model",
            parameter_count=1_000_000_000,
        ),
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
        estimated_tok_per_sec=speed,
        speed_confidence=confidence,
    )


def test_format_speed_colors_by_runtime_speed_not_confidence():
    assert _format_speed(speed_result(2.5, "medium")) == "[red]2.5 tok/s ~[/red]"
    assert _format_speed(speed_result(6.0, "low")) == "[yellow]6.0 tok/s ?[/yellow]"
    assert _format_speed(speed_result(12.0, "medium")) == "[green]12.0 tok/s ~[/green]"
    assert (
        _format_speed(speed_result(30.0, "low"))
        == "[bright_green]30.0 tok/s ?[/bright_green]"
    )
