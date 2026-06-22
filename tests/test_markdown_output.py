from io import StringIO

from rich.console import Console

from whichvlm.engine.types import CompatibilityResult
from whichvlm.hardware.types import GPUInfo, HardwareInfo
from whichvlm.models.types import GGUFVariant, ModelInfo
from whichvlm.output.markdown import display_markdown


def capture_markdown(
    results: list[CompatibilityResult],
    hardware: HardwareInfo,
    *,
    show_status: bool,
    empty_message: str | None = None,
) -> str:
    import whichvlm.output.console as console_mod

    buf = StringIO()
    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_markdown(
            results,
            hardware,
            show_status=show_status,
            empty_message=empty_message,
        )
    finally:
        console_mod.console = orig_console
    return buf.getvalue().strip()


def hardware_fixture() -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="RTX 4090",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                memory_bandwidth_gbps=1008.0,
            )
        ],
        cpu_name="Test CPU",
        cpu_cores=16,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
    )


def markdown_result(
    index: int,
    *,
    benchmark_status: str = "direct",
    speed_confidence: str = "medium",
) -> CompatibilityResult:
    model = ModelInfo(
        id=f"org/Test-{index}|Model",
        family_id=f"test-{index}",
        name=f"Test-{index}",
        parameter_count=7_000_000_000 + index,
        downloads=1_500 * index,
        likes=index,
        license="apache-2.0",
        published_at=f"2026-01-0{index}T00:00:00Z",
    )
    return CompatibilityResult(
        model=model,
        gguf_variant=GGUFVariant(
            filename=f"test-{index}.gguf",
            quant_type="Q4_K_M",
            file_size_bytes=4 * 1024**3,
        ),
        can_run=True,
        vram_required_bytes=(4 + index) * 1024**3,
        vram_available_bytes=24 * 1024**3,
        estimated_tok_per_sec=10.0 * index,
        speed_confidence=speed_confidence,
        quality_score=80.0 - index,
        fit_type="full_gpu",
        benchmark_status=benchmark_status,
        benchmark_source=benchmark_status,
        benchmark_confidence=1.0,
    )


def test_display_markdown_runtime_table_top_three():
    output = capture_markdown(
        [
            markdown_result(1, speed_confidence="medium"),
            markdown_result(2, benchmark_status="estimated", speed_confidence="low"),
            markdown_result(3, benchmark_status="none", speed_confidence="high"),
        ],
        hardware_fixture(),
        show_status=True,
    )

    assert output.startswith("## Recommended Models")
    assert (
        "| # | Model | Params | Quant | Fit | VRAM | Speed | Published | Score | License |"
        in output
    )
    assert (
        "| 1 | org/Test-1\\|Model | 7.0B | Q4_K_M | Full GPU | 5.0 GB | 10.0 tok/s ~ | 2026-01-01 | 79.0 | apache-2.0 |"
        in output
    )
    assert "20.0 tok/s ?" in output
    assert "78.0 ~" in output
    assert "77.0 ?" in output


def test_display_markdown_details_table_uses_metadata_columns():
    output = capture_markdown(
        [markdown_result(1)], hardware_fixture(), show_status=False
    )

    assert (
        "| # | Model | Params | Quant | Published | Downloads | Score | License |"
        in output
    )
    assert "Fit | VRAM | Speed" not in output
    assert (
        "| 1 | org/Test-1\\|Model | 7.0B | Q4_K_M | 2026-01-01 | 1.5K | 79.0 | apache-2.0 |"
        in output
    )


def test_display_markdown_empty_results():
    output = capture_markdown(
        [],
        hardware_fixture(),
        show_status=True,
        empty_message="Nothing matched.",
    )

    assert output == "## Recommended Models\n\nNothing matched."
