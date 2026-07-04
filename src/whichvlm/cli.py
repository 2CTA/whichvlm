from __future__ import annotations

import asyncio
import re
from typing import Optional

import httpx
import typer
from rich.console import Console

from whichvlm.constants import BYTES_PER_GIB
from whichvlm.engine.workload import Workload
from whichvlm.hardware.types import HardwareInfo, ensure_backend_capabilities
from whichvlm.models.types import GGUFVariant, ModelInfo
from whichvlm.runtime import (
    RuntimeRequest,
    RuntimeUnsupportedError,
    ServeRequest,
    generate_run_script,
    normalize_backend_name,
    requires_image,
    resolve_model_deps,
    run_request,
    select_backend,
    select_serve_backend,
    serve_request,
)
from whichvlm.utils import current_version, CONTEXT_LENGTH

# CLI hub. Turns flags into load, rank, run, and render work.
app = typer.Typer(
    name="whichvlm",
    help="Find local vision-language models that fit your hardware.",
    no_args_is_help=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()
FETCH_ERRORS = (httpx.HTTPError, OSError, ValueError)
PROFILE_CHOICES = (
    "general",
    "coding",
    "vision",
    "math",
    "any",
    "image_qa",
    "ocr",
    "document",
    "chart",
    "video",
    "audio",
    "general_multimodal",
)
PROFILE_HELP = f"Ranking profile: {' | '.join(PROFILE_CHOICES)}"
DATA_PANEL = "Data"
RANKING_PANEL = "Ranking"
WORKLOAD_PANEL = "Workload"
HARDWARE_PANEL = "Hardware"
OUTPUT_PANEL = "Output"


def vlm_progress():
    # Progress widget. Keeps network-heavy steps readable in terminal.
    from rich.progress import Progress, SpinnerColumn, TextColumn

    return Progress(
        SpinnerColumn("dots12", style="bold #f472b6"),
        TextColumn("[#67e8f9]VLM[/] [#f472b6]{task.description}"),
        console=console,
        transient=True,
    )


def format_fetch_error(error: Exception) -> str:
    # Error flattener. Gives one short message even for empty HTTP errors.
    detail = str(error).strip()
    if detail:
        return detail

    response = getattr(error, "response", None)
    request = getattr(error, "request", None) or getattr(response, "request", None)
    status_code = getattr(response, "status_code", None)
    url = getattr(request, "url", None)
    if status_code and url:
        return f"{type(error).__name__}: HTTP {status_code} for {url}"
    if url:
        return f"{type(error).__name__} while requesting {url}"
    return f"{type(error).__name__} with no detail from the network layer"


def load_benchmark_index(refresh: bool) -> dict[str, float]:
    from whichvlm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )

    cached = None if refresh else load_benchmark_cache()
    if cached is not None:
        return cached

    try:
        scores = asyncio.run(fetch_benchmark_scores())
    except FETCH_ERRORS as error:
        console.print(
            "[yellow]Warning:[/] Benchmark data unavailable: "
            f"{format_fetch_error(error)}"
        )
        return load_benchmark_cache(allow_stale=True) or {}

    save_benchmark_cache(scores)
    return scores


def print_version(value: bool) -> None:
    if value:
        console.print(current_version())
        raise typer.Exit()


def validate_gpu_flags(
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
) -> None:
    if cpu_only and gpu:
        console.print("[red]Error:[/] --cpu-only and --gpu are mutually exclusive.")
        raise typer.Exit(code=1)
    if vram is not None and not gpu:
        console.print("[red]Error:[/] --vram requires --gpu.")
        raise typer.Exit(code=1)


def validate_output_flags(json_output: bool, markdown_output: bool) -> None:
    if json_output and markdown_output:
        console.print("[red]Error:[/] --json and --markdown are mutually exclusive.")
        raise typer.Exit(code=1)


def validate_profile(profile: str) -> str:
    p = profile.lower()
    if p not in PROFILE_CHOICES:
        choices = ", ".join(PROFILE_CHOICES)
        console.print(f"[red]Error:[/] --profile must be one of: {choices}.")
        raise typer.Exit(code=1)
    return p


def validate_evidence(evidence: str) -> str:
    valid = {"strict", "base", "any"}
    mode = evidence.lower()
    if mode not in valid:
        console.print("[red]Error:[/] --evidence must be one of: strict, base, any.")
        raise typer.Exit(code=1)
    return mode


def validate_freshness_weight(value: float) -> float:
    if value < 0.0 or value > 1.0:
        console.print("[red]Error:[/] --freshness-weight must be between 0 and 1.")
        raise typer.Exit(code=1)
    return value


def resolve_evidence_mode(evidence: str, direct: bool) -> str:
    mode = validate_evidence(evidence)
    if direct:
        return "strict"
    return mode


def resolve_fit_filter(fit: str, gpu_only: bool) -> str:
    mode = fit.lower().replace("_", "-").replace(" ", "-")
    if mode not in {"any", "gpu", "full-gpu", "fullgpu"}:
        console.print("[red]Error:[/] --fit must be one of: any, gpu, full-gpu.")
        raise typer.Exit(code=1)
    if gpu_only:
        return "full_gpu"
    return "full_gpu" if mode in {"gpu", "full-gpu", "fullgpu"} else "any"


def resolve_speed_filter(speed: str, min_speed: float | None) -> float | None:
    if min_speed is not None:
        return min_speed
    mode = speed.lower().replace("_", "-")
    presets = {
        "any": None,
        "usable": 10.0,
        "fast": 30.0,
    }
    if mode not in presets:
        console.print("[red]Error:[/] --speed must be one of: any, usable, fast.")
        raise typer.Exit(code=1)
    return presets[mode]


MEMORY_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>gib|gb|g|mib|mb|m)?$",
    re.IGNORECASE,
)


def parse_memory_amount(
    value: str, *, option_name: str, total_bytes: int | None = None
) -> int:
    # Memory parser. Accepts GiB, MiB, and percent budget inputs.
    raw = value.strip()
    if not raw:
        console.print(f"[red]Error:[/] {option_name} cannot be empty.")
        raise typer.Exit(code=1)

    if raw.endswith("%"):
        if total_bytes is None:
            console.print(f"[red]Error:[/] {option_name} percentage needs a base size.")
            raise typer.Exit(code=1)
        try:
            pct = float(raw[:-1])
        except ValueError:
            console.print(f"[red]Error:[/] Invalid {option_name}: {value!r}.")
            raise typer.Exit(code=1)
        if pct < 0:
            console.print(f"[red]Error:[/] {option_name} must be non-negative.")
            raise typer.Exit(code=1)
        return int(total_bytes * pct / 100.0)

    match = MEMORY_RE.match(raw)
    if not match:
        console.print(
            f"[red]Error:[/] Invalid {option_name}: {value!r}. "
            "Use values like 1.5GB, 512MB, 10%, or 8."
        )
        raise typer.Exit(code=1)

    number = float(match.group("number"))
    unit = (match.group("unit") or "gb").lower()
    if number < 0:
        console.print(f"[red]Error:[/] {option_name} must be non-negative.")
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


def apply_memory_budgets(
    hardware: HardwareInfo,
    *,
    vram_headroom: str,
    perf_vram: str = "none",
    ram_budget: str | None,
) -> HardwareInfo:
    # Budget pass. Writes usable memory limits onto detected hardware.
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
            note = (
                "Performance VRAM: "
                f"{format_budget_bytes(unique_reserved[0])} reserved per GPU"
            )
        else:
            note = "Performance VRAM: reserve applied per GPU"
        hardware.budget_notes.append(note)

    if ram_budget:
        mode = ram_budget.strip().lower()
        if mode == "available":
            from whichvlm.hardware.memory import detect_available_ram_bytes

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


def format_budget_bytes(value: int) -> str:
    if value >= BYTES_PER_GIB:
        return f"{value / BYTES_PER_GIB:.1f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.0f} MB"
    return f"{value / 1024:.0f} KB"


def apply_gpu_overrides(
    hardware: HardwareInfo,
    cpu_only: bool,
    gpu: list[str] | None,
    vram: float | None,
) -> HardwareInfo:
    if cpu_only:
        hardware.gpus = []
    elif gpu:
        from whichvlm.hardware.gpu_simulator import create_synthetic_gpus

        try:
            hardware.gpus = create_synthetic_gpus(gpu, vram)
            for gpu_info in hardware.gpus:
                ensure_backend_capabilities(gpu_info, hardware.os)
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(code=1)
    return hardware


def auto_min_params_for_profile(hardware: HardwareInfo, profile: str) -> float | None:

    if profile != "general":
        return None
    if not hardware.gpus:
        return 2.0
    from whichvlm.hardware.memory import effective_usable_ram

    usable_ram = effective_usable_ram(hardware.ram_bytes, hardware.ram_budget_bytes)
    best_vram_gb = max(
        (
            usable_ram
            if g.shared_memory
            and (g.vram_bytes == 0 or hardware.ram_budget_bytes is not None)
            else (
                g.usable_vram_bytes if g.usable_vram_bytes is not None else g.vram_bytes
            )
        )
        for g in hardware.gpus
    ) / (1024**3)
    if best_vram_gb >= 30:
        return 12.0
    if best_vram_gb >= 20:
        return 10.0
    if best_vram_gb >= 12:
        return 8.0
    if best_vram_gb >= 8:
        return 5.0
    if best_vram_gb >= 5:
        return 3.0
    return 2.0


def include_vision_candidates(profile: str) -> bool:
    return profile.lower() in {
        "vision",
        "any",
        "image_qa",
        "ocr",
        "document",
        "chart",
        "video",
        "audio",
        "general_multimodal",
    }


def workload_for_profile(
    profile: str,
    *,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
    audio_seconds: float = 0.0,
    batch_size: int = 1,
    context_length: int = 4096,
) -> Workload | None:
    task_by_profile = {
        "any": "general_multimodal",
        "vision": "image_qa",
        "image_qa": "image_qa",
        "ocr": "ocr",
        "document": "document",
        "chart": "chart",
        "video": "video",
        "audio": "audio",
        "general_multimodal": "general_multimodal",
    }
    task = task_by_profile.get(profile.lower())
    if task is None:
        return None
    if task == "video" and video_frames == 0:
        video_frames = 8
    if task == "audio" and audio_seconds == 0:
        audio_seconds = 30.0
    return Workload(
        task=task,
        image_count=image_count,
        image_size=image_size,
        video_frames=video_frames,
        audio_seconds=audio_seconds,
        batch_size=batch_size,
        context_length=context_length,
    ).normalized()


def fill_missing_published_at(
    all_models: list,
    results: list,
    fetch_model_published_at,
) -> bool:
    missing_ids = [r.model.id for r in results if not r.model.published_at]
    if not missing_ids:
        return False
    published_map = asyncio.run(fetch_model_published_at(missing_ids))
    if not published_map:
        return False

    updated = False
    for model in all_models:
        published_at = published_map.get(model.id)
        if published_at and not model.published_at:
            model.published_at = published_at
            updated = True
    return updated


def merge_model_eval_benchmarks(
    models: list,
    benchmark_scores: dict[str, float],
) -> tuple[dict[str, float], int]:

    return benchmark_scores, 0


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit",
        callback=print_version,
        is_eager=True,
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Refresh Hugging Face model metadata",
        rich_help_panel=DATA_PANEL,
    ),
    top: int = typer.Option(
        10,
        "--top",
        "-n",
        help="Number of top models to show",
        rich_help_panel=RANKING_PANEL,
    ),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    image_count: int = typer.Option(
        1,
        "--image-count",
        help="Images per request for VLM memory estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    image_size: int = typer.Option(
        448,
        "--image-size",
        help="Input image edge size for VLM memory estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    video_frames: int = typer.Option(
        0,
        "--video-frames",
        help="Video frames per request for workload estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    audio_seconds: float = typer.Option(
        0.0,
        "--audio-seconds",
        help="Audio seconds per request for workload estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="Requests per batch for memory and speed estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    quant: Optional[str] = typer.Option(
        None,
        "--quant",
        "-q",
        help="Filter by quantization type (e.g. Q4_K_M)",
        rich_help_panel=RANKING_PANEL,
    ),
    min_speed: Optional[float] = typer.Option(
        None,
        "--min-speed",
        help="Minimum estimated decode tok/s",
        rich_help_panel=RANKING_PANEL,
    ),
    speed: str = typer.Option(
        "any",
        "--speed",
        help="Speed preset: any | usable | fast",
        rich_help_panel=RANKING_PANEL,
    ),
    fit: str = typer.Option(
        "any",
        "--fit",
        help="Memory fit: any | gpu | full-gpu",
        rich_help_panel=RANKING_PANEL,
    ),
    gpu_only: bool = typer.Option(
        False,
        "--gpu-only",
        help="Only show full-GPU fits",
        rich_help_panel=RANKING_PANEL,
    ),
    evidence: str = typer.Option(
        "any",
        "--evidence",
        help="Benchmark evidence filter: strict | base | any",
        rich_help_panel=RANKING_PANEL,
    ),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Alias of --evidence strict",
        rich_help_panel=RANKING_PANEL,
    ),
    status: bool = typer.Option(
        False,
        "--status",
        help="Show runtime columns (default; kept for compatibility)",
        rich_help_panel=OUTPUT_PANEL,
    ),
    details: bool = typer.Option(
        False,
        "--details",
        help="Show metadata columns; with --json, emit full diagnostic JSON",
        rich_help_panel=OUTPUT_PANEL,
    ),
    min_params: Optional[float] = typer.Option(
        None,
        "--min-params",
        help="Minimum effective parameter size in billions (e.g. 7)",
        rich_help_panel=RANKING_PANEL,
    ),
    profile: str = typer.Option(
        "vision",
        "--profile",
        help=PROFILE_HELP,
        rich_help_panel=RANKING_PANEL,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
        rich_help_panel=OUTPUT_PANEL,
    ),
    markdown_output: bool = typer.Option(
        False,
        "--markdown",
        "-m",
        help="Output as GitHub-Flavored Markdown",
        rich_help_panel=OUTPUT_PANEL,
    ),
    cpu_only: bool = typer.Option(
        False,
        "--cpu-only",
        help="Ignore GPUs and rank CPU/RAM fallback",
        rich_help_panel=HARDWARE_PANEL,
    ),
    gpu: Optional[list[str]] = typer.Option(
        None,
        "--gpu",
        help="Simulate GPU hardware, e.g. 'RTX 4090', 'Apple M3 Max', or repeat --gpu",
        rich_help_panel=HARDWARE_PANEL,
    ),
    vram: Optional[float] = typer.Option(
        None,
        "--vram",
        help="Override simulated VRAM in GB; requires --gpu",
        rich_help_panel=HARDWARE_PANEL,
    ),
    vram_headroom: str = typer.Option(
        "auto",
        "--vram-headroom",
        help="Reserve GPU memory for the OS/runtime: auto | none | 1GB | 10%",
        rich_help_panel=HARDWARE_PANEL,
    ),
    perf_vram: str = typer.Option(
        "none",
        "--perf-vram",
        help="Reserve GPU memory for inference performance features: none | 1GB | 10%",
    ),
    ram_budget: Optional[str] = typer.Option(
        None,
        "--ram-budget",
        help="RAM budget for CPU/offload fallback: available | 8GB | 50%",
        rich_help_panel=HARDWARE_PANEL,
    ),
    freshness_weight: float = typer.Option(
        1.0,
        "--freshness-weight",
        help="Scale lineage freshness in ranking scores: 0 disables it, 1 uses full weight",
        rich_help_panel=RANKING_PANEL,
    ),
):

    if ctx.invoked_subcommand is not None:
        return

    validate_gpu_flags(cpu_only, gpu, vram)
    validate_output_flags(json_output, markdown_output)
    profile = validate_profile(profile)
    evidence_mode = resolve_evidence_mode(evidence, direct)
    fit_filter = resolve_fit_filter(fit, gpu_only)
    speed_filter = resolve_speed_filter(speed, min_speed)
    freshness_weight = validate_freshness_weight(freshness_weight)

    from whichvlm.engine.ranker import rank_models
    from whichvlm.hardware.detector import detect_hardware
    from whichvlm.models.cache import save_cache
    from whichvlm.models.fetcher import (
        fetch_model_published_at,
        inventory_source_provenance,
        models_to_dicts,
    )
    from whichvlm.models.grouper import group_models
    from whichvlm.output.display import (
        display_hardware,
        display_json,
        display_markdown,
        display_ranking,
    )

    with vlm_progress() as progress:
        task = progress.add_task("scanning silicon...", total=None)
        hardware = detect_hardware()
        apply_gpu_overrides(hardware, cpu_only, gpu, vram)
        apply_memory_budgets(
            hardware,
            vram_headroom=vram_headroom,
            perf_vram=perf_vram,
            ram_budget=ram_budget,
        )
        progress.update(task, description="hardware mapped")

        progress.update(task, description="loading VLM packages...")
        models = load_model_catalog(
            refresh, include_vision=include_vision_candidates(profile)
        )

        progress.update(task, description="loading benchmark index...")
        bench_scores = load_benchmark_index(refresh)

        progress.update(task, description="scoring multimodal fit...")
        families = group_models(models)

        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        auto_min_params = (
            auto_min_params_for_profile(hardware, profile)
            if min_params is None
            else min_params
        )
        workload = workload_for_profile(
            profile,
            image_count=image_count,
            image_size=image_size,
            video_frames=video_frames,
            audio_seconds=audio_seconds,
            batch_size=batch_size,
            context_length=context_length,
        )

        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=top,
            quant_filter=quant,
            min_speed=speed_filter,
            benchmark_scores=bench_scores,
            task_profile=profile,
            require_direct_top=True,
            min_params_b=auto_min_params,
            evidence_filter=evidence_mode,
            fit_filter=fit_filter,
            vision_workload=workload,
            workload=workload,
            freshness_weight=freshness_weight,
        )

        if not results and auto_min_params is not None and min_params is None:
            results = rank_models(
                all_models,
                hardware,
                context_length=context_length,
                top_n=top,
                quant_filter=quant,
                min_speed=speed_filter,
                benchmark_scores=bench_scores,
                task_profile=profile,
                require_direct_top=True,
                min_params_b=None,
                evidence_filter=evidence_mode,
                fit_filter=fit_filter,
                vision_workload=workload,
                workload=workload,
                freshness_weight=freshness_weight,
            )

        if results:
            try:
                if fill_missing_published_at(
                    all_models, results, fetch_model_published_at
                ):
                    save_cache(
                        models_to_dicts(models),
                        source=inventory_source_provenance(
                            include_vision=include_vision_candidates(profile)
                        ),
                    )
            except FETCH_ERRORS as e:
                progress.update(
                    task, description=f"Published date backfill skipped: {e}"
                )

    empty_message = None
    if fit_filter == "full_gpu":
        empty_message = (
            "No full-GPU models found for this hardware. "
            "Remove --gpu-only or use --fit any to include partial offload "
            "and CPU-only candidates."
        )
    if json_output:
        display_json(results, hardware, details=details)
    elif markdown_output:
        display_markdown(
            results,
            hardware,
            show_status=status or not details,
            empty_message=empty_message,
        )
    else:
        console.print()
        display_hardware(hardware)
        console.print()
        display_ranking(
            results,
            has_gpu=bool(hardware.gpus),
            show_status=status or not details,
            empty_message=empty_message,
        )
        console.print()


@app.command(help="Plan memory, quantization, and GPU fit for a model.")
def plan(
    model_name: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Target quantization (default: Q4_K_M)"
    ),
    image_count: int = typer.Option(
        1,
        "--image-count",
        help="Images per request for VLM memory estimation",
    ),
    image_size: int = typer.Option(
        448,
        "--image-size",
        help="Input image edge size for VLM memory estimation",
    ),
    video_frames: int = typer.Option(
        0,
        "--video-frames",
        help="Video frames to budget as visual inputs",
    ),
    ram: Optional[str] = typer.Option(
        None,
        "--ram",
        help="System RAM budget for partial offload, e.g. 64GB",
    ),
    min_speed: Optional[float] = typer.Option(
        None,
        "--min-speed",
        help="Minimum estimated generation speed in tok/s",
    ),
    os_name: str = typer.Option(
        "linux",
        "--os",
        help="Target OS for backend compatibility",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch models"
    ),
):

    from whichvlm.output.display import display_plan, display_plan_json

    with vlm_progress() as progress:
        progress.add_task("loading VLM packages...", total=None)
        models = load_model_catalog(refresh, include_vision=True)

    model = resolve_model_match(models, model_name)

    target_quant = quant.upper() if quant else "Q4_K_M"
    from whichvlm.hardware.catalog import PLAN_SYSTEM_RAM_BYTES

    system_ram_bytes = (
        parse_memory_amount(ram, option_name="--ram") if ram else PLAN_SYSTEM_RAM_BYTES
    )

    if json_output:
        display_plan_json(
            model,
            context_length,
            target_quant,
            image_count,
            image_size,
            video_frames,
            system_ram_bytes,
            min_speed,
            os_name,
        )
    else:
        console.print()
        display_plan(
            model,
            context_length,
            target_quant,
            image_count,
            image_size,
            video_frames,
            system_ram_bytes,
            min_speed,
            os_name,
        )
        console.print()


@app.command("hardware-plan", help="Rank models for a target GPU.")
def hardware_plan(
    gpu_name: str = typer.Argument(..., help="Target GPU, e.g. 'RTX 4070'"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    quant: Optional[str] = typer.Option(
        None,
        "--quant",
        "-q",
        help="Target quantization",
        rich_help_panel=RANKING_PANEL,
    ),
    top: int = typer.Option(
        10,
        "--top",
        "-n",
        help="Number of models to show",
        rich_help_panel=RANKING_PANEL,
    ),
    profile: str = typer.Option(
        "vision",
        "--profile",
        help=PROFILE_HELP,
        rich_help_panel=RANKING_PANEL,
    ),
    image_count: int = typer.Option(
        1,
        "--image-count",
        help="Images per request for VLM memory estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    image_size: int = typer.Option(
        448,
        "--image-size",
        help="Input image edge size for VLM memory estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    video_frames: int = typer.Option(
        0,
        "--video-frames",
        help="Video frames to budget as visual inputs",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    audio_seconds: float = typer.Option(
        0.0,
        "--audio-seconds",
        help="Audio seconds per request for workload estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="Requests per batch for memory and speed estimation",
        rich_help_panel=WORKLOAD_PANEL,
    ),
    ram: Optional[str] = typer.Option(
        None,
        "--ram",
        help="System RAM budget for partial offload, e.g. 64GB",
        rich_help_panel=HARDWARE_PANEL,
    ),
    vram: Optional[float] = typer.Option(
        None,
        "--vram",
        help="Override target GPU VRAM in GB",
        rich_help_panel=HARDWARE_PANEL,
    ),
    min_speed: Optional[float] = typer.Option(
        None,
        "--min-speed",
        help="Minimum estimated generation speed in tok/s",
        rich_help_panel=RANKING_PANEL,
    ),
    os_name: str = typer.Option(
        "linux",
        "--os",
        help="Target OS for backend compatibility",
        rich_help_panel=HARDWARE_PANEL,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON",
        rich_help_panel=OUTPUT_PANEL,
    ),
    refresh: bool = typer.Option(
        False,
        "--refresh",
        help="Ignore cache and re-fetch models",
        rich_help_panel=DATA_PANEL,
    ),
):

    from whichvlm.engine.ranker import rank_models
    from whichvlm.hardware.catalog import (
        PLAN_SYSTEM_RAM_BYTES,
        PLAN_VRAM_HEADROOM_RATIO,
        lookup_catalog_entry,
    )
    from whichvlm.hardware.gpu_simulator import create_synthetic_gpu
    from whichvlm.hardware.types import HardwareInfo
    from whichvlm.models.grouper import group_models
    from whichvlm.output.display import display_hardware, display_json, display_ranking

    profile = validate_profile(profile)
    system_ram_bytes = (
        parse_memory_amount(ram, option_name="--ram") if ram else PLAN_SYSTEM_RAM_BYTES
    )

    catalog_entry = lookup_catalog_entry(gpu_name)
    if catalog_entry is not None and vram is None:
        hardware = catalog_entry.to_hardware(system_ram_bytes, os_name)
    else:
        try:
            gpu = create_synthetic_gpu(gpu_name, vram)
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(code=1)
        gpu.usable_vram_bytes = int(gpu.vram_bytes * (1.0 - PLAN_VRAM_HEADROOM_RATIO))
        ensure_backend_capabilities(gpu, os_name)
        hardware = HardwareInfo(
            gpus=[gpu],
            ram_bytes=system_ram_bytes,
            disk_free_bytes=1_000 * BYTES_PER_GIB,
            os=os_name,
        )

    with vlm_progress() as progress:
        task = progress.add_task("loading VLM packages...", total=None)
        models = load_model_catalog(
            refresh, include_vision=include_vision_candidates(profile)
        )
        progress.update(task, description="loading benchmark index...")
        bench_scores = load_benchmark_index(refresh)
        progress.update(task, description=f"scoring {gpu_name}...")

        all_models = []
        for family in group_models(models):
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=top,
            quant_filter=quant,
            min_speed=min_speed,
            benchmark_scores=bench_scores,
            task_profile=profile,
            require_direct_top=True,
            vision_workload=workload_for_profile(
                profile,
                image_count=image_count,
                image_size=image_size,
                video_frames=video_frames,
                audio_seconds=audio_seconds,
                batch_size=batch_size,
                context_length=context_length,
            ),
        )

    if json_output:
        display_json(results, hardware, details=True)
    else:
        console.print()
        display_hardware(hardware)
        console.print()
        display_ranking(results, has_gpu=True, show_status=True)
        console.print()


@app.command(help="Compare current hardware with target GPU upgrades.")
def upgrade(
    target_gpus: list[str] = typer.Argument(
        ...,
        help="GPUs to compare against (e.g. 'RTX 4090' 'RTX 5090' 'H100')",
    ),
    context_length: int = typer.Option(
        8192,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for ranking (e.g. 8192, 64k, 128k)",
    ),
    top: int = typer.Option(3, "--top", "-n", help="Best-N models to compare per GPU"),
    profile: str = typer.Option(
        "vision",
        "--profile",
        help="Ranking profile or workload task",
    ),
    image_count: int = typer.Option(
        1,
        "--image-count",
        help="Images per request for VLM memory estimation",
    ),
    image_size: int = typer.Option(
        448,
        "--image-size",
        help="Input image edge size for VLM memory estimation",
    ),
    video_frames: int = typer.Option(
        0,
        "--video-frames",
        help="Video frames per request for workload estimation",
    ),
    audio_seconds: float = typer.Option(
        0.0,
        "--audio-seconds",
        help="Audio seconds per request for workload estimation",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="Requests per batch for memory and speed estimation",
    ),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Compare against a CPU-only baseline"
    ),
    json_output: bool = typer.Option(False, "--json"),
    refresh: bool = typer.Option(False, "--refresh"),
):

    from whichvlm.engine.ranker import rank_models
    from whichvlm.hardware.detector import detect_hardware
    from whichvlm.hardware.gpu_simulator import create_synthetic_gpu
    from whichvlm.hardware.types import HardwareInfo
    from whichvlm.models.grouper import group_models
    from whichvlm.output.display import display_upgrade, display_upgrade_json

    profile = validate_profile(profile)

    with vlm_progress() as progress:
        task = progress.add_task("scanning silicon...", total=None)
        current_hw = detect_hardware()
        if cpu_only:
            current_hw.gpus = []

        progress.update(task, description="loading VLM packages...")
        models = load_model_catalog(
            refresh, include_vision=include_vision_candidates(profile)
        )

        progress.update(task, description="loading benchmark index...")
        bench_scores = load_benchmark_index(refresh)

        all_models: list = []
        for family in group_models(models):
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        def rank_for(hw: HardwareInfo):
            min_p = auto_min_params_for_profile(hw, profile)
            workload = workload_for_profile(
                profile,
                image_count=image_count,
                image_size=image_size,
                video_frames=video_frames,
                audio_seconds=audio_seconds,
                batch_size=batch_size,
                context_length=context_length,
            )
            results = rank_models(
                all_models,
                hw,
                context_length=context_length,
                top_n=top,
                benchmark_scores=bench_scores,
                task_profile=profile,
                require_direct_top=True,
                min_params_b=min_p,
                vision_workload=workload,
                workload=workload,
            )
            if not results and min_p is not None:
                results = rank_models(
                    all_models,
                    hw,
                    context_length=context_length,
                    top_n=top,
                    benchmark_scores=bench_scores,
                    task_profile=profile,
                    require_direct_top=True,
                    min_params_b=None,
                    vision_workload=workload,
                    workload=workload,
                )
            return results

        progress.update(task, description="scoring current hardware...")
        current_results = rank_for(current_hw)

        target_results: list[tuple[str, HardwareInfo, list]] = []
        for raw_name in target_gpus:
            progress.update(task, description=f"scoring {raw_name}...")
            try:
                synthetic = create_synthetic_gpu(raw_name)
            except ValueError as e:
                console.print(f"[yellow]Skipping {raw_name}:[/] {e}")
                continue
            sim_hw = HardwareInfo(
                gpus=[synthetic],
                cpu_name=current_hw.cpu_name,
                cpu_cores=current_hw.cpu_cores,
                has_avx2=current_hw.has_avx2,
                has_avx512=current_hw.has_avx512,
                ram_bytes=current_hw.ram_bytes,
                disk_free_bytes=current_hw.disk_free_bytes,
                os=current_hw.os,
            )
            sim_results = rank_for(sim_hw)
            target_results.append((raw_name, sim_hw, sim_results))

    if json_output:
        display_upgrade_json(current_hw, current_results, target_results)
    else:
        console.print()
        display_upgrade(current_hw, current_results, target_results)
        console.print()


def load_model_catalog(refresh: bool, include_vision: bool = True) -> list[ModelInfo]:
    # Model loader. Reuses cache first, then falls back to live HF fetch.
    from whichvlm.models.cache import load_cache, save_cache
    from whichvlm.models.fetcher import (
        dicts_to_models,
        fetch_models,
        inventory_source_provenance,
        models_to_dicts,
    )

    if not refresh:
        cached = load_cache()
        if cached is not None:
            return dicts_to_models(cached)
    try:
        models = asyncio.run(fetch_models(include_vision=include_vision))
        save_cache(
            models_to_dicts(models),
            source=inventory_source_provenance(include_vision=include_vision),
        )
        return models
    except FETCH_ERRORS as e:
        cached = load_cache(allow_stale=True)
        if cached is not None:
            console.print(
                f"[yellow]Warning:[/] Hugging Face unavailable; using cached model metadata: "
                f"{format_fetch_error(e)}"
            )
            return dicts_to_models(cached)
        console.print(f"[red]Error fetching models:[/] {format_fetch_error(e)}")
        raise typer.Exit(code=1) from e


def resolve_model_match(models: list[ModelInfo], model_name: str) -> ModelInfo:
    # Model resolver. Turns fuzzy CLI text into one concrete repo id.
    query_lower = model_name.lower()
    terms = query_lower.split()

    matches = [m for m in models if m.id.lower() == query_lower]
    if not matches:
        matches = [m for m in models if m.id.lower().endswith("/" + query_lower)]
    if not matches:
        matches = [m for m in models if all(t in m.id.lower() for t in terms)]

    if not matches:
        console.print(f"[red]No model found matching '{model_name}'.[/]")
        suggestions = [m for m in models if any(t in m.id.lower() for t in terms)]
        if suggestions:
            suggestions.sort(key=lambda m: m.downloads, reverse=True)
            console.print("\n[yellow]Did you mean:[/]")
            for m in suggestions[:5]:
                p = (
                    f"{m.parameter_count / 1e9:.1f}B"
                    if m.parameter_count >= 1e9
                    else f"{m.parameter_count / 1e6:.0f}M"
                )
                console.print(f"  • {m.id} ({p})")
        raise typer.Exit(code=1)

    matches.sort(key=lambda m: m.downloads, reverse=True)
    model = matches[0]
    if len(matches) > 1:
        console.print(f"[dim]Found {len(matches)} matches, using: {model.id}[/]")
    return model


def select_gguf_variant(
    model: ModelInfo, quant_filter: str | None = None
) -> GGUFVariant | None:
    # Variant chooser. Picks the best local GGUF file for the request.
    from whichvlm.constants import QUANT_PREFERENCE_ORDER

    if not model.gguf_variants:
        return None

    if quant_filter:
        variant = lookup_gguf_variant(model, quant_filter)
        if variant is not None:
            return variant
        console.print(
            f"[yellow]Warning:[/] {quant_filter} not available, using best match."
        )

    variant_map = {v.quant_type.upper(): v for v in model.gguf_variants}
    for qt in QUANT_PREFERENCE_ORDER:
        if qt in variant_map:
            return variant_map[qt]
    return model.gguf_variants[0]


def should_select_gguf(backend_name: str | None) -> bool:
    if backend_name is None or backend_name == "auto":
        return True
    return normalize_backend_name(backend_name) == "llama.cpp"


def lookup_gguf_variant(model: ModelInfo, quant_type: str) -> GGUFVariant | None:
    for variant in model.gguf_variants:
        if variant.quant_type.upper() == quant_type.upper():
            return variant
    return None


def same_model_family(candidate: ModelInfo, selected: ModelInfo) -> bool:
    if candidate.id == selected.id:
        return True
    if candidate.family_id and selected.family_id:
        if candidate.family_id == selected.family_id:
            return True
    if candidate.base_model and candidate.base_model == selected.id:
        return True
    if selected.base_model and selected.base_model == candidate.id:
        return True
    if candidate.base_model and selected.base_model:
        return candidate.base_model == selected.base_model
    return False


def parameter_counts_compatible(candidate: ModelInfo, selected: ModelInfo) -> bool:
    if candidate.parameter_count <= 0 or selected.parameter_count <= 0:
        return True
    smaller = min(candidate.parameter_count, selected.parameter_count)
    larger = max(candidate.parameter_count, selected.parameter_count)
    return (larger / smaller) <= 2.0


def resolve_ranked_gguf_for_run(
    selected_model: ModelInfo,
    selected_variant: GGUFVariant,
    models: list[ModelInfo],
    quant_filter: str | None = None,
) -> tuple[ModelInfo, GGUFVariant] | None:
    # Runner resolver. Maps synthetic ranked variants to real GGUF repos.
    desired_quant = quant_filter or selected_variant.quant_type

    if selected_model.gguf_variants:
        variant = lookup_gguf_variant(selected_model, desired_quant)
        return (selected_model, variant) if variant else None

    candidates: list[tuple[bool, int, int, ModelInfo, GGUFVariant]] = []
    for model in models:
        if not model.gguf_variants or not same_model_family(model, selected_model):
            continue
        if not parameter_counts_compatible(model, selected_model):
            continue
        variant = lookup_gguf_variant(model, desired_quant)
        if not variant:
            continue
        explicit_base = model.base_model == selected_model.id
        candidates.append(
            (
                explicit_base,
                model.downloads,
                model.likes,
                model,
                variant,
            )
        )

    if not candidates:
        return None

    _, _, _, model, variant = max(candidates, key=lambda item: item[:3])
    return model, variant


@app.command(help="Run a selected or recommended model.")
def run(
    model_name: Optional[str] = typer.Argument(
        None, help="Model to run (default: auto-pick best)"
    ),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Refresh model metadata"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="CPU-only mode"),
    image: Optional[str] = typer.Option(
        None, "--image", "-i", help="Image path for VLM runners"
    ),
    backend_name: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Runtime backend: auto, transformers, llama.cpp, mlx, vllm, sglang",
    ),
):

    import shutil

    if not shutil.which("uv"):
        console.print("[red]uv is required.[/]")
        console.print(
            "Install: [bold]curl -LsSf https://astral.sh/uv/install.sh | sh[/]"
        )
        raise typer.Exit(code=1)

    with vlm_progress() as progress:
        task = progress.add_task("loading VLM packages...", total=None)
        models = load_model_catalog(refresh)
        progress.remove_task(task)

    variant = None
    hardware = None
    if model_name:
        model = resolve_model_match(models, model_name)
    else:
        from whichvlm.engine.ranker import rank_models
        from whichvlm.hardware.detector import detect_hardware
        from whichvlm.models.benchmark import load_benchmark_cache
        from whichvlm.models.grouper import group_models

        hardware = detect_hardware()
        if cpu_only:
            hardware.gpus = []
        bench_scores = load_benchmark_cache() or {}
        families = group_models(models)
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        results = rank_models(
            all_models,
            hardware,
            context_length=context_length,
            top_n=5,
            quant_filter=quant,
            benchmark_scores=bench_scores,
            task_profile="vision",
            vision_workload=Workload(
                task="image_qa",
                context_length=context_length,
                image_count=1,
            ),
            workload=Workload(
                task="image_qa",
                context_length=context_length,
                image_count=1,
            ),
        )
        if not results:
            console.print("[red]No runnable model found for your hardware.[/]")
            raise typer.Exit(code=1)
        skipped_gguf: list[str] = []
        model = None
        for ranked in results:
            if ranked.gguf_variant:
                resolved = resolve_ranked_gguf_for_run(
                    ranked.model,
                    ranked.gguf_variant,
                    all_models,
                    quant_filter=quant,
                )
                if resolved:
                    resolved_model, variant = resolved
                    if resolved_model.id != ranked.model.id:
                        console.print(
                            "[dim]Resolved GGUF runtime: "
                            f"{ranked.model.id} -> {resolved_model.id} "
                            f"({variant.quant_type})[/]"
                        )
                    model = resolved_model
                    quant = variant.quant_type
                    break
                skipped_gguf.append(ranked.model.id)
                continue

            model = ranked.model
            break

        if skipped_gguf:
            skipped = ", ".join(skipped_gguf[:3])
            suffix = "..." if len(skipped_gguf) > 3 else ""
            console.print(
                "[yellow]Warning:[/] Skipped GGUF-ranked candidate(s) without "
                f"a matching runnable GGUF repo: {skipped}{suffix}"
            )
        if model is None:
            console.print(
                "[red]Error:[/] Top recommendations require GGUF builds, "
                "but no matching GGUF repos were found."
            )
            console.print(
                "[dim]Try specifying a GGUF model explicitly, for example "
                '`whichvlm run "qwen gguf"`.[/]'
            )
            raise typer.Exit(code=1)

    assert model is not None
    if variant is None and should_select_gguf(backend_name):
        variant = select_gguf_variant(model, quant)
    if requires_image(model) and image is None:
        console.print("[red]Error:[/] VLM models require --image PATH.")
        raise typer.Exit(code=1)
    try:
        if hardware is None and backend_name and backend_name != "auto":
            from whichvlm.hardware.detector import detect_hardware

            hardware = detect_hardware()
            if cpu_only:
                hardware.gpus = []
        backend = select_backend(model, variant, hardware, backend_name)
        deps = backend.dependencies(model, variant)
        _, script_type = resolve_model_deps(model, variant, backend_name, hardware)
    except RuntimeUnsupportedError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)

    fmt = variant.quant_type if variant else script_type.upper()
    console.print(f"\n[bold green]Running {model.id}[/] [dim]({fmt})[/]")
    console.print(f"[dim]Setting up isolated env with: {', '.join(deps)}[/]\n")

    try:
        request = RuntimeRequest(
            model=model,
            artifact=variant,
            context_length=context_length,
            cpu_only=cpu_only,
            image_path=image,
            hardware=hardware,
        )
        raise typer.Exit(code=run_request(request, backend.name))
    except RuntimeUnsupportedError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


@app.command(help="Serve a selected model over HTTP.")
def serve(
    model_name: str = typer.Argument(..., help="Model to serve"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length (e.g. 4096, 64k, 128k)",
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Refresh model metadata"),
    cpu_only: bool = typer.Option(False, "--cpu-only", help="CPU-only mode"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    backend_name: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Server backend: auto, llama.cpp, vllm, sglang",
    ),
):
    import shutil

    if not shutil.which("uv"):
        console.print("[red]uv is required.[/]")
        console.print(
            "Install: [bold]curl -LsSf https://astral.sh/uv/install.sh | sh[/]"
        )
        raise typer.Exit(code=1)

    with vlm_progress() as progress:
        task = progress.add_task("loading VLM packages...", total=None)
        models = load_model_catalog(refresh)
        progress.remove_task(task)

    from whichvlm.hardware.detector import detect_hardware

    model = resolve_model_match(models, model_name)
    hardware = detect_hardware()
    if cpu_only:
        hardware.gpus = []

    variant = (
        select_gguf_variant(model, quant) if should_select_gguf(backend_name) else None
    )
    try:
        backend = select_serve_backend(model, variant, hardware, backend_name)
        deps = backend.serve_dependencies(model, variant)
    except RuntimeUnsupportedError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)

    fmt = variant.quant_type if variant else backend.name.upper()
    console.print(f"\n[bold green]Serving {model.id}[/] [dim]({fmt})[/]")
    console.print(f"[dim]Setting up isolated env with: {', '.join(deps)}[/]")
    console.print(f"[dim]Listening on http://{host}:{port}[/]\n")

    try:
        request = ServeRequest(
            model=model,
            artifact=variant,
            context_length=context_length,
            cpu_only=cpu_only,
            hardware=hardware,
            host=host,
            port=port,
        )
        raise typer.Exit(code=serve_request(request, backend.name))
    except RuntimeUnsupportedError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)


@app.command(help="Print a runnable script for a selected model.")
def snippet(
    model_name: Optional[str] = typer.Argument(
        None, help="Model to show snippet for (default: auto-pick best)"
    ),
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Quantization type"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Refresh model metadata"),
    image: Optional[str] = typer.Option(
        None, "--image", "-i", help="Image path for VLM snippets"
    ),
    backend_name: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="Runtime backend: auto, transformers, llama.cpp, mlx, vllm, sglang",
    ),
):

    from rich.syntax import Syntax

    with vlm_progress() as progress:
        task = progress.add_task("loading VLM packages...", total=None)
        models = load_model_catalog(refresh)
        progress.remove_task(task)

    if model_name:
        model = resolve_model_match(models, model_name)
    else:
        gguf_models = [m for m in models if m.gguf_variants]
        if not gguf_models:
            console.print("[red]No GGUF models found.[/]")
            raise typer.Exit(code=1)
        gguf_models.sort(key=lambda m: m.downloads, reverse=True)
        model = gguf_models[0]

    variant = (
        select_gguf_variant(model, quant) if should_select_gguf(backend_name) else None
    )
    if requires_image(model) and image is None:
        console.print("[red]Error:[/] VLM models require --image PATH.")
        raise typer.Exit(code=1)
    try:
        hardware = None
        if backend_name and backend_name != "auto":
            from whichvlm.hardware.detector import detect_hardware

            hardware = detect_hardware()
        deps, _ = resolve_model_deps(model, variant, backend_name, hardware)
        if image is None:
            code = generate_run_script(
                model,
                variant,
                4096,
                False,
                backend_name=backend_name,
                hardware=hardware,
            )
        else:
            code = generate_run_script(
                model,
                variant,
                4096,
                False,
                image_path=image,
                backend_name=backend_name,
                hardware=hardware,
            )
    except RuntimeUnsupportedError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)

    dep_str = " ".join(f"--with {d}" for d in deps)
    console.print(f"\n[bold]{model.id}[/]")
    console.print(f"[dim]# Run directly:[/]  whichvlm run '{model.id}'")
    console.print(f"[dim]# Or manually:[/]   uv run --no-project {dep_str} script.py\n")
    console.print(Syntax(code, "python", theme="monokai"))


@app.command(help="Show detected or simulated hardware.")
def hardware(
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Ignore GPU and run in CPU-only mode"
    ),
    gpu: Optional[list[str]] = typer.Option(
        None,
        "--gpu",
        help="Simulate GPU(s), e.g. 'RTX 4090', '2x RTX 4090', or repeat --gpu",
    ),
    vram: Optional[float] = typer.Option(
        None, "--vram", help="Override VRAM in GB (requires --gpu)"
    ),
):

    validate_gpu_flags(cpu_only, gpu, vram)

    from whichvlm.hardware.detector import detect_hardware
    from whichvlm.output.display import display_hardware

    with vlm_progress() as progress:
        task = progress.add_task("scanning silicon...", total=None)
        hw = detect_hardware()
        apply_gpu_overrides(hw, cpu_only, gpu, vram)
        progress.remove_task(task)

    console.print()
    display_hardware(hw)
    console.print()


if __name__ == "__main__":
    app()
