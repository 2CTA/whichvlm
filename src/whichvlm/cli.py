"""CLI entry point using typer."""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Optional

import typer
from rich.console import Console

from whichvlm.constants import _GiB
from whichvlm.engine.workload import VisionWorkload
from whichvlm.hardware.types import HardwareInfo
from whichvlm.models.types import GGUFVariant, ModelInfo
from whichvlm.runtime import (
    RuntimeUnsupportedError,
    generate_run_script,
    requires_image,
    resolve_model_deps,
)
from whichvlm.utils import _current_version, CONTEXT_LENGTH

app = typer.Typer(
    name="whichvlm",
    help="Find local vision-language models that fit your hardware.",
    no_args_is_help=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()


def _run_async(coro):
    return asyncio.run(coro)


def vlm_progress():
    from rich.progress import Progress, SpinnerColumn, TextColumn

    return Progress(
        SpinnerColumn("dots12", style="bold #f472b6"),
        TextColumn("[#67e8f9]VLM[/] [#f472b6]{task.description}"),
        console=console,
        transient=True,
    )


def _format_fetch_error(error: Exception) -> str:
    """Return a useful one-line fetch error even when str(error) is empty."""
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


def _print_version(value: bool) -> None:
    if value:
        console.print(_current_version())
        raise typer.Exit()


def _validate_gpu_flags(
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


def _validate_output_flags(json_output: bool, markdown_output: bool) -> None:
    if json_output and markdown_output:
        console.print("[red]Error:[/] --json and --markdown are mutually exclusive.")
        raise typer.Exit(code=1)


def _validate_profile(profile: str) -> str:
    valid = {"general", "coding", "vision", "math", "any"}
    p = profile.lower()
    if p not in valid:
        console.print(
            "[red]Error:[/] --profile must be one of: general, coding, vision, math, any."
        )
        raise typer.Exit(code=1)
    return p


def _validate_evidence(evidence: str) -> str:
    valid = {"strict", "base", "any"}
    mode = evidence.lower()
    if mode not in valid:
        console.print("[red]Error:[/] --evidence must be one of: strict, base, any.")
        raise typer.Exit(code=1)
    return mode


def _resolve_evidence_mode(evidence: str, direct: bool) -> str:
    mode = _validate_evidence(evidence)
    if direct:
        # 互換性維持のため --direct は strict と同義に固定する。
        return "strict"
    return mode


def _resolve_fit_filter(fit: str, gpu_only: bool) -> str:
    mode = fit.lower().replace("_", "-").replace(" ", "-")
    if mode not in {"any", "gpu", "full-gpu", "fullgpu"}:
        console.print("[red]Error:[/] --fit must be one of: any, gpu, full-gpu.")
        raise typer.Exit(code=1)
    if gpu_only:
        return "full_gpu"
    return "full_gpu" if mode in {"gpu", "full-gpu", "fullgpu"} else "any"


def _resolve_speed_filter(speed: str, min_speed: float | None) -> float | None:
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


_MEMORY_RE = re.compile(
    r"^(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>gib|gb|g|mib|mb|m)?$",
    re.IGNORECASE,
)


def _parse_memory_amount(
    value: str, *, option_name: str, total_bytes: int | None = None
) -> int:
    """Parse memory CLI values. Bare numbers are treated as GiB."""
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

    match = _MEMORY_RE.match(raw)
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
        return int(number * _GiB)
    return int(number * 1024**2)


def _auto_vram_headroom(vram_bytes: int) -> int:
    if vram_bytes <= 0:
        return 0
    return int(max(512 * 1024**2, min(vram_bytes * 0.05, 2 * _GiB)))


def _parse_vram_headroom(value: str, vram_bytes: int) -> int:
    mode = value.strip().lower()
    if mode == "auto":
        return _auto_vram_headroom(vram_bytes)
    if mode in {"none", "off", "0"}:
        return 0
    return _parse_memory_amount(
        value,
        option_name="--vram-headroom",
        total_bytes=vram_bytes,
    )


def _apply_memory_budgets(
    hardware: HardwareInfo,
    *,
    vram_headroom: str,
    ram_budget: str | None,
) -> HardwareInfo:
    """Apply user-facing memory budgets without mutating detected raw sizes."""
    headroom_mode = vram_headroom.strip().lower()
    if not hardware.gpus and headroom_mode not in {"auto", "none", "off", "0"}:
        _parse_memory_amount(
            vram_headroom,
            option_name="--vram-headroom",
            total_bytes=_GiB,
        )

    reserved_values: list[int] = []
    for gpu in hardware.gpus:
        reserved = _parse_vram_headroom(vram_headroom, gpu.vram_bytes)
        gpu.usable_vram_bytes = max(0, gpu.vram_bytes - reserved)
        if reserved > 0:
            reserved_values.append(reserved)

    if reserved_values:
        unique_reserved = sorted(set(reserved_values))
        if len(unique_reserved) == 1:
            note = f"VRAM headroom: {_format_budget_bytes(unique_reserved[0])} reserved per GPU"
        else:
            note = "VRAM headroom: auto reserve applied per GPU"
        hardware.budget_notes.append(note)

    if ram_budget:
        mode = ram_budget.strip().lower()
        if mode == "available":
            from whichvlm.hardware.memory import detect_available_ram_bytes

            hardware.ram_budget_bytes = detect_available_ram_bytes()
            hardware.budget_notes.append(
                f"RAM budget: current available {_format_budget_bytes(hardware.ram_budget_bytes)}"
            )
        elif mode not in {"auto", "none", "off"}:
            hardware.ram_budget_bytes = _parse_memory_amount(
                ram_budget, option_name="--ram-budget", total_bytes=hardware.ram_bytes
            )
            hardware.budget_notes.append(
                f"RAM budget: {_format_budget_bytes(hardware.ram_budget_bytes)}"
            )
    return hardware


def _format_budget_bytes(value: int) -> str:
    if value >= _GiB:
        return f"{value / _GiB:.1f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.0f} MB"
    return f"{value / 1024:.0f} KB"


def _apply_gpu_overrides(
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
        except ValueError as e:
            console.print(f"[red]Error:[/] {e}")
            raise typer.Exit(code=1)
    return hardware


def _auto_min_params_for_profile(hardware: HardwareInfo, profile: str) -> float | None:
    """Pick automatic min-params threshold for strongest general ranking.

    The threshold rises with VRAM so a 24GB GPU is steered away from 3-4B
    toys, but tiny GPUs (4-8GB) still see full-GPU options instead of being
    forced into 7B+ partial-offload-only results.
    """
    if profile != "general":
        return None
    if not hardware.gpus:
        return 2.0  # CPU-only: tiny is the only practical choice
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


def _include_vision_candidates(profile: str) -> bool:
    return profile.lower() in {"vision", "any"}


def _vision_workload_for_profile(
    profile: str,
    *,
    image_count: int = 1,
    image_size: int = 448,
    context_length: int = 4096,
) -> VisionWorkload | None:
    if profile.lower() not in {"vision", "any"}:
        return None
    return VisionWorkload(
        image_count=image_count,
        image_size=image_size,
        context_length=context_length,
    ).normalized()


def _fill_missing_published_at(
    all_models: list,
    results: list,
    fetch_model_published_at,
) -> bool:
    """上位表示で欠けている公開日時を補完し、更新有無を返す。"""
    missing_ids = [r.model.id for r in results if not r.model.published_at]
    if not missing_ids:
        return False
    published_map = _run_async(fetch_model_published_at(missing_ids))
    if not published_map:
        return False

    updated = False
    for model in all_models:
        published_at = published_map.get(model.id)
        if published_at and not model.published_at:
            model.published_at = published_at
            updated = True
    return updated


def _merge_model_eval_benchmarks(
    models: list,
    benchmark_scores: dict[str, float],
) -> tuple[dict[str, float], int]:
    """Deprecated no-op kept for backward API compatibility.

    Previously this injected each model's uploader-reported ``hf_eval``
    value into the leaderboard scores dict under the model's id, which
    caused those values to be treated as ``direct`` benchmark evidence
    by the ranker. That elevated any account that wrote a high number
    in their model card to the top of the rankings.

    The hf_eval value is now consumed inside ``rank_models`` via
    ``BenchmarkEvidence.source == "self_reported"`` with a much lower
    weight and a dedicated display tag, so we no longer need to mutate
    the leaderboard dict here. Returning the input unchanged keeps any
    external callers working.
    """
    return benchmark_scores, 0


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    show_version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit",
        callback=_print_version,
        is_eager=True,
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Refresh Hugging Face model metadata"
    ),
    top: int = typer.Option(10, "--top", "-n", help="Number of top models to show"),
    context_length: int = typer.Option(
        4096,
        "--context-length",
        "-c",
        click_type=CONTEXT_LENGTH,
        help="Context length for KV cache estimation (e.g. 4096, 64k, 128k)",
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
    quant: Optional[str] = typer.Option(
        None, "--quant", "-q", help="Filter by quantization type (e.g. Q4_K_M)"
    ),
    min_speed: Optional[float] = typer.Option(
        None, "--min-speed", help="Minimum estimated decode tok/s"
    ),
    speed: str = typer.Option(
        "any",
        "--speed",
        help="Speed preset: any | usable | fast",
    ),
    fit: str = typer.Option(
        "any",
        "--fit",
        help="Memory fit: any | gpu | full-gpu",
    ),
    gpu_only: bool = typer.Option(
        False,
        "--gpu-only",
        help="Only show full-GPU fits",
    ),
    evidence: str = typer.Option(
        "any",
        "--evidence",
        help="Benchmark evidence filter: strict | base | any",
    ),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Alias of --evidence strict",
    ),
    status: bool = typer.Option(
        False,
        "--status",
        help="Show runtime columns (default; kept for compatibility)",
    ),
    details: bool = typer.Option(
        False,
        "--details",
        help="Show Downloads metadata instead of runtime columns",
    ),
    min_params: Optional[float] = typer.Option(
        None,
        "--min-params",
        help="Minimum effective parameter size in billions (e.g. 7)",
    ),
    profile: str = typer.Option(
        "vision",
        "--profile",
        help="Ranking profile: general | coding | vision | math | any",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    markdown_output: bool = typer.Option(
        False,
        "--markdown",
        "-m",
        help="Output as GitHub-Flavored Markdown",
    ),
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Ignore GPUs and rank CPU/RAM fallback"
    ),
    gpu: Optional[list[str]] = typer.Option(
        None,
        "--gpu",
        help="Simulate GPU hardware, e.g. 'RTX 4090', 'Apple M3 Max', or repeat --gpu",
    ),
    vram: Optional[float] = typer.Option(
        None, "--vram", help="Override simulated VRAM in GB; requires --gpu"
    ),
    vram_headroom: str = typer.Option(
        "auto",
        "--vram-headroom",
        help="Reserve GPU memory for the OS/runtime: auto | none | 1GB | 10%",
    ),
    ram_budget: Optional[str] = typer.Option(
        None,
        "--ram-budget",
        help="RAM budget for CPU/offload fallback: available | 8GB | 50%",
    ),
):
    """Detect hardware and recommend local VLM candidates."""
    if ctx.invoked_subcommand is not None:
        return

    _validate_gpu_flags(cpu_only, gpu, vram)
    _validate_output_flags(json_output, markdown_output)
    profile = _validate_profile(profile)
    evidence_mode = _resolve_evidence_mode(evidence, direct)
    fit_filter = _resolve_fit_filter(fit, gpu_only)
    speed_filter = _resolve_speed_filter(speed, min_speed)

    from whichvlm.engine.ranker import rank_models
    from whichvlm.hardware.detector import detect_hardware
    from whichvlm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )
    from whichvlm.models.cache import load_cache, save_cache
    from whichvlm.models.fetcher import (
        dicts_to_models,
        fetch_model_published_at,
        fetch_models,
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
        _apply_gpu_overrides(hardware, cpu_only, gpu, vram)
        _apply_memory_budgets(
            hardware, vram_headroom=vram_headroom, ram_budget=ram_budget
        )
        progress.update(task, description="hardware mapped")

        progress.update(task, description="loading VLM packages...")
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
            progress.update(
                task, description=f"loaded {len(models)} packages from cache"
            )
        else:
            progress.update(
                task, description="fetching package graph from Hugging Face..."
            )
            try:
                models = _run_async(
                    fetch_models(include_vision=_include_vision_candidates(profile))
                )
                save_cache(models_to_dicts(models))
                progress.update(task, description=f"fetched {len(models)} packages")
            except Exception as e:
                console.print(
                    f"[red]Error fetching models:[/] {_format_fetch_error(e)}"
                )
                sys.exit(1)

        progress.update(task, description="loading benchmark index...")
        bench_scores = None if refresh else load_benchmark_cache()
        if bench_scores is None:
            try:
                progress.update(task, description="fetching benchmark index...")
                bench_scores = _run_async(fetch_benchmark_scores())
                save_benchmark_cache(bench_scores)
            except Exception as e:
                console.print(f"[yellow]Warning:[/] Benchmark data unavailable: {e}")
                bench_scores = {}

        progress.update(task, description="scoring multimodal fit...")
        families = group_models(models)

        # Flatten all models with their family IDs set by grouper
        all_models = []
        for family in families:
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        # NOTE: We no longer merge uploader-reported hf_eval values into the
        # leaderboard scores dict — the ranker now treats them as a separate
        # "self_reported" evidence tier with much lower trust. See
        # ranker.lookup_benchmark_evidence + _SOURCE_WEIGHTS.

        # general用途はGPUクラスに応じた自動しきい値で小さすぎるモデルを抑制する
        auto_min_params = (
            _auto_min_params_for_profile(hardware, profile)
            if min_params is None
            else min_params
        )
        vision_workload = _vision_workload_for_profile(
            profile,
            image_count=image_count,
            image_size=image_size,
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
            vision_workload=vision_workload,
        )

        # 自動しきい値で候補ゼロなら緩和して表示を維持する
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
                vision_workload=vision_workload,
            )

        # 上位候補の公開日時が欠けている場合のみ補完して表示品質を上げる
        if results:
            try:
                if _fill_missing_published_at(
                    all_models, results, fetch_model_published_at
                ):
                    save_cache(models_to_dicts(models))
            except Exception as e:
                progress.update(
                    task, description=f"Published date backfill skipped: {e}"
                )

    # Display results
    empty_message = None
    if fit_filter == "full_gpu":
        empty_message = (
            "No full-GPU models found for this hardware. "
            "Remove --gpu-only or use --fit any to include partial offload "
            "and CPU-only candidates."
        )
    if json_output:
        display_json(results, hardware)
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


@app.command()
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
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore cache and re-fetch models"
    ),
):
    """Show what GPU you need to run a specific model."""
    from whichvlm.models.cache import load_cache, save_cache
    from whichvlm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from whichvlm.output.display import display_plan, display_plan_json

    with vlm_progress() as progress:
        task = progress.add_task("loading VLM packages...", total=None)
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
        else:
            progress.update(
                task, description="fetching package graph from Hugging Face..."
            )
            try:
                models = _run_async(fetch_models(include_vision=True))
                save_cache(models_to_dicts(models))
            except Exception as e:
                console.print(
                    f"[red]Error fetching models:[/] {_format_fetch_error(e)}"
                )
                sys.exit(1)

    model = _search_model(models, model_name)

    target_quant = quant.upper() if quant else "Q4_K_M"

    if json_output:
        display_plan_json(model, context_length, target_quant)
    else:
        console.print()
        display_plan(model, context_length, target_quant)
        console.print()


@app.command()
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
        "vision", "--profile", help="Ranking profile: general | coding | vision | math | any"
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
    cpu_only: bool = typer.Option(
        False, "--cpu-only", help="Compare against a CPU-only baseline"
    ),
    json_output: bool = typer.Option(False, "--json"),
    refresh: bool = typer.Option(False, "--refresh"),
):
    """Compare the current machine against potential GPU upgrades.

    For each GPU passed on the command line, simulate a system with the same
    CPU/RAM but that GPU, run the ranker, and show the best-N models you'd
    be able to run. Useful for answering "is upgrading from a 3090 to a 4090
    worth it?" — the table shows the quality jump and the speed jump for
    each option.
    """
    from whichvlm.engine.ranker import rank_models
    from whichvlm.hardware.detector import detect_hardware
    from whichvlm.hardware.gpu_simulator import create_synthetic_gpu
    from whichvlm.hardware.types import HardwareInfo
    from whichvlm.models.benchmark import (
        fetch_benchmark_scores,
        load_benchmark_cache,
        save_benchmark_cache,
    )
    from whichvlm.models.cache import load_cache, save_cache
    from whichvlm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts
    from whichvlm.models.grouper import group_models
    from whichvlm.output.display import display_upgrade, display_upgrade_json

    profile = _validate_profile(profile)

    with vlm_progress() as progress:
        task = progress.add_task("scanning silicon...", total=None)
        current_hw = detect_hardware()
        if cpu_only:
            current_hw.gpus = []

        progress.update(task, description="loading VLM packages...")
        cached_data = None if refresh else load_cache()
        if cached_data is not None:
            models = dicts_to_models(cached_data)
        else:
            progress.update(
                task, description="fetching package graph from Hugging Face..."
            )
            try:
                models = _run_async(
                    fetch_models(include_vision=_include_vision_candidates(profile))
                )
                save_cache(models_to_dicts(models))
            except Exception as e:
                console.print(
                    f"[red]Error fetching models:[/] {_format_fetch_error(e)}"
                )
                raise typer.Exit(code=1)

        progress.update(task, description="loading benchmark index...")
        bench_scores = None if refresh else load_benchmark_cache()
        if bench_scores is None:
            try:
                bench_scores = _run_async(fetch_benchmark_scores())
                save_benchmark_cache(bench_scores)
            except Exception as e:
                console.print(f"[yellow]Warning:[/] Benchmark data unavailable: {e}")
                bench_scores = {}

        all_models: list = []
        for family in group_models(models):
            all_models.append(family.base_model)
            all_models.extend(family.variants)

        def _rank_for(hw: HardwareInfo):
            min_p = _auto_min_params_for_profile(hw, profile)
            vision_workload = _vision_workload_for_profile(
                profile,
                image_count=image_count,
                image_size=image_size,
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
                vision_workload=vision_workload,
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
                    vision_workload=vision_workload,
                )
            return results

        progress.update(task, description="scoring current hardware...")
        current_results = _rank_for(current_hw)

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
            sim_results = _rank_for(sim_hw)
            target_results.append((raw_name, sim_hw, sim_results))

    if json_output:
        display_upgrade_json(current_hw, current_results, target_results)
    else:
        console.print()
        display_upgrade(current_hw, current_results, target_results)
        console.print()


def _load_models(refresh: bool, include_vision: bool = True):
    from whichvlm.models.cache import load_cache, save_cache
    from whichvlm.models.fetcher import dicts_to_models, fetch_models, models_to_dicts

    cached_data = None if refresh else load_cache()
    if cached_data is not None:
        return dicts_to_models(cached_data)
    try:
        models = _run_async(fetch_models(include_vision=include_vision))
        save_cache(models_to_dicts(models))
        return models
    except Exception as e:
        console.print(f"[red]Error fetching models:[/] {_format_fetch_error(e)}")
        sys.exit(1)


def _search_model(models: list, model_name: str):
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


def _pick_gguf_variant(model, quant_filter: str | None = None):
    from whichvlm.constants import QUANT_PREFERENCE_ORDER

    if not model.gguf_variants:
        return None

    if quant_filter:
        for v in model.gguf_variants:
            if v.quant_type.upper() == quant_filter.upper():
                return v
        console.print(
            f"[yellow]Warning:[/] {quant_filter} not available, using best match."
        )

    variant_map = {v.quant_type.upper(): v for v in model.gguf_variants}
    for qt in QUANT_PREFERENCE_ORDER:
        if qt in variant_map:
            return variant_map[qt]
    return model.gguf_variants[0]


def _find_gguf_variant(model: ModelInfo, quant_type: str) -> GGUFVariant | None:
    for variant in model.gguf_variants:
        if variant.quant_type.upper() == quant_type.upper():
            return variant
    return None


def _is_same_model_family(candidate: ModelInfo, selected: ModelInfo) -> bool:
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


def _has_compatible_parameter_count(candidate: ModelInfo, selected: ModelInfo) -> bool:
    if candidate.parameter_count <= 0 or selected.parameter_count <= 0:
        return True
    smaller = min(candidate.parameter_count, selected.parameter_count)
    larger = max(candidate.parameter_count, selected.parameter_count)
    return (larger / smaller) <= 2.0


def _resolve_ranked_gguf_for_run(
    selected_model: ModelInfo,
    selected_variant: GGUFVariant,
    models: list[ModelInfo],
    quant_filter: str | None = None,
) -> tuple[ModelInfo, GGUFVariant] | None:
    """Resolve a ranked GGUF candidate to a real GGUF repo/file for `run`.

    The ranker may synthesize GGUF variants for official safetensors-only repos
    so they can be scored realistically. `run` cannot execute those synthetic
    files directly, so it must find a real GGUF sibling before launching.
    """
    desired_quant = quant_filter or selected_variant.quant_type

    if selected_model.gguf_variants:
        variant = _find_gguf_variant(selected_model, desired_quant)
        return (selected_model, variant) if variant else None

    candidates: list[tuple[bool, int, int, ModelInfo, GGUFVariant]] = []
    for model in models:
        if not model.gguf_variants or not _is_same_model_family(model, selected_model):
            continue
        if not _has_compatible_parameter_count(model, selected_model):
            continue
        variant = _find_gguf_variant(model, desired_quant)
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


@app.command()
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
):
    """Download and run a model. Picks the best one if none specified."""
    import os
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("uv"):
        console.print("[red]uv is required.[/]")
        console.print(
            "Install: [bold]curl -LsSf https://astral.sh/uv/install.sh | sh[/]"
        )
        raise typer.Exit(code=1)

    with vlm_progress() as progress:
        task = progress.add_task("loading VLM packages...", total=None)
        models = _load_models(refresh)
        progress.remove_task(task)

    variant = None
    if model_name:
        model = _search_model(models, model_name)
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
            vision_workload=VisionWorkload(context_length=context_length),
        )
        if not results:
            console.print("[red]No runnable model found for your hardware.[/]")
            raise typer.Exit(code=1)
        skipped_gguf: list[str] = []
        model = None
        for ranked in results:
            if ranked.gguf_variant:
                resolved = _resolve_ranked_gguf_for_run(
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

    if variant is None:
        variant = _pick_gguf_variant(model, quant)
    if requires_image(model) and image is None:
        console.print("[red]Error:[/] VLM models require --image PATH.")
        raise typer.Exit(code=1)
    deps, script_type = resolve_model_deps(model, variant)
    try:
        if image is None:
            script = generate_run_script(model, variant, context_length, cpu_only)
        else:
            script = generate_run_script(
                model,
                variant,
                context_length,
                cpu_only,
                image_path=image,
            )
    except RuntimeUnsupportedError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)

    fmt = variant.quant_type if variant else script_type.upper()
    console.print(f"\n[bold green]Running {model.id}[/] [dim]({fmt})[/]")
    console.print(f"[dim]Setting up isolated env with: {', '.join(deps)}[/]\n")

    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="whichvlm_run_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(script)
        cmd = ["uv", "run", "--no-project"]
        for dep in deps:
            cmd.extend(["--with", dep])
        cmd.append(script_path)
        result = subprocess.run(cmd)
        raise typer.Exit(code=result.returncode)
    finally:
        os.unlink(script_path)


@app.command()
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
):
    """Print a ready-to-run Python script for a model."""
    from rich.syntax import Syntax

    with vlm_progress() as progress:
        task = progress.add_task("loading VLM packages...", total=None)
        models = _load_models(refresh)
        progress.remove_task(task)

    if model_name:
        model = _search_model(models, model_name)
    else:
        gguf_models = [m for m in models if m.gguf_variants]
        if not gguf_models:
            console.print("[red]No GGUF models found.[/]")
            raise typer.Exit(code=1)
        gguf_models.sort(key=lambda m: m.downloads, reverse=True)
        model = gguf_models[0]

    variant = _pick_gguf_variant(model, quant)
    deps, _ = resolve_model_deps(model, variant)
    if requires_image(model) and image is None:
        console.print("[red]Error:[/] VLM models require --image PATH.")
        raise typer.Exit(code=1)
    try:
        if image is None:
            code = generate_run_script(model, variant, 4096, False)
        else:
            code = generate_run_script(model, variant, 4096, False, image_path=image)
    except RuntimeUnsupportedError as e:
        console.print(f"[red]Error:[/] {e}")
        raise typer.Exit(code=1)

    dep_str = " ".join(f"--with {d}" for d in deps)
    console.print(f"\n[bold]{model.id}[/]")
    console.print(f"[dim]# Run directly:[/]  whichvlm run '{model.id}'")
    console.print(f"[dim]# Or manually:[/]   uv run --no-project {dep_str} script.py\n")
    console.print(Syntax(code, "python", theme="monokai"))


@app.command()
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
    """Show detected hardware information only."""
    _validate_gpu_flags(cpu_only, gpu, vram)

    from whichvlm.hardware.detector import detect_hardware
    from whichvlm.output.display import display_hardware

    with vlm_progress() as progress:
        task = progress.add_task("scanning silicon...", total=None)
        hw = detect_hardware()
        _apply_gpu_overrides(hw, cpu_only, gpu, vram)
        progress.remove_task(task)

    console.print()
    display_hardware(hw)
    console.print()


if __name__ == "__main__":
    app()
