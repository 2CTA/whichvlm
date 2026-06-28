import json
from io import StringIO

import httpx
import pytest
from rich.console import Console
from typer import Exit
from typer.testing import CliRunner

import whichvlm.cli as cli_mod
import whichvlm.__main__ as main_mod
import whichvlm.output.console as console_mod
from whichvlm.cli import (
    apply_memory_budgets,
    apply_gpu_overrides,
    auto_min_params_for_profile,
    fill_missing_published_at,
    format_fetch_error,
    include_vision_candidates,
    load_benchmark_index,
    merge_model_eval_benchmarks,
    parse_memory_amount,
    resolve_model_match,
    resolve_ranked_gguf_for_run,
    resolve_evidence_mode,
    resolve_fit_filter,
    resolve_speed_filter,
    select_gguf_variant,
    validate_evidence,
    validate_freshness_weight,
    vision_workload_for_profile,
    app,
)
from whichvlm.runtime import generate_run_script
from whichvlm.utils import current_version
from whichvlm.engine.types import CompatibilityResult
from whichvlm.hardware.types import GPUInfo, HardwareInfo, has_backend
from whichvlm.models.types import GGUFVariant, ModelArtifact, ModelInfo
from whichvlm.output.display import display_json


def hw_with_gpu(vram_gb: int) -> HardwareInfo:
    return HardwareInfo(
        gpus=[
            GPUInfo(
                name="GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * 1024**3,
                memory_bandwidth_gbps=1.0,
            )
        ],
        cpu_name="CPU",
        cpu_cores=1,
        ram_bytes=16 * 1024**3,
        disk_free_bytes=100 * 1024**3,
        os="linux",
    )


def test_auto_min_params_general_by_vram():
    assert auto_min_params_for_profile(hw_with_gpu(4), "general") == 2.0
    assert auto_min_params_for_profile(hw_with_gpu(6), "general") == 3.0
    assert auto_min_params_for_profile(hw_with_gpu(8), "general") == 5.0
    assert auto_min_params_for_profile(hw_with_gpu(12), "general") == 8.0
    assert auto_min_params_for_profile(hw_with_gpu(24), "general") == 10.0
    assert auto_min_params_for_profile(hw_with_gpu(32), "general") == 12.0


def test_auto_min_params_non_general_disabled():
    assert auto_min_params_for_profile(hw_with_gpu(24), "coding") is None


def test_auto_min_params_uses_usable_vram_budget():
    hw = hw_with_gpu(20)
    hw.gpus[0].usable_vram_bytes = int(19.0 * 1024**3)

    assert auto_min_params_for_profile(hw, "general") == 8.0


def test_auto_min_params_uses_ram_budget_for_shared_memory_gpu():
    hw = HardwareInfo(
        gpus=[
            GPUInfo(
                name="Apple M2",
                vendor="apple",
                vram_bytes=16 * 1024**3,
                usable_vram_bytes=15 * 1024**3,
                shared_memory=True,
            )
        ],
        ram_bytes=16 * 1024**3,
        ram_budget_bytes=4 * 1024**3,
    )

    assert auto_min_params_for_profile(hw, "general") == 2.0


def test_apply_gpu_overrides_accepts_multiple_simulated_gpus():
    hw = HardwareInfo(gpus=[], ram_bytes=64 * 1024**3, os="linux")

    apply_gpu_overrides(hw, cpu_only=False, gpu=["2x RTX 4090"], vram=None)

    assert len(hw.gpus) == 2
    assert all(gpu.vendor == "nvidia" for gpu in hw.gpus)
    assert all(gpu.vram_bytes == 24 * 1024**3 for gpu in hw.gpus)
    assert all(has_backend(gpu, "cuda") for gpu in hw.gpus)
    assert all(has_backend(gpu, "vulkan") for gpu in hw.gpus)


def test_json_simulated_nvidia_gpu_includes_backend_capabilities():
    hardware = HardwareInfo(gpus=[], ram_bytes=64 * 1024**3, os="linux")
    apply_gpu_overrides(hardware, cpu_only=False, gpu=["RTX 4090"], vram=None)

    buffer = StringIO()
    original_console = console_mod.console
    console_mod.console = Console(file=buffer, force_terminal=False)
    try:
        display_json([], hardware, details=True)
    finally:
        console_mod.console = original_console

    data = json.loads(buffer.getvalue())
    gpu = data["hardware"]["gpus"][0]
    assert gpu["vendor"] == "nvidia"
    assert {c["name"] for c in gpu["backend_capabilities"] if c["available"]} == {
        "cuda",
        "vulkan",
    }
    assert "ranking" in data
    assert "cache_snapshots" in data


def test_include_vision_candidates_by_profile():
    assert include_vision_candidates("vision") is True
    assert include_vision_candidates("ocr") is True
    assert include_vision_candidates("any") is True
    assert include_vision_candidates("general") is False
    assert include_vision_candidates("coding") is False


def test_vision_workload_for_profile_defaults_and_overrides():
    wl = vision_workload_for_profile("vision", context_length=8192)
    assert wl is not None
    assert wl.image_count == 1
    assert wl.image_size == 448
    assert wl.context_length == 8192
    assert vision_workload_for_profile("ocr") is not None

    custom = vision_workload_for_profile(
        "any", image_count=2, image_size=896, context_length=2048
    )
    assert custom is not None
    assert custom.image_count == 2
    assert custom.image_size == 896
    assert custom.context_length == 2048
    assert vision_workload_for_profile("general") is None


def test_fill_missing_published_at_updates_models():
    model = ModelInfo(
        id="Qwen/Qwen3-8B-AWQ",
        family_id="qwen3-8b",
        name="Qwen3-8B-AWQ",
        parameter_count=8_000_000_000,
        downloads=1,
        likes=1,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=0,
        vram_available_bytes=0,
    )

    async def fake_fetch(ids: list[str]) -> dict[str, str]:
        assert ids == ["Qwen/Qwen3-8B-AWQ"]
        return {"Qwen/Qwen3-8B-AWQ": "2026-03-05T08:00:00.000Z"}

    updated = fill_missing_published_at([model], [result], fake_fetch)
    assert updated is True
    assert model.published_at == "2026-03-05T08:00:00.000Z"


def test_version_option_prints_version_and_exits():
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert current_version() in result.stdout


def test_module_entrypoint_uses_cli_app():
    assert main_mod.app is app


def test_format_fetch_error_uses_exception_class_when_message_is_empty():
    class EmptyNetworkError(Exception):
        def __str__(self) -> str:
            return ""

    assert format_fetch_error(EmptyNetworkError()) == (
        "EmptyNetworkError with no detail from the network layer"
    )


def test_format_fetch_error_includes_status_and_url_for_empty_http_error():
    request = httpx.Request("GET", "https://huggingface.co/api/models")
    response = httpx.Response(429, request=request)
    error = httpx.HTTPStatusError("", request=request, response=response)

    assert format_fetch_error(error) == (
        "HTTPStatusError: HTTP 429 for https://huggingface.co/api/models"
    )


def test_load_benchmark_index_uses_stale_cache_after_fetch_failure(monkeypatch):
    cache_calls = []

    def fake_load_cache(*, allow_stale: bool = False):
        cache_calls.append(allow_stale)
        if allow_stale:
            return {"test/stale": 1.0}
        return None

    async def fail_fetch():
        raise httpx.ConnectError("offline")

    def fail_save(scores):
        raise AssertionError("failed benchmark fetch should not save cache")

    monkeypatch.setattr(
        "whichvlm.models.benchmark.load_benchmark_cache", fake_load_cache
    )
    monkeypatch.setattr("whichvlm.models.benchmark.fetch_benchmark_scores", fail_fetch)
    monkeypatch.setattr("whichvlm.models.benchmark.save_benchmark_cache", fail_save)

    assert load_benchmark_index(refresh=False) == {"test/stale": 1.0}
    assert cache_calls == [False, True]


def test_merge_model_eval_benchmarks_is_now_a_noop():

    model_direct_missing = ModelInfo(
        id="meta-llama/Llama-3.1-8B-Instruct",
        family_id="llama-3.1-8b",
        name="Llama-3.1-8B-Instruct",
        parameter_count=8_000_000_000,
        downloads=1,
        likes=1,
        benchmark_scores={"hf_eval": 66.4},
    )
    model_already_present = ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        family_id="qwen2.5-7b",
        name="Qwen2.5-7B-Instruct",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        benchmark_scores={"hf_eval": 70.0},
    )
    original = {"Qwen/Qwen2.5-7B-Instruct": 71.2}
    merged, injected = merge_model_eval_benchmarks(
        [model_direct_missing, model_already_present],
        original,
    )

    assert injected == 0
    assert merged is original or merged == original


    assert "meta-llama/Llama-3.1-8B-Instruct" not in merged


def test_validate_evidence_accepts_all_modes():
    assert validate_evidence("strict") == "strict"
    assert validate_evidence("base") == "base"
    assert validate_evidence("any") == "any"


def test_validate_evidence_rejects_unknown_mode():
    with pytest.raises(Exit):
        validate_evidence("foo")


def test_validate_freshness_weight_bounds():
    assert validate_freshness_weight(0.0) == 0.0
    assert validate_freshness_weight(1.0) == 1.0
    with pytest.raises(Exit):
        validate_freshness_weight(-0.1)
    with pytest.raises(Exit):
        validate_freshness_weight(1.1)


def test_resolve_evidence_mode_direct_alias_wins():
    assert resolve_evidence_mode("base", direct=True) == "strict"


def test_resolve_fit_filter_accepts_gpu_only_alias():
    assert resolve_fit_filter("any", gpu_only=False) == "any"
    assert resolve_fit_filter("gpu", gpu_only=False) == "full_gpu"
    assert resolve_fit_filter("full-gpu", gpu_only=False) == "full_gpu"
    assert resolve_fit_filter("full_gpu", gpu_only=False) == "full_gpu"
    assert resolve_fit_filter("any", gpu_only=True) == "full_gpu"


def test_resolve_fit_filter_rejects_unknown_mode():
    with pytest.raises(Exit):
        resolve_fit_filter("partial", gpu_only=False)


def test_resolve_speed_filter_presets_and_min_speed_override():
    assert resolve_speed_filter("any", min_speed=None) is None
    assert resolve_speed_filter("usable", min_speed=None) == 10.0
    assert resolve_speed_filter("fast", min_speed=None) == 30.0
    assert resolve_speed_filter("fast", min_speed=2.5) == 2.5


def test_resolve_speed_filter_rejects_unknown_mode():
    with pytest.raises(Exit):
        resolve_speed_filter("slowish", min_speed=None)


def test_parse_memory_amount_supports_gb_mb_and_percent():
    assert parse_memory_amount("1.5GB", option_name="--x") == int(1.5 * 1024**3)
    assert parse_memory_amount("512MB", option_name="--x") == 512 * 1024**2
    assert parse_memory_amount("8", option_name="--x") == 8 * 1024**3
    assert (
        parse_memory_amount("10%", option_name="--x", total_bytes=20 * 1024**3)
        == 2 * 1024**3
    )


def test_apply_memory_budgets_sets_vram_headroom_and_ram_budget():
    hw = hw_with_gpu(16)

    apply_memory_budgets(hw, vram_headroom="1GB", ram_budget="8GB")

    assert hw.gpus[0].vram_bytes == 16 * 1024**3
    assert hw.gpus[0].usable_vram_bytes == 15 * 1024**3
    assert hw.ram_budget_bytes == 8 * 1024**3
    assert any("VRAM headroom" in note for note in hw.budget_notes)
    assert any("RAM budget" in note for note in hw.budget_notes)


def test_apply_memory_budgets_validates_vram_headroom_without_gpus():
    hw = HardwareInfo(gpus=[], ram_bytes=16 * 1024**3)

    with pytest.raises(Exit):
        apply_memory_budgets(hw, vram_headroom="nope", ram_budget=None)


def test_apply_memory_budgets_accepts_valid_noop_vram_headroom_without_gpus():
    hw = HardwareInfo(gpus=[], ram_bytes=16 * 1024**3)

    apply_memory_budgets(hw, vram_headroom="10%", ram_budget=None)

    assert hw.gpus == []
    assert hw.ram_budget_bytes is None


def test_main_passes_gpu_only_fit_filter(monkeypatch):
    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        published_at="2026-01-01T00:00:00.000Z",
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["fit_filter"] = kwargs.get("fit_filter")
        captured["task_profile"] = kwargs.get("task_profile")
        captured["vision_workload"] = kwargs.get("vision_workload")
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=None,
                can_run=True,
                vram_required_bytes=4 * 1024**3,
                vram_available_bytes=8 * 1024**3,
                fit_type="full_gpu",
                quality_score=80.0,
            )
        ]

    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware", lambda: hw_with_gpu(8)
    )
    monkeypatch.setattr("whichvlm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichvlm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichvlm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichvlm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr(
        "whichvlm.output.display.display_ranking",
        lambda results, **kwargs: None,
    )

    result = CliRunner().invoke(app, ["--gpu-only"])

    assert result.exit_code == 0
    assert captured["fit_filter"] == "full_gpu"
    assert captured["task_profile"] == "vision"
    assert captured["vision_workload"].image_count == 1


def test_main_passes_speed_preset_and_default_runtime_columns(monkeypatch):
    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        published_at="2026-01-01T00:00:00.000Z",
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["min_speed"] = kwargs.get("min_speed")
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=None,
                can_run=True,
                vram_required_bytes=4 * 1024**3,
                vram_available_bytes=8 * 1024**3,
                fit_type="full_gpu",
                estimated_tok_per_sec=8.0,
                quality_score=80.0,
            )
        ]

    def fake_display_ranking(results, **kwargs):
        captured["show_status"] = kwargs.get("show_status")

    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware", lambda: hw_with_gpu(8)
    )
    monkeypatch.setattr("whichvlm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichvlm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichvlm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichvlm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr("whichvlm.output.display.display_ranking", fake_display_ranking)

    result = CliRunner().invoke(app, ["--speed", "usable"])

    assert result.exit_code == 0
    assert captured["min_speed"] == 10.0
    assert captured["show_status"] is True


def test_main_details_flag_restores_metadata_columns(monkeypatch):
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        return []

    def fake_display_ranking(results, **kwargs):
        captured["show_status"] = kwargs.get("show_status")

    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware", lambda: hw_with_gpu(8)
    )
    monkeypatch.setattr("whichvlm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichvlm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichvlm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichvlm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr("whichvlm.output.display.display_ranking", fake_display_ranking)

    result = CliRunner().invoke(app, ["--details", "--min-params", "1"])

    assert result.exit_code == 0
    assert captured["show_status"] is False


def test_main_markdown_alias_dispatches_markdown_output(monkeypatch):
    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=1,
        likes=1,
        published_at="2026-01-01T00:00:00.000Z",
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=None,
                can_run=True,
                vram_required_bytes=4 * 1024**3,
                vram_available_bytes=8 * 1024**3,
                fit_type="full_gpu",
                estimated_tok_per_sec=12.0,
                quality_score=80.0,
            )
        ]

    def fake_display_markdown(results, hardware, **kwargs):
        captured["called"] = True
        captured["show_status"] = kwargs.get("show_status")

    def fail_display_hardware(hardware):
        raise AssertionError("markdown output should not render Rich hardware panel")

    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware", lambda: hw_with_gpu(8)
    )
    monkeypatch.setattr("whichvlm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichvlm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichvlm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichvlm.output.display.display_markdown", fake_display_markdown
    )
    monkeypatch.setattr(
        "whichvlm.output.display.display_hardware", fail_display_hardware
    )

    result = CliRunner().invoke(app, ["-m"])

    assert result.exit_code == 0
    assert captured["called"] is True
    assert captured["show_status"] is True


def test_main_json_and_markdown_are_mutually_exclusive():
    result = CliRunner().invoke(app, ["--json", "--markdown"])

    assert result.exit_code == 1
    assert "--json and --markdown are mutually exclusive" in result.stdout


def test_main_empty_gpu_only_result_shows_fit_message(monkeypatch):
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        return []

    def fake_display_ranking(results, **kwargs):
        captured["empty_message"] = kwargs.get("empty_message")

    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware", lambda: hw_with_gpu(8)
    )
    monkeypatch.setattr("whichvlm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichvlm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichvlm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(
        "whichvlm.output.display.display_hardware", lambda hardware: None
    )
    monkeypatch.setattr("whichvlm.output.display.display_ranking", fake_display_ranking)

    result = CliRunner().invoke(app, ["--fit", "full-gpu", "--min-params", "1"])

    assert result.exit_code == 0
    assert "No full-GPU models found" in captured["empty_message"]


def test_main_json_smoke_profile_vision(monkeypatch):
    model = ModelInfo(
        id="org/Test-VL-7B",
        family_id="test-vl-7b",
        name="Test-VL-7B",
        parameter_count=7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        downloads=1,
        likes=1,
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["task_profile"] = kwargs.get("task_profile")
        captured["vision_workload"] = kwargs.get("vision_workload")
        return [
            CompatibilityResult(
                model=model,
                gguf_variant=None,
                can_run=True,
                vram_required_bytes=4 * 1024**3,
                vram_available_bytes=8 * 1024**3,
                fit_type="full_gpu",
                quality_score=80.0,
            )
        ]

    def fake_display_json(results, hardware, details=False):
        captured["json_called"] = True
        captured["json_results"] = results
        captured["json_details"] = details

    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware", lambda: hw_with_gpu(8)
    )
    monkeypatch.setattr("whichvlm.models.cache.load_cache", lambda: [])
    monkeypatch.setattr("whichvlm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichvlm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr("whichvlm.output.display.display_json", fake_display_json)

    result = CliRunner().invoke(
        app,
        ["--profile", "vision", "--json", "--image-count", "2", "--image-size", "896"],
    )

    assert result.exit_code == 0
    assert captured["json_called"] is True
    assert captured["json_details"] is False
    assert captured["task_profile"] == "vision"
    assert captured["vision_workload"].image_count == 2
    assert captured["vision_workload"].image_size == 896

    captured.clear()
    result = CliRunner().invoke(app, ["--json", "--details"])
    assert result.exit_code == 0
    assert captured["json_details"] is True


def test_hardware_command_smoke(monkeypatch):
    captured: dict[str, object] = {}

    def fake_display_hardware(hw):
        captured["hardware"] = hw

    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware", lambda: hw_with_gpu(8)
    )
    monkeypatch.setattr("whichvlm.output.display.display_hardware", fake_display_hardware)

    result = CliRunner().invoke(app, ["hardware"])

    assert result.exit_code == 0
    assert captured["hardware"].gpus[0].name == "GPU"


def test_hardware_command_simulated_apple_silicon(monkeypatch):
    captured: dict[str, object] = {}

    def fake_display_hardware(hw):
        captured["hardware"] = hw

    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware",
        lambda: HardwareInfo(gpus=[], ram_bytes=16 * 1024**3, os="darwin"),
    )
    monkeypatch.setattr("whichvlm.output.display.display_hardware", fake_display_hardware)

    result = CliRunner().invoke(app, ["hardware", "--gpu", "Apple M3 Max"])

    assert result.exit_code == 0
    gpu = captured["hardware"].gpus[0]
    assert gpu.vendor == "apple"
    assert gpu.shared_memory is True
    assert has_backend(gpu, "metal")
    assert has_backend(gpu, "mlx")


def test_plan_no_model_found_shows_error(monkeypatch):
    monkeypatch.setattr("whichvlm.models.cache.load_cache", lambda: [])
    runner = CliRunner()
    result = runner.invoke(app, ["plan", "nonexistent_model_xyz_999"])
    assert result.exit_code != 0
    assert "No model found" in result.stdout


def test_plan_display_plan_renders_tables():
    from io import StringIO

    from rich.console import Console

    from whichvlm.output.display import display_plan

    model = ModelInfo(
        id="test-org/Test-Model-7B-GGUF",
        family_id="test-7b",
        name="Test-Model-7B",
        parameter_count=7_000_000_000,
        architecture="llama",
        context_length=4096,
        license="mit",
        downloads=100,
        likes=10,
    )
    buf = StringIO()
    import whichvlm.output.console as console_mod

    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False, width=120)
    try:
        display_plan(model, context_length=4096, target_quant="Q4_K_M")
    finally:
        console_mod.console = orig_console

    output = buf.getvalue()
    assert "Model Info" in output
    assert "VRAM Required" in output
    assert "GPU Compatibility" in output
    assert "test-org/Test-Model-7B-GGUF" in output


def test_plan_display_plan_json_outputs_valid_json():

    import json as json_mod
    from io import StringIO

    from rich.console import Console

    from whichvlm.output.display import display_plan_json

    model = ModelInfo(
        id="test-org/Test-Model-7B-GGUF",
        family_id="test-7b",
        name="Test-Model-7B",
        parameter_count=7_000_000_000,
        architecture="llama",
        context_length=4096,
        license="mit",
        downloads=100,
        likes=10,
    )

    buf = StringIO()
    import whichvlm.output.console as console_mod

    orig_console = console_mod.console
    console_mod.console = Console(file=buf, force_terminal=False)
    try:
        display_plan_json(model, context_length=4096, target_quant="Q4_K_M")
    finally:
        console_mod.console = orig_console
    raw = buf.getvalue().strip()
    data = json_mod.loads(raw)
    assert data["model"]["id"] == "test-org/Test-Model-7B-GGUF"
    assert "vram_by_quant" in data
    assert "gpu_compatibility" in data
    assert data["target_quant"] == "Q4_K_M"


def test_hardware_plan_scores_target_gpu(monkeypatch):
    captured: dict[str, object] = {}
    model = ModelInfo(
        id="test-org/Test-Vision-7B",
        family_id="test-vision-7b",
        name="Test-Vision-7B",
        parameter_count=7_000_000_000,
        architecture="qwen2_vl",
        context_length=8192,
        hf_pipeline_tag="image-text-to-text",
        downloads=10,
    )

    def fake_display_json(results, hardware, details=False):
        captured["results"] = results
        captured["hardware"] = hardware
        captured["details"] = details

    monkeypatch.setattr(cli_mod, "load_model_catalog", lambda *args, **kwargs: [model])
    monkeypatch.setattr(cli_mod, "load_benchmark_index", lambda refresh: {})
    monkeypatch.setattr("whichvlm.output.display.display_json", fake_display_json)

    result = CliRunner().invoke(
        app,
        ["hardware-plan", "RTX 4070", "--json", "--top", "1", "--context-length", "4096"],
    )

    assert result.exit_code == 0
    hardware = captured["hardware"]
    results = captured["results"]
    assert hardware.gpus[0].name == "RTX 4070"
    assert results[0].model.id == "test-org/Test-Vision-7B"
    assert results[0].can_run is True
    assert captured["details"] is True


def make_model(model_id="org/Test-7B-GGUF", downloads=100, gguf_variants=None):
    return ModelInfo(
        id=model_id,
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        downloads=downloads,
        likes=10,
        gguf_variants=gguf_variants or [],
    )


def test_search_model_exact_match():
    models = [make_model("org/Llama-8B"), make_model("org/Qwen-7B")]
    result = resolve_model_match(models, "org/Llama-8B")
    assert result.id == "org/Llama-8B"


def test_search_model_endswith_match():
    models = [make_model("org/Llama-8B"), make_model("org/Qwen-7B")]
    result = resolve_model_match(models, "Llama-8B")
    assert result.id == "org/Llama-8B"


def test_search_model_term_match():
    models = [make_model("org/Llama-3.1-8B-GGUF"), make_model("org/Qwen-7B")]
    result = resolve_model_match(models, "llama 8b")
    assert result.id == "org/Llama-3.1-8B-GGUF"


def test_search_model_not_found():
    models = [make_model("org/Llama-8B")]
    with pytest.raises(Exit):
        resolve_model_match(models, "nonexistent_xyz")


def test_pick_gguf_variant_by_preference():
    variants = [
        GGUFVariant(filename="q2.gguf", quant_type="Q2_K", file_size_bytes=1000),
        GGUFVariant(filename="q4km.gguf", quant_type="Q4_K_M", file_size_bytes=2000),
    ]
    model = make_model(gguf_variants=variants)
    result = select_gguf_variant(model)
    assert result.quant_type == "Q4_K_M"


def test_pick_gguf_variant_with_filter():
    variants = [
        GGUFVariant(filename="q2.gguf", quant_type="Q2_K", file_size_bytes=1000),
        GGUFVariant(filename="q4km.gguf", quant_type="Q4_K_M", file_size_bytes=2000),
    ]
    model = make_model(gguf_variants=variants)
    result = select_gguf_variant(model, quant_filter="Q2_K")
    assert result.quant_type == "Q2_K"


def test_pick_gguf_variant_no_variants():
    model = make_model(gguf_variants=[])
    result = select_gguf_variant(model)
    assert result is None


def test_resolve_ranked_synthetic_gguf_to_real_repo():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
        downloads=50_000,
    )
    real_gguf = ModelInfo(
        id="unsloth/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=200_000,
        base_model="Qwen/Qwen3.6-27B",
        gguf_variants=[
            GGUFVariant(
                filename="Qwen3.6-27B-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = resolve_ranked_gguf_for_run(selected, synthetic, [selected, real_gguf])

    assert resolved is not None
    model, variant = resolved
    assert model.id == "unsloth/Qwen3.6-27B-GGUF"
    assert variant.filename == "Qwen3.6-27B-Q4_K_M.gguf"


def test_resolve_ranked_synthetic_gguf_prefers_exact_quant():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    q5_only = ModelInfo(
        id="converter/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=1_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="q5.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=18_000_000_000,
            )
        ],
    )
    q4_match = ModelInfo(
        id="smaller/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=10,
        gguf_variants=[
            GGUFVariant(
                filename="q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = resolve_ranked_gguf_for_run(
        selected,
        synthetic,
        [selected, q5_only, q4_match],
    )

    assert resolved is not None
    model, variant = resolved
    assert model.id == "smaller/Qwen3.6-27B-GGUF"
    assert variant.quant_type == "Q4_K_M"


def test_resolve_ranked_synthetic_gguf_rejects_quant_mismatch():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    q5_only = ModelInfo(
        id="converter/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=1_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="q5.gguf",
                quant_type="Q5_K_M",
                file_size_bytes=18_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    resolved = resolve_ranked_gguf_for_run(selected, synthetic, [selected, q5_only])

    assert resolved is None


def test_resolve_ranked_synthetic_gguf_without_real_repo_returns_none():
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
    )
    unrelated = ModelInfo(
        id="other/Model-7B-GGUF",
        family_id="model-7b",
        name="Model-7B-GGUF",
        parameter_count=7_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="other.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )

    assert (
        resolve_ranked_gguf_for_run(selected, synthetic, [selected, unrelated]) is None
    )


def test_resolve_ranked_synthetic_gguf_rejects_size_mismatch():
    selected = ModelInfo(
        id="deepseek-ai/DeepSeek-V4-Flash",
        family_id="deepseek-v4-flash",
        name="DeepSeek-V4-Flash",
        parameter_count=158_000_000_000,
    )
    mtp_head = ModelInfo(
        id="converter/deepseek-v4-flash-mtp-gguf",
        family_id="deepseek-v4-flash",
        name="DeepSeek-V4-Flash-MTP-GGUF",
        parameter_count=6_600_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="mtp.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="DeepSeek-V4-Flash.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=90_000_000_000,
    )

    resolved = resolve_ranked_gguf_for_run(selected, synthetic, [selected, mtp_head])

    assert resolved is None


def test_run_reports_missing_model(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(cli_mod, "load_model_catalog", lambda refresh: [])

    runner = CliRunner()
    result = runner.invoke(app, ["run", "some-model"])

    assert result.exit_code != 0
    assert "No model found matching 'some-model'" in result.stdout


def test_transformers_chat_script_passes_tokenizer_mapping_to_generate():
    model = make_model(model_id="org/Test-7B")

    script = generate_run_script(
        model, variant=None, context_length=4096, cpu_only=False
    )

    assert "return_dict=True" in script
    assert "kwargs=dict(**inputs, max_new_tokens=512, streamer=streamer)" in script
    assert "kwargs=dict(input_ids=inputs" not in script


def test_transformers_chat_script_provides_disk_offload_folder():
    model = make_model(model_id="org/Test-7B")

    script = generate_run_script(
        model, variant=None, context_length=4096, cpu_only=False
    )

    assert 'tempfile.mkdtemp(prefix="whichvlm_transformers_offload_")' in script
    assert "offload_folder=offload_folder" in script
    assert "shutil.rmtree(offload_folder, ignore_errors=True)" in script


def test_run_auto_pick_resolves_ranked_gguf_before_launch(monkeypatch):
    selected = ModelInfo(
        id="Qwen/Qwen3.6-27B",
        family_id="qwen3-27b",
        name="Qwen3.6-27B",
        parameter_count=27_000_000_000,
        downloads=50_000,
    )
    real_gguf = ModelInfo(
        id="unsloth/Qwen3.6-27B-GGUF",
        family_id="qwen3-27b",
        name="Qwen3.6-27B-GGUF",
        parameter_count=27_000_000_000,
        downloads=200_000,
        base_model="Qwen/Qwen3.6-27B",
        gguf_variants=[
            GGUFVariant(
                filename="q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=16_000_000_000,
            )
        ],
    )
    synthetic = GGUFVariant(
        filename="Qwen3.6-27B.Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=16_000_000_000,
    )
    captured: dict[str, object] = {}

    def fake_rank_models(models, hardware, **kwargs):
        captured["quant_filter"] = kwargs.get("quant_filter")
        return [
            CompatibilityResult(
                model=selected,
                gguf_variant=synthetic,
                can_run=True,
                vram_required_bytes=0,
                vram_available_bytes=0,
                quality_score=90.0,
            )
        ]

    def fake_generate_run_script(model, variant, context_length, cpu_only):
        captured["model_id"] = model.id
        captured["variant"] = variant
        return "print('ok')"

    class Completed:
        returncode = 0

    def fake_run(cmd):
        captured["cmd"] = cmd
        return Completed()

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(
        cli_mod, "load_model_catalog", lambda refresh: [selected, real_gguf]
    )
    monkeypatch.setattr(
        "whichvlm.hardware.detector.detect_hardware", lambda: hw_with_gpu(8)
    )
    monkeypatch.setattr("whichvlm.models.benchmark.load_benchmark_cache", lambda: {})
    monkeypatch.setattr("whichvlm.engine.ranker.rank_models", fake_rank_models)
    monkeypatch.setattr(cli_mod, "generate_run_script", fake_generate_run_script)
    monkeypatch.setattr("subprocess.run", fake_run)

    result = CliRunner().invoke(app, ["run", "--quant", "Q4_K_M"])

    assert result.exit_code == 0
    assert captured["quant_filter"] == "Q4_K_M"
    assert captured["model_id"] == "unsloth/Qwen3.6-27B-GGUF"
    assert captured["variant"].filename == "q4.gguf"
    assert "llama-cpp-python" in captured["cmd"]
    assert "transformers" not in captured["cmd"]


def test_run_vlm_requires_image(monkeypatch):
    model = ModelInfo(
        id="org/Test-VL-7B",
        family_id="test-vl",
        name="Test-VL-7B",
        parameter_count=7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
    )

    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(cli_mod, "load_model_catalog", lambda refresh: [model])

    result = CliRunner().invoke(app, ["run", "org/Test-VL-7B"])

    assert result.exit_code == 1
    assert "VLM models require --image PATH" in result.stdout


def test_snippet_no_model_found(monkeypatch):
    monkeypatch.setattr(cli_mod, "load_model_catalog", lambda refresh: [])
    runner = CliRunner()
    result = runner.invoke(app, ["snippet", "nonexistent_model_xyz_999"])
    assert result.exit_code != 0
    assert "No model found" in result.stdout


def render_json_output(
    result: CompatibilityResult,
    hardware: HardwareInfo,
    details: bool = False,
) -> dict:
    buffer = StringIO()
    original_console = console_mod.console
    console_mod.console = Console(file=buffer, force_terminal=False)
    try:
        display_json([result], hardware, details=details)
    finally:
        console_mod.console = original_console
    return json.loads(buffer.getvalue().strip())


def json_output_case() -> tuple[CompatibilityResult, HardwareInfo]:
    model = ModelInfo(
        id="test-org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
        base_models=["base/Test-7B"],
        artifacts=[
            ModelArtifact(
                repo_id="test-org/Test-7B",
                format="mlx",
                quantization="MLX",
                access="gated",
                backend_support=["mlx", "metal"],
                source_kind="mlx_variant",
            )
        ],
        downloads=100,
        likes=10,
    )
    result = CompatibilityResult(
        model=model,
        gguf_variant=None,
        can_run=True,
        vram_required_bytes=8_000_000_000,
        vram_available_bytes=24_000_000_000,
        vram_required_range_bytes=(7_000_000_000, 10_000_000_000),
        vram_confidence="medium",
        vram_breakdown_bytes={
            "weights": 6_000_000_000,
            "kv_cache": 500_000_000,
            "activations": 700_000_000,
            "vision": 0,
            "runtime_overhead": 800_000_000,
        },
        vram_notes=["KV cache uses parameter-count fallback"],
        quality_score=55.0,
        benchmark_status="estimated",
        benchmark_source="line_interp",
        benchmark_confidence=0.34,
    )
    hw = HardwareInfo(
        gpus=[],
        cpu_name="Test CPU",
        cpu_cores=8,
        ram_budget_bytes=32 * 1024**3,
        ram_bytes=64 * 1024**3,
        disk_free_bytes=500 * 1024**3,
        os="linux",
        budget_notes=["RAM budget: 32.0 GB"],
    )
    return result, hw


def test_json_output_defaults_to_compact():
    result, hardware = json_output_case()
    compact = render_json_output(result, hardware)
    compact_entry = compact["models"][0]

    assert compact_entry["model_id"] == "test-org/Test-7B"
    assert compact_entry["benchmark_source"] == "line_interp"
    assert compact_entry["vram_required_range_bytes"] == [
        7_000_000_000,
        10_000_000_000,
    ]
    assert compact_entry["vram_confidence"] == "medium"
    assert "artifacts" not in compact_entry
    assert "lineage" not in compact_entry
    assert "budget_notes" not in compact["hardware"]


def test_json_output_includes_diagnostics_when_requested():
    result, hardware = json_output_case()
    data = render_json_output(result, hardware, details=True)
    entry = data["models"][0]
    artifact = entry["artifacts"][0]

    assert data["hardware"]["ram_budget_bytes"] == 32 * 1024**3
    assert data["hardware"]["budget_notes"] == ["RAM budget: 32.0 GB"]
    assert entry["benchmark_status"] == "estimated"
    assert entry["benchmark_source"] == "line_interp"
    assert entry["benchmark_confidence"] == 0.34
    assert entry["vram_breakdown_bytes"]["weights"] == 6_000_000_000
    assert entry["vram_notes"] == ["KV cache uses parameter-count fallback"]
    assert entry["base_models"] == ["base/Test-7B"]
    assert artifact["format"] == "mlx"
    assert artifact["access"] == "gated"
    assert artifact["backend_support"] == ["mlx", "metal"]
    assert entry["lineage"]["base_model_ids"] == ["base/Test-7B"]
