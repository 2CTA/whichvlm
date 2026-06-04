"""Tests for VRAM estimation."""

from whichvlm.engine.vram import estimate_kv_cache, estimate_vram
from whichvlm.engine.workload import VisionWorkload
from whichvlm.models.types import GGUFVariant, ModelComponent, ModelInfo


def _make_model(params: int, **kwargs) -> ModelInfo:
    return ModelInfo(
        id="test/model",
        family_id="test/model",
        name="model",
        parameter_count=params,
        **kwargs,
    )


def test_estimate_vram_gguf_variant():
    model = _make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )
    vram = estimate_vram(model, variant, context_length=4096)
    # Should be: 4GB weights + KV cache + activation + framework overhead
    assert vram > 4_000_000_000
    assert vram < 7_000_000_000  # should be well under FP16 size


def test_estimate_vram_fp16_fallback():
    model = _make_model(7_000_000_000)
    vram = estimate_vram(model, None, context_length=4096)
    # FP16: 7B * 2 = 14GB + overhead
    assert vram > 14_000_000_000
    assert vram < 20_000_000_000


def test_estimate_vram_increases_with_context():
    model = _make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )
    vram_4k = estimate_vram(model, variant, context_length=4096)
    vram_32k = estimate_vram(model, variant, context_length=32768)
    assert vram_32k > vram_4k


def test_estimate_kv_cache_scales_with_params():
    small = _make_model(1_000_000_000)
    large = _make_model(70_000_000_000)
    kv_small = estimate_kv_cache(small, 4096)
    kv_large = estimate_kv_cache(large, 4096)
    assert kv_large > kv_small


def test_estimate_vram_small_model():
    model = _make_model(500_000_000)  # 0.5B
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=300_000_000
    )
    vram = estimate_vram(model, variant, context_length=4096)
    # Should be reasonable for a tiny model
    assert vram > 300_000_000
    assert vram < 3_000_000_000


def test_vision_workload_increases_vram_predictably():
    model = _make_model(7_000_000_000, hf_pipeline_tag="image-text-to-text")
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )

    text_only = estimate_vram(model, variant, context_length=4096)
    one_image = estimate_vram(
        model,
        variant,
        context_length=4096,
        vision_workload=VisionWorkload(image_count=1, image_size=448),
    )
    two_large_images = estimate_vram(
        model,
        variant,
        context_length=4096,
        vision_workload=VisionWorkload(image_count=2, image_size=896),
    )

    assert one_image > text_only
    assert two_large_images > one_image


def test_vision_workload_does_not_change_text_model_vram():
    model = _make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )

    text_only = estimate_vram(model, variant, context_length=4096)
    with_image_workload = estimate_vram(
        model,
        variant,
        context_length=4096,
        vision_workload=VisionWorkload(image_count=1, image_size=448),
    )

    assert with_image_workload == text_only


def test_vision_component_sizes_increase_vlm_overhead():
    small = _make_model(
        7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        components=[
            ModelComponent(
                role="vision_encoder",
                repo_id="test/model",
                parameter_count=300_000_000,
            ),
            ModelComponent(
                role="projector",
                repo_id="test/model",
                parameter_count=50_000_000,
            ),
        ],
    )
    large = _make_model(
        7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        components=[
            ModelComponent(
                role="vision_encoder",
                repo_id="test/model",
                parameter_count=1_000_000_000,
            ),
            ModelComponent(
                role="projector",
                repo_id="test/model",
                parameter_count=200_000_000,
            ),
        ],
    )
    workload = VisionWorkload(image_count=1, image_size=448)

    assert estimate_vram(large, None, vision_workload=workload) > estimate_vram(
        small,
        None,
        vision_workload=workload,
    )
