from engine.vram import estimate_kv_cache, estimate_vram, estimate_vram_details
from engine.workload import VisionWorkload
from models.types import (
    GGUFVariant,
    ModelCapabilities,
    ModelComponent,
    ModelInfo,
)


def make_model(params: int, model_id: str = "test/model", **kwargs) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=model_id,
        name="model",
        parameter_count=params,
        **kwargs,
    )


def vlm_calibration_model(
    *,
    params: int = 7_000_000_000,
    architecture: str = "qwen2_5_vl",
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_type: str | None = None,
) -> ModelInfo:
    return make_model(
        params,
        model_id=model_id,
        hf_pipeline_tag="image-text-to-text",
        architecture=architecture,
        model_format="safetensors",
        quantization_type=quantization_type,
        layer_count=28,
        hidden_size=3584,
        intermediate_size=18944,
        attention_heads=28,
        kv_heads=4,
        dtype="bfloat16",
        vision_layer_count=32,
        vision_hidden_size=1280,
        vision_intermediate_size=3420,
        vision_attention_heads=16,
        projector_hidden_size=3584,
        patch_size=14,
        spatial_merge_size=2,
        components=[
            ModelComponent(
                role="vision_encoder",
                repo_id=model_id,
                parameter_count=300_000_000,
                quantization="FP16",
            ),
            ModelComponent(
                role="projector",
                repo_id=model_id,
                parameter_count=50_000_000,
                quantization="FP16",
            ),
        ],
    )


def qwen3_moe_model() -> ModelInfo:
    return make_model(
        30_000_000_000,
        model_id="Qwen/Qwen3-30B-A3B",
        parameter_count_active=3_000_000_000,
        architecture="qwen3_5moe",
        is_moe=True,
        model_format="gguf",
        layer_count=48,
        hidden_size=2048,
        intermediate_size=6144,
        attention_heads=32,
        kv_heads=4,
        dtype="bfloat16",
    )


def qwen3_dense_model() -> ModelInfo:
    return make_model(
        30_000_000_000,
        model_id="Qwen/Qwen3-30B",
        architecture="qwen3",
        model_format="gguf",
        layer_count=48,
        hidden_size=2048,
        intermediate_size=6144,
        attention_heads=32,
        kv_heads=4,
        dtype="bfloat16",
    )


def test_estimate_vram_gguf_variant():
    model = make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )
    vram = estimate_vram(model, variant, context_length=4096)

    assert vram > 4_000_000_000
    assert vram < 7_000_000_000


def test_estimate_vram_fp16_fallback():
    model = make_model(7_000_000_000)
    vram = estimate_vram(model, None, context_length=4096)

    assert vram > 14_000_000_000
    assert vram < 20_000_000_000


def test_estimate_vram_increases_with_context():
    model = make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )
    vram_4k = estimate_vram(model, variant, context_length=4096)
    vram_32k = estimate_vram(model, variant, context_length=32768)
    assert vram_32k > vram_4k


def test_estimate_kv_cache_scales_with_params():
    small = make_model(1_000_000_000)
    large = make_model(70_000_000_000)
    kv_small = estimate_kv_cache(small, 4096)
    kv_large = estimate_kv_cache(large, 4096)
    assert kv_large > kv_small


def test_estimate_kv_cache_uses_architecture_dimensions():
    dense = make_model(
        70_000_000_000,
        layer_count=32,
        hidden_size=4096,
        attention_heads=32,
        kv_heads=32,
        dtype="bfloat16",
    )
    grouped_query = make_model(
        70_000_000_000,
        layer_count=32,
        hidden_size=4096,
        attention_heads=32,
        kv_heads=8,
        dtype="bfloat16",
    )

    assert estimate_kv_cache(dense, 4096) == 2_147_483_648
    assert estimate_kv_cache(grouped_query, 4096) == 536_870_912


def test_vram_details_returns_components_range_and_confidence():
    model = make_model(
        7_000_000_000,
        architecture="llama",
        model_format="gguf",
        layer_count=32,
        hidden_size=4096,
        attention_heads=32,
        kv_heads=8,
        dtype="bfloat16",
    )
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )

    estimate = estimate_vram_details(model, variant, context_length=4096)

    assert estimate.confidence == "high"
    assert estimate.notes == []
    assert estimate.components.weights == 4_000_000_000
    assert estimate.components.kv_cache == 536_870_912
    assert estimate.components.runtime_overhead > 800_000_000
    assert estimate.lower_bytes < estimate.required_bytes < estimate.upper_bytes


def test_full_metadata_without_calibration_returns_medium_confidence():
    model = make_model(
        7_000_000_000,
        architecture="gemma",
        model_format="safetensors",
        layer_count=28,
        hidden_size=3072,
        intermediate_size=24576,
        attention_heads=16,
        kv_heads=16,
        dtype="bfloat16",
    )

    estimate = estimate_vram_details(model, None, context_length=4096)

    assert estimate.confidence == "medium"
    assert estimate.notes == [
        "no matching peak-memory calibration "
        "(transformers, safetensors, FP16, context=4096)"
    ]


def test_missing_architecture_metadata_returns_low_confidence_range():
    model = make_model(7_000_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=4_000_000_000
    )

    estimate = estimate_vram_details(model, variant, context_length=4096)

    assert estimate.confidence == "low"
    assert "KV cache uses parameter-count fallback" in estimate.notes
    assert estimate.upper_bytes > int(estimate.required_bytes * 1.5)


def test_moe_vram_stores_total_weights_but_uses_active_activation_cost():
    base = dict(
        architecture="mixtral",
        model_format="gguf",
        is_moe=True,
        layer_count=32,
        hidden_size=4096,
        attention_heads=32,
        kv_heads=8,
        dtype="bfloat16",
    )
    small_active = make_model(
        45_000_000_000,
        parameter_count_active=12_000_000_000,
        **base,
    )
    large_active = make_model(
        45_000_000_000,
        parameter_count_active=24_000_000_000,
        **base,
    )
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=24_000_000_000,
    )

    small_estimate = estimate_vram_details(small_active, variant)
    large_estimate = estimate_vram_details(large_active, variant)

    assert small_estimate.components.weights == large_estimate.components.weights
    assert large_estimate.components.activations > small_estimate.components.activations


def test_vision_architecture_metadata_changes_image_token_cost():
    base = dict(
        hf_pipeline_tag="image-text-to-text",
        architecture="qwen2vl",
        model_format="safetensors",
        layer_count=28,
        hidden_size=3584,
        attention_heads=28,
        kv_heads=4,
        dtype="bfloat16",
        vision_layer_count=32,
        vision_hidden_size=1280,
        projector_hidden_size=3584,
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
    patch_14 = make_model(7_000_000_000, patch_size=14, **base)
    patch_28 = make_model(7_000_000_000, patch_size=28, **base)
    workload = VisionWorkload(image_count=1, image_size=448)

    small_patch = estimate_vram_details(patch_14, None, vision_workload=workload)
    large_patch = estimate_vram_details(patch_28, None, vision_workload=workload)

    assert small_patch.confidence == "high"
    assert small_patch.components.vision > large_patch.components.vision


def test_quantized_transformers_variant_uses_quantized_calibration():
    awq = vlm_calibration_model(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        quantization_type="AWQ",
    )
    fp16 = vlm_calibration_model()
    workload = VisionWorkload(image_count=1, image_size=448)

    awq_estimate = estimate_vram_details(awq, None, vision_workload=workload)
    fp16_estimate = estimate_vram_details(fp16, None, vision_workload=workload)

    assert awq_estimate.confidence == "high"
    assert awq_estimate.notes == []
    assert awq_estimate.required_bytes < fp16_estimate.required_bytes


def test_gguf_vlm_calibration_raises_confidence_for_architecture_alias():
    variant = GGUFVariant(
        filename="qwen2.5-vl-7b-q4_k_m.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=4_500_000_000,
    )
    workload = VisionWorkload(image_count=1, image_size=448)
    calibrated = estimate_vram_details(
        vlm_calibration_model(architecture="qwen2_vl"),
        variant,
        vision_workload=workload,
    )
    fallback = estimate_vram_details(
        vlm_calibration_model(architecture="other_vlm"),
        variant,
        vision_workload=workload,
    )

    assert calibrated.confidence == "high"
    assert calibrated.notes == []
    assert fallback.confidence == "medium"
    assert any("no matching peak-memory calibration" in note for note in fallback.notes)
    assert calibrated.upper_bytes - calibrated.lower_bytes < (
        fallback.upper_bytes - fallback.lower_bytes
    )


def test_transformers_vlm_calibration_covers_internvl_and_gptq():
    workload = VisionWorkload(image_count=1, image_size=448)
    internvl = vlm_calibration_model(
        params=8_000_000_000,
        model_id="OpenGVLab/InternVL3-8B",
        architecture="internvl",
    )
    gemma_gptq = vlm_calibration_model(
        params=27_000_000_000,
        model_id="ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        architecture="gemma",
        quantization_type="GPTQ",
    )

    estimates = [
        estimate_vram_details(internvl, None, vision_workload=workload),
        estimate_vram_details(gemma_gptq, None, vision_workload=workload),
    ]

    assert all(estimate.confidence == "high" for estimate in estimates)
    assert all(estimate.notes == [] for estimate in estimates)


def test_moe_calibration_is_specific_to_moe_architecture():
    variant = GGUFVariant(
        filename="qwen3-30b-a3b-q4_k_m.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=int(17.1 * 1024**3),
    )

    moe_estimate = estimate_vram_details(qwen3_moe_model(), variant)
    dense_family_estimate = estimate_vram_details(qwen3_dense_model(), variant)

    assert moe_estimate.confidence == "high"
    assert moe_estimate.notes == []
    assert dense_family_estimate.confidence == "medium"
    assert any(
        "no matching peak-memory calibration" in note
        for note in dense_family_estimate.notes
    )


def test_calibration_requires_matching_vlm_workload():
    variant = GGUFVariant(
        filename="qwen2.5-vl-7b-q4_k_m.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=4_500_000_000,
    )

    calibrated = estimate_vram_details(
        vlm_calibration_model(),
        variant,
        vision_workload=VisionWorkload(image_count=1, image_size=448),
    )
    fallback = estimate_vram_details(
        vlm_calibration_model(),
        variant,
        vision_workload=VisionWorkload(image_count=2, image_size=448),
    )

    assert calibrated.confidence == "high"
    assert fallback.confidence == "medium"
    assert any("images=2@448px" in note for note in fallback.notes)


def test_calibration_requires_nearby_model_size():
    variant = GGUFVariant(
        filename="qwen2.5-vl-120b-q4_k_m.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=int(120_000_000_000 * 0.5625),
    )
    workload = VisionWorkload(image_count=1, image_size=448)

    estimate = estimate_vram_details(
        vlm_calibration_model(params=120_000_000_000, architecture="qwen2_vl"),
        variant,
        vision_workload=workload,
    )

    assert estimate.confidence == "medium"
    assert any("no matching peak-memory calibration" in note for note in estimate.notes)


def test_spatial_merge_reduces_vision_tokens():
    base = dict(
        hf_pipeline_tag="image-text-to-text",
        architecture="qwen2vl",
        model_format="safetensors",
        layer_count=28,
        hidden_size=3584,
        attention_heads=28,
        kv_heads=4,
        dtype="bfloat16",
        vision_layer_count=32,
        vision_hidden_size=1280,
        vision_intermediate_size=3420,
        vision_attention_heads=16,
        projector_hidden_size=3584,
        patch_size=14,
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
    no_merge = make_model(7_000_000_000, **base)
    merge_2 = make_model(7_000_000_000, spatial_merge_size=2, **base)
    workload = VisionWorkload(image_count=1, image_size=448)

    assert (
        estimate_vram_details(
            no_merge,
            None,
            vision_workload=workload,
        ).components.vision
        > estimate_vram_details(
            merge_2,
            None,
            vision_workload=workload,
        ).components.vision
    )


def test_estimate_vram_small_model():
    model = make_model(500_000_000)
    variant = GGUFVariant(
        filename="model-Q4_K_M.gguf", quant_type="Q4_K_M", file_size_bytes=300_000_000
    )
    vram = estimate_vram(model, variant, context_length=4096)

    assert vram > 300_000_000
    assert vram < 3_000_000_000


def test_vision_workload_increases_vram_predictably():
    model = make_model(7_000_000_000, hf_pipeline_tag="image-text-to-text")
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


def test_vision_architecture_without_pipeline_tag_increases_vram():
    model = make_model(3_000_000_000, architecture="paligemma")
    workload = VisionWorkload(image_count=1, image_size=448)

    assert estimate_vram(model, None, vision_workload=workload) > estimate_vram(
        model, None
    )


def test_vision_workload_does_not_change_text_model_vram():
    model = make_model(7_000_000_000)
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


def test_audio_processor_does_not_enable_visual_vram_overhead():
    model = make_model(
        7_000_000_000,
        capabilities=ModelCapabilities(audio=True),
        components=[
            ModelComponent(role="language", repo_id="test/model"),
            ModelComponent(role="audio_encoder", repo_id="test/model"),
            ModelComponent(role="processor", repo_id="test/model"),
        ],
    )
    workload = VisionWorkload(image_count=1, image_size=448)

    assert estimate_vram(model, None, vision_workload=workload) == estimate_vram(
        model,
        None,
    )


def test_vision_capability_enables_vram_overhead():
    model = make_model(7_000_000_000, capabilities=ModelCapabilities(image=True))
    workload = VisionWorkload(image_count=1, image_size=448)

    assert estimate_vram(model, None, vision_workload=workload) > estimate_vram(
        model,
        None,
    )


def test_vision_component_sizes_increase_vlm_overhead():
    small = make_model(
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
    large = make_model(
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
