import pytest

from whichvlm.models.types import GGUFVariant, ModelArtifact, ModelInfo
from whichvlm.runtime import (
    RuntimeUnsupportedError,
    generate_run_script,
    requires_image,
    resolve_model_deps,
)


def vlm_model(**kwargs) -> ModelInfo:
    values = {
        "id": "org/Test-VL-7B",
        "family_id": "test-vl",
        "name": "Test-VL-7B",
        "parameter_count": 7_000_000_000,
        "hf_pipeline_tag": "image-text-to-text",
    }
    values.update(kwargs)
    return ModelInfo(**values)


def test_vlm_runtime_requires_image():
    model = vlm_model()

    assert requires_image(model)
    with pytest.raises(RuntimeUnsupportedError, match="--image"):
        generate_run_script(model, None, 4096, False)


def test_transformers_vlm_script_uses_processor_and_image_path():
    model = vlm_model()

    deps, script_type = resolve_model_deps(model, None)
    script = generate_run_script(model, None, 4096, False, image_path="/tmp/image.png")

    assert "pillow" in deps
    assert script_type == "transformers_vlm"
    assert "AutoProcessor" in script
    assert "AutoModelForImageTextToText" in script
    assert 'image_path = \'/tmp/image.png\'' in script
    assert '{"type": "image", "image": image}' in script
    assert "TextIteratorStreamer" in script
    assert "torch.inference_mode()" in script
    assert "[metrics] ttft=" in script


def test_transformers_vlm_script_uses_family_profile():
    model = vlm_model(
        id="Qwen/Qwen2.5-VL-7B-Instruct",
        family_id="qwen2.5-vl",
        name="Qwen2.5-VL-7B-Instruct",
    )

    script = generate_run_script(model, None, 4096, False, image_path="/tmp/image.png")

    assert "Qwen2_5_VLForConditionalGeneration" in script
    assert "min_pixels=256 * 28 * 28" in script
    assert "max_pixels=1280 * 28 * 28" in script


def test_transformers_quantized_script_uses_bitsandbytes_loader():
    model = ModelInfo(
        id="org/Test-7B-BNB-4bit",
        family_id="test-7b",
        name="Test-7B-BNB-4bit",
        parameter_count=7_000_000_000,
    )

    deps, script_type = resolve_model_deps(model, None)
    script = generate_run_script(model, None, 4096, False)

    assert script_type == "transformers"
    assert "bitsandbytes" in deps
    assert "BitsAndBytesConfig" in script
    assert 'model_kwargs["quantization_config"]' in script
    assert "attn_implementation=\"sdpa\"" in script
    assert "max_memory=cuda_memory_limits()" in script


def test_generated_scripts_compile():
    gguf_variant = GGUFVariant(
        filename="test-q4.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=4_000_000_000,
    )
    text_model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
    )
    gguf_model = ModelInfo(
        id="org/Test-7B-GGUF",
        family_id="test-7b",
        name="Test-7B-GGUF",
        parameter_count=7_000_000_000,
        gguf_variants=[gguf_variant],
        model_format="gguf",
    )
    gguf_vlm = vlm_model(
        gguf_variants=[gguf_variant],
        model_format="gguf",
        artifacts=[
            ModelArtifact(
                repo_id="org/Test-VL-7B",
                format="adapter",
                filename="mmproj-test-f16.gguf",
                source_kind="mmproj",
            ),
        ],
    )
    mlx_vlm = vlm_model(model_format="mlx")
    scripts = [
        generate_run_script(text_model, None, 4096, False),
        generate_run_script(vlm_model(), None, 4096, False, image_path="/tmp/image.png"),
        generate_run_script(gguf_model, gguf_variant, 4096, False),
        generate_run_script(gguf_vlm, gguf_variant, 4096, False, image_path="/tmp/image.png"),
        generate_run_script(mlx_vlm, None, 4096, False, image_path="/tmp/image.png"),
    ]

    for script in scripts:
        compile(script, "<whichvlm-generated>", "exec")


def test_gguf_vlm_runtime_requires_projector_artifact():
    model = vlm_model(
        gguf_variants=[
            GGUFVariant(
                filename="test-q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
        model_format="gguf",
    )

    with pytest.raises(RuntimeUnsupportedError, match="mmproj"):
        generate_run_script(
            model,
            model.gguf_variants[0],
            4096,
            False,
            image_path="/tmp/image.png",
        )


def test_gguf_vlm_script_uses_llama_cpp_projector_artifact():
    model = vlm_model(
        gguf_variants=[
            GGUFVariant(
                filename="test-q4.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
        model_format="gguf",
        artifacts=[
            ModelArtifact(
                repo_id="org/Test-VL-7B",
                format="gguf",
                filename="test-q4.gguf",
                source_kind="gguf_variant",
            ),
            ModelArtifact(
                repo_id="org/Test-VL-7B",
                format="adapter",
                filename="mmproj-test-f16.gguf",
                source_kind="mmproj",
            ),
        ],
    )

    deps, script_type = resolve_model_deps(model, model.gguf_variants[0])
    script = generate_run_script(
        model,
        model.gguf_variants[0],
        4096,
        False,
        image_path="/tmp/image.png",
    )

    assert "pillow" in deps
    assert script_type == "gguf_vlm"
    assert "Llava15ChatHandler" in script
    assert "clip_model_path=mmproj_path" in script
    assert 'projector_filename = "mmproj-test-f16.gguf"' in script
    assert "image_data_url" in script


def test_mlx_vlm_script_uses_mlx_vlm_runner():
    model = vlm_model(
        model_format="mlx",
        artifacts=[
            ModelArtifact(
                repo_id="org/Test-VL-7B-MLX",
                format="mlx",
                source_kind="mlx_variant",
            )
        ],
    )

    deps, script_type = resolve_model_deps(model, None)
    script = generate_run_script(model, None, 4096, False, image_path="/tmp/image.png")

    assert deps == ["mlx-vlm", "pillow"]
    assert script_type == "mlx_vlm"
    assert "from mlx_vlm import generate, load" in script
    assert "apply_chat_template" in script
    assert "except ImportError:" in script
    assert "except Exception:" not in script
    assert "[image_path]" in script
