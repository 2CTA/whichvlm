import pytest

from whichvlm.models.types import GGUFVariant, ModelArtifact, ModelInfo
from whichvlm.runtime import (
    RuntimeUnsupportedError,
    generate_run_script,
    requires_image,
    resolve_model_deps,
)


def vlm_model(**kwargs) -> ModelInfo:
    return ModelInfo(
        id="org/Test-VL-7B",
        family_id="test-vl",
        name="Test-VL-7B",
        parameter_count=7_000_000_000,
        hf_pipeline_tag="image-text-to-text",
        **kwargs,
    )


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


def test_runtime_detects_vlm_from_architecture():
    model = ModelInfo(
        id="org/ConfigOnly-3B",
        family_id="config-only",
        name="ConfigOnly-3B",
        parameter_count=3_000_000_000,
        architecture="paligemma",
    )

    deps, script_type = resolve_model_deps(model, None)

    assert requires_image(model)
    assert "pillow" in deps
    assert script_type == "transformers_vlm"


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
