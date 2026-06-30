import pytest

from whichvlm.models.types import GGUFVariant, ModelArtifact, ModelInfo
from whichvlm.hardware.types import BackendCapability, GPUInfo, HardwareInfo
from whichvlm.runtime import (
    ServeRequest,
    RuntimeUnsupportedError,
    generate_run_script,
    requires_image,
    resolve_model_deps,
    select_serve_backend,
    serve_request,
)


def vlm_model(**kwargs) -> ModelInfo:
    return ModelInfo(
        id=kwargs.pop("id", "Qwen/Qwen2.5-VL-7B-Instruct"),
        family_id=kwargs.pop("family_id", "qwen-vl"),
        name=kwargs.pop("name", "Qwen2.5-VL-7B-Instruct"),
        parameter_count=kwargs.pop("parameter_count", 7_000_000_000),
        architecture=kwargs.pop("architecture", "qwen2"),
        hf_pipeline_tag=kwargs.pop("hf_pipeline_tag", "image-text-to-text"),
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
    assert "image_path = '/tmp/image.png'" in script
    assert '{"type": "image", "image": image}' in script


def test_unknown_transformers_vlm_is_not_claimed_supported():
    model = vlm_model(
        id="org/Unknown-VL-7B",
        family_id="unknown-vl",
        name="Unknown-VL-7B",
        architecture="unknownvl",
    )

    with pytest.raises(RuntimeUnsupportedError, match="No supported run backend"):
        generate_run_script(model, None, 4096, False, image_path="/tmp/image.png")


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
    hardware = darwin_mlx_hardware()

    deps, script_type = resolve_model_deps(model, None, hardware=hardware)
    script = generate_run_script(
        model,
        None,
        4096,
        False,
        image_path="/tmp/image.png",
        hardware=hardware,
    )

    assert deps == ["mlx-vlm", "pillow"]
    assert script_type == "mlx_vlm"
    assert "from mlx_vlm import generate, load" in script
    assert "apply_chat_template" in script
    assert "except ImportError:" in script
    assert "except Exception:" not in script
    assert "[image_path]" in script


def darwin_mlx_hardware() -> HardwareInfo:
    return HardwareInfo(
        os="darwin",
        gpus=[
            GPUInfo(
                name="Apple Test GPU",
                vendor="apple",
                vram_bytes=36_000_000_000,
                backend_capabilities=[
                    BackendCapability("metal", True),
                    BackendCapability("mlx", True),
                ],
            )
        ],
    )


def linux_cuda_hardware() -> HardwareInfo:
    return HardwareInfo(
        os="linux",
        gpus=[
            GPUInfo(
                name="NVIDIA Test GPU",
                vendor="nvidia",
                vram_bytes=24_000_000_000,
                backend_capabilities=[BackendCapability("cuda", True)],
            )
        ],
    )


def test_vllm_vlm_backend_requires_explicit_linux_cuda_support():
    model = vlm_model()

    deps, script_type = resolve_model_deps(
        model,
        None,
        backend_name="vllm",
        hardware=linux_cuda_hardware(),
    )
    script = generate_run_script(
        model,
        None,
        4096,
        False,
        image_path="/tmp/image.png",
        backend_name="vllm",
        hardware=linux_cuda_hardware(),
    )

    assert deps == ["vllm"]
    assert script_type == "vllm"
    assert "from vllm import LLM, SamplingParams" in script
    assert "llm.chat" in script
    assert "image_data_url" in script


def test_sglang_vlm_backend_uses_offline_engine():
    model = vlm_model()

    deps, script_type = resolve_model_deps(
        model,
        None,
        backend_name="sglang",
        hardware=linux_cuda_hardware(),
    )
    script = generate_run_script(
        model,
        None,
        4096,
        False,
        image_path="/tmp/image.png",
        backend_name="sglang",
        hardware=linux_cuda_hardware(),
    )

    assert deps == ["sglang"]
    assert script_type == "sglang"
    assert "from sglang import Engine" in script
    assert "engine.generate" in script
    assert "image_data=image_path" in script


def test_transformers_backend_is_not_a_server_backend():
    model = vlm_model()

    with pytest.raises(RuntimeUnsupportedError, match="does not support serve"):
        select_serve_backend(
            model,
            None,
            linux_cuda_hardware(),
            backend_name="transformers",
        )


def test_vllm_serve_uses_openai_server_command(monkeypatch):
    model = vlm_model()
    captured: dict[str, list[str]] = {}

    class Result:
        returncode = 0

    def fake_run(cmd):
        captured["cmd"] = cmd
        return Result()

    monkeypatch.setattr("whichvlm.runtime.subprocess.run", fake_run)

    code = serve_request(
        ServeRequest(
            model=model,
            artifact=None,
            context_length=8192,
            cpu_only=False,
            hardware=linux_cuda_hardware(),
            host="0.0.0.0",
            port=9000,
        ),
        backend_name="vllm",
    )

    assert code == 0
    assert captured["cmd"] == [
        "uv",
        "run",
        "--no-project",
        "--with",
        "vllm",
        "vllm",
        "serve",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "--host",
        "0.0.0.0",
        "--port",
        "9000",
        "--max-model-len",
        "8192",
        "--trust-remote-code",
    ]
