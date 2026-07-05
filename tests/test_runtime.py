import pytest

from whichvlm.models.fetcher import parse_model
from whichvlm.models.types import (
    GGUFVariant,
    ModelArtifact,
    ModelCapabilities,
    ModelComponent,
    ModelInfo,
)
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


def test_runtime_uses_cached_vision_capability():
    model = ModelInfo(
        id="Qwen/Qwen2-VL-7B",
        family_id="qwen-vl",
        name="Qwen2-VL-7B",
        parameter_count=7_000_000_000,
        capabilities=ModelCapabilities(image=True),
    )

    deps, script_type = resolve_model_deps(model, None)

    assert requires_image(model)
    assert "pillow" in deps
    assert script_type == "transformers_vlm"


def test_audio_processor_does_not_require_image():
    model = ModelInfo(
        id="org/Audio-7B",
        family_id="audio-7b",
        name="Audio-7B",
        parameter_count=7_000_000_000,
        capabilities=ModelCapabilities(audio=True),
        components=[
            ModelComponent(role="language", repo_id="org/Audio-7B"),
            ModelComponent(role="audio_encoder", repo_id="org/Audio-7B"),
            ModelComponent(role="processor", repo_id="org/Audio-7B"),
        ],
    )

    assert not requires_image(model)


def test_transformers_vlm_script_uses_processor_and_image_path():
    model = vlm_model()

    deps, script_type = resolve_model_deps(model, None)
    script = generate_run_script(
        model, None, 4096, False, image_path="/tmp/image.png", max_tokens=128
    )

    assert "pillow" in deps
    assert script_type == "transformers_vlm"
    assert "AutoProcessor" in script
    assert "Qwen2_5_VLForConditionalGeneration" in script
    assert "image_path = '/tmp/image.png'" in script
    assert '{"type": "image", "image": image}' in script
    assert "max_new_tokens=128" in script
    assert "min_pixels=256 * 28 * 28" in script
    assert "max_pixels=1280 * 28 * 28" in script
    assert "TextIteratorStreamer" in script
    assert "torch.inference_mode()" in script
    assert "[metrics] ttft=" in script


def test_text_runtime_scripts_use_custom_max_tokens():
    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
    )
    variant = GGUFVariant(
        filename="test-q4.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=4_000_000_000,
    )

    transformers = generate_run_script(model, None, 4096, False, max_tokens=96)
    gguf = generate_run_script(model, variant, 4096, False, max_tokens=96)

    assert "max_new_tokens=96" in transformers
    assert "max_tokens=96" in gguf


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


def test_transformers_text_script_uses_inference_mode_and_joined_stream():
    model = ModelInfo(
        id="org/Test-7B",
        family_id="test-7b",
        name="Test-7B",
        parameter_count=7_000_000_000,
    )

    script = generate_run_script(model, None, 4096, False)

    assert "model.eval()" in script
    assert "with torch.inference_mode():" in script
    assert "output_parts.append(text)" in script
    assert '"".join(output_parts)' in script
    assert "full +=" not in script


def test_llama_cpp_text_script_joins_streamed_response():
    model = ModelInfo(
        id="org/Test-7B-GGUF",
        family_id="test-7b",
        name="Test-7B-GGUF",
        parameter_count=7_000_000_000,
        gguf_variants=[
            GGUFVariant(
                filename="test-Q4_K_M.gguf",
                quant_type="Q4_K_M",
                file_size_bytes=4_000_000_000,
            )
        ],
    )

    script = generate_run_script(model, model.gguf_variants[0], 4096, False)

    assert "output_parts.append(content)" in script
    assert '"".join(output_parts)' in script
    assert "full +=" not in script


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
        generate_run_script(
            vlm_model(), None, 4096, False, image_path="/tmp/image.png"
        ),
        generate_run_script(gguf_model, gguf_variant, 4096, False),
        generate_run_script(
            gguf_vlm,
            gguf_variant,
            4096,
            False,
            image_path="/tmp/image.png",
        ),
        generate_run_script(
            mlx_vlm,
            None,
            4096,
            False,
            image_path="/tmp/image.png",
            hardware=darwin_mlx_hardware(),
        ),
        generate_run_script(
            vlm_model(),
            None,
            4096,
            False,
            image_path="/tmp/image.png",
            backend_name="vllm",
            hardware=linux_cuda_hardware(),
        ),
        generate_run_script(
            vlm_model(),
            None,
            4096,
            False,
            image_path="/tmp/image.png",
            backend_name="sglang",
            hardware=linux_cuda_hardware(),
        ),
    ]

    for script in scripts:
        compile(script, "<whichvlm-generated>", "exec")


def test_runtime_detects_vlm_from_architecture():
    model = parse_model(
        {
            "id": "org/ConfigOnly-3B",
            "tags": ["transformers", "safetensors"],
            "config": {
                "architectures": ["PaliGemmaForConditionalGeneration"],
                "model_type": "paligemma",
            },
            "safetensors": {"total": 3_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert model is not None
    deps, script_type = resolve_model_deps(model, None)

    assert requires_image(model)
    assert "pillow" in deps
    assert script_type == "transformers_vlm"


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
        max_tokens=128,
    )

    assert "pillow" in deps
    assert script_type == "gguf_vlm"
    assert "Llava15ChatHandler" in script
    assert "clip_model_path=mmproj_path" in script
    assert 'projector_filename = "mmproj-test-f16.gguf"' in script
    assert "image_data_url" in script
    assert "max_tokens=128" in script


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
        max_tokens=96,
        hardware=hardware,
    )

    assert deps == ["mlx-vlm", "pillow"]
    assert script_type == "mlx_vlm"
    assert "from mlx_vlm import generate, load" in script
    assert "apply_chat_template" in script
    assert "except ImportError:" in script
    assert "except Exception:" not in script
    assert "[image_path]" in script
    assert "max_tokens=96" in script


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
    model = vlm_model(quantization_type="AWQ")

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
        max_tokens=96,
        backend_name="vllm",
        hardware=linux_cuda_hardware(),
    )

    assert deps == ["vllm", "psutil"]
    assert script_type == "vllm"
    assert "from vllm import LLM, SamplingParams" in script
    assert "llm.chat" in script
    assert "image_data_url" in script
    assert "SamplingParams(max_tokens=96)" in script
    assert "quantization = 'awq'" in script
    assert "gpu_memory_utilization=0.90" in script
    assert "[metrics] ttft=" in script


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
        max_tokens=96,
        backend_name="sglang",
        hardware=linux_cuda_hardware(),
    )

    assert deps == ["sglang", "psutil"]
    assert script_type == "sglang"
    assert "from sglang import Engine" in script
    assert "engine.generate" in script
    assert "stream=True" in script
    assert "image_data=image_path" in script
    assert '"max_new_tokens": 96' in script


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
