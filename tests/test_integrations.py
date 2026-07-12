import pytest

from engine.ranker import PROFILE_TO_WORKLOAD_TASK, detect_specializations, rank_models
from engine.vram import estimate_vram
from engine.workload import Workload
from hardware.types import BackendCapability, GPUInfo, HardwareInfo
from models.fetcher import parse_model
from models.integrations import (
    INTEGRATION_PROFILES,
    capabilities_for_data,
    integration_ids_for_capabilities,
    runtime_backends_for_capabilities,
    runtime_backends_for_data,
)
from runtime import generate_run_script, recommended_runtime_backend, resolve_model_deps


def linux_cuda_hardware() -> HardwareInfo:
    return HardwareInfo(
        os="linux",
        gpus=[
            GPUInfo(
                name="NVIDIA Test GPU",
                vendor="nvidia",
                vram_bytes=24 * 1024**3,
                compute_capability=(8, 9),
                backend_capabilities=[BackendCapability("cuda", True)],
            )
        ],
        ram_bytes=64 * 1024**3,
    )


def test_registered_profiles_define_complete_contract():
    for profile in INTEGRATION_PROFILES:
        assert profile.integration_id
        assert profile.capability_names
        assert profile.pipeline_tags or profile.tag_patterns
        assert profile.component_roles
        assert profile.workload_tasks
        if not set(profile.capability_names) & {"video", "audio"}:
            assert profile.runtime_backends


def test_registered_profiles_cover_ranker_workloads():
    registered_tasks = {
        task for profile in INTEGRATION_PROFILES for task in profile.workload_tasks
    }

    assert set(PROFILE_TO_WORKLOAD_TASK.values()) <= registered_tasks


def test_plain_image_to_text_is_not_document_ocr():
    capabilities = capabilities_for_data(
        "org/Captioner-7B",
        "image-to-text",
        ["safetensors"],
    )

    assert capabilities.image is True
    assert capabilities.ocr is False
    assert capabilities.document is False


@pytest.mark.parametrize(
    ("model_id", "pipeline_tag", "tags", "expected"),
    [
        ("org/VideoChat-7B", "video-text-to-text", ["safetensors"], {"video"}),
        ("org/AudioChat-7B", "audio-text-to-text", ["safetensors"], {"audio"}),
        ("org/Whisper-7B", "automatic-speech-recognition", [], {"audio"}),
        (
            "org/ChartQA-7B",
            "image-text-to-text",
            ["chartqa", "safetensors"],
            {"image", "chart"},
        ),
    ],
)
def test_registry_classifies_media_profiles(
    model_id: str,
    pipeline_tag: str,
    tags: list[str],
    expected: set[str],
):
    capabilities = capabilities_for_data(model_id, pipeline_tag, tags)

    enabled = {
        name
        for name in ("image", "video", "audio", "ocr", "document", "chart")
        if getattr(capabilities, name)
    }
    assert expected <= enabled


def test_parse_model_uses_registered_video_profile_without_image_runtime():
    model = parse_model(
        {
            "id": "org/VideoChat-7B",
            "pipeline_tag": "video-text-to-text",
            "tags": ["safetensors"],
            "config": {"architectures": ["VideoChatForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert model is not None
    assert model.capabilities.video is True
    assert model.capabilities.image is False
    assert "video-language" in integration_ids_for_capabilities(model.capabilities)
    assert "video" in detect_specializations(model)
    assert "vision" not in detect_specializations(model)
    assert "video_encoder" in {component.role for component in model.components}
    assert runtime_backends_for_capabilities(model.capabilities) == []


def test_parse_model_uses_registered_audio_profile_without_runtime_claim():
    model = parse_model(
        {
            "id": "org/AudioChat-7B",
            "pipeline_tag": "audio-text-to-text",
            "tags": ["safetensors"],
            "config": {"architectures": ["AudioChatForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert model is not None
    assert model.capabilities.audio is True
    assert "audio-language" in integration_ids_for_capabilities(model.capabilities)
    assert "audio" in detect_specializations(model)
    assert "audio_encoder" in {component.role for component in model.components}
    assert runtime_backends_for_capabilities(model.capabilities) == []


def test_qwen25_vl_video_profile_has_transformers_runtime_path():
    model = parse_model(
        {
            "id": "Qwen/Qwen2.5-VL-7B-Instruct",
            "pipeline_tag": "image-text-to-text",
            "tags": ["safetensors"],
            "config": {"architectures": ["Qwen2_5_VLForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert model is not None
    assert model.capabilities.image is True
    assert model.capabilities.video is True
    assert "video-language" in integration_ids_for_capabilities(model.capabilities)
    assert runtime_backends_for_data(
        model.id,
        model.hf_pipeline_tag,
        model.tags,
        model.architecture,
    ) == ["transformers", "llama.cpp", "mlx", "vllm", "sglang"]

    deps, script_type = resolve_model_deps(
        model,
        None,
        backend_name="transformers",
        hardware=linux_cuda_hardware(),
    )
    script = generate_run_script(
        model,
        None,
        4096,
        False,
        video_path="/tmp/video.mp4",
        backend_name="transformers",
        hardware=linux_cuda_hardware(),
    )

    assert recommended_runtime_backend(model, None, linux_cuda_hardware()) == "vllm"
    assert "torchvision" in deps
    assert "qwen-vl-utils" in deps
    assert script_type == "transformers_vlm"
    assert "Qwen2_5_VLForConditionalGeneration" in script
    assert "video_path = '/tmp/video.mp4'" in script
    assert '{"type": "video", "video": video_uri, "fps": 1.0}' in script
    assert "process_vision_info" in script


def test_qwen2_audio_profile_has_transformers_runtime_path():
    model = parse_model(
        {
            "id": "Qwen/Qwen2-Audio-7B-Instruct",
            "pipeline_tag": "audio-text-to-text",
            "tags": ["safetensors"],
            "config": {"architectures": ["Qwen2AudioForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert model is not None
    assert model.capabilities.audio is True
    assert "audio-language" in integration_ids_for_capabilities(model.capabilities)
    assert runtime_backends_for_capabilities(model.capabilities) == []
    assert runtime_backends_for_data(
        model.id,
        model.hf_pipeline_tag,
        model.tags,
        model.architecture,
    ) == ["transformers"]

    deps, script_type = resolve_model_deps(
        model,
        None,
        backend_name="transformers",
        hardware=linux_cuda_hardware(),
    )
    script = generate_run_script(
        model,
        None,
        4096,
        False,
        audio_path="/tmp/audio.wav",
        backend_name="transformers",
        hardware=linux_cuda_hardware(),
    )

    assert (
        recommended_runtime_backend(model, None, linux_cuda_hardware())
        == "transformers"
    )
    assert "librosa" in deps
    assert script_type == "transformers_audio"
    assert "Qwen2AudioForConditionalGeneration" in script
    assert "audio_path = '/tmp/audio.wav'" in script
    assert "audios=[audio]" in script


def test_parse_model_uses_registered_chart_profile_with_image_runtime():
    model = parse_model(
        {
            "id": "org/ChartQA-7B",
            "pipeline_tag": "image-text-to-text",
            "tags": ["chartqa", "safetensors"],
            "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
            "safetensors": {"total": 7_000_000_000},
            "siblings": [],
            "cardData": {},
        }
    )

    assert model is not None
    assert model.capabilities.image is True
    assert model.capabilities.chart is True
    assert "chart-document" in integration_ids_for_capabilities(model.capabilities)
    assert "chart" in detect_specializations(model)
    assert "transformers" in runtime_backends_for_capabilities(model.capabilities)


@pytest.mark.parametrize(
    ("integration_id", "model_data", "workload"),
    [
        (
            "vision-language",
            {
                "id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "pipeline_tag": "image-text-to-text",
                "tags": ["vision-language", "safetensors"],
                "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
                "safetensors": {"total": 7_000_000_000},
                "siblings": [],
                "cardData": {},
            },
            Workload(task="image_qa", context_length=4096, image_count=1),
        ),
        (
            "document-ocr",
            {
                "id": "org/DocVQA-OCR-7B",
                "pipeline_tag": "image-to-text",
                "tags": ["document", "ocr", "safetensors"],
                "config": {"architectures": ["Qwen2VLForConditionalGeneration"]},
                "safetensors": {"total": 7_000_000_000},
                "siblings": [],
                "cardData": {},
                "evalResults": [
                    {
                        "filename": ".eval_results/docvqa.yaml",
                        "data": {
                            "dataset": {"id": "DocVQA"},
                            "value": 80.0,
                        },
                    }
                ],
            },
            Workload(task="ocr", context_length=4096, image_count=1),
        ),
    ],
)
def test_registered_integration_has_complete_path(
    integration_id: str,
    model_data: dict,
    workload: Workload,
):
    model = parse_model(model_data)
    assert model is not None
    assert integration_id in integration_ids_for_capabilities(model.capabilities)
    assert model.artifacts
    assert model.components

    text_vram = estimate_vram(model, None, context_length=workload.context_length)
    media_vram = estimate_vram(
        model,
        None,
        context_length=workload.context_length,
        vision_workload=workload,
    )
    assert media_vram > text_vram

    results = rank_models(
        [model],
        linux_cuda_hardware(),
        top_n=1,
        task_profile=workload.task,
        workload=workload,
        benchmark_scores={model.id: 70.0},
    )
    assert results
    assert results[0].model.id == model.id

    backends = runtime_backends_for_capabilities(model.capabilities)
    assert "transformers" in backends
    deps, script_type = resolve_model_deps(
        model,
        None,
        backend_name="transformers",
        hardware=linux_cuda_hardware(),
    )
    script = generate_run_script(
        model,
        None,
        workload.context_length,
        False,
        image_path="/tmp/image.png",
        backend_name="transformers",
        hardware=linux_cuda_hardware(),
    )

    assert "pillow" in deps
    assert script_type == "transformers_vlm"
    assert "AutoProcessor" in script
    assert "TextIteratorStreamer" in script
    assert "image_path = '/tmp/image.png'" in script
