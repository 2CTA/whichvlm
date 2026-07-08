from __future__ import annotations

from constants import BYTES_PER_GIB
from engine.ranker import rank_models
from hardware.types import GPUInfo, HardwareInfo
from models.types import GGUFVariant, ModelCapabilities, ModelInfo


VISION_BENCHMARKS = {
    "Qwen/Qwen2.5-VL-7B-Instruct": 72.0,
    "OpenGVLab/InternVL3-8B": 74.0,
    "Qwen/Qwen3-VL-8B-Instruct": 76.0,
    "Qwen/Qwen3-VL-32B-Instruct": 88.0,
}


def calibration_hardware(vram_gb: int | None, ram_gb: int = 64) -> HardwareInfo:
    gpus = []
    if vram_gb is not None:
        gpus = [
            GPUInfo(
                name="Calibration GPU",
                vendor="nvidia",
                vram_bytes=vram_gb * BYTES_PER_GIB,
                compute_capability=(8, 9),
                memory_bandwidth_gbps=1000.0,
            )
        ]
    return HardwareInfo(
        gpus=gpus,
        cpu_name="Calibration CPU",
        cpu_cores=16,
        has_avx2=True,
        ram_bytes=ram_gb * BYTES_PER_GIB,
        disk_free_bytes=500 * BYTES_PER_GIB,
        os="linux",
    )


def gguf_variant(size_gib: float) -> GGUFVariant:
    return GGUFVariant(
        filename="model-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=int(size_gib * BYTES_PER_GIB),
    )


def vision_model(
    model_id: str,
    family_id: str,
    params_b: float,
    artifact_size_gib: float,
    downloads: int,
    likes: int,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        family_id=family_id,
        name=model_id.split("/")[-1],
        parameter_count=int(params_b * 1e9),
        hf_pipeline_tag="image-text-to-text",
        tags=["vision-language"],
        capabilities=ModelCapabilities(image=True),
        downloads=downloads,
        likes=likes,
        gguf_variants=[gguf_variant(artifact_size_gib)],
    )


def vision_calibration_models() -> list[ModelInfo]:
    return [
        vision_model(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen2.5-vl-7b",
            7.0,
            4.5,
            2_000_000,
            500,
        ),
        vision_model(
            "OpenGVLab/InternVL3-8B",
            "internvl3-8b",
            8.0,
            5.0,
            800_000,
            300,
        ),
        vision_model(
            "Qwen/Qwen3-VL-8B-Instruct",
            "qwen3-vl-8b",
            8.0,
            5.0,
            500_000,
            250,
        ),
        vision_model(
            "Qwen/Qwen3-VL-32B-Instruct",
            "qwen3-vl-32b",
            32.0,
            19.0,
            300_000,
            150,
        ),
        vision_model(
            "community/PopularVision-7B-GGUF",
            "popular-vision-7b",
            7.0,
            4.4,
            5_000_000,
            2_000,
        ),
    ]


def test_vision_ranking_calibration_24gb_keeps_benchmark_order():
    results = rank_models(
        vision_calibration_models(),
        calibration_hardware(24),
        top_n=5,
        task_profile="vision",
        benchmark_scores=VISION_BENCHMARKS,
    )

    assert [result.model.id for result in results[:3]] == [
        "Qwen/Qwen3-VL-32B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "OpenGVLab/InternVL3-8B",
    ]
    assert results[0].fit_type == "partial_offload"
    assert all(result.ranking_evidence == "benchmark score" for result in results[:3])


def test_vision_ranking_calibration_48gb_prefers_full_gpu_leader():
    results = rank_models(
        vision_calibration_models(),
        calibration_hardware(48),
        top_n=5,
        task_profile="vision",
        benchmark_scores=VISION_BENCHMARKS,
    )

    assert [result.model.id for result in results[:3]] == [
        "Qwen/Qwen3-VL-32B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "OpenGVLab/InternVL3-8B",
    ]
    assert results[0].fit_type == "full_gpu"
    assert results[0].quality_score > results[1].quality_score


def test_vision_ranking_calibration_cpu_tier_prefers_small_benchmark_leader():
    results = rank_models(
        vision_calibration_models(),
        calibration_hardware(None),
        top_n=5,
        task_profile="vision",
        benchmark_scores=VISION_BENCHMARKS,
    )
    ids = [result.model.id for result in results]

    assert ids[:3] == [
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-32B-Instruct",
        "OpenGVLab/InternVL3-8B",
    ]
    assert all(result.fit_type == "cpu_only" for result in results[:3])
    assert "community/PopularVision-7B-GGUF" not in ids[:3]
