from __future__ import annotations

import json

from engine.quantization import effective_quant_type, estimate_weight_bytes
from engine.types import CompatibilityResult
from hardware.types import BackendCapability, HardwareInfo
from models.types import (
    ModelArtifact,
    ModelComponent,
    ModelInfo,
    ModelLineage,
)
from models.package_graph import capabilities_to_dict
from output import console
from output.upgrade import summarize_upgrade_row
from runtime import recommended_runtime_backend


def backend_capability_dict(capability: BackendCapability) -> dict:
    return {
        "name": capability.name,
        "available": capability.available,
        "version": capability.version,
        "details": capability.details,
    }


def artifact_dict(artifact: ModelArtifact) -> dict:
    return {
        "repo_id": artifact.repo_id,
        "format": artifact.format,
        "quantization": artifact.quantization,
        "file_size_bytes": artifact.file_size_bytes,
        "access": artifact.access,
        "backend_support": artifact.backend_support,
        "source_kind": artifact.source_kind,
        "filename": artifact.filename,
    }


def component_dict(component: ModelComponent) -> dict:
    return {
        "role": component.role,
        "repo_id": component.repo_id,
        "parameter_count": component.parameter_count,
        "quantization": component.quantization,
    }


def lineage_dict(lineage: ModelLineage) -> dict:
    return {
        "base_model_ids": lineage.base_model_ids,
        "merged_parent_ids": lineage.merged_parent_ids,
        "variant_of": lineage.variant_of,
        "relationship": lineage.relationship,
        "is_merged": lineage.is_merged,
    }


def hardware_dict(hardware: HardwareInfo, details: bool = False) -> dict:
    gpus = []
    for gpu in hardware.gpus:
        gpu_data = {
            "name": gpu.name,
            "vram_bytes": gpu.vram_bytes,
            "usable_vram_bytes": gpu.usable_vram_bytes,
        }
        if details:
            gpu_data.update(
                {
                    "vendor": gpu.vendor,
                    "memory_bandwidth_gbps": gpu.memory_bandwidth_gbps,
                    "shared_memory": gpu.shared_memory,
                    "backend_capabilities": [
                        backend_capability_dict(c) for c in gpu.backend_capabilities
                    ],
                    "neural_engine_available": gpu.neural_engine_available,
                }
            )
        gpus.append(gpu_data)

    data = {
        "gpus": gpus,
        "cpu": hardware.cpu_name,
        "cpu_cores": hardware.cpu_cores,
        "ram_bytes": hardware.ram_bytes,
        "ram_budget_bytes": hardware.ram_budget_bytes,
        "os": hardware.os,
    }
    if details:
        data["budget_notes"] = hardware.budget_notes
        data["backend_capabilities"] = [
            backend_capability_dict(c) for c in hardware.backend_capabilities
        ]
    return data


def result_binding_constraint(result: CompatibilityResult) -> str:
    if not result.context_fits:
        return "context length"
    if not result.can_run:
        return "memory"
    if result.estimated_tok_per_sec is None:
        return "bandwidth"
    if result.fit_type == "partial_offload":
        return "VRAM"
    if result.uses_multi_gpu:
        return "multi-GPU split"
    return "none"


def model_dict(
    rank: int,
    result: CompatibilityResult,
    hardware: HardwareInfo,
    details: bool = False,
) -> dict:
    model = result.model
    data = {
        "rank": rank,
        "model_id": model.id,
        "recommended_runtime_backend": recommended_runtime_backend(
            model, result.gguf_variant, hardware
        ),
        "parameter_count": model.parameter_count,
        "license": model.license,
        "quant_type": effective_quant_type(model, result.gguf_variant),
        "file_size_bytes": (
            result.gguf_variant.file_size_bytes
            if result.gguf_variant
            else estimate_weight_bytes(model, None)
        ),
        "vram_required_bytes": result.vram_required_bytes,
        "vram_required_range_bytes": (
            list(result.vram_required_range_bytes)
            if result.vram_required_range_bytes
            else None
        ),
        "vram_confidence": result.vram_confidence,
        "vram_available_bytes": result.vram_available_bytes,
        "estimated_tok_per_sec": result.estimated_tok_per_sec,
        "benchmark_status": result.benchmark_status,
        "benchmark_source": result.benchmark_source,
        "ranking_evidence": result.ranking_evidence,
        "fit_type": result.fit_type,
        "can_run": result.can_run,
        "context_fits": result.context_fits,
        "offload_ratio": result.offload_ratio,
        "binding_constraint": result_binding_constraint(result),
        "warnings": result.warnings,
        "quality_score": round(result.quality_score, 2),
        "benchmark_confidence": round(result.benchmark_confidence, 2),
    }
    if details:
        data.update(
            {
                "family_id": model.family_id,
                "architecture": model.architecture,
                "hf_pipeline_tag": model.hf_pipeline_tag,
                "tags": model.tags,
                "capabilities": capabilities_to_dict(model.capabilities),
                "access": model.access,
                "is_official": model.is_official,
                "model_format": model.model_format,
                "variant_kind": model.variant_kind,
                "quantization_type": model.quantization_type,
                "base_model": model.base_model,
                "base_models": model.base_models,
                "variant_of": model.variant_of,
                "artifacts": [artifact_dict(a) for a in model.artifacts],
                "components": [component_dict(c) for c in model.components],
                "lineage": lineage_dict(model.lineage),
                "published_at": model.published_at,
                "downloads": model.downloads,
                "uses_multi_gpu": result.uses_multi_gpu,
                "multi_gpu_effective_vram_bytes": (
                    result.multi_gpu_effective_vram_bytes
                ),
                "speed_confidence": result.speed_confidence,
                "speed_range_tok_per_sec": (
                    list(result.speed_range_tok_per_sec)
                    if result.speed_range_tok_per_sec
                    else None
                ),
                "speed_notes": result.speed_notes,
                "vram_breakdown_bytes": result.vram_breakdown_bytes,
                "vram_notes": result.vram_notes,
                "ranking_freshness_weight": result.ranking_freshness_weight,
            }
        )
    return data


def display_json(
    results: list[CompatibilityResult],
    hardware: HardwareInfo,
    details: bool = False,
) -> None:
    output = {
        "hardware": hardware_dict(hardware, details),
        "models": [
            model_dict(i, result, hardware, details)
            for i, result in enumerate(results, 1)
        ],
    }
    if details:
        from engine.ranker import RANKING_ALGORITHM_VERSION
        from models.benchmark import benchmark_cache_snapshot
        from models.cache import cache_snapshot

        output["ranking"] = {
            "algorithm_version": RANKING_ALGORITHM_VERSION,
            "freshness_weight": (
                results[0].ranking_freshness_weight if results else None
            ),
        }
        output["cache_snapshots"] = {
            "model_metadata": cache_snapshot(),
            "benchmarks": benchmark_cache_snapshot(),
        }
    console.console.print_json(json.dumps(output, ensure_ascii=False))


def display_plan_json(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
    image_count: int = 1,
    image_size: int = 448,
    video_frames: int = 0,
    system_ram_bytes: int | None = None,
    min_speed: float | None = None,
    os_name: str = "linux",
    perf_vram: str = "none",
) -> None:
    from hardware.catalog import PLAN_SYSTEM_RAM_BYTES
    from output.plan import (
        plan_multi_gpu_compatibility,
        plan_gpu_compatibility,
        plan_recommendations,
        plan_vram_by_quant,
    )

    system_ram_bytes = system_ram_bytes or PLAN_SYSTEM_RAM_BYTES
    vram_by_quant = plan_vram_by_quant(
        model, context_length, image_count, image_size, video_frames
    )
    single_gpu_rows = plan_gpu_compatibility(
        model,
        target_quant,
        context_length,
        image_count,
        image_size,
        video_frames,
        system_ram_bytes,
        min_speed,
        os_name,
        perf_vram,
    )
    multi_gpu_rows = plan_multi_gpu_compatibility(
        model,
        target_quant,
        context_length,
        image_count,
        image_size,
        video_frames,
        system_ram_bytes,
        min_speed,
        os_name,
        perf_vram,
    )

    output = {
        "model": {
            "id": model.id,
            "parameter_count": model.parameter_count,
            "architecture": model.architecture,
            "context_length": model.context_length,
            "license": model.license,
        },
        "target_quant": target_quant,
        "context_length": context_length,
        "workload": {
            "image_count": image_count,
            "image_size": image_size,
            "video_frames": video_frames,
            "system_ram_bytes": system_ram_bytes,
            "min_speed": min_speed,
            "os": os_name,
            "vram_headroom": "auto",
            "perf_vram": perf_vram,
        },
        "vram_by_quant": vram_by_quant,
        "gpu_compatibility": single_gpu_rows,
        "reverse_lookup": plan_recommendations(single_gpu_rows, multi_gpu_rows),
    }
    console.console.print_json(json.dumps(output, ensure_ascii=False))


def display_upgrade_json(
    current_hw: HardwareInfo,
    current_results: list,
    target_results: list[tuple[str, HardwareInfo, list]],
) -> None:
    current_row = summarize_upgrade_row("Current", current_hw, current_results)
    rows = []
    for name, hw, res in target_results:
        row = summarize_upgrade_row(name, hw, res)
        row["delta_quality"] = row["top_quality"] - current_row["top_quality"]
        row["delta_tok_s"] = row["top_tok_s"] - current_row["top_tok_s"]
        rows.append(row)
    console.console.print_json(
        json.dumps(
            {"current": current_row, "targets": rows},
            ensure_ascii=False,
        )
    )
