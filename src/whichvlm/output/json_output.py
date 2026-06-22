from __future__ import annotations

import json

from whichvlm.engine.quantization import effective_quant_type, estimate_weight_bytes
from whichvlm.engine.types import CompatibilityResult
from whichvlm.hardware.types import BackendCapability, HardwareInfo
from whichvlm.models.types import (
    ModelArtifact,
    ModelComponent,
    ModelInfo,
    ModelLineage,
)
from whichvlm.output import console
from whichvlm.output.upgrade import summarize_upgrade_row


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


def model_dict(rank: int, result: CompatibilityResult, details: bool = False) -> dict:
    model = result.model
    data = {
        "rank": rank,
        "model_id": model.id,
        "parameter_count": model.parameter_count,
        "license": model.license,
        "quant_type": effective_quant_type(model, result.gguf_variant),
        "file_size_bytes": (
            result.gguf_variant.file_size_bytes
            if result.gguf_variant
            else estimate_weight_bytes(model, None)
        ),
        "vram_required_bytes": result.vram_required_bytes,
        "vram_available_bytes": result.vram_available_bytes,
        "estimated_tok_per_sec": result.estimated_tok_per_sec,
        "benchmark_status": result.benchmark_status,
        "benchmark_source": result.benchmark_source,
        "fit_type": result.fit_type,
        "can_run": result.can_run,
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
            model_dict(i, result, details)
            for i, result in enumerate(results, 1)
        ],
    }
    console.console.print_json(json.dumps(output, ensure_ascii=False))


def display_plan_json(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
) -> None:
    from whichvlm.output.plan import (
        plan_gpu_compatibility,
        plan_target_vram,
        plan_vram_by_quant,
    )

    vram_by_quant = plan_vram_by_quant(model, context_length)
    target_vram = plan_target_vram(
        model, context_length, target_quant, vram_by_quant
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
        "vram_by_quant": vram_by_quant,
        "gpu_compatibility": plan_gpu_compatibility(
            model, target_quant, target_vram
        ),
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
