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


def display_json(results: list[CompatibilityResult], hardware: HardwareInfo) -> None:
    output = {
        "hardware": {
            "gpus": [
                {
                    "name": g.name,
                    "vendor": g.vendor,
                    "vram_bytes": g.vram_bytes,
                    "usable_vram_bytes": g.usable_vram_bytes,
                    "memory_bandwidth_gbps": g.memory_bandwidth_gbps,
                    "shared_memory": g.shared_memory,
                    "backend_capabilities": [
                        backend_capability_dict(c) for c in g.backend_capabilities
                    ],
                    "neural_engine_available": g.neural_engine_available,
                }
                for g in hardware.gpus
            ],
            "cpu": hardware.cpu_name,
            "cpu_cores": hardware.cpu_cores,
            "ram_bytes": hardware.ram_bytes,
            "ram_budget_bytes": hardware.ram_budget_bytes,
            "budget_notes": hardware.budget_notes,
            "os": hardware.os,
            "backend_capabilities": [
                backend_capability_dict(c) for c in hardware.backend_capabilities
            ],
        },
        "models": [
            {
                "rank": i,
                "model_id": r.model.id,
                "family_id": r.model.family_id,
                "architecture": r.model.architecture,
                "hf_pipeline_tag": r.model.hf_pipeline_tag,
                "tags": r.model.tags,
                "access": r.model.access,
                "is_official": r.model.is_official,
                "model_format": r.model.model_format,
                "variant_kind": r.model.variant_kind,
                "quantization_type": r.model.quantization_type,
                "base_model": r.model.base_model,
                "base_models": r.model.base_models,
                "variant_of": r.model.variant_of,
                "artifacts": [artifact_dict(a) for a in r.model.artifacts],
                "components": [component_dict(c) for c in r.model.components],
                "lineage": lineage_dict(r.model.lineage),
                "parameter_count": r.model.parameter_count,
                "published_at": r.model.published_at,
                "downloads": r.model.downloads,
                "quant_type": effective_quant_type(r.model, r.gguf_variant),
                "file_size_bytes": (
                    r.gguf_variant.file_size_bytes
                    if r.gguf_variant
                    else estimate_weight_bytes(r.model, None)
                ),
                "vram_required_bytes": r.vram_required_bytes,
                "vram_available_bytes": r.vram_available_bytes,
                "uses_multi_gpu": r.uses_multi_gpu,
                "multi_gpu_effective_vram_bytes": r.multi_gpu_effective_vram_bytes,
                "estimated_tok_per_sec": r.estimated_tok_per_sec,
                "speed_confidence": r.speed_confidence,
                "speed_range_tok_per_sec": (
                    list(r.speed_range_tok_per_sec)
                    if r.speed_range_tok_per_sec
                    else None
                ),
                "speed_notes": r.speed_notes,
                "quality_score": round(r.quality_score, 2),
                "benchmark_status": r.benchmark_status,
                "benchmark_source": r.benchmark_source,
                "benchmark_confidence": round(r.benchmark_confidence, 2),
                "fit_type": r.fit_type,
                "can_run": r.can_run,
                "warnings": r.warnings,
                "license": r.model.license,
            }
            for i, r in enumerate(results, 1)
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
