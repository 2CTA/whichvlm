"""Machine-readable JSON output for ranking, plan, and upgrade surfaces."""

from __future__ import annotations

import json

from whichvlm.engine.quantization import effective_quant_type, estimate_weight_bytes
from whichvlm.engine.types import CompatibilityResult
from whichvlm.hardware.types import BackendCapability, HardwareInfo
from whichvlm.models.types import (
    GGUFVariant,
    ModelArtifact,
    ModelComponent,
    ModelInfo,
    ModelLineage,
)
from whichvlm.output import _console
from whichvlm.output.upgrade import _summarize_row


def _backend_capability_dict(capability: BackendCapability) -> dict:
    return {
        "name": capability.name,
        "available": capability.available,
        "version": capability.version,
        "details": capability.details,
    }


def _artifact_dict(artifact: ModelArtifact) -> dict:
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


def _component_dict(component: ModelComponent) -> dict:
    return {
        "role": component.role,
        "repo_id": component.repo_id,
        "parameter_count": component.parameter_count,
        "quantization": component.quantization,
    }


def _lineage_dict(lineage: ModelLineage) -> dict:
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
                        _backend_capability_dict(c) for c in g.backend_capabilities
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
                _backend_capability_dict(c) for c in hardware.backend_capabilities
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
                "artifacts": [_artifact_dict(a) for a in r.model.artifacts],
                "components": [_component_dict(c) for c in r.model.components],
                "lineage": _lineage_dict(r.model.lineage),
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
    _console.console.print_json(json.dumps(output, ensure_ascii=False))


def display_plan_json(
    model: ModelInfo,
    context_length: int,
    target_quant: str,
) -> None:
    from whichvlm.constants import (
        GPU_BANDWIDTH,
        QUANT_BYTES_PER_WEIGHT,
        QUANT_QUALITY_PENALTY,
    )
    from whichvlm.engine.performance import estimate_tok_per_sec
    from whichvlm.engine.vram import estimate_vram
    from whichvlm.hardware.types import GPUInfo

    _GiB = 1024**3

    quant_levels = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
    vram_by_quant = {}
    for qt in quant_levels:
        bpw = QUANT_BYTES_PER_WEIGHT.get(qt)
        if bpw is None:
            continue
        fake_size = int(model.parameter_count * bpw)
        fake_variant = GGUFVariant(
            filename="", quant_type=qt, file_size_bytes=fake_size
        )
        vram_bytes = estimate_vram(model, fake_variant, context_length)
        vram_by_quant[qt] = {
            "vram_bytes": vram_bytes,
            "quality_loss": QUANT_QUALITY_PENALTY.get(qt, 0.0),
        }

    target_vram = vram_by_quant.get(target_quant.upper(), {}).get("vram_bytes", 0)
    if target_vram == 0:
        bpw = QUANT_BYTES_PER_WEIGHT.get(target_quant.upper(), 0.5625)
        fake_size = int(model.parameter_count * bpw)
        fake_variant = GGUFVariant(
            filename="", quant_type=target_quant, file_size_bytes=fake_size
        )
        target_vram = estimate_vram(model, fake_variant, context_length)

    _PLAN_GPUS: list[tuple[str, int]] = [
        ("RTX 4060", 8),
        ("RTX 3060", 12),
        ("RTX 4070", 12),
        ("RTX 4080", 16),
        ("RTX 4090", 24),
        ("RX 7900 XTX", 24),
        ("RTX 5090", 32),
        ("A100 40GB", 40),
        ("L40S", 48),
        ("A100 80GB", 80),
        ("H100", 80),
        ("H200", 141),
    ]

    bpw = QUANT_BYTES_PER_WEIGHT.get(target_quant.upper(), 0.5625)
    fake_size = int(model.parameter_count * bpw)
    fake_variant = GGUFVariant(
        filename="", quant_type=target_quant, file_size_bytes=fake_size
    )

    gpus = []
    for gpu_name, vram_gb in _PLAN_GPUS:
        vram_bytes = int(vram_gb * _GiB)
        bandwidth = GPU_BANDWIDTH.get(gpu_name)
        gpu_info = GPUInfo(
            name=gpu_name,
            vendor="nvidia",
            vram_bytes=vram_bytes,
            memory_bandwidth_gbps=bandwidth,
        )
        if vram_bytes >= target_vram:
            fit_type = "full_gpu"
        elif vram_bytes >= target_vram * 0.4:
            fit_type = "partial_offload"
        else:
            fit_type = "too_small"

        speed = None
        if fit_type != "too_small" and bandwidth:
            speed = round(
                estimate_tok_per_sec(model, fake_variant, gpu_info, fit_type), 1
            )

        gpus.append(
            {
                "name": gpu_name,
                "vram_gb": vram_gb,
                "fit_type": fit_type,
                "estimated_tok_per_sec": speed,
            }
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
        "gpu_compatibility": gpus,
    }
    _console.console.print_json(json.dumps(output, ensure_ascii=False))


def display_upgrade_json(
    current_hw: HardwareInfo,
    current_results: list,
    target_results: list[tuple[str, HardwareInfo, list]],
) -> None:
    """Emit the upgrade comparison as JSON for scripting."""
    current_row = _summarize_row("Current", current_hw, current_results)
    rows = []
    for name, hw, res in target_results:
        row = _summarize_row(name, hw, res)
        row["delta_quality"] = row["top_quality"] - current_row["top_quality"]
        row["delta_tok_s"] = row["top_tok_s"] - current_row["top_tok_s"]
        rows.append(row)
    _console.console.print_json(
        json.dumps(
            {"current": current_row, "targets": rows},
            ensure_ascii=False,
        )
    )
