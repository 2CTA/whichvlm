"""VLM package graph parsing and serialization helpers."""

from __future__ import annotations

import re

from whichvlm.models.types import (
    GGUFVariant,
    ModelArtifact,
    ModelComponent,
    ModelInfo,
    ModelLineage,
)


def looks_quantized_repo_name(model_id: str) -> bool:
    lower = model_id.lower()
    return bool(
        re.search(r"(gptq|awq|bnb|4bit|int4|int8|fp8|gguf|mlx|quant)", lower)
    )


def artifact_format(model_format: str, quantization_type: str | None) -> str:
    if quantization_type:
        lower = quantization_type.lower()
        if lower in {"awq", "gptq", "fp8", "mlx"}:
            return lower
        if lower == "bnb_4bit":
            return "bnb"
    if model_format in {"safetensors", "gguf", "mlx"}:
        return model_format
    if model_format == "quantized":
        return (quantization_type or "other").lower()
    return "other"


def backend_support_for_artifact(
    artifact_format_value: str,
    quantization_type: str | None,
) -> list[str]:
    fmt = artifact_format_value.lower()
    quant = (quantization_type or "").upper()
    if fmt == "gguf":
        return ["metal", "cuda", "vulkan", "cpu"]
    if fmt == "mlx":
        return ["mlx", "metal"]
    if fmt == "safetensors":
        return ["cuda", "mps", "cpu"]
    if fmt in {"awq", "gptq", "bnb", "fp8"} or quant in {
        "AWQ",
        "GPTQ",
        "BNB_4BIT",
        "FP8",
    }:
        return ["cuda"]
    return ["cpu"]


def is_projector_filename(filename: str) -> bool:
    lower = filename.lower()
    return "mmproj" in lower or "projector" in lower


def is_vision_model(model_id: str, pipeline_tag: object, tags: list[str]) -> bool:
    haystack = " ".join([model_id, str(pipeline_tag or ""), *tags]).lower()
    return bool(
        re.search(
            r"(image-text-to-text|visual-question-answering|image-to-text|"
            r"vision-language|multimodal|qwen.*vl|llava|pixtral|internvl|"
            r"deepseek-vl|vision)",
            haystack,
        )
    )


def lineage_relationship(
    card_data: dict,
    base_models: list[str],
    tags: list[str],
) -> str:
    raw = card_data.get("base_model_relation") or card_data.get("model_relation")
    if isinstance(raw, str) and raw:
        return raw.lower()
    haystack = " ".join(tags).lower()
    if len(base_models) > 1 or re.search(r"(merge|merged|fused|fusion)", haystack):
        return "merged"
    if base_models:
        return "variant"
    return "unknown"


def build_lineage(
    base_models: list[str],
    tags: list[str],
    card_data: dict,
) -> ModelLineage:
    relationship = lineage_relationship(card_data, base_models, tags)
    is_merged = relationship in {"merge", "merged", "fused", "fusion"} or len(
        base_models
    ) > 1
    return ModelLineage(
        base_model_ids=list(base_models),
        merged_parent_ids=list(base_models) if is_merged else [],
        variant_of=base_models[0] if base_models and not is_merged else None,
        relationship="merged" if is_merged else relationship,
        is_merged=is_merged,
    )


def build_artifacts(
    model_id: str,
    *,
    model_format: str,
    quantization_type: str | None,
    access: str,
    variant_kind: str,
    gguf_variants: list[GGUFVariant],
    parameter_count: int,
    projector_files: list[tuple[str, int | None]] | None = None,
) -> list[ModelArtifact]:
    projector_artifacts = [
        ModelArtifact(
            repo_id=model_id,
            format="adapter",
            quantization=None,
            file_size_bytes=size,
            access=access,
            backend_support=["metal", "cuda", "vulkan", "cpu"],
            source_kind="mmproj",
            filename=filename,
        )
        for filename, size in (projector_files or [])
    ]

    if gguf_variants:
        artifacts = [
            ModelArtifact(
                repo_id=model_id,
                format="gguf",
                quantization=v.quant_type,
                file_size_bytes=v.file_size_bytes,
                access=access,
                backend_support=backend_support_for_artifact("gguf", v.quant_type),
                source_kind=variant_kind,
                filename=v.filename,
            )
            for v in gguf_variants
        ]
        artifacts.extend(projector_artifacts)
        return artifacts

    fmt = artifact_format(model_format, quantization_type)
    file_size = parameter_count * 2 if fmt == "safetensors" else None
    artifacts = [
        ModelArtifact(
            repo_id=model_id,
            format=fmt,
            quantization=quantization_type,
            file_size_bytes=file_size,
            access=access,
            backend_support=backend_support_for_artifact(fmt, quantization_type),
            source_kind=variant_kind,
        )
    ]
    artifacts.extend(projector_artifacts)
    return artifacts


def build_components(
    model_id: str,
    *,
    parameter_count: int,
    quantization_type: str | None,
    pipeline_tag: object,
    tags: list[str],
    lineage: ModelLineage,
) -> list[ModelComponent]:
    if lineage.is_merged:
        return [
            ModelComponent(
                role="merged_checkpoint",
                repo_id=model_id,
                parameter_count=parameter_count,
                quantization=quantization_type,
            )
        ]
    if not is_vision_model(model_id, pipeline_tag, tags):
        return []
    return [
        ModelComponent(
            role="language",
            repo_id=model_id,
            parameter_count=parameter_count,
            quantization=quantization_type,
        ),
        ModelComponent(role="vision_encoder", repo_id=model_id),
        ModelComponent(role="projector", repo_id=model_id),
        ModelComponent(role="processor", repo_id=model_id),
    ]


def infer_variant_kind(
    *,
    model_id: str,
    base_models: list[str],
    model_format: str,
    is_official: bool,
    tags: list[str],
    card_data: dict | None = None,
) -> str:
    relationship = lineage_relationship(card_data or {}, base_models, tags)
    if relationship in {"merge", "merged", "fused", "fusion"} or len(base_models) > 1:
        return "merged_model"
    if base_models:
        if model_format == "gguf":
            return "gguf_variant"
        if model_format == "mlx":
            return "mlx_variant"
        if model_format == "quantized" or looks_quantized_repo_name(model_id):
            return "quantized_variant"
        return "derived_variant"
    if is_official:
        return "official"
    if model_format == "gguf":
        return "community_gguf"
    if model_format == "mlx":
        return "community_mlx"
    if model_format == "quantized" or looks_quantized_repo_name(model_id):
        return "community_quantization"
    return "community"


def artifact_to_dict(artifact: ModelArtifact) -> dict:
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


def artifact_from_dict(data: dict) -> ModelArtifact:
    return ModelArtifact(
        repo_id=data.get("repo_id", ""),
        format=data.get("format", "other"),
        quantization=data.get("quantization"),
        file_size_bytes=data.get("file_size_bytes"),
        access=data.get("access", "unknown"),
        backend_support=[
            str(v) for v in data.get("backend_support", []) if isinstance(v, str)
        ],
        source_kind=data.get("source_kind", "unknown"),
        filename=data.get("filename"),
    )


def component_to_dict(component: ModelComponent) -> dict:
    return {
        "role": component.role,
        "repo_id": component.repo_id,
        "parameter_count": component.parameter_count,
        "quantization": component.quantization,
    }


def component_from_dict(data: dict) -> ModelComponent:
    return ModelComponent(
        role=data.get("role", "language"),
        repo_id=data.get("repo_id", ""),
        parameter_count=data.get("parameter_count"),
        quantization=data.get("quantization"),
    )


def lineage_to_dict(lineage: ModelLineage) -> dict:
    return {
        "base_model_ids": lineage.base_model_ids,
        "merged_parent_ids": lineage.merged_parent_ids,
        "variant_of": lineage.variant_of,
        "relationship": lineage.relationship,
        "is_merged": lineage.is_merged,
    }


def lineage_from_dict(data: dict | None, base_model: str | None) -> ModelLineage:
    if not isinstance(data, dict):
        base_models = [base_model] if base_model else []
        return ModelLineage(base_model_ids=base_models, variant_of=base_model)
    return ModelLineage(
        base_model_ids=[
            str(v) for v in data.get("base_model_ids", []) if isinstance(v, str)
        ],
        merged_parent_ids=[
            str(v) for v in data.get("merged_parent_ids", []) if isinstance(v, str)
        ],
        variant_of=data.get("variant_of"),
        relationship=data.get("relationship", "unknown"),
        is_merged=bool(data.get("is_merged", False)),
    )


def _artifact_key(artifact: ModelArtifact) -> tuple:
    return (
        artifact.repo_id,
        artifact.format,
        artifact.quantization,
        artifact.filename,
        artifact.file_size_bytes,
    )


def _component_key(component: ModelComponent) -> tuple:
    return (
        component.role,
        component.repo_id,
        component.parameter_count,
        component.quantization,
    )


def _merge_family_lineage(group: list[ModelInfo]) -> ModelLineage:
    base_ids: list[str] = []
    merged_ids: list[str] = []
    variant_of = None
    relationship = "unknown"
    is_merged = False

    for model in group:
        for base_id in model.base_models or ([] if not model.base_model else [model.base_model]):
            if base_id not in base_ids:
                base_ids.append(base_id)
        for parent_id in model.lineage.merged_parent_ids:
            if parent_id not in merged_ids:
                merged_ids.append(parent_id)
        if variant_of is None and model.lineage.variant_of:
            variant_of = model.lineage.variant_of
        if model.lineage.is_merged:
            is_merged = True
            relationship = "merged"
        elif relationship == "unknown" and model.lineage.relationship != "unknown":
            relationship = model.lineage.relationship

    if len(base_ids) > 1 and not merged_ids:
        merged_ids = list(base_ids)
        is_merged = True
        relationship = "merged"

    return ModelLineage(
        base_model_ids=base_ids,
        merged_parent_ids=merged_ids,
        variant_of=None if is_merged else variant_of,
        relationship=relationship,
        is_merged=is_merged,
    )


def merge_family_graph(
    group: list[ModelInfo],
) -> tuple[list[ModelArtifact], list[ModelComponent], ModelLineage]:
    artifacts_by_key: dict[tuple, ModelArtifact] = {}
    components_by_key: dict[tuple, ModelComponent] = {}
    for model in group:
        for artifact in model.artifacts:
            artifacts_by_key.setdefault(_artifact_key(artifact), artifact)
        for component in model.components:
            components_by_key.setdefault(_component_key(component), component)
    return (
        list(artifacts_by_key.values()),
        list(components_by_key.values()),
        _merge_family_lineage(group),
    )
