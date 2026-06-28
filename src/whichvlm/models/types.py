from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GGUFVariant:
    filename: str
    quant_type: str
    file_size_bytes: int


@dataclass
class ModelArtifact:
    repo_id: str
    format: str
    quantization: str | None = None
    file_size_bytes: int | None = None
    access: str = "unknown"
    backend_support: list[str] = field(default_factory=list)
    source_kind: str = "unknown"
    filename: str | None = None


@dataclass
class ModelComponent:
    role: str
    repo_id: str
    parameter_count: int | None = None
    quantization: str | None = None


@dataclass
class ModelLineage:
    base_model_ids: list[str] = field(default_factory=list)
    merged_parent_ids: list[str] = field(default_factory=list)
    variant_of: str | None = None
    relationship: str = "unknown"
    is_merged: bool = False


@dataclass
class ModelInfo:
    id: str
    family_id: str
    name: str
    parameter_count: int
    parameter_count_active: int | None = None
    architecture: str = ""
    is_moe: bool = False
    context_length: int | None = None
    layer_count: int | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    attention_heads: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    dtype: str | None = None
    kv_cache_dtype: str | None = None
    vision_layer_count: int | None = None
    vision_hidden_size: int | None = None
    vision_intermediate_size: int | None = None
    vision_attention_heads: int | None = None
    projector_hidden_size: int | None = None
    patch_size: int | None = None
    spatial_merge_size: int | None = None
    image_token_strategy: str | None = None
    license: str | None = None
    published_at: str | None = None
    downloads: int = 0
    likes: int = 0
    gguf_variants: list[GGUFVariant] = field(default_factory=list)
    benchmark_scores: dict[str, float] = field(default_factory=dict)
    base_model: str | None = None
    hf_pipeline_tag: str | None = None
    tags: list[str] = field(default_factory=list)
    access: str = "unknown"
    is_official: bool = False
    model_format: str = "unknown"
    variant_kind: str = "base"
    quantization_type: str | None = None
    variant_of: str | None = None
    base_models: list[str] = field(default_factory=list)
    artifacts: list[ModelArtifact] = field(default_factory=list)
    components: list[ModelComponent] = field(default_factory=list)
    lineage: ModelLineage = field(default_factory=ModelLineage)

    def __post_init__(self) -> None:
        if self.base_model and not self.base_models:
            self.base_models = [self.base_model]
        elif self.base_models and self.base_model is None:
            self.base_model = self.base_models[0]

        lineage_empty = (
            not self.lineage.base_model_ids
            and not self.lineage.merged_parent_ids
            and self.lineage.variant_of is None
            and self.lineage.relationship == "unknown"
            and not self.lineage.is_merged
        )
        if self.base_models and lineage_empty:
            is_merged = len(self.base_models) > 1
            self.lineage = ModelLineage(
                base_model_ids=list(self.base_models),
                merged_parent_ids=list(self.base_models) if is_merged else [],
                variant_of=None if is_merged else self.base_models[0],
                relationship="merged" if is_merged else "variant",
                is_merged=is_merged,
            )
        if self.variant_of is None and self.lineage.variant_of:
            self.variant_of = self.lineage.variant_of


@dataclass
class ModelFamily:
    family_id: str
    display_name: str
    base_model: ModelInfo
    variants: list[ModelInfo] = field(default_factory=list)
    best_benchmark: dict[str, float] = field(default_factory=dict)
    artifacts: list[ModelArtifact] = field(default_factory=list)
    components: list[ModelComponent] = field(default_factory=list)
    lineage: ModelLineage = field(default_factory=ModelLineage)
