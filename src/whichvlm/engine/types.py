from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from whichvlm.models.types import GGUFVariant, ModelInfo

SpeedConfidence = Literal["high", "medium", "low"]
FitType = Literal["full_gpu", "partial_offload", "cpu_only"]
BenchmarkStatus = Literal["direct", "estimated", "self_reported", "none"]
BenchmarkSource = Literal[
    "direct",
    "variant",
    "base_model",
    "line_interp",
    "self_reported",
    "none",
]


@dataclass
class CompatibilityResult:
    model: ModelInfo
    gguf_variant: GGUFVariant | None
    can_run: bool
    vram_required_bytes: int
    vram_available_bytes: int
    offload_ratio: float = 0.0  # 0.0-1.0 fraction of weights spilled to CPU RAM
    estimated_tok_per_sec: float | None = None
    speed_confidence: SpeedConfidence = "medium"
    speed_range_tok_per_sec: tuple[float, float] | None = None
    speed_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0  # 0-100 for ranking
    fit_type: FitType = "full_gpu"
    benchmark_status: BenchmarkStatus = "none"
    benchmark_source: BenchmarkSource = "none"
    benchmark_confidence: float = 0.0  # 0.0-1.0 from BenchmarkEvidence
    context_fits: bool = True  # False when known model max context < requested
    uses_multi_gpu: bool = False
    multi_gpu_effective_vram_bytes: int | None = None
