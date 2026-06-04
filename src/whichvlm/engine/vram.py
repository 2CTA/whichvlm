"""VRAM usage estimation."""

from __future__ import annotations

from whichvlm.constants import FRAMEWORK_OVERHEAD_BYTES
from whichvlm.engine.quantization import estimate_weight_bytes
from whichvlm.engine.workload import VisionWorkload
from whichvlm.models.types import GGUFVariant, ModelInfo

# Empirical KV-cache coefficient: bytes per B-active-param per K-context-token
# for FP16 K/V tensors. Calibrated against published llama.cpp memory reports
# for Qwen2.5-7B (0.45 GB @ 8K), Qwen3-32B (3.1 GB @ 32K), and Llama-3.1-70B
# (5.4 GB @ 32K with GQA), then bumped slightly because real llama.cpp also
# allocates a graph-compute buffer proportional to KV size.
_KV_BYTES_PER_BPARAM_PER_KCTX = 3.5 * 1024 * 1024  # 3.5 MB

# MoE attention scales with the *attention-layer count*, which is roughly
# proportional to active_params * this multiplier. For Qwen3-Next-80B-A3B
# (3B active, 48 layers), the multiplier lands near 4.
_MOE_ATTENTION_PARAM_MULTIPLIER = 4.0

# KV cache still follows decoder-tower scaling. VLM overhead is accounted for
# separately through component size, image-token expansion, and prefill scratch;
# per-family calibration is required before treating these as benchmark-quality
# memory numbers.


def estimate_kv_cache(model: ModelInfo, context_length: int) -> int:
    """Estimate KV cache size in bytes for a given context length.

    Dense models: KV ≈ 3 MB × params_b × ctx_k (FP16 K+V across all layers).
    MoE models: scale from active params × an empirical multiplier because
    attention shares across experts.
    """
    if model.is_moe and model.parameter_count_active:
        # Active-params × MoE multiplier gives a reasonable proxy for the
        # attention-layer footprint without needing config.num_hidden_layers.
        active_b = model.parameter_count_active / 1e9
        params_b = active_b * _MOE_ATTENTION_PARAM_MULTIPLIER
    else:
        params_b = model.parameter_count / 1e9

    ctx_k = context_length / 1024
    kv_bytes = int(params_b * ctx_k * _KV_BYTES_PER_BPARAM_PER_KCTX)
    return max(kv_bytes, 0)


def _activation_bytes(model: ModelInfo, context_length: int) -> int:
    """Activation/scratch buffer size.

    Empirically activation memory grows mildly with both model size and
    context length. The prior constant-plus-linear-param formula
    over-counted small models and under-counted long contexts.
    """
    # Use effective (active for MoE) size as the param-dependent base
    if model.is_moe and model.parameter_count_active:
        effective_p = model.parameter_count_active
    else:
        effective_p = model.parameter_count

    base = 400_000_000  # 400 MB framework activation floor
    param_term = int(effective_p * 0.08)  # ~0.08 byte/param
    ctx_term = int((context_length / 4096) * 150_000_000)  # +150 MB per 4K
    return base + param_term + ctx_term


def _component_params(model: ModelInfo, roles: set[str]) -> int:
    total = 0
    for component in model.components:
        if component.role in roles and component.parameter_count:
            total += component.parameter_count
    return total


def _effective_params(model: ModelInfo) -> int:
    if model.is_moe and model.parameter_count_active:
        return model.parameter_count_active
    return model.parameter_count


def _is_vlm(model: ModelInfo) -> bool:
    if model.hf_pipeline_tag in {
        "image-text-to-text",
        "visual-question-answering",
        "image-to-text",
    }:
        return True
    return any(
        component.role in {"vision_encoder", "projector", "processor"}
        for component in model.components
    )


def estimate_vision_overhead(model: ModelInfo, workload: VisionWorkload | None) -> int:
    """Estimate VLM-specific prefill/projector overhead.

    This is conservative scaffolding, not a benchmark claim. Use explicit
    component counts when present, then fall back to a bounded VLM heuristic.
    """
    if workload is None:
        return 0
    wl = workload.normalized()
    if wl.image_count == 0:
        return 0
    if not _is_vlm(model):
        return 0

    effective_p = _effective_params(model)
    vision_params = _component_params(model, {"vision_encoder", "projector"})
    if vision_params <= 0:
        vision_params = int(
            min(max(model.parameter_count * 0.18, 300_000_000), 4_000_000_000)
        )

    image_scale = (wl.image_size / 448) ** 2
    vision_weights = int(vision_params * 2)
    projector_scratch = int(128 * 1024**2 + vision_params * 0.15)
    image_tokens = max(1, (wl.image_size // 14) ** 2) * wl.image_count
    image_token_scratch = int(image_tokens * max(effective_p / 1e9, 1.0) * 96 * 1024)
    prefill = int((192 * 1024**2 + effective_p * 0.008) * image_scale * wl.image_count)
    return vision_weights + projector_scratch + image_token_scratch + prefill


def estimate_vram(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
    vision_workload: VisionWorkload | None = None,
) -> int:
    """Estimate total VRAM required to run a model."""
    weights = estimate_weight_bytes(model, variant)
    kv_cache = estimate_kv_cache(model, context_length)
    activation = _activation_bytes(model, context_length)
    vision = estimate_vision_overhead(model, vision_workload)
    framework = FRAMEWORK_OVERHEAD_BYTES
    return weights + kv_cache + activation + vision + framework
