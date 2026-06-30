from __future__ import annotations

from whichvlm.engine.quantization import estimate_weight_bytes
from whichvlm.engine.quantization import effective_quant_type
from whichvlm.hardware.types import GPUInfo
from whichvlm.models.package_graph import is_vision_model
from whichvlm.models.types import GGUFVariant, ModelInfo

# Speed model. Uses bandwidth, quant, and fit as the main signals.

QUANT_EFFICIENCY: dict[str, float] = {
    "F32": 0.30,
    "F16": 0.40,
    "BF16": 0.40,
    "Q8_0": 0.45,
    "Q6_K": 0.50,
    "Q5_K_M": 0.52,
    "Q5_K_S": 0.52,
    "Q5_0": 0.50,
    "Q4_K_M": 0.55,
    "Q4_K_S": 0.55,
    "Q4_0": 0.53,


    "NVFP4": 0.56,
    "MXFP4": 0.55,
    "Q3_K_M": 0.50,
    "Q3_K_S": 0.48,
    "Q3_K_L": 0.50,
    "Q2_K": 0.45,
    "IQ4_XS": 0.52,
    "IQ4_NL": 0.50,
    "IQ3_S": 0.45,
    "IQ3_M": 0.45,
    "IQ3_XS": 0.45,
    "IQ3_XXS": 0.42,
    "IQ2_S": 0.40,
    "IQ2_M": 0.40,
    "IQ2_XXS": 0.38,
    "IQ1_M": 0.35,
    "IQ1_S": 0.35,
    "Q2_0": 0.38,
    "Q1_0": 0.32,
    "TQ2_0": 0.35,
    "TQ1_0": 0.32,
}

DEFAULT_QUANT_EFFICIENCY = 0.45


BACKEND_FACTOR: dict[str, float] = {
    "nvidia": 1.00,
    "amd": 0.78,
    "apple": 0.82,
    "intel": 0.65,
}


MOE_REFERENCE_BANDWIDTH_GBPS = 256.0
MOE_MIN_READ_RATIO_AT_REFERENCE = 0.05
MOE_MAX_READ_RATIO_FLOOR = 0.25

SPEED_CONFIDENCE_RANGE_FACTORS: dict[str, tuple[float, float]] = {
    "high": (0.85, 1.20),
    "medium": (0.60, 1.60),
    "low": (0.35, 2.00),
}

SPEED_CONFIDENCE_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def backend_factor(gpu: GPUInfo) -> float:
    if gpu.vendor in BACKEND_FACTOR:
        return BACKEND_FACTOR[gpu.vendor]
    return 0.7


def quant_efficiency(model: ModelInfo, variant: GGUFVariant | None) -> float:
    quant = effective_quant_type(model, variant)
    if not quant:
        return DEFAULT_QUANT_EFFICIENCY
    return QUANT_EFFICIENCY.get(quant.upper(), DEFAULT_QUANT_EFFICIENCY)


def moe_effective_read_ratio(model: ModelInfo, gpu: GPUInfo) -> float:
    # MoE read model. Shrinks stored params down to active token reads.
    if not model.is_moe or not model.parameter_count_active:
        return 1.0
    if model.parameter_count <= 0:
        return 1.0

    active_ratio = model.parameter_count_active / model.parameter_count
    if active_ratio <= 0:
        return 1.0

    bandwidth = gpu.memory_bandwidth_gbps or 0.0
    if bandwidth > 0:
        floor = MOE_MIN_READ_RATIO_AT_REFERENCE * max(
            1.0, bandwidth / MOE_REFERENCE_BANDWIDTH_GBPS
        )
    else:
        floor = MOE_MAX_READ_RATIO_FLOOR
    floor = min(MOE_MAX_READ_RATIO_FLOOR, floor)

    return min(1.0, max(active_ratio, floor))


def lower_speed_confidence(current: str, candidate: str) -> str:
    if SPEED_CONFIDENCE_ORDER[candidate] < SPEED_CONFIDENCE_ORDER[current]:
        return candidate
    return current


def looks_synthetic_gguf(model: ModelInfo, variant: GGUFVariant | None) -> bool:
    if variant is None:
        return False
    if not variant.filename:
        return False
    expected = f"{model.name}.{variant.quant_type}.gguf"
    return variant.filename == expected


def is_vlm_model(model: ModelInfo) -> bool:
    if is_vision_model(
        model.id, model.hf_pipeline_tag, model.tags, model.architecture
    ):
        return True
    return any(
        component.role in {"vision_encoder", "projector", "processor"}
        for component in model.components
    )


def vlm_decode_factor(model: ModelInfo, gpu: GPUInfo | None, fit_type: str) -> float:
    if not is_vlm_model(model):
        return 1.0
    factor = 0.78
    if fit_type == "partial_offload":
        factor *= 0.90
    if gpu is not None and gpu.vendor == "apple":
        factor *= 0.95
    return factor


def estimate_speed_uncertainty(
    model: ModelInfo,
    variant: GGUFVariant | None,
    gpu: GPUInfo | None,
    fit_type: str,
    estimated_tok_per_sec: float | None,
) -> tuple[str, tuple[float, float] | None, list[str]]:
    # Confidence layer. Explains how shaky the speed point estimate is.
    notes = [
        "Speed is estimated from memory bandwidth, quantization, backend, and fit type."
    ]
    confidence = "medium"

    if estimated_tok_per_sec is None or estimated_tok_per_sec <= 0:
        return (
            "low",
            None,
            notes + ["No usable bandwidth estimate was available for this setup."],
        )

    if gpu is None or fit_type == "cpu_only":
        confidence = "low"
        notes.append(
            "CPU-only speed varies heavily with memory channels and BLAS/kernel path."
        )
    else:
        if not gpu.memory_bandwidth_gbps:
            confidence = "low"
            notes.append(
                "GPU memory bandwidth is unknown, so speed is especially uncertain."
            )

        if fit_type == "partial_offload":
            confidence = "low"
            if gpu.vendor == "apple" or gpu.shared_memory:
                notes.append(
                    "Partial offload on unified memory is backend-sensitive but avoids a PCIe cliff."
                )
            else:
                notes.append(
                    "Partial offload on a discrete GPU depends strongly on PCIe and CPU RAM bandwidth."
                )

        if model.is_moe:
            notes.append(
                "MoE speed uses active parameters plus a bandwidth-scaled dispatch/read floor."
            )
            if gpu.vendor == "apple":
                confidence = lower_speed_confidence(confidence, "low")
                notes.append(
                    "Apple Silicon MoE throughput is especially sensitive to Metal/MLX runtime kernels."
                )
            elif gpu.vendor == "amd" and gpu.shared_memory:
                confidence = lower_speed_confidence(confidence, "medium")
                notes.append(
                    "AMD shared-memory APU estimates are calibrated by bandwidth, but ROCm/Vulkan kernels can differ."
                )

    if looks_synthetic_gguf(model, variant):
        confidence = lower_speed_confidence(confidence, "medium")
        notes.append(
            "This is a synthetic GGUF estimate for an official repo, not a measured GGUF file."
        )

    if is_vlm_model(model):
        confidence = lower_speed_confidence(confidence, "medium")
        notes.append(
            "VLM speed includes a conservative discount for image prefill and projector overhead."
        )

    low_factor, high_factor = SPEED_CONFIDENCE_RANGE_FACTORS[confidence]
    speed_range = (
        round(estimated_tok_per_sec * low_factor, 1),
        round(estimated_tok_per_sec * high_factor, 1),
    )
    return confidence, speed_range, notes


def estimate_tok_per_sec(
    model: ModelInfo,
    variant: GGUFVariant | None,
    gpu: GPUInfo | None,
    fit_type: str = "full_gpu",
) -> float:
    # Main speed pass. Estimates decode rate for one model and backend.
    if gpu is None or fit_type == "cpu_only":
        params_b = model.parameter_count / 1e9
        if model.is_moe and model.parameter_count_active:
            params_b = model.parameter_count_active / 1e9
        if params_b <= 0:
            return 0.0


        quant_factor = quant_efficiency(model, variant) / DEFAULT_QUANT_EFFICIENCY
        text_speed = max(0.3, 18.0 / max(params_b, 0.5) * quant_factor)
        return text_speed * vlm_decode_factor(model, gpu, fit_type)

    model_size = estimate_weight_bytes(model, variant)


    if model.is_moe and model.parameter_count_active:
        effective_read = model_size * moe_effective_read_ratio(model, gpu)
    else:
        effective_read = model_size

    bandwidth = gpu.memory_bandwidth_gbps * 1e9 if gpu.memory_bandwidth_gbps else 0
    if bandwidth == 0:
        return 0.0

    theoretical = bandwidth / effective_read


    efficiency = quant_efficiency(model, variant) * backend_factor(gpu)


    if fit_type == "partial_offload":
        if gpu.vendor == "apple" or gpu.shared_memory:
            efficiency *= 0.85
        else:
            efficiency *= 0.45

    return theoretical * efficiency * vlm_decode_factor(model, gpu, fit_type)
