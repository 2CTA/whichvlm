from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from whichvlm.constants import FRAMEWORK_OVERHEAD_BYTES
from whichvlm.engine.quantization import estimate_weight_bytes
from whichvlm.engine.workload import Workload
from whichvlm.models.types import GGUFVariant, ModelInfo

KV_BYTES_PER_BPARAM_PER_KCTX = 3.5 * 1024 * 1024
MOE_ATTENTION_PARAM_MULTIPLIER = 4.0
VramConfidence = Literal["high", "medium", "low"]
VISUAL_PIPELINE_TAGS = {
    "image-text-to-text",
    "visual-question-answering",
    "image-to-text",
    "video-text-to-text",
}
AUDIO_PIPELINE_TAGS = {"audio-text-to-text", "automatic-speech-recognition"}


@dataclass(frozen=True)
class VramComponents:
    weights: int
    kv_cache: int
    activations: int
    vision: int
    runtime_overhead: int


@dataclass(frozen=True)
class VramEstimate:
    required_bytes: int
    lower_bytes: int
    upper_bytes: int
    confidence: VramConfidence
    components: VramComponents
    notes: list[str]


@dataclass(frozen=True)
class VramCalibration:
    architecture: str
    backend: str
    quant_type: str | None
    model_format: str
    context_length: int
    image_count: int
    image_size: int
    estimate_bytes: int
    measured_peak_bytes: int


VRAM_RANGE_FACTORS: dict[VramConfidence, tuple[float, float]] = {
    "high": (0.96, 1.08),
    "medium": (0.82, 1.28),
    "low": (0.65, 1.60),
}

VRAM_CALIBRATIONS: tuple[VramCalibration, ...] = (
    VramCalibration(
        architecture="llama",
        backend="llama.cpp",
        quant_type="Q4_K_M",
        model_format="gguf",
        context_length=4096,
        image_count=0,
        image_size=0,
        estimate_bytes=7_245_493_760,
        measured_peak_bytes=7_760_000_000,
    ),
    VramCalibration(
        architecture="mixtral",
        backend="llama.cpp",
        quant_type="Q4_K_M",
        model_format="gguf",
        context_length=4096,
        image_count=0,
        image_size=0,
        estimate_bytes=29_074_327_193,
        measured_peak_bytes=31_200_000_000,
    ),
    VramCalibration(
        architecture="qwen2vl",
        backend="transformers",
        quant_type=None,
        model_format="safetensors",
        context_length=4096,
        image_count=1,
        image_size=448,
        estimate_bytes=16_769_715_744,
        measured_peak_bytes=19_900_000_000,
    ),
)


def dtype_bytes(dtype: str | None) -> float:
    if not dtype:
        return 2.0
    normalized = dtype.lower().replace("torch.", "")
    if normalized in {"float32", "fp32", "f32"}:
        return 4.0
    if normalized in {"float8", "fp8", "int8"}:
        return 1.0
    return 2.0


def resolved_head_dim(model: ModelInfo) -> int | None:
    if model.head_dim:
        return model.head_dim
    if model.hidden_size and model.attention_heads:
        return model.hidden_size // model.attention_heads
    return None


def model_format(model: ModelInfo, variant: GGUFVariant | None) -> str:
    if variant:
        return "gguf"
    return model.model_format


def backend_name(model: ModelInfo, variant: GGUFVariant | None) -> str:
    fmt = model_format(model, variant)
    if fmt == "gguf":
        return "llama.cpp"
    if fmt == "mlx":
        return "mlx"
    return "transformers"


def quant_type(model: ModelInfo, variant: GGUFVariant | None) -> str | None:
    if variant:
        return variant.quant_type
    return model.quantization_type


def has_kv_shape(model: ModelInfo) -> bool:
    return bool(
        model.layer_count
        and (model.kv_heads or model.attention_heads)
        and resolved_head_dim(model)
    )


def estimate_kv_cache(
    model: ModelInfo,
    context_length: int,
    batch_size: int = 1,
) -> int:
    kv_heads = model.kv_heads or model.attention_heads
    head_dim = resolved_head_dim(model)
    if model.layer_count and kv_heads and head_dim:
        bytes_per_value = dtype_bytes(model.kv_cache_dtype or model.dtype)
        return int(
            model.layer_count
            * kv_heads
            * head_dim
            * 2
            * context_length
            * batch_size
            * bytes_per_value
        )

    if model.is_moe and model.parameter_count_active:
        active_b = model.parameter_count_active / 1e9
        params_b = active_b * MOE_ATTENTION_PARAM_MULTIPLIER
    else:
        params_b = model.parameter_count / 1e9

    ctx_k = context_length / 1024
    kv_bytes = int(params_b * ctx_k * KV_BYTES_PER_BPARAM_PER_KCTX)
    return max(kv_bytes, 0)


def has_activation_shape(model: ModelInfo) -> bool:
    return bool(model.layer_count and model.hidden_size)


def activation_bytes(
    model: ModelInfo,
    context_length: int,
    batch_size: int = 1,
) -> int:
    if model.layer_count and model.hidden_size:
        tokens = context_length * batch_size
        bytes_per_value = dtype_bytes(model.dtype)
        head_dim = resolved_head_dim(model) or model.hidden_size
        kv_heads = model.kv_heads or model.attention_heads or 1
        ffn_size = model.intermediate_size or model.hidden_size * 4
        hidden_bytes = tokens * model.hidden_size * bytes_per_value
        attention_bytes = tokens * kv_heads * head_dim * bytes_per_value
        ffn_bytes = tokens * ffn_size * bytes_per_value
        layer_factor = max(2, min(8, model.layer_count // 8))
        expert_scratch = int((effective_params(model) / 1e9) * 64 * 1024**2)
        return int(
            (hidden_bytes * 2 + attention_bytes + ffn_bytes) * layer_factor
            + expert_scratch
        )

    if model.is_moe and model.parameter_count_active:
        effective_p = model.parameter_count_active
    else:
        effective_p = model.parameter_count

    base = 400_000_000
    param_term = int(effective_p * 0.08)
    ctx_term = int((context_length / 4096) * 150_000_000)
    return base + param_term + ctx_term


def calibration_match(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int,
    vision_workload: Workload | None,
) -> VramCalibration | None:
    wl = vision_workload.normalized() if vision_workload else None
    image_count = wl.image_count if wl else 0
    image_size = wl.image_size if wl and wl.image_count else 0
    fmt = model_format(model, variant)
    backend = backend_name(model, variant)
    quant = quant_type(model, variant)
    architecture = model.architecture.lower()
    for sample in VRAM_CALIBRATIONS:
        if not architecture.startswith(sample.architecture):
            continue
        if sample.backend != backend or sample.model_format != fmt:
            continue
        if sample.quant_type is not None and sample.quant_type != quant:
            continue
        if sample.image_count != image_count or sample.image_size != image_size:
            continue
        if sample.context_length != context_length:
            continue
        return sample
    return None


def model_confidence(
    model: ModelInfo,
    vision_workload: Workload | None,
    calibration: VramCalibration | None,
) -> VramConfidence:
    if not has_kv_shape(model):
        return "low"
    if (
        vision_workload is not None
        and supports_visual_inputs(model)
        and component_params(model, {"vision_encoder", "video_encoder", "projector"})
        <= 0
    ):
        return "low"
    if not has_activation_shape(model):
        return "medium"
    if (
        vision_workload is not None
        and supports_visual_inputs(model)
        and not (model.vision_layer_count and model.vision_hidden_size)
    ):
        return "medium"
    if calibration is None:
        return "medium"
    return "high"


def component_params(model: ModelInfo, roles: set[str]) -> int:
    total = 0
    for component in model.components:
        if component.role in roles and component.parameter_count:
            total += component.parameter_count
    return total


def effective_params(model: ModelInfo) -> int:
    if model.is_moe and model.parameter_count_active:
        return model.parameter_count_active
    return model.parameter_count


def supports_visual_inputs(model: ModelInfo) -> bool:
    if model.capabilities.image or model.capabilities.video:
        return True
    if model.hf_pipeline_tag in VISUAL_PIPELINE_TAGS:
        return True
    return any(
        component.role in {"vision_encoder", "video_encoder", "projector", "processor"}
        for component in model.components
    )


def supports_audio_inputs(model: ModelInfo) -> bool:
    if model.capabilities.audio:
        return True
    if model.hf_pipeline_tag in AUDIO_PIPELINE_TAGS:
        return True
    return any(component.role == "audio_encoder" for component in model.components)


def image_tokens(model: ModelInfo, workload: Workload) -> int:
    patch_size = model.patch_size or 14
    grid_size = workload.image_size // patch_size
    if model.spatial_merge_size:
        grid_size //= model.spatial_merge_size
    tokens_per_image = max(1, grid_size**2)
    if model.image_token_strategy == "default":
        tokens_per_image = max(1, tokens_per_image - 1)
    return tokens_per_image * (workload.image_count + workload.video_frames)


def estimate_vision_overhead(model: ModelInfo, workload: Workload | None) -> int:
    if workload is None:
        return 0
    wl = workload.normalized()
    visual_inputs = wl.image_count + wl.video_frames
    if visual_inputs == 0:
        return 0
    if not supports_visual_inputs(model):
        return 0

    effective_p = effective_params(model)
    vision_params = component_params(
        model, {"vision_encoder", "video_encoder", "projector"}
    )
    if vision_params <= 0:
        vision_params = int(
            min(max(model.parameter_count * 0.18, 300_000_000), 4_000_000_000)
        )

    image_scale = (wl.image_size / 448) ** 2
    vision_weights = int(vision_params * 2)
    projector_scratch = int(128 * 1024**2 + vision_params * 0.15)
    tokens = image_tokens(model, wl)
    language_hidden = model.hidden_size or int(max(effective_p / 1e9, 1.0) * 1024)
    image_token_scratch = int(tokens * language_hidden * dtype_bytes(model.dtype) * 4)
    if model.vision_layer_count and model.vision_hidden_size:
        bytes_per_value = dtype_bytes(model.dtype)
        vision_ffn = model.vision_intermediate_size or model.vision_hidden_size * 4
        vision_heads = model.vision_attention_heads or 1
        vision_head_dim = model.vision_hidden_size // vision_heads
        vision_layer_window = max(2, min(4, model.vision_layer_count // 8))
        vision_hidden = tokens * model.vision_hidden_size * bytes_per_value
        vision_attention = tokens * vision_heads * vision_head_dim * bytes_per_value
        vision_mlp = tokens * vision_ffn * bytes_per_value
        projector_hidden = model.projector_hidden_size or model.hidden_size or 0
        projector_scratch += int(tokens * projector_hidden * bytes_per_value)
        prefill = int(
            (vision_hidden * 2 + vision_attention + vision_mlp)
            * vision_layer_window
            * image_scale
        )
    else:
        prefill = int(
            (192 * 1024**2 + effective_p * 0.008) * image_scale * visual_inputs
        )
    return (
        vision_weights + projector_scratch + image_token_scratch + prefill
    ) * wl.batch_size


def estimate_audio_overhead(model: ModelInfo, workload: Workload | None) -> int:
    if workload is None:
        return 0
    wl = workload.normalized()
    if wl.audio_seconds <= 0 or not supports_audio_inputs(model):
        return 0
    return int((64 * 1024**2 + wl.audio_seconds * 2 * 1024**2) * wl.batch_size)


def vram_notes(
    model: ModelInfo,
    vision_workload: Workload | None,
    calibration: VramCalibration | None,
) -> list[str]:
    notes = []
    if not has_kv_shape(model):
        notes.append("KV cache uses parameter-count fallback")
    if not has_activation_shape(model):
        notes.append("activations use parameter-count fallback")
    if supports_visual_inputs(model) and vision_workload is not None:
        if (
            component_params(model, {"vision_encoder", "video_encoder", "projector"})
            <= 0
        ):
            notes.append("vision parameters use VLM size fallback")
        if not (model.vision_layer_count and model.vision_hidden_size):
            notes.append("vision activations use image-size fallback")
    if calibration is None:
        notes.append("no matching peak-memory calibration")
    return notes


def apply_calibration(required: int, calibration: VramCalibration | None) -> int:
    if calibration is None:
        return required
    return int(required * calibration.measured_peak_bytes / calibration.estimate_bytes)


def estimate_vram_details(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
    vision_workload: Workload | None = None,
    batch_size: int = 1,
) -> VramEstimate:
    workload = vision_workload.normalized() if vision_workload else None
    effective_context = workload.context_length if workload else context_length
    effective_batch = workload.batch_size if workload else batch_size
    weights = estimate_weight_bytes(model, variant)
    kv_cache = estimate_kv_cache(model, effective_context, effective_batch)
    activation = activation_bytes(model, effective_context, effective_batch)
    media = estimate_vision_overhead(model, workload) + estimate_audio_overhead(
        model, workload
    )
    subtotal = weights + kv_cache + activation + media
    calibration = calibration_match(model, variant, effective_context, workload)
    required = apply_calibration(subtotal + FRAMEWORK_OVERHEAD_BYTES, calibration)
    runtime = required - subtotal
    notes = vram_notes(model, vision_workload, calibration)
    confidence = model_confidence(model, vision_workload, calibration)
    low_factor, high_factor = VRAM_RANGE_FACTORS[confidence]
    return VramEstimate(
        required_bytes=required,
        lower_bytes=int(required * low_factor),
        upper_bytes=int(required * high_factor),
        confidence=confidence,
        components=VramComponents(
            weights=weights,
            kv_cache=kv_cache,
            activations=activation,
            vision=media,
            runtime_overhead=runtime,
        ),
        notes=notes,
    )


def estimate_vram(
    model: ModelInfo,
    variant: GGUFVariant | None,
    context_length: int = 4096,
    vision_workload: Workload | None = None,
    batch_size: int = 1,
) -> int:
    return estimate_vram_details(
        model,
        variant,
        context_length,
        vision_workload,
        batch_size,
    ).required_bytes
