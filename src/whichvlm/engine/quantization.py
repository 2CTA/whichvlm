from __future__ import annotations

import re

from whichvlm.constants import QUANT_QUALITY_PENALTY
from whichvlm.models.types import GGUFVariant, ModelInfo


NON_GGUF_PATTERNS: list[tuple[str, str]] = [
    (r"(^|[-_/])awq($|[-_/])", "AWQ"),
    (r"(^|[-_/])gptq($|[-_/])", "GPTQ"),


    (r"(^|[-_/])mxfp4($|[-_/])", "MXFP4"),
    (r"(^|[-_/])nvfp4($|[-_/])", "NVFP4"),
    (r"(bnb[-_/]?4bit|nf4|int4|4bit)", "BNB_4BIT"),
    (r"(int8|8bit)", "INT8"),
    (r"(^|[-_/])fp8($|[-_/])", "FP8"),
    (r"(^|[-_/])bf16($|[-_/])", "BF16"),
    (r"(^|[-_/])(fp16|f16)($|[-_/])", "FP16"),
]


NON_GGUF_BYTES_PER_WEIGHT: dict[str, float] = {
    "AWQ": 0.5,
    "GPTQ": 0.5,
    "BNB_4BIT": 0.5,
    "MXFP4": 0.53125,
    "NVFP4": 0.5625,
    "INT8": 1.0,
    "FP8": 1.0,
    "BF16": 2.0,
    "FP16": 2.0,
}


NON_GGUF_QUALITY_PENALTY: dict[str, float] = {
    "AWQ": 0.05,
    "GPTQ": 0.05,
    "BNB_4BIT": 0.07,
    "MXFP4": 0.06,
    "NVFP4": 0.05,
    "INT8": 0.02,
    "FP8": 0.02,
    "BF16": 0.0,
    "FP16": 0.0,
}


def infer_non_gguf_quant_type(model_id: str) -> str:
    lower = model_id.lower()
    for pattern, quant_type in NON_GGUF_PATTERNS:
        if re.search(pattern, lower):
            return quant_type
    return "FP16"


def effective_quant_type(model: ModelInfo, variant: GGUFVariant | None) -> str:
    if variant:
        return variant.quant_type.upper()
    return infer_non_gguf_quant_type(model.id)


def bytes_per_weight(quant_type: str) -> float:
    return NON_GGUF_BYTES_PER_WEIGHT.get(quant_type.upper(), 2.0)


def estimate_weight_bytes(model: ModelInfo, variant: GGUFVariant | None) -> int:
    if variant:
        return variant.file_size_bytes
    quant_type = infer_non_gguf_quant_type(model.id)
    if model.components:
        known_params = 0
        component_bytes = 0.0
        for component in model.components:
            if component.role in {"processor", "tokenizer"}:
                continue
            if not component.parameter_count:
                continue
            known_params += component.parameter_count
            component_quant = component.quantization or quant_type
            component_bytes += component.parameter_count * bytes_per_weight(
                component_quant
            )
        if known_params > 0:
            remaining = max(0, model.parameter_count - known_params)
            component_bytes += remaining * bytes_per_weight(quant_type)
            return int(component_bytes)

    return int(model.parameter_count * bytes_per_weight(quant_type))


def quant_quality_penalty(model: ModelInfo, variant: GGUFVariant | None) -> float:
    quant_type = effective_quant_type(model, variant).upper()
    if quant_type in QUANT_QUALITY_PENALTY:
        return QUANT_QUALITY_PENALTY[quant_type]
    return NON_GGUF_QUALITY_PENALTY.get(quant_type, 0.05)
