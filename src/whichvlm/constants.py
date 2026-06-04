"""Compatibility shim: curated registries now live under ``whichvlm.data``.

This module re-exports the same names so existing imports
(``from whichvlm.constants import ...``) keep working. New code should import
from the specific ``whichvlm.data.*`` submodule instead.
"""

from whichvlm.data.framework import (
    FRAMEWORK_OVERHEAD_BYTES,
    MIN_COMPUTE_CAPABILITY_OLLAMA,
    MIN_COMPUTE_CAPABILITY_VLLM,
)
from whichvlm.data.gpu import (
    _GiB,
    AMD_SHARED_MEMORY_APU_MARKERS,
    GPU_BANDWIDTH,
    GPU_MEMORY_CLOCK_VARIANTS,
    NVIDIA_COMPUTE_CAPABILITY,
    VULKAN_ONLY_GPUS,
)
from whichvlm.data.lineage import (
    MODEL_GENERATION_BONUS_MAX,
    MODEL_GENERATION_PENALTY_MAX,
    MODEL_LINEAGE_VERSIONS,
)
from whichvlm.data.quantization import (
    QUANT_BYTES_PER_WEIGHT,
    QUANT_PREFERENCE_ORDER,
    QUANT_QUALITY_PENALTY,
)

__all__ = [
    "_GiB",
    "AMD_SHARED_MEMORY_APU_MARKERS",
    "FRAMEWORK_OVERHEAD_BYTES",
    "GPU_BANDWIDTH",
    "GPU_MEMORY_CLOCK_VARIANTS",
    "MIN_COMPUTE_CAPABILITY_OLLAMA",
    "MIN_COMPUTE_CAPABILITY_VLLM",
    "MODEL_GENERATION_BONUS_MAX",
    "MODEL_GENERATION_PENALTY_MAX",
    "MODEL_LINEAGE_VERSIONS",
    "NVIDIA_COMPUTE_CAPABILITY",
    "QUANT_BYTES_PER_WEIGHT",
    "QUANT_PREFERENCE_ORDER",
    "QUANT_QUALITY_PENALTY",
    "VULKAN_ONLY_GPUS",
]
