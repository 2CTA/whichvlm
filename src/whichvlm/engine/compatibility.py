from __future__ import annotations

from whichvlm.constants import BYTES_PER_GIB
from whichvlm.constants import MIN_COMPUTE_CAPABILITY_OLLAMA
from whichvlm.constants import VULKAN_ONLY_GPUS
from whichvlm.engine.quantization import estimate_weight_bytes
from whichvlm.engine.types import CompatibilityResult
from whichvlm.engine.vram import estimate_vram
from whichvlm.engine.workload import VisionWorkload
from whichvlm.hardware.memory import effective_usable_ram
from whichvlm.hardware.types import GPUInfo, HardwareInfo
from whichvlm.models.types import GGUFVariant, ModelInfo

# Fit layer. Turns memory pools into full, partial, or cpu-only results.
MULTI_GPU_FRAMEWORK_OVERHEAD_BYTES = int(0.3 * BYTES_PER_GIB)
MULTI_GPU_HOMOGENEOUS_UTILIZATION = 0.95
MULTI_GPU_HETEROGENEOUS_UTILIZATION = 0.90


def gpu_available_memory(
    gpu: GPUInfo, usable_ram: int, *, ram_budget_active: bool = False
) -> int:
    # Pool calculator. Resolves the bytes this GPU can really use.
    vram_bytes = (
        gpu.usable_vram_bytes if gpu.usable_vram_bytes is not None else gpu.vram_bytes
    )
    if gpu.shared_memory and vram_bytes < 2 * BYTES_PER_GIB:
        return usable_ram
    if gpu.shared_memory and ram_budget_active:
        return min(vram_bytes, usable_ram)
    return vram_bytes


def uses_shared_system_pool(gpu: GPUInfo) -> bool:
    return gpu.shared_memory and gpu.vram_bytes < 2 * BYTES_PER_GIB


def is_vulkan_only_gpu(gpu: GPUInfo) -> bool:

    if gpu.vendor != "nvidia":
        return False
    name_upper = gpu.name.upper()
    return any(marker.upper() in name_upper for marker in VULKAN_ONLY_GPUS)


def fit_candidate_gpus(gpus: list[GPUInfo]) -> list[GPUInfo]:
    has_dedicated_gpu = any(
        not uses_shared_system_pool(gpu) and gpu.vram_bytes > 0 for gpu in gpus
    )
    if not has_dedicated_gpu:
        return gpus
    return [gpu for gpu in gpus if not uses_shared_system_pool(gpu)]


def gpu_identity(gpu: GPUInfo) -> str:
    name = gpu.name.lower().replace("(simulated)", "")
    return " ".join(name.split())


def is_homogeneous_gpu_set(gpus: list[GPUInfo], available: list[int]) -> bool:
    if not gpus:
        return True
    first = gpus[0]
    first_identity = gpu_identity(first)
    first_available = available[0]
    vram_tolerance = max(256 * 1024**2, int(first_available * 0.02))
    return all(
        gpu.vendor == first.vendor
        and gpu_identity(gpu) == first_identity
        and abs(gpu_available - first_available) <= vram_tolerance
        for gpu, gpu_available in zip(gpus, available, strict=True)
    )


def multi_gpu_effective_vram(
    gpus: list[GPUInfo],
    available: list[int],
    warnings: list[str],
) -> tuple[int, bool, int | None]:
    # Multi-GPU fit model. Shrinks raw VRAM into a conservative split budget.
    raw_total = sum(available)
    if len(gpus) <= 1:
        return raw_total, False, None

    if any(gpu.shared_memory or gpu.vendor == "apple" for gpu in gpus):
        effective = max(available)
        warnings.append(
            "Multiple shared-memory GPUs are not pooled; using the largest "
            "reported memory pool for fit checks"
        )
        return effective, False, None

    homogeneous = is_homogeneous_gpu_set(gpus, available)
    utilization = (
        MULTI_GPU_HOMOGENEOUS_UTILIZATION
        if homogeneous
        else MULTI_GPU_HETEROGENEOUS_UTILIZATION
    )
    overhead = min(raw_total, len(gpus) * MULTI_GPU_FRAMEWORK_OVERHEAD_BYTES)
    effective = int((raw_total - overhead) * utilization)

    warnings.append(
        "Multi-GPU fit uses a conservative layer-split budget: "
        f"{effective / BYTES_PER_GIB:.1f} GB effective from {raw_total / BYTES_PER_GIB:.1f} GB raw VRAM"
    )
    if not homogeneous:
        warnings.append(
            "Heterogeneous multi-GPU setup: fit assumes uneven layer placement; "
            "speed depends on backend split mode and interconnect"
        )
    return effective, True, effective


def check_compatibility(
    model: ModelInfo,
    variant: GGUFVariant | None,
    hardware: HardwareInfo,
    context_length: int = 4096,
    vision_workload: VisionWorkload | None = None,
) -> CompatibilityResult:
    # Main fit pass. Produces run type, budgets, and hardware warnings.
    warnings: list[str] = []

    vram_required = estimate_vram(model, variant, context_length, vision_workload)

    usable_ram = effective_usable_ram(hardware.ram_bytes, hardware.ram_budget_bytes)

    best_gpu: GPUInfo | None = None
    best_gpu_available = 0
    gpu_available_values: list[int] = []
    candidate_gpus = fit_candidate_gpus(hardware.gpus)
    ram_budget_active = hardware.ram_budget_bytes is not None
    for gpu in candidate_gpus:
        gpu_available = gpu_available_memory(
            gpu, usable_ram, ram_budget_active=ram_budget_active
        )
        gpu_available_values.append(gpu_available)
        if best_gpu is None or gpu_available > best_gpu_available:
            best_gpu = gpu
            best_gpu_available = gpu_available

    vram_available = sum(gpu_available_values) if gpu_available_values else 0
    fit_vram_available, uses_multi_gpu, multi_gpu_effective_vram_bytes = (
        multi_gpu_effective_vram(candidate_gpus, gpu_available_values, warnings)
    )
    if (
        len(candidate_gpus) > 1
        and not uses_multi_gpu
        and any(gpu.shared_memory or gpu.vendor == "apple" for gpu in candidate_gpus)
    ):
        vram_available = fit_vram_available
    offload_ram_available = (
        0
        if best_gpu and (best_gpu.shared_memory or best_gpu.vendor == "apple")
        else usable_ram
    )

    if best_gpu and best_gpu.vendor == "nvidia" and best_gpu.compute_capability:
        if best_gpu.compute_capability < MIN_COMPUTE_CAPABILITY_OLLAMA:
            warnings.append(
                f"Compute capability {best_gpu.compute_capability} is below "
                f"minimum {MIN_COMPUTE_CAPABILITY_OLLAMA} for Ollama"
            )


    if best_gpu and is_vulkan_only_gpu(best_gpu):
        warnings.append(
            "Legacy Kepler GPU: no CUDA support in modern llama.cpp; "
            "use the Vulkan backend (Linux) instead"
        )

    if (
        best_gpu
        and best_gpu.vendor == "amd"
        and hardware.os not in ("linux", "windows")
    ):
        warnings.append("ROCm requires Linux for AMD GPU inference")

    if best_gpu and best_gpu.vendor == "apple" and hardware.os != "darwin":
        warnings.append("Metal requires macOS for Apple Silicon inference")

    if fit_vram_available >= vram_required:
        fit_type = "full_gpu"
        can_run = True
        offload_ratio = 0.0
    elif (
        fit_vram_available > 0
        and (fit_vram_available + offload_ram_available) >= vram_required
    ):
        fit_type = "partial_offload"
        can_run = True
        offload_ratio = (
            (vram_required - fit_vram_available) / vram_required
            if vram_required > 0
            else 0.0
        )
        offload_pct = offload_ratio * 100
        if best_gpu and (best_gpu.shared_memory or best_gpu.vendor == "apple"):
            warnings.append("Will use shared system memory")
        else:
            warnings.append(
                f"~{offload_pct:.0f}% of layers will be offloaded to CPU RAM"
            )
    elif usable_ram >= vram_required:
        fit_type = "cpu_only"
        can_run = True
        offload_ratio = 0.0
        warnings.append("Will run on CPU only (much slower)")
    else:
        fit_type = "cpu_only"
        can_run = False
        offload_ratio = 0.0
        warnings.append("Insufficient memory (GPU VRAM + RAM) to run this model")


    context_fits = not (
        model.context_length is not None and model.context_length < context_length
    )
    if not context_fits:
        warnings.append(
            f"Model max context {model.context_length} < requested "
            f"{context_length}; runtime will truncate or reject"
        )
    elif (
        context_length > 8192
        and model.context_length
        and model.context_length >= context_length
    ):
        warnings.append(
            f"Large context ({context_length}) increases VRAM usage significantly"
        )


    file_size = estimate_weight_bytes(model, variant)
    if hardware.disk_free_bytes > 0 and file_size > hardware.disk_free_bytes:
        warnings.append("Insufficient disk space to download this model")
        can_run = False

    return CompatibilityResult(
        model=model,
        gguf_variant=variant,
        can_run=can_run,
        vram_required_bytes=vram_required,
        vram_available_bytes=vram_available,
        offload_ratio=offload_ratio,
        uses_multi_gpu=uses_multi_gpu,
        multi_gpu_effective_vram_bytes=multi_gpu_effective_vram_bytes,
        warnings=warnings,
        fit_type=fit_type,
        context_fits=context_fits,
    )
