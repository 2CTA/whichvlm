from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackendCapability:
    name: str
    available: bool = True
    version: str | None = None
    details: str | None = None


@dataclass
class GPUInfo:
    name: str
    vendor: str
    vram_bytes: int
    usable_vram_bytes: int | None = None
    compute_capability: tuple[int, int] | None = None
    cuda_version: str | None = None
    rocm_version: str | None = None
    memory_bandwidth_gbps: float | None = None
    shared_memory: bool = False
    backend_capabilities: list[BackendCapability] = field(default_factory=list)
    neural_engine_available: bool = False


@dataclass
class HardwareInfo:
    gpus: list[GPUInfo] = field(default_factory=list)
    cpu_name: str = "Unknown"
    cpu_cores: int = 0
    has_avx2: bool = False
    has_avx512: bool = False
    ram_bytes: int = 0
    ram_budget_bytes: int | None = None
    disk_free_bytes: int = 0
    os: str = "linux"
    budget_notes: list[str] = field(default_factory=list)
    backend_capabilities: list[BackendCapability] = field(default_factory=list)


def infer_backend_capabilities(gpu: GPUInfo, os_name: str) -> list[BackendCapability]:
    if gpu.vendor == "apple":
        if os_name == "darwin":
            return [
                BackendCapability("metal", True, details="Apple Silicon Metal GPU"),
                BackendCapability("mps", True, details="PyTorch MPS-compatible"),
                BackendCapability("mlx", True, details="MLX-ready Apple Silicon"),
            ]
        return [
            BackendCapability("vulkan", True, details="Apple GPU via Linux driver"),
            BackendCapability("metal", False, details="Metal requires macOS"),
            BackendCapability("mlx", False, details="MLX requires macOS"),
        ]
    if gpu.vendor == "nvidia":
        return [
            BackendCapability("cuda", True, gpu.cuda_version),
            BackendCapability("vulkan", True),
        ]
    if gpu.vendor == "amd":
        return [
            BackendCapability("rocm", os_name == "linux", gpu.rocm_version),
            BackendCapability("vulkan", True),
        ]
    if gpu.vendor == "intel":
        return [BackendCapability("vulkan", True)]
    return []


def ensure_backend_capabilities(gpu: GPUInfo, os_name: str) -> GPUInfo:
    if not gpu.backend_capabilities:
        gpu.backend_capabilities = infer_backend_capabilities(gpu, os_name)
    if gpu.vendor == "apple" and os_name == "darwin":
        gpu.neural_engine_available = True
    return gpu


def has_backend(gpu: GPUInfo, backend_name: str) -> bool:
    target = backend_name.lower()
    return any(
        c.name.lower() == target and c.available for c in gpu.backend_capabilities
    )
