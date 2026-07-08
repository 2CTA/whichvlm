from __future__ import annotations

from dataclasses import dataclass

from data.gpu import BYTES_PER_GIB, GPU_BANDWIDTH, NVIDIA_COMPUTE_CAPABILITY
from hardware.types import BackendCapability, GPUInfo, HardwareInfo


PLAN_VRAM_HEADROOM_RATIO = 0.10
PLAN_SYSTEM_RAM_BYTES = 64 * BYTES_PER_GIB


@dataclass(frozen=True)
class HardwareCatalogEntry:
    name: str
    vendor: str
    vram_gb: int
    memory_bandwidth_gbps: float | None
    compute_capability: tuple[int, int] | None
    supported_backends: tuple[str, ...]
    os_names: tuple[str, ...]
    shared_memory: bool = False
    shared_memory_behavior: str = "dedicated VRAM"
    multi_gpu_backends: tuple[str, ...] = ()
    interconnect: str | None = None
    price_usd: int | None = None
    availability: str | None = None

    def to_hardware(
        self,
        system_ram_bytes: int = PLAN_SYSTEM_RAM_BYTES,
        os_name: str | None = None,
    ) -> HardwareInfo:
        os_name = os_name or self.os_names[0]
        vram_bytes = self.vram_gb * BYTES_PER_GIB
        gpu = GPUInfo(
            name=self.name,
            vendor=self.vendor,
            vram_bytes=vram_bytes,
            usable_vram_bytes=int(vram_bytes * (1.0 - PLAN_VRAM_HEADROOM_RATIO)),
            compute_capability=self.compute_capability,
            memory_bandwidth_gbps=self.memory_bandwidth_gbps,
            shared_memory=self.shared_memory,
            backend_capabilities=[
                BackendCapability(
                    backend, backend_supported_on_os(self, backend, os_name)
                )
                for backend in self.supported_backends
            ],
        )
        return HardwareInfo(
            gpus=[gpu],
            ram_bytes=system_ram_bytes,
            disk_free_bytes=1_000 * BYTES_PER_GIB,
            os=os_name,
        )


def catalog_entry(
    name: str,
    vendor: str,
    vram_gb: int,
    supported_backends: tuple[str, ...],
    os_names: tuple[str, ...],
    shared_memory: bool = False,
    shared_memory_behavior: str = "dedicated VRAM",
    multi_gpu_backends: tuple[str, ...] = (),
    interconnect: str | None = None,
    price_usd: int | None = None,
    availability: str | None = None,
) -> HardwareCatalogEntry:
    return HardwareCatalogEntry(
        name=name,
        vendor=vendor,
        vram_gb=vram_gb,
        memory_bandwidth_gbps=catalog_bandwidth(name),
        compute_capability=(
            nvidia_compute_capability(name) if vendor == "nvidia" else None
        ),
        supported_backends=supported_backends,
        os_names=os_names,
        shared_memory=shared_memory,
        shared_memory_behavior=shared_memory_behavior,
        multi_gpu_backends=multi_gpu_backends,
        interconnect=interconnect,
        price_usd=price_usd,
        availability=availability,
    )


def catalog_bandwidth(name: str) -> float | None:
    if name in GPU_BANDWIDTH:
        return GPU_BANDWIDTH[name]
    return next(
        (
            bandwidth
            for key, bandwidth in GPU_BANDWIDTH.items()
            if key in name or name.endswith(key)
        ),
        None,
    )


def nvidia_compute_capability(name: str) -> tuple[int, int] | None:
    if name in NVIDIA_COMPUTE_CAPABILITY:
        return NVIDIA_COMPUTE_CAPABILITY[name]
    return next(
        (
            capability
            for key, capability in NVIDIA_COMPUTE_CAPABILITY.items()
            if key in name
        ),
        None,
    )


def backend_supported_on_os(
    entry: HardwareCatalogEntry, backend: str, os_name: str
) -> bool:
    if os_name not in entry.os_names:
        return False
    return backend != "rocm" or os_name == "linux"


def catalog_key(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def lookup_catalog_entry(name: str) -> HardwareCatalogEntry | None:
    target = catalog_key(name.removesuffix("(simulated)").strip())
    for entry in HARDWARE_CATALOG:
        entry_key = catalog_key(entry.name)
        if target == entry_key or entry_key.endswith(target):
            return entry
        if target in {"a100", "h100", "h200"} and entry_key.startswith(target):
            return entry
        if entry_key in {"h100", "h200"} and target.startswith(entry_key):
            return entry
    return None


HARDWARE_CATALOG: tuple[HardwareCatalogEntry, ...] = (
    catalog_entry("RTX 3050", "nvidia", 8, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("RTX 4060", "nvidia", 8, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("RTX 3060", "nvidia", 12, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("RTX 4070", "nvidia", 12, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("RTX A4000", "nvidia", 16, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry(
        "RTX 4060 Ti",
        "nvidia",
        16,
        ("cuda", "vulkan"),
        ("linux", "windows"),
    ),
    catalog_entry(
        "RTX 5060 Ti",
        "nvidia",
        16,
        ("cuda", "vulkan"),
        ("linux", "windows"),
    ),
    catalog_entry("RTX 4080", "nvidia", 16, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("RX 7800 XT", "amd", 16, ("rocm", "vulkan"), ("linux", "windows")),
    catalog_entry(
        "RTX 4070 Ti SUPER",
        "nvidia",
        16,
        ("cuda", "vulkan"),
        ("linux", "windows"),
    ),
    catalog_entry(
        "RTX 5070 Ti",
        "nvidia",
        16,
        ("cuda", "vulkan"),
        ("linux", "windows"),
    ),
    catalog_entry(
        "L4",
        "nvidia",
        24,
        ("cuda",),
        ("linux",),
        price_usd=2500,
        availability="cloud and used market",
    ),
    catalog_entry("RTX 4090", "nvidia", 24, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("RTX 3090", "nvidia", 24, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("RTX 5080", "nvidia", 16, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("A5000", "nvidia", 24, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry("RX 7900 XT", "amd", 20, ("rocm", "vulkan"), ("linux", "windows")),
    catalog_entry("RX 7900 XTX", "amd", 24, ("rocm", "vulkan"), ("linux", "windows")),
    catalog_entry("RX 9070 XT", "amd", 16, ("rocm", "vulkan"), ("linux", "windows")),
    catalog_entry("RTX 5090", "nvidia", 32, ("cuda", "vulkan"), ("linux", "windows")),
    catalog_entry(
        "RTX A6000",
        "nvidia",
        48,
        ("cuda", "vulkan"),
        ("linux", "windows"),
        multi_gpu_backends=("cuda",),
        interconnect="NVLink on supported boards",
        price_usd=4500,
        availability="used market",
    ),
    catalog_entry(
        "A100 40GB",
        "nvidia",
        40,
        ("cuda",),
        ("linux",),
        multi_gpu_backends=("cuda",),
        interconnect="NVLink/SXM or PCIe",
        price_usd=7000,
        availability="cloud and used market",
    ),
    catalog_entry(
        "L40S",
        "nvidia",
        48,
        ("cuda",),
        ("linux",),
        multi_gpu_backends=("cuda",),
        interconnect="PCIe",
        price_usd=8000,
        availability="cloud and workstation",
    ),
    catalog_entry(
        "A100 80GB",
        "nvidia",
        80,
        ("cuda",),
        ("linux",),
        multi_gpu_backends=("cuda",),
        interconnect="NVLink/SXM or PCIe",
        price_usd=12000,
        availability="cloud and used market",
    ),
    catalog_entry(
        "H100",
        "nvidia",
        80,
        ("cuda",),
        ("linux",),
        multi_gpu_backends=("cuda",),
        interconnect="NVLink/SXM or PCIe",
        price_usd=25000,
        availability="cloud and datacenter",
    ),
    catalog_entry(
        "MI210",
        "amd",
        64,
        ("rocm",),
        ("linux",),
        multi_gpu_backends=("rocm",),
        interconnect="Infinity Fabric on supported systems",
        price_usd=6000,
        availability="datacenter and used market",
    ),
    catalog_entry(
        "H200",
        "nvidia",
        141,
        ("cuda",),
        ("linux",),
        multi_gpu_backends=("cuda",),
        interconnect="NVLink/SXM or PCIe",
        availability="cloud and datacenter",
    ),
    catalog_entry(
        "MI300X",
        "amd",
        192,
        ("rocm",),
        ("linux",),
        multi_gpu_backends=("rocm",),
        interconnect="Infinity Fabric",
        availability="cloud and datacenter",
    ),
    catalog_entry(
        "Apple M4 Max",
        "apple",
        36,
        ("metal", "mps", "mlx"),
        ("darwin",),
        shared_memory=True,
        shared_memory_behavior="unified system memory",
        price_usd=3200,
        availability="new systems",
    ),
)
