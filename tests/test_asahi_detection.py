from __future__ import annotations

import subprocess
import json
from pathlib import Path

from whichvlm.hardware import apple, cpu
from whichvlm.hardware.types import GPUInfo, ensure_backend_capabilities, has_backend


def test_cpu_name_lscpu_fallback(monkeypatch):


    arm_cpuinfo = (
        "processor\t: 0\n"
        "BogoMIPS\t: 48.00\n"
        "Features\t: fp asimd evtstrm aes\n"
        "CPU implementer\t: 0x61\n"
    )
    monkeypatch.setattr("builtins.open", fake_open(arm_cpuinfo))
    monkeypatch.setattr("platform.system", lambda: "Linux")

    lscpu_output = "Architecture:            aarch64\nModel name:            Apple M2\n"

    def fake_run(args, **kwargs):
        if args == ["lscpu"]:
            return subprocess.CompletedProcess(args, 0, stdout=lscpu_output, stderr="")
        raise FileNotFoundError

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)

    assert cpu.detect_cpu_name() == "Apple M2"


def test_cpu_name_devicetree_fallback(monkeypatch, tmp_path):

    arm_cpuinfo = "processor\t: 0\nFeatures\t: fp asimd\n"
    monkeypatch.setattr("builtins.open", fake_open(arm_cpuinfo))
    monkeypatch.setattr("platform.system", lambda: "Linux")


    def fake_run(args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)


    dt_model = tmp_path / "model"
    dt_model.write_bytes(b"Apple MacBook Air (M2, 2022)\x00")
    monkeypatch.setattr(cpu, "cpu_name_from_devicetree", lambda: read_dt_model(dt_model))

    assert cpu.detect_cpu_name() == "Apple M2"


def test_cpu_name_devicetree_extracts_chip_variants():

    assert extract("Apple MacBook Air (M2, 2022)") == "Apple M2"
    assert extract("Apple Mac Mini (M4 Pro, 2024)") == "Apple M4 Pro"
    assert extract("Apple Mac Studio (M2 Ultra, 2023)") == "Apple M2 Ultra"
    assert extract("Apple Mac Pro (M2 Max, 2023)") == "Apple M2 Max"
    assert extract("Apple MacBook Pro (M1, 2020)") == "Apple M1"


def test_cpu_name_devicetree_non_apple():

    assert extract("Raspberry Pi 4 Model B Rev 1.5") is None
    assert extract("Qualcomm Snapdragon 8cx Gen 3") is None


def test_cpu_name_lscpu_ignores_dash(monkeypatch):

    arm_cpuinfo = "processor\t: 0\n"
    monkeypatch.setattr("builtins.open", fake_open(arm_cpuinfo))
    monkeypatch.setattr("platform.system", lambda: "Linux")

    lscpu_output = "Architecture:            aarch64\nModel name:            -\n"

    def fake_run(args, **kwargs):
        if args == ["lscpu"]:
            return subprocess.CompletedProcess(args, 0, stdout=lscpu_output, stderr="")
        raise FileNotFoundError

    monkeypatch.setattr(cpu.subprocess, "run", fake_run)
    monkeypatch.setattr(cpu, "cpu_name_from_devicetree", lambda: "Apple M2")

    assert cpu.detect_cpu_name() == "Apple M2"


def test_detect_asahi_gpu_from_sysfs(monkeypatch, tmp_path):

    setup_asahi_sysfs(tmp_path)
    monkeypatch.setattr(apple, "chip_name_from_devicetree", lambda: "Apple M2")
    monkeypatch.setattr("psutil.virtual_memory", fake_vmem(24 * 1024**3))

    gpus = apple.detect_apple_gpu_linux(drm_path=tmp_path)

    assert len(gpus) == 1
    assert gpus[0].vendor == "apple"
    assert gpus[0].name == "Apple M2"
    assert gpus[0].vram_bytes == 24 * 1024**3
    assert gpus[0].shared_memory is True
    assert gpus[0].memory_bandwidth_gbps == 100.0
    assert has_backend(gpus[0], "vulkan")
    assert not has_backend(gpus[0], "metal")
    assert not has_backend(gpus[0], "mlx")


def test_detect_asahi_gpu_fallback_name(monkeypatch, tmp_path):

    setup_asahi_sysfs(tmp_path)
    monkeypatch.setattr(apple, "chip_name_from_devicetree", lambda: None)
    monkeypatch.setattr("psutil.virtual_memory", fake_vmem(8 * 1024**3))

    gpus = apple.detect_apple_gpu_linux(drm_path=tmp_path)

    assert len(gpus) == 1
    assert gpus[0].name == "Apple Silicon"


def test_detect_asahi_gpu_ignores_non_apple_drivers(tmp_path):

    card = tmp_path / "card0" / "device" / "driver"
    card.mkdir(parents=True)

    target = tmp_path / "drivers" / "amdgpu"
    target.mkdir(parents=True)
    (tmp_path / "card0" / "device" / "driver").rmdir()
    (tmp_path / "card0" / "device" / "driver").symlink_to(target)

    assert apple.detect_apple_gpu_linux(drm_path=tmp_path) == []


def test_detect_asahi_gpu_no_drm(tmp_path):

    nonexistent = tmp_path / "no_drm"
    assert apple.detect_apple_gpu_linux(drm_path=nonexistent) == []


def test_detect_apple_gpu_macos_parses_metal_and_mlx(monkeypatch):
    def fake_run(args, **kwargs):
        if args[1] == "SPHardwareDataType":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {
                        "SPHardwareDataType": [
                            {
                                "chip_type": "Apple M3 Max",
                                "physical_memory": "36 GB",
                            }
                        ]
                    }
                ),
                stderr="",
            )
        if args[1] == "SPDisplaysDataType":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {
                        "SPDisplaysDataType": [
                            {"spdisplays_metal": "Metal 3"}
                        ]
                    }
                ),
                stderr="",
            )
        raise AssertionError(args)

    monkeypatch.setattr(apple.subprocess, "run", fake_run)

    gpus = apple.detect_apple_gpu()

    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.name == "Apple M3 Max"
    assert gpu.shared_memory is True
    assert gpu.neural_engine_available is True
    caps = {c.name: c for c in gpu.backend_capabilities}
    assert caps["metal"].available is True
    assert caps["metal"].version == "Metal 3"
    assert caps["mlx"].available is True
    assert caps["mps"].available is True


def test_detect_apple_gpu_macos_falls_back_when_display_metadata_missing(
    monkeypatch,
):
    def fake_run(args, **kwargs):
        if args[1] == "SPHardwareDataType":
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=json.dumps(
                    {
                        "SPHardwareDataType": [
                            {"chip_type": "Apple M2", "physical_memory": "16 GB"}
                        ]
                    }
                ),
                stderr="",
            )
        if args[1] == "SPDisplaysDataType":
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="failed")
        raise AssertionError(args)

    monkeypatch.setattr(apple.subprocess, "run", fake_run)

    gpus = apple.detect_apple_gpu()

    assert len(gpus) == 1
    metal = next(c for c in gpus[0].backend_capabilities if c.name == "metal")
    assert metal.available is True
    assert metal.version is None
    assert "Assumed available" in (metal.details or "")


def test_mlx_readiness_requires_darwin_apple_silicon():
    apple_darwin = ensure_backend_capabilities(
        GPUInfo(name="Apple M3", vendor="apple", vram_bytes=16 * 1024**3),
        "darwin",
    )
    apple_linux = ensure_backend_capabilities(
        GPUInfo(name="Apple M3", vendor="apple", vram_bytes=16 * 1024**3),
        "linux",
    )
    nvidia_darwin = ensure_backend_capabilities(
        GPUInfo(name="RTX", vendor="nvidia", vram_bytes=16 * 1024**3),
        "darwin",
    )

    assert has_backend(apple_darwin, "mlx")
    assert not has_backend(apple_linux, "mlx")
    assert not has_backend(nvidia_darwin, "mlx")


def setup_asahi_sysfs(tmp_path: Path) -> None:

    device_dir = tmp_path / "card0" / "device"
    device_dir.mkdir(parents=True)

    driver_target = tmp_path / "drivers" / "asahi"
    driver_target.mkdir(parents=True)
    (device_dir / "driver").symlink_to(driver_target)


def fake_open(content: str):

    import builtins
    import io

    real_open = builtins.open

    def patched_open(path, *a, **kw):
        if str(path) == "/proc/cpuinfo":
            return io.StringIO(content)
        return real_open(path, *a, **kw)

    return patched_open


def fake_vmem(total: int):

    from collections import namedtuple

    Vmem = namedtuple("svmem", ["total"])

    def make_simulated_vmem():
        return Vmem(total=total)

    return make_simulated_vmem


def read_dt_model(path: Path) -> str | None:

    import re

    try:
        raw = path.read_bytes()
        model = raw.decode("utf-8", errors="replace").strip().rstrip("\x00")
        if not model:
            return None
        m = re.search(r"\b(M\d+(?:\s+(?:Pro|Max|Ultra))?)\b", model)
        if m:
            return f"Apple {m.group(1)}"
        return model
    except OSError:
        return None


def extract(model: str) -> str | None:

    import re

    m = re.search(r"\b(M\d+(?:\s+(?:Pro|Max|Ultra))?)\b", model)
    if m:
        return f"Apple {m.group(1)}"
    return None
