import importlib
import subprocess

import pytest

from data.gpu import GPU_BANDWIDTH, GPU_MEMORY_CLOCK_VARIANTS
from hardware import nvidia
from hardware.gpu_db import resolve_detected_bandwidth
from hardware.types import GPUInfo
from engine.performance import estimate_tok_per_sec
from models.types import GGUFVariant, ModelInfo

GTX1650_NAME = "NVIDIA GeForce GTX 1650"
GDDR6_CLOCK = 6001.0
GDDR5_CLOCK = 4001.0


def test_variant_table_preserves_gtx1650_defaults():
    legacy_constants = importlib.import_module("whichvlm.constants")
    thresholds = [t for t, _ in GPU_MEMORY_CLOCK_VARIANTS["GTX 1650"]]

    assert thresholds == sorted(thresholds, reverse=True)
    assert GPU_BANDWIDTH["GTX 1650"] == 128.0
    assert legacy_constants.GPU_BANDWIDTH is GPU_BANDWIDTH
    assert legacy_constants.GPU_MEMORY_CLOCK_VARIANTS is GPU_MEMORY_CLOCK_VARIANTS


def test_gddr6_clock_resolves_to_192():
    assert resolve_detected_bandwidth(GTX1650_NAME, 4 * 1024**3, GDDR6_CLOCK) == 192.0


def test_gddr5_clock_resolves_to_128():
    assert resolve_detected_bandwidth(GTX1650_NAME, 4 * 1024**3, GDDR5_CLOCK) == 128.0


def test_unknown_clock_falls_back_to_curated_default():
    assert resolve_detected_bandwidth(GTX1650_NAME, 4 * 1024**3) == 128.0
    assert resolve_detected_bandwidth(GTX1650_NAME, 4 * 1024**3, None) == 128.0
    assert resolve_detected_bandwidth(GTX1650_NAME, 4 * 1024**3, 0.0) == 128.0


@pytest.mark.parametrize(
    "clock, expected", [(5499.0, 128.0), (5500.0, 192.0), (12000.0, 192.0)]
)
def test_threshold_boundary(clock, expected):
    assert resolve_detected_bandwidth(GTX1650_NAME, 4 * 1024**3, clock) == expected


def test_non_variant_card_ignores_memory_clock():
    assert (
        resolve_detected_bandwidth("NVIDIA GeForce GTX 1660", 6 * 1024**3, 9999.0)
        == 192.0
    )
    assert (
        resolve_detected_bandwidth("NVIDIA GeForce GTX 1660", 6 * 1024**3, 100.0)
        == 192.0
    )


def test_gtx1650_super_does_not_fall_through_to_base_1650():
    assert (
        resolve_detected_bandwidth("NVIDIA GeForce GTX 1650 SUPER", 4 * 1024**3)
        == 192.0
    )
    assert (
        resolve_detected_bandwidth("NVIDIA GeForce GTX 1650 SUPER", 4 * 1024**3, 0.0)
        == 192.0
    )


def test_gddr6_estimate_scales_with_bandwidth_and_matches_measured():
    model = ModelInfo(
        id="Qwen/Qwen3-1.7B",
        family_id="Qwen/Qwen3-1.7B",
        name="Qwen3-1.7B",
        parameter_count=1_720_000_000,
    )
    variant = GGUFVariant(
        filename="Qwen3-1.7B-Q4_K_M.gguf",
        quant_type="Q4_K_M",
        file_size_bytes=1_353_000_000,
    )
    gpu6 = GPUInfo(
        "NVIDIA GeForce GTX 1650",
        "nvidia",
        4 * 1024**3,
        compute_capability=(7, 5),
        memory_bandwidth_gbps=192.0,
    )
    gpu5 = GPUInfo(
        "NVIDIA GeForce GTX 1650",
        "nvidia",
        4 * 1024**3,
        compute_capability=(7, 5),
        memory_bandwidth_gbps=128.0,
    )
    est6 = estimate_tok_per_sec(model, variant, gpu6)
    est5 = estimate_tok_per_sec(model, variant, gpu5)
    assert est6 > est5
    assert est6 / est5 == pytest.approx(192.0 / 128.0, rel=1e-3)

    assert 60.0 <= est6 <= 95.0


def smi_bw(monkeypatch, stdout: str) -> float | None:
    monkeypatch.setattr(nvidia, "run_smi_query", lambda fields: stdout)
    gpus = nvidia.detect_nvidia_gpus_via_smi()
    assert len(gpus) == 1
    return gpus[0].memory_bandwidth_gbps


def test_smi_gddr6_clock_resolves_192(monkeypatch):
    assert smi_bw(monkeypatch, "NVIDIA GeForce GTX 1650, 4096, 6001\n") == 192.0


def test_smi_gddr5_clock_resolves_128(monkeypatch):
    assert smi_bw(monkeypatch, "NVIDIA GeForce GTX 1650, 4096, 4001\n") == 128.0


def test_smi_na_clock_falls_back_to_curated(monkeypatch):
    assert smi_bw(monkeypatch, "NVIDIA GeForce GTX 1650, 4096, [N/A]\n") == 128.0


def test_smi_3field_query_failure_retries_without_clock(monkeypatch):
    def fake_query(fields: str) -> str:
        if "clocks" in fields:
            raise subprocess.CalledProcessError(6, "nvidia-smi")
        return "NVIDIA GeForce GTX 1650, 4096\n"

    monkeypatch.setattr(nvidia, "run_smi_query", fake_query)
    gpus = nvidia.detect_nvidia_gpus_via_smi()
    assert len(gpus) == 1
    assert gpus[0].memory_bandwidth_gbps == 128.0


def test_smi_both_queries_fail_returns_empty(monkeypatch):
    def always_fail(fields: str) -> str:
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(nvidia, "run_smi_query", always_fail)
    assert nvidia.detect_nvidia_gpus_via_smi() == []
