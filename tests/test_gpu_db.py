from whichvlm.hardware.gpu_db import (
    normalize_detected_gpu_name,
    static_bandwidth,
    resolve_detected_bandwidth,
)

BYTES_PER_GIB = 1024**3


def test_normalize_strips_vendor_trademark_and_maps_laptop_to_mobile():
    assert (
        normalize_detected_gpu_name("NVIDIA GeForce RTX 5090 Laptop GPU")
        == "GeForce RTX 5090 Mobile"
    )
    assert normalize_detected_gpu_name("Intel(R) Arc(TM) A770 Graphics") == "Arc A770"
    assert normalize_detected_gpu_name("AMD Radeon RX 6750 XT") == "Radeon RX 6750 XT"


def test_normalize_empty_or_vendor_only():
    assert normalize_detected_gpu_name("") == ""
    assert normalize_detected_gpu_name("AMD") == ""


def test_normalize_adds_space_to_vram_bin_suffix():

    assert normalize_detected_gpu_name("NVIDIA RTX A2000 12GB") == "RTX A2000 12 GB"


def test_static_bandwidth_desktop_key_does_not_claim_laptop_card():

    assert static_bandwidth("NVIDIA GeForce RTX 5090 Laptop GPU") is None

    assert static_bandwidth("NVIDIA GeForce RTX 5090") == 1792.0


def test_static_bandwidth_keeps_curated_laptop_entry():

    assert static_bandwidth("NVIDIA RTX A3000 Laptop GPU") == 264.0


def test_resolve_desktop_uses_curated_value():
    assert resolve_detected_bandwidth("NVIDIA GeForce RTX 5090") == 1792.0


def test_resolve_laptop_5090_is_mobile_not_desktop():

    bw = resolve_detected_bandwidth("NVIDIA GeForce RTX 5090 Laptop GPU", 24 * BYTES_PER_GIB)
    assert bw is not None
    assert bw < 1500.0


def test_resolve_a3000_laptop_preserves_curated_264():
    assert resolve_detected_bandwidth("NVIDIA RTX A3000 Laptop GPU", 6 * BYTES_PER_GIB) == 264.0


def test_resolve_rx_6750_xt():


    bw = resolve_detected_bandwidth("AMD Radeon RX 6750 XT", 12 * BYTES_PER_GIB)
    assert bw == 432.0


def test_resolve_variant_qualifier_is_preserved():

    bw = resolve_detected_bandwidth("NVIDIA GeForce RTX 4060 Ti", 16 * BYTES_PER_GIB)
    assert bw is not None


    assert 200 < bw < 400


def test_resolve_unknown_gpu_returns_none_not_wrong_guess():

    assert resolve_detected_bandwidth("Intel(R) Arc(TM) Pro B70 Graphics") is None


def test_resolve_empty_name_returns_none():
    assert resolve_detected_bandwidth("") is None
    assert resolve_detected_bandwidth("Some Totally Made Up GPU 9xZ") is None


def test_resolve_known_amd_desktop_from_curated_table():
    assert resolve_detected_bandwidth("AMD Radeon RX 9070 XT") == 640.0


def test_resolve_a2000_12gb_vram_bin_from_dbgpu():


    assert resolve_detected_bandwidth("NVIDIA RTX A2000 12GB", 12 * BYTES_PER_GIB) == 288.0


def test_static_bandwidth_compound_lspci_name():


    compound = "Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT]"
    bw = static_bandwidth(compound)
    assert bw is not None
    assert bw > 0

    assert resolve_detected_bandwidth(compound, 12 * BYTES_PER_GIB) == bw
