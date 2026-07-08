import pytest

from hardware.memory import estimate_usable_ram

BYTES_PER_GIB = 1024**3


def expected_usable(total: int) -> int:
    reserve = int(total * 0.15)
    reserve = max(4 * BYTES_PER_GIB, min(reserve, 32 * BYTES_PER_GIB))
    return total - reserve


@pytest.mark.parametrize(
    "total_gb",
    [16, 32, 64, 128, 1024],
    ids=["16GB", "32GB", "64GB", "128GB", "1TB"],
)
def test_estimate_usable_ram(total_gb):
    total = total_gb * BYTES_PER_GIB
    assert estimate_usable_ram(total) == expected_usable(total)


def test_16gb_hits_min_reserve():
    total = 16 * BYTES_PER_GIB
    assert estimate_usable_ram(total) == total - 4 * BYTES_PER_GIB


def test_1tb_hits_max_reserve():
    total = 1024 * BYTES_PER_GIB
    assert estimate_usable_ram(total) == total - 32 * BYTES_PER_GIB


def test_midrange_uses_percentage():
    total = 64 * BYTES_PER_GIB
    expected_reserve = int(total * 0.15)
    assert 4 * BYTES_PER_GIB < expected_reserve < 32 * BYTES_PER_GIB
    assert estimate_usable_ram(total) == total - expected_reserve
