"""Tests for hardware.memory — estimate_usable_ram bounded-reserve formula."""

import pytest

from whichvlm.hardware.memory import estimate_usable_ram

_GiB = 1024**3


def _expected_usable(total: int) -> int:
    reserve = int(total * 0.15)
    reserve = max(4 * _GiB, min(reserve, 32 * _GiB))
    return total - reserve


@pytest.mark.parametrize(
    "total_gb",
    [16, 32, 64, 128, 1024],
    ids=["16GB", "32GB", "64GB", "128GB", "1TB"],
)
def test_estimate_usable_ram(total_gb):
    total = total_gb * _GiB
    assert estimate_usable_ram(total) == _expected_usable(total)


def test_16gb_hits_min_reserve():
    total = 16 * _GiB
    assert estimate_usable_ram(total) == total - 4 * _GiB


def test_1tb_hits_max_reserve():
    total = 1024 * _GiB
    assert estimate_usable_ram(total) == total - 32 * _GiB


def test_midrange_uses_percentage():
    total = 64 * _GiB
    expected_reserve = int(total * 0.15)
    assert 4 * _GiB < expected_reserve < 32 * _GiB
    assert estimate_usable_ram(total) == total - expected_reserve
