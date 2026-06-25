# Copyright Sandia National Laboratories. All rights reserved.

"""Tests for voxel size validation and memory estimation."""

import pytest
import numpy as np

from pyffiam.app_config import (
    MIN_VOXEL_SIZE, MAX_VOXEL_SIZE, MAX_VOXELS, VOXEL_POOL_SIZE,
    MAX_FIELD_RADIUS, MAX_HEIGHT, GB
)
from pyffiam.utils import estimate_voxel_memory, print_voxel_config_table


class TestVoxelMemoryEstimation:
    """Tests for voxel memory estimation utility."""

    def test_estimate_basic(self):
        """Test basic memory estimation."""
        num_voxels, memory_gb, is_valid = estimate_voxel_memory(300, 0, 120, 2)

        # 300m radius: 300 voxels per side
        # 120m height at 2m: 61 levels
        expected_voxels = 300 * 300 * 61
        assert num_voxels == expected_voxels
        assert is_valid is True
        assert memory_gb < 1.0  # Should be well under 1 GB

    def test_estimate_1m_voxels_small_field(self):
        """Test 1m voxels work for small fields."""
        num_voxels, memory_gb, is_valid = estimate_voxel_memory(300, 0, 120, 1)

        # 300m radius at 1m: 600 voxels per side
        # 120m height at 1m: 121 levels
        expected_voxels = 600 * 600 * 121
        assert num_voxels == expected_voxels
        assert is_valid is True
        assert memory_gb < 0.2  # Should be ~0.16 GB

    def test_estimate_1m_voxels_600m_field(self):
        """Test 1m voxels work for 600m radius field."""
        num_voxels, memory_gb, is_valid = estimate_voxel_memory(600, 0, 100, 1)

        # 600m radius at 1m: 1200 voxels per side
        # 100m height at 1m: 101 levels
        expected_voxels = 1200 * 1200 * 101
        assert num_voxels == expected_voxels
        assert is_valid is True
        assert memory_gb < 1.0  # Should be ~0.54 GB

    def test_estimate_rejects_too_large(self):
        """Test that very large configurations are marked invalid."""
        # 1000m radius at 1m voxels with 200m altitude
        num_voxels, memory_gb, is_valid = estimate_voxel_memory(1000, 0, 200, 1)

        # 1000m radius at 1m: 2000 voxels per side
        # 200m height at 1m: 201 levels
        expected_voxels = 2000 * 2000 * 201
        assert num_voxels == expected_voxels
        assert is_valid is False  # 804M > 500M limit
        assert memory_gb > 2.0  # Should be ~3 GB


class TestMaxFieldSizeConstraints:
    """Tests ensuring max field sizes work with 2m voxels."""

    def test_max_field_with_2m_voxels(self):
        """Test maximum field radius with default 2m voxels fits in memory."""
        num_voxels, memory_gb, is_valid = estimate_voxel_memory(
            MAX_FIELD_RADIUS, 0, MAX_HEIGHT, 2
        )

        # 1700m radius at 2m: 1700 voxels per side
        # 310m height at 2m: 156 levels
        expected_voxels = 1700 * 1700 * 156
        assert num_voxels == expected_voxels
        assert num_voxels < MAX_VOXELS, f"Max config exceeds limit: {num_voxels:,} > {MAX_VOXELS:,}"
        assert is_valid is True

        # Memory should be well under pool size
        memory_bytes = num_voxels * 4
        assert memory_bytes < VOXEL_POOL_SIZE

    def test_1600m_300m_2m_constraint(self):
        """Test the specific constraint mentioned: 1600m radius, 300m altitude, 2m voxel."""
        num_voxels, memory_gb, is_valid = estimate_voxel_memory(1600, 0, 300, 2)

        # Should work
        assert is_valid is True
        print(f"1600m/300m/2m: {num_voxels:,} voxels, {memory_gb:.2f} GB")

    def test_max_field_with_1m_voxels_fails(self):
        """Test that maximum field with 1m voxels exceeds limits."""
        num_voxels, memory_gb, is_valid = estimate_voxel_memory(
            MAX_FIELD_RADIUS, 0, MAX_HEIGHT, 1
        )

        # This should exceed memory limits
        # 1700m radius at 1m: 3400 voxels per side
        # 310m height at 1m: 311 levels
        expected_voxels = 3400 * 3400 * 311
        assert num_voxels == expected_voxels
        assert is_valid is False  # Should be ~3.6 billion voxels


class TestVoxelSizeConstraints:
    """Tests for voxel size validation constants."""

    def test_min_voxel_size(self):
        """Test minimum voxel size is 1m."""
        assert MIN_VOXEL_SIZE == 1

    def test_max_voxel_size(self):
        """Test maximum voxel size is 10m."""
        assert MAX_VOXEL_SIZE == 10

    def test_max_voxels_allows_reasonable_configs(self):
        """Test MAX_VOXELS allows typical use cases."""
        # Typical validation config: 300m radius, 0-120m, 2m voxels
        num_voxels, _, is_valid = estimate_voxel_memory(300, 0, 120, 2)
        assert is_valid is True

        # High-resolution config: 600m radius, 0-100m, 1m voxels
        num_voxels, _, is_valid = estimate_voxel_memory(600, 0, 100, 1)
        assert is_valid is True


class TestPrintVoxelConfigTable:
    """Tests for the config table printer (visual verification)."""

    def test_print_table_runs(self, capsys):
        """Test that print_voxel_config_table runs without error."""
        print_voxel_config_table(300, 0, 120)
        captured = capsys.readouterr()
        assert "Voxel Size" in captured.out
        assert "1m" in captured.out
        assert "OK" in captured.out


if __name__ == "__main__":
    # Quick verification
    print("Testing voxel memory estimation...")

    print("\n1m voxels, 600m radius, 0-100m altitude:")
    num, mem, valid = estimate_voxel_memory(600, 0, 100, 1)
    print(f"  {num:,} voxels, {mem:.2f} GB, valid={valid}")

    print("\nMax field (1700m radius, 0-310m altitude, 2m voxels):")
    num, mem, valid = estimate_voxel_memory(1700, 0, 310, 2)
    print(f"  {num:,} voxels, {mem:.2f} GB, valid={valid}")

    print("\n1600m radius, 0-300m altitude, 2m voxels:")
    num, mem, valid = estimate_voxel_memory(1600, 0, 300, 2)
    print(f"  {num:,} voxels, {mem:.2f} GB, valid={valid}")

    print("\nConfiguration table for 600m radius, 0-100m:")
    print_voxel_config_table(600, 0, 100)
