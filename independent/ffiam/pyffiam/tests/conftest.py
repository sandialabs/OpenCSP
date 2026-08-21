# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Pytest fixtures for FFIAM testing.

This module provides reusable fixtures for testing pyffiam components,
including sample configurations, mock data, and synthetic results.
"""

import sys
from ctypes.util import find_library

import numpy as np
import pytest


def _has_cuda_runtime() -> bool:
    """Probe whether the FFIAM CUDA backend is actually usable on this machine.

    Finding *a* cudart is not enough: the shipped libffiam_lib.so is linked
    against whichever CUDA the build host had, so a box with a different major
    version (e.g. the sandbox's 12.8 vs a host build's cudart 13) resolves
    find_library() fine and then fails to dlopen. Load it for real instead.
    """
    import ctypes
    from pathlib import Path

    name = "cudart64_12" if sys.platform == "win32" else "cudart"
    path = find_library(name)
    if path is None or "cudart" not in path:
        return False

    gpu_lib = "ffiam_lib.dll" if sys.platform == "win32" else "libffiam_lib.so"
    gpu_path = Path(__file__).parent.parent / "src" / "pyffiam" / "external" / gpu_lib
    if not gpu_path.exists():
        return False
    try:
        ctypes.CDLL(str(gpu_path))
    except OSError:
        return False
    return True


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: skip when no CUDA runtime is present")


def pytest_collection_modifyitems(config, items):
    if _has_cuda_runtime():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA runtime not found")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_cuda)


from pyffiam.ffiam_types import CspSite, AimType
from pyffiam.config import (
    AnalysisConfig,
    TimeConfig,
    LocationConfig,
    FieldConfig,
    HeliostatConfig,
    AimConfig,
    VoxelConfig,
    OpticalConfig,
)
from pyffiam.results import AnalysisResults, HeliostatResults, IrradianceResults, ThresholdResults
from pyffiam.cpp_interface import RawResults


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_time_config() -> TimeConfig:
    """Sample time configuration for summer solstice at noon."""
    return TimeConfig(year=2025, month=6, day=21, hour=12)


@pytest.fixture
def sample_location_config() -> LocationConfig:
    """Sample location configuration for NSTTF site."""
    return LocationConfig(latitude=34.96, longitude=-106.51, timezone=-7.0)


@pytest.fixture
def sample_field_config() -> FieldConfig:
    """Sample field configuration with typical parameters."""
    return FieldConfig(radius=600, num_heliostats=218, min_height=4, max_height=200, tower_height=90.0)


@pytest.fixture
def sample_heliostat_config() -> HeliostatConfig:
    """Sample heliostat configuration."""
    return HeliostatConfig(
        num_facets=25, num_facet_cols=5, facet_width=1.2, facet_height=1.2, heliostat_file="", facet_file=""
    )


@pytest.fixture
def sample_aim_config() -> AimConfig:
    """Sample aim configuration with point aiming."""
    return AimConfig(strategy=AimType.Point, parameters=np.array([0.0, 0.0, 90.0]), aim_file="")


@pytest.fixture
def sample_voxel_config() -> VoxelConfig:
    """Sample voxel configuration."""
    return VoxelConfig(size=2)


@pytest.fixture
def sample_optical_config() -> OpticalConfig:
    """Sample optical configuration."""
    return OpticalConfig(reflectivity=0.9, peak_dni=0.1, sun_angle=0.0093, slope_error=0.0012, beta=0.0094)


@pytest.fixture
def sample_analysis_config(
    sample_time_config,
    sample_location_config,
    sample_field_config,
    sample_heliostat_config,
    sample_aim_config,
    sample_voxel_config,
    sample_optical_config,
) -> AnalysisConfig:
    """Complete sample analysis configuration."""
    return AnalysisConfig(
        name="Test Analysis",
        site=CspSite.NSTTF,
        time=sample_time_config,
        location=sample_location_config,
        field=sample_field_config,
        heliostat=sample_heliostat_config,
        aim=sample_aim_config,
        voxel=sample_voxel_config,
        optical=sample_optical_config,
        threshold=4.0,
    )


# =============================================================================
# Mock Data Fixtures (for testing without GPU)
# =============================================================================


@pytest.fixture
def mock_heliostat_locations() -> np.ndarray:
    """Generate mock heliostat locations in a grid pattern."""
    n_heliostats = 100
    # Create a 10x10 grid of heliostats
    x = np.linspace(-200, 200, 10)
    y = np.linspace(-200, 200, 10)
    xx, yy = np.meshgrid(x, y)
    locations = np.column_stack([xx.flatten(), yy.flatten(), np.zeros(n_heliostats)])  # All at ground level
    return locations


@pytest.fixture
def mock_voxel_locations() -> np.ndarray:
    """Generate mock voxel locations for a small test grid."""
    # Create a 10x10x5 voxel grid
    voxel_size = 2
    field_r = 10
    min_height = 4
    max_height = 14

    x = np.arange(-field_r, field_r, voxel_size)
    y = np.arange(-field_r, field_r, voxel_size)
    z = np.arange(min_height, max_height, voxel_size)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    locations = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])
    return locations


@pytest.fixture
def mock_irradiance_values(mock_voxel_locations) -> np.ndarray:
    """Generate mock irradiance values with realistic distribution."""
    n_voxels = len(mock_voxel_locations)
    # Most voxels have low irradiance, few have high
    irrads = np.random.exponential(scale=2.0, size=n_voxels)
    # Add some zeros
    irrads[irrads < 0.1] = 0
    return irrads


@pytest.fixture
def mock_raw_results(mock_heliostat_locations, mock_voxel_locations, mock_irradiance_values) -> RawResults:
    """Create mock RawResults for testing without GPU.

    This fixture provides synthetic results that can be used to test
    the Python analysis pipeline without requiring CUDA.
    """
    n_heliostats = len(mock_heliostat_locations)

    return RawResults(
        helio_locs=mock_heliostat_locations,
        helio_aim_vs=np.tile([0, 0, 1], (n_heliostats, 1)).astype(np.float64),
        helio_angles=np.random.uniform(0, 360, n_heliostats),
        facet_origins=np.zeros((n_heliostats, 25, 3)),  # 25 facets per heliostat
        irrads=mock_irradiance_values,
        movement=45.0,
    )


# =============================================================================
# Results Fixtures
# =============================================================================


@pytest.fixture
def sample_heliostat_results(mock_heliostat_locations) -> HeliostatResults:
    """Sample heliostat results."""
    n_heliostats = len(mock_heliostat_locations)
    return HeliostatResults(
        locations=mock_heliostat_locations,
        aim_vectors=np.tile([0, 0, 1], (n_heliostats, 1)).astype(np.float64),
        movement_angles=np.random.uniform(0, 360, n_heliostats),
        facet_origins=np.zeros((n_heliostats, 25, 3)),
        total_movement=45.0,
    )


@pytest.fixture
def sample_irradiance_results(mock_irradiance_values) -> IrradianceResults:
    """Sample irradiance results."""
    return IrradianceResults.from_array(mock_irradiance_values)


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def small_voxel_params() -> dict:
    """Parameters for a small voxel grid used in utility tests."""
    return {'vox_size': 2, 'field_r': 10, 'field_zmin': 4}


@pytest.fixture
def standard_voxel_params() -> dict:
    """Parameters for a standard voxel grid."""
    return {'vox_size': 2, 'field_r': 600, 'field_zmin': 4}
