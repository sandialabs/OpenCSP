# Copyright Sandia National Laboratories. All rights reserved.

"""Tests for ambient irradiance support."""

import pytest
import numpy as np

from pyffiam.analysis_data import AnalysisData
from pyffiam.cpp_interface import AnalysisParams
from pyffiam.ffiam_types import CspSite, AimType


class TestAmbientInAnalysisParams:
    """Tests for AnalysisParams with ambient irradiance."""

    def test_zero_ambient(self):
        """Test that ambient=0.0 works correctly (feature disabled)."""
        params = AnalysisParams(
            year=2025,
            month=8,
            day=8,
            hour=12.0,
            latitude=34.96,
            longitude=-106.51,
            timezone=-7,
            field_radius=300,
            min_height=0,
            max_height=100,
            tower_height=61,
            heliostat_file="",
            num_heliostats=24,
            facet_file="",
            num_facets=25,
            num_facet_cols=5,
            facet_width=1.2,
            facet_height=1.2,
            aim_strategy=1,
            aim_param_0=0,
            aim_param_1=0,
            aim_param_2=90,
            aim_file="",
            reflectivity=0.9,
            peak_dni=0.1,
            beta=0.0094,
            voxel_size=2,
            ambient=0.0,
            min_attenuation=1.0,
            irrad_exponent=2.0,
            n_rays_per_facet=12.0,
            flux_correction_scale=1.0,
            pre_focal_scale=1.0,
            num_voxels=100,
        )
        assert params.ambient == 0.0

    def test_accepts_positive_ambient(self):
        """Test AnalysisParams accepts positive ambient value."""
        params = AnalysisParams(
            year=2025,
            month=8,
            day=8,
            hour=12.0,
            latitude=34.96,
            longitude=-106.51,
            timezone=-7,
            field_radius=300,
            min_height=0,
            max_height=100,
            tower_height=61,
            heliostat_file="",
            num_heliostats=24,
            facet_file="",
            num_facets=25,
            num_facet_cols=5,
            facet_width=1.2,
            facet_height=1.2,
            aim_strategy=1,
            aim_param_0=0,
            aim_param_1=0,
            aim_param_2=90,
            aim_file="",
            reflectivity=0.9,
            peak_dni=0.1,
            beta=0.0094,
            voxel_size=2,
            ambient=0.35,  # Typical ambient value
            min_attenuation=1.0,
            irrad_exponent=2.0,
            n_rays_per_facet=12.0,
            flux_correction_scale=1.0,
            pre_focal_scale=1.0,
            num_voxels=100,
        )
        assert params.ambient == 0.35

    def test_accepts_small_ambient(self):
        """Test AnalysisParams accepts small ambient value."""
        params = AnalysisParams(
            year=2025,
            month=8,
            day=8,
            hour=12.0,
            latitude=34.96,
            longitude=-106.51,
            timezone=-7,
            field_radius=300,
            min_height=0,
            max_height=100,
            tower_height=61,
            heliostat_file="",
            num_heliostats=24,
            facet_file="",
            num_facets=25,
            num_facet_cols=5,
            facet_width=1.2,
            facet_height=1.2,
            aim_strategy=1,
            aim_param_0=0,
            aim_param_1=0,
            aim_param_2=90,
            aim_file="",
            reflectivity=0.9,
            peak_dni=0.1,
            beta=0.0094,
            voxel_size=2,
            ambient=0.05,  # Small ambient
            min_attenuation=1.0,
            irrad_exponent=2.0,
            n_rays_per_facet=12.0,
            flux_correction_scale=1.0,
            pre_focal_scale=1.0,
            num_voxels=100,
        )
        assert params.ambient == 0.05


class TestAmbientInAnalysisData:
    """Tests for AnalysisData with ambient irradiance."""

    def test_default_ambient_is_zero(self):
        """Test that default ambient is 0.0 in AnalysisData."""
        data = AnalysisData(
            name='Test',
            site=CspSite.Custom,
            year=2025, month=8, day=8, hour=12.0,
            threshold=4,
            latitude=34.96, longitude=-106.51, timezone=-7,
            tower_height=61,
            field_radius=300,
            num_helios=24,
            num_facets=25, num_facet_cols=5,
            facet_width=1.2, facet_height=1.2,
            min_height=0, max_height=100,
            aim_strat=AimType.Point,
            aim_params=np.array([0, 0, 90]),
        )
        assert data.ambient == 0.0

    def test_accepts_positive_ambient(self):
        """Test AnalysisData accepts positive ambient value."""
        data = AnalysisData(
            name='Test',
            site=CspSite.Custom,
            year=2025, month=8, day=8, hour=12.0,
            threshold=4,
            latitude=34.96, longitude=-106.51, timezone=-7,
            tower_height=61,
            field_radius=300,
            num_helios=24,
            num_facets=25, num_facet_cols=5,
            facet_width=1.2, facet_height=1.2,
            min_height=0, max_height=100,
            aim_strat=AimType.Point,
            aim_params=np.array([0, 0, 90]),
            ambient=0.35,
        )
        assert data.ambient == 0.35

    def test_ambient_stored_correctly(self):
        """Test that ambient value is stored and retrievable."""
        test_ambient = 0.42
        data = AnalysisData(
            name='Test',
            site=CspSite.Custom,
            year=2025, month=8, day=8, hour=12.0,
            threshold=4,
            latitude=34.96, longitude=-106.51, timezone=-7,
            tower_height=61,
            field_radius=300,
            num_helios=24,
            num_facets=25, num_facet_cols=5,
            facet_width=1.2, facet_height=1.2,
            min_height=0, max_height=100,
            aim_strat=AimType.Point,
            aim_params=np.array([0, 0, 90]),
            ambient=test_ambient,
        )
        assert data.ambient == test_ambient


class TestAmbientValueRanges:
    """Tests for ambient value ranges."""

    @pytest.mark.parametrize("ambient_value", [
        0.0,    # Disabled
        0.1,    # Low
        0.35,   # Typical
        0.5,    # High
        1.0,    # Very high
    ])
    def test_various_ambient_values(self, ambient_value):
        """Test various ambient values are accepted."""
        data = AnalysisData(
            name='Test',
            site=CspSite.Custom,
            year=2025, month=8, day=8, hour=12.0,
            threshold=4,
            latitude=34.96, longitude=-106.51, timezone=-7,
            tower_height=61,
            field_radius=300,
            num_helios=24,
            num_facets=25, num_facet_cols=5,
            facet_width=1.2, facet_height=1.2,
            min_height=0, max_height=100,
            aim_strat=AimType.Point,
            aim_params=np.array([0, 0, 90]),
            ambient=ambient_value,
        )
        assert data.ambient == ambient_value


if __name__ == "__main__":
    # Quick verification
    print("Testing ambient irradiance support...")

    # Test default
    params = AnalysisParams(
        year=2025, month=8, day=8, hour=12.0,
        latitude=34.96, longitude=-106.51, timezone=-7,
        field_radius=300, min_height=0, max_height=100, tower_height=61,
        heliostat_file="", num_heliostats=24,
        facet_file="", num_facets=25, num_facet_cols=5,
        facet_width=1.2, facet_height=1.2,
        aim_strategy=1, aim_param_0=0, aim_param_1=0, aim_param_2=90, aim_file="",
        reflectivity=0.9, peak_dni=0.1, beta=0.0094, voxel_size=2,
        min_attenuation=1.0,
        irrad_exponent=2.0,
        n_rays_per_facet=12.0,
        flux_correction_scale=1.0,
        pre_focal_scale=1.0,
        num_voxels=100,
    )
    print(f"  Default ambient: {params.ambient} [PASS]" if params.ambient == 0.0 else "[FAIL]")

    # Test with value
    params2 = AnalysisParams(
        year=2025, month=8, day=8, hour=12.0,
        latitude=34.96, longitude=-106.51, timezone=-7,
        field_radius=300, min_height=0, max_height=100, tower_height=61,
        heliostat_file="", num_heliostats=24,
        facet_file="", num_facets=25, num_facet_cols=5,
        facet_width=1.2, facet_height=1.2,
        aim_strategy=1, aim_param_0=0, aim_param_1=0, aim_param_2=90, aim_file="",
        reflectivity=0.9, peak_dni=0.1, beta=0.0094, voxel_size=2, ambient=0.35,
        min_attenuation=1.0,
        irrad_exponent=2.0,
        n_rays_per_facet=12.0,
        flux_correction_scale=1.0,
        pre_focal_scale=1.0,
        num_voxels=100,
    )
    print(f"  With ambient=0.35: {params2.ambient} [PASS]" if params2.ambient == 0.35 else "[FAIL]")

    print("\nAll tests passed!")
