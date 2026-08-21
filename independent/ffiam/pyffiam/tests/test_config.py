# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Unit tests for pyffiam config and results dataclasses.

These tests cover the configuration classes and results processing,
including factory methods and computed properties.
"""

from datetime import datetime

import numpy as np
import pytest

from pyffiam.ffiam_types import CspSite, AimType
from pyffiam.config import (
    TimeConfig,
    LocationConfig,
    FieldConfig,
    HeliostatConfig,
    AimConfig,
    VoxelConfig,
    OpticalConfig,
    AnalysisConfig,
)
from pyffiam.results import HeliostatResults, IrradianceResults, ThresholdResults, AnalysisResults


class TestTimeConfig:
    """Tests for TimeConfig dataclass."""

    def test_datetime_property(self, sample_time_config):
        """Test datetime property returns correct datetime object."""
        dt = sample_time_config.datetime
        assert dt.year == 2025
        assert dt.month == 6
        assert dt.day == 21
        assert dt.hour == 12

    def test_datetime_str_format(self, sample_time_config):
        """Test datetime_str returns correct format."""
        # Should be YYYYMMDDHHMM
        assert sample_time_config.datetime_str == "202506211200"

    def test_immutable(self, sample_time_config):
        """Test that TimeConfig is frozen/immutable."""
        with pytest.raises(AttributeError):
            sample_time_config.year = 2026


class TestLocationConfig:
    """Tests for LocationConfig dataclass."""

    def test_creation(self, sample_location_config):
        """Test basic creation and attribute access."""
        assert sample_location_config.latitude == 34.96
        assert sample_location_config.longitude == -106.51
        assert sample_location_config.timezone == -7.0

    def test_immutable(self, sample_location_config):
        """Test that LocationConfig is frozen/immutable."""
        with pytest.raises(AttributeError):
            sample_location_config.latitude = 40.0


class TestFieldConfig:
    """Tests for FieldConfig dataclass."""

    def test_creation(self, sample_field_config):
        """Test basic creation and attribute access."""
        assert sample_field_config.radius == 600
        assert sample_field_config.num_heliostats == 218
        assert sample_field_config.min_height == 4
        assert sample_field_config.max_height == 200
        assert sample_field_config.tower_height == 90.0


class TestHeliostatConfig:
    """Tests for HeliostatConfig dataclass."""

    def test_creation(self, sample_heliostat_config):
        """Test basic creation and attribute access."""
        assert sample_heliostat_config.num_facets == 25
        assert sample_heliostat_config.num_facet_cols == 5
        assert sample_heliostat_config.facet_width == 1.2
        assert sample_heliostat_config.facet_height == 1.2

    def test_heliostat_width_calculation(self, sample_heliostat_config):
        """Test heliostat_width property calculates correctly."""
        # width = facet_width * num_facet_cols = 1.2 * 5 = 6.0
        assert sample_heliostat_config.heliostat_width == 6.0

    def test_default_files(self):
        """Test default empty strings for file paths."""
        config = HeliostatConfig(num_facets=25, num_facet_cols=5, facet_width=1.0, facet_height=1.0)
        assert config.heliostat_file == ""
        assert config.facet_file == ""


class TestAimConfig:
    """Tests for AimConfig dataclass."""

    def test_creation(self, sample_aim_config):
        """Test basic creation and attribute access."""
        assert sample_aim_config.strategy == AimType.Point
        np.testing.assert_array_equal(sample_aim_config.parameters, np.array([0.0, 0.0, 90.0]))

    def test_default_aim_file(self):
        """Test default empty string for aim file."""
        config = AimConfig(strategy=AimType.Point, parameters=np.array([0.0, 0.0, 90.0]))
        assert config.aim_file == ""


class TestVoxelConfig:
    """Tests for VoxelConfig dataclass."""

    def test_default_size(self):
        """Test default voxel size is 2."""
        config = VoxelConfig()
        assert config.size == 2

    def test_area_property(self, sample_voxel_config):
        """Test voxel face area calculation."""
        # size=2, area = 2^2 = 4
        assert sample_voxel_config.area == 4

    def test_volume_property(self, sample_voxel_config):
        """Test voxel volume calculation."""
        # size=2, volume = 2^3 = 8
        assert sample_voxel_config.volume == 8

    def test_compute_layout(self, sample_voxel_config):
        """Test voxel grid layout computation."""
        layout = sample_voxel_config.compute_layout(field_radius=100, min_height=4, max_height=104)
        # num_x = 2 * 100 / 2 = 100
        # num_y = 2 * 100 / 2 = 100
        # num_z = (104 - 4) / 2 + 1 = 51
        assert layout == (100, 100, 51)

    def test_compute_num_voxels(self, sample_voxel_config):
        """Test total voxel count computation."""
        n = sample_voxel_config.compute_num_voxels(field_radius=10, min_height=4, max_height=14)
        # layout = (10, 10, 6)
        # total = 10 * 10 * 6 = 600
        assert n == 600


class TestOpticalConfig:
    """Tests for OpticalConfig dataclass."""

    def test_defaults(self):
        """Test default optical parameters."""
        config = OpticalConfig()
        assert config.reflectivity == 0.9
        assert config.peak_dni == 0.1
        assert config.sun_angle == 0.0093
        assert config.slope_error == 0.0012
        assert config.beta == 0.0094

    def test_custom_values(self, sample_optical_config):
        """Test custom optical parameters."""
        assert sample_optical_config.reflectivity == 0.9
        assert sample_optical_config.peak_dni == 0.1


class TestAnalysisConfig:
    """Tests for AnalysisConfig dataclass."""

    def test_creation(self, sample_analysis_config):
        """Test complete analysis config creation."""
        assert sample_analysis_config.name == "Test Analysis"
        assert sample_analysis_config.site == CspSite.NSTTF
        assert sample_analysis_config.threshold == 4.0

    def test_has_paths_empty(self, sample_analysis_config):
        """Test has_paths when no paths configured."""
        assert sample_analysis_config.has_paths is False

    def test_has_paths_with_paths(self, sample_analysis_config):
        """Test has_paths when paths are configured."""
        sample_analysis_config.paths = [[[0, 0, 0], [1, 1, 1]]]
        assert sample_analysis_config.has_paths is True

    def test_slug_property(self, sample_analysis_config):
        """Test slug property converts name correctly."""
        # "Test Analysis" -> "test_analysis"
        assert sample_analysis_config.slug == "test_analysis"

    def test_created_at_str(self, sample_analysis_config):
        """Test created_at_str format."""
        # Should be YYYYMMDDHHMM format
        ts = sample_analysis_config.created_at_str
        assert len(ts) == 12
        # Should be parseable as a date
        datetime.strptime(ts, "%Y%m%d%H%M")

    def test_from_params_factory(self):
        """Test AnalysisConfig.from_params factory method."""
        config = AnalysisConfig.from_params(
            name="Factory Test",
            site=CspSite.NSTTF,
            year=2025,
            month=6,
            day=21,
            hour=12,
            latitude=34.96,
            longitude=-106.51,
            timezone=-7.0,
            field_radius=600,
            num_heliostats=218,
            min_height=4,
            max_height=200,
            tower_height=90.0,
            num_facets=25,
            num_facet_cols=5,
            facet_width=1.2,
            facet_height=1.2,
            aim_strategy=AimType.Point,
            aim_parameters=np.array([0.0, 0.0, 90.0]),
            threshold=4.0,
        )

        assert config.name == "Factory Test"
        assert config.site == CspSite.NSTTF
        assert config.time.year == 2025
        assert config.location.latitude == 34.96
        assert config.field.radius == 600
        assert config.heliostat.num_facets == 25
        assert config.aim.strategy == AimType.Point
        assert config.threshold == 4.0


class TestIrradianceResults:
    """Tests for IrradianceResults dataclass."""

    def test_from_array_basic(self):
        """Test from_array factory with simple data."""
        irrads = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        results = IrradianceResults.from_array(irrads)

        assert results.total == 10  # sum
        assert results.peak == 4.0  # max
        assert results.num_impacted == 4  # count > 0

    def test_from_array_with_zeros(self):
        """Test from_array with zero values."""
        irrads = np.array([0.0, 0.0, 5.0, 0.0, 3.0])
        results = IrradianceResults.from_array(irrads)

        assert results.total == 8
        assert results.peak == 5.0
        assert results.num_impacted == 2

    def test_from_array_all_zeros(self):
        """Test from_array with all zero values."""
        irrads = np.zeros(100)
        results = IrradianceResults.from_array(irrads)

        assert results.total == 0
        assert results.peak == 0.0
        assert results.num_impacted == 0

    def test_from_array_with_nan(self):
        """Test from_array handles NaN values."""
        irrads = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        results = IrradianceResults.from_array(irrads)

        # nansum ignores NaN
        assert results.total == 6
        # nanmax ignores NaN
        assert results.peak == 3.0


class TestThresholdResults:
    """Tests for ThresholdResults dataclass."""

    def test_from_irradiance_basic(self, small_voxel_params):
        """Test from_irradiance factory method."""
        irrads = np.array([1.0, 2.0, 5.0, 10.0, 0.5])
        threshold = 4.0

        results = ThresholdResults.from_irradiance(
            irrads=irrads,
            threshold=threshold,
            voxel_size=small_voxel_params['vox_size'],
            field_radius=small_voxel_params['field_r'],
            min_height=small_voxel_params['field_zmin'],
        )

        # Only values > 4.0 are 5.0 and 10.0
        assert results.num_voxels == 2
        assert results.has_glare is True
        np.testing.assert_array_equal(results.irradiances, np.array([5.0, 10.0]))
        np.testing.assert_array_equal(results.voxel_ids, np.array([2, 3]))

    def test_from_irradiance_no_glare(self, small_voxel_params):
        """Test from_irradiance when no values exceed threshold."""
        irrads = np.array([1.0, 2.0, 3.0])
        threshold = 5.0

        results = ThresholdResults.from_irradiance(
            irrads=irrads,
            threshold=threshold,
            voxel_size=small_voxel_params['vox_size'],
            field_radius=small_voxel_params['field_r'],
            min_height=small_voxel_params['field_zmin'],
        )

        assert results.num_voxels == 0
        assert results.has_glare is False
        assert len(results.irradiances) == 0

    def test_total_irradiance_calculation(self, small_voxel_params):
        """Test total_irradiance sums only above-threshold values."""
        irrads = np.array([1.0, 2.0, 5.0, 10.0, 3.0])
        threshold = 4.0

        results = ThresholdResults.from_irradiance(
            irrads=irrads,
            threshold=threshold,
            voxel_size=small_voxel_params['vox_size'],
            field_radius=small_voxel_params['field_r'],
            min_height=small_voxel_params['field_zmin'],
        )

        # 5.0 + 10.0 = 15
        assert results.total_irradiance == 15


class TestHeliostatResults:
    """Tests for HeliostatResults dataclass."""

    def test_creation(self, sample_heliostat_results):
        """Test basic creation and attribute access."""
        assert sample_heliostat_results.locations.shape[1] == 3
        assert sample_heliostat_results.aim_vectors.shape[1] == 3
        assert sample_heliostat_results.total_movement >= 0


class TestAnalysisResults:
    """Tests for AnalysisResults dataclass."""

    def test_has_results(self, sample_heliostat_results, sample_irradiance_results):
        """Test has_results property."""
        threshold = ThresholdResults(
            mask=np.array([True, False]),
            irradiances=np.array([5.0]),
            voxel_ids=np.array([0]),
            voxel_locations=np.array([[0, 0, 4]]),
            total_irradiance=5,
            num_voxels=1,
            has_glare=True,
        )

        results = AnalysisResults(
            heliostats=sample_heliostat_results,
            irradiance=sample_irradiance_results,
            threshold=threshold,
            voxel_locations=np.array([[0, 0, 0]]),
            voxel_layout=(10, 10, 5),
        )

        assert results.has_results is True

    def test_convenience_accessors(self, sample_heliostat_results, sample_irradiance_results):
        """Test convenience property accessors."""
        threshold = ThresholdResults(
            mask=np.array([True, False]),
            irradiances=np.array([5.0]),
            voxel_ids=np.array([0]),
            voxel_locations=np.array([[0, 0, 4]]),
            total_irradiance=5,
            num_voxels=1,
            has_glare=True,
        )

        results = AnalysisResults(
            heliostats=sample_heliostat_results,
            irradiance=sample_irradiance_results,
            threshold=threshold,
            voxel_locations=np.array([[0, 0, 0]]),
            voxel_layout=(10, 10, 5),
        )

        # These should delegate to nested objects
        assert results.total_irrad == sample_irradiance_results.total
        assert results.peak_irrad == sample_irradiance_results.peak
        assert results.n_glaring_voxels == 1
        assert results.has_glare is True

    def test_from_raw_data_factory(self, small_voxel_params):
        """Test from_raw_data factory method."""
        n_heliostats = 10
        n_voxels = 100

        results = AnalysisResults.from_raw_data(
            helio_locs=np.random.randn(n_heliostats, 3),
            helio_aim_vs=np.tile([0, 0, 1], (n_heliostats, 1)).astype(np.float64),
            helio_angles=np.random.uniform(0, 360, n_heliostats),
            facet_origins=np.zeros((n_heliostats, 25, 3)),
            irrads=np.random.exponential(2.0, n_voxels),
            movement=45.0,
            voxel_locs=np.random.randn(n_voxels, 3),
            voxel_layout=(10, 10, 1),
            threshold=4.0,
            voxel_size=small_voxel_params['vox_size'],
            field_radius=small_voxel_params['field_r'],
            min_height=small_voxel_params['field_zmin'],
        )

        assert results.heliostats.locations.shape == (n_heliostats, 3)
        assert results.irradiance.values.shape == (n_voxels,)
        assert results.voxel_layout == (10, 10, 1)
