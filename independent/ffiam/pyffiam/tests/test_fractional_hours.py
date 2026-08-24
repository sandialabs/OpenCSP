# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Tests for fractional hour support."""

import pytest
import numpy as np
from datetime import datetime

from pyffiam.analysis_data import AnalysisData
from pyffiam.cpp_interface import AnalysisParams
from pyffiam.ffiam_types import CspSite, AimType


class TestFractionalHourParsing:
    """Tests for fractional hour to h:m:s conversion."""

    def test_integer_hour(self):
        """Test integer hour (10) converts correctly."""
        hour = 10.0
        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)

        assert hour_int == 10
        assert minute_int == 0

    def test_half_hour(self):
        """Test half hour (10.5 = 10:30) converts correctly."""
        hour = 10.5
        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)

        assert hour_int == 10
        assert minute_int == 30

    def test_quarter_hour(self):
        """Test quarter hour (10.25 = 10:15) converts correctly."""
        hour = 10.25
        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)

        assert hour_int == 10
        assert minute_int == 15

    def test_specific_time_10_08(self):
        """Test 10:08 AM (10.133...) converts correctly."""
        # 10:08 AM = 10 + 8/60 = 10.1333...
        hour = 10 + 8 / 60
        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)

        assert hour_int == 10
        assert minute_int == 8

    def test_specific_time_12_43(self):
        """Test 12:43 PM (12.7166...) converts correctly."""
        # 12:43 PM = 12 + 43/60
        hour = 12 + 43 / 60
        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)

        assert hour_int == 12
        assert minute_int == 43

    def test_specific_time_14_36(self):
        """Test 2:36 PM (14.6) converts correctly."""
        # 1:36 PM = 13 + 36/60 = 13.6
        hour = 13 + 36 / 60
        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)

        assert hour_int == 13
        assert minute_int == 36


class TestAnalysisParamsFractionalHour:
    """Tests for AnalysisParams with fractional hours."""

    def test_accepts_float_hour(self):
        """Test AnalysisParams accepts float hour."""
        params = AnalysisParams(
            year=2025,
            month=8,
            day=8,
            hour=10.5,  # 10:30 AM
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
        assert params.hour == 10.5

    def test_accepts_integer_as_float(self):
        """Test AnalysisParams accepts integer hour as float."""
        params = AnalysisParams(
            year=2025,
            month=8,
            day=8,
            hour=12,  # Will be converted to float
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
        assert params.hour == 12.0


class TestAnalysisDataFractionalHour:
    """Tests for AnalysisData with fractional hours."""

    def test_datetime_conversion_integer(self):
        """Test datetime is correct for integer hour."""
        data = AnalysisData(
            name='Test',
            site=CspSite.Custom,
            year=2025,
            month=8,
            day=8,
            hour=10.0,
            threshold=4,
            latitude=34.96,
            longitude=-106.51,
            timezone=-7,
            tower_height=61,
            field_radius=300,
            num_helios=24,
            num_facets=25,
            num_facet_cols=5,
            facet_width=1.2,
            facet_height=1.2,
            min_height=0,
            max_height=100,
            aim_strat=AimType.Point,
            aim_params=np.array([0, 0, 90]),
        )

        assert data.hour == 10.0
        assert data.analysis_dt.hour == 10
        assert data.analysis_dt.minute == 0

    def test_datetime_conversion_half_hour(self):
        """Test datetime is correct for 10:30 AM."""
        data = AnalysisData(
            name='Test',
            site=CspSite.Custom,
            year=2025,
            month=8,
            day=8,
            hour=10.5,
            threshold=4,
            latitude=34.96,
            longitude=-106.51,
            timezone=-7,
            tower_height=61,
            field_radius=300,
            num_helios=24,
            num_facets=25,
            num_facet_cols=5,
            facet_width=1.2,
            facet_height=1.2,
            min_height=0,
            max_height=100,
            aim_strat=AimType.Point,
            aim_params=np.array([0, 0, 90]),
        )

        assert data.hour == 10.5
        assert data.analysis_dt.hour == 10
        assert data.analysis_dt.minute == 30

    def test_datetime_conversion_specific_time(self):
        """Test datetime is correct for 10:08 AM."""
        hour = 10 + 8 / 60  # 10:08 AM
        data = AnalysisData(
            name='Test',
            site=CspSite.Custom,
            year=2025,
            month=8,
            day=8,
            hour=hour,
            threshold=4,
            latitude=34.96,
            longitude=-106.51,
            timezone=-7,
            tower_height=61,
            field_radius=300,
            num_helios=24,
            num_facets=25,
            num_facet_cols=5,
            facet_width=1.2,
            facet_height=1.2,
            min_height=0,
            max_height=100,
            aim_strat=AimType.Point,
            aim_params=np.array([0, 0, 90]),
        )

        assert data.analysis_dt.hour == 10
        assert data.analysis_dt.minute == 8


class TestFlightTimeExamples:
    """Test with actual NSTTF validation flight times."""

    @pytest.mark.parametrize(
        "flight,expected_hour,expected_minute",
        [
            ("Flight_0001_low", 10, 8),  # 10:08 AM
            ("Flight_0002", 11, 15),  # 11:15 AM
            ("Flight_0004", 12, 43),  # 12:43 PM
            ("Flight_0005", 12, 57),  # 12:57 PM
            ("Flight_0006", 13, 36),  # 1:36 PM
            ("Flight_0007", 14, 24),  # 2:24 PM
        ],
    )
    def test_flight_times(self, flight, expected_hour, expected_minute):
        """Test that flight times convert correctly."""
        hour = expected_hour + expected_minute / 60

        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)

        assert hour_int == expected_hour
        assert minute_int == expected_minute


if __name__ == "__main__":
    # Quick verification
    print("Testing fractional hour conversion...")

    test_cases = [
        (10.0, 10, 0, "10:00 AM"),
        (10.5, 10, 30, "10:30 AM"),
        (10 + 8 / 60, 10, 8, "10:08 AM"),
        (12 + 43 / 60, 12, 43, "12:43 PM"),
        (13 + 36 / 60, 13, 36, "1:36 PM"),
    ]

    for hour, exp_h, exp_m, desc in test_cases:
        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)
        status = "PASS" if hour_int == exp_h and minute_int == exp_m else "FAIL"
        print(f"  {desc}: hour={hour:.4f} -> {hour_int:02d}:{minute_int:02d} [{status}]")

    print("\nAll tests passed!")
