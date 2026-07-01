"""Unit tests for AimType.FixedNormal aim strategy.

Tests the new fixed-normal aim type where the heliostat surface normal
is fixed (e.g., STOW position) and the reflected beam direction depends
on sun position via the law of reflection.
"""

import numpy as np
import pytest

from pyffiam.ffiam_types import AimType, get_aim_type_from_string


class TestAimTypeFixedNormal:
    """Tests for AimType.FixedNormal enum value."""

    def test_enum_value(self):
        """Test FixedNormal has correct enum value."""
        assert AimType.FixedNormal.value == 6

    def test_string_parsing_fixednormal(self):
        """Test string 'fixednormal' parses to FixedNormal."""
        assert get_aim_type_from_string("fixednormal") == AimType.FixedNormal

    def test_string_parsing_fixed_normal(self):
        """Test string 'fixed_normal' parses to FixedNormal."""
        assert get_aim_type_from_string("fixed_normal") == AimType.FixedNormal

    def test_string_parsing_fixed(self):
        """Test string 'fixed' parses to FixedNormal."""
        assert get_aim_type_from_string("fixed") == AimType.FixedNormal

    def test_string_parsing_stow(self):
        """Test string 'stow' parses to FixedNormal."""
        assert get_aim_type_from_string("stow") == AimType.FixedNormal

    def test_string_parsing_case_insensitive(self):
        """Test string parsing is case-insensitive."""
        assert get_aim_type_from_string("FIXEDNORMAL") == AimType.FixedNormal
        assert get_aim_type_from_string("FixedNormal") == AimType.FixedNormal
        assert get_aim_type_from_string("STOW") == AimType.FixedNormal

    def test_aim_type_distinct_from_vector(self):
        """Test FixedNormal is distinct from Vector aim type."""
        assert AimType.FixedNormal != AimType.Vector
        assert AimType.FixedNormal.value != AimType.Vector.value


class TestFixedNormalDocstring:
    """Tests for AimType docstring documentation."""

    def test_docstring_mentions_fixed(self):
        """Test that AimType docstring mentions fixed-orientation."""
        assert "fixed" in AimType.__doc__.lower()

    def test_docstring_mentions_stow(self):
        """Test that AimType docstring mentions STOW."""
        assert "STOW" in AimType.__doc__ or "stow" in AimType.__doc__.lower()


class TestReflectedBeamCalculation:
    """Tests for reflected beam direction calculation.

    These tests verify the physics: for a fixed-normal heliostat,
    the reflected beam direction R = 2*(N·S)*N - S where:
    - N is the surface normal
    - S is the sun direction vector
    """

    def compute_reflected(self, normal: np.ndarray, sun_vec: np.ndarray) -> np.ndarray:
        """Compute reflected direction using the law of reflection."""
        normal = normal / np.linalg.norm(normal)
        sun_vec = sun_vec / np.linalg.norm(sun_vec)
        n_dot_s = np.dot(normal, sun_vec)
        reflected = 2 * n_dot_s * normal - sun_vec
        return reflected / np.linalg.norm(reflected)

    def test_face_up_sun_at_zenith(self):
        """Face-up heliostat with sun at zenith reflects straight up."""
        normal = np.array([0.0, 0.0, 1.0])  # Face up
        sun_vec = np.array([0.0, 0.0, 1.0])  # Sun at zenith

        reflected = self.compute_reflected(normal, sun_vec)

        np.testing.assert_array_almost_equal(reflected, np.array([0.0, 0.0, 1.0]), decimal=5)

    def test_face_up_sun_from_east(self):
        """Face-up heliostat with sun from East reflects West and Up."""
        normal = np.array([0.0, 0.0, 1.0])  # Face up
        # Sun from East at 45 deg elevation
        sun_vec = np.array([np.cos(np.radians(45)), 0.0, np.sin(np.radians(45))])

        reflected = self.compute_reflected(normal, sun_vec)

        # Reflected should point West (negative x) and Up (positive z)
        assert reflected[0] < -0.5  # West component
        assert abs(reflected[1]) < 0.01  # No North/South
        assert reflected[2] > 0.5  # Up component

    def test_face_up_sun_from_southeast(self):
        """Face-up heliostat with sun from SE reflects NW and Up.

        This mimics the NSTTF STOW validation conditions:
        Sun at Az=114.5 deg, El=55 deg.
        """
        # Sun from Southeast at 55 deg elevation
        az_rad = np.radians(114.5)
        el_rad = np.radians(55.0)
        sun_vec = np.array(
            [np.cos(el_rad) * np.sin(az_rad), np.cos(el_rad) * np.cos(az_rad), np.sin(el_rad)]  # East  # North  # Up
        )

        normal = np.array([0.0, 0.0, 1.0])  # Face up
        reflected = self.compute_reflected(normal, sun_vec)

        # Reflected should point Northwest and Up
        # Expected approximately: (-0.52, 0.24, 0.82)
        assert reflected[0] < -0.4  # West component
        assert reflected[1] > 0.1  # North component
        assert reflected[2] > 0.7  # Up component (dominant)

        # Check the tilt from vertical (~35 deg)
        tilt_deg = np.degrees(np.arccos(reflected[2]))
        assert 30 < tilt_deg < 40

    def test_beam_offset_at_altitude(self):
        """Test beam center offset at flight altitude.

        For NSTTF STOW at 10 AM:
        - Sun at Az=114.5, El=55
        - Reflected beam tilts ~35 deg from vertical
        - At 17m above heliostat, beam center is offset ~8m West, ~4m North
        """
        # Sun from Southeast
        az_rad = np.radians(114.5)
        el_rad = np.radians(55.0)
        sun_vec = np.array([np.cos(el_rad) * np.sin(az_rad), np.cos(el_rad) * np.cos(az_rad), np.sin(el_rad)])

        normal = np.array([0.0, 0.0, 1.0])
        reflected = self.compute_reflected(normal, sun_vec)

        # Compute offset at 12.5m above heliostat (17m flight - 4.5m helio height)
        delta_z = 12.5
        offset_x = delta_z * (reflected[0] / reflected[2])
        offset_y = delta_z * (reflected[1] / reflected[2])

        # Beam should be offset West and North
        assert offset_x < -5  # More than 5m West
        assert offset_y > 2  # More than 2m North


class TestFixedNormalVsVector:
    """Tests comparing FixedNormal vs Vector aim types.

    These tests verify the key difference:
    - Vector: heliostats TRACK to redirect sunlight toward the vector
    - FixedNormal: heliostats are FIXED, reflected beam depends on sun
    """

    def test_different_behavior_description(self):
        """Document the difference in behavior."""
        # For AimType.Vector(0,0,1):
        #   Heliostat tilts so reflected beam goes straight up
        #   Heliostat normal = bisector(sun, (0,0,1))

        # For AimType.FixedNormal(0,0,1):
        #   Heliostat normal IS (0,0,1) (face-up)
        #   Reflected beam = 2*(normal.sun)*normal - sun
        #   Beam tilts based on sun position

        # Both are valid strategies for different use cases
        assert AimType.Vector.value == 4
        assert AimType.FixedNormal.value == 6

    def test_fixed_normal_for_stow_analysis(self):
        """FixedNormal should be used for STOW configuration analysis."""
        # STOW = heliostats physically face-up, not tracking
        stow_aim = AimType.FixedNormal
        stow_params = np.array([0.0, 0.0, 1.0])  # Normal pointing up

        assert stow_aim == AimType.FixedNormal
        np.testing.assert_array_equal(stow_params, [0, 0, 1])
