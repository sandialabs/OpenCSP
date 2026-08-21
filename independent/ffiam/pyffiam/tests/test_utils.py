# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Unit tests for pyffiam utility functions.

These tests cover the core utility functions including voxel indexing,
interpolation, and string manipulation.
"""

import numpy as np
import pytest

from pyffiam.utils import (
    slug,
    get_voxel_loc_from_index,
    get_voxel_locs_from_indexes,
    compute_vox_locs,
    get_voxel_indexes_from_locs,
    interpolate_3d,
    get_reduced_voxels_with_irrads,
)


class TestSlug:
    """Tests for the slug() function."""

    def test_lowercase_conversion(self):
        """Test that uppercase letters are converted to lowercase."""
        assert slug("Hello World") == "hello_world"

    def test_space_replacement(self):
        """Test that spaces are replaced with underscores."""
        assert slug("test string") == "test_string"

    def test_multiple_spaces(self):
        """Test handling of multiple spaces."""
        assert slug("a  b   c") == "a__b___c"

    def test_already_slug(self):
        """Test that valid slugs remain unchanged."""
        assert slug("already_valid") == "already_valid"

    def test_empty_string(self):
        """Test handling of empty string."""
        assert slug("") == ""

    def test_mixed_case_with_spaces(self):
        """Test mixed case with spaces."""
        assert slug("NSTTF Site Analysis") == "nsttf_site_analysis"


class TestVoxelIndexing:
    """Tests for voxel coordinate conversion functions."""

    def test_get_voxel_loc_from_index_origin(self, small_voxel_params):
        """Test that index 0 maps to the correct origin location."""
        loc = get_voxel_loc_from_index(
            idx=0,
            vox_size=small_voxel_params['vox_size'],
            field_r=small_voxel_params['field_r'],
            field_zmin=small_voxel_params['field_zmin'],
        )
        expected = np.array(
            [-small_voxel_params['field_r'], -small_voxel_params['field_r'], small_voxel_params['field_zmin']]
        )
        np.testing.assert_array_almost_equal(loc, expected)

    def test_get_voxel_loc_from_index_second_plane(self, small_voxel_params):
        """Test voxel location for first voxel on second z-plane."""
        # n_per_plane = (field_r * 2 / vox_size) ** 2 = (10 * 2 / 2) ** 2 = 100
        n_per_plane = int((small_voxel_params['field_r'] * 2 / small_voxel_params['vox_size']) ** 2)
        loc = get_voxel_loc_from_index(
            idx=n_per_plane,
            vox_size=small_voxel_params['vox_size'],
            field_r=small_voxel_params['field_r'],
            field_zmin=small_voxel_params['field_zmin'],
        )
        # Should be at origin x,y but one voxel_size up in z
        expected_z = small_voxel_params['field_zmin'] + small_voxel_params['vox_size']
        assert loc[2] == expected_z

    def test_get_voxel_locs_from_indexes_shape(self, small_voxel_params):
        """Test that output shape is correct for multiple indices."""
        ids = np.array([0, 1, 2, 3, 4])
        locs = get_voxel_locs_from_indexes(
            ids=ids,
            vox_size=small_voxel_params['vox_size'],
            field_r=small_voxel_params['field_r'],
            field_zmin=small_voxel_params['field_zmin'],
        )
        assert locs.shape == (5, 3)

    def test_compute_vox_locs_count(self, small_voxel_params):
        """Test that correct number of voxel locations are computed."""
        n_voxels = 100
        locs = compute_vox_locs(
            n_voxels=n_voxels,
            vox_size=small_voxel_params['vox_size'],
            field_r=small_voxel_params['field_r'],
            field_zmin=small_voxel_params['field_zmin'],
        )
        assert len(locs) == n_voxels

    def test_voxel_index_round_trip(self, small_voxel_params):
        """Test that converting index->loc->index gives original index."""
        original_ids = np.array([0, 5, 10, 50])
        locs = get_voxel_locs_from_indexes(
            ids=original_ids,
            vox_size=small_voxel_params['vox_size'],
            field_r=small_voxel_params['field_r'],
            field_zmin=small_voxel_params['field_zmin'],
        )
        recovered_ids = get_voxel_indexes_from_locs(
            locs=locs,
            vox_size=small_voxel_params['vox_size'],
            field_r=small_voxel_params['field_r'],
            field_zmin=small_voxel_params['field_zmin'],
        )
        np.testing.assert_array_almost_equal(recovered_ids, original_ids)

    def test_voxel_z_coordinates_increase(self, small_voxel_params):
        """Test that z coordinates increase as expected with index."""
        n_per_side = small_voxel_params['field_r'] * 2 / small_voxel_params['vox_size']
        n_per_plane = int(n_per_side**2)

        # Get locations at index 0 and n_per_plane (should be one z level up)
        loc0 = get_voxel_loc_from_index(
            idx=0,
            vox_size=small_voxel_params['vox_size'],
            field_r=small_voxel_params['field_r'],
            field_zmin=small_voxel_params['field_zmin'],
        )
        loc_z1 = get_voxel_loc_from_index(
            idx=n_per_plane,
            vox_size=small_voxel_params['vox_size'],
            field_r=small_voxel_params['field_r'],
            field_zmin=small_voxel_params['field_zmin'],
        )

        # Z should increase by voxel_size
        assert loc_z1[2] - loc0[2] == small_voxel_params['vox_size']


class TestInterpolation:
    """Tests for the interpolate_3d() function."""

    def test_interpolate_3d_step_count(self):
        """Test that correct number of interpolation points are generated."""
        points = interpolate_3d([0, 0, 0], [10, 0, 0], step_size=2)
        # Distance is 10, step_size is 2, so steps = 5
        # Points: 0, 2, 4, 6, 8 (excluding endpoint)
        assert len(points) == 5

    def test_interpolate_3d_includes_start(self):
        """Test that start point is included."""
        points = interpolate_3d([0, 0, 0], [10, 0, 0], step_size=2)
        np.testing.assert_array_almost_equal(points[0], [0, 0, 0])

    def test_interpolate_3d_excludes_end(self):
        """Test that end point is excluded."""
        points = interpolate_3d([0, 0, 0], [10, 0, 0], step_size=2)
        # Last point should be at 8, not 10
        np.testing.assert_array_almost_equal(points[-1], [8, 0, 0])

    def test_interpolate_3d_diagonal(self):
        """Test interpolation along a diagonal."""
        points = interpolate_3d([0, 0, 0], [3, 4, 0], step_size=1)
        # Distance is 5, so 5 points
        assert len(points) == 5

    def test_interpolate_3d_3d_direction(self):
        """Test interpolation in 3D space."""
        points = interpolate_3d([0, 0, 0], [0, 0, 10], step_size=2)
        # Should move only in z direction
        for p in points:
            assert p[0] == 0
            assert p[1] == 0

    def test_interpolate_3d_single_step(self):
        """Test when distance equals step size."""
        points = interpolate_3d([0, 0, 0], [1, 0, 0], step_size=1)
        assert len(points) == 1
        np.testing.assert_array_almost_equal(points[0], [0, 0, 0])

    def test_interpolate_3d_numpy_input(self):
        """Test that numpy arrays work as input."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([6.0, 0.0, 0.0])
        points = interpolate_3d(p1, p2, step_size=2)
        assert len(points) == 3


class TestReducedVoxels:
    """Tests for get_reduced_voxels_with_irrads()."""

    def test_reduces_to_max_irrad_per_xy(self):
        """Test that only maximum irradiance per x-y position is kept."""
        # Two voxels at same x,y but different z
        voxel_xs = np.array([0.0, 0.0, 1.0])
        voxel_ys = np.array([0.0, 0.0, 1.0])
        irrads = np.array([5.0, 10.0, 3.0])

        result = get_reduced_voxels_with_irrads(voxel_xs, voxel_ys, irrads)

        # Should have 2 unique x-y pairs
        assert result.shape[1] == 2

        # The (0,0) position should have max irrad of 10
        mask_00 = (result[0] == 0) & (result[1] == 0)
        assert result[2][mask_00][0] == 10.0

    def test_output_shape(self):
        """Test output shape is (3, N)."""
        voxel_xs = np.array([0.0, 1.0, 2.0])
        voxel_ys = np.array([0.0, 1.0, 2.0])
        irrads = np.array([1.0, 2.0, 3.0])

        result = get_reduced_voxels_with_irrads(voxel_xs, voxel_ys, irrads)

        assert result.shape[0] == 3  # x, y, irrad

    def test_single_voxel(self):
        """Test with single voxel input."""
        voxel_xs = np.array([0.0])
        voxel_ys = np.array([0.0])
        irrads = np.array([5.0])

        result = get_reduced_voxels_with_irrads(voxel_xs, voxel_ys, irrads)

        assert result.shape == (3, 1)
        assert result[2][0] == 5.0


class TestVoxelConfigComputation:
    """Tests for VoxelConfig computation methods."""

    def test_compute_layout(self):
        """Test voxel layout computation."""
        from pyffiam.config import VoxelConfig

        config = VoxelConfig(size=2)
        layout = config.compute_layout(field_radius=100, min_height=4, max_height=104)

        # With radius 100 and size 2: 2*100/2 = 100 voxels per side
        assert layout[0] == 100
        assert layout[1] == 100
        # Height: (104-4)/2 + 1 = 51
        assert layout[2] == 51

    def test_compute_num_voxels(self):
        """Test total voxel count computation."""
        from pyffiam.config import VoxelConfig

        config = VoxelConfig(size=2)
        n = config.compute_num_voxels(field_radius=10, min_height=4, max_height=14)

        # 10x10x6 grid
        expected = 10 * 10 * 6
        assert n == expected

    def test_voxel_area(self):
        """Test voxel face area calculation."""
        from pyffiam.config import VoxelConfig

        config = VoxelConfig(size=2)
        assert config.area == 4  # 2^2

    def test_voxel_volume(self):
        """Test voxel volume calculation."""
        from pyffiam.config import VoxelConfig

        config = VoxelConfig(size=3)
        assert config.volume == 27  # 3^3
