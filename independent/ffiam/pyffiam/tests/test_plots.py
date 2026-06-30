"""Unit tests for pyffiam.plots — focuses on the pure helper functions that
don't require rendering. Heavy rendering paths (write_image, GIF generation)
are exercised by integration tests in test_outputs.py; here we assert the
math/transformations that *produce* plot inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyffiam.ffiam_types import AimType, Direction
from pyffiam.plots import (
    ZOOM_SPAN,
    get_zoom_lims,
    _get_axis_plotting_data,
    _get_gif_depth_axis_data,
    _calculate_plot_parameters,
)


# =============================================================================
# get_zoom_lims — axis limits for zoomed plots
# =============================================================================

class TestGetZoomLims:

    def test_point_aim_east(self):
        """Point aim, East axis: window spans ±ZOOM_SPAN/2 around aim_params[0]."""
        aim = np.array([15.0, 20.0, 60.0])
        lo, hi = get_zoom_lims(AimType.Point, aim, Direction.East)
        assert lo == pytest.approx(15.0 - ZOOM_SPAN / 2)
        assert hi == pytest.approx(15.0 + ZOOM_SPAN / 2)

    def test_point_aim_north(self):
        aim = np.array([15.0, 20.0, 60.0])
        lo, hi = get_zoom_lims(AimType.Point, aim, Direction.North)
        assert lo == pytest.approx(20.0 - ZOOM_SPAN / 2)
        assert hi == pytest.approx(20.0 + ZOOM_SPAN / 2)

    def test_point_aim_up_uses_min_minus_ten(self):
        """Up axis: lower bound is hard-coded -10 (ground clearance)."""
        aim = np.array([15.0, 20.0, 60.0])
        lo, hi = get_zoom_lims(AimType.Point, aim, Direction.Up)
        assert lo == -10
        assert hi == pytest.approx(60.0 + ZOOM_SPAN / 2)

    def test_ring_aim_east_small_radius_uses_half_span(self):
        """Ring with offset < half-span: bounds are ±half-span, not ±(offset+50)."""
        aim = np.array([10.0, 60.0, 0.0])  # r=10, ht=60
        lo, hi = get_zoom_lims(AimType.Ring, aim, Direction.East)
        assert lo == pytest.approx(-ZOOM_SPAN / 2)
        assert hi == pytest.approx(ZOOM_SPAN / 2)

    def test_ring_aim_east_large_radius_uses_offset_plus_pad(self):
        """Ring with offset > half-span: bounds are ±(offset + 50)."""
        aim = np.array([200.0, 60.0, 0.0])  # r=200 > 50
        lo, hi = get_zoom_lims(AimType.Ring, aim, Direction.East)
        assert lo == pytest.approx(-200.0 - 50)
        assert hi == pytest.approx(200.0 + 50)

    def test_ring_aim_up(self):
        aim = np.array([20.0, 75.0, 0.0])  # ht=75
        lo, hi = get_zoom_lims(AimType.Ring, aim, Direction.Up)
        assert lo == -10
        assert hi == pytest.approx(75.0 + ZOOM_SPAN / 2)


# =============================================================================
# _get_axis_plotting_data — slices the per-axis row out of (3, N) arrays
# =============================================================================

class TestGetAxisPlottingData:

    def setup_method(self):
        # Voxel locations: (3, 4) — east, north, up rows
        self.vox = np.array([
            [1.0, 2.0, 3.0, 4.0],   # E
            [5.0, 6.0, 7.0, 8.0],   # N
            [9.0, 10.0, 11.0, 12.0]  # U
        ])
        self.hel = np.array([
            [-1.0, -2.0],
            [-3.0, -4.0],
            [-5.0, -6.0],
        ])

    def test_east_returns_x_rows(self):
        vox_row, hel_row, label, slug = _get_axis_plotting_data(
            Direction.East, self.hel, self.vox)
        assert np.array_equal(vox_row, self.vox[0])
        assert np.array_equal(hel_row, self.hel[0])
        assert label == "East"
        assert slug == "E"

    def test_north_returns_y_rows(self):
        vox_row, hel_row, label, slug = _get_axis_plotting_data(
            Direction.North, self.hel, self.vox)
        assert np.array_equal(vox_row, self.vox[1])
        assert label == "North"
        assert slug == "N"

    def test_up_returns_z_rows(self):
        vox_row, hel_row, label, slug = _get_axis_plotting_data(
            Direction.Up, self.hel, self.vox)
        assert np.array_equal(vox_row, self.vox[2])
        assert label == "Up"
        assert slug == "U"


# =============================================================================
# _get_gif_depth_axis_data — picks the orthogonal axis for the GIF depth dim.
# =============================================================================

class TestGifDepthAxisData:

    def test_en_view_depth_is_altitude(self):
        vxs, vys, vzs = np.array([1.0]), np.array([2.0]), np.array([3.0])
        depth, label = _get_gif_depth_axis_data(Direction.East, Direction.North, vxs, vys, vzs)
        assert depth[0] == 3.0
        assert label == "Altitude"

    def test_eu_view_depth_is_north(self):
        vxs, vys, vzs = np.array([1.0]), np.array([2.0]), np.array([3.0])
        depth, label = _get_gif_depth_axis_data(Direction.East, Direction.Up, vxs, vys, vzs)
        assert depth[0] == 2.0
        assert label == "North"

    def test_nu_view_depth_is_east(self):
        vxs, vys, vzs = np.array([1.0]), np.array([2.0]), np.array([3.0])
        depth, label = _get_gif_depth_axis_data(Direction.North, Direction.Up, vxs, vys, vzs)
        assert depth[0] == 1.0
        assert label == "East"

    def test_invalid_combination_raises(self):
        with pytest.raises(ValueError, match="Invalid dimension combination"):
            _get_gif_depth_axis_data(Direction.Up, Direction.East,
                                     np.array([0.0]), np.array([0.0]), np.array([0.0]))


# =============================================================================
# _calculate_plot_parameters — marker size derivation from extent + pixel width.
# =============================================================================

class TestCalculatePlotParameters:

    def test_returns_four_positive_sizes(self):
        xs = np.array([-100.0, 100.0])
        ys = np.array([-100.0, 100.0])
        hel, vox, hel_z, vox_z = _calculate_plot_parameters(xs, ys, hel_size=2.0, vox_size=2)
        assert hel > 0
        assert vox >= 1  # vox_mkr_size has a floor of 1
        assert hel_z > 0
        assert vox_z >= 1

    def test_zero_width_falls_back_to_unit(self):
        """When all xlocs are equal, plot_w_m becomes 0 -> falls back to 1.0
        (sanity check that we don't divide by zero)."""
        xs = np.array([5.0, 5.0])
        ys = np.array([5.0, 5.0])
        hel, vox, hel_z, vox_z = _calculate_plot_parameters(xs, ys, hel_size=2.0, vox_size=2)
        assert np.isfinite(hel)
        assert np.isfinite(vox)
        assert hel > 0  # 2.0 * (PLOT_W_PX_DEFAULT / 1.0)

    def test_larger_extent_yields_smaller_marker(self):
        """A wider plot spreads voxels apart -> markers should shrink in px."""
        small_extent = np.array([-50.0, 50.0])
        large_extent = np.array([-500.0, 500.0])
        ys = np.array([0.0, 0.0])
        _, vox_small, _, _ = _calculate_plot_parameters(small_extent, ys, 2.0, 2)
        _, vox_large, _, _ = _calculate_plot_parameters(large_extent, ys, 2.0, 2)
        assert vox_small > vox_large

    def test_zoom_sizes_independent_of_extent(self):
        """Zoom marker sizes use ZOOM_SPAN, not the full extent — so they're
        identical for small vs. large input extents."""
        ys = np.array([0.0, 0.0])
        _, _, hel_z_small, vox_z_small = _calculate_plot_parameters(
            np.array([-50.0, 50.0]), ys, 2.0, 2)
        _, _, hel_z_large, vox_z_large = _calculate_plot_parameters(
            np.array([-500.0, 500.0]), ys, 2.0, 2)
        assert hel_z_small == pytest.approx(hel_z_large)
        assert vox_z_small == pytest.approx(vox_z_large)


# =============================================================================
# convert_fig_to_array — round-trips a plotly figure to RGB bytes. Exercises
# the kaleido + PIL path. Skipped if kaleido isn't installed properly.
# =============================================================================

def test_convert_fig_to_array_returns_rgb_array():
    import plotly.graph_objects as go
    from pyffiam.plots import convert_fig_to_array
    fig = go.Figure(data=go.Scatter(x=[0, 1], y=[0, 1]))
    try:
        arr = convert_fig_to_array(fig)
    except Exception as e:
        pytest.skip(f"kaleido/PIL backend unavailable: {e}")
    assert arr.ndim == 3
    assert arr.shape[2] in (3, 4)  # RGB or RGBA
    assert arr.dtype == np.uint8
