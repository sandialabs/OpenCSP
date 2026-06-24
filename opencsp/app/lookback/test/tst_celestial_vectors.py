# test_time_history_transitions_npz_mp.py
#
# Pytest suite for: opencsp.app.lookback.time_history_transitions_npz_mp
# Imported as requested:
#   import opencsp.app.lookback.time_history_transitions_npz_mp as astro_vec
#
# This suite intentionally loads the real Skyfield ephemeris file ("de430t.bsp")
# using Skyfield directly (not via astro_vec) to ensure it is available in the
# execution environment, then runs function-level tests that use the module.

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from skyfield.api import load as sf_load

import opencsp.app.lookback.celestial_vectors as astro_vec

# -----------------------------------------------------------------------------
# Ephemeris availability + shared fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ephemeris():
    """
    Critical environment test: ensure the DE430 ephemeris file can be loaded.

    This uses Skyfield directly (sf_load) rather than astro_vec.skf.load.
    """
    eph = sf_load("de430t.bsp")
    assert "earth" in eph
    assert "sun" in eph
    assert "moon" in eph
    return eph


@pytest.fixture(scope="session")
def sample_locations():
    target_location = (35.0542, -106.6170, 1600.0)  # (lat deg, lon deg, elev m)
    observer_location = (35.0844, -106.6504, 1615.0)
    return target_location, observer_location


# -----------------------------------------------------------------------------
# Pure algorithm tests (don’t require Skyfield)
# -----------------------------------------------------------------------------


def test_maximum_frame_range_alternating():
    frame_ranges = [(1, 10), (1, 20), (0, 50), (1, 60), (0, 90)]
    max_pair, max_dur = astro_vec.maximum_frame_range(frame_ranges)
    assert max_dur == 30
    assert max_pair == [(1, 60), (0, 90)]


def test_maximum_frame_range_no_alternation_returns_none():
    frame_ranges = [(1, 10), (1, 40), (1, 70)]
    max_pair, max_dur = astro_vec.maximum_frame_range(frame_ranges)
    assert max_dur == 0
    assert max_pair is None


def test_trilaterate_two_intersections_symmetric_case():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    radii = np.array([1.0, 1.0, 1.0])
    sols = astro_vec.trilaterate(positions, radii, raise_on_no_solution=True)

    assert sols.shape == (2, 3)
    assert np.allclose(sols[:, 0], 0.5, atol=1e-7)
    assert np.allclose(sols[:, 1], 0.5, atol=1e-7)
    assert np.isclose(sols[0, 2], -sols[1, 2], atol=1e-7)
    assert not np.isclose(sols[0, 2], 0.0)


def test_trilaterate_negative_radicand_returns_close_enough_when_not_raising():
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    radii = np.array([1.0, 1.0, 1.0])  # no intersection
    sol = astro_vec.trilaterate(positions, radii, raise_on_no_solution=False)
    assert sol.shape == (1, 3)


def test_calculate_slope_normalizes():
    observer_vec_start = np.array([1.0, 0.0, 0.0])
    observer_vec_end = np.array([1.0, 0.0, 0.0])
    inter_1 = np.array([0.0, 1.0, 0.0])
    inter_2 = np.array([0.0, 0.0, 1.0])

    observer_vec, slope_1, slope_2 = astro_vec.calculate_slope(observer_vec_start, observer_vec_end, inter_1, inter_2)
    assert np.isclose(np.linalg.norm(slope_1), 1.0)
    assert np.isclose(np.linalg.norm(slope_2), 1.0)
    assert np.allclose(observer_vec, np.array([1.0, 0.0, 0.0]))


# -----------------------------------------------------------------------------
# Skyfield-backed tests (require ephemeris fixture)
# -----------------------------------------------------------------------------


def test_ephemeris_loads(ephemeris):
    assert ephemeris is not None


@pytest.mark.parametrize("celestial_object_name", ["moon", "sun"])
def test_calculate_vectors_celestial_observer_outputs_unit_vectors(
    ephemeris, sample_locations, celestial_object_name  # ensures ephemeris load succeeds in environment
):
    target_location, observer_location = sample_locations

    out = astro_vec.calculate_vectors_celestial_observer(
        celestial_object_name=celestial_object_name,
        target_location=target_location,
        observer_location=observer_location,
        observation_time=(2026, 1, 1, 0, 0, 0),
    )

    for k in [
        "celestial_to_target",
        "cel_to_target_cartesian",
        "earth_to_target_cartesian",
        "target_to_observer",
        "angular_size_radians",
    ]:
        assert k in out

    assert np.isfinite(out["angular_size_radians"])
    assert out["angular_size_radians"] > 0.0

    assert np.isclose(np.linalg.norm(out["celestial_to_target"]), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(out["cel_to_target_cartesian"]), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(out["earth_to_target_cartesian"]), 1.0, atol=1e-12)
    assert np.isclose(np.linalg.norm(out["target_to_observer"]), 1.0, atol=1e-12)


def test_calculate_vectors_accepts_datetime(ephemeris, sample_locations):
    target_location, observer_location = sample_locations
    out = astro_vec.calculate_vectors_celestial_observer(
        celestial_object_name="moon",
        target_location=target_location,
        observer_location=observer_location,
        observation_time=datetime(2026, 1, 1, 0, 0, 0),
    )
    assert out["angular_size_radians"] > 0.0


def test_angular_size_plausible_bounds(ephemeris, sample_locations):
    """
    Weak-but-meaningful regression test: angular diameters should be in plausible
    ranges for typical Earth observers.
    """
    target_location, observer_location = sample_locations
    t = (2026, 1, 1, 0, 0, 0)

    moon_ang = astro_vec.calculate_vectors_celestial_observer("moon", target_location, observer_location, t)[
        "angular_size_radians"
    ]
    sun_ang = astro_vec.calculate_vectors_celestial_observer("sun", target_location, observer_location, t)[
        "angular_size_radians"
    ]

    # Plausible bounds in radians:
    assert 0.0075 < moon_ang < 0.0115
    assert 0.0080 < sun_ang < 0.0110


# -----------------------------------------------------------------------------
# process_pixel() tests
# -----------------------------------------------------------------------------


def test_process_pixel_no_transitions_returns_none_pixel_data():
    pixel = "(10, 20)"
    transitions = []
    video_metadata = {"frame_rate": 30.0, "create_date": "2026:01:01 00:00:00"}
    video_start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("UTC"))

    args = (
        pixel,
        transitions,
        video_metadata,
        video_start_time,
        0.0,
        "moon",
        (35.0, -106.0, 1600.0),
        (35.1, -106.1, 1600.0),
        ZoneInfo("UTC"),
    )

    p, pixel_data, chk = astro_vec.process_pixel(args)
    assert p == pixel
    assert pixel_data is None
    assert pixel in chk["processed_pixels"]
    assert pixel in chk["pixels_without_data"]


def test_process_pixel_two_transitions_happy_path(monkeypatch, ephemeris):
    """
    Uses real ephemeris (via astro_vec internals), but patches only lookback-tools
    helpers that parse frame names and produce observation times.
    """

    def fake_frame_number_from_img_name(name: str) -> int:
        digits = "".join([c for c in name if c.isdigit()])
        return int(digits) if digits else 0

    def fake_define_observation_time_skyfield(dt, _shift):
        # calculate_vectors_celestial_observer accepts datetime; keep it timezone-aware UTC.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo("UTC"))

    monkeypatch.setattr(astro_vec.lbt, "frame_number_from_img_name", fake_frame_number_from_img_name, raising=True)
    monkeypatch.setattr(
        astro_vec.lbt, "define_observation_time_skyfield", fake_define_observation_time_skyfield, raising=True
    )

    pixel = "(10, 20)"
    transitions = [
        {"transition": "bright", "to_frame": "frame_000010.png"},
        {"transition": "dark", "to_frame": "frame_000070.png"},
    ]
    video_metadata = {"frame_rate": 30.0, "create_date": "2026:01:01 00:00:00"}
    video_start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=ZoneInfo("UTC"))

    args = (
        pixel,
        transitions,
        video_metadata,
        video_start_time,
        0.0,
        "moon",
        (35.0, -106.0, 1600.0),
        (35.1, -106.1, 1600.0),
        ZoneInfo("UTC"),
    )

    p, pixel_data, chk = astro_vec.process_pixel(args)
    assert p == pixel
    assert pixel_data is not None
    assert pixel in chk["processed_pixels"]
    assert pixel in chk["pixels_with_data"]

    assert "start_vector" in pixel_data and "end_vector" in pixel_data
    assert "observer_vector" in pixel_data

    if pixel_data["slope_1"] is not None:
        assert np.isclose(np.linalg.norm(pixel_data["slope_1"]), 1.0, atol=1e-12)
    if pixel_data["slope_2"] is not None:
        assert np.isclose(np.linalg.norm(pixel_data["slope_2"]), 1.0, atol=1e-12)
