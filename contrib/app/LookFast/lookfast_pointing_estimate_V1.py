'''
Utilize the time intervals from the main analysis to determine where the beamlet from each mixel (assumed to be an elipse)
crosses the camera system.

A perfectly focused and made optic would have each and every beamlet produce the maximum crossing time.
'''

import os
import math
from typing import Dict, Any, Tuple, Optional, List
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import DEBUG, ERROR
from datetime import datetime, timedelta, timezone

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


import opencsp.common.lib.tool.file_tools as ft

# import opencsp.app.lookback.lookback_tools as lbt
import contrib.app.LookFast.lookback_tools as lbt


@dataclass(frozen=True)
class EllipticalSunSpot:
    """
    Rotated ellipse on a 2D wall.

    a_m, b_m: semi-axes (m)
    orientation_deg: rotation of +a axis CCW from +x (deg)
    center_xy: (cx, cy) in wall coordinates (m)
    """

    distance_m: float
    angular_diameter_rad: float
    diameter_small_angle: float
    a_m: float
    b_m: float
    orientation_deg: float = 0.0
    center_xy: Tuple[float, float] = (0.0, 0.0)


def reflected_sun_spot_ellipse(
    distance_m: float,
    *,
    angular_diameter_rad: float = 9.3e-3,
    axis_ratio: float = 1.0,  # b/a
    orientation_deg: float = 0.0,
    center_xy: Tuple[float, float] = (0.0, 0.0),
) -> EllipticalSunSpot:
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    if angular_diameter_rad <= 0:
        raise ValueError("angular_diameter_rad must be > 0")
    if axis_ratio <= 0:
        raise ValueError("axis_ratio must be > 0")

    diameter = distance_m * angular_diameter_rad  # small-angle approximation
    a = 0.5 * diameter
    b = a * axis_ratio

    return EllipticalSunSpot(
        distance_m=distance_m,
        angular_diameter_rad=angular_diameter_rad,
        a_m=a,
        b_m=b,
        orientation_deg=orientation_deg,
        center_xy=center_xy,
        diameter_small_angle=diameter,
    )


def ellipse_boundary_points(
    a_m: float, b_m: float, *, orientation_deg: float = 0.0, center_xy: Tuple[float, float] = (0.0, 0.0), n: int = 400
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (x, y) points for a rotated ellipse boundary.

    Ellipse local param:
      x' = a cos t
      y' = b sin t
    Then rotate by orientation_deg CCW and translate by center_xy.
    """
    if a_m <= 0 or b_m <= 0:
        raise ValueError("a_m and b_m must be > 0")
    if n < 10:
        raise ValueError("n must be >= 10")

    t = np.linspace(0.0, 2.0 * math.pi, n, endpoint=True)
    xp = a_m * np.cos(t)
    yp = b_m * np.sin(t)

    th = math.radians(orientation_deg)
    c, s = math.cos(th), math.sin(th)

    x = c * xp - s * yp + center_xy[0]
    y = s * xp + c * yp + center_xy[1]
    return x, y


def _to_local_rotation(orientation_deg: float) -> Tuple[float, float]:
    """Return cos(theta), sin(theta) for theta=orientation."""
    th = math.radians(orientation_deg)
    return math.cos(th), math.sin(th)


def _global_to_local_vec(x: float, y: float, orientation_deg: float) -> Tuple[float, float]:
    """
    Rotate a vector from global coords into ellipse-local coords:
      v' = R(-theta) v
    """
    c, s = _to_local_rotation(orientation_deg)
    xp = c * x + s * y
    yp = -s * x + c * y
    return xp, yp


def _global_to_local_point(
    x: float, y: float, *, center_xy: Tuple[float, float], orientation_deg: float
) -> Tuple[float, float]:
    """
    Convert a global point to ellipse-local coordinates:
      p' = R(-theta) (p - center)
    """
    dx = x - center_xy[0]
    dy = y - center_xy[1]
    return _global_to_local_vec(dx, dy, orientation_deg)


# -----------------------------
# Intersections / chord length
# -----------------------------
def ellipse_line_intersections(
    a_m: float,
    b_m: float,
    *,
    center_xy: Tuple[float, float] = (0.0, 0.0),
    orientation_deg: float = 0.0,
    direction_deg: float,
    eps: float = 1e-12,
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Intersections between the ellipse and the infinite line that passes through the origin with
    direction `direction_deg`.

    Line: p(t) = t * u, where u = (cosφ, sinφ), φ in degrees.

    Returns
    -------
    - ((x1, y1), (x2, y2)) for two distinct intersection points
    - None if tangent (single intersection) or no intersection

    """
    if a_m <= 0 or b_m <= 0:
        raise ValueError("a_m and b_m must be > 0")

    phi = math.radians(direction_deg)
    ux, uy = math.cos(phi), math.sin(phi)

    # Convert the line into ellipse-local coordinates:
    # local ellipse: (x'/a)^2 + (y'/b)^2 = 1
    # local line: p'(t) = p0' + t*u', where p0' is origin expressed in local coords.
    p0x, p0y = _global_to_local_point(0.0, 0.0, center_xy=center_xy, orientation_deg=orientation_deg)
    uxp, uyp = _global_to_local_vec(ux, uy, orientation_deg)

    # Solve quadratic:
    # ((p0x + t*uxp)/a)^2 + ((p0y + t*uyp)/b)^2 = 1
    A = (uxp * uxp) / (a_m * a_m) + (uyp * uyp) / (b_m * b_m)
    B = 2.0 * (p0x * uxp) / (a_m * a_m) + 2.0 * (p0y * uyp) / (b_m * b_m)
    C = (p0x * p0x) / (a_m * a_m) + (p0y * p0y) / (b_m * b_m) - 1.0

    D = B * B - 4.0 * A * C

    if D < -eps:
        return None  # no intersection
    if abs(D) <= eps:
        return None  # tangent: single intersection requested to return None

    sqrtD = math.sqrt(D)
    t1 = (-B - sqrtD) / (2.0 * A)
    t2 = (-B + sqrtD) / (2.0 * A)

    p1 = (t1 * ux, t1 * uy)
    p2 = (t2 * ux, t2 * uy)
    return p1, p2


def ellipse_chord_length_along_direction(
    a_m: float,
    b_m: float,
    *,
    center_xy: Tuple[float, float] = (0.0, 0.0),
    orientation_deg: float = 0.0,
    direction_deg: float,
    eps: float = 1e-12,
) -> Optional[float]:
    """
    Chord length created by intersecting the ellipse with the infinite line through the origin
    at angle `direction_deg`.

    Returns
    -------
    - float chord length if there are two distinct intersection points
    - None if tangent or no intersection
    """
    pts = ellipse_line_intersections(
        a_m, b_m, center_xy=center_xy, orientation_deg=orientation_deg, direction_deg=direction_deg, eps=eps
    )
    if pts is None:
        return None

    (x1, y1), (x2, y2) = pts
    return math.hypot(x2 - x1, y2 - y1)


def time_for_ellipse_to_pass_chord(chord_length_m: float, *, distance_m: float, angular_speed_rad_s: float) -> float:
    """
    Convert the trajectory chord length into a time using small-angle mapping:
        v ≈ distance_m * angular_speed_rad_s
        T = chord / v
    """
    if chord_length_m < 0:
        raise ValueError("chord_length_m must be >= 0")
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    if angular_speed_rad_s <= 0:
        raise ValueError("angular_speed_rad_s must be > 0")

    v = distance_m * angular_speed_rad_s
    return chord_length_m / v


def chord_length_from_transit_time(transit_time_s: float, *, distance_m: float, angular_speed_rad_s: float) -> float:
    """
    Inverse of time_for_ellipse_to_pass_chord().

    Using the same small-angle mapping:
        v ≈ distance_m * angular_speed_rad_s
        T = chord / v
    so:
        chord = T * v = T * distance_m * angular_speed_rad_s
    """
    if transit_time_s < 0:
        raise ValueError("transit_time_s must be >= 0")
    if distance_m <= 0:
        raise ValueError("distance_m must be > 0")
    if angular_speed_rad_s <= 0:
        raise ValueError("angular_speed_rad_s must be > 0")

    v = distance_m * angular_speed_rad_s
    return transit_time_s * v


def fit_y_shifts_for_chord_length(
    spot: EllipticalSunSpot,
    target_chord_length_m: float,
    *,
    direction_deg: float = 0.0,
    y_bounds: Optional[Tuple[float, float]] = None,
    n_grid: int = 4001,
    refine_factor: int = 25,
    refine_half_window_pts: int = 20,
    show: bool = False,
) -> Dict[str, Any]:
    """
    Find y-shifts (changing only ellipse center y) that make the chord length along `direction_deg`
    match `target_chord_length_m`.

    Algorithm
    ---------
    1) Coarse scan: chord(y) over a grid in y.
    2) Take the two grid points with smallest |chord(y) - target|.
    3) For each candidate, do a local refinement scan in a smaller y-window around that y.
    4) Deduplicate nearly-identical solutions.
    5) Return exactly two shifts: (best, second_or_None), ordered by increasing |shift|.

    Notes
    -----
    - chord(y) is treated as invalid if ellipse_line_intersections yields 0 or 1 intersection
      (i.e., ellipse_chord_length_along_direction returns None). Those ys are ignored.
    - If *no* valid chord exists in the whole scan range, returns (None, None).

    Returns
    -------
    dict with:
      shifts: (s1, s2_or_None)
      y_solutions: (y1, y2_or_None)
      chord_at_solutions: (L1, L2_or_None)
      errors: (e1, e2_or_None)  where e=abs(L-target)
      n_solutions_found: int
      scan_coarse: (ys, chords, diffs)
      scan_refined: list of per-candidate scans [(ys_ref, chords_ref, diffs_ref), ...]
      bounds: (ymin, ymax)
    """

    def chord_at(y: Optional[float]) -> Optional[float]:
        if y is None:
            return None
        L = ellipse_chord_length_along_direction(
            spot.a_m,
            spot.b_m,
            center_xy=(cx, float(y)),
            orientation_deg=spot.orientation_deg,
            direction_deg=direction_deg,
        )
        return None if L is None else float(L)

    if target_chord_length_m < 0:
        raise ValueError("target_chord_length_m must be >= 0")
    if n_grid < 200:
        raise ValueError("n_grid should be reasonably large (>=200)")
    if refine_factor < 2:
        raise ValueError("refine_factor must be >= 2")
    if refine_half_window_pts < 2:
        raise ValueError("refine_half_window_pts must be >= 2")

    cx, cy0 = spot.center_xy

    if y_bounds is None:
        R = 3.0 * max(spot.a_m, spot.b_m)
        ymin, ymax = cy0 - R, cy0 + R
    else:
        ymin, ymax = y_bounds
        if ymin >= ymax:
            raise ValueError("y_bounds must be (ymin, ymax) with ymin < ymax")

    # ---- coarse scan ----
    ys = np.linspace(ymin, ymax, n_grid)
    chords = np.full_like(ys, np.nan, dtype=float)

    for i, y in enumerate(ys):
        L = chord_at(float(y))
        chords[i] = np.nan if L is None else float(L)

    valid = np.isfinite(chords)
    diffs = np.full_like(chords, np.nan, dtype=float)
    diffs[valid] = np.abs(chords[valid] - target_chord_length_m)

    if not np.any(valid):
        return {
            "shifts": (None, None),
            "y_solutions": (None, None),
            "chord_at_solutions": (None, None),
            "errors": (None, None),
            "n_solutions_found": 0,
            "scan_coarse": (ys, chords, diffs),
            "scan_refined": [],
            "bounds": (ymin, ymax),
        }

    # Find up to two best coarse indices (unique)
    order = np.argsort(diffs[valid])
    idx_valid = np.where(valid)[0]
    best_idxs = []
    for j in order:
        ii = int(idx_valid[j])
        best_idxs.append(ii)
        if len(best_idxs) == 2:
            break

    # ---- local refinement around each coarse candidate ----
    dy = float(ys[1] - ys[0])
    refined_scans = []
    refined_candidates: List[Tuple[float, float, float]] = []  # (y, chord, diff)

    for ii in best_idxs:
        y0 = float(ys[ii])

        # window around y0, but keep within global bounds
        half_window = refine_half_window_pts * dy
        a = max(ymin, y0 - half_window)
        b = min(ymax, y0 + half_window)

        # refined step size
        n_ref = max(200, int(refine_factor * (refine_half_window_pts * 2 + 1)))
        ys_ref = np.linspace(a, b, n_ref)

        chords_ref = np.full_like(ys_ref, np.nan, dtype=float)
        diffs_ref = np.full_like(ys_ref, np.nan, dtype=float)

        for k, y in enumerate(ys_ref):
            L = chord_at(float(y))
            if L is None:
                continue
            chords_ref[k] = L
            diffs_ref[k] = abs(L - target_chord_length_m)

        refined_scans.append((ys_ref, chords_ref, diffs_ref))

        vref = np.isfinite(diffs_ref)
        if np.any(vref):
            kbest = int(np.nanargmin(diffs_ref))
            refined_candidates.append((float(ys_ref[kbest]), float(chords_ref[kbest]), float(diffs_ref[kbest])))

    # If refinement produced nothing (e.g., windows had no valid chords), fall back to coarse best points
    if not refined_candidates:
        for ii in best_idxs:
            refined_candidates.append((float(ys[ii]), float(chords[ii]), float(diffs[ii])))

    y1 = refined_candidates[0][0] if len(refined_candidates) >= 1 else None
    y2 = refined_candidates[1][0] if len(refined_candidates) >= 2 else None

    s1 = (y1 - cy0) if y1 is not None else None
    s2 = (y2 - cy0) if y2 is not None else None

    L1 = refined_candidates[0][1] if len(refined_candidates) >= 1 else None
    L2 = refined_candidates[1][1] if len(refined_candidates) >= 2 else None

    e1 = refined_candidates[0][2] if len(refined_candidates) >= 1 else None
    e2 = refined_candidates[1][2] if len(refined_candidates) >= 2 else None

    if show:
        spot_adj1 = reflected_sun_spot_ellipse(
            distance_m=spot.distance_m,
            angular_diameter_rad=spot.angular_diameter_rad,
            axis_ratio=spot.b_m / spot.a_m,
            orientation_deg=spot.orientation_deg,
            center_xy=(spot.center_xy[0], spot.center_xy[1] + s1),
        )
        spot_adj2 = reflected_sun_spot_ellipse(
            distance_m=spot.distance_m,
            angular_diameter_rad=spot.angular_diameter_rad,
            axis_ratio=spot.b_m / spot.a_m,
            orientation_deg=spot.orientation_deg,
            center_xy=(spot.center_xy[0], spot.center_xy[1] + s2),
        )
        fig, ax, details = plot_ellipse_and_center_chord(spot_obj=spot, direction_deg=direction_deg, show=False)
        fig, ax, details = plot_ellipse_and_center_chord(
            spot_obj=spot_adj1, ax=ax, direction_deg=direction_deg, show=False
        )
        fig, ax, details = plot_ellipse_and_center_chord(
            spot_obj=spot_adj2, ax=ax, direction_deg=direction_deg, show=True
        )

    return {
        "shifts": (s1, s2),
        "y_solutions": (y1, y2),
        "chord_at_solutions": (L1, L2),
        "chord_length_diffs": (e1, e2),
        "n_solutions_found": int(min(2, len(refined_candidates))),
        "scan_coarse": (ys, chords, diffs),
        "scan_refined": refined_scans,
        "bounds": (ymin, ymax),
    }


def plot_ellipse_and_center_chord(
    a_m: Optional[float] = None,
    b_m: Optional[float] = None,
    spot_obj: Optional[EllipticalSunSpot] = None,
    *,
    center_xy: Tuple[float, float] = (0.0, 0.0),
    orientation_deg: float = 0.0,
    direction_deg: float = 0.0,
    ax: Optional[plt.Axes] = None,
    n: int = 600,
    ray_span_factor: float = 1.25,
    show: bool = True,
    output_dir: str = None,
    figure_name: str = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Plot:
      - ellipse boundary
      - the line (vector) through the origin at direction_deg
      - intersection points (if they exist)
      - chord segment (if two intersections exist)

    Returns meta containing intersection points and chord length (or None).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
        adding_shapes = False
    else:
        fig = ax.figure
        adding_shapes = True

    if spot_obj:
        a_m = spot_obj.a_m
        b_m = spot_obj.b_m
        center_xy = spot_obj.center_xy
        orientation_deg = spot_obj.orientation_deg
    elif None in [a_m, b_m, center_xy, orientation_deg]:
        raise ValueError("Either EllipticalSunSpot object needs to be provided or individual parameters")

    # ellipse boundary
    ex, ey = ellipse_boundary_points(a_m, b_m, orientation_deg=orientation_deg, center_xy=center_xy, n=n)
    # ax.plot(ex, ey, lw=2.0, color="C0", label="Ellipse boundary")
    ax.plot(ex, ey, lw=2.0, label="Ellipse boundary")

    # origin + ellipse center markers
    ax.plot([0.0], [0.0], "ko", ms=6, label="Origin")
    ax.plot([center_xy[0]], [center_xy[1]], "C3x", ms=8, mew=2, label="Ellipse center")

    # direction vector line (draw a long segment both directions)
    phi = math.radians(direction_deg)
    ux, uy = math.cos(phi), math.sin(phi)

    lim = ray_span_factor * (max(a_m, b_m) + math.hypot(center_xy[0], center_xy[1]) + 1e-9)
    x_line = np.array([-lim, lim]) * ux
    y_line = np.array([-lim, lim]) * uy
    ax.plot(x_line, y_line, "C1--", lw=1.8, label=f"Direction line ({direction_deg:g}°)")

    # intersections + chord
    pts = ellipse_line_intersections(
        a_m, b_m, center_xy=center_xy, orientation_deg=orientation_deg, direction_deg=direction_deg
    )
    chord_len = None
    if pts is not None:
        (x1, y1), (x2, y2) = pts
        ax.plot([x1, x2], [y1, y2], "C2-", lw=4.0, alpha=0.5, label="Chord")
        # ax.plot([x1, x2], [y1, y2], lw=4.0, alpha=0.5, label="Chord")
        ax.plot([x1, x2], [y1, y2], "C2o", ms=7, alpha=0.5, label="Intersections")
        # ax.plot([x1, x2], [y1, y2], ms=7, alpha=0.2, label="Intersections")
        chord_len = math.hypot(x2 - x1, y2 - y1)
    else:
        # Still label that there are no intersections/tangent
        ax.text(0.02, 0.98, "No chord: 0 or 1 intersection", transform=ax.transAxes, va="top", ha="left")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    if adding_shapes:
        ax.set_title("Ellipse / direction intersections\n" f"a={a_m:.4g} m, b={b_m:.4g} m, dir={direction_deg:g}°")
    else:
        ax.set_title(
            "Ellipse / direction intersections\n"
            f"a={a_m:.4g} m, b={b_m:.4g} m, orient={orientation_deg:g}°, center={center_xy}, dir={direction_deg:g}°"
        )
    ax.grid(True, alpha=0.3)
    if ax.get_legend():
        pass
    else:
        ax.legend(loc="upper right")

    # set view to include ellipse nicely
    # (rough bound: ellipse extent plus center offset)
    view = 1.2 * (max(a_m, b_m) + math.hypot(center_xy[0], center_xy[1]) + 1e-9)
    ax.set_xlim(-view, view)
    ax.set_ylim(-view, view)

    if show:
        plt.show()

    if output_dir:
        if not ft.directory_exists(output_dir):
            ft.create_directories_if_necessary(output_dir)
        file_path = ft.join(output_dir, figure_name)
        plt.savefig(file_path)

    return fig, ax, {"intersections": pts, "chord_length_m": chord_len}


def _parse_pixel_key(k: str) -> Tuple[int, int]:
    """
    Parse a key like "(row, col)" into (row, col) ints.
    Handles whitespace variations.
    """
    s = k.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    r_str, c_str = s.split(",")
    return int(r_str.strip()), int(c_str.strip())


def ensure_tz(dt: datetime, tz: ZoneInfo) -> datetime:
    """
    Return tz-aware datetime in timezone tz.
    - If dt is naive: interpret it as wall time in tz.
    - If dt is aware: convert to tz.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def dt_to_mpl_num_local(dt: datetime, tz: ZoneInfo) -> float:
    """
    Convert datetime to Matplotlib date number after normalizing to tz.
    """
    if dt is None:
        return float("nan")
    return mdates.date2num(ensure_tz(dt, tz))


def _to_epoch_seconds(dt: datetime, tz: ZoneInfo) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)  # interpret naive as local
    else:
        dt = dt.astimezone(tz)
    return dt.timestamp()


def _cropped_bbox_from_keys(pixel_dict: Dict[str, Dict[str, Any]], pad: int) -> Tuple[int, int, int, int]:
    parsed_keys = [_parse_pixel_key(k) for k in pixel_dict.keys()]
    if not parsed_keys:
        raise ValueError("pixel_dict is empty.")
    rows = np.array([r for r, _ in parsed_keys], dtype=int)
    cols = np.array([c for _, c in parsed_keys], dtype=int)

    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())
    return (rmin - pad, rmax + pad, cmin - pad, cmax + pad)  # rmin_p, rmax_p, cmin_p, cmax_p


def _extent_from_bbox(rmin_p: int, rmax_p: int, cmin_p: int, cmax_p: int, origin: str):
    return (
        [cmin_p - 0.5, cmax_p + 0.5, rmax_p + 0.5, rmin_p - 0.5]
        if origin == "upper"
        else [cmin_p - 0.5, cmax_p + 0.5, rmin_p - 0.5, rmax_p + 0.5]
    )


def plot_light_maps_cropped(
    pixel_dict: Dict[str, Dict[str, Any]],
    *,
    tz: ZoneInfo = ZoneInfo("America/Denver"),
    pad: int = 0,  # extra pixels around the bounding box
    origin: str = "upper",
    cmap_duration: str = "viridis",
    cmap_time: str = "cool",
    nan_color: str = "white",
    duration_unit: str = "s",
    show: bool = True,
):
    """
    Like plot_light_maps(), but crops the plotted arrays to the bounding box
    of pixels present in pixel_dict (optionally padded).
    """
    # Parse keys once
    parsed = [(_parse_pixel_key(k), v) for k, v in pixel_dict.items()]
    if not parsed:
        raise ValueError("pixel_dict is empty.")

    rows = np.array([rc[0] for rc, _ in parsed], dtype=int)
    cols = np.array([rc[1] for rc, _ in parsed], dtype=int)

    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    # Apply padding
    rmin_p = rmin - pad
    cmin_p = cmin - pad
    rmax_p = rmax + pad
    cmax_p = cmax + pad

    # Local array shape for the cropped region
    nrows = (rmax_p - rmin_p) + 1
    ncols = (cmax_p - cmin_p) + 1

    dur = np.full((nrows, ncols), np.nan, dtype=float)
    t0 = np.full((nrows, ncols), np.nan, dtype=float)
    t1 = np.full((nrows, ncols), np.nan, dtype=float)

    # Fill into cropped coordinates
    for (r, c), info in parsed:
        rr = r - rmin_p
        cc = c - cmin_p
        dur[rr, cc] = info.get("light_duration", np.nan)
        t0[rr, cc] = dt_to_mpl_num_local(info.get("light_start_time", None), tz)
        t1[rr, cc] = dt_to_mpl_num_local(info.get("light_end_time", None), tz)

    # Colormaps with defined NaN color
    cmap_d = plt.colormaps[cmap_duration].copy()
    cmap_t = plt.colormaps[cmap_time].copy()
    cmap_d.set_bad(nan_color)
    cmap_t.set_bad(nan_color)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Use extent so axes show original pixel coordinates (not 0..cropped-1)
    # For imshow, extent is [xmin, xmax, ymin, ymax] in data coords.
    extent = (
        [cmin_p - 0.5, cmax_p + 0.5, rmax_p + 0.5, rmin_p - 0.5]
        if origin == "upper"
        else [cmin_p - 0.5, cmax_p + 0.5, rmin_p - 0.5, rmax_p + 0.5]
    )

    # Duration
    im0 = axes[0].imshow(dur, origin=origin, cmap=cmap_d, extent=extent)
    axes[0].set_title(f"Light duration ({duration_unit})")
    axes[0].set_xlabel("col")
    axes[0].set_ylabel("row")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label(f"Duration ({duration_unit})")

    # Start time
    im1 = axes[1].imshow(t0, origin=origin, cmap=cmap_t, extent=extent)
    axes[1].set_title("Light start time")
    axes[1].set_xlabel("col")
    axes[1].set_ylabel("row")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label("Datetime")
    cb1.formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S", tz=tz)
    cb1.update_ticks()

    # End time
    im2 = axes[2].imshow(t1, origin=origin, cmap=cmap_t, extent=extent)
    axes[2].set_title("Light end time")
    axes[2].set_xlabel("col")
    axes[2].set_ylabel("row")
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cb2.set_label("Datetime")
    cb2.formatter = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S", tz=tz)
    cb2.update_ticks()

    # Tighten view to the cropped region bounds (optional; extent already does this)
    for ax in axes:
        ax.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
        ax.set_ylim(rmax_p + 0.5, rmin_p - 0.5) if origin == "upper" else ax.set_ylim(rmin_p - 0.5, rmax_p + 0.5)

    if show:
        plt.show()

    bbox = {"rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax, "pad": pad}
    return fig, axes, {"duration": dur, "start_num": t0, "end_num": t1, "bbox": bbox}


def plot_scalar_map_cropped(
    pixel_dict: Dict[str, Dict[str, Any]],
    *,
    value_key: str,
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    pad: int = 0,
    origin: str = "upper",
    cmap: str = "viridis",
    data_range: tuple = (None, None),
    nan_color: str = "white",
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Generic cropped scalar (float) map plot.

    Parameters
    ----------
    value_key:
        Key inside each pixel's info dict to plot (e.g., "light_duration").
    title, cbar_label:
        Plot title / colorbar label. If None, defaults are derived from value_key.
    """
    rmin_p, rmax_p, cmin_p, cmax_p = _cropped_bbox_from_keys(pixel_dict, pad=pad)
    nrows = (rmax_p - rmin_p) + 1
    ncols = (cmax_p - cmin_p) + 1

    arr = np.full((nrows, ncols), np.nan, dtype=float)

    for k, info in pixel_dict.items():
        r, c = _parse_pixel_key(k)
        rr, cc = r - rmin_p, c - cmin_p
        v = info.get(value_key, np.nan)
        arr[rr, cc] = np.nan if v is None else float(v)

    cm = plt.colormaps[cmap].copy()
    cm.set_bad(nan_color)

    extent = _extent_from_bbox(rmin_p, rmax_p, cmin_p, cmax_p, origin)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    else:
        fig = ax.figure

    im = ax.imshow(
        arr,
        origin=origin,
        cmap=cm,
        extent=extent,
        vmin=data_range[0] if data_range[0] else None,
        vmax=data_range[1] if data_range[1] else None,
    )
    ax.set_title(title if title is not None else value_key)
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label if cbar_label is not None else value_key)

    # tighten bounds
    ax.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
    if origin == "upper":
        ax.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
    else:
        ax.set_ylim(rmin_p - 0.5, rmax_p + 0.5)

    if show:
        plt.show()

    meta = {
        "bbox": {"rmin": rmin_p + pad, "rmax": rmax_p - pad, "cmin": cmin_p + pad, "cmax": cmax_p - pad, "pad": pad}
    }
    return fig, ax, {"array": arr, "extent": extent, **meta}


def plot_time_map_cropped(
    pixel_dict: Dict[str, Dict[str, Any]],
    *,
    time_key: str,
    tz: ZoneInfo = ZoneInfo("America/Denver"),
    title: Optional[str] = None,
    cbar_label: str = "Datetime",
    pad: int = 0,
    origin: str = "upper",
    cmap: str = "cool",
    nan_color: str = "white",
    time_fmt: str = "%Y-%m-%d\n%H:%M:%S",
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Generic cropped datetime map plot.

    Converts datetimes to matplotlib date floats for imshow,
    and formats the colorbar as datetimes in the specified timezone.
    """
    rmin_p, rmax_p, cmin_p, cmax_p = _cropped_bbox_from_keys(pixel_dict, pad=pad)
    nrows = (rmax_p - rmin_p) + 1
    ncols = (cmax_p - cmin_p) + 1

    arr = np.full((nrows, ncols), np.nan, dtype=float)

    for k, info in pixel_dict.items():
        r, c = _parse_pixel_key(k)
        rr, cc = r - rmin_p, c - cmin_p
        arr[rr, cc] = dt_to_mpl_num_local(info.get(time_key, None), tz)

    cm = plt.colormaps[cmap].copy()
    cm.set_bad(nan_color)

    extent = _extent_from_bbox(rmin_p, rmax_p, cmin_p, cmax_p, origin)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    else:
        fig = ax.figure

    im = ax.imshow(arr, origin=origin, cmap=cm, extent=extent)
    ax.set_title(title if title is not None else time_key)
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label)
    cb.formatter = mdates.DateFormatter(time_fmt, tz=tz)
    cb.update_ticks()

    # tighten bounds
    ax.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
    if origin == "upper":
        ax.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
    else:
        ax.set_ylim(rmin_p - 0.5, rmax_p + 0.5)

    if show:
        plt.show()

    meta = {
        "bbox": {"rmin": rmin_p + pad, "rmax": rmax_p - pad, "cmin": cmin_p + pad, "cmax": cmax_p - pad, "pad": pad}
    }
    return fig, ax, {"array_mpl_dates": arr, "extent": extent, **meta}


def plot_midpoint_offset_heatmap_tz(
    pixel_dict: Dict[str, Dict[str, Any]],
    ideal_mid_time: datetime,
    *,
    tz: ZoneInfo = ZoneInfo("America/Denver"),
    pad: int = 0,
    origin: str = "upper",
    cmap: str = "coolwarm",
    nan_color: str = "white",
    units: str = "seconds",  # "seconds" | "minutes" | "hours"
    robust: bool = True,  # use percentiles for color limits (less outlier-sensitive)
    robust_pct: Tuple[float, float] = (2, 98),
    show: bool = True,
):
    """
    For each pixel:
      midpoint = light_start_time + (light_end_time - light_start_time)/2
      delta = midpoint - ideal_mid_time

    Timezone handling:
      - start/end/ideal are coerced to `tz`
      - naive datetimes are treated as local in `tz`
    """
    if units not in {"seconds", "minutes", "hours"}:
        raise ValueError("units must be one of: 'seconds', 'minutes', 'hours'")

    # Coerce ideal to tz (also makes naive-safe)
    ideal_mid_time = ensure_tz(ideal_mid_time, tz)
    if ideal_mid_time is None:
        raise ValueError("ideal_mid_time cannot be None")

    # Parse keys
    parsed = [(_parse_pixel_key(k), v) for k, v in pixel_dict.items()]
    if not parsed:
        raise ValueError("pixel_dict is empty.")

    rows = np.array([rc[0] for rc, _ in parsed], dtype=int)
    cols = np.array([rc[1] for rc, _ in parsed], dtype=int)
    rmin, rmax = int(rows.min()), int(rows.max())
    cmin, cmax = int(cols.min()), int(cols.max())

    rmin_p, rmax_p = rmin - pad, rmax + pad
    cmin_p, cmax_p = cmin - pad, cmax + pad

    nrows = (rmax_p - rmin_p) + 1
    ncols = (cmax_p - cmin_p) + 1
    delta = np.full((nrows, ncols), np.nan, dtype=float)

    scale = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[units]

    # Fill
    for (r, c), info in parsed:
        start = ensure_tz(info.get("light_start_time", None), tz)
        end = ensure_tz(info.get("light_end_time", None), tz)
        if start is None or end is None:
            continue

        # Midpoint (timezone-aware timedeltas are fine)
        mid = start + (end - start) / 2

        # Signed offset: positive means midpoint is AFTER ideal_mid_time
        dsec = (mid - ideal_mid_time).total_seconds()

        rr, cc = r - rmin_p, c - cmin_p
        delta[rr, cc] = dsec / scale

    # Colormap (NaNs)
    cm = plt.colormaps[cmap].copy()
    cm.set_bad(nan_color)

    # Symmetric color limits around 0
    finite = delta[np.isfinite(delta)]
    if finite.size:
        if robust and finite.size >= 10:
            lo, hi = np.percentile(finite, robust_pct)
            m = max(abs(lo), abs(hi))
        else:
            m = float(np.nanmax(np.abs(finite)))
        vmin, vmax = -m, m
    else:
        vmin, vmax = -1.0, 1.0

    # Pixel-coordinate extent
    extent = (
        [cmin_p - 0.5, cmax_p + 0.5, rmax_p + 0.5, rmin_p - 0.5]
        if origin == "upper"
        else [cmin_p - 0.5, cmax_p + 0.5, rmin_p - 0.5, rmax_p + 0.5]
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(delta, origin=origin, cmap=cm, extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(f"Midpoint offset vs ideal ({units})\nideal = {ideal_mid_time.isoformat()}")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"midpoint - ideal ({units})")

    if show:
        plt.show()

    return (
        fig,
        ax,
        {
            "delta": delta,
            "ideal_mid_time": ideal_mid_time,
            "bbox": {"rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax, "pad": pad},
        },
    )


def plot_midpoint_offset_heatmap_autoideal(
    pixel_dict: Dict[str, Dict[str, Any]],
    *,
    tz: ZoneInfo = ZoneInfo("America/Denver"),
    pad: int = 0,
    origin: str = "upper",
    cmap: str = "coolwarm",
    nan_color: str = "white",
    units: str = "seconds",  # "seconds" | "minutes" | "hours"
    robust: bool = True,
    robust_pct: Tuple[float, float] = (2, 98),
    show: bool = True,
):
    """
    1) Compute midpoint for every pixel with valid start/end.
    2) Set ideal_mid_time = median(midpoints) (median in epoch seconds).
    3) Heatmap shows signed delta: midpoint - ideal_mid_time, in requested units.

    Returns the ideal time used (timezone-converted to `tz`) along with arrays.
    """
    if units not in {"seconds", "minutes", "hours"}:
        raise ValueError("units must be one of: 'seconds', 'minutes', 'hours'")

    parsed = [(_parse_pixel_key(k), v) for k, v in pixel_dict.items()]
    if not parsed:
        raise ValueError("pixel_dict is empty.")

    # Bounding box based on provided keys
    rows = np.array([rc[0] for rc, _ in parsed], dtype=int)
    cols = np.array([rc[1] for rc, _ in parsed], dtype=int)
    rmin, rmax = rows.min(), rows.max()
    cmin, cmax = cols.min(), cols.max()

    rmin_p, rmax_p = rmin - pad, rmax + pad
    cmin_p, cmax_p = cmin - pad, cmax + pad
    nrows = (rmax_p - rmin_p) + 1
    ncols = (cmax_p - cmin_p) + 1

    # First pass: collect midpoint times (epoch seconds)
    mids_epoch = []
    for (_, _), info in parsed:
        start = info.get("light_start_time")
        end = info.get("light_end_time")
        if start is None or end is None:
            continue
        # Ensure both are tz-aware for safe arithmetic; if naive, treat as tz
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz)
        if end.tzinfo is None:
            end = end.replace(tzinfo=tz)

        mid = start + (end - start) / 2
        mids_epoch.append(_to_epoch_seconds(mid, tz))

    if not mids_epoch:
        raise ValueError("No valid (light_start_time, light_end_time) pairs found.")

    ideal_epoch = float(np.median(np.array(mids_epoch, dtype=float)))
    ideal_mid_time = datetime.fromtimestamp(ideal_epoch, tz=timezone.utc).astimezone(tz)

    # Second pass: fill delta grid
    delta = np.full((nrows, ncols), np.nan, dtype=float)
    scale = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[units]

    for (r, c), info in parsed:
        start = info.get("light_start_time")
        end = info.get("light_end_time")
        if start is None or end is None:
            continue
        if start.tzinfo is None:
            start = start.replace(tzinfo=tz)
        if end.tzinfo is None:
            end = end.replace(tzinfo=tz)

        mid = start + (end - start) / 2
        dsec = _to_epoch_seconds(mid, tz) - ideal_epoch
        rr, cc = r - rmin_p, c - cmin_p
        delta[rr, cc] = dsec / scale

    # Colormap with NaN color
    cm = plt.colormaps[cmap].copy()
    cm.set_bad(nan_color)

    finite = delta[np.isfinite(delta)]
    if finite.size:
        if robust and finite.size >= 10:
            lo, hi = np.percentile(finite, robust_pct)
            m = max(abs(lo), abs(hi))
        else:
            m = np.nanmax(np.abs(finite))
        vmin, vmax = -m, m
    else:
        vmin, vmax = -1, 1

    extent = (
        [cmin_p - 0.5, cmax_p + 0.5, rmax_p + 0.5, rmin_p - 0.5]
        if origin == "upper"
        else [cmin_p - 0.5, cmax_p + 0.5, rmin_p - 0.5, rmax_p + 0.5]
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(delta, origin=origin, cmap=cm, extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(f"Midpoint offset vs median midpoint ({units})\nmedian = {ideal_mid_time.isoformat()}")
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"midpoint - median(midpoints) ({units})")

    if show:
        plt.show()

    return fig, ax, {"delta": delta, "ideal_mid_time": ideal_mid_time, "ideal_epoch": ideal_epoch}


def scatter_start_time_vs_duration(
    pixel_dict: Dict[str, Dict[str, Any]],
    *,
    tz: ZoneInfo = ZoneInfo("America/Denver"),
    ax: Optional[plt.Axes] = None,
    duration_key: str = "light_duration",
    start_key: str = "light_start_time",
    duration_units: str = "s",
    alpha: float = 0.4,
    s: float = 12,
    color: str = "C0",
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of illumination duration vs start time across all pixels.

    x-axis: light_start_time (datetime; tz-aware OK)
    y-axis: light_duration (assumed seconds unless your data says otherwise)
    """
    xs, ys = [], []
    for info in pixel_dict.values():
        start = info.get(start_key, None)
        dur = info.get(duration_key, None)
        if start is None or dur is None:
            continue
        if not isinstance(start, datetime):
            raise TypeError(f"{start_key} must be datetime; got {type(start)}")

        start = ensure_tz(start, tz)

        xs.append(start)
        ys.append(float(dur))

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    else:
        fig = ax.figure

    ax.scatter(xs, ys, s=s, alpha=alpha, c=color, edgecolors="none", label=duration_key)

    ax.set_xlabel("Start time")
    ax.set_ylabel(f"Illumination duration ({duration_units})")
    ax.set_title("Illumination duration vs start time")

    # Date formatting on x-axis
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=tz))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator(tz=tz), tz=tz))
    fig.autofmt_xdate()

    ax.grid(True, alpha=0.25)
    ax.legend()

    if show:
        plt.show()

    return fig, ax


def pointing_info_extract(timing_data_location, video_metadata, camera_time_shift, data_time_zone, ideal_midpoint_time):

    video_start_time = datetime.strptime(video_metadata['create_date'], "%Y:%m:%d %H:%M:%S")
    tz = ZoneInfo(data_time_zone)
    # interpret create_date as local time in tz (if that's what it is)
    video_start_time = video_start_time.replace(tzinfo=tz)

    # Coerce ideal to tz (also makes naive-safe)
    ideal_mid_time = ensure_tz(ideal_midpoint_time, tz)
    if ideal_mid_time is None:
        raise ValueError("ideal_mid_time cannot be None")

    # apply camera shift if intended (positive shift moves timestamps forward)
    video_start_time = video_start_time + camera_time_shift

    data = lbt.read_compressed_json(timing_data_location)
    details = {}

    for pix, trans in list(data.items()):
        frame_range_pixel = []
        if len(trans) < 1:
            continue
        else:
            for transition in trans:
                if transition["transition"] == "bright":
                    frame_range_pixel.append(tuple((1, lbt.frame_number_from_img_name(transition['to_frame']))))
                elif transition["transition"] == "dark":
                    frame_range_pixel.append(tuple((0, lbt.frame_number_from_img_name(transition['to_frame']))))

            first_light = min([item for item in frame_range_pixel if item[0] == 1])
            last_light = max([item for item in frame_range_pixel if item[0] == 0])
            frame_diff = last_light[1] - first_light[1]
            light_duration = frame_diff / video_metadata['frame_rate']
            light_start_time_shift = first_light[1] / video_metadata['frame_rate']
            light_end_time_shift = last_light[1] / video_metadata['frame_rate']
            light_start_time = video_start_time + timedelta(seconds=light_start_time_shift)
            light_end_time = video_start_time + timedelta(seconds=light_end_time_shift)

            cel_vec_max_range_frames, cel_vec_max_frame_span = lbt.maximum_frame_range(frame_ranges=frame_range_pixel)
            cel_vec_bright_start_time = video_start_time + timedelta(
                seconds=cel_vec_max_range_frames[0][1] / video_metadata['frame_rate']
            )
            cel_vec_bright_end_time = (
                video_start_time
                + timedelta(seconds=cel_vec_max_range_frames[0][1] / video_metadata['frame_rate'])
                + timedelta(seconds=cel_vec_max_frame_span / video_metadata['frame_rate'])
            )
            cel_vec_bright_duration = cel_vec_max_frame_span / video_metadata['frame_rate']

        details[pix] = {
            "first_light": first_light,
            "last_light": last_light,
            "cel_vec_start_end": cel_vec_max_range_frames,
            "light_duration": light_duration,
            "light_duration_analysis": cel_vec_bright_duration,
            "light_duration_delta": light_duration - cel_vec_bright_duration,
            "light_start_time": light_start_time,
            "light_end_time": light_end_time,
            "light_midpoint": light_start_time + (light_end_time - light_start_time) / 2,
            "light_midpoint_delta": (light_start_time + (light_end_time - light_start_time) / 2) - ideal_mid_time,
            "cel_vec_light_start_time": cel_vec_bright_start_time,
            "cel_vec_light_end_time": cel_vec_bright_end_time,
            "cel_vec_light_midpoint": cel_vec_bright_start_time
            + (cel_vec_bright_end_time - cel_vec_bright_start_time) / 2,
            "cel_vec_light_midpoint_delta": (
                cel_vec_bright_start_time + (cel_vec_bright_end_time - cel_vec_bright_start_time) / 2
            )
            - ideal_mid_time,
            "ideal_midpoint_time": ideal_mid_time,
        }

    return details


def _per_pixel_vertical_shift_task(args):
    """
    Returns (rr, cc, ch_len_float_or_nan, (shift0, shift1) or (nan, nan))
    """
    k, info, value_key, rmin_p, cmin_p, distance_ref, sun_trav_rad_s, ideal_spot, trav_direction = args

    r, c = _parse_pixel_key(k)
    rr, cc = r - rmin_p, c - cmin_p

    v = info.get(value_key, np.nan)

    ch_len = chord_length_from_transit_time(
        transit_time_s=v, distance_m=distance_ref, angular_speed_rad_s=sun_trav_rad_s
    )
    ch_len_out = np.nan if ch_len is None else float(ch_len)

    temp = fit_y_shifts_for_chord_length(
        spot=ideal_spot,
        target_chord_length_m=ch_len,
        direction_deg=trav_direction,
        show=False,
        n_grid=5001,
        refine_factor=25,
        refine_half_window_pts=20,
    )

    shifts = temp.get("shifts", (np.nan, np.nan))
    if shifts is None or (isinstance(shifts, (list, tuple)) and any(s is None for s in shifts)):
        shifts_out = (np.nan, np.nan)
    else:
        # ensure tuple of floats
        shifts_out = (float(shifts[0]), float(shifts[1]))

    return rr, cc, ch_len_out, shifts_out


def extract_vertical_shift_parallel(
    pixel_dict,
    value_key,
    ideal_spot,
    distance_ref,
    sun_trav_rad_s,
    pad,
    trav_direction=0,
    render=False,
    cmap="coolwarm",
    nan_color="white",
    max_workers=None,  # None -> os.cpu_count()
    show=False,
):
    rmin_p, rmax_p, cmin_p, cmax_p = _cropped_bbox_from_keys(pixel_dict, pad=pad)
    nrows = (rmax_p - rmin_p) + 1
    ncols = (cmax_p - cmin_p) + 1

    ch_lengths = np.full((nrows, ncols), np.nan, dtype=float)
    shifts_map = np.full((nrows, ncols), np.nan, dtype=object)  # store tuples

    # Build tasks (avoid sending huge structures; we send per-item info)
    items = list(pixel_dict.items())
    tasks = [
        (k, info, value_key, rmin_p, cmin_p, distance_ref, sun_trav_rad_s, ideal_spot, trav_direction)
        for (k, info) in items
    ]

    # Use processes for CPU-bound work
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Submit tasks; for very large lists you can submit in batches,
        # but this is fine for many use cases.
        futures = [ex.submit(_per_pixel_vertical_shift_task, t) for t in tasks]

        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting Vertical Pointing Shifts", unit="pixel"
        ):
            rr, cc, ch_len_out, shifts_out = fut.result()
            ch_lengths[rr, cc] = ch_len_out
            shifts_map[rr, cc] = shifts_out

    # Build shift0/shift1 maps (vectorized-ish)
    shift0 = np.full((nrows, ncols), np.nan, dtype=float)
    shift1 = np.full((nrows, ncols), np.nan, dtype=float)
    shiftmin = np.full((nrows, ncols), np.nan, dtype=float)
    shiftmax = np.full((nrows, ncols), np.nan, dtype=float)

    for i in range(nrows):
        for j in range(ncols):
            t = shifts_map[i, j]
            if isinstance(t, tuple) and len(t) >= 2:
                shift0[i, j] = t[0]
                shift1[i, j] = t[1]
                shiftmin[i, j] = np.min(t)
                shiftmax[i, j] = np.max(t)

    vmin = np.nanmin(shiftmin)
    vmax = np.nanmax(shiftmax)

    if render:
        extent = _extent_from_bbox(rmin_p, rmax_p, cmin_p, cmax_p, "upper")
        cm = plt.colormaps[cmap].copy()
        cm.set_bad(nan_color)

        fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8, 8))

        im0 = ax0.imshow(shift0, origin="upper", cmap=cm, extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
        ax0.set_title(value_key + " Shift map First")
        ax0.set_xlabel("col")
        ax0.set_ylabel("row")
        ax0.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
        ax0.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
        fig.colorbar(im0, ax=ax0, shrink=0.9)

        im1 = ax1.imshow(shift1, origin="upper", cmap=cm, extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
        ax1.set_title(value_key + " Shift map Second")
        ax1.set_xlabel("col")
        ax1.set_ylabel("row")
        ax1.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
        ax1.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
        fig.colorbar(im1, ax=ax1, shrink=0.9)

        im2 = ax2.imshow(shiftmin, origin="upper", cmap=cm, extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
        ax2.set_title(value_key + " Shift map Min")
        ax2.set_xlabel("col")
        ax2.set_ylabel("row")
        ax2.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
        ax2.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
        fig.colorbar(im2, ax=ax2, shrink=0.9)

        im3 = ax3.imshow(shiftmax, origin="upper", cmap=cm, extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
        ax3.set_title(value_key + " Shift map Max")
        ax3.set_xlabel("col")
        ax3.set_ylabel("row")
        ax3.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
        ax3.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
        fig.colorbar(im3, ax=ax3, shrink=0.9)

        if show:
            plt.show()

    return fig if fig else None, ch_lengths, shifts_map, shift0, shift1, shiftmin, shiftmax


def extract_vertical_shift(
    pixel_dict,
    value_key,
    ideal_spot,
    distance_ref,
    sun_trav_rad_s,
    pad,
    trav_direction=0,
    show_full_maps=False,
    cmap="coolwarm",
    nan_color="white",
):

    rmin_p, rmax_p, cmin_p, cmax_p = _cropped_bbox_from_keys(pixel_dict, pad=pad)
    nrows = (rmax_p - rmin_p) + 1
    ncols = (cmax_p - cmin_p) + 1

    ch_lengths = np.full((nrows, ncols), np.nan, dtype=float)
    shifts_map = np.full((nrows, ncols), np.nan, dtype=tuple)

    for k, info in tqdm(
        pixel_dict.items(), total=len(pixel_dict), desc="Extracting Vertical Pointing Shifts", unit="pixel"
    ):
        r, c = _parse_pixel_key(k)
        rr, cc = r - rmin_p, c - cmin_p
        v = info.get(value_key, np.nan)
        ch_len = chord_length_from_transit_time(
            transit_time_s=v, distance_m=distance_ref, angular_speed_rad_s=sun_trav_rad_s
        )
        ch_lengths[rr, cc] = np.nan if ch_len is None else float(ch_len)
        temp = fit_y_shifts_for_chord_length(
            spot=ideal_spot,
            target_chord_length_m=ch_len,
            direction_deg=trav_direction,
            show=False,
            n_grid=5001,
            refine_factor=25,
            refine_half_window_pts=20,
        )
        shifts_map[rr, cc] = (np.nan, np.nan) if None in temp["shifts"] else temp["shifts"]

    shift0 = np.full((nrows, ncols), np.nan, dtype=float)
    shift1 = np.full((nrows, ncols), np.nan, dtype=float)

    for i in range(nrows):
        for j in range(ncols):
            t = shifts_map[i, j]
            if isinstance(t, tuple) and len(t) >= 2:
                shift0[i, j] = t[0]
                shift1[i, j] = t[1]

    if show_full_maps:
        extent = _extent_from_bbox(rmin_p, rmax_p, cmin_p, cmax_p, "upper")
        cm = plt.colormaps[cmap].copy()
        cm.set_bad(nan_color)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        im0 = ax0.imshow(shift0, origin="upper", cmap=cm, extent=extent, aspect="equal")
        ax0.set_title("Shift map (tuple[0])")
        ax0.set_xlabel("col")
        ax0.set_ylabel("row")
        ax0.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
        ax0.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
        fig.colorbar(im0, ax=ax0, shrink=0.9)

        im1 = ax1.imshow(shift1, origin="upper", cmap=cm, extent=extent, aspect="equal")
        ax1.set_title("Shift map (tuple[1])")
        ax1.set_xlabel("col")
        ax1.set_ylabel("row")
        ax1.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
        ax1.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
        fig.colorbar(im1, ax=ax1, shrink=0.9)

        plt.show()

    return ch_lengths, shifts_map, shift0, shift1


def extract_horizontal_shift(
    distance_ref: float,
    sun_trav_rad_s: float,
    pixel_dict: Dict[str, Dict[str, Any]],
    *,
    value_key: str,
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,
    pad: int = 0,
    origin: str = "upper",
    cmap: str = "viridis",
    data_range: tuple = (None, None),
    nan_color: str = "white",
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Parameters
    ----------
    value_key:
        Key inside each pixel's info dict to plot (e.g., "light_duration").
    title, cbar_label:
        Plot title / colorbar label. If None, defaults are derived from value_key.
    """
    velocity = distance_ref * sun_trav_rad_s  # m/s

    rmin_p, rmax_p, cmin_p, cmax_p = _cropped_bbox_from_keys(pixel_dict, pad=pad)
    nrows = (rmax_p - rmin_p) + 1
    ncols = (cmax_p - cmin_p) + 1

    arr = np.full((nrows, ncols), np.nan, dtype=float)

    for k, info in pixel_dict.items():
        r, c = _parse_pixel_key(k)
        rr, cc = r - rmin_p, c - cmin_p
        v = info.get(value_key, np.nan)
        arr[rr, cc] = np.nan if v is None else float(v.total_seconds() * velocity)

    cm = plt.colormaps[cmap].copy()
    cm.set_bad(nan_color)

    extent = _extent_from_bbox(rmin_p, rmax_p, cmin_p, cmax_p, origin)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    else:
        fig = ax.figure

    im = ax.imshow(
        arr,
        origin=origin,
        cmap=cm,
        extent=extent,
        vmin=data_range[0] if data_range[0] else None,
        vmax=data_range[1] if data_range[1] else None,
    )
    ax.set_title(title if title is not None else value_key)
    ax.set_xlabel("col")
    ax.set_ylabel("row")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label if cbar_label is not None else value_key)

    # tighten bounds
    ax.set_xlim(cmin_p - 0.5, cmax_p + 0.5)
    if origin == "upper":
        ax.set_ylim(rmax_p + 0.5, rmin_p - 0.5)
    else:
        ax.set_ylim(rmin_p - 0.5, rmax_p + 0.5)

    if show:
        plt.show()

    meta = {
        "bbox": {"rmin": rmin_p + pad, "rmax": rmax_p - pad, "cmin": cmin_p + pad, "cmax": cmax_p - pad, "pad": pad}
    }
    return fig, ax, {"array": arr, "extent": extent, **meta}


def array_statistics(x, *, ignore_nan=True, percentiles=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    """
    Compute a dictionary of useful summary statistics for a numpy array.

    Parameters
    ----------
    x : array-like
        Input data (any shape). Will be flattened for summary stats.
    ignore_nan : bool, default True
        If True, use NaN-aware stats (nanmean, nanpercentile, etc.).
    percentiles : tuple of numbers
        Percentiles to compute (0-100).

    Returns
    -------
    stats : dict
        Dictionary of statistics. If all values are NaN (or input empty),
        many fields will be NaN and counts will reflect that.
    """
    a = np.asarray(x).ravel()

    # Handle empty input early
    if a.size == 0:
        return {
            "count_total": 0,
            "count_finite": 0,
            "count_nan": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "var": np.nan,
            "iqr": np.nan,
            "range": np.nan,
            "percentiles": {p: np.nan for p in percentiles},
        }

    finite_mask = np.isfinite(a)
    n_total = a.size
    n_finite = int(finite_mask.sum())
    n_nan = int(np.isnan(a).sum())

    # Choose reduction functions
    if ignore_nan:
        amin = np.nanmin
        amax = np.nanmax
        amean = np.nanmean
        amedian = np.nanmedian
        astd = np.nanstd
        avar = np.nanvar
        aperc = np.nanpercentile
        data_for_geom = a[finite_mask]
    else:
        amin = np.min
        amax = np.max
        amean = np.mean
        amedian = np.median
        astd = np.std
        avar = np.var
        aperc = np.percentile
        data_for_geom = a

    # If ignoring NaNs and nothing is finite, return mostly NaNs
    if ignore_nan and n_finite == 0:
        return {
            "count_total": n_total,
            "count_finite": 0,
            "count_nan": n_nan,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "var": np.nan,
            "iqr": np.nan,
            "range": np.nan,
            "mad": np.nan,
            "percentiles": {p: np.nan for p in percentiles},
        }

    # Core stats
    minv = float(amin(a))
    maxv = float(amax(a))
    meanv = float(amean(a))
    medv = float(amedian(a))
    stdv = float(astd(a))
    varv = float(avar(a))

    # Quartiles + IQR
    q25, q50, q75 = aperc(a, [25, 50, 75])
    iqr = float(q75 - q25)

    # Median absolute deviation (robust spread). "raw" MAD (no scaling).
    # (If you want approx std for normal data: mad_sigma ≈ 1.4826 * MAD)
    mad = float(amean(np.abs((a if not ignore_nan else a[finite_mask]) - medv))) if False else None
    # Use true MAD definition:
    if ignore_nan:
        mad = float(np.nanmedian(np.abs(a - medv)))
    else:
        mad = float(np.median(np.abs(a - medv)))

    # Percentiles (including quartiles if included in percentiles list)
    pvals = aperc(a, list(percentiles))
    percentiles_dict = {float(p): float(v) for p, v in zip(percentiles, pvals)}

    # A couple of commonly useful extras
    rng = float(maxv - minv)
    sem = float(stdv / np.sqrt(n_finite if ignore_nan else n_total))

    stats = {
        "count_total": n_total,
        "count_finite": n_finite,
        "count_nan": n_nan,
        "min": minv,
        "max": maxv,
        "range": rng,
        "mean": meanv,
        "median": medv,
        "std": stdv,
        "var": varv,
        "sem": sem,  # standard error of the mean
        "q25": float(q25),
        "q50": float(q50),
        "q75": float(q75),
        "iqr": iqr,
        "mad": mad,  # median absolute deviation (robust)
        "percentiles": percentiles_dict,
    }

    return stats


def pointing_estimate(
    timing_data_path,
    video_path,
    output_dir,
    camera_time_shift,
    ideal_time_local,
    ref_distance,
    timezone="America/Denver",
    ideal_spot_parameters={"center": (0, 0), "axis_ratio": 1, "axis_orientation_angle": 0, "traverse_direction": 0.0},
    sun_diam_pad_mul=1.2,
):
    ft.create_directories_if_necessary(output_dir)
    video_metadata = lbt.extract_detailed_video_metadata(video_path)
    camera_time_shift = camera_time_shift if camera_time_shift else timedelta(0, 0, 0)
    timezone = timezone if timezone else "America/Denver"
    tz = ZoneInfo(timezone)
    # ideal_time = datetime(2025, 6, 5, 15, 18, 20, tzinfo=tz)
    ideal_time = ideal_time_local

    results = pointing_info_extract(
        timing_data_location=timing_data_path,
        video_metadata=video_metadata,
        camera_time_shift=camera_time_shift,
        data_time_zone=timezone,
        ideal_midpoint_time=ideal_time,
    )

    # distance = 99.94392
    distance = ref_distance
    # traverse_direction = traverse_direction
    omega = 2 * np.pi / 86400  # rad/s, ~solar apparent motion rate (solar day seconds is 86,400)

    Ideal_spot = reflected_sun_spot_ellipse(
        distance,
        angular_diameter_rad=sun_diam_pad_mul * 9.3e-3,
        axis_ratio=ideal_spot_parameters["axis_ratio"],  # circle special case is ratio of 1
        orientation_deg=ideal_spot_parameters[
            "axis_orientation_angle"
        ],  # major axis aligned with +x (irrelevant if circle)
        center_xy=ideal_spot_parameters["center"],
    )
    Ideal_spot_chord_length = ellipse_chord_length_along_direction(
        Ideal_spot.a_m,
        Ideal_spot.b_m,
        center_xy=Ideal_spot.center_xy,
        orientation_deg=Ideal_spot.orientation_deg,
        direction_deg=ideal_spot_parameters["traverse_direction"],
    )
    Ideal_spot_chord_traverse_time = time_for_ellipse_to_pass_chord(
        chord_length_m=Ideal_spot_chord_length, distance_m=distance, angular_speed_rad_s=omega
    )

    fig_ito, axes_ito, arrays_ito = plot_midpoint_offset_heatmap_tz(
        results, ideal_mid_time=ideal_time, tz=tz, pad=2, units="seconds", show=False
    )
    fig_ito.savefig(ft.join(output_dir, "ideal_time_midpoint_offset_full.png"))

    fig_horz_full, ax_horz_full, array_horz_full = extract_horizontal_shift(
        distance_ref=distance,
        sun_trav_rad_s=omega,
        pixel_dict=results,
        value_key="light_midpoint_delta",
        title="Horizontal Shift Full Range",
        cbar_label="Horizontal Shift [m]",
        pad=2,
        cmap="coolwarm",
        show=False,
    )
    fig_horz_full.savefig(ft.join(output_dir, "ideal_time_midpoint_offset_distance_full.png"))

    array_horz_full_stats = array_statistics(array_horz_full['array'])
    lbt.write_json(array_horz_full_stats, ft.join(output_dir, "ideal_time_midpoint_offset_distance_full_stats.json"))

    fig_horz_ana, ax_horz_ana, array_horz_ana = extract_horizontal_shift(
        distance_ref=distance,
        sun_trav_rad_s=omega,
        pixel_dict=results,
        value_key="cel_vec_light_midpoint_delta",
        title="Horizontal Shift Analysis",
        cbar_label="Horizontal Shift [m]",
        pad=2,
        cmap="coolwarm",
        show=False,
    )
    fig_horz_ana.savefig(ft.join(output_dir, "ideal_time_midpoint_offset_distance_analysis.png"))
    array_horz_ana_stats = array_statistics(array_horz_ana['array'])
    lbt.write_json(array_horz_ana_stats, ft.join(output_dir, "ideal_time_midpoint_offset_distance_analysis_stats.json"))

    (
        fig_v_shift_map_full,
        ch_len_full,
        v_shift_map_full,
        v_shift_full_0,
        v_shift_full_1,
        v_shift_full_min,
        v_shift_full_max,
    ) = extract_vertical_shift_parallel(
        pixel_dict=results,
        value_key="light_duration",
        ideal_spot=Ideal_spot,
        distance_ref=distance,
        sun_trav_rad_s=omega,
        pad=2,
        trav_direction=ideal_spot_parameters["traverse_direction"],
        render=True,
        show=False,
    )
    fig_v_shift_map_full.savefig(ft.join(output_dir, "vertical_shift_full.png"))
    v_shift_full_min_stats = array_statistics(v_shift_full_min)
    lbt.write_json(v_shift_full_min_stats, ft.join(output_dir, "vertical_shift_full_minimum_stats.json"))
    v_shift_full_max_stats = array_statistics(v_shift_full_max)
    lbt.write_json(v_shift_full_max_stats, ft.join(output_dir, "vertical_shift_full_maximum_stats.json"))

    fig_v_shift_map_ana, ch_len_ana, v_shift_map_ana, v_shift_ana_0, v_shift_ana_1, v_shift_ana_min, v_shift_ana_max = (
        extract_vertical_shift_parallel(
            pixel_dict=results,
            value_key="light_duration_analysis",
            ideal_spot=Ideal_spot,
            distance_ref=distance,
            sun_trav_rad_s=omega,
            pad=2,
            trav_direction=ideal_spot_parameters["traverse_direction"],
            render=True,
            show=False,
        )
    )
    fig_v_shift_map_full.savefig(ft.join(output_dir, "vertical_shift_analysis.png"))
    v_shift_ana_min_stats = array_statistics(v_shift_ana_min)
    lbt.write_json(v_shift_ana_min_stats, ft.join(output_dir, "vertical_shift_analysis_minimum_stats.json"))
    v_shift_ana_max_stats = array_statistics(v_shift_ana_max)
    lbt.write_json(v_shift_ana_max_stats, ft.join(output_dir, "vertical_shift_analysis_maximum_stats.json"))

    # fig_time, axes_time, arrays_time = plot_light_maps_cropped(results, tz=tz, pad=2, show=False)
    # duration (scalar)
    fig_mmdur, axes_mmdur, dat_mmdur = plot_scalar_map_cropped(
        results,
        value_key="light_duration",
        title="Light Duration (s)",
        cbar_label="Duration (s)",
        pad=2,
        cmap="viridis",
        show=False,
        data_range=(0, 200),
    )
    fig_mmdur.savefig(ft.join(output_dir, "light_duration_full.png"))

    fig_mmdur_ana, axes_mmdur_ana, dat_mmdur_ana = plot_scalar_map_cropped(
        results,
        value_key="light_duration_analysis",
        title="Light Duration Analysis (s)",
        cbar_label="Duration (s)",
        pad=2,
        cmap="viridis",
        show=False,
        data_range=(0, 200),
    )
    fig_mmdur_ana.savefig(ft.join(output_dir, "light_duration_analysis.png"))

    # start time (datetime)
    fig_lst, ax_lst, dat_lst = plot_time_map_cropped(
        results, time_key="light_start_time", title="Light start time", pad=2, cmap="cool", show=False
    )
    fig_lst.savefig(ft.join(output_dir, "light_start_time_full.png"))

    # end time (datetime)
    fig_let, ax_let, dat_let = plot_time_map_cropped(
        results, time_key="light_end_time", title="Light end time", pad=2, cmap="cool", show=False
    )
    fig_let.savefig(ft.join(output_dir, "light_end_time_full.png"))

    # start time (datetime)
    fig_lst_ana, ax_lst_ana, dat_lst_ana = plot_time_map_cropped(
        results, time_key="cel_vec_light_start_time", title="Analysis Light start time", pad=2, cmap="cool", show=False
    )
    fig_lst_ana.savefig(ft.join(output_dir, "light_start_time_analysis.png"))

    # end time (datetime)
    fig_let_ana, ax_let_ana, dat_let_ana = plot_time_map_cropped(
        results, time_key="cel_vec_light_end_time", title="Analysis Light end time", pad=2, cmap="cool", show=False
    )
    fig_let_ana.savefig(ft.join(output_dir, "light_end_time_analysis.png"))

    fig_scatter, axes_scatter = scatter_start_time_vs_duration(results, tz=tz, show=False)
    fig_scatter.savefig(ft.join(output_dir, "light_start_time_vs_duration_full.png"))

    fig_scat_ana, axes_scat_ana = scatter_start_time_vs_duration(
        results, tz=tz, color="C3", duration_key="light_duration_analysis", show=False
    )
    fig_scat_ana.savefig(ft.join(output_dir, "light_start_time_vs_duration_analysis.png"))
    fig_scat_delta, axes_scat_delta = scatter_start_time_vs_duration(
        results, tz=tz, color="C2", duration_key="light_duration_delta", show=False
    )
    fig_scat_delta.savefig(ft.join(output_dir, "durations_delta_vs_start_time_full.png"))
    print("Completed Pointing Estimate Module")


if __name__ == "__main__":
    timing_data_temp_location = r"\\snl\Collaborative\NSTTF_Optics\Projects\_Directories\NSTTF_Optics_LookbackExEx\Experiments\2025-06_05_NsttfTunedFacetScan1dof\3_Post\ini_script_test_3\7_pixel_timing_interrogation\50\time_history_transition_parallel_50_final.json.gz"
    video_temp_path = r"\\snl\Collaborative\NSTTF_Optics\Projects\_Directories\NSTTF_Optics_LookbackExEx\Experiments\2025-06_05_NsttfTunedFacetScan1dof\3_Post\ini_script_test_3\DSC_0025.MOV"
    output_dir = r"\\snl\Collaborative\NSTTF_Optics\Projects\_Directories\NSTTF_Optics_LookbackExEx\Experiments\2025-06_05_NsttfTunedFacetScan1dof\3_Post\ini_script_test_3\10_pointing_estimate"
    video_metadata_temp = lbt.extract_detailed_video_metadata(video_temp_path)
    camera_time_shift_temp = timedelta(0, 0, 0)
    timezone_temp = "America/Denver"
    tz = ZoneInfo(timezone_temp)
    ideal_time = datetime(2025, 6, 5, 15, 18, 20, tzinfo=tz)

    results = pointing_info_extract(
        timing_data_location=timing_data_temp_location,
        video_metadata=video_metadata_temp,
        camera_time_shift=camera_time_shift_temp,
        data_time_zone=timezone_temp,
        ideal_midpoint_time=ideal_time,
    )

    distance = 99.94392
    traverse_direction = 0
    omega = 2 * np.pi / 86400  # rad/s, ~solar apparent motion rate (solar day seconds is 86,400)
    sun_diam_pad_mul = 1.2

    Ideal_spot = reflected_sun_spot_ellipse(
        distance,
        angular_diameter_rad=sun_diam_pad_mul * 9.3e-3,
        axis_ratio=1,  # circle special case is ratio of 1
        orientation_deg=0,  # major axis aligned with +x (irrelevant if circle)
        center_xy=(0, 0),
    )
    Ideal_spot_chord_length = ellipse_chord_length_along_direction(
        Ideal_spot.a_m,
        Ideal_spot.b_m,
        center_xy=Ideal_spot.center_xy,
        orientation_deg=Ideal_spot.orientation_deg,
        direction_deg=traverse_direction,
    )
    Ideal_spot_chord_traverse_time = time_for_ellipse_to_pass_chord(
        chord_length_m=Ideal_spot_chord_length, distance_m=distance, angular_speed_rad_s=omega
    )

    plot_ellipse_and_center_chord(spot_obj=Ideal_spot, direction_deg=traverse_direction, show=False)

    shift_test_chord_length = 1.5 * Ideal_spot.a_m
    out = fit_y_shifts_for_chord_length(
        Ideal_spot,
        shift_test_chord_length,
        n_grid=10001,
        direction_deg=traverse_direction,
        show=False,
        refine_factor=50,
        refine_half_window_pts=50,
    )
    fig_ito, axes_ito, arrays_ito = plot_midpoint_offset_heatmap_tz(
        results, ideal_mid_time=ideal_time, tz=tz, pad=2, units="seconds", show=False
    )
    fig_ito.savefig(ft.join(output_dir, "ideal_time_midpoint_offset_full.png"))
    # fig_mid, axes_mid, arrays_mid = plot_midpoint_offset_heatmap_autoideal(
    #     results, tz=tz, pad=2, units="seconds", show=False
    # )
    fig_horz_full, ax_horz_full, array_horz_full = extract_horizontal_shift(
        distance_ref=distance,
        sun_trav_rad_s=omega,
        pixel_dict=results,
        value_key="light_midpoint_delta",
        title="Horizontal Shift Full Range",
        cbar_label="Horizontal Shift [m]",
        pad=2,
        cmap="coolwarm",
        show=False,
    )
    fig_horz_full.savefig(ft.join(output_dir, "ideal_time_midpoint_offset_distance_full.png"))

    array_horz_full_stats = array_statistics(array_horz_full['array'])
    lbt.write_json(array_horz_full_stats, ft.join(output_dir, "ideal_time_midpoint_offset_distance_full_stats.json"))

    fig_horz_ana, ax_horz_ana, array_horz_ana = extract_horizontal_shift(
        distance_ref=distance,
        sun_trav_rad_s=omega,
        pixel_dict=results,
        value_key="cel_vec_light_midpoint_delta",
        title="Horizontal Shift Analysis",
        cbar_label="Horizontal Shift [m]",
        pad=2,
        cmap="coolwarm",
        show=False,
    )
    fig_horz_ana.savefig(ft.join(output_dir, "ideal_time_midpoint_offset_distance_analysis.png"))
    array_horz_ana_stats = array_statistics(array_horz_ana['array'])
    lbt.write_json(array_horz_ana_stats, ft.join(output_dir, "ideal_time_midpoint_offset_distance_analysis_stats.json"))

    (
        fig_v_shift_map_full,
        ch_len_full,
        v_shift_map_full,
        v_shift_full_0,
        v_shift_full_1,
        v_shift_full_min,
        v_shift_full_max,
    ) = extract_vertical_shift_parallel(
        pixel_dict=results,
        value_key="light_duration",
        ideal_spot=Ideal_spot,
        distance_ref=distance,
        sun_trav_rad_s=omega,
        pad=2,
        trav_direction=traverse_direction,
        render=True,
        show=False,
    )
    fig_v_shift_map_full.savefig(ft.join(output_dir, "vertical_shift_full.png"))
    v_shift_full_min_stats = array_statistics(v_shift_full_min)
    lbt.write_json(v_shift_full_min_stats, ft.join(output_dir, "vertical_shift_full_minimum_stats.json"))
    v_shift_full_max_stats = array_statistics(v_shift_full_max)
    lbt.write_json(v_shift_full_max_stats, ft.join(output_dir, "vertical_shift_full_maximum_stats.json"))

    fig_v_shift_map_ana, ch_len_ana, v_shift_map_ana, v_shift_ana_0, v_shift_ana_1, v_shift_ana_min, v_shift_ana_max = (
        extract_vertical_shift_parallel(
            pixel_dict=results,
            value_key="light_duration_analysis",
            ideal_spot=Ideal_spot,
            distance_ref=distance,
            sun_trav_rad_s=omega,
            pad=2,
            trav_direction=traverse_direction,
            render=True,
            show=False,
        )
    )
    fig_v_shift_map_full.savefig(ft.join(output_dir, "vertical_shift_analysis.png"))
    v_shift_ana_min_stats = array_statistics(v_shift_ana_min)
    lbt.write_json(v_shift_ana_min_stats, ft.join(output_dir, "vertical_shift_analysis_minimum_stats.json"))
    v_shift_ana_max_stats = array_statistics(v_shift_ana_max)
    lbt.write_json(v_shift_ana_max_stats, ft.join(output_dir, "vertical_shift_analysis_maximum_stats.json"))

    # fig_time, axes_time, arrays_time = plot_light_maps_cropped(results, tz=tz, pad=2, show=False)
    # duration (scalar)
    fig_mmdur, axes_mmdur, dat_mmdur = plot_scalar_map_cropped(
        results,
        value_key="light_duration",
        title="Light Duration (s)",
        cbar_label="Duration (s)",
        pad=2,
        cmap="viridis",
        show=False,
        data_range=(0, 200),
    )
    fig_mmdur.savefig(ft.join(output_dir, "light_duration_full.png"))

    fig_mmdur_ana, axes_mmdur_ana, dat_mmdur_ana = plot_scalar_map_cropped(
        results,
        value_key="light_duration_analysis",
        title="Light Duration Analysis (s)",
        cbar_label="Duration (s)",
        pad=2,
        cmap="viridis",
        show=False,
        data_range=(0, 200),
    )
    fig_mmdur_ana.savefig(ft.join(output_dir, "light_duration_analysis.png"))

    # start time (datetime)
    fig_lst, ax_lst, dat_lst = plot_time_map_cropped(
        results, time_key="light_start_time", title="Light start time", pad=2, cmap="cool", show=False
    )
    fig_lst.savefig(ft.join(output_dir, "light_start_time_full.png"))

    # end time (datetime)
    fig_let, ax_let, dat_let = plot_time_map_cropped(
        results, time_key="light_end_time", title="Light end time", pad=2, cmap="cool", show=False
    )
    fig_let.savefig(ft.join(output_dir, "light_end_time_full.png"))

    # start time (datetime)
    fig_lst_ana, ax_lst_ana, dat_lst_ana = plot_time_map_cropped(
        results, time_key="cel_vec_light_start_time", title="Analysis Light start time", pad=2, cmap="cool", show=False
    )
    fig_lst_ana.savefig(ft.join(output_dir, "light_start_time_analysis.png"))

    # end time (datetime)
    fig_let_ana, ax_let_ana, dat_let_ana = plot_time_map_cropped(
        results, time_key="cel_vec_light_end_time", title="Analysis Light end time", pad=2, cmap="cool", show=False
    )
    fig_let_ana.savefig(ft.join(output_dir, "light_end_time_analysis.png"))

    fig_scatter, axes_scatter = scatter_start_time_vs_duration(results, tz=tz, show=False)
    fig_scatter.savefig(ft.join(output_dir, "light_start_time_vs_duration_full.png"))

    fig_scat_ana, axes_scat_ana = scatter_start_time_vs_duration(
        results, tz=tz, color="C3", duration_key="light_duration_analysis", show=False
    )
    fig_scat_ana.savefig(ft.join(output_dir, "light_start_time_vs_duration_analysis.png"))
    fig_scat_delta, axes_scat_delta = scatter_start_time_vs_duration(
        results, tz=tz, color="C2", duration_key="light_duration_delta", show=False
    )
    fig_scat_delta.savefig(
        ft.join(output_dir, "delta_between_durations_full_and_analysis_interval_vs_start_time_full.png")
    )

    plt.show()
    print("stop here")
