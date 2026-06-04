'''
Steps Outline for camera coordinate system relative pointing estimate.
1 - Select a set of sun position times during test start with 1 every minute.
2 - Transform the sun vectors into camera coordinates and mirror coordinates
        Propogate all alignment rotations as well
3 - Using camera slope information BEFORE TRANSFORMING TO ORTHORECTIFIED ALIGNMENT and perform a simple ray trace where all rays are intersecting the mirror points with data at the sun direction
4 - Propogate those reflected rays out until they intersect with the camera coordinate system XY plane
5 - Plot


Alternative plan
Utilize the time intervals from the main analysis to determine where the beamlet from each mixel (assumed to be an elipse)
crosses the camera system.

A perfectly focused and made optic would have each and every beamlet produce the maximum crossing time.
'''

import os
from logging import DEBUG, ERROR
from datetime import datetime, timedelta, timezone
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from zoneinfo import ZoneInfo
from typing import Dict, Any, Tuple, Optional

import opencsp.app.lookback.lookback_tools as lbt
import opencsp.common.lib.tool.file_tools as ft


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


def pointing_info_extract(timing_data_location, video_metadata, camera_time_shift, data_time_zone):

    video_start_time = datetime.strptime(video_metadata['create_date'], "%Y:%m:%d %H:%M:%S")
    tz = ZoneInfo(data_time_zone)
    # interpret create_date as local time in tz (if that's what it is)
    video_start_time = video_start_time.replace(tzinfo=tz)

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
            "cel_vec_light_start_time": cel_vec_bright_start_time,
            "cel_vec_light_end_time": cel_vec_bright_end_time,
            "cel_vec_light_midpoint": cel_vec_bright_start_time
            + (cel_vec_bright_end_time - cel_vec_bright_start_time) / 2,
        }
    return details


if __name__ == "__main__":
    timing_data_temp_location = r"\\snl\Collaborative\NSTTF_Optics\Projects\_Directories\NSTTF_Optics_LookbackExEx\Experiments\2025-06_05_NsttfTunedFacetScan1dof\3_Post\ini_script_test_3\7_pixel_timing_interrogation\50\time_history_transition_parallel_50_final.json.gz"
    video_temp_path = r"\\snl\Collaborative\NSTTF_Optics\Projects\_Directories\NSTTF_Optics_LookbackExEx\Experiments\2025-06_05_NsttfTunedFacetScan1dof\3_Post\ini_script_test_3\DSC_0025.MOV"
    video_metadata_temp = lbt.extract_detailed_video_metadata(video_temp_path)
    camera_time_shift_temp = timedelta(0, 0, 0)
    timezone_temp = "America/Denver"
    tz = ZoneInfo(timezone_temp)

    results = pointing_info_extract(
        timing_data_location=timing_data_temp_location,
        video_metadata=video_metadata_temp,
        camera_time_shift=camera_time_shift_temp,
        data_time_zone=timezone_temp,
    )

    # fig_time, axes_time, arrays_time = plot_light_maps_cropped(results, tz=tz, pad=2, show=False)
    # duration (scalar)
    fig_mmdur, axes_mmdur, array_mmdur = plot_scalar_map_cropped(
        results,
        value_key="light_duration",
        title="Light Duration (s)",
        cbar_label="Duration (s)",
        pad=2,
        cmap="viridis",
        show=False,
        data_range=(0, 200),
    )

    fig_mmdur_ana, axes_mmdur_ana, array_mmdur_ana = plot_scalar_map_cropped(
        results,
        value_key="light_duration_analysis",
        title="Light Duration Analysis (s)",
        cbar_label="Duration (s)",
        pad=2,
        cmap="viridis",
        show=False,
        data_range=(0, 200),
    )

    # start time (datetime)
    _ = plot_time_map_cropped(
        results, time_key="light_start_time", title="Light start time", pad=2, cmap="cool", show=False
    )

    # end time (datetime)
    _ = plot_time_map_cropped(
        results, time_key="light_end_time", title="Light end time", pad=2, cmap="cool", show=False
    )

    # start time (datetime)
    _ = plot_time_map_cropped(
        results, time_key="cel_vec_light_start_time", title="Analysis Light start time", pad=2, cmap="cool", show=False
    )

    # end time (datetime)
    _ = plot_time_map_cropped(
        results, time_key="cel_vec_light_end_time", title="Analysis Light end time", pad=2, cmap="cool", show=False
    )

    fig_mid, axes_mid, arrays_mid = plot_midpoint_offset_heatmap_autoideal(
        results, tz=tz, pad=2, units="seconds", show=False
    )
    fig_scatter, axes_scatter = scatter_start_time_vs_duration(results, tz=tz, show=False)
    fig_scat_ana, axes_scat_ana = scatter_start_time_vs_duration(
        results, tz=tz, color="C3", duration_key="light_duration_analysis", show=False
    )
    fig_scat_delta, axes_scat_delta = scatter_start_time_vs_duration(
        results, tz=tz, color="C2", duration_key="light_duration_delta", show=False
    )
    plt.show()
    print("stop here")
