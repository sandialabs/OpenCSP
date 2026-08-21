# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

"""Heatmap plots and animated GIFs for irradiance analysis results."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from moviepy import VideoClip

from .utils import get_reduced_voxels_with_irrads, slug
from .ffiam_types import AimType, Direction


@dataclass
class AnimationFrameData:
    """Data for a single GIF animation frame."""
    vox_xs: NDArray[np.floating]
    vox_ys: NDArray[np.floating]
    vox_irrads: NDArray[np.floating]
    depth_value: float
    depth_label: str                       # e.g. "Altitude", "North", "East"
    base_title: str
    xlims: Tuple[float, float]
    ylims: Tuple[float, float]
    marker_size: float
    color_range: Tuple[float, float]

ZOOM_SPAN = 100
PLOT_W_PX_DEFAULT = 700
PLOT_SCALE_DEFAULT = 2


def get_zoom_lims(
    aim_strat: AimType,
    aim: NDArray[np.floating],
    dim: Direction,
) -> Tuple[float, float]:
    """Axis limits for the zoomed plot, centered on the aim point."""
    half_span = ZOOM_SPAN / 2

    if aim_strat == AimType.Point:
        if dim in [Direction.East, Direction.North]:
            min_val = aim[dim.value] - half_span
            max_val = aim[dim.value] + half_span
        else:
            min_val = -10
            max_val = aim[dim.value] + half_span

    elif aim_strat in (AimType.Ring, AimType.SplitRing):
        # ring/split-ring: aim is [inner_radius, outer_radius, height]; extent set by outer radius
        r = aim[1]
        ht = aim[2]
        if dim in [Direction.East, Direction.North]:
            min_val = (-r - 50) if r > half_span else -half_span
            max_val = (r + 50) if r > half_span else half_span
        else:
            min_val = -10
            max_val = ht + half_span

    else:
        # fallback (e.g. vector/csv/fixed-normal): center on aim[0]/aim[1]
        r = aim[0]
        ht = aim[1]
        if dim in [Direction.East, Direction.North]:
            min_val = (-r - 50) if r > half_span else -half_span
            max_val = (r + 50) if r > half_span else half_span
        else:
            min_val = -10
            max_val = ht + half_span

    return min_val, max_val


def convert_fig_to_array(fig: go.Figure) -> NDArray[np.uint8]:
    """Convert plotly figure to RGB numpy array."""
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def create_path_cumsum_plot(
    time_per_voxel: float,
    cumulative_exposures: NDArray[np.floating],
    site: str,
    path_id: int = 1,
    out_dir: Optional[Path] = None,
) -> Path:
    """Plot cumulative radiant exposure [J/m2] over time along a flight path."""
    if out_dir is None:
        out_dir = Path(__file__).parent.resolve()

    n_elems = len(cumulative_exposures)
    time_arr = np.array([time_per_voxel] * n_elems)
    time_arr = np.cumsum(time_arr)

    fig = go.Figure()

    plot_fpath = out_dir.joinpath(f"{slug(site)}_path{path_id}_cumulative_radiant_exposure.png")

    fig.add_trace(go.Scatter(x=time_arr, y=cumulative_exposures, mode='lines', name="ECDF"))
    fig.update_layout(
            title=f'Radiant Exposure Along Path {path_id}',
            xaxis_title='Time (s)',
            yaxis_title='Radiant Exposure (J/m<sup>2</sup>)',
    )
    fig.write_image(plot_fpath, format='png')
    return plot_fpath


#
# HEAT MAP PLOTS
#

def create_profile_plots(
    hel_locs: NDArray[np.floating],
    vox_locs: NDArray[np.floating],
    irrads: NDArray[np.floating],
    dim1: Direction,
    dim2: Direction,
    site: str,
    threshold: float,
    vox_size: int,
    hel_size: float,
    aim_strat: AimType,
    aim_params: NDArray[np.floating],
    tower_h: float = 0,
    create_zoomed: bool = True,
    create_gif: bool = True,
    filename: Optional[str] = None,
    title: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Create 2D profile heatmap for the given Cartesian axis pair.

    Returns (plot_path, zoomed_plot_path, gif_path); zoomed and gif may be None.
    """
    if out_dir is None:
        out_dir = Path(__file__).parent.resolve()
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if vox_locs.shape[0] != 3:
        vox_locs = vox_locs.T
    vxs_all, vys_all, vzs_all = vox_locs

    if hel_locs.shape[0] != 3:
        hel_locs = hel_locs.T

    plot_xs_all, h_plot_xs, label1, slug1 = _get_axis_plotting_data(dim1, hel_locs, vox_locs)
    plot_ys_all, h_plot_ys, label2, slug2 = _get_axis_plotting_data(dim2, hel_locs, vox_locs)

    gif_depth_coords, label3 = _get_gif_depth_axis_data(dim1, dim2, vxs_all, vys_all, vzs_all)

    # Static plots use max-reduced voxels; GIF uses full per-slice data
    red_plot_xs, red_plot_ys, red_irrads = get_reduced_voxels_with_irrads(plot_xs_all, plot_ys_all, irrads)

    all_xlocs_reduced = np.concatenate((h_plot_xs, red_plot_xs))
    all_ylocs_reduced = np.concatenate((h_plot_ys, red_plot_ys))

    hel_mkr_size, vox_mkr_size, hel_mkr_size_zoom, vox_mkr_size_zoom = \
        _calculate_plot_parameters(all_xlocs_reduced, all_ylocs_reduced, hel_size, vox_size)

    base_filename = filename if filename else f"{slug(site)}_{slug1}{slug2}"
    thres_label = f"{threshold} kW/m<sup>2</sup> threshold"

    main_plot_title = title
    if main_plot_title is None:
        main_plot_title = f"{site}: {label1}-{label2}\n{thres_label}, max irradiance/depth"

    if title is None:
        zoomed_plot_title = f"{main_plot_title.splitlines()[0]} (Zoomed)\n{thres_label}"
    else:
        zoomed_plot_title = f"{title} (Zoomed)"

    plot_fpath = out_dir.joinpath(base_filename + ".png")
    plot_zoom_fpath = out_dir.joinpath(base_filename + "_zoom.png")
    gif_fpath = out_dir.joinpath(base_filename + ".gif") if create_gif else None

    dim2_is_up = (dim2 == Direction.Up)
    xtitle_str = f"{label1} (m)"
    ytitle_str = f"{label2} (m)"

    _create_static_plot(plot_fpath,
                        h_plot_xs, h_plot_ys, hel_mkr_size,
                        red_plot_xs, red_plot_ys, red_irrads, vox_mkr_size,
                        main_plot_title, xtitle_str, ytitle_str,
                        tower_h=tower_h, dim2_is_up=dim2_is_up)

    if create_zoomed:
        xlims_zoom = get_zoom_lims(aim_strat, aim_params, dim1)
        ylims_zoom = get_zoom_lims(aim_strat, aim_params, dim2)
        _create_static_plot(plot_zoom_fpath,
                            h_plot_xs, h_plot_ys, hel_mkr_size_zoom,
                            red_plot_xs, red_plot_ys, red_irrads, vox_mkr_size_zoom,
                            zoomed_plot_title, xtitle_str, ytitle_str,
                            xlims=xlims_zoom, ylims=ylims_zoom,
                            tower_h=tower_h, dim2_is_up=dim2_is_up) # Tower might be out of zoom

    if create_gif:
        all_xlocs_full = np.concatenate((h_plot_xs, plot_xs_all))
        all_ylocs_full = np.concatenate((h_plot_ys, plot_ys_all))

        gif_xlims = (np.nanmin(all_xlocs_full) - 3, np.nanmax(all_xlocs_full) + 3)
        gif_ylims = (np.nanmin(all_ylocs_full) - 3, np.nanmax(all_ylocs_full) + 3)
        gif_color_range = (threshold, np.nanmax(irrads))

        gif_base_title = f"{site}: {label1}-{label2}\n{thres_label}"

        _create_animation_gif(gif_fpath,
                              h_plot_xs, h_plot_ys, hel_mkr_size,
                              plot_xs_all, plot_ys_all, gif_depth_coords, irrads,
                              gif_base_title, xtitle_str, ytitle_str, label3,
                              gif_xlims, gif_ylims, vox_mkr_size, gif_color_range,
                              tower_h=tower_h, dim2_is_up=dim2_is_up)

    returned_gif_fpath = gif_fpath if create_gif else None
    return plot_fpath, plot_zoom_fpath, returned_gif_fpath


#
# PLOT HELPERS
#

def _add_scatter(
    fig: go.Figure,
    xs: NDArray[np.floating],
    ys: NDArray[np.floating],
    colors: NDArray[np.floating],
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    marker_size: Optional[float] = None,
    color_range: Optional[Tuple[float, float]] = None,
    title: str = "",
    xtitle: str = "East (m)",
    ytitle: str = "North (m)",
) -> None:
    """Add scatter data to existing plotly figure."""
    if marker_size is None:
        xmin = np.nanmin(xs) - 5
        xmax = np.nanmax(xs) + 5
        marker_size = int((xmax - xmin) / 4)

    mkr_dict = dict(size=marker_size,
                    symbol='square',
                    color=colors,
                    colorscale='YlOrRd',
                    colorbar=dict(title="Irradiance (kW/m<sup>2</sup>)"),
                    showscale=True)
    if color_range is not None:
        mkr_dict['cmin'] = color_range[0]
        mkr_dict['cmax'] = color_range[1]

    fig.add_trace(go.Scattergl(
            x=xs,
            y=ys,
            mode='markers',
            marker=mkr_dict, name="voxels"))

    fig.update_xaxes(title_text=xtitle, range=xlims, constrain="domain")
    fig.update_yaxes(title_text=ytitle, range=ylims, scaleanchor="x", scaleratio=1)  # , range=[-20, 20]) # scaleanchor="x", scaleratio=1)
    # fig.update_layout(title=title, height=800, showlegend=False)
    fig.update_layout(title=title, showlegend=False)


def _update_scatter(
    fig: go.Figure,
    xs: NDArray[np.floating],
    ys: NDArray[np.floating],
    colors: NDArray[np.floating],
    xlims: Tuple[float, float],
    ylims: Tuple[float, float],
    mkr_size: float,
    color_range: Tuple[float, float],
    title: Optional[str] = None,
) -> None:
    """Update scatter data in existing plot during GIF creation."""
    fig.update_traces(go.Scattergl(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(
                    size=mkr_size,
                    symbol='square',
                    color=colors,
                    colorscale='YlOrRd',
                    cmin=color_range[0],
                    cmax=color_range[1],
                    # colorbar=dict(title="Irradiance (kW/m<sup>2</sup>)"),
                    # showscale=True
            )),
            selector={"name": "voxels"}
    )
    fig.update_xaxes(range=xlims, constrain="domain")
    fig.update_yaxes(range=ylims, scaleanchor="x", scaleratio=1)

    if title:
        fig.update_layout(title=title, showlegend=False)


def _add_heliostats(
    fig: go.Figure,
    xs: NDArray[np.floating],
    ys: NDArray[np.floating],
    marker_size: float = 10,
    color: str = "#0099C6",
) -> None:
    """Plot heliostat representations as squares in existing figure."""
    fig.add_trace(go.Scattergl(
            x=xs,
            y=ys,
            mode='markers',
            marker=dict(size=int(marker_size), symbol='square', color=color)
    ))


def _add_tower(fig: go.Figure, h: float, w: float = 5) -> None:
    """Plot CSP tower representation as filled rectangle."""
    tower_color = "#bab0ac"  # "#6E899C"
    fig.add_trace(go.Scattergl(
            x=[-w, -w, w, w],
            y=[0, h, h, 0],
            fill="toself",
            fillcolor=tower_color,
            line_width=0,
            mode="none"
    ))


def _get_axis_plotting_data(
    dim_enum: Direction,
    hel_locs: NDArray[np.floating],
    vox_locs: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], NDArray[np.floating], str, str]:
    """Return (voxel_coords, helio_coords, label, slug) for the given axis."""
    if dim_enum == Direction.East:
        return vox_locs[0], hel_locs[0], "East", "E"
    elif dim_enum == Direction.North:
        return vox_locs[1], hel_locs[1], "North", "N"
    elif dim_enum == Direction.Up:  # Primarily for dim2
        return vox_locs[2], hel_locs[2], "Up", "U"
    raise ValueError(f"Invalid Direction: {dim_enum}")


def _get_gif_depth_axis_data(
    dim1: Direction,
    dim2: Direction,
    vxs_all: NDArray[np.floating],
    vys_all: NDArray[np.floating],
    vzs_all: NDArray[np.floating],
) -> Tuple[NDArray[np.floating], str]:
    """Return (depth_coordinates, label) for the third axis not shown in the 2D plot."""
    if dim1 == Direction.East and dim2 == Direction.North:
        return vzs_all, "Altitude"
    elif dim1 == Direction.East and dim2 == Direction.Up:
        return vys_all, "North"
    elif dim1 == Direction.North and dim2 == Direction.Up:
        return vxs_all, "East"
    raise ValueError(f"Invalid dimension combination for GIF depth: {dim1}, {dim2}")


def _calculate_plot_parameters(
    all_xlocs: NDArray[np.floating],
    all_ylocs: NDArray[np.floating],
    hel_size: float,
    vox_size: int,
    plot_w_px: int = PLOT_W_PX_DEFAULT,
    zoom_span: int = ZOOM_SPAN,
) -> Tuple[float, float, float, float]:
    """Return pixel marker sizes for (heliostats, voxels, heliostats_zoom, voxels_zoom)."""
    plot_w_m = 1.4 * (np.nanmax(all_xlocs) - np.nanmin(all_xlocs))
    if plot_w_m == 0: plot_w_m = 1.0

    m_to_px = plot_w_px / plot_w_m
    m_to_px_zoom = plot_w_px / (1.9 * zoom_span)

    hel_mkr_size = hel_size * m_to_px
    vox_mkr_size = max(1, vox_size * m_to_px)

    hel_mkr_size_zoom = hel_size * m_to_px_zoom
    vox_mkr_size_zoom = max(1, vox_size * m_to_px_zoom)

    return hel_mkr_size, vox_mkr_size, hel_mkr_size_zoom, vox_mkr_size_zoom


def _create_static_plot(
    filepath: Path,
    h_plot_xs: NDArray[np.floating],
    h_plot_ys: NDArray[np.floating],
    hel_marker_size: float,
    vox_plot_xs: NDArray[np.floating],
    vox_plot_ys: NDArray[np.floating],
    vox_plot_colors: NDArray[np.floating],
    vox_marker_size: float,
    title_str: str,
    xtitle_str: str,
    ytitle_str: str,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    tower_h: float = 0.,
    dim2_is_up: bool = False,
    plot_width: Optional[int] = None,
    plot_scale: int = PLOT_SCALE_DEFAULT,
) -> None:
    """Create and save a single static profile plot."""
    fig = go.Figure()
    _add_heliostats(fig, xs=h_plot_xs, ys=h_plot_ys, marker_size=int(hel_marker_size))

    if dim2_is_up and tower_h > 0.1:
        _add_tower(fig, h=int(tower_h), w=5)

    _add_scatter(fig,
                 xs=vox_plot_xs,
                 ys=vox_plot_ys,
                 colors=vox_plot_colors,
                 xlims=xlims,
                 ylims=ylims,
                 title=title_str,
                 xtitle=xtitle_str,
                 ytitle=ytitle_str,
                 marker_size=vox_marker_size)
    fig.write_image(filepath, format='png', width=plot_width, scale=plot_scale)


def render_animation_frame(
    fig: go.Figure,
    frame_data: AnimationFrameData,
) -> NDArray[np.uint8]:
    """Render one GIF frame. Extracted from _create_animation_gif for testability."""
    frame_title = f"{frame_data.base_title}, {frame_data.depth_label} {int(frame_data.depth_value)} m"

    _update_scatter(
        fig,
        xs=frame_data.vox_xs,
        ys=frame_data.vox_ys,
        colors=frame_data.vox_irrads,
        xlims=frame_data.xlims,
        ylims=frame_data.ylims,
        mkr_size=frame_data.marker_size,
        color_range=frame_data.color_range,
        title=frame_title,
    )
    return convert_fig_to_array(fig)


def _create_animation_gif(
    gif_fpath: Path,
    h_plot_xs: NDArray[np.floating],
    h_plot_ys: NDArray[np.floating],
    base_hel_mkr_size: float,
    vox_data_xs: NDArray[np.floating],
    vox_data_ys: NDArray[np.floating],
    vox_data_depth_coords: NDArray[np.floating],
    vox_data_irrads: NDArray[np.floating],
    gif_base_title: str,
    gif_xtitle: str,
    gif_ytitle: str,
    gif_depth_label: str,
    gif_xlims: Tuple[float, float],
    gif_ylims: Tuple[float, float],
    base_vox_mkr_size: float,
    gif_color_range: Tuple[float, float],
    tower_h: float = 0.,
    dim2_is_up: bool = False,
) -> None:
    """Create and save animated GIF for profile plots."""
    gfig = go.Figure()
    _add_heliostats(gfig, xs=h_plot_xs, ys=h_plot_ys, marker_size=base_hel_mkr_size)
    if dim2_is_up and tower_h > 0.1:
        _add_tower(gfig, h=tower_h, w=5)

    _add_scatter(gfig, xs=[], ys=[], colors=[],
                 title=gif_base_title,
                 xlims=gif_xlims,
                 ylims=gif_ylims,
                 marker_size=base_vox_mkr_size,
                 color_range=gif_color_range,
                 xtitle=gif_xtitle,
                 ytitle=gif_ytitle)
    gfig.update_layout(template="plotly_dark")

    unique_depth_values = np.unique(vox_data_depth_coords)
    if unique_depth_values.size == 0:
        print(f"Warning: No data slices to animate for GIF: {gif_fpath}. Skipping GIF creation.")
        return

    depth_values_list = unique_depth_values.tolist()
    num_depth_slices = len(depth_values_list)
    fps = 12 if num_depth_slices > 6 else max(1, num_depth_slices)
    duration = num_depth_slices / fps if num_depth_slices > 6 else 1

    initial_title = str(gfig.layout.title.text) if gfig.layout.title.text else gif_base_title

    frame_data_list: List[AnimationFrameData] = []
    for depth_val in depth_values_list:
        mask = vox_data_depth_coords == depth_val
        frame_data_list.append(AnimationFrameData(
            vox_xs=vox_data_xs[mask],
            vox_ys=vox_data_ys[mask],
            vox_irrads=vox_data_irrads[mask],
            depth_value=depth_val,
            depth_label=gif_depth_label,
            base_title=initial_title,
            xlims=gif_xlims,
            ylims=gif_ylims,
            marker_size=base_vox_mkr_size,
            color_range=gif_color_range,
        ))

    frame_index = [0]  # mutable container so the closure can increment it

    def _make_frame(t: float) -> NDArray[np.uint8]:
        if frame_index[0] >= len(frame_data_list):
            return convert_fig_to_array(gfig)

        frame_data = frame_data_list[frame_index[0]]
        frame_index[0] += 1
        return render_animation_frame(gfig, frame_data)

    animation = VideoClip(_make_frame, duration=duration).resized(2)
    animation.write_gif(gif_fpath, fps=fps)


