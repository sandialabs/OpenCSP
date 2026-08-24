# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import pandas as pd
import scipy
import plotly.graph_objects as go

from pyffiam.ffiam_types import CspSite, FieldLayout, get_csp_site_from_string, get_aim_type_from_string
from pyffiam.app_config import GB, MAX_VOXELS, VOXEL_POOL_SIZE


def slug(in_str: str) -> str:
    """Convert string to URL-safe slug format."""
    return in_str.lower().replace(' ', '_')


def print_params(func: Callable) -> Callable:
    """Decorator to print parameters passed to a function."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"\nFunction '{func.__name__}' called with:")
        if args:
            print("Args:", args)
        if kwargs:
            print("Kwargs:", kwargs)
        return func(*args, **kwargs)

    return wrapper


def load_preset_sites_from_json() -> Dict[CspSite, Dict[str, Any]]:
    """Load all preset site config JSON files.

    JSON format matches UnrealEngine - do not change the schema.
    Files with a BOM (ï»¿) must have it removed (PyCharm: UTF-8 > Remove BOM).
    """
    app_dir = Path(__file__).parent.resolve()
    data_path = app_dir.joinpath('site_configs/presets')
    results = {}

    preset_files = {
        CspSite.NSTTF: 'Nsttf.json',
        CspSite.RadialSmall: 'RadialSm.json',
        CspSite.Radial: 'Radial.json',
        CspSite.SampleV1: 'SampleV1.json',
        CspSite.SampleV2: 'SampleV2.json',
        CspSite.SampleV3: 'SampleV3.json',
    }

    for site_key, filename in preset_files.items():
        file_path = data_path.joinpath(filename)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Preset configuration file not found ({file_path.as_posix()}). Verify setup of FFIAM."
            )

        file_content = file_path.read_text()

        if not file_content.strip():
            raise ValueError(f"Preset configuration file is empty: {file_path.as_posix()}")

        try:
            site_data = json.loads(file_content)
            results[site_key] = site_data
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in preset configuration file {file_path.as_posix()}: {str(e)}\n"
                f"First 100 characters of file: {file_content[:100]!r}"
            )
    return results


def _parse_helio_design(json_dict: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Extract heliostat design parameters from a site config JSON dict into result."""
    if "HelioDesign" not in json_dict:
        return

    helio = json_dict["HelioDesign"]
    result["n_facets"] = helio["NumFacets"]
    result["n_facet_cols"] = helio["NumCols"]
    result["facet_w"] = helio["FacetWidth"]
    result["facet_h"] = helio["FacetHeight"]
    result["facet_file"] = helio["FacetFile"]


def _parse_field_config(json_dict: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Extract field/location parameters from a site config JSON dict into result."""
    if "Field" not in json_dict:
        return

    field = json_dict["Field"]
    result["lat"] = field["Latitude"]
    result["lng"] = field["Longitude"]
    result["timezone"] = field["Timezone"]
    result["field_r"] = field["FieldRadius"]
    result["n_helios"] = field["NumHelios"]
    result["min_height"] = field["MinHeight"]
    result["max_height"] = field["MaxHeight"]
    result["tower_height"] = field["TowerHeight"]
    result["helio_file"] = field["HelioCoordinateFile"]
    result["layout"] = FieldLayout.from_str(field.get("Layout", "grid")).value


def _parse_datetime(json_dict: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Extract date/time parameters from a site config JSON dict into result."""
    if "DateTime" not in json_dict:
        return

    dt = json_dict["DateTime"]
    result["year"] = dt["Year"]
    result["month"] = dt["Month"]
    result["day"] = dt["Day"]
    result["hour"] = dt["Hour"]


def _parse_aim_strategy(json_dict: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Extract aim strategy type, parameters, and optional file from a site config JSON dict into result."""
    if "AimStrategy" not in json_dict:
        return

    aim = json_dict["AimStrategy"]
    result["aim_strat"] = get_aim_type_from_string(str(aim["Type"]))

    if "Params" in aim and aim["Params"] is not None:
        params = aim["Params"]
        result["aim_params"] = np.array([params['X'], params['Y'], params['Z']])

    if "File" in aim and aim["File"] is not None:
        result["aim_file"] = aim["File"]


def _parse_paths(json_dict: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Extract UAS path waypoints and speeds from a site config JSON dict into result."""
    if "Paths" not in json_dict:
        return

    all_paths = []
    all_speeds = []
    for path in json_dict["Paths"]:
        all_paths.append(path['Points'])
        all_speeds.append(path['Speed'])

    result['paths'] = all_paths
    result['path_speeds'] = all_speeds


def get_site_config_dict_from_json(filepath: Path) -> Dict[str, Any]:
    """Parse a site config JSON file into an analysis-ready dict.

    JSON must match the UnrealEngine schema.
    """
    with open(filepath, 'r') as f:
        try:
            json_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise e

    if isinstance(json_dict, list):
        print(
            "Error: Expected a single site configuration, not a list. "
            "Use separate JSON files to analyze a list of sites."
        )
        return {}

    run_id = json_dict.pop("RunId", "Unidentified site")
    result: Dict[str, Any] = {}

    result["site"] = get_csp_site_from_string(str(json_dict["SiteType"]))
    result['threshold'] = json_dict.get('Threshold', 4)

    _parse_helio_design(json_dict, result)
    _parse_field_config(json_dict, result)
    _parse_datetime(json_dict, result)
    _parse_aim_strategy(json_dict, result)
    _parse_paths(json_dict, result)

    if 'create_gifs' in json_dict:
        if not isinstance(json_dict['create_gifs'], bool):
            print(f"Warning for {run_id}: 'create_gifs' should be boolean. " f"Got: {json_dict['create_gifs']}")
            result['create_gifs'] = False
        else:
            result['create_gifs'] = json_dict['create_gifs']

    return result


def get_reduced_voxels_with_irrads(
    voxel_xs: NDArray[np.floating], voxel_ys: NDArray[np.floating], irrads: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Flatten 3D voxel positions into 2D with max irradiance values for each x-y."""
    xy_irrads_flat = np.array([voxel_xs, voxel_ys, irrads]).T
    xy_irrads_flat_sorted = xy_irrads_flat[np.argsort(xy_irrads_flat[:, 2])[::-1]]
    idxs_max = np.unique(xy_irrads_flat_sorted[:, :2], return_index=True, axis=0)[1]
    xy_irrads_max = xy_irrads_flat_sorted[idxs_max]
    return xy_irrads_max.T


def get_voxel_loc_from_index(idx: int, vox_size: int, field_r: int, field_zmin: int) -> NDArray[np.floating]:
    """Convert flat voxel index to [x, y, z] center coordinates (meters)."""
    n_per_side = field_r * 2 / vox_size
    n_per_plane = n_per_side**2
    base_z = idx / n_per_plane
    rem = idx - base_z * n_per_plane
    base_y = rem / n_per_side
    base_x = rem - base_y * n_per_side

    # tower is at origin; shift index-space origin to center
    x = base_x * vox_size - field_r
    y = base_y * vox_size - field_r
    z = base_z * vox_size + field_zmin
    return np.array([x, y, z])


def get_voxel_locs_from_indexes(
    ids: NDArray[np.integer], vox_size: int, field_r: int, field_zmin: int
) -> NDArray[np.floating]:
    """Convert array of voxel indices to (N, 3) [x, y, z] coordinates (meters)."""
    n_per_side = field_r * 2 / vox_size
    n_per_plane = n_per_side**2
    base_zs = (ids / n_per_plane).astype(int)
    rem = ids - base_zs * n_per_plane
    base_ys = (rem / n_per_side).astype(int)
    base_xs = rem - base_ys * n_per_side

    xs = base_xs * vox_size - field_r
    ys = base_ys * vox_size - field_r
    zs = base_zs * vox_size + field_zmin
    result = np.array([xs, ys, zs]).T
    return result


def compute_vox_locs(n_voxels: int, vox_size: int, field_r: int, field_zmin: int) -> NDArray[np.floating]:
    """Return (n_voxels, 3) array of [x, y, z] voxel center coordinates (meters)."""
    ids = np.arange(n_voxels)
    locs = get_voxel_locs_from_indexes(ids, vox_size, field_r, field_zmin)
    return locs


def get_voxel_indexes_from_locs(
    locs: NDArray[np.floating], vox_size: int, field_r: int, field_zmin: int
) -> NDArray[np.floating]:
    """Convert (N, 3) [x, y, z] coordinates to flat voxel indices."""
    xs, ys, zs = locs.T
    n_per_side = field_r * 2 / vox_size
    n_per_plane = n_per_side**2
    new_zs = np.floor((zs - field_zmin) / vox_size)
    new_ys = np.floor((ys + field_r) / vox_size)
    new_xs = np.floor((xs + field_r) / vox_size)
    indexes = new_zs * n_per_plane + new_ys * n_per_side + new_xs
    return indexes


def interpolate_3d(
    p1: Union[List[float], Tuple[float, float, float], NDArray[np.floating]],
    p2: Union[List[float], Tuple[float, float, float], NDArray[np.floating]],
    step_size: float = 1.0,
) -> List[NDArray[np.floating]]:
    """Interpolate points between p1 and p2 at step_size intervals.

    Includes p1, excludes p2. Used to enumerate voxels along a flight path.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    distance = np.linalg.norm(p2 - p1)
    steps = int(distance / step_size)

    direction = (p2 - p1) / distance
    interpolated_points = [p1 + i * step_size * direction for i in range(steps)]
    return interpolated_points


def sample_2d_with_constrained_median(array: NDArray[np.floating], step_size: int) -> NDArray[np.floating]:
    """Downsample a 2D array using median over the inner 70% of each block.

    Uses max instead of median when median is near zero - important near the
    focal point where a single high-flux cell would otherwise be missed.
    """
    rows, cols = array.shape

    out_rows = rows // step_size + (1 if rows % step_size != 0 else 0)
    out_cols = cols // step_size + (1 if cols % step_size != 0 else 0)

    result = np.zeros((out_rows, out_cols))
    # exclude outer 15% of each block to match SolTrace's sampling behaviour
    limit = int(step_size * 0.15)

    for i in range(out_rows):
        for j in range(out_cols):
            r_start = i * step_size + limit
            r_end = min((i + 1) * step_size - limit, rows)
            c_start = j * step_size + limit
            c_end = min((j + 1) * step_size - limit, cols)

            section = array[r_start:r_end, c_start:c_end]
            med_val = np.median(section)
            max_val = np.max(section)
            final_val = med_val
            if (med_val - 0.1) < 0:
                final_val = max_val

            result[i, j] = final_val

    return result


def conv_ffiam_to_dataframe(
    irrads: NDArray[np.floating],
    vox_locs: NDArray[np.floating],
    altitude: float,
    n_voxels_x: int,
    voxel_size: float = 2,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    step_size: Optional[float] = None,
) -> pd.DataFrame:
    """Slice irradiance data at a given altitude into a spatial DataFrame.

    step_size triggers median downsampling (useful for matching validation grids).
    """
    vox_xs, vox_ys, vox_zs = vox_locs.T
    all_xs = np.unique(vox_xs)
    all_ys = np.unique(vox_ys)

    mask = vox_zs == altitude
    m_irrads = irrads[mask]
    if m_irrads.size == 0:
        raise ValueError(f'No irradiance data for specified altitude! ({altitude})')

    grid_irrads = m_irrads.reshape(int(n_voxels_x), -1)

    if step_size is not None:
        arr_step = int(step_size / voxel_size)
        ds_irrads = sample_2d_with_constrained_median(grid_irrads, arr_step)
        ds_xs = all_xs[::arr_step]
        ds_ys = all_ys[::arr_step]

    else:
        ds_irrads = grid_irrads
        ds_xs = all_xs
        ds_ys = all_ys

    ff_df = pd.DataFrame(ds_irrads, index=ds_ys, columns=ds_xs)

    if xmin is not None:
        ff_df = ff_df[ff_df.columns[ff_df.columns >= xmin]]
        ff_df = ff_df[ff_df.columns[ff_df.columns <= xmax]]
    if ymin is not None:
        ff_df = ff_df[ff_df.index >= ymin]
        ff_df = ff_df[ff_df.index <= ymax]

    return ff_df


def combine_datasets_for_parity(
    ff_sets: Tuple[NDArray[np.floating], ...], st_sets: Tuple[NDArray[np.floating], ...], skip_zeros: bool = False
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Concatenate FFIAM and SolTrace dataset tuples for parity comparison."""
    ff_comb = np.concatenate(ff_sets)
    st_comb = np.concatenate(st_sets)
    ff_len = len(ff_comb)
    st_len = len(st_comb)
    if ff_len > st_len:
        ff_comb = ff_comb[:st_len]
    elif st_len > ff_len:
        st_comb = st_comb[:ff_len]

    if skip_zeros:
        min_val = 1e-8
        zero_mask = (ff_comb < min_val) & (st_comb < min_val)
        print(f"{np.nansum(zero_mask)} 0-pairs out of {ff_len}")
        ff_comb = ff_comb[~zero_mask]
        st_comb = st_comb[~zero_mask]

    return ff_comb, st_comb


def combine_datasets_for_parity_aligned(
    ff_sets: Tuple[pd.Series, ...], st_sets: Tuple[pd.Series, ...], skip_zeros: bool = False
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Like combine_datasets_for_parity but aligns by x-coordinate index first.

    Ensures spatially co-located values are paired before concatenating.
    """
    ff_aligned = []
    st_aligned = []

    for ff_t, st_t in zip(ff_sets, st_sets):
        common_idx = ff_t.index.intersection(st_t.index)
        ff_aligned.append(ff_t.loc[common_idx].values)
        st_aligned.append(st_t.loc[common_idx].values)

    ff_comb = np.concatenate(ff_aligned)
    st_comb = np.concatenate(st_aligned)

    if skip_zeros:
        min_val = 1e-8
        zero_mask = (ff_comb < min_val) & (st_comb < min_val)
        print(f"{np.nansum(zero_mask)} 0-pairs out of {len(ff_comb)}")
        ff_comb = ff_comb[~zero_mask]
        st_comb = st_comb[~zero_mask]

    return ff_comb, st_comb


def prep_ffiam_data_for_soltrace(
    vox_locs: NDArray[np.floating],
    irrads: NDArray[np.floating],
    height: float,
    xlims: Tuple[float, float],
    ylims: Tuple[float, float],
    downsampling: int = 8,
) -> NDArray[np.floating]:
    """Extract and sort FFIAM irradiance at a given height for SolTrace comparison."""
    mask = vox_locs.T[2] == height
    data = np.copy(vox_locs[mask])
    data[:, 2] = irrads[mask]

    data = data[np.lexsort((data[:, 1], data[:, 0]))]  # sort by row then col to match SolTrace
    return data


def add_trendline(
    fig: go.Figure,
    row: int,
    col: int,
    xs: NDArray[np.floating],
    ys: NDArray[np.floating],
    ln_name: str = '',
    group: str = 'group1',
    grouptitle: str = 'Trendline',
) -> None:
    """Add a linear best-fit line with R² label to a plotly subplot."""
    slope, y_int, r_value, p_value, std_err = scipy.stats.linregress(xs, ys)
    r2 = r_value**2
    if ln_name in ['', None]:
        label = f"Best Fit: {slope:,.1f}x + {y_int:,.1f}, R<sup>2</sup> = {r2:.1f}"
    else:
        label = f"Best Fit ({ln_name}): {slope:,.1f}x + {y_int:,.1f}, R<sup>2</sup> = {r2:.1f}"
    bestfit_vals = slope * xs + y_int
    fig.add_trace(
        row=row,
        col=col,
        trace=go.Scatter(
            name=label,
            x=xs,
            y=bestfit_vals,
            mode='lines',
            line_color='royalblue',
            line_width=2,
            legendgroup=group,
            legendgrouptitle_text=grouptitle,
        ),
    )


def add_parity_data(
    fig: go.Figure,
    col: int,
    xs: Union[NDArray[np.floating], pd.Series],
    ys: Union[NDArray[np.floating], pd.Series],
    line_name: str,
) -> None:
    """Add scatter data with y=x reference line and best-fit trendline."""
    xmax = np.max(xs) * 1.2
    yx_lc = 'slategrey'
    fig.add_trace(
        row=1, col=col, trace=go.Scatter(x=[0, xmax], y=[0, xmax], mode='lines', line_color=yx_lc, showlegend=False)
    )
    fig.add_annotation(
        row=1, col=col, x=xmax, y=xmax, xref='x', yref='y', text='y=x', showarrow=False, xshift=-12, yshift=-25
    )

    if type(xs) is np.ndarray:
        add_trendline(fig, row=1, col=col, xs=xs, ys=ys, ln_name=line_name)
    else:
        add_trendline(fig, row=1, col=col, xs=xs.values, ys=ys.values, ln_name=line_name)

    fig.add_trace(
        row=1, col=col, trace=go.Scatter(mode='markers', x=xs, y=ys, showlegend=False, marker=dict(color='steelblue'))
    )


def estimate_voxel_memory(
    field_radius: int, min_height: int, max_height: int, voxel_size: int
) -> Tuple[int, float, bool]:
    """Return (num_voxels, memory_gb, is_valid) for the given grid configuration."""
    num_voxels_x = 2 * field_radius // voxel_size
    num_voxels_y = 2 * field_radius // voxel_size
    num_voxels_z = (max_height - min_height) // voxel_size + 1
    num_voxels = num_voxels_x * num_voxels_y * num_voxels_z
    memory_gb = num_voxels * 4 / GB  # float32
    is_valid = num_voxels <= MAX_VOXELS
    return num_voxels, memory_gb, is_valid


def print_voxel_config_table(field_radius: int, min_height: int, max_height: int) -> None:
    """Print voxel count and memory for common voxel sizes. Handy for choosing grid resolution."""
    print(f"Voxel configuration options for {field_radius}m radius, {min_height}-{max_height}m altitude:")
    print(f"{'Voxel Size':<11}| {'Voxels':<12}| {'Memory':<8}| Status")
    print("-" * 11 + "|" + "-" * 13 + "|" + "-" * 9 + "|" + "-" * 8)

    for voxel_size in [1, 2, 3, 4, 5, 8, 10]:
        num_voxels, memory_gb, is_valid = estimate_voxel_memory(field_radius, min_height, max_height, voxel_size)
        status = "OK" if is_valid else "TOO LARGE"
        print(f"{voxel_size}m{'':<9}| {num_voxels:<12,}| {memory_gb:.2f} GB  | {status}")
