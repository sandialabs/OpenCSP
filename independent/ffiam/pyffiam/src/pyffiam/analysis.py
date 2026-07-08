from pathlib import Path
import logging

import numpy as np
import xlsxwriter

from pyffiam.app_config import app_name, app_author
from pyffiam.utils import slug
from pyffiam import plots, utils, xls
from pyffiam.ffiam_types import AimType, CspSite, Direction
from pyffiam.analysis_data import AnalysisData, PathData
from pyffiam.cpp_interface import FFIAMLibrary, AnalysisParams

from platformdirs import user_data_dir
import os
from typing import List, Tuple, Optional

app_data_fpath = user_data_dir(app_name, app_author)
app_data_path = Path(app_data_fpath)

os.makedirs(app_data_fpath, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(os.path.join(app_data_fpath, 'ffiam.log'))
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

log.addHandler(file_handler)
log.addHandler(console_handler)

app_dir = Path(__file__).parent.resolve()


def _generate_all_plots(als: AnalysisData, create_gifs: bool) -> None:
    """Create and store EN, EU, and NU profile heatmaps (and optional GIFs) on the AnalysisData object."""
    # (dim1, dim2, suffix, needs_tower_height)
    dimension_pairs = [
        (Direction.East, Direction.North, 'en', False),
        (Direction.East, Direction.Up, 'eu', True),
        (Direction.North, Direction.Up, 'nu', True),
    ]

    for dim1, dim2, suffix, needs_tower in dimension_pairs:
        plot_kwargs = dict(
            hel_locs=als.helio_locs,
            vox_locs=als.threshold_voxel_locs,
            irrads=als.threshold_irrads,
            dim1=dim1,
            dim2=dim2,
            site=als.name,
            threshold=als.threshold,
            vox_size=als.voxel_size,
            hel_size=als.helio_width,
            aim_strat=als.aim_strategy,
            aim_params=als.aim_parameters,
            create_gif=create_gifs,
            out_dir=als.output_directory,
        )

        if needs_tower:
            plot_kwargs['tower_h'] = als.tower_height

        heatmap, heatmap_zoomed, heatmap_gif = plots.create_profile_plots(**plot_kwargs)

        setattr(als, f'{suffix}_heatmap', heatmap)
        setattr(als, f'{suffix}_heatmap_zoomed', heatmap_zoomed)
        setattr(als, f'{suffix}_heatmap_gif', heatmap_gif)


def _generate_excel(als: AnalysisData, create_gifs: bool) -> xlsxwriter.Workbook:
    """Returns open workbook; caller must close."""
    workbook = xlsxwriter.Workbook(als.xls_file.as_posix())
    xls.initialize(workbook)
    xls.write_results(workbook, als, has_gifs=create_gifs)
    xls.write_heliostat_positions(workbook, als)
    xls.write_irrad_voxel_positions(workbook, als)
    return workbook


def analysis(
    name: str = "",
    year: int = 2025,
    month: int = 6,
    day: int = 21,
    hour: float = 12.0,  # Supports fractional hours (e.g., 10.5 = 10:30 AM)
    threshold: float = 4,
    site: CspSite = CspSite.Custom,
    lat: float = None,
    lng: float = None,
    timezone: float = -7,
    field_r: float = 600,
    n_helios: int = 2000,
    min_height: float = 4,
    max_height: float = 100,
    tower_height: float = 100,
    helio_file: str = "",
    facet_file: str = "",
    layout: int = 0,  # generation layout when no helio_file: 0=grid, 1=radial (FieldLayout)
    n_facets: int = 35,
    n_facet_cols: int = 5,
    facet_w: float = 1.8,
    facet_h: float = 1.8,
    aim_strat: AimType = AimType.Point,
    aim_params: np.array = None,
    aim_file: str = "",
    paths=None,
    path_speeds=None,
    create_xls=True,
    create_gifs=True,
    create_plots=True,
    open_output_dir=True,
    voxel_size: int = 2,
    beta: float = 0.0094,  # Beam spread coefficient (radians)
    ambient: float = 0.0,  # Ambient irradiance baseline [kW/m²] to add to all voxels
    min_attenuation: float = 1.0,  # Hybrid blend floor (1.0 = disabled/original, 0.42 = UAS-calibrated)
    irrad_exponent: float = 2.0,  # Beam concentration exponent (2.0 = original, 1.7 = UAS-calibrated)
    n_rays_per_facet: float = 12.0,  # Rays per facet (12 = 9 core + 3 outer, 9 = core only)
    flux_correction_scale: float = 1.0,  # Flux correction scale (1.0 = full, 0.0 = disabled)
    pre_focal_scale: float = 1.0,  # Pre-focal attenuation scale (1.0 = original symmetric, 0.6 = UAS-calibrated)
    force_cpu: bool = False,  # Use the CPU backend even when CUDA is available; also honored via FFIAM_FORCE_CPU=1
    verbose=False,
):
    """Run a full FFIAM irradiance analysis and write plots/XLS output.

    Coordinates (x, y, z) = (east, north, up) with tower at origin. Distances in meters.

    Args:
        name: Display name for this analysis run; defaults to site enum name.
        year: Analysis year.
        month: Analysis month (1–12).
        day: Analysis day of month.
        hour: Solar time in decimal hours (e.g. 10.5 = 10:30 AM).
        threshold: Irradiance threshold for glare reporting (kW/m²); voxels below this are excluded.
        site: Preset CSP site enum (CspSite.NSTTF, CspSite.Radial, CspSite.SampleV2, etc.). Use CspSite.Custom to supply params manually.
        lat: Site latitude (°); required when site=CspSite.Custom.
        lng: Site longitude (°); required when site=CspSite.Custom.
        timezone: UTC offset in hours (e.g. -7 for US Mountain Standard Time).
        field_r: Heliostat field radius (m); sets the x/y extent of the voxel grid.
        n_helios: Number of heliostats; ignored when a heliostat CSV file is provided.
        min_height: Lower bound of analysis airspace (m AGL).
        max_height: Upper bound of analysis airspace (m AGL).
        tower_height: Receiver tower height (m); used for plot annotations.
        helio_file: CSV filename (relative to pyffiam/data/) with heliostat positions; leave blank to use n_helios.
        facet_file: CSV filename (relative to pyffiam/data/) with facet offsets; leave blank to use default layout.
        n_facets: Total facets per heliostat.
        n_facet_cols: Number of facet columns per heliostat.
        facet_w: Individual facet width (m).
        facet_h: Individual facet height (m).
        aim_strat: Aiming strategy enum (AimType.Point, AimType.Horizontal, etc.).
        aim_params: Strategy-specific aim parameters as a 3-element array (e.g. [x, y, z] aim point in meters for AimType.Point).
        aim_file: CSV filename (relative to pyffiam/data/) with per-heliostat aim data.
        paths: List of polyline paths, each a sequence of (x, y, z) waypoints (m), for UAS exposure analysis.
        path_speeds: Travel speed (m/s) for each path in `paths`.
        create_xls: Write an Excel summary workbook to the output directory.
        create_gifs: Generate animated GIF plots (slow; requires kaleido).
        create_plots: Generate PNG heatmap plots.
        open_output_dir: Open the output folder in Explorer after completion (Windows only).
        voxel_size: Voxel edge length (m); smaller values increase resolution and memory use.
        beta: Beam spread coefficient (radians); controls flux falloff with distance.
        ambient: Ambient irradiance baseline added to every voxel (kW/m²).
        min_attenuation: Hybrid blend floor for pre/post-focal attenuation (1.0 = disabled, 0.42 = UAS-calibrated).
        irrad_exponent: Beam concentration exponent controlling peak sharpness (2.0 = default, 1.7 = UAS-calibrated).
        n_rays_per_facet: Rays traced per facet (12 = 9 core + 3 outer ring, 9 = core only).
        flux_correction_scale: Scale factor for flux correction (1.0 = full, 0.0 = disabled).
        pre_focal_scale: Pre-focal attenuation scale (1.0 = symmetric default, 0.6 = UAS-calibrated).
        force_cpu: Use the CPU backend even when CUDA is available (useful for testing on GPU
                   machines, or when running where the GPU lib can't be loaded). Equivalent to
                   FFIAM_FORCE_CPU=1. The CPU backend is feature-equivalent but slower.

    Returns:
        AnalysisData populated with all inputs, computed irradiance arrays, and output file paths.
    """
    if name in ["", None]:
        name = site.name

    log.warning(f"Initializing FFIAM Analysis for: {name}")

    if verbose:
        log.setLevel(logging.DEBUG)
        cwd = Path(os.getcwd())
        log.info(f"CWD: {cwd}")

    if aim_params is None:
        aim_params = np.array([0, 0, 100])

    if paths is None:
        paths = []
        path_speeds = []
    else:
        paths = np.array(paths)
        path_speeds = np.array(path_speeds)
        # must be list of list of points, even if only one path given
        if len(paths.shape) != 3:
            paths = paths[None]

    als = AnalysisData(
        name=name,
        year=year,
        month=month,
        day=day,
        hour=hour,
        threshold=threshold,
        site=site,
        latitude=lat,
        longitude=lng,
        timezone=timezone,
        field_radius=field_r,
        num_helios=n_helios,
        min_height=min_height,
        max_height=max_height,
        tower_height=tower_height,
        num_facets=n_facets,
        num_facet_cols=n_facet_cols,
        facet_width=facet_w,
        facet_height=facet_h,
        helio_file=helio_file,
        aim_file=aim_file,
        facet_file=facet_file,
        layout=layout,
        aim_strat=aim_strat,
        aim_params=aim_params,
        paths=paths,
        path_speeds=path_speeds,
        voxel_size=voxel_size,
        beta=beta,
        ambient=ambient,
        min_attenuation=min_attenuation,
        irrad_exponent=irrad_exponent,
        n_rays_per_facet=n_rays_per_facet,
        flux_correction_scale=flux_correction_scale,
        pre_focal_scale=pre_focal_scale,
        app_result_dir=app_data_path,
        verbose=verbose,
    )

    als = lib_ffiam_analysis(
        als, create_xls=create_xls, create_gifs=create_gifs, create_plots=create_plots, force_cpu=force_cpu
    )

    if open_output_dir:
        import platform

        if platform.system() == "Windows":
            os.startfile(als.output_directory)

    return als


def lib_ffiam_analysis(
    als: AnalysisData, create_xls=False, create_gifs=False, create_plots=True, force_cpu: bool = False
):
    """Call the C++ library with a pre-built AnalysisData object and populate results in place."""
    params = AnalysisParams(
        year=als.year,
        month=als.month,
        day=als.day,
        hour=als.hour,
        latitude=als.latitude,
        longitude=als.longitude,
        timezone=als.timezone,
        field_radius=als.field_radius,
        min_height=als.min_height,
        max_height=als.max_height,
        tower_height=als.tower_height,
        heliostat_file=als.heliostat_file,
        num_heliostats=als.num_heliostats,
        facet_file=als.facet_file,
        num_facets=als.num_facets,
        num_facet_cols=als.num_facet_cols,
        facet_width=als.facet_width,
        facet_height=als.facet_height,
        aim_strategy=int(als.aim_strategy.value),
        aim_param_0=als.aim_parameters[0],
        aim_param_1=als.aim_parameters[1],
        aim_param_2=als.aim_parameters[2],
        aim_file=als.aim_file,
        reflectivity=als.reflectivity,
        peak_dni=als.peak_dni,
        beta=als.beta,
        voxel_size=als.voxel_size,
        ambient=als.ambient,
        min_attenuation=als.min_attenuation,
        irrad_exponent=als.irrad_exponent,
        n_rays_per_facet=als.n_rays_per_facet,
        flux_correction_scale=als.flux_correction_scale,
        pre_focal_scale=als.pre_focal_scale,
        num_voxels=als.num_voxels,
        layout=als.layout,
    )

    try:
        ffiam_lib = FFIAMLibrary(force_cpu=force_cpu)
        raw_results = ffiam_lib.run_analysis(params)
    except OSError as err:
        log.error(f'OSError encountered during analysis: \n{str(err)}')
        raise RuntimeError(f'OSError during C++ analysis: {err}') from err
    except Exception as err:
        log.error(f'Unknown error encountered during analysis: \n{str(err)}')
        raise RuntimeError(f'Error during C++ analysis: {err}') from err

    log.warning(f"Analysis complete. Processing data...")

    facet_data = raw_results.facet_data.reshape((als.num_heliostats, 2 * als.num_facets, 3))
    facet_origins = facet_data[:, 0::2]

    log.info(f'Expected # voxels: {als.num_voxels:,.0f} ({als.num_voxels_x}, {als.num_voxels_y}, {als.num_voxels_z})')
    log.info(f'Helio locs: {raw_results.helio_locs.shape}')

    als.add_results(
        helio_locs=raw_results.helio_locs,
        helio_aim_vs=raw_results.helio_aim_vs,
        helio_angles=raw_results.helio_angles,
        facet_origins=facet_origins,
        irrads=raw_results.irrads,
        movement=raw_results.movement,
    )

    log.warning(
        f'Python: total irradiance {als.total_irrad:,.0f}, '
        f'Total irrad above threshold {als.total_irrad_threshold:,d}, '
        f'Max irradiance {als.peak_irrad:,.0f}, '
        f'Impacted voxels {als.n_glaring_voxels:,d}'
    )

    if create_plots and als.has_glare:
        log.info("Creating plots. Warning: if enabled, GIF creation may take awhile")
        _generate_all_plots(als, create_gifs)
        log.info("Plots complete")

    if create_xls:
        log.info("Creating XLS data-file")
        workbook = _generate_excel(als, create_gifs)
    else:
        workbook = None

    if als.has_paths:
        log.info("Beginning path analyses")

        for i, path_vertices in enumerate(als.paths):
            path_id = i + 1
            log.info(f"\nAssessing path {path_id}")

            path = PathData(path_id, path_vertices, als.path_speeds[i], parent_dir=als.output_directory)
            als.path_objects.append(path)

            point_voxel_ids = utils.get_voxel_indexes_from_locs(
                path.interpolated_points, als.voxel_size, als.field_radius, als.min_height
            )
            point_voxel_ids = point_voxel_ids.astype(int)
            path_voxel_locs = utils.get_voxel_locs_from_indexes(
                point_voxel_ids, als.voxel_size, als.field_radius, als.min_height
            )

            path_irrads = als.irrads[point_voxel_ids]

            path.set_results(path_irradiances=path_irrads, voxel_ids=point_voxel_ids, voxel_locs=path_voxel_locs)

            # speed = als.path_speeds[i]
            # quantity is "radiant exposure" in units of W*s/m2 (recall Watt == J/s)
            # exposures = path_irrads * time_per_voxel
            # exposure_over_time = np.cumsum(exposures)
            # total_exposure = np.nansum(exposures)

            if create_plots:
                log.info(f"Creating plots for path {path_id}")
                ne_filename = f"{slug(als.name)}_path{path_id}_NE"
                path_plot_en_fpath, path_plot_en_zoom_fpath, _ = plots.create_profile_plots(
                    als.helio_locs,
                    path_voxel_locs,
                    path_irrads,
                    dim1=Direction.East,
                    dim2=Direction.North,
                    site=als.name,
                    threshold=als.threshold,
                    vox_size=als.voxel_size,
                    hel_size=als.helio_width,
                    aim_strat=als.aim_strategy,
                    aim_params=als.aim_parameters,
                    create_gif=False,
                    filename=ne_filename,
                    out_dir=path.output_directory,
                )

                eu_filename = f"{slug(als.name)}_path{path_id}_EU"
                path_plot_eu_fpath, path_plot_eu_zoom_fpath, _ = plots.create_profile_plots(
                    als.helio_locs,
                    path_voxel_locs,
                    path_irrads,
                    dim1=Direction.East,
                    dim2=Direction.Up,
                    site=als.name,
                    threshold=als.threshold,
                    vox_size=als.voxel_size,
                    hel_size=als.helio_width,
                    aim_strat=als.aim_strategy,
                    aim_params=als.aim_parameters,
                    tower_h=als.tower_height,
                    create_gif=False,
                    filename=eu_filename,
                    out_dir=path.output_directory,
                )

                cumsum_plot = plots.create_path_cumsum_plot(
                    time_per_voxel=path.time_per_voxel,
                    cumulative_exposures=path.exposures,
                    site=als.name,
                    path_id=path.id,
                    out_dir=path.output_directory,
                )

                path.set_plots(
                    path_plot_en_fpath,
                    path_plot_en_zoom_fpath,
                    path_plot_eu_fpath,
                    path_plot_eu_zoom_fpath,
                    cumsum_plot,
                )

            else:
                log.info(f"Plot creation disabled - skipping path plots")
                path.set_plots(None, None, None, None, None)

            log.info(f"Path {path_id} complete")

            if create_xls:
                xls.write_path_data(workbook, als, path)

        agg_voxel_locs = []
        agg_irrads = []
        for path in als.path_objects:
            agg_voxel_locs.append(path.voxel_locs)
            agg_irrads.append(path.irrads)

        agg_voxel_locs = np.array(agg_voxel_locs).reshape(-1, 3)
        agg_irrads = np.array(agg_irrads).flatten()
        eu_filename = f"aggregate_path_irradiance_east_up"
        agg_path_plot_eu, _, _ = plots.create_profile_plots(
            als.helio_locs,
            agg_voxel_locs,
            agg_irrads,
            dim1=Direction.East,
            dim2=Direction.Up,
            site=als.name,
            threshold=als.threshold,
            vox_size=als.voxel_size,
            hel_size=als.helio_width,
            aim_strat=als.aim_strategy,
            aim_params=als.aim_parameters,
            tower_h=als.tower_height,
            create_zoomed=False,
            create_gif=False,
            filename=eu_filename,
            out_dir=als.output_directory,
        )
        als.agg_path_eu_heatmap = agg_path_plot_eu
        log.info(f"Aggregate path plots complete")

    else:
        log.info("No paths provided - skipping path calculations")

    if create_xls:
        workbook.close()

    log.warning('\nFFIAM analysis complete! Exiting...\n\n')
    return als
