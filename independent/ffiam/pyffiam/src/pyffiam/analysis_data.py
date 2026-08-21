# Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import numpy as np
from datetime import datetime

from pyffiam.app_config import *
from pyffiam import utils
from pyffiam.ffiam_types import CspSite, AimType, FieldLayout
from pyffiam.utils import slug


log = logging.getLogger(__name__)


@dataclass
class AnalysisData:
    """Central data container for a single analysis run: inputs, results, and output paths.

    Coordinates (x, y, z) = (east, north, up) with the tower at the origin. All distances in meters.
    Populated in two stages: construction sets inputs/geometry, add_results() sets computed metrics.
    """
    # Basic analysis parameters
    name: str
    slug: str
    created_at: datetime
    created_at_str: str
    year: int
    month: int
    day: int
    hour: float          # Decimal hours (e.g. 10.5 = 10:30 AM)
    analysis_dt: datetime
    analysis_dt_str: str

    # Site parameters
    site: 'CspSite'
    latitude: float      # Degrees north
    longitude: float     # Degrees east
    timezone: float      # UTC offset (hours)
    field_radius: float  # Heliostat field radius (m); sets x/y extent of voxel grid
    num_heliostats: int
    min_height: float    # Lower airspace bound (m AGL)
    max_height: float    # Upper airspace bound (m AGL)
    tower_height: float  # Receiver tower height (m); used for plot annotations
    reflectivity: float  # Heliostat mirror reflectivity (0–1)
    peak_dni: float      # Peak direct normal irradiance (W/cm²)
    sun_angle: float     # Solar disk half-angle (radians)
    slope_error: float   # Mirror slope error (radians)
    beta: float          # Beam spread coefficient (radians); controls flux falloff with distance
    ambient: float       # Ambient irradiance baseline added to every voxel (kW/m²)
    threshold: float     # Irradiance threshold for glare reporting (kW/m²); voxels below this are excluded

    # File paths
    heliostat_file: str          # CSV filename relative to pyffiam/data/ with heliostat positions
    facet_file: str              # CSV filename relative to pyffiam/data/ with facet offsets
    output_directory: Path       # Directory where plots, XLS, and logs are written

    # Aiming parameters
    aim_strategy: 'AimType'
    aim_parameters: Optional[np.ndarray]  # 3-element strategy-specific params (e.g. aim point in m for AimType.Point)
    aim_file: str                # CSV filename relative to pyffiam/data/ with per-heliostat aim data

    # Voxel grid parameters
    voxel_size: int = 2          # Voxel edge length (m)

    # Path parameters (optional UAS exposure analysis)
    has_paths: bool = False
    paths: Optional[List] = None         # List of polyline paths, each a sequence of (x, y, z) waypoints (m)
    path_speeds: Optional[List] = None   # Travel speed (m/s) for each path
    path_objects: Optional[List] = None  # PathData results, populated after analysis

    # Heliostat geometry (populated after C++ call)
    helio_locs: np.ndarray = None        # (num_heliostats, 3) heliostat positions (m)
    helio_aim_vs: np.ndarray = None      # (num_heliostats, 3) aim unit vectors
    helio_angles: np.ndarray = None      # (num_heliostats,) tracking rotation angles (degrees)
    num_facets: Optional[int] = None
    num_facet_cols: Optional[int] = None
    facet_width: Optional[float] = None  # Individual facet width (m)
    facet_height: Optional[float] = None # Individual facet height (m)
    facet_origins: np.ndarray = None     # (num_heliostats, num_facets, 3) facet center positions (m)
    helio_width: float = None            # Total heliostat width = facet_width * num_facet_cols (m)
    voxel_layout: tuple = None           # (nx, ny, nz) voxel grid dimensions
    num_voxels: int = None               # Total voxel count = nx * ny * nz
    # (num_voxels, 3) voxel center positions (m). Materialized lazily by the
    # voxel_locs property -- see the note there; at 1 m grids this array is tens
    # of GB and most runs never touch it.
    _voxel_locs: np.ndarray = None
    num_voxels_x: int = None
    num_voxels_y: int = None
    num_voxels_z: int = None
    voxel_area: float = None             # Voxel face area (m²)
    voxel_volume: float = None           # Voxel volume (m³)

    # Analysis results (populated by add_results())
    has_results: bool = False
    irrads: np.ndarray = None            # (num_voxels,) irradiance per voxel (kW/m²)
    total_irrad: float = None            # Sum of all voxel irradiances (kW/m²)
    peak_irrad: float = None             # Maximum single-voxel irradiance (kW/m²)
    movement: float = None               # Total heliostat tracking movement (degrees)

    threshold_mask: np.ndarray = None        # Boolean mask: True where irrads > threshold
    threshold_irrads: np.ndarray = None      # Irradiance values for above-threshold voxels (kW/m²)
    total_irrad_threshold: float = None      # Sum of above-threshold irradiances (kW/m²)
    n_glaring_voxels: int = None             # Count of voxels exceeding threshold
    has_glare: bool = None                   # True if any voxel exceeds threshold

    threshold_voxel_ids: np.ndarray = None   # Flat indices into irrads array for above-threshold voxels
    threshold_voxel_locs: np.ndarray = None  # (N, 3) Cartesian positions of above-threshold voxels (m)

    # Plot files (set in analysis function)
    en_heatmap: [str, None] = None
    en_heatmap_zoomed: [str, None] = None
    en_heatmap_gif: [str, None] = None
    eu_heatmap: [str, None] = None
    eu_heatmap_zoomed: [str, None] = None
    eu_heatmap_gif: [str, None] = None
    nu_heatmap: [str, None] = None
    nu_heatmap_zoomed: [str, None] = None
    nu_heatmap_gif: [str, None] = None

    agg_path_eu_heatmap: [str, None] = None

    xls_filename: [str, None] = None
    xls_file: Path = None

    ffiam_version: str = FFIAM_VERSION

    def __init__(self,
                 name: str,
                 year: int,
                 month: int,
                 day: int,
                 hour: float,  # Decimal hours (e.g. 10.5 = 10:30 AM)
                 threshold: float,

                 site: CspSite,
                 latitude: float,
                 longitude: float,
                 timezone: float,
                 field_radius: float,
                 num_helios: int,
                 min_height: float,
                 max_height: float,
                 tower_height: float,
                 num_facets: int,
                 num_facet_cols: int,
                 facet_width: float,
                 facet_height: float,

                 helio_file: str = "",
                 facet_file: str = "",
                 layout: int = FieldLayout.Grid.value,  # used only when no helio_file (0=grid, 1=radial)
                 aim_strat: 'AimType' = None,
                 aim_params: np.ndarray = None,
                 aim_file: str = "",

                 paths=None,
                 path_speeds=None,

                 voxel_size: int = 2,
                 refl=0.9,
                 peak_dni=0.1,
                 sun_angle=0.0093,
                 slope_error=0.0012,
                 beta=0.0094,
                 ambient: float = 0.0,          # Ambient irradiance baseline (kW/m²)
                 min_attenuation: float = 1.0,  # Hybrid blend floor (1.0 = disabled, 0.42 = UAS-calibrated)
                 irrad_exponent: float = 2.0,   # Beam concentration exponent (2.0 = default, 1.7 = UAS-calibrated)
                 n_rays_per_facet: float = 12.0, # Rays per facet (12 = 9 core + 3 outer, 9 = core only)
                 flux_correction_scale: float = 1.0,  # Flux correction scale (1.0 = full, 0.0 = disabled)
                 pre_focal_scale: float = 1.0,  # Pre-focal attenuation scale (1.0 = symmetric default, 0.6 = UAS-calibrated)

                 app_result_dir: Path = None,
                 verbose: bool = False
                 ):
        """Validate inputs, resolve preset site configs, and compute voxel grid geometry.

        Args:
            name: Display name; overridden by preset RunId when site != CspSite.Custom.
            threshold: Irradiance reporting threshold (kW/m²).
            site: Preset site enum; if not CspSite.Custom, most field/heliostat params are loaded from JSON.
            field_radius: Heliostat field radius (m); sets x/y extent of voxel grid.
            num_helios: Number of heliostats; ignored when helio_file is provided.
            min_height: Lower airspace bound (m AGL).
            max_height: Upper airspace bound (m AGL).
            tower_height: Receiver tower height (m).
            helio_file: CSV filename (relative to pyffiam/data/) with heliostat positions.
            facet_file: CSV filename (relative to pyffiam/data/) with facet offsets.
            aim_strat: Aiming strategy enum (AimType.Point, AimType.Horizontal, etc.).
            aim_params: 3-element strategy-specific parameter array (e.g. aim point in m for AimType.Point).
            aim_file: CSV filename (relative to pyffiam/data/) with per-heliostat aim data.
            voxel_size: Voxel edge length (m); must be between MIN_VOXEL_SIZE and MAX_VOXEL_SIZE.
            beta: Beam spread coefficient (radians).
            ambient: Ambient irradiance baseline added to every voxel (kW/m²).
            min_attenuation: Hybrid blend floor for pre/post-focal attenuation.
            irrad_exponent: Beam concentration exponent controlling peak sharpness.
            n_rays_per_facet: Rays traced per facet.
            flux_correction_scale: Scale factor for flux correction.
            pre_focal_scale: Pre-focal attenuation scale.
            app_result_dir: Root directory for output files; defaults to pyffiam/out/ if None.
        """
        self.site = site
        self.created_at = datetime.now()
        self.created_at_str = self.created_at.strftime("%Y%m%d%H%M")

        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        # round() avoids floating-point precision issues (e.g., 8/60*60 = 7.999...)
        hour_int = int(hour)
        minute_int = round((hour - hour_int) * 60)
        self.analysis_dt = datetime(year, month, day, hour_int, minute=minute_int)
        self.analysis_dt_str = self.analysis_dt.strftime('%Y%m%d%H%M')

        if self.site != CspSite.Custom:
            presets_dict = utils.load_preset_sites_from_json()
            if self.site not in presets_dict:
                raise ValueError(f"Site name '{site}' not found in site config data.")

            site_data = presets_dict[site]  # JSON entry must match enum str name
            name = site_data['RunId']
            helio_data = site_data['HelioDesign']
            field_data = site_data['Field']
            dt_data = site_data['DateTime']
            aim_data = site_data['AimStrategy']

            latitude = field_data['Latitude']
            longitude = field_data['Longitude']
            timezone = field_data['Timezone']
            field_radius = field_data['FieldRadius']
            num_helios = field_data['NumHelios']
            tower_height = field_data['TowerHeight']
            max_height = field_data['MaxHeight']
            min_height = field_data['MinHeight']
            helio_file = field_data['HelioCoordinateFile']
            # Generation layout (grid vs radial) for sites with no heliostat CSV.
            layout = FieldLayout.from_str(field_data.get('Layout', 'grid')).value

            facet_file = helio_data['FacetFile']
            num_facets = helio_data['NumFacets']
            num_facet_cols = helio_data['NumCols']
            facet_width = helio_data['FacetWidth']
            facet_height = helio_data['FacetHeight']

            # Only use preset aim file if not provided one by user
            if aim_file in ["", None] and 'File' in aim_data:
                aim_file = aim_data['File']

        self.name = name
        self.slug = slug(name)

        if app_result_dir is None:  # for tests
            app_result_dir = Path(__file__).parent.resolve().joinpath('out')
            app_result_dir.mkdir(parents=True, exist_ok=True)

        session_fname = f"{self.slug}_on{self.analysis_dt_str}_created{self.created_at_str}"
        self.output_directory = app_result_dir.joinpath(session_fname)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.xls_filename = f"{self.slug}_on{self.analysis_dt_str}_created{self.created_at_str}.xlsx"
        self.xls_file = self.output_directory.joinpath(self.xls_filename)

        self.threshold = threshold
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.field_radius = field_radius
        self.num_heliostats = num_helios
        self.min_height = int(min_height)
        self.max_height = int(max_height)
        self.tower_height = tower_height

        self.num_facets = num_facets
        self.num_facet_cols = num_facet_cols
        self.facet_width = facet_width
        self.facet_height = facet_height
        self.helio_width = self.facet_width * self.num_facet_cols

        self.reflectivity = refl
        self.peak_dni = peak_dni
        self.sun_angle = sun_angle
        self.slope_error = slope_error
        self.beta = beta
        self.ambient = ambient
        self.min_attenuation = min_attenuation
        self.irrad_exponent = irrad_exponent
        self.n_rays_per_facet = n_rays_per_facet
        self.flux_correction_scale = flux_correction_scale
        self.pre_focal_scale = pre_focal_scale

        # C++ prepends data/ to these filenames, so store basename only
        self.heliostat_file = helio_file
        self.facet_file = facet_file
        self.layout = layout
        self.aim_file = aim_file
        self.aim_strategy = aim_strat
        self.aim_parameters = aim_params

        self.voxel_size = int(voxel_size)

        if not (MIN_VOXEL_SIZE <= self.voxel_size <= MAX_VOXEL_SIZE):
            raise ValueError(
                f"voxel_size must be between {MIN_VOXEL_SIZE} and {MAX_VOXEL_SIZE}m "
                f"({self.voxel_size}m provided)"
            )

        self.voxel_area = voxel_size ** 2
        self.voxel_volume = voxel_size ** 3
        self.num_voxels_z = int((self.max_height - self.min_height) / self.voxel_size + 1)
        self.num_voxels_y = int(2 * self.field_radius / self.voxel_size)
        self.num_voxels_x = int(2 * self.field_radius / self.voxel_size)
        self.voxel_layout = (self.num_voxels_x, self.num_voxels_y, self.num_voxels_z)
        self.num_voxels = self.num_voxels_x * self.num_voxels_y * self.num_voxels_z

        if self.num_voxels > MAX_VOXELS:
            memory_required_gb = self.num_voxels * 4 / GB
            memory_available_gb = VOXEL_POOL_SIZE / GB
            raise ValueError(
                f"Configuration creates {self.num_voxels:,} voxels requiring ~{memory_required_gb:.1f}GB, "
                f"exceeding limit of {MAX_VOXELS:,} voxels (~{memory_available_gb:.0f}GB). "
                f"Reduce field_radius ({self.field_radius}m), altitude range "
                f"({self.min_height}-{self.max_height}m), or increase voxel_size ({self.voxel_size}m)."
            )

        self._voxel_locs = None  # see the voxel_locs property

        self.has_paths = paths is not None
        self.paths = [] if paths is None else paths
        self.path_speeds = [] if paths is None else path_speeds
        self.path_objects = []

        error_msg = ""
        if self.num_heliostats > MAX_NUM_HELIOSTATS:
            error_msg = f"FFIAM v3.0 can assess up to 6,000 heliostats but {self.num_heliostats} provided."

        if self.max_height > MAX_HEIGHT:
            error_msg = f"FFIAM v3.0 can assess airspace with vertical span <= 310 m ({self.max_height} m provided)"

        if self.timezone < -12 or self.timezone > 14:
            error_msg = f"Enter a valid timezone offset ({self.timezone} provided)."

        if self.field_radius > MAX_FIELD_RADIUS:
            error_msg = f"FFIAM v3.0 can assess field with effective radius <= 1600 m. ({self.field_radius} entered)"

        if error_msg:
            log.error(error_msg)
            raise ValueError(error_msg)

        # Use absolute path so validation doesn't depend on cwd
        data_dir = Path(__file__).parent.resolve() / 'data'

        if self.heliostat_file != "":
            helio_file_fpath = data_dir / self.heliostat_file
            if not helio_file_fpath.exists():
                log.error(f"File not found ({self.heliostat_file}). If providing a heliostat data-file, place it in the 'data' subdirectory.")
                raise FileNotFoundError(helio_file_fpath)

        if self.aim_file != "":
            aim_file_fpath = data_dir / self.aim_file
            if not aim_file_fpath.exists():
                log.error(f"File not found ({self.aim_file}). If providing a aim data-file, place it in the 'data' subdirectory.")
                raise FileNotFoundError(aim_file_fpath)

        if self.facet_file != "":
            facet_file_fpath = data_dir / self.facet_file
            if not facet_file_fpath.exists():
                log.error(f"File not found ({self.facet_file}). If providing a facet data-file, place it in the 'data' subdirectory.")
                raise FileNotFoundError(facet_file_fpath)

        if verbose:
            import pprint
            print("Analysis parameters:")
            pprint.pprint(vars(self))


    @property
    def voxel_locs(self) -> np.ndarray:
        """(num_voxels, 3) voxel center positions (m), computed on first access.

        This is float64, so it costs 24 bytes per voxel -- six times the float32
        irradiance array it describes. A 1 km site at 1 m voxels is 788M voxels,
        i.e. ~18 GB for this array alone, which is why it is not built eagerly:
        analysis, plotting and Excel export all work from `threshold_voxel_ids`
        and only ever need the above-threshold subset. Prefer
        `utils.get_voxel_locs_from_indexes(ids, ...)` over touching this.
        """
        if self._voxel_locs is None:
            self._voxel_locs = utils.compute_vox_locs(
                self.num_voxels, self.voxel_size, self.field_radius, self.min_height
            )
        return self._voxel_locs

    @voxel_locs.setter
    def voxel_locs(self, value: np.ndarray) -> None:
        self._voxel_locs = value


    def add_results(self,
                    helio_locs,
                    helio_aim_vs,
                    helio_angles,
                    facet_origins,
                    irrads,
                    movement,
                    ):
        """Populate result fields from raw C++ output arrays and compute derived summary stats."""
        self.helio_locs = helio_locs
        self.helio_aim_vs = helio_aim_vs
        self.helio_angles = helio_angles
        self.facet_origins = facet_origins

        self.irrads = irrads
        self.total_irrad = int(np.nansum(irrads))
        self.peak_irrad = np.nanmax(irrads)
        self.movement = movement

        self.threshold_mask = irrads > self.threshold
        self.n_glaring_voxels = int(np.nansum(self.threshold_mask))
        self.threshold_irrads = irrads[self.threshold_mask]
        self.total_irrad_threshold = int(np.nansum(self.threshold_irrads))
        self.has_glare = self.n_glaring_voxels > 0

        self.threshold_voxel_ids = np.where(self.threshold_mask)[0]
        self.threshold_voxel_locs = utils.get_voxel_locs_from_indexes(self.threshold_voxel_ids,
                                                                      self.voxel_size,
                                                                      self.field_radius,
                                                                      self.min_height)

        self.has_results = True

    def add_path_data(self, path_data: 'PathData'):
        self.path_objects.append(path_data)

    @property
    def has_plots(self):
        return self.en_heatmap is not None


class PathData:
    """Irradiance results along a single polyline path."""
    id: int
    name: str
    points: np.ndarray
    interpolated_points: np.ndarray
    speed: float
    num_points: int
    time_per_voxel: float

    voxel_ids: np.ndarray
    voxel_locs: np.ndarray
    irrads: np.ndarray
    total_time: float
    exposures: np.ndarray
    exposures_over_time: np.ndarray
    total_exposure: float

    output_directory: Path
    en_heatmap: Path
    en_heatmap_zoomed: Path
    eu_heatmap: Path
    eu_heatmap_zoomed: Path
    exposure_cumsum_plot: Path

    def __init__(self, id, points, speed, parent_dir, voxel_size=2):
        """Interpolate waypoints at voxel-sized steps and create the output subdirectory."""
        self.id = id
        self.name = f"Path_{self.id}"
        self.points = points
        if type(self.points) is list:
            self.points = np.array(self.points)

        self.speed = speed
        self.time_per_voxel = voxel_size / self.speed
        self.num_points = len(self.points)

        self.output_directory = parent_dir.joinpath(self.name)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # One interpolated point per voxel so every voxel along the path is sampled
        interp_points = []
        for j in range(len(self.points) - 1):
            p1 = self.points[j]
            p2 = self.points[j + 1]
            new_points = utils.interpolate_3d(p1, p2, step_size=voxel_size)
            interp_points += new_points

        self.interpolated_points = np.array(interp_points)

    def set_results(self, path_irradiances, voxel_ids, voxel_locs):
        """Store irradiance arrays and compute cumulative radiant exposure along the path."""
        self.irrads = path_irradiances
        self.voxel_ids = voxel_ids
        self.voxel_locs = voxel_locs

        # quantity is "radiant exposure" in units of W*s/m2 (recall Watt == J/s)
        self.exposures = self.irrads * self.time_per_voxel
        self.exposures_over_time = np.cumsum(self.exposures)
        self.total_exposure = np.nansum(self.exposures)


    def set_plots(self,
                  en_heatmap, en_heatmap_zoomed,
                  eu_heatmap, eu_heatmap_zoomed,
                  cumsum_plot,
                  ):
        """Store generated plot file paths for this path."""
        self.en_heatmap = en_heatmap
        self.en_heatmap_zoomed = en_heatmap_zoomed
        self.eu_heatmap = eu_heatmap
        self.eu_heatmap_zoomed = eu_heatmap_zoomed
        self.exposure_cumsum_plot = cumsum_plot
