"""Class used to display/save the suite of standard output plots after measuring a CSP Optic object."""

import configparser
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import h5py
import opencsp.common.lib.render.view_spec as vs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional, Tuple

import opencsp.common.lib.render_control.RenderControlAxis as rca
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
import opencsp.common.lib.csp.RayTrace as rt
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.geometry.Resolution import Resolution
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlEnclosedEnergy as rcee
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.string_tools as st


@dataclass
class _OptionsFileOutput:
    to_save: bool = False
    """Flag to save figures or not. (default False)"""
    output_dir: str = ''
    """Output path to save directory. (default '')"""
    save_dpi: int = 200
    """Dots Per Inch (DPI) of saved figures. (default 200)"""
    save_format: str = 'png'
    """Saved figure format. (default 'png')"""
    close_after_save: bool = False
    """To close figures after save. (default False)"""
    number_in_name: bool = True
    """To keep figure number in save name. (default True)"""
    file_prefix: str = ''
    """String to prefix each output file, including separator"""


@dataclass
class _OptionsSave:
    curvature_deviation: bool = True
    """Save curvature deviation data in HDF5 format"""
    slope_deviation: bool = True
    """Save slope deviation data in HDF5 format"""
    enclosed_energy: bool = True
    """Save enclosed energy data in HDF5 format"""
    curvature: bool = True
    """Save curvature data for both measured and reference in HDF5 format"""
    slope: bool = True
    """Save slope data for both measured and reference in HDF5 format"""


@dataclass
class _OptionsSliceOutput:
    to_plot: bool = True
    """Plot and save slice output data"""
    slice_x_loc: List[float] = field(default_factory=lambda: [0])
    """X slice locations for slice output data"""
    slice_y_loc: List[float] = field(default_factory=lambda: [0])
    """Y slice locations for slice output data"""
    plot_x_slope_offset: float = 0
    """Plot offset for slope X slice output"""
    plot_y_slope_offset: float = 0
    """Plot offset for slope Y slice output"""
    plot_x_slope_deviation_offset: float = 0
    """Plot offset for slope deviation X slice output"""
    plot_y_slope_deviation_offset: float = 0
    """Plot offset for slope deviation Y slice output"""
    plot_x_curvature_offset: float = 0
    """Plot offset for curvature X slice output"""
    plot_y_curvature_offset: float = 0
    """Plot offset for curvature Y slice output"""
    plot_x_curvature_deviation_offset: float = 0
    """Plot offset for curvature deviation X slice output"""
    plot_y_curvature_deviation_offset: float = 0
    """Plot offset for curvature deviation Y slice output"""
    uncertainty_to_plot: bool = False
    """Plot uncertainty bars on slice output"""
    xlim: Optional[Tuple[float, float]] = None
    """Plot xlim on X slice output"""
    ylim: Optional[Tuple[float, float]] = None
    """Plot xlim on Y slice output"""


@dataclass
class _OptionsSlopeVis:
    resolution: float = 0.01
    """Plot x/y sample resolution (meters) (default 0.01)"""
    clim: float | tuple[float, float, float] = 5
    """Sets colorbar limits (mrad). Plot limits set to [-clim, clim] for x or y slope plots and [0, clim] for slope xy plots.
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 5)"""
    quiver_density: float | tuple[float, float, float] = 0.1
    """The density of the quiver arrows (meters).
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 0.1)"""
    quiver_scale: float | tuple[float, float, float] = 25
    """The scale of the quiver arrows.
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 25)"""
    quiver_color: str | tuple[str, str, str] = 'white'
    """The color of the quiver arrows.
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 'white')"""
    to_plot: bool = True
    """Flag to produce plots or not. (default True)"""


@dataclass
class _OptionsSlopeDeviationVis:
    resolution: float = 0.01
    """Plot x/y sample resolution (meters) (default 0.01)"""
    clim: float | tuple[float, float, float] = 5
    """Sets colorbar limits (mrad). Plot limits set to [-clim, clim] for x or y slope plots and [0, clim] for slope xy plots.
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 5)"""
    quiver_density: float | tuple[float, float, float] = 0.1
    """The density of the quiver arrows (meters).
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 0.1)"""
    quiver_scale: float | tuple[float, float, float] = 25
    """The scale of the quiver arrows.
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 25)"""
    quiver_color: str | tuple[str, str, str] = 'white'
    """The color of the quiver arrows.
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 'white')"""
    to_plot: bool = True
    """Flag to produce plots or not. (default True)"""


@dataclass
class _OptionsCurvatureVis:
    resolution: float = 0.01
    """Plot x/y sample resolution (meters) (default 0.01)"""
    clim: float | tuple[float, float, float] = 50
    """Sets colorbar limits (mrad/meter). Plot limits set to [-clim, clim].
    Can be single value of tuple of three values to map to [x, y, xy] plots individually. (default 50)"""
    processing: list[str] | tuple[list[str], list[str], list[str]] = field(default_factory=list)
    """Processing string to apply when in MirrorAbstract.plot_orthorectified_curvature().
    Can be single value or tuple of three values to map to [x, y, xy] plots individually.
    (default [])"""
    smooth_kernel_width: float | tuple[float, float, float] = 1
    """Width of square smoothing kernel (pixels) to apply to curvature images in MirrorAbstract.plot_orthorectified_curvature().
    Can be single value of tuple of three values to map to [x, y, xy] plots individually.
    (default 1)"""
    to_plot: bool = True
    """Flag to produce plots or not. (default True)"""


@dataclass
class _OptionsRayTraceVis:
    ray_trace_optic_res: float = 0.05
    """Raytracing sampling resolution of optic in meters. (default 0.05)"""
    hist_bin_res: float = 0.07
    """Bin resolution (meters) when creating 2d histogram images. (default 0.07)"""
    hist_extent: float = 3
    """Width of histogram image in meters. (default 3)"""
    enclosed_energy_max_semi_width: float = 2
    """The max semi-width of square aperture (meters) used when computing enclosed energy plots. (default 2)"""
    to_plot: bool = True
    """Flag to produce plots or not. (default True)"""


@dataclass
class _RayTraceParameters:
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=20)
    """Light Source to use when producing ray trace image. (default `LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=20)`)"""
    v_target_center = Vxyz((0, 0, 50))
    """Location of target in scene (meters). (default `Vxyz((0, 0, 50))`)"""
    v_target_normal = Vxyz((0, 0, -1))
    """Orientation of target in scene. (default `Vxyz((0, 0, -1))`)"""


@dataclass
class _RayTraceOutput:
    ray_trace: rt.RayTrace
    histogram: np.ndarray
    histogram_x: np.ndarray
    histogram_y: np.ndarray
    ensquared_energy_values: np.ndarray
    ensquared_energy_widths: np.ndarray


class StandardPlotOutput:
    """Used to orchestrate the plotting and saving of the standard output plot suite of CSP mirrors"""

    def __init__(self):
        self.options_slope_vis = _OptionsSlopeVis()
        """Slope visualization options"""
        self.options_slope_deviation_vis = _OptionsSlopeDeviationVis()
        """Slope deviation visualization options"""
        self.options_curvature_vis = _OptionsCurvatureVis()
        """Curvature visualization options"""
        self.options_ray_trace_vis = _OptionsRayTraceVis()
        """Ray trace visualization options"""
        self.options_file_output = _OptionsFileOutput()
        """File output options"""
        self.options_save = _OptionsSave()
        """File save options"""
        self.options_slice_output = _OptionsSliceOutput()
        """Slice output options"""

        self.params_ray_trace = _RayTraceParameters()
        """Parameters to perform ray trace"""

        self.optic_measured: MirrorAbstract = None
        """Measured optic object"""
        self.optic_reference: MirrorAbstract = None
        """Reference optic object"""

        # Set up figure control objects for plots
        self.fig_control = RenderControlFigure(tile_array=(4, 2), tile_square=True)
        self.axis_control = rca.meters()

        # Define output data storage classes
        self._ray_trace_output_measured: _RayTraceOutput = None
        self._ray_trace_output_reference: _RayTraceOutput = None

    @property
    def _has_reference_optic(self) -> bool:
        return self.optic_reference is not None

    @property
    def _has_measured_optic(self) -> bool:
        return self.optic_measured is not None

    @property
    def _has_measured_ray_trace(self) -> bool:
        return self._ray_trace_output_measured is not None

    @property
    def _has_reference_ray_trace(self) -> bool:
        return self._ray_trace_output_reference is not None

    def set_plot_control_from_settings(self, settings: configparser.ConfigParser):
        """
        Fills in plot control fields that are provided in settings.
        If the settings does not contain a key, then the corresponding plot control field is left unchanged.
        """

        # File output control.
        # This should be kept in synch with class _OptionsFileOutput above.
        if ("Default" in settings) and ("plots.options_file_output.to_save" in settings["Default"]):
            self.options_file_output.to_save = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_file_output.to_save"]
            )
        # We expect output_dir to be handled by other routines.
        if ("Default" in settings) and ("plots.options_file_output.save_dpi" in settings["Default"]):
            self.options_file_output.save_dpi = int(settings["Default"]["plots.options_file_output.save_dpi"])
        if ("Default" in settings) and ("plots.options_file_output.save_format" in settings["Default"]):
            self.options_file_output.save_format = str(settings["Default"]["plots.options_file_output.save_format"])
        if ("Default" in settings) and ("plots.options_file_output.close_after_save" in settings["Default"]):
            self.options_file_output.close_after_save = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_file_output.close_after_save"]
            )
        if ("Default" in settings) and ("plots.options_file_output.number_in_name" in settings["Default"]):
            self.options_file_output.number_in_name = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_file_output.number_in_name"]
            )
        # We expect file_prefix to be handled by other routines.

        # File Save control.
        # This should be kept in synch with class _OptionsSave above.
        if ("Default" in settings) and ("plots.options_save.curvature_deviation" in settings["Default"]):
            self.options_save.curvature_deviation = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_save.curvature_deviation"]
            )
        if ("Default" in settings) and ("plots.options_save.slope_deviation" in settings["Default"]):
            self.options_save.slope_deviation = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_save.slope_deviation"]
            )
        if ("Default" in settings) and ("plots.options_save.enclosed_energy" in settings["Default"]):
            self.options_save.enclosed_energy = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_save.enclosed_energy"]
            )
        if ("Default" in settings) and ("plots.options_save.curvature" in settings["Default"]):
            self.options_save.curvature = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_save.curvature"]
            )
        if ("Default" in settings) and ("plots.options_save.slope" in settings["Default"]):
            self.options_save.slope = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_save.slope"]
            )

        # Slice output control.
        # This should be kept in synch with class _OptionsSlopeVis above.
        if ("Default" in settings) and ("plots.options_slice_output.slice_x_loc" in settings["Default"]):
            slice_x_loc_str = settings["Default"]["plots.options_slice_output.slice_x_loc"]
            try:
                slice_x_loc_list = [float(x.strip()) for x in slice_x_loc_str.split(',')]
                self.options_slice_output.slice_x_loc = slice_x_loc_list
            except Exception as e:
                lt.error_and_raise(ValueError, f"Failed to parse slice_x_loc: {e}")
        if ("Default" in settings) and ("plots.options_slice_output.slice_y_loc" in settings["Default"]):
            slice_y_loc_str = settings["Default"]["plots.options_slice_output.slice_y_loc"]
            try:
                slice_y_loc_list = [float(y.strip()) for y in slice_y_loc_str.split(',')]
                self.options_slice_output.slice_y_loc = slice_y_loc_list
            except Exception as e:
                lt.error_and_raise(ValueError, f"Failed to parse slice_x_loc: {e}")
        if ("Default" in settings) and ("plots.options_slice_output.plot_x_slope_offset" in settings["Default"]):
            self.options_slice_output.plot_x_slope_offset = float(
                settings["Default"]["plots.options_slice_output.plot_x_slope_offset"]
            )
        if ("Default" in settings) and ("plots.options_slice_output.plot_y_slope_offset" in settings["Default"]):
            self.options_slice_output.plot_y_slope_offset = float(
                settings["Default"]["plots.options_slice_output.plot_y_slope_offset"]
            )
        if ("Default" in settings) and (
            "plots.options_slice_output.plot_x_slope_deviation_offset" in settings["Default"]
        ):
            self.options_slice_output.plot_x_slope_deviation_offset = float(
                settings["Default"]["plots.options_slice_output.plot_x_slope_deviation_offset"]
            )
        if ("Default" in settings) and (
            "plots.options_slice_output.plot_y_slope_deviation_offset" in settings["Default"]
        ):
            self.options_slice_output.plot_y_slope_deviation_offset = float(
                settings["Default"]["plots.options_slice_output.plot_y_slope_deviation_offset"]
            )
        if ("Default" in settings) and ("plots.options_slice_output.plot_x_curvature_offset" in settings["Default"]):
            self.options_slice_output.plot_x_curvature_offset = float(
                settings["Default"]["plots.options_slice_output.plot_x_curvature_offset"]
            )
        if ("Default" in settings) and ("plots.options_slice_output.plot_y_curvature_offset" in settings["Default"]):
            self.options_slice_output.plot_y_curvature_offset = float(
                settings["Default"]["plots.options_slice_output.plot_y_curvature_offset"]
            )
        if ("Default" in settings) and (
            "plots.options_slice_output.plot_x_curvature_deviation_offset" in settings["Default"]
        ):
            self.options_slice_output.plot_x_curvature_deviation_offset = float(
                settings["Default"]["plots.options_slice_output.plot_x_curvature_deviation_offset"]
            )
        if ("Default" in settings) and (
            "plots.options_slice_output.plot_y_curvature_deviation_offset" in settings["Default"]
        ):
            self.options_slice_output.plot_y_curvature_deviation_offset = float(
                settings["Default"]["plots.options_slice_output.plot_y_curvature_deviation_offset"]
            )
        if ("Default" in settings) and ("plots.options_slice_output.to_plot" in settings["Default"]):
            self.options_slice_output.to_plot = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_slice_output.to_plot"]
            )
        if ("Default" in settings) and ("plots.options_slice_output.uncertainty_to_plot" in settings["Default"]):
            self.options_slice_output.uncertainty_to_plot = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_slice_output.uncertainty_to_plot"]
            )
        if ("Default" in settings) and ("plots.options_slice_output.xlim" in settings["Default"]):
            xlim_str = settings["Default"]["plots.options_slice_output.xlim"]
            try:
                xlim_vals = [float(x.strip()) for x in xlim_str.split(',')]
                if len(xlim_vals) == 2:
                    self.options_slice_output.xlim = (xlim_vals[0], xlim_vals[1])
                else:
                    lt.error_and_raise(ValueError, "xlim must have exactly two float values")
            except Exception as e:
                lt.error_and_raise(ValueError, f"Failed to parse xlim: {e}")

        if ("Default" in settings) and ("plots.options_slice_output.ylim" in settings["Default"]):
            ylim_str = settings["Default"]["plots.options_slice_output.ylim"]
            try:
                ylim_vals = [float(y.strip()) for y in ylim_str.split(',')]
                if len(ylim_vals) == 2:
                    self.options_slice_output.ylim = (ylim_vals[0], ylim_vals[1])
                else:
                    lt.error_and_raise(ValueError, "ylim must have exactly two float values")
            except Exception as e:
                lt.error_and_raise(ValueError, f"Failed to parse ylim: {e}")

        # Slope plot control.
        # This should be kept in synch with class _OptionsSlopeVis above.
        if ("Default" in settings) and ("plots.options_slope_vis.resolution" in settings["Default"]):
            self.options_slope_vis.resolution = float(settings["Default"]["plots.options_slope_vis.resolution"])
        if ("Default" in settings) and ("plots.options_slope_vis.clim" in settings["Default"]):
            self.options_slope_vis.clim = float(settings["Default"]["plots.options_slope_vis.clim"])
        if ("Default" in settings) and ("plots.options_slope_vis.quiver_density" in settings["Default"]):
            self.options_slope_vis.quiver_density = float(settings["Default"]["plots.options_slope_vis.quiver_density"])
        if ("Default" in settings) and ("plots.options_slope_vis.quiver_scale" in settings["Default"]):
            self.options_slope_vis.quiver_scale = float(settings["Default"]["plots.options_slope_vis.quiver_scale"])
        if ("Default" in settings) and ("plots.options_slope_vis.quiver_color" in settings["Default"]):
            self.options_slope_vis.quiver_color = str(settings["Default"]["plots.options_slope_vis.quiver_color"])
        if ("Default" in settings) and ("plots.options_slope_vis.to_plot" in settings["Default"]):
            self.options_slope_vis.to_plot = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_slope_vis.to_plot"]
            )

        # Slope deviation plot control.
        # This should be kept in synch with class _OptionsSlopeDeviationVis above.
        if ("Default" in settings) and ("plots.options_slope_deviation_vis.resolution" in settings["Default"]):
            self.options_slope_deviation_vis.resolution = float(
                settings["Default"]["plots.options_slope_deviation_vis.resolution"]
            )
        if ("Default" in settings) and ("plots.options_slope_deviation_vis.clim" in settings["Default"]):
            self.options_slope_deviation_vis.clim = float(settings["Default"]["plots.options_slope_deviation_vis.clim"])
        if ("Default" in settings) and ("plots.options_slope_deviation_vis.quiver_density" in settings["Default"]):
            self.options_slope_deviation_vis.quiver_density = float(
                settings["Default"]["plots.options_slope_deviation_vis.quiver_density"]
            )
        if ("Default" in settings) and ("plots.options_slope_deviation_vis.quiver_scale" in settings["Default"]):
            self.options_slope_deviation_vis.quiver_scale = float(
                settings["Default"]["plots.options_slope_deviation_vis.quiver_scale"]
            )
        if ("Default" in settings) and ("plots.options_slope_deviation_vis.quiver_color" in settings["Default"]):
            self.options_slope_deviation_vis.quiver_color = settings["Default"][
                "plots.options_slope_deviation_vis.quiver_color"
            ]
        if ("Default" in settings) and ("plots.options_slope_deviation_vis.to_plot" in settings["Default"]):
            self.options_slope_deviation_vis.to_plot = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_slope_deviation_vis.to_plot"]
            )

        # Curvature plot control.
        # This should be kept in synch with class _OptionsCurvatureVis above.
        if ("Default" in settings) and ("plots.options_curvature_vis.resolution" in settings["Default"]):
            self.options_curvature_vis.resolution = float(settings["Default"]["plots.options_curvature_vis.resolution"])
        if ("Default" in settings) and ("plots.options_curvature_vis.clim" in settings["Default"]):
            self.options_curvature_vis.clim = float(settings["Default"]["plots.options_curvature_vis.clim"])
        processing = []
        processing_options = [
            "log",
            "smooth",
        ]  # See VisualizeOrthorectifiedSlopeAbstract.py, routine plot_orthorectified_curvature().
        if ("Default" in settings) and ("plots.options_curvature_vis.processing_1" in settings["Default"]):
            processing_1 = settings["Default"]["plots.options_curvature_vis.processing_1"]
            if processing_1 in processing_options:
                processing.append(processing_1)
            elif processing_1 == "None":
                pass
            else:
                lt.error_and_raise(
                    ValueError, f'Curvature procesing option "{processing_1}" is not one of {processing_options}.'
                )
        if ("Default" in settings) and ("plots.options_curvature_vis.processing_2" in settings["Default"]):
            processing_2 = settings["Default"]["plots.options_curvature_vis.processing_2"]
            if processing_2 in processing_options:
                processing.append(processing_2)
            elif processing_2 == "None":
                pass
            else:
                lt.error_and_raise(
                    ValueError, f'Curvature procesing option "{processing_2}" is not one of {processing_options}.'
                )
        self.options_curvature_vis.processing = processing
        if ("Default" in settings) and ("plots.options_curvature_vis.smooth_kernel_width" in settings["Default"]):
            self.options_curvature_vis.smooth_kernel_width = float(
                settings["Default"]["plots.options_curvature_vis.smooth_kernel_width"]
            )
        if ("Default" in settings) and ("plots.options_curvature_vis.to_plot" in settings["Default"]):
            self.options_curvature_vis.to_plot = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_curvature_vis.to_plot"]
            )

        # Ray trace control.
        # This should be kept in synch with class _OptionsRayTraceVis above.
        if ("Default" in settings) and ("plots.options_ray_trace_vis.ray_trace_optic_res" in settings["Default"]):
            self.options_ray_trace_vis.ray_trace_optic_res = float(
                settings["Default"]["plots.options_ray_trace_vis.ray_trace_optic_res"]
            )
        if ("Default" in settings) and ("plots.options_ray_trace_vis.hist_bin_res" in settings["Default"]):
            self.options_ray_trace_vis.hist_bin_res = float(
                settings["Default"]["plots.options_ray_trace_vis.hist_bin_res"]
            )
        if ("Default" in settings) and ("plots.options_ray_trace_vis.hist_extent" in settings["Default"]):
            self.options_ray_trace_vis.hist_extent = float(
                settings["Default"]["plots.options_ray_trace_vis.hist_extent"]
            )
        if ("Default" in settings) and (
            "plots.options_ray_trace_vis.enclosed_energy_max_semi_width" in settings["Default"]
        ):
            self.options_ray_trace_vis.enclosed_energy_max_semi_width = float(
                settings["Default"]["plots.options_ray_trace_vis.enclosed_energy_max_semi_width"]
            )
        if ("Default" in settings) and ("plots.options_ray_trace_vis.to_plot" in settings["Default"]):
            self.options_ray_trace_vis.to_plot = st.convert_true_false_string_to_boolean(
                settings["Default"]["plots.options_ray_trace_vis.to_plot"]
            )

        # Ray trace parameters.
        # This should be kept in synch with class _RayTraceParameters above.
        # Light source.
        if (
            ("Default" in settings)
            and ("plots.options_ray_trace_vis.sun_direction_x" in settings["Default"])
            and ("plots.options_ray_trace_vis.sun_direction_y" in settings["Default"])
            and ("plots.options_ray_trace_vis.sun_direction_z" in settings["Default"])
            and ("plots.options_ray_trace_vis.sun_sample_resolution" in settings["Default"])
        ):
            sun_direction_x = float(settings["Default"]["plots.options_ray_trace_vis.sun_direction_x"])
            sun_direction_y = float(settings["Default"]["plots.options_ray_trace_vis.sun_direction_y"])
            sun_direction_z = float(settings["Default"]["plots.options_ray_trace_vis.sun_direction_z"])
            sun_direction = Uxyz((sun_direction_x, sun_direction_y, sun_direction_z))
            sun_sample_resolution = int(settings["Default"]["plots.options_ray_trace_vis.sun_sample_resolution"])
            self.params_ray_trace.source = LightSourceSun.from_given_sun_position(
                sun_direction, resolution=sun_sample_resolution
            )
        # Target center.
        if (
            ("Default" in settings)
            and ("plots.options_ray_trace_vis.v_target_center_x" in settings["Default"])
            and ("plots.options_ray_trace_vis.v_target_center_y" in settings["Default"])
            and ("plots.options_ray_trace_vis.v_target_center_z" in settings["Default"])
        ):
            v_target_center_x = float(settings["Default"]["plots.options_ray_trace_vis.v_target_center_x"])
            v_target_center_y = float(settings["Default"]["plots.options_ray_trace_vis.v_target_center_y"])
            v_target_center_z = float(settings["Default"]["plots.options_ray_trace_vis.v_target_center_z"])
            self.params_ray_trace.v_target_center = Vxyz((v_target_center_x, v_target_center_y, v_target_center_z))
        # Target normal.
        if (
            ("Default" in settings)
            and ("plots.options_ray_trace_vis.v_target_normal_x" in settings["Default"])
            and ("plots.options_ray_trace_vis.v_target_normal_y" in settings["Default"])
            and ("plots.options_ray_trace_vis.v_target_normal_z" in settings["Default"])
        ):
            v_target_normal_x = float(settings["Default"]["plots.options_ray_trace_vis.v_target_normal_x"])
            v_target_normal_y = float(settings["Default"]["plots.options_ray_trace_vis.v_target_normal_y"])
            v_target_normal_z = float(settings["Default"]["plots.options_ray_trace_vis.v_target_normal_z"])
            self.params_ray_trace.v_target_normal = Vxyz((v_target_normal_x, v_target_normal_y, v_target_normal_z))

    def plot(self):
        """Creates standard output plot suite"""
        # This function checks if plotting is turned on
        # Individual functions check if measured/reference optics/data exist

        # Plot slope/curvature, if able
        if self.options_slope_vis.to_plot:
            self._plot_slope_measured_optic()
            self._plot_slope_reference_optic()

            # Call the combined slope plotting function after individual plots
        #            if self.options_slice_output.to_plot:
        #                self.plot_combined_measured_reference_slopes('MeasRef')

        else:
            lt.info('Slope plotting turned off; skipping measured/reference optic slope plots.')

        if self.options_curvature_vis.to_plot:
            self._plot_curvature_measured_optic()
            self._plot_curvature_reference_optic()
        else:
            lt.info('Curvature plotting turned off; skipping measured/reference optic curvature plots.')

        if self.options_slope_deviation_vis.to_plot:
            self._plot_slope_deviation()
            self._plot_curvature_deviation()
        else:
            lt.info('Slope deviation plotting turned off; skipping slope deviation plots.')

        if self.options_ray_trace_vis.to_plot:
            # Perform ray tracing, if set
            self._perform_ray_trace_optic_measured()
            self._perform_ray_trace_optic_reference()
            # Plot ray trace data
            self._plot_ray_trace_image_measured_optic()
            self._plot_ray_trace_image_reference_optic()
            self._plot_enclosed_energy()
        else:
            lt.info('Ray tracing turned off; skipping all ray tracing plots.')

    def _plot_slope_measured_optic(self):
        # Plots optic slope for measured optic
        if self._has_measured_optic:
            self._plot_slope(self.optic_measured, 'Meas')
        else:
            lt.info('No measured optic; skipping measured optic slope plots.')

    def _plot_curvature_measured_optic(self):
        # Plots optic curvature for measured optic
        if self._has_measured_optic:
            self._plot_curvature(self.optic_measured, 'Meas')
        else:
            lt.info('No measured optic; skipping measured optic curvature plots.')

    def _plot_slope_reference_optic(self):
        # Plots optic slope for reference optic
        if self._has_reference_optic:
            self._plot_slope(self.optic_reference, 'Ref')
        else:
            lt.info('No reference optic; skipping reference optic slope plots.')

    def _plot_curvature_reference_optic(self):
        # Plots optic curvature for reference optic
        if self._has_reference_optic:
            self._plot_curvature(self.optic_reference, 'Ref')
        else:
            lt.info('No reference optic; skipping reference optic curvature plots.')

    def _process_plot_options(self, value) -> list:
        # If given a single value or length 1 tuple/list, returns length 3 list of the copied value.
        # If given a tuple/list of length 3, returns the same value input
        if isinstance(value, (tuple, list)):
            if len(value) == 3:
                return value
            elif len(value) in [0, 1]:
                return [value] * 3
            else:
                lt.error_and_raise(
                    ValueError, f'Plot option "{value}" must be length 3 or 1, not length {len(value):d}'
                )
        else:
            return [value] * 3

    '''
    def plot_combined_measured_reference_slopes(self, which_data: str):
        """Plots combined measured reference slopes"""
        measured_x_csv = f"{self.options_file_output.output_dir}/slope_slice_data_x_x_all_Meas.csv"
        x1 = pd.read_csv(measured_x_csv)
        measured_x_csv = f"{self.options_file_output.output_dir}/slope_slice_data_y_x_all_Meas.csv"
        y1 = pd.read_csv(measured_x_csv)
        num_slices = max(len(x1.columns), len(y1.columns)) - 1
        cmap = cm.get_cmap('tab10', num_slices)

        # Paths to saved CSV files for measured and reference slopes
        measured_x_csv = f"{self.options_file_output.output_dir}/slope_slice_data_y_x_all_Meas.csv"
        reference_x_csv = f"{self.options_file_output.output_dir}/slope_slice_data_y_x_all_Ref.csv"
        measured_y_csv = f"{self.options_file_output.output_dir}/slope_slice_data_y_y_all_Meas.csv"
        reference_y_csv = f"{self.options_file_output.output_dir}/slope_slice_data_y_y_all_Ref.csv"
        measured_mag_csv = f"{self.options_file_output.output_dir}/slope_slice_data_y_mag_all_Meas.csv"
        reference_mag_csv = f"{self.options_file_output.output_dir}/slope_slice_data_y_mag_all_Ref.csv"

        # Load data
        meas_x = pd.read_csv(measured_x_csv)
        ref_x = pd.read_csv(reference_x_csv)
        meas_y = pd.read_csv(measured_y_csv)
        ref_y = pd.read_csv(reference_y_csv)
        meas_mag = pd.read_csv(measured_mag_csv)
        ref_mag = pd.read_csv(reference_mag_csv)

        # Combined X slope plot
        plt.figure(figsize=(10, 6))
        i = 0
        for col in meas_x.columns[1:]:
            color = cmap(i)

            plt.plot(meas_x.iloc[:, 0], meas_x[col], linestyle='-', color=color, label=f'Measured {col}')
            plt.plot(ref_x.iloc[:, 0], ref_x[col], linestyle='--', color=color, label=f'Reference {col}')
            i += 1
        plt.xlabel('Coordinate (meters)')
        plt.ylabel('X Slope (mrad)')
        plt.title('Combined Measured and Reference X Slopes (Y Slices)')
        ax = plt.gca()
        if self.options_slice_output.xlim is not None:
            ax.set_xlim(self.options_slice_output.xlim)
        plt.legend()
        plt.grid()
        plt.savefig(
            f"{self.options_file_output.output_dir}/combined_slope_y_x_{which_data}.png",
            dpi=self.options_file_output.save_dpi,
        )

        # Combined Y slope plot
        plt.figure(figsize=(10, 6))
        i = 0
        for col in meas_y.columns[1:]:
            color = cmap(i)
            plt.plot(meas_y.iloc[:, 0], meas_y[col], linestyle='-', color=color, label=f'Measured {col}')
            plt.plot(ref_y.iloc[:, 0], ref_y[col], linestyle='--', color=color, label=f'Reference {col}')
            i += 1
        plt.xlabel('Coordinate (meters)')
        plt.ylabel('Y Slope (mrad)')
        plt.title('Combined Measured and Reference Y Slopes (Y Slices)')
        ax = plt.gca()
        if self.options_slice_output.xlim is not None:
            ax.set_xlim(self.options_slice_output.xlim)
        plt.legend()
        plt.grid()
        plt.savefig(
            f"{self.options_file_output.output_dir}/combined_slope_y_y_{which_data}.png",
            dpi=self.options_file_output.save_dpi,
        )

        # Combined Magnitude plot
        plt.figure(figsize=(10, 6))
        i = 0
        for col in meas_mag.columns[1:]:
            color = cmap(i)
            plt.plot(meas_mag.iloc[:, 0], meas_mag[col], linestyle='-', color=color, label=f'Measured {col}')
            plt.plot(ref_mag.iloc[:, 0], ref_mag[col], linestyle='--', color=color, label=f'Reference {col}')
            i += 1
        plt.xlabel('Coordinate (meters)')
        plt.ylabel('Slope Magnitude (mrad)')
        plt.title('Combined Measured and Reference Slope Magnitudes (Y Slices)')
        ax = plt.gca()
        if self.options_slice_output.xlim is not None:
            ax.set_xlim(self.options_slice_output.xlim)
        plt.legend()
        plt.grid()
        plt.savefig(
            f"{self.options_file_output.output_dir}/combined_slope_y_magnitude_{which_data}.png",
            dpi=self.options_file_output.save_dpi,
        )

        # Paths to saved CSV files for measured and reference slopes
        measured_x_csv = f"{self.options_file_output.output_dir}/slope_slice_data_x_x_all_Meas.csv"
        reference_x_csv = f"{self.options_file_output.output_dir}/slope_slice_data_x_x_all_Ref.csv"
        measured_y_csv = f"{self.options_file_output.output_dir}/slope_slice_data_x_y_all_Meas.csv"
        reference_y_csv = f"{self.options_file_output.output_dir}/slope_slice_data_x_y_all_Ref.csv"
        measured_mag_csv = f"{self.options_file_output.output_dir}/slope_slice_data_x_mag_all_Meas.csv"
        reference_mag_csv = f"{self.options_file_output.output_dir}/slope_slice_data_x_mag_all_Ref.csv"

        # Load data
        meas_x = pd.read_csv(measured_x_csv)
        ref_x = pd.read_csv(reference_x_csv)
        meas_y = pd.read_csv(measured_y_csv)
        ref_y = pd.read_csv(reference_y_csv)
        meas_mag = pd.read_csv(measured_mag_csv)
        ref_mag = pd.read_csv(reference_mag_csv)

        # Combined X slope plot
        plt.figure(figsize=(10, 6))
        i = 0
        for col in meas_x.columns[1:]:
            color = cmap(i)
            plt.plot(meas_x.iloc[:, 0], meas_x[col], linestyle='-', color=color, label=f'Measured {col}')
            plt.plot(ref_x.iloc[:, 0], ref_x[col], linestyle='--', color=color, label=f'Reference {col}')
            i += 1
        plt.xlabel('Coordinate (meters)')
        plt.ylabel('X Slope (mrad)')
        plt.title('Combined Measured and Reference X Slopes (X Slices)')
        ax = plt.gca()
        if self.options_slice_output.ylim is not None:
            ax.set_xlim(self.options_slice_output.ylim)
        plt.legend()
        plt.grid()
        plt.savefig(
            f"{self.options_file_output.output_dir}/combined_slope_x_x_{which_data}.png",
            dpi=self.options_file_output.save_dpi,
        )

        # Combined Y slope plot
        plt.figure(figsize=(10, 6))
        i = 0
        for col in meas_y.columns[1:]:
            color = cmap(i)
            plt.plot(meas_y.iloc[:, 0], meas_y[col], linestyle='-', color=color, label=f'Measured {col}')
            plt.plot(ref_y.iloc[:, 0], ref_y[col], linestyle='--', color=color, label=f'Reference {col}')
            i += 1
        plt.xlabel('Coordinate (meters)')
        plt.ylabel('Y Slope (mrad)')
        plt.title('Combined Measured and Reference Y Slopes (X Slices)')
        ax = plt.gca()
        if self.options_slice_output.ylim is not None:
            ax.set_xlim(self.options_slice_output.ylim)
        plt.legend()
        plt.grid()
        plt.savefig(
            f"{self.options_file_output.output_dir}/combined_slope_x_y_{which_data}.png",
            dpi=self.options_file_output.save_dpi,
        )

        # Combined Magnitude plot
        plt.figure(figsize=(10, 6))
        i = 0
        for col in meas_mag.columns[1:]:
            color = cmap(i)
            plt.plot(meas_mag.iloc[:, 0], meas_mag[col], linestyle='-', color=color, label=f'Measured {col}')
            plt.plot(ref_mag.iloc[:, 0], ref_mag[col], linestyle='--', color=color, label=f'Reference {col}')
            i += 1
        plt.xlabel('Coordinate (meters)')
        plt.ylabel('Slope Magnitude (mrad)')
        plt.title('Combined Measured and Reference Slope Magnitudes (X Slices)')
        ax = plt.gca()
        if self.options_slice_output.ylim is not None:
            ax.set_xlim(self.options_slice_output.ylim)
        plt.legend()
        plt.grid()
        plt.savefig(
            f"{self.options_file_output.output_dir}/combined_slope_x_magnitude_{which_data}.png",
            dpi=self.options_file_output.save_dpi,
        )
    '''

    def _plot_curvature_deviation(self):
        # Plots curvature deviation
        if self._has_measured_optic and self._has_reference_optic:
            # Separate outputs
            quiver_densities = self._process_plot_options(self.options_slope_vis.quiver_density)
            quiver_scales = self._process_plot_options(self.options_slope_vis.quiver_scale)
            quiver_colors = self._process_plot_options(self.options_slope_vis.quiver_color)

            # Curvature deviation magnitude
            fig_rec = fm.setup_figure(
                self.fig_control,
                self.axis_control,
                name=self.options_file_output.file_prefix + "Curvature Deviation XY",
                number_in_name=self.options_file_output.number_in_name,
            )
            x_vec, y_vec, image = self.optic_measured.plot_orthorectified_curvature_error(
                self.optic_reference,
                self.options_curvature_vis.resolution,
                type_='xy',
                clim=self.options_curvature_vis.clim,
                axis=fig_rec.axis,
                processing=(
                    self.options_curvature_vis.processing if hasattr(self.options_curvature_vis, 'processing') else []
                ),
                smooth_kernel_width=(
                    self.options_curvature_vis.smooth_kernel_width
                    if hasattr(self.options_curvature_vis, 'smooth_kernel_width')
                    else 1
                ),
                return_data=True,
            )
            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )

            if self.options_save.curvature_deviation:
                hdf5_filename = f"{self.options_file_output.output_dir}/curvature_deviation_image_xy.h5"
                with h5py.File(hdf5_filename, 'w') as h5f:
                    h5f.create_dataset('x_vec', data=x_vec)
                    h5f.create_dataset('y_vec', data=y_vec)
                    h5f.create_dataset('image', data=image)

            # Curvature deviation X
            fig_rec = fm.setup_figure(
                self.fig_control,
                self.axis_control,
                name=self.options_file_output.file_prefix + "Curvature Deviation X",
                number_in_name=self.options_file_output.number_in_name,
            )
            x_vec, y_vec, image_x = self.optic_measured.plot_orthorectified_curvature_error(
                self.optic_reference,
                self.options_curvature_vis.resolution,
                type_='x',
                clim=self.options_curvature_vis.clim,
                axis=fig_rec.axis,
                processing=(
                    self.options_curvature_vis.processing if hasattr(self.options_curvature_vis, 'processing') else []
                ),
                smooth_kernel_width=(
                    self.options_curvature_vis.smooth_kernel_width
                    if hasattr(self.options_curvature_vis, 'smooth_kernel_width')
                    else 1
                ),
                return_data=True,
            )
            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )

            if self.options_save.curvature_deviation:
                hdf5_filename = f"{self.options_file_output.output_dir}/curvature_deviation_image_x.h5"
                with h5py.File(hdf5_filename, 'w') as h5f:
                    h5f.create_dataset('x_vec', data=x_vec)
                    h5f.create_dataset('y_vec', data=y_vec)
                    h5f.create_dataset('image_x', data=image_x)

            # Curvature deviation Y
            fig_rec = fm.setup_figure(
                self.fig_control,
                self.axis_control,
                name=self.options_file_output.file_prefix + "Curvature Deviation Y",
                number_in_name=self.options_file_output.number_in_name,
            )
            x_vec, y_vec, image_y = self.optic_measured.plot_orthorectified_curvature_error(
                self.optic_reference,
                self.options_curvature_vis.resolution,
                type_='y',
                clim=self.options_curvature_vis.clim,
                axis=fig_rec.axis,
                processing=(
                    self.options_curvature_vis.processing if hasattr(self.options_curvature_vis, 'processing') else []
                ),
                smooth_kernel_width=(
                    self.options_curvature_vis.smooth_kernel_width
                    if hasattr(self.options_curvature_vis, 'smooth_kernel_width')
                    else 1
                ),
                return_data=True,
            )

            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )

            if self.options_save.curvature_deviation:
                hdf5_filename = f"{self.options_file_output.output_dir}/curvature_deviation_image_y.h5"
                with h5py.File(hdf5_filename, 'w') as h5f:
                    h5f.create_dataset('x_vec', data=x_vec)
                    h5f.create_dataset('y_vec', data=y_vec)
                    h5f.create_dataset('image_y', data=image_y)

            if self.options_slice_output.to_plot:

                x_coords = self.options_slice_output.slice_x_loc
                x_indices = coords_to_indices(x_coords, x_vec)

                y_coords = self.options_slice_output.slice_y_loc
                y_indices = coords_to_indices(y_coords, y_vec)

                offset_x = self.options_slice_output.plot_x_curvature_deviation_offset  # mrad offset between x slices
                offset_y = self.options_slice_output.plot_y_curvature_deviation_offset  # mrad offset between y slices
                offset_counter_x = 0  # Initialize counter
                offset_counter_y = 0  # Initialize counter

                # Plot contour plot with slice locations
                x_slices = [x_vec[idx] for idx in x_indices]  # x coordinate values for slices
                y_slices = [y_vec[idx] for idx in y_indices]  # y coordinate values for slices

                num_slices = max(len(x_indices), len(y_indices))
                cmap = cm.get_cmap('tab10', num_slices)

                # Initialize lists to accumulate data for all y slices
                all_slice_data_x = []
                all_slice_data_y = []
                all_slice_data_mag = []

                # Create separate figures for X curvature, Y curvature, and Magnitude
                plt.figure(figsize=(10, 6))
                plt.title('2D Line Plot of X Curvature Deviation')
                plt.xlabel('X Coordinate (meters)')
                plt.ylabel('Curvature Deviation (mrad/m)')
                plt.grid()

                plt.figure(figsize=(10, 6))
                plt.title('2D Line Plot of Y Curvature Deviation')
                plt.xlabel('X Coordinate (meters)')
                plt.ylabel('Curvature Deviation (mrad/m)')
                plt.grid()

                plt.figure(figsize=(10, 6))
                plt.title('2D Line Plot of Curvature Deviation Magnitude')
                plt.xlabel('X Coordinate (meters)')
                plt.ylabel('Curvature Deviation (mrad/m)')
                plt.grid()

                # Store figure handles to plot on them
                fig_x = plt.gcf()
                fig_y = plt.gcf()
                fig_mag = plt.gcf()

                # Actually, we need separate figure objects for each plot, so create them explicitly:
                fig_x, ax_x = plt.subplots(figsize=(10, 6))
                fig_y, ax_y = plt.subplots(figsize=(10, 6))
                fig_mag, ax_mag = plt.subplots(figsize=(10, 6))

                i = 0

                for y_coordinate_index in y_indices:
                    # Extract the slices
                    slice_data_x1 = image_x[y_coordinate_index, :]  # X slope
                    slice_data_y1 = image_y[y_coordinate_index, :]  # Y slope
                    slice_data1 = image[y_coordinate_index, :]
                    coordinate_label1 = f"Y Location = {y_vec[y_coordinate_index]:.4f}"
                    x_values1 = x_vec  # Use x_vec for x-axis

                    offset1 = offset_counter_y * offset_y

                    if self.options_slice_output.uncertainty_to_plot:

                        # Define subset range for error bars
                        indx = [50, 150, 250, 350]
                        x_un = x_values1[indx]
                        yx_un = slice_data_x1[indx]
                        yy_un = slice_data_y1[indx]
                        y_un = slice_data1[indx]
                        profx_un = [1, 0.5, 1, 0.75]
                        profy_un = [1, 0.5, 1, 0.75]
                        prof_un = [1, 0.5, 1, 0.75]

                    # Get color from colormap
                    color = cmap(i)

                    # Plot on separate axes
                    (line_x,) = ax_x.plot(
                        x_values1[1:], slice_data_x1 + offset1, label=f'{coordinate_label1}', color=color, linestyle='-'
                    )
                    (line_y,) = ax_y.plot(
                        x_values1, slice_data_y1 + offset1, label=f'{coordinate_label1}', color=color, linestyle='-'
                    )
                    (line_mag,) = ax_mag.plot(
                        x_values1[1:], slice_data1 + offset1, label=f'{coordinate_label1}', color=color, linestyle='-'
                    )

                    if self.options_slice_output.uncertainty_to_plot:
                        ax_x.errorbar(x_un, yx_un + offset1, yerr=profx_un, fmt='none', ecolor=color, capsize=3)
                        ax_y.errorbar(x_un, yy_un + offset1, yerr=profy_un, fmt='none', ecolor=color, capsize=3)
                        ax_mag.errorbar(x_un, y_un + offset1, yerr=prof_un, fmt='none', ecolor=color, capsize=3)

                    # Accumulate data for saving
                    all_slice_data_x.append(slice_data_x1)
                    all_slice_data_y.append(slice_data_y1)
                    all_slice_data_mag.append(slice_data1)

                    offset_counter_y += 1  # Increase counter
                    i += 1

                # Finalize X curvature plot
                ax_x.set_xlabel('X Coordinate (meters)')
                ax_x.set_ylabel('Curvature Deviation (mrad/m)')
                ax_x.set_title('X Curvature Deviation (Y Slices)')
                if self.options_slice_output.xlim is not None:
                    ax_x.set_xlim(self.options_slice_output.xlim)
                ax_x.legend()
                ax_x.grid()
                fig_x.savefig(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_y_x_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_x)

                # Finalize Y curvature plot
                ax_y.set_xlabel('X Coordinate (meters)')
                ax_y.set_ylabel('Curvature Deviation (mrad/m)')
                ax_y.set_title('Y Curvature Deviation (Y Slices)')
                if self.options_slice_output.xlim is not None:
                    ax_y.set_xlim(self.options_slice_output.xlim)
                ax_y.legend()
                ax_y.grid()
                fig_y.savefig(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_y_y_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_y)

                # Finalize Magnitude plot
                ax_mag.set_xlabel('X Coordinate (meters)')
                ax_mag.set_ylabel('Curvature Deviation (mrad/m)')
                ax_mag.set_title('Curvature Deviation Magnitude (Y Slices)')
                if self.options_slice_output.xlim is not None:
                    ax_mag.set_xlim(self.options_slice_output.xlim)
                ax_mag.legend()
                ax_mag.grid()
                fig_mag.savefig(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_y_mag_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_mag)

                # Convert accumulated lists to arrays (shape: num_slices x slice_length)
                all_slice_data_x = np.array(all_slice_data_x)
                all_slice_data_y = np.array(all_slice_data_y)
                all_slice_data_mag = np.array(all_slice_data_mag)

                # Save X curvature data (all slices) to CSV and HDF5
                header_x = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
                data_to_save = np.column_stack((x_vec[1:].flatten(), all_slice_data_x.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_y_x_all.csv",
                    data_to_save,
                    header=header_x,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_y_x_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('X Values', data=x_vec[1:].flatten())
                    h5f.create_dataset('Y Indices', data=np.array(y_indices))
                    h5f.create_dataset('X Curvature Deviation', data=all_slice_data_x)

                # Save Y curvature data (all slices) to CSV and HDF5
                header_y = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
                data_to_save = np.column_stack((x_vec.flatten(), all_slice_data_y.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_y_y_all.csv",
                    data_to_save,
                    header=header_y,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_y_y_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('X Values', data=x_vec.flatten())
                    h5f.create_dataset('Y Indices', data=np.array(y_indices))
                    h5f.create_dataset('Y Curvature Deviation', data=all_slice_data_y)

                # Save Magnitude data (all slices) to CSV and HDF5
                header_mag = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
                data_to_save = np.column_stack((x_vec[1:].flatten(), all_slice_data_mag.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_y_mag_all.csv",
                    data_to_save,
                    header=header_mag,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_y_mag_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('X Values', data=x_vec[1:].flatten())
                    h5f.create_dataset('Y Indices', data=np.array(y_indices))
                    h5f.create_dataset('Curvature Deviation Magnitude', data=all_slice_data_mag)

                # Initialize lists to accumulate data for all x slices
                all_slice_data_x_x = []
                all_slice_data_y_x = []
                all_slice_data_mag_x = []

                # Create separate figures for X curvature, Y curvature, and Magnitude (x slices)
                fig_x_x, ax_x_x = plt.subplots(figsize=(10, 6))
                fig_y_x, ax_y_x = plt.subplots(figsize=(10, 6))
                fig_mag_x, ax_mag_x = plt.subplots(figsize=(10, 6))

                i = 0
                for x_coordinate_index in x_indices:
                    # Extract the slices
                    slice_data_x2 = image_x[:, x_coordinate_index]  # X slope difference
                    slice_data_y2 = image_y[:, x_coordinate_index]  # Y slope difference
                    slice_data2 = image[:, x_coordinate_index]  # Slope magnitude
                    coordinate_label2 = f"X Location = {x_vec[x_coordinate_index]:.4f}"
                    x_values2 = y_vec  # Use y_vec for x-axis

                    offset2 = offset_counter_x * offset_x

                    if self.options_slice_output.uncertainty_to_plot:
                        # Define subset range for error bars
                        indx = [50, 150, 250, 350]
                        x_un = x_values2[indx]
                        yx_un = slice_data_x2[indx]
                        yy_un = slice_data_y2[indx]
                        y_un = slice_data2[indx]
                        profx_un = [1, 0.5, 1, 0.75]
                        profy_un = [1, 0.5, 1, 0.75]
                        prof_un = [1, 0.5, 1, 0.75]

                    # Get color from colormap
                    color = cmap(i)

                    # Plot on separate axes
                    ax_x_x.plot(
                        x_values2, slice_data_x2 + offset2, label=f'{coordinate_label2}', color=color, linestyle='-'
                    )
                    ax_y_x.plot(
                        x_values2[1:], slice_data_y2 + offset2, label=f'{coordinate_label2}', color=color, linestyle='-'
                    )
                    ax_mag_x.plot(
                        x_values2[1:], slice_data2 + offset2, label=f'{coordinate_label2}', color=color, linestyle='-'
                    )

                    if self.options_slice_output.uncertainty_to_plot:
                        ax_x_x.errorbar(x_un, yx_un + offset2, yerr=profx_un, fmt='none', ecolor=color, capsize=3)
                        ax_y_x.errorbar(x_un, yy_un + offset2, yerr=profy_un, fmt='none', ecolor=color, capsize=3)
                        ax_mag_x.errorbar(x_un, y_un + offset2, yerr=prof_un, fmt='none', ecolor=color, capsize=3)

                    # Accumulate data for saving
                    all_slice_data_x_x.append(slice_data_x2)
                    all_slice_data_y_x.append(slice_data_y2)
                    all_slice_data_mag_x.append(slice_data2)

                    offset_counter_x += 1  # Increase counter
                    i += 1

                # Finalize X curvature plot (x slices)
                ax_x_x.set_xlabel('Y Coordinate (meters)')
                ax_x_x.set_ylabel('Curvature Deviation (mrad/m)')
                ax_x_x.set_title('X Curvature Deviation (X slices)')
                if self.options_slice_output.ylim is not None:
                    ax_x_x.set_xlim(self.options_slice_output.ylim)
                ax_x_x.legend()
                ax_x_x.grid()
                fig_x_x.savefig(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_x_x_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_x_x)

                # Finalize Y curvature plot (x slices)
                ax_y_x.set_xlabel('Y Coordinate (meters)')
                ax_y_x.set_ylabel('Curvature Deviation (mrad/m)')
                ax_y_x.set_title('Y Curvature Deviation (X slices)')
                if self.options_slice_output.ylim is not None:
                    ax_y_x.set_xlim(self.options_slice_output.ylim)
                ax_y_x.legend()
                ax_y_x.grid()
                fig_y_x.savefig(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_x_y_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_y_x)

                # Finalize Magnitude plot (x slices)
                ax_mag_x.set_xlabel('Y Coordinate (meters)')
                ax_mag_x.set_ylabel('Curvature Deviation (mrad/m)')
                ax_mag_x.set_title('Curvature Deviation Magnitude (X slices)')
                if self.options_slice_output.ylim is not None:
                    ax_mag_x.set_xlim(self.options_slice_output.ylim)
                ax_mag_x.legend()
                ax_mag_x.grid()
                fig_mag_x.savefig(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_x_mag_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_mag_x)

                # Convert accumulated lists to arrays (shape: num_slices x slice_length)
                all_slice_data_x_x = np.array(all_slice_data_x_x)
                all_slice_data_y_x = np.array(all_slice_data_y_x)
                all_slice_data_mag_x = np.array(all_slice_data_mag_x)

                # Save X curvature data (all x slices) to CSV and HDF5
                header_x_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
                data_to_save = np.column_stack((y_vec.flatten(), all_slice_data_x_x.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_x_x_all.csv",
                    data_to_save,
                    header=header_x_x,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_x_x_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('Y Values', data=y_vec.flatten())
                    h5f.create_dataset('X Indices', data=np.array(x_indices))
                    h5f.create_dataset('X Curvature Deviation', data=all_slice_data_x_x)

                # Save Y curvature data (all x slices) to CSV and HDF5
                header_y_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
                data_to_save = np.column_stack((y_vec[1:].flatten(), all_slice_data_y_x.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_x_y_all.csv",
                    data_to_save,
                    header=header_y_x,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_x_y_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('Y Values', data=y_vec[1:].flatten())
                    h5f.create_dataset('X Indices', data=np.array(x_indices))
                    h5f.create_dataset('Y Curvature Deviation', data=all_slice_data_y_x)

                # Save Magnitude data (all x slices) to CSV and HDF5
                header_mag_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
                data_to_save = np.column_stack((y_vec[1:].flatten(), all_slice_data_mag_x.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_x_mag_all.csv",
                    data_to_save,
                    header=header_mag_x,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/curvature_deviation_slice_data_x_mag_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('Y Values', data=y_vec[1:].flatten())
                    h5f.create_dataset('X Indices', data=np.array(x_indices))
                    h5f.create_dataset('Curvature Deviation Magnitude', data=all_slice_data_mag_x)

    def _plot_slope_deviation(self):
        # Plots slope deviation
        if self._has_measured_optic and self._has_reference_optic:
            # Separate outputs
            quiver_densities = self._process_plot_options(self.options_slope_vis.quiver_density)
            quiver_scales = self._process_plot_options(self.options_slope_vis.quiver_scale)
            quiver_colors = self._process_plot_options(self.options_slope_vis.quiver_color)

            # Slope deviation magnitude
            fig_rec = fm.setup_figure(
                self.fig_control,
                self.axis_control,
                name=self.options_file_output.file_prefix + "Slope Deviation XY",
                number_in_name=self.options_file_output.number_in_name,
            )
            x_vec, y_vec, image = self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='xy',
                quiver_density=quiver_densities[0],
                quiver_scale=quiver_scales[0],
                quiver_color=quiver_colors[0],
                clim=self.options_slope_deviation_vis.clim,
                axis=fig_rec.axis,
                return_data=True,  # Request the data to be returned
            )

            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )

            if self.options_save.slope_deviation:
                hdf5_filename = f"{self.options_file_output.output_dir}/slope_deviation_image_xy.h5"
                with h5py.File(hdf5_filename, 'w') as h5f:
                    h5f.create_dataset('x_vec', data=x_vec)
                    h5f.create_dataset('y_vec', data=y_vec)
                    h5f.create_dataset('image', data=image)

            # Slope x
            fig_rec = fm.setup_figure(
                self.fig_control,
                self.axis_control,
                name=self.options_file_output.file_prefix + "Slope Deviation X",
                number_in_name=self.options_file_output.number_in_name,
            )
            x_vec, y_vec, image_x = self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='x',
                quiver_density=quiver_densities[1],
                quiver_scale=quiver_scales[1],
                quiver_color=quiver_colors[1],
                clim=self.options_slope_deviation_vis.clim,
                axis=fig_rec.axis,
                return_data=True,  # Request the data to be returned
            )
            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )

            if self.options_save.slope_deviation:
                hdf5_filename = f"{self.options_file_output.output_dir}/slope_deviation_image_x.h5"
                with h5py.File(hdf5_filename, 'w') as h5f:
                    h5f.create_dataset('x_vec', data=x_vec)
                    h5f.create_dataset('y_vec', data=y_vec)
                    h5f.create_dataset('image_x', data=image_x)

            # Slope Y
            fig_rec = fm.setup_figure(
                self.fig_control,
                self.axis_control,
                name=self.options_file_output.file_prefix + "Slope Deviation Y",
                number_in_name=self.options_file_output.number_in_name,
            )
            x_vec, y_vec, image_y = self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='y',
                quiver_density=quiver_densities[2],
                quiver_scale=quiver_scales[2],
                quiver_color=quiver_colors[2],
                clim=self.options_slope_deviation_vis.clim,
                axis=fig_rec.axis,
                return_data=True,  # Request the data to be returned
            )
            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )

            if self.options_save.slope_deviation:
                hdf5_filename = f"{self.options_file_output.output_dir}/slope_deviation_image_y.h5"
                with h5py.File(hdf5_filename, 'w') as h5f:
                    h5f.create_dataset('x_vec', data=x_vec)
                    h5f.create_dataset('y_vec', data=y_vec)
                    h5f.create_dataset('image_y', data=image_y)

            if self.options_slice_output.to_plot:

                x_coords = self.options_slice_output.slice_x_loc
                x_indices = coords_to_indices(x_coords, x_vec)

                y_coords = self.options_slice_output.slice_y_loc
                y_indices = coords_to_indices(y_coords, y_vec)

                offset_x = self.options_slice_output.plot_x_slope_deviation_offset  # mrad offset between x slices
                offset_y = self.options_slice_output.plot_y_slope_deviation_offset  # mrad offset between y slices
                offset_counter_x = 0  # Initialize counter
                offset_counter_y = 0  # Initialize counter

                # Plot contour plot with slice locations
                x_slices = [x_vec[idx] for idx in x_indices]  # x coordinate values for slices
                y_slices = [y_vec[idx] for idx in y_indices]  # y coordinate values for slices

                num_slices = max(len(x_indices), len(y_indices))
                cmap = cm.get_cmap('tab10', num_slices)

                # Initialize lists to accumulate data for all y slices
                all_slice_data_x = []
                all_slice_data_y = []
                all_slice_data_mag = []

                # Create separate figures for X slope, Y slope, and Magnitude
                plt.figure(figsize=(10, 6))
                plt.title('2D Line Plot of X Slope Deviation')
                plt.xlabel('X Coordinate (meters)')
                plt.ylabel('Slope Deviation (mrad)')
                plt.grid()

                plt.figure(figsize=(10, 6))
                plt.title('2D Line Plot of Y Slope Deviation')
                plt.xlabel('X Coordinate (meters)')
                plt.ylabel('Slope Deviation (mrad)')
                plt.grid()

                plt.figure(figsize=(10, 6))
                plt.title('2D Line Plot of Slope Deviation Magnitude')
                plt.xlabel('X Coordinate (meters)')
                plt.ylabel('Slope Deviation (mrad)')
                plt.grid()

                # Store figure handles to plot on them
                fig_x = plt.gcf()
                fig_y = plt.gcf()
                fig_mag = plt.gcf()

                # Actually, we need separate figure objects for each plot, so create them explicitly:
                fig_x, ax_x = plt.subplots(figsize=(10, 6))
                fig_y, ax_y = plt.subplots(figsize=(10, 6))
                fig_mag, ax_mag = plt.subplots(figsize=(10, 6))

                i = 0

                for y_coordinate_index in y_indices:
                    # Extract the slices
                    slice_data_x1 = image_x[y_coordinate_index, :]  # X slope
                    slice_data_y1 = image_y[y_coordinate_index, :]  # Y slope
                    slice_data1 = image[y_coordinate_index, :]
                    coordinate_label1 = f"Y Location = {y_vec[y_coordinate_index]:.4f}"
                    x_values1 = x_vec  # Use x_vec for x-axis

                    offset1 = offset_counter_y * offset_y

                    if self.options_slice_output.uncertainty_to_plot:

                        # Define subset range for error bars
                        indx = [50, 150, 250, 350]
                        x_un = x_values1[indx]
                        yx_un = slice_data_x1[indx]
                        yy_un = slice_data_y1[indx]
                        y_un = slice_data1[indx]
                        profx_un = [1, 0.5, 1, 0.75]
                        profy_un = [1, 0.5, 1, 0.75]
                        prof_un = [1, 0.5, 1, 0.75]

                    # Get color from colormap
                    color = cmap(i)

                    # Plot on separate axes
                    (line_x,) = ax_x.plot(
                        x_values1, slice_data_x1 + offset1, label=f'{coordinate_label1}', color=color, linestyle='-'
                    )
                    (line_y,) = ax_y.plot(
                        x_values1, slice_data_y1 + offset1, label=f'{coordinate_label1}', color=color, linestyle='-'
                    )
                    (line_mag,) = ax_mag.plot(
                        x_values1, slice_data1 + offset1, label=f'{coordinate_label1}', color=color, linestyle='-'
                    )

                    if self.options_slice_output.uncertainty_to_plot:
                        ax_x.errorbar(x_un, yx_un + offset1, yerr=profx_un, fmt='none', ecolor=color, capsize=3)
                        ax_y.errorbar(x_un, yy_un + offset1, yerr=profy_un, fmt='none', ecolor=color, capsize=3)
                        ax_mag.errorbar(x_un, y_un + offset1, yerr=prof_un, fmt='none', ecolor=color, capsize=3)

                    # Accumulate data for saving
                    all_slice_data_x.append(slice_data_x1)
                    all_slice_data_y.append(slice_data_y1)
                    all_slice_data_mag.append(slice_data1)

                    offset_counter_y += 1  # Increase counter
                    i += 1

                # Finalize X slope plot
                ax_x.set_xlabel('X Coordinate (meters)')
                ax_x.set_ylabel('Slope Deviation (mrad)')
                ax_x.set_title('X Slope Deviation (Y Slices)')
                if self.options_slice_output.xlim is not None:
                    ax_x.set_xlim(self.options_slice_output.xlim)
                ax_x.legend()
                ax_x.grid()
                fig_x.savefig(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_y_x_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_x)

                # Finalize Y slope plot
                ax_y.set_xlabel('X Coordinate (meters)')
                ax_y.set_ylabel('Slope Deviation (mrad)')
                ax_y.set_title('Y Slope Deviation (Y Slices)')
                if self.options_slice_output.xlim is not None:
                    ax_y.set_xlim(self.options_slice_output.xlim)
                ax_y.legend()
                ax_y.grid()
                fig_y.savefig(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_y_y_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_y)

                # Finalize Magnitude plot
                ax_mag.set_xlabel('X Coordinate (meters)')
                ax_mag.set_ylabel('Slope Deviation (mrad)')
                ax_mag.set_title('Slope Deviation Magnitude (Y Slices)')
                if self.options_slice_output.xlim is not None:
                    ax_mag.set_xlim(self.options_slice_output.xlim)
                ax_mag.legend()
                ax_mag.grid()
                fig_mag.savefig(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_y_mag_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_mag)

                # Convert accumulated lists to arrays (shape: num_slices x slice_length)
                all_slice_data_x = np.array(all_slice_data_x)
                all_slice_data_y = np.array(all_slice_data_y)
                all_slice_data_mag = np.array(all_slice_data_mag)

                # Save X slope data (all slices) to CSV and HDF5
                header_x = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
                data_to_save = np.column_stack((x_vec.flatten(), all_slice_data_x.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_y_x_all.csv",
                    data_to_save,
                    header=header_x,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_y_x_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('X Values', data=x_vec.flatten())
                    h5f.create_dataset('Y Indices', data=np.array(y_indices))
                    h5f.create_dataset('X Slope Deviation', data=all_slice_data_x)

                # Save Y slope data (all slices) to CSV and HDF5
                header_y = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
                data_to_save = np.column_stack((x_vec.flatten(), all_slice_data_y.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_y_y_all.csv",
                    data_to_save,
                    header=header_y,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_y_y_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('X Values', data=x_vec.flatten())
                    h5f.create_dataset('Y Indices', data=np.array(y_indices))
                    h5f.create_dataset('Y Slope Deviation', data=all_slice_data_y)

                # Save Magnitude data (all slices) to CSV and HDF5
                header_mag = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
                data_to_save = np.column_stack((x_vec.flatten(), all_slice_data_mag.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_y_mag_all.csv",
                    data_to_save,
                    header=header_mag,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_y_mag_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('X Values', data=x_vec.flatten())
                    h5f.create_dataset('Y Indices', data=np.array(y_indices))
                    h5f.create_dataset('Slope Deviation Magnitude', data=all_slice_data_mag)

                # Initialize lists to accumulate data for all x slices
                all_slice_data_x_x = []
                all_slice_data_y_x = []
                all_slice_data_mag_x = []

                # Create separate figures for X slope, Y slope, and Magnitude (x slices)
                fig_x_x, ax_x_x = plt.subplots(figsize=(10, 6))
                fig_y_x, ax_y_x = plt.subplots(figsize=(10, 6))
                fig_mag_x, ax_mag_x = plt.subplots(figsize=(10, 6))

                i = 0
                for x_coordinate_index in x_indices:
                    # Extract the slices
                    slice_data_x2 = image_x[:, x_coordinate_index]  # X slope difference
                    slice_data_y2 = image_y[:, x_coordinate_index]  # Y slope difference
                    slice_data2 = image[:, x_coordinate_index]  # Slope magnitude
                    coordinate_label2 = f"X Location = {x_vec[x_coordinate_index]:.4f}"
                    x_values2 = y_vec  # Use y_vec for x-axis

                    offset2 = offset_counter_x * offset_x

                    if self.options_slice_output.uncertainty_to_plot:
                        # Define subset range for error bars
                        indx = [50, 150, 250, 350]
                        x_un = x_values2[indx]
                        yx_un = slice_data_x2[indx]
                        yy_un = slice_data_y2[indx]
                        y_un = slice_data2[indx]
                        profx_un = [1, 0.5, 1, 0.75]
                        profy_un = [1, 0.5, 1, 0.75]
                        prof_un = [1, 0.5, 1, 0.75]

                    # Get color from colormap
                    color = cmap(i)

                    # Plot on separate axes
                    ax_x_x.plot(
                        x_values2, slice_data_x2 + offset2, label=f'{coordinate_label2}', color=color, linestyle='-'
                    )
                    ax_y_x.plot(
                        x_values2, slice_data_y2 + offset2, label=f'{coordinate_label2}', color=color, linestyle='-'
                    )
                    ax_mag_x.plot(
                        x_values2, slice_data2 + offset2, label=f'{coordinate_label2}', color=color, linestyle='-'
                    )

                    if self.options_slice_output.uncertainty_to_plot:
                        ax_x_x.errorbar(x_un, yx_un + offset2, yerr=profx_un, fmt='none', ecolor=color, capsize=3)
                        ax_y_x.errorbar(x_un, yy_un + offset2, yerr=profy_un, fmt='none', ecolor=color, capsize=3)
                        ax_mag_x.errorbar(x_un, y_un + offset2, yerr=prof_un, fmt='none', ecolor=color, capsize=3)

                    # Accumulate data for saving
                    all_slice_data_x_x.append(slice_data_x2)
                    all_slice_data_y_x.append(slice_data_y2)
                    all_slice_data_mag_x.append(slice_data2)

                    offset_counter_x += 1  # Increase counter
                    i += 1

                # Finalize X slope plot (x slices)
                ax_x_x.set_xlabel('Y Coordinate (meters)')
                ax_x_x.set_ylabel('Slope Deviation (mrad)')
                ax_x_x.set_title('X Slope Deviation (X slices)')
                if self.options_slice_output.ylim is not None:
                    ax_x_x.set_xlim(self.options_slice_output.ylim)
                ax_x_x.legend()
                ax_x_x.grid()
                fig_x_x.savefig(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_x_x_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_x_x)

                # Finalize Y slope plot (x slices)
                ax_y_x.set_xlabel('Y Coordinate (meters)')
                ax_y_x.set_ylabel('Slope Deviation (mrad)')
                ax_y_x.set_title('Y Slope Deviation (X slices)')
                if self.options_slice_output.ylim is not None:
                    ax_y_x.set_xlim(self.options_slice_output.ylim)
                ax_y_x.legend()
                ax_y_x.grid()
                fig_y_x.savefig(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_x_y_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_y_x)

                # Finalize Magnitude plot (x slices)
                ax_mag_x.set_xlabel('Y Coordinate (meters)')
                ax_mag_x.set_ylabel('Slope Deviation (mrad)')
                ax_mag_x.set_title('Slope Deviation Magnitude (X slices)')
                if self.options_slice_output.ylim is not None:
                    ax_mag_x.set_xlim(self.options_slice_output.ylim)
                ax_mag_x.legend()
                ax_mag_x.grid()
                fig_mag_x.savefig(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_x_mag_all.png",
                    dpi=self.options_file_output.save_dpi,
                )
                plt.close(fig_mag_x)

                # Convert accumulated lists to arrays (shape: num_slices x slice_length)
                all_slice_data_x_x = np.array(all_slice_data_x_x)
                all_slice_data_y_x = np.array(all_slice_data_y_x)
                all_slice_data_mag_x = np.array(all_slice_data_mag_x)

                # Save X slope data (all x slices) to CSV and HDF5
                header_x_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
                data_to_save = np.column_stack((y_vec.flatten(), all_slice_data_x_x.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_x_x_all.csv",
                    data_to_save,
                    header=header_x_x,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_x_x_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('Y Values', data=y_vec.flatten())
                    h5f.create_dataset('X Indices', data=np.array(x_indices))
                    h5f.create_dataset('X Slope Deviation', data=all_slice_data_x_x)

                # Save Y slope data (all x slices) to CSV and HDF5
                header_y_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
                data_to_save = np.column_stack((y_vec.flatten(), all_slice_data_y_x.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_x_y_all.csv",
                    data_to_save,
                    header=header_y_x,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_x_y_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('Y Values', data=y_vec.flatten())
                    h5f.create_dataset('X Indices', data=np.array(x_indices))
                    h5f.create_dataset('Y Slope Deviation', data=all_slice_data_y_x)

                # Save Magnitude data (all x slices) to CSV and HDF5
                header_mag_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
                data_to_save = np.column_stack((y_vec.flatten(), all_slice_data_mag_x.T))
                np.savetxt(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_x_mag_all.csv",
                    data_to_save,
                    header=header_mag_x,
                    delimiter=",",
                    fmt="%.6e",
                )
                with h5py.File(
                    f"{self.options_file_output.output_dir}/slope_deviation_slice_data_x_mag_all.h5", 'w'
                ) as h5f:
                    h5f.create_dataset('Y Values', data=y_vec.flatten())
                    h5f.create_dataset('X Indices', data=np.array(x_indices))
                    h5f.create_dataset('Slope Deviation Magnitude', data=all_slice_data_mag_x)

        else:
            lt.info('Do not have both measured and reference optic; skipping slope deviation plots.')

    def _plot_ray_trace_image_measured_optic(self):
        # Plot ray trace image for measured optic
        if self._has_measured_ray_trace:
            self._plot_ray_trace_image(self._ray_trace_output_measured, 'Meas')
        else:
            lt.info('No measured ray trace data; skipping measured ray trace image.')

    def _plot_ray_trace_image_reference_optic(self):
        # Plot ray trace image for reference optic
        if self._has_reference_ray_trace:
            self._plot_ray_trace_image(self._ray_trace_output_reference, 'Ref')
        else:
            lt.info('No reference ray trace data; skipping reference ray trace image.')

    def _plot_enclosed_energy(self):
        # Makes measured and/or reference enclosed energy plots

        if (not self._has_reference_ray_trace) and (not self._has_measured_ray_trace):
            lt.info('No measured or reference ray trace data; skipping enclosed energy plot.')
            return

        # Make figure
        fig_rec = fm.setup_figure(
            self.fig_control,
            name=self.options_file_output.file_prefix + 'Ensquared Energy',
            number_in_name=self.options_file_output.number_in_name,
        )

        # Draw reference if available
        if self._has_reference_ray_trace:
            # fig_rec.axis.plot(
            #     self._ray_trace_output_reference.ensquared_energy_widths,
            #     self._ray_trace_output_reference.ensquared_energy_values,
            #     label="Reference",
            #     color="k",
            #     linestyle="--",
            # )
            widths_and_vals = list(
                zip(
                    self._ray_trace_output_reference.ensquared_energy_widths,
                    self._ray_trace_output_reference.ensquared_energy_values,
                )
            )
            fig_rec.view.draw_pq_list(widths_and_vals, style=rcee.default().theoretical, label='Reference')
        else:
            lt.info('Reference ray trace data not available, skipping reference enclosed energy curve.')

        # Draw measured if available
        if self._has_measured_ray_trace:
            # fig_rec.axis.plot(
            #     self._ray_trace_output_measured.ensquared_energy_widths,
            #     self._ray_trace_output_measured.ensquared_energy_values,
            #     label="Measured",
            #     color="k",
            #     linestyle="-",
            # )
            widths_and_vals = list(
                zip(
                    self._ray_trace_output_measured.ensquared_energy_widths,
                    self._ray_trace_output_measured.ensquared_energy_values,
                )
            )
            fig_rec.view.draw_pq_list(widths_and_vals, style=rcee.default().measured, label='Measured')
        else:
            lt.info('Measured ray trace data not available, skipping measured enclosed energy curve.')

        # Format plot
        fig_rec.axis.legend()
        fig_rec.axis.grid()
        fig_rec.axis.set_xlabel('Semi-width (meters)')
        fig_rec.axis.set_ylabel('Ensquared Energy')
        fig_rec.axis.set_title('Ensquared Energy')
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        if self.options_save.enclosed_energy:
            hdf5_filename = f"{self.options_file_output.output_dir}/enclosed_energy_image.h5"
            with h5py.File(hdf5_filename, 'w') as h5f:
                h5f.create_dataset('energy_widths', data=self._ray_trace_output_measured.ensquared_energy_widths)
                h5f.create_dataset('energy_values', data=self._ray_trace_output_measured.ensquared_energy_values)

    def _plot_curvature(self, optic: MirrorAbstract, which_data: str):
        # Separate outputs
        processings = self._process_plot_options(self.options_curvature_vis.processing)
        widths = self._process_plot_options(self.options_curvature_vis.smooth_kernel_width)

        # Curvature XY
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Curvature XY',
            number_in_name=self.options_file_output.number_in_name,
        )
        x_vec, y_vec, image = optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='xy',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[0],
            smooth_kernel_width=widths[0],
            return_data=True,  # Request the data to be returned
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        if self.options_save.curvature:
            hdf5_filename = f"{self.options_file_output.output_dir}/curvature_image_xy_{which_data}.h5"
            with h5py.File(hdf5_filename, 'w') as h5f:
                h5f.create_dataset('x_vec', data=x_vec)
                h5f.create_dataset('y_vec', data=y_vec)
                h5f.create_dataset('image', data=image)

        # Curvature X
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Curvature X',
            number_in_name=self.options_file_output.number_in_name,
        )
        x_vec, y_vec, image_x = optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='x',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[1],
            smooth_kernel_width=widths[1],
            return_data=True,  # Request the data to be returned
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        if self.options_save.curvature:
            hdf5_filename = f"{self.options_file_output.output_dir}/curvature_image_x_{which_data}.h5"
            with h5py.File(hdf5_filename, 'w') as h5f:
                h5f.create_dataset('x_vec', data=x_vec)
                h5f.create_dataset('y_vec', data=y_vec)
                h5f.create_dataset('image_x', data=image_x)

        # Curvature Y
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Curvature Y',
            number_in_name=self.options_file_output.number_in_name,
        )
        x_vec, y_vec, image_y = optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='y',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[2],
            smooth_kernel_width=widths[2],
            return_data=True,  # Request the data to be returned
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        if self.options_save.curvature:
            hdf5_filename = f"{self.options_file_output.output_dir}/curvature_image_y_{which_data}.h5"
            with h5py.File(hdf5_filename, 'w') as h5f:
                h5f.create_dataset('x_vec', data=x_vec)
                h5f.create_dataset('y_vec', data=y_vec)
                h5f.create_dataset('image_y', data=image_y)

        if self.options_slice_output.to_plot:
            x_coords = self.options_slice_output.slice_x_loc
            x_indices = coords_to_indices(x_coords, x_vec)

            y_coords = self.options_slice_output.slice_y_loc
            y_indices = coords_to_indices(y_coords, y_vec)

            offset_x = self.options_slice_output.plot_x_curvature_offset  # mrad offset between x slices
            offset_y = self.options_slice_output.plot_y_curvature_offset  # mrad offset between y slices
            offset_counter_x = 0  # Initialize counter
            offset_counter_y = 0  # Initialize counter

            # Plot contour plot with slice locations
            x_slices = [x_vec[idx] for idx in x_indices]  # x coordinate values for slices
            y_slices = [y_vec[idx] for idx in y_indices]  # y coordinate values for slices

            num_slices = max(len(x_indices), len(y_indices))
            cmap = cm.get_cmap('tab10', num_slices)

            # Initialize lists to accumulate data for all y slices
            all_slice_data_x = []
            all_slice_data_y = []
            all_slice_data_mag = []

            # Create separate figures for X curvature, Y curvature, and Magnitude
            plt.figure(figsize=(10, 6))
            plt.title('2D Line Plot of X Curvature')
            plt.xlabel('X Coordinate (meters)')
            plt.ylabel('Curvature (mrad/meter)')
            plt.grid()

            plt.figure(figsize=(10, 6))
            plt.title('2D Line Plot of Y Curvature')
            plt.xlabel('X Coordinate (meters)')
            plt.ylabel('Curvature (mrad/meter)')
            plt.grid()

            plt.figure(figsize=(10, 6))
            plt.title('2D Line Plot of Curvature Magnitude')
            plt.xlabel('X Coordinate (meters)')
            plt.ylabel('Curvature (mrad/meter)')
            plt.grid()

            # Store figure handles to plot on them
            fig_x = plt.gcf()
            fig_y = plt.gcf()
            fig_mag = plt.gcf()

            # Actually, we need separate figure objects for each plot, so create them explicitly:
            fig_x, ax_x = plt.subplots(figsize=(10, 6))
            fig_y, ax_y = plt.subplots(figsize=(10, 6))
            fig_mag, ax_mag = plt.subplots(figsize=(10, 6))

            i = 0

            for y_coordinate_index in y_indices:
                # Extract the slices
                slice_data_x1 = image_x[y_coordinate_index, :]  # X slope
                slice_data_y1 = image_y[y_coordinate_index, :]  # Y slope
                slice_data1 = image[y_coordinate_index, :]
                coordinate_label1 = f"Y Location = {y_vec[y_coordinate_index]:.4f}"
                x_values1 = x_vec  # Use x_vec for x-axis

                offset1 = offset_counter_y * offset_y

                if self.options_slice_output.uncertainty_to_plot:
                    # Define subset range for error bars
                    indx = [50, 150, 250, 350]
                    x_un = x_values1[indx]
                    yx_un = slice_data_x1[indx]
                    yy_un = slice_data_y1[indx]
                    y_un = slice_data1[indx]
                    profx_un = [1, 0.5, 1, 0.75]
                    profy_un = [1, 0.5, 1, 0.75]
                    prof_un = [1, 0.5, 1, 0.75]

                # Get color from colormap
                color = cmap(i)

                # Set linestyle based on which_data
                linestyle = '-' if which_data.lower() == 'meas' else '--'

                # Plot on separate axes
                (line_x,) = ax_x.plot(
                    x_values1[1:],
                    slice_data_x1 + offset1,
                    label=f'{coordinate_label1}',
                    color=color,
                    linestyle=linestyle,
                )
                (line_y,) = ax_y.plot(
                    x_values1, slice_data_y1 + offset1, label=f'{coordinate_label1}', color=color, linestyle=linestyle
                )
                (line_mag,) = ax_mag.plot(
                    x_values1[1:], slice_data1 + offset1, label=f'{coordinate_label1}', color=color, linestyle=linestyle
                )

                if self.options_slice_output.uncertainty_to_plot:
                    ax_x.errorbar(x_un, yx_un + offset1, yerr=profx_un, fmt='none', ecolor=color, capsize=3)
                    ax_y.errorbar(x_un, yy_un + offset1, yerr=profy_un, fmt='none', ecolor=color, capsize=3)
                    ax_mag.errorbar(x_un, y_un + offset1, yerr=prof_un, fmt='none', ecolor=color, capsize=3)

                # Accumulate data for saving
                all_slice_data_x.append(slice_data_x1)
                all_slice_data_y.append(slice_data_y1)
                all_slice_data_mag.append(slice_data1)

                offset_counter_y += 1  # Increase counter
                i += 1

            # Finalize X curvature plot
            ax_x.set_xlabel('X Coordinate (meters)')
            ax_x.set_ylabel('Curvature (mrad/meter)')
            ax_x.set_title('X Curvature (Y Slices)')
            if self.options_slice_output.xlim is not None:
                ax_x.set_xlim(self.options_slice_output.xlim)
            ax_x.legend()
            ax_x.grid()
            fig_x.savefig(
                f"{self.options_file_output.output_dir}/curvature_slice_y_x_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_x)

            # Finalize Y curvature plot
            ax_y.set_xlabel('X Coordinate (meters)')
            ax_y.set_ylabel('Curvature (mrad/meter)')
            ax_y.set_title('Y Curvature (Y Slices)')
            if self.options_slice_output.xlim is not None:
                ax_y.set_xlim(self.options_slice_output.xlim)
            ax_y.legend()
            ax_y.grid()
            fig_y.savefig(
                f"{self.options_file_output.output_dir}/curvature_slice_y_y_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_y)

            # Finalize Magnitude plot
            ax_mag.set_xlabel('X Coordinate (meters)')
            ax_mag.set_ylabel('Curvature (mrad/meter)')
            ax_mag.set_title('Curvature Magnitude (Y Slices)')
            if self.options_slice_output.xlim is not None:
                ax_mag.set_xlim(self.options_slice_output.xlim)
            ax_mag.legend()
            ax_mag.grid()
            fig_mag.savefig(
                f"{self.options_file_output.output_dir}/curvature_slice_y_mag_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_mag)

            # Convert accumulated lists to arrays (shape: num_slices x slice_length)
            all_slice_data_x = np.array(all_slice_data_x)
            all_slice_data_y = np.array(all_slice_data_y)
            all_slice_data_mag = np.array(all_slice_data_mag)

            # Save X curvature data (all slices) to CSV and HDF5
            header_x = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
            data_to_save = np.column_stack((x_vec[1:].flatten(), all_slice_data_x.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/curvature_slice_data_y_x_all_{which_data}.csv",
                data_to_save,
                header=header_x,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/curvature_slice_data_y_x_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('X Values', data=x_vec[1:].flatten())
                h5f.create_dataset('Y Indices', data=np.array(y_indices))
                h5f.create_dataset('X Curvature', data=all_slice_data_x)

            # Save Y curvature data (all slices) to CSV and HDF5
            header_y = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
            data_to_save = np.column_stack((x_vec.flatten(), all_slice_data_y.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/curvature_slice_data_y_y_all_{which_data}.csv",
                data_to_save,
                header=header_y,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/curvature_slice_data_y_y_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('X Values', data=x_vec.flatten())
                h5f.create_dataset('Y Indices', data=np.array(y_indices))
                h5f.create_dataset('Y Curvature', data=all_slice_data_y)

            # Save Magnitude data (all slices) to CSV and HDF5
            header_mag = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
            data_to_save = np.column_stack((x_vec[1:].flatten(), all_slice_data_mag.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/curvature_slice_data_y_mag_all_{which_data}.csv",
                data_to_save,
                header=header_mag,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/curvature_slice_data_y_mag_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('X Values', data=x_vec[1:].flatten())
                h5f.create_dataset('Y Indices', data=np.array(y_indices))
                h5f.create_dataset('Curvature Magnitude', data=all_slice_data_mag)

            # Initialize lists to accumulate data for all x slices
            all_slice_data_x_x = []
            all_slice_data_y_x = []
            all_slice_data_mag_x = []

            # Create separate figures for X curvature, Y curvature, and Magnitude (x slices)
            fig_x_x, ax_x_x = plt.subplots(figsize=(10, 6))
            fig_y_x, ax_y_x = plt.subplots(figsize=(10, 6))
            fig_mag_x, ax_mag_x = plt.subplots(figsize=(10, 6))

            i = 0
            for x_coordinate_index in x_indices:
                # Extract the slices
                slice_data_x2 = image_x[:, x_coordinate_index]  # X curvature difference
                slice_data_y2 = image_y[:, x_coordinate_index]  # Y curvature difference
                slice_data2 = image[:, x_coordinate_index]  # Curvature magnitude
                coordinate_label2 = f"X Location = {x_vec[x_coordinate_index]:.4f}"
                x_values2 = y_vec  # Use y_vec for x-axis

                offset2 = offset_counter_x * offset_x

                if self.options_slice_output.uncertainty_to_plot:
                    # Define subset range for error bars
                    indx = [50, 150, 250, 350]
                    x_un = x_values2[indx]
                    yx_un = slice_data_x2[indx]
                    yy_un = slice_data_y2[indx]
                    y_un = slice_data2[indx]
                    profx_un = [1, 0.5, 1, 0.75]
                    profy_un = [1, 0.5, 1, 0.75]
                    prof_un = [1, 0.5, 1, 0.75]

                # Get color from colormap
                color = cmap(i)

                # Set linestyle based on which_data
                linestyle = '-' if which_data.lower() == 'meas' else '--'

                # Plot on separate axes
                (line_x,) = ax_x_x.plot(
                    x_values2, slice_data_x2 + offset2, label=f'{coordinate_label2}', color=color, linestyle=linestyle
                )
                (line_y,) = ax_y_x.plot(
                    x_values2[1:],
                    slice_data_y2 + offset2,
                    label=f'{coordinate_label2}',
                    color=color,
                    linestyle=linestyle,
                )
                (line_mag,) = ax_mag_x.plot(
                    x_values2[1:], slice_data2 + offset2, label=f'{coordinate_label2}', color=color, linestyle=linestyle
                )

                if self.options_slice_output.uncertainty_to_plot:

                    ax_x_x.errorbar(x_un, yx_un + offset2, yerr=profx_un, fmt='none', ecolor=color, capsize=3)
                    ax_y_x.errorbar(x_un, yy_un + offset2, yerr=profy_un, fmt='none', ecolor=color, capsize=3)
                    ax_mag_x.errorbar(x_un, y_un + offset2, yerr=prof_un, fmt='none', ecolor=color, capsize=3)

                # Accumulate data for saving
                all_slice_data_x_x.append(slice_data_x2)
                all_slice_data_y_x.append(slice_data_y2)
                all_slice_data_mag_x.append(slice_data2)

                offset_counter_x += 1  # Increase counter
                i += 1

            # Finalize X curvature plot (x slices)
            ax_x_x.set_xlabel('Y Coordinate (meters)')
            ax_x_x.set_ylabel('Curvature (mrad/meter)')
            ax_x_x.set_title('X Curvature (X slices)')
            if self.options_slice_output.ylim is not None:
                ax_x_x.set_xlim(self.options_slice_output.ylim)
            ax_x_x.legend()
            ax_x_x.grid()
            fig_x_x.savefig(
                f"{self.options_file_output.output_dir}/curvature_slice_x_x_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_x_x)

            # Finalize Y curvature plot (x slices)
            ax_y_x.set_xlabel('Y Coordinate (meters)')
            ax_x_x.set_ylabel('Curvature (mrad/meter)')
            ax_y_x.set_title('Y Curvature (X slices)')
            if self.options_slice_output.ylim is not None:
                ax_y_x.set_xlim(self.options_slice_output.ylim)
            ax_y_x.legend()
            ax_y_x.grid()
            fig_y_x.savefig(
                f"{self.options_file_output.output_dir}/curvature_slice_x_y_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_y_x)

            # Finalize Magnitude plot (x slices)
            ax_mag_x.set_xlabel('Y Coordinate (meters)')
            ax_x_x.set_ylabel('Curvature (mrad/meter)')
            ax_mag_x.set_title('Curvature Magnitude (X slices)')
            if self.options_slice_output.ylim is not None:
                ax_mag_x.set_xlim(self.options_slice_output.ylim)
            ax_mag_x.legend()
            ax_mag_x.grid()
            fig_mag_x.savefig(
                f"{self.options_file_output.output_dir}/curvature_slice_x_mag_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_mag_x)

            # Convert accumulated lists to arrays (shape: num_slices x slice_length)
            all_slice_data_x_x = np.array(all_slice_data_x_x)
            all_slice_data_y_x = np.array(all_slice_data_y_x)
            all_slice_data_mag_x = np.array(all_slice_data_mag_x)

            # Save X curvature data (all x slices) to CSV and HDF5
            header_x_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
            data_to_save = np.column_stack((y_vec.flatten(), all_slice_data_x_x.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/curvature_slice_data_x_x_all_{which_data}.csv",
                data_to_save,
                header=header_x_x,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/curvature_slice_data_x_x_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('Y Values', data=y_vec.flatten())
                h5f.create_dataset('X Indices', data=np.array(x_indices))
                h5f.create_dataset('X Curvature', data=all_slice_data_x_x)

            # Save Y curvature data (all x slices) to CSV and HDF5
            header_y_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
            data_to_save = np.column_stack((y_vec[1:].flatten(), all_slice_data_y_x.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/curvature_slice_data_x_y_all_{which_data}.csv",
                data_to_save,
                header=header_y_x,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/curvature_slice_data_x_y_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('Y Values', data=y_vec[1:].flatten())
                h5f.create_dataset('X Indices', data=np.array(x_indices))
                h5f.create_dataset('Y Curvature', data=all_slice_data_y_x)

            # Save Magnitude data (all x slices) to CSV and HDF5
            header_mag_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
            data_to_save = np.column_stack((y_vec[1:].flatten(), all_slice_data_mag_x.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/curvature_slice_data_x_mag_all_{which_data}.csv",
                data_to_save,
                header=header_mag_x,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/curvature_slice_data_x_mag_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('Y Values', data=y_vec[1:].flatten())
                h5f.create_dataset('X Indices', data=np.array(x_indices))
                h5f.create_dataset('Curvature Magnitude', data=all_slice_data_mag_x)

    def _plot_slope(self, optic: MirrorAbstract, which_data: str):
        # Separate outputs
        quiver_densities = self._process_plot_options(self.options_slope_vis.quiver_density)
        quiver_scales = self._process_plot_options(self.options_slope_vis.quiver_scale)
        quiver_colors = self._process_plot_options(self.options_slope_vis.quiver_color)

        # Slope Magnitude
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Slope XY',
            number_in_name=self.options_file_output.number_in_name,
        )
        x_vec, y_vec, image = optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='xy',
            quiver_density=quiver_densities[0],
            quiver_scale=quiver_scales[0],
            quiver_color=quiver_colors[0],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
            return_data=True,  # Request the data to be returned
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        if self.options_save.slope:
            hdf5_filename = f"{self.options_file_output.output_dir}/slope_image_xy_{which_data}.h5"
            with h5py.File(hdf5_filename, 'w') as h5f:
                h5f.create_dataset('x_vec', data=x_vec)
                h5f.create_dataset('y_vec', data=y_vec)
                h5f.create_dataset('image', data=image)

        # X Slope
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Slope X',
            number_in_name=self.options_file_output.number_in_name,
        )
        x_vec, y_vec, image_x = optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='x',
            quiver_density=quiver_densities[1],
            quiver_scale=quiver_scales[1],
            quiver_color=quiver_colors[1],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
            return_data=True,  # Request the data to be returned
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        if self.options_save.slope:
            hdf5_filename = f"{self.options_file_output.output_dir}/slope_image_x_{which_data}.h5"
            with h5py.File(hdf5_filename, 'w') as h5f:
                h5f.create_dataset('x_vec', data=x_vec)
                h5f.create_dataset('y_vec', data=y_vec)
                h5f.create_dataset('image_x', data=image_x)

        # Y Slope
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Slope Y',
            number_in_name=self.options_file_output.number_in_name,
        )
        x_vec, y_vec, image_y = optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='y',
            quiver_density=quiver_densities[2],
            quiver_scale=quiver_scales[2],
            quiver_color=quiver_colors[2],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
            return_data=True,  # Request the data to be returned
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        if self.options_save.slope:
            hdf5_filename = f"{self.options_file_output.output_dir}/slope_image_y_{which_data}.h5"
            with h5py.File(hdf5_filename, 'w') as h5f:
                h5f.create_dataset('x_vec', data=x_vec)
                h5f.create_dataset('y_vec', data=y_vec)
                h5f.create_dataset('image_y', data=image_y)

        if self.options_slice_output.to_plot:

            x_coords = self.options_slice_output.slice_x_loc
            x_indices = coords_to_indices(x_coords, x_vec)

            y_coords = self.options_slice_output.slice_y_loc
            y_indices = coords_to_indices(y_coords, y_vec)

            offset_x = self.options_slice_output.plot_x_slope_offset  # mrad offset between x slices
            offset_y = self.options_slice_output.plot_y_slope_offset  # mrad offset between y slices
            offset_counter_x = 0  # Initialize counter
            offset_counter_y = 0  # Initialize counter

            # Plot contour plot with slice locations
            x_slices = [x_vec[idx] for idx in x_indices]  # x coordinate values for slices
            y_slices = [y_vec[idx] for idx in y_indices]  # y coordinate values for slices

            num_slices = max(len(x_indices), len(y_indices))
            cmap = cm.get_cmap('tab10', num_slices)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Create contour plot
            contour = ax.contourf(x_vec, y_vec, image, levels=50, cmap='jet')  # transpose image if needed
            fig.colorbar(contour, ax=ax, label='Slope Magnitude (mrad)')

            # Determine line style based on which_data
            linestyle = '-' if which_data.lower() == 'meas' else '--'

            # Add vertical lines for x slices
            i = 0
            for x_val in x_slices:
                color = cmap(i)
                ax.axvline(x=x_val, color=color, linestyle=linestyle, linewidth=1.5, label=f'x = {x_val:.3f}')
                i += 1

            # Add horizontal lines for y slices
            i = 0
            for y_val in y_slices:
                color = cmap(i)
                ax.axhline(y=y_val, color=color, linestyle=linestyle, linewidth=1.5, label=f'y = {y_val:.3f}')
                i += 1

            # To avoid duplicate labels in legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')

            ax.set_xlabel('X Coordinate (meters)')
            ax.set_ylabel('Y Coordinate (meters)')
            ax.set_title('Slope Plot with Slice Lines')

            fig.savefig(
                f"{self.options_file_output.output_dir}/slice_lines_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )

            # Initialize lists to accumulate data for all y slices
            all_slice_data_x = []
            all_slice_data_y = []
            all_slice_data_mag = []

            # Create separate figures for X slope, Y slope, and Magnitude
            plt.figure(figsize=(10, 6))
            plt.title('2D Line Plot of X Slope')
            plt.xlabel('X Coordinate (meters)')
            plt.ylabel('Slope (mrad)')
            plt.grid()

            plt.figure(figsize=(10, 6))
            plt.title('2D Line Plot of Y Slope')
            plt.xlabel('X Coordinate (meters)')
            plt.ylabel('Slope (mrad)')
            plt.grid()

            plt.figure(figsize=(10, 6))
            plt.title('2D Line Plot of Slope Magnitude')
            plt.xlabel('X Coordinate (meters)')
            plt.ylabel('Slope (mrad)')
            plt.grid()

            # Store figure handles to plot on them
            fig_x = plt.gcf()
            fig_y = plt.gcf()
            fig_mag = plt.gcf()

            # Actually, we need separate figure objects for each plot, so create them explicitly:
            fig_x, ax_x = plt.subplots(figsize=(10, 6))
            fig_y, ax_y = plt.subplots(figsize=(10, 6))
            fig_mag, ax_mag = plt.subplots(figsize=(10, 6))

            i = 0

            for y_coordinate_index in y_indices:
                # Extract the slices
                slice_data_x1 = image_x[y_coordinate_index, :]  # X slope
                slice_data_y1 = image_y[y_coordinate_index, :]  # Y slope
                slice_data1 = image[y_coordinate_index, :]
                coordinate_label1 = f"Y Location = {y_vec[y_coordinate_index]:.4f}"
                x_values1 = x_vec  # Use x_vec for x-axis

                offset1 = offset_counter_y * offset_y

                if self.options_slice_output.uncertainty_to_plot:

                    # Define subset range for error bars
                    indx = [50, 150, 250, 350]
                    x_un = x_values1[indx]
                    yx_un = slice_data_x1[indx]
                    yy_un = slice_data_y1[indx]
                    y_un = slice_data1[indx]
                    profx_un = [1, 0.5, 1, 0.75]
                    profy_un = [1, 0.5, 1, 0.75]
                    prof_un = [1, 0.5, 1, 0.75]

                # Get color from colormap
                color = cmap(i)

                # Set linestyle based on which_data
                linestyle = '-' if which_data.lower() == 'meas' else '--'

                # Plot on separate axes
                (line_x,) = ax_x.plot(
                    x_values1, slice_data_x1 + offset1, label=f'{coordinate_label1}', color=color, linestyle=linestyle
                )
                (line_y,) = ax_y.plot(
                    x_values1, slice_data_y1 + offset1, label=f'{coordinate_label1}', color=color, linestyle=linestyle
                )
                (line_mag,) = ax_mag.plot(
                    x_values1, slice_data1 + offset1, label=f'{coordinate_label1}', color=color, linestyle=linestyle
                )

                if self.options_slice_output.uncertainty_to_plot:
                    ax_x.errorbar(x_un, yx_un + offset1, yerr=profx_un, fmt='none', ecolor=color, capsize=3)
                    ax_y.errorbar(x_un, yy_un + offset1, yerr=profy_un, fmt='none', ecolor=color, capsize=3)
                    ax_mag.errorbar(x_un, y_un + offset1, yerr=prof_un, fmt='none', ecolor=color, capsize=3)

                # Accumulate data for saving
                all_slice_data_x.append(slice_data_x1)
                all_slice_data_y.append(slice_data_y1)
                all_slice_data_mag.append(slice_data1)

                offset_counter_y += 1  # Increase counter
                i += 1

            # Finalize X slope plot
            ax_x.set_xlabel('X Coordinate (meters)')
            ax_x.set_ylabel('Slope (mrad)')
            ax_x.set_title('X Slope (Y Slices)')
            if self.options_slice_output.xlim is not None:
                ax_x.set_xlim(self.options_slice_output.xlim)
            ax_x.legend()
            ax_x.grid()
            fig_x.savefig(
                f"{self.options_file_output.output_dir}/slope_slice_y_x_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_x)

            # Finalize Y slope plot
            ax_y.set_xlabel('X Coordinate (meters)')
            ax_y.set_ylabel('Slope (mrad)')
            ax_y.set_title('Y Slope (Y Slices)')
            if self.options_slice_output.xlim is not None:
                ax_y.set_xlim(self.options_slice_output.xlim)
            ax_y.legend()
            ax_y.grid()
            fig_y.savefig(
                f"{self.options_file_output.output_dir}/slope_slice_y_y_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_y)

            # Finalize Magnitude plot
            ax_mag.set_xlabel('X Coordinate (meters)')
            ax_mag.set_ylabel('Slope (mrad)')
            ax_mag.set_title('Slope Magnitude (Y Slices)')
            if self.options_slice_output.xlim is not None:
                ax_mag.set_xlim(self.options_slice_output.xlim)
            ax_mag.legend()
            ax_mag.grid()
            fig_mag.savefig(
                f"{self.options_file_output.output_dir}/slope_slice_y_mag_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_mag)

            # Convert accumulated lists to arrays (shape: num_slices x slice_length)
            all_slice_data_x = np.array(all_slice_data_x)
            all_slice_data_y = np.array(all_slice_data_y)
            all_slice_data_mag = np.array(all_slice_data_mag)

            # Save X slope data (all slices) to CSV and HDF5
            header_x = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
            data_to_save = np.column_stack((x_vec.flatten(), all_slice_data_x.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/slope_slice_data_y_x_all_{which_data}.csv",
                data_to_save,
                header=header_x,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/slope_slice_data_y_x_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('X Values', data=x_vec.flatten())
                h5f.create_dataset('Y Indices', data=np.array(y_indices))
                h5f.create_dataset('X Slope', data=all_slice_data_x)

            # Save Y slope data (all slices) to CSV and HDF5
            header_y = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
            data_to_save = np.column_stack((x_vec.flatten(), all_slice_data_y.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/slope_slice_data_y_y_all_{which_data}.csv",
                data_to_save,
                header=header_y,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/slope_slice_data_y_y_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('X Values', data=x_vec.flatten())
                h5f.create_dataset('Y Indices', data=np.array(y_indices))
                h5f.create_dataset('Y Slope', data=all_slice_data_y)

            # Save Magnitude data (all slices) to CSV and HDF5
            header_mag = 'X Values,' + ','.join([f'Y Location = {y_vec[idx]:.4f}' for idx in y_indices])
            data_to_save = np.column_stack((x_vec.flatten(), all_slice_data_mag.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/slope_slice_data_y_mag_all_{which_data}.csv",
                data_to_save,
                header=header_mag,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/slope_slice_data_y_mag_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('X Values', data=x_vec.flatten())
                h5f.create_dataset('Y Indices', data=np.array(y_indices))
                h5f.create_dataset('Slope Magnitude', data=all_slice_data_mag)

            # Initialize lists to accumulate data for all x slices
            all_slice_data_x_x = []
            all_slice_data_y_x = []
            all_slice_data_mag_x = []

            # Create separate figures for X slope, Y slope, and Magnitude (x slices)
            fig_x_x, ax_x_x = plt.subplots(figsize=(10, 6))
            fig_y_x, ax_y_x = plt.subplots(figsize=(10, 6))
            fig_mag_x, ax_mag_x = plt.subplots(figsize=(10, 6))

            i = 0
            for x_coordinate_index in x_indices:
                # Extract the slices
                slice_data_x2 = image_x[:, x_coordinate_index]  # X slope difference
                slice_data_y2 = image_y[:, x_coordinate_index]  # Y slope difference
                slice_data2 = image[:, x_coordinate_index]  # Slope magnitude
                coordinate_label2 = f"X Location = {x_vec[x_coordinate_index]:.4f}"
                x_values2 = y_vec  # Use y_vec for x-axis

                offset2 = offset_counter_x * offset_x

                if self.options_slice_output.uncertainty_to_plot:
                    # Define subset range for error bars
                    indx = [50, 150, 250, 350]
                    x_un = x_values2[indx]
                    yx_un = slice_data_x2[indx]
                    yy_un = slice_data_y2[indx]
                    y_un = slice_data2[indx]
                    profx_un = [1, 0.5, 1, 0.75]
                    profy_un = [1, 0.5, 1, 0.75]
                    prof_un = [1, 0.5, 1, 0.75]

                # Get color from colormap
                color = cmap(i)

                # Set linestyle based on which_data
                linestyle = '-' if which_data.lower() == 'meas' else '--'

                # Plot on separate axes
                ax_x_x.plot(
                    x_values2, slice_data_x2 + offset2, label=f'{coordinate_label2}', color=color, linestyle=linestyle
                )
                ax_y_x.plot(
                    x_values2, slice_data_y2 + offset2, label=f'{coordinate_label2}', color=color, linestyle=linestyle
                )
                ax_mag_x.plot(
                    x_values2, slice_data2 + offset2, label=f'{coordinate_label2}', color=color, linestyle=linestyle
                )

                if self.options_slice_output.uncertainty_to_plot:
                    ax_x_x.errorbar(x_un, yx_un + offset2, yerr=profx_un, fmt='none', ecolor=color, capsize=3)
                    ax_y_x.errorbar(x_un, yy_un + offset2, yerr=profy_un, fmt='none', ecolor=color, capsize=3)
                    ax_mag_x.errorbar(x_un, y_un + offset2, yerr=prof_un, fmt='none', ecolor=color, capsize=3)

                # Accumulate data for saving
                all_slice_data_x_x.append(slice_data_x2)
                all_slice_data_y_x.append(slice_data_y2)
                all_slice_data_mag_x.append(slice_data2)

                offset_counter_x += 1  # Increase counter
                i += 1

            # Finalize X slope plot (x slices)
            ax_x_x.set_xlabel('Y Coordinate (meters)')
            ax_x_x.set_ylabel('Slope (mrad)')
            ax_x_x.set_title('X Slope (X slices)')
            if self.options_slice_output.ylim is not None:
                ax_x_x.set_xlim(self.options_slice_output.ylim)
            ax_x_x.legend()
            ax_x_x.grid()
            fig_x_x.savefig(
                f"{self.options_file_output.output_dir}/slope_slice_x_x_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_x_x)

            # Finalize Y slope plot (x slices)
            ax_y_x.set_xlabel('Y Coordinate (meters)')
            ax_y_x.set_ylabel('Slope (mrad)')
            ax_y_x.set_title('Y Slope (X slices)')
            if self.options_slice_output.ylim is not None:
                ax_y_x.set_xlim(self.options_slice_output.ylim)
            ax_y_x.legend()
            ax_y_x.grid()
            fig_y_x.savefig(
                f"{self.options_file_output.output_dir}/slope_slice_x_y_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_y_x)

            # Finalize Magnitude plot (x slices)
            ax_mag_x.set_xlabel('Y Coordinate (meters)')
            ax_mag_x.set_ylabel('Slope (mrad)')
            ax_mag_x.set_title('Slope Magnitude (X slices)')
            if self.options_slice_output.ylim is not None:
                ax_mag_x.set_xlim(self.options_slice_output.ylim)
            ax_mag_x.legend()
            ax_mag_x.grid()
            fig_mag_x.savefig(
                f"{self.options_file_output.output_dir}/slope_slice_x_mag_all_{which_data}.png",
                dpi=self.options_file_output.save_dpi,
            )
            plt.close(fig_mag_x)

            # Convert accumulated lists to arrays (shape: num_slices x slice_length)
            all_slice_data_x_x = np.array(all_slice_data_x_x)
            all_slice_data_y_x = np.array(all_slice_data_y_x)
            all_slice_data_mag_x = np.array(all_slice_data_mag_x)

            # Save X slope data (all x slices) to CSV and HDF5
            header_x_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
            data_to_save = np.column_stack((y_vec.flatten(), all_slice_data_x_x.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/slope_slice_data_x_x_all_{which_data}.csv",
                data_to_save,
                header=header_x_x,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/slope_slice_data_x_x_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('Y Values', data=y_vec.flatten())
                h5f.create_dataset('X Indices', data=np.array(x_indices))
                h5f.create_dataset('X Slope', data=all_slice_data_x_x)

            # Save Y slope data (all x slices) to CSV and HDF5
            header_y_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
            data_to_save = np.column_stack((y_vec.flatten(), all_slice_data_y_x.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/slope_slice_data_x_y_all_{which_data}.csv",
                data_to_save,
                header=header_y_x,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/slope_slice_data_x_y_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('Y Values', data=y_vec.flatten())
                h5f.create_dataset('X Indices', data=np.array(x_indices))
                h5f.create_dataset('Y Slope', data=all_slice_data_y_x)

            # Save Magnitude data (all x slices) to CSV and HDF5
            header_mag_x = 'Y Values,' + ','.join([f'X location = {x_vec[idx]:.4f}' for idx in x_indices])
            data_to_save = np.column_stack((y_vec.flatten(), all_slice_data_mag_x.T))
            np.savetxt(
                f"{self.options_file_output.output_dir}/slope_slice_data_x_mag_all_{which_data}.csv",
                data_to_save,
                header=header_mag_x,
                delimiter=",",
                fmt="%.6e",
            )
            with h5py.File(
                f"{self.options_file_output.output_dir}/slope_slice_data_x_mag_all_{which_data}.h5", 'w'
            ) as h5f:
                h5f.create_dataset('Y Values', data=y_vec.flatten())
                h5f.create_dataset('X Indices', data=np.array(x_indices))
                h5f.create_dataset('Slope Magnitude', data=all_slice_data_mag_x)

    def _plot_ray_trace_image(self, ray_trace_data: _RayTraceOutput, which_data: str):
        # Draw sun image on target
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Ray Trace Image',
            number_in_name=self.options_file_output.number_in_name,
        )
        fig_rec.axis.imshow(
            ray_trace_data.histogram,
            cmap='jet',
            extent=(
                ray_trace_data.histogram_x.min(),
                ray_trace_data.histogram_x.max(),
                ray_trace_data.histogram_y.min(),
                ray_trace_data.histogram_y.max(),
            ),
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

    def _perform_ray_trace_optic_measured(self):
        # Performs ray trace on measured optic
        if self._has_measured_optic:
            # Perfom ray trace and intersection
            ray_trace = self._ray_trace_scene(self.optic_measured)
            ray_pts_meas = rt.plane_intersect(
                ray_trace, self.params_ray_trace.v_target_center, self.params_ray_trace.v_target_normal
            )

            # Create image
            image, xv, yv = rt.histogram_image(
                bin_res=self.options_ray_trace_vis.hist_bin_res,
                extent=self.options_ray_trace_vis.hist_extent,
                pts=ray_pts_meas,
            )

            # Create ensquared energy curve
            ee, ws = rt.ensquared_energy(ray_pts_meas, self.options_ray_trace_vis.enclosed_energy_max_semi_width)

            # Save
            self._ray_trace_output_measured = _RayTraceOutput(ray_trace, image, xv, yv, ee, ws)
        else:
            lt.info('No measured optic; skipping measured optic ray trace.')

    def _perform_ray_trace_optic_reference(self):
        if self._has_reference_optic:
            # Perform ray trace and intersection
            ray_trace = self._ray_trace_scene(self.optic_reference)
            ray_pts = rt.plane_intersect(
                ray_trace, self.params_ray_trace.v_target_center, self.params_ray_trace.v_target_normal
            )

            # Create image
            image, xv, yv = rt.histogram_image(
                bin_res=self.options_ray_trace_vis.hist_bin_res,
                extent=self.options_ray_trace_vis.hist_extent,
                pts=ray_pts,
            )

            # Create ensquared energy curve
            ee, ws = rt.ensquared_energy(ray_pts, self.options_ray_trace_vis.enclosed_energy_max_semi_width)

            # Save data
            self._ray_trace_output_reference = _RayTraceOutput(ray_trace, image, xv, yv, ee, ws)
        else:
            lt.info('No reference optic; skipping reference optic ray trace.')

    def _ray_trace_scene(self, obj: RayTraceable) -> rt.RayTrace:
        # Performs a raytrace of a simple scene with a source and an optic.
        # Input an optic (obj) and a RayTrace object is returned.

        # Create scene with source and optic
        scene = Scene()
        scene.add_light_source(self.params_ray_trace.source)
        scene.add_object(obj)

        # Trace scene
        res = Resolution.separation(self.options_ray_trace_vis.ray_trace_optic_res)
        trace = rt.trace_scene(scene, obj_resolution=res)

        # Calculate intersection with plane
        ray_trace = rt.RayTrace(scene)
        ray_trace.add_many_light_paths(trace.light_paths)

        return ray_trace


def coords_to_indices(coords, vec):
    if isinstance(coords, (float, int)):
        coords = [coords]
    indices = []
    max_index = len(vec) - 2  # Because image_y is smaller by 1, subtract 1 more for safety
    for c in coords:
        idx = np.abs(vec - c).argmin()
        idx = min(idx, max_index)  # Clip to max valid index
        indices.append(idx)
    return indices
