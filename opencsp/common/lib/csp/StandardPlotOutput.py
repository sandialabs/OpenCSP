"""Class used to display/save the suite of standard output plots after measuring a CSP Optic object."""

import configparser
from dataclasses import dataclass, field
import numpy as np

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
        else:
            lt.info('Slope plotting turned off; skipping measured/reference optic slope plots.')

        if self.options_curvature_vis.to_plot:
            self._plot_curvature_measured_optic()
            self._plot_curvature_reference_optic()
        else:
            lt.info('Curvature plotting turned off; skipping measured/reference optic curvature plots.')

        if self.options_slope_deviation_vis.to_plot:
            self._plot_slope_deviation()
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
            self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='xy',
                quiver_density=quiver_densities[0],
                quiver_scale=quiver_scales[0],
                quiver_color=quiver_colors[0],
                clim=self.options_slope_deviation_vis.clim,
                axis=fig_rec.axis,
            )
            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )

            # Slope x
            fig_rec = fm.setup_figure(
                self.fig_control,
                self.axis_control,
                name=self.options_file_output.file_prefix + "Slope Deviation X",
                number_in_name=self.options_file_output.number_in_name,
            )
            self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='x',
                quiver_density=quiver_densities[1],
                quiver_scale=quiver_scales[1],
                quiver_color=quiver_colors[1],
                clim=self.options_slope_deviation_vis.clim,
                axis=fig_rec.axis,
            )
            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )

            # Slope Y
            fig_rec = fm.setup_figure(
                self.fig_control,
                self.axis_control,
                name=self.options_file_output.file_prefix + "Slope Deviation Y",
                number_in_name=self.options_file_output.number_in_name,
            )
            self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='y',
                quiver_density=quiver_densities[2],
                quiver_scale=quiver_scales[2],
                quiver_color=quiver_colors[2],
                clim=self.options_slope_deviation_vis.clim,
                axis=fig_rec.axis,
            )
            if self.options_file_output.to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                    include_view_suffix=False,
                )
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
        optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='xy',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[0],
            smooth_kernel_width=widths[0],
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        # Curvature X
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Curvature X',
            number_in_name=self.options_file_output.number_in_name,
        )
        optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='x',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[1],
            smooth_kernel_width=widths[1],
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        # Curvature Y
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Curvature Y',
            number_in_name=self.options_file_output.number_in_name,
        )
        optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='y',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[2],
            smooth_kernel_width=widths[2],
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

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
        optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='xy',
            quiver_density=quiver_densities[0],
            quiver_scale=quiver_scales[0],
            quiver_color=quiver_colors[0],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        # X Slope
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Slope X',
            number_in_name=self.options_file_output.number_in_name,
        )
        optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='x',
            quiver_density=quiver_densities[1],
            quiver_scale=quiver_scales[1],
            quiver_color=quiver_colors[1],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

        # Y Slope
        fig_rec = fm.setup_figure(
            self.fig_control,
            self.axis_control,
            name=self.options_file_output.file_prefix + which_data + ' Slope Y',
            number_in_name=self.options_file_output.number_in_name,
        )
        optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='y',
            quiver_density=quiver_densities[2],
            quiver_scale=quiver_scales[2],
            quiver_color=quiver_colors[2],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
        )
        if self.options_file_output.to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
                include_view_suffix=False,
            )

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
