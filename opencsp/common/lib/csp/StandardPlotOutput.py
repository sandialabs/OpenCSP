"""Class used to display/save the suite of standard output plots after measuring a CSP Mirror/FacetEnsemble.
"""

from dataclasses import dataclass, field

import numpy as np

import opencsp.common.lib.render_control.RenderControlAxis as rca
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlLightPath import RenderControlLightPath
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
import opencsp.common.lib.csp.RayTrace as rt
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.tool.log_tools as lt


@dataclass
class _OptionsSlopeVis:
    resolution: float = 0.01
    clim: float = 5
    quiver_density: float | tuple[float, float, float] = 0.1
    quiver_scale: float | tuple[float, float, float] = 25
    quiver_color: str | tuple[str, str, str] = 'white'
    deviation_clim: float = 5
    to_plot_slope: bool = True
    to_plot_slope_deviation: bool = True


@dataclass
class _OptionsCurvatureVis:
    resolution: float = 0.01
    clim: float = 50
    processing: list[str] | tuple[list[str], list[str], list[str]] = field(default_factory=list)
    smooth_kernel_width: float | tuple[float, float, float] = 1
    to_plot: bool = True


@dataclass
class _OptionsRayTraceVis:
    to_ray_trace: bool = True
    ray_trace_optic_res: int = 20
    hist_bin_res: float = 0.07
    hist_extent: float = 3
    ensquared_energy_max_semi_width: float = 2
    ray_trace_plot_ray_length: float = 80


@dataclass
class _OptionsFileOutput:
    to_save: bool = False
    output_dir: str = ''
    save_dpi: int = 200
    save_format: str = 'png'
    close_after_save: bool = False


@dataclass
class _RayTraceParameters:
    source = LightSourceSun.from_given_sun_position(Uxyz((0, 0, -1)), resolution=20)
    v_target_center = Vxyz((0, 0, 50))
    v_target_normal = Vxyz((0, 0, -1))


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
        self.point_styles = RenderControlPointSeq(linestyle='--', color='k', markersize=0)
        self.mirror_control = RenderControlMirror(surface_normals=True, norm_res=1, point_styles=self.point_styles)
        self.light_path_control = RenderControlLightPath(
            current_length=self.options_ray_trace_vis.ray_trace_plot_ray_length
        )
        self.ray_trace_control = RenderControlRayTrace(light_path_control=self.light_path_control)

        # Define output data storage classes
        self._ray_trace_output_measured: _RayTraceOutput = None
        self._ray_trace_output_reference: _RayTraceOutput = None

    @property
    def _is_reference_optic(self) -> bool:
        """To make reference optic plots"""
        return self.optic_reference is not None

    @property
    def _to_save(self) -> bool:
        """To save plots or not"""
        return self.options_file_output.to_save

    def plot(self):
        """Creates standard output plot suite"""
        # Plot slope/curvature, if able
        self._plot_slope_curvature_measured_optic()
        self._plot_slope_curvature_reference_optic()
        self._plot_slope_deviation()

        if self.options_ray_trace_vis.to_ray_trace:
            # Perform ray tracing, if set
            self._perform_ray_trace_optic_measured()
            self._perform_ray_trace_optic_reference()
            # Plot ray trace data
            self._plot_ray_trace_image_measured_optic()
            self._plot_ray_trace_image_reference_optic()
            self._plot_enclosed_energy()
        else:
            lt.info(
                'Ray tracing turned off in self.options_ray_trace_vis.to_ray_trace; skipping all ray tracing plots.'
            )

    def _plot_slope_curvature_measured_optic(self):
        """Plots optic slope/curvature plots for measured optic"""
        if self.optic_measured is None:
            lt.info('No measured optic; skipping measured optic slope/curvature plots.')
            return

        if self.options_slope_vis:
            self._plot_slope(self.optic_measured, 'measured')
        else:
            lt.info('Slope plotting turned off; skipping measured optic slope plots.')

        if self.options_curvature_vis.to_plot:
            self._plot_curvature(self.optic_measured, 'measured')
        else:
            lt.info('Curvature plotting turned off; skipping measured optic curvature plots.')

    def _plot_slope_curvature_reference_optic(self):
        """Plots optic slope/curvature plots for reference optic"""
        if self.optic_reference is None:
            lt.info('No reference optic; skipping reference optic slope/curvature plots.')
            return

        if self.options_slope_vis:
            self._plot_slope(self.optic_reference, 'reference')
        else:
            lt.info('Slope plotting turned off; skipping reference optic slope plots.')

        if self.options_curvature_vis.to_plot:
            self._plot_curvature(self.optic_reference, 'reference')
        else:
            lt.info('Curvature plotting turned off; skipping reference optic curvature plots.')

    def _process_plot_options(self, value) -> list:
        if isinstance(value, (tuple, list)):
            if len(value) == 3:
                return value
            elif len(value) in [0, 1]:
                return [value] * 3
            else:
                lt.error_and_raise(ValueError, f'Plot option must be length 3 or 1, not length {len(value):d}')
        else:
            return [value] * 3

    def _plot_slope_deviation(self):
        """Plots slope deviation"""
        if not self.options_slope_vis.to_plot_slope_deviation:
            lt.info('Slope deviation plotting turned off; skipping slope deviation plots.')

        if (self.optic_measured is not None) and (self.optic_reference is not None):
            # Separate outputs
            quiver_densities = self._process_plot_options(self.options_slope_vis.quiver_density)
            quiver_scales = self._process_plot_options(self.options_slope_vis.quiver_scale)
            quiver_colors = self._process_plot_options(self.options_slope_vis.quiver_color)

            # Slope magnitude
            fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name="Slope Deviation Magnitude")
            self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='magnitude',
                quiver_density=quiver_densities[0],
                quiver_scale=quiver_scales[0],
                quiver_color=quiver_colors[0],
                clim=self.options_slope_vis.deviation_clim,
                axis=fig_rec.axis,
            )
            if self._to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                )

            # Slope x
            fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name="Slope Deviation X")
            self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='x',
                quiver_density=quiver_densities[1],
                quiver_scale=quiver_scales[1],
                quiver_color=quiver_colors[1],
                clim=self.options_slope_vis.deviation_clim,
                axis=fig_rec.axis,
            )
            if self._to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                )

            # Slope Y
            fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name="Slope Deviation Y")
            self.optic_measured.plot_orthorectified_slope_error(
                self.optic_reference,
                self.options_slope_vis.resolution,
                type_='y',
                quiver_density=quiver_densities[2],
                quiver_scale=quiver_scales[2],
                quiver_color=quiver_colors[2],
                clim=self.options_slope_vis.deviation_clim,
                axis=fig_rec.axis,
            )
            if self._to_save:
                fig_rec.save(
                    output_dir=self.options_file_output.output_dir,
                    dpi=self.options_file_output.save_dpi,
                    format='png',
                    close_after_save=self.options_file_output.close_after_save,
                )
        else:
            lt.info('Do not have both measured and reference optic; skipping slope deviation plots.')

    def _plot_ray_trace_image_measured_optic(self):
        """Plot ray trace image for measured optic"""
        if self._ray_trace_output_measured is not None:
            self._plot_ray_trace_image(self._ray_trace_output_measured, 'measured')

    def _plot_ray_trace_image_reference_optic(self):
        """Plot ray trace image for reference optic"""
        if self._ray_trace_output_reference is not None:
            self._plot_ray_trace_image(self._ray_trace_output_reference, 'reference')

    def _plot_enclosed_energy(self):
        """Makes measured and/or reference enclosed energy plots"""
        fig_rec = fm.setup_figure(self.fig_control, name='Ensquared Energy')

        # Dray reference if available
        if self._ray_trace_output_reference is not None:
            fig_rec.axis.plot(
                self._ray_trace_output_reference.ensquared_energy_widths,
                self._ray_trace_output_reference.ensquared_energy_values,
                label='Reference',
                color='k',
                linestyle='--',
            )
        else:
            lt.info('Reference ray trace data not available, skipping reference enclosed energy curve.')

        # Draw measured if available
        if self._ray_trace_output_measured is not None:
            fig_rec.axis.plot(
                self._ray_trace_output_measured.ensquared_energy_widths,
                self._ray_trace_output_measured.ensquared_energy_values,
                label='Measured',
                color='k',
                linestyle='-',
            )
        else:
            lt.info('Measured ray trace data not available, skipping measured enclosed energy curve.')

        # Format plot
        fig_rec.axis.legend()
        fig_rec.axis.grid()
        fig_rec.axis.set_xlabel('Semi-width (meters)')
        fig_rec.axis.set_ylabel('Ensquared Energy')
        fig_rec.axis.set_title('Ensquared Energy')
        if self._to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
            )

    def _plot_curvature(self, optic: MirrorAbstract, suffix: str):
        # Separate outputs
        processings = self._process_plot_options(self.options_curvature_vis.processing)
        widths = self._process_plot_options(self.options_curvature_vis.smooth_kernel_width)

        # Curvature combined
        fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name='Curvature Combined ' + suffix)
        optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='combined',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[0],
            smooth_kernel_width=widths[0],
        )
        if self._to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
            )

        # Curvature X
        fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name='Curvature X ' + suffix)
        optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='x',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[1],
            smooth_kernel_width=widths[1],
        )
        if self._to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
            )

        # Curvature Y
        fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name='Curvature Y ' + suffix)
        optic.plot_orthorectified_curvature(
            res=self.options_curvature_vis.resolution,
            type_='y',
            clim=self.options_curvature_vis.clim,
            axis=fig_rec.axis,
            processing=processings[2],
            smooth_kernel_width=widths[2],
        )
        if self._to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
            )

    def _plot_slope(self, optic: MirrorAbstract, suffix: str):
        # Separate outputs
        quiver_densities = self._process_plot_options(self.options_slope_vis.quiver_density)
        quiver_scales = self._process_plot_options(self.options_slope_vis.quiver_scale)
        quiver_colors = self._process_plot_options(self.options_slope_vis.quiver_color)

        # Slope Magnitude
        fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name="Slope Magnitude " + suffix)
        optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='magnitude',
            quiver_density=quiver_densities[0],
            quiver_scale=quiver_scales[0],
            quiver_color=quiver_colors[0],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
        )
        if self._to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
            )

        # X Slope
        fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name="Slope X " + suffix)
        optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='x',
            quiver_density=quiver_densities[1],
            quiver_scale=quiver_scales[1],
            quiver_color=quiver_colors[1],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
        )
        if self._to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
            )

        # Y Slope
        fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name="Slope Y " + suffix)
        optic.plot_orthorectified_slope(
            self.options_slope_vis.resolution,
            type_='y',
            quiver_density=quiver_densities[2],
            quiver_scale=quiver_scales[2],
            quiver_color=quiver_colors[2],
            clim=self.options_slope_vis.clim,
            axis=fig_rec.axis,
        )
        if self._to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
            )

    def _plot_ray_trace_image(self, ray_trace_data: _RayTraceOutput, suffix: str):
        # Draw sun image on target
        fig_rec = fm.setup_figure(self.fig_control, self.axis_control, name='Ray Trace Image ' + suffix)
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
        if self._to_save:
            fig_rec.save(
                output_dir=self.options_file_output.output_dir,
                dpi=self.options_file_output.save_dpi,
                format='png',
                close_after_save=self.options_file_output.close_after_save,
            )

    def _perform_ray_trace_optic_measured(self):
        """Performs ray trace on measured optic"""
        if self.optic_measured is not None:
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
            ee, ws = rt.ensquared_energy(ray_pts_meas, self.options_ray_trace_vis.ensquared_energy_max_semi_width)

            # Save
            self._ray_trace_output_measured = _RayTraceOutput(ray_trace, image, xv, yv, ee, ws)
        else:
            lt.info('No measured optic; skipping measured optic ray trace.')

    def _perform_ray_trace_optic_reference(self):
        if self.optic_reference is not None:
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
            ee, ws = rt.ensquared_energy(ray_pts, self.options_ray_trace_vis.ensquared_energy_max_semi_width)

            # Save data
            self._ray_trace_output_reference = _RayTraceOutput(ray_trace, image, xv, yv, ee, ws)
        else:
            lt.info('No reference optic; skipping reference optic ray trace.')

    def _ray_trace_scene(self, obj: RayTraceable) -> rt.RayTrace:
        """Performs a raytrace of a simple scene with a source and an optic.

        Parameters
        ----------
        obj : RayTraceable
            Optic
        source : LightSource
            Source
        obj_resolution : int, optional
            Mirror ray resolution, by default 1

        Returns
        -------
        RayTrace
            Output raytrace object
        """
        # Create scene with source and optic
        scene = Scene()
        scene.add_light_source(self.params_ray_trace.source)
        scene.add_object(obj)

        # Trace scene
        trace = rt.trace_scene(scene, obj_resolution=self.options_ray_trace_vis.ray_trace_optic_res)

        # Calculate intersection with plane
        ray_trace = rt.RayTrace(scene)
        ray_trace.add_many_light_paths(trace.light_paths)

        return ray_trace
