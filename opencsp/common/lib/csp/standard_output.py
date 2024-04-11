"""Library of functions used to display/save the suite of standard output
plots after measuring a CSP Mirror/FacetEnsemble.
"""

from dataclasses import dataclass

import opencsp.common.lib.render_control.RenderControlAxis as rca
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlLightPath import RenderControlLightPath
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror
from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq
from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace
from opencsp.common.lib.csp.LightSource import LightSource
from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
import opencsp.common.lib.csp.RayTrace as rt
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.figure_management as fm


@dataclass
class VisualizationOptions:
    """Provides defaults for CSP optic visualization"""

    ray_trace_optic_res: int = 20
    hist_bin_res: float = 0.07
    hist_extent: float = 3
    ensquared_energy_max_semi_width: float = 2
    curvature_clim: float = 50
    slope_map_quiver_density: float = 0.1
    slope_map_resolution: float = 0.01
    slope_clim: float = 5
    slope_error_clim: float = 5
    save_dpi: int = 200
    save_format: str = 'png'
    mirror_plot_normals_res: int = 1
    ray_trace_plot_ray_length: float = 80
    close_after_save: bool = False
    to_save: bool = False
    output_dir: str = ''


def standard_output(
    optic_meas: MirrorAbstract,
    optic_ref: MirrorAbstract | None = None,
    source: LightSource | None = None,
    v_target_center: Vxyz | None = None,
    v_target_normal: Uxyz | None = None,
    vis_options=VisualizationOptions(),
):
    """Plots the OpenCSP standard ouptut plot suite given a measured
    CSP optic, a reference CSP optic, and scenario data for ray tracing.
    If no reference or scenario data exists, those plots will not be
    created.

    Parameters
    ----------
    optic_meas : MirrorAbstract
        Measured optic representation
    optic_ref : MirrorAbstract | None, optional
        Reference optic representation, by default None
    source : LightSource | None, optional
        Light source representation, by default None
    v_target_center : Vxyz | None, optional
        Target center in space, by default None
    v_target_normal : Uxyz | None, optional
        Target normal orientation in space, by default None
    vis_options : VisualizationOptions
        Data class of visualization options. By default, defualt class values.
    """
    # Determine which plots to create
    plot_reference = optic_ref is not None
    plot_ray_trace = (source is not None) and (v_target_center is not None) and (v_target_normal is not None)

    # Perform measured optic ray trace
    ray_trace_meas = ray_trace_scene(optic_meas, source, obj_resolution=vis_options.ray_trace_optic_res)
    ray_pts_meas = rt.plane_intersect(ray_trace_meas, v_target_center, v_target_normal)
    image_meas, xv_meas, yv_meas = rt.histogram_image(
        bin_res=vis_options.hist_bin_res, extent=vis_options.hist_extent, pts=ray_pts_meas
    )
    ee_meas, ws_meas = rt.ensquared_energy(ray_pts_meas, vis_options.ensquared_energy_max_semi_width)

    # Perform reference optic ray trace
    if plot_reference:
        ray_trace_ref = ray_trace_scene(optic_ref, source, obj_resolution=vis_options.ray_trace_optic_res)
        ray_pts_ref = rt.plane_intersect(ray_trace_ref, v_target_center, v_target_normal)
        image_ref, xv_ref, yv_ref = rt.histogram_image(
            bin_res=vis_options.hist_bin_res, extent=vis_options.hist_extent, pts=ray_pts_ref
        )
        ee_ref, ws_ref = rt.ensquared_energy(ray_pts_ref, vis_options.ensquared_energy_max_semi_width)

    # Set up figure control objects for 3d plots
    fig_control = RenderControlFigure(tile_array=(4, 2), tile_square=True)
    axis_control = rca.meters()
    point_styles = RenderControlPointSeq(linestyle='--', color='k', markersize=0)
    mirror_control = RenderControlMirror(surface_normals=True, norm_res=1, point_styles=point_styles)
    light_path_control = RenderControlLightPath(current_length=vis_options.ray_trace_plot_ray_length)
    ray_trace_control = RenderControlRayTrace(light_path_control=light_path_control)

    # Plot measured slope maps
    fig_rec = fm.setup_figure(fig_control, axis_control, name="Measured Slope")
    optic_meas.plot_orthorectified_slope(
        vis_options.slope_map_resolution,
        type_='magnitude',
        quiver_density=vis_options.slope_map_quiver_density,
        clim=vis_options.slope_clim,
        axis=fig_rec.axis,
    )
    if vis_options.to_save:
        fig_rec.save(
            vis_options.output_dir,
            dpi=vis_options.save_dpi,
            format='png',
            close_after_save=vis_options.close_after_save,
        )

    fig_rec = fm.setup_figure(fig_control, axis_control, name="Measured X Slope")
    optic_meas.plot_orthorectified_slope(
        vis_options.slope_map_resolution,
        type_='x',
        quiver_density=vis_options.slope_map_quiver_density,
        clim=vis_options.slope_clim,
        axis=fig_rec.axis,
    )
    if vis_options.to_save:
        fig_rec.save(
            vis_options.output_dir,
            dpi=vis_options.save_dpi,
            format='png',
            close_after_save=vis_options.close_after_save,
        )

    fig_rec = fm.setup_figure(fig_control, axis_control, name="Measured Y Slope")
    optic_meas.plot_orthorectified_slope(
        vis_options.slope_map_resolution,
        type_='y',
        quiver_density=vis_options.slope_map_quiver_density,
        clim=vis_options.slope_clim,
        axis=fig_rec.axis,
    )
    if vis_options.to_save:
        fig_rec.save(
            vis_options.output_dir,
            dpi=vis_options.save_dpi,
            format='png',
            close_after_save=vis_options.close_after_save,
        )

    # Plot measured curvature maps
    fig_rec = fm.setup_figure(fig_control, axis_control, name='Curvature X')
    optic_meas.plot_orthorectified_curvature(
        vis_options.slope_map_resolution, type_='x', clim=vis_options.curvature_clim, axis=fig_rec.axis
    )
    if vis_options.to_save:
        fig_rec.save(
            vis_options.output_dir,
            dpi=vis_options.save_dpi,
            format='png',
            close_after_save=vis_options.close_after_save,
        )

    fig_rec = fm.setup_figure(fig_control, axis_control, name='Curvature Y')
    optic_meas.plot_orthorectified_curvature(
        vis_options.slope_map_resolution, type_='y', clim=vis_options.curvature_clim, axis=fig_rec.axis
    )
    if vis_options.to_save:
        fig_rec.save(
            vis_options.output_dir,
            dpi=vis_options.save_dpi,
            format='png',
            close_after_save=vis_options.close_after_save,
        )

    # Plot slope error (requires reference optic)
    if plot_reference:
        fig_rec = fm.setup_figure(fig_control, axis_control, name="Slope Error")
        optic_meas.plot_orthorectified_slope_error(
            optic_ref,
            vis_options.slope_map_resolution,
            type_='magnitude',
            quiver_density=vis_options.slope_map_quiver_density,
            clim=vis_options.slope_error_clim,
            axis=fig_rec.axis,
        )
        if vis_options.to_save:
            fig_rec.save(
                vis_options.output_dir,
                dpi=vis_options.save_dpi,
                format='png',
                close_after_save=vis_options.close_after_save,
            )

        fig_rec = fm.setup_figure(fig_control, axis_control, name="X Slope Error")
        optic_meas.plot_orthorectified_slope_error(
            optic_ref,
            vis_options.slope_map_resolution,
            type_='x',
            quiver_density=vis_options.slope_map_quiver_density,
            clim=vis_options.slope_error_clim,
            axis=fig_rec.axis,
        )
        if vis_options.to_save:
            fig_rec.save(
                vis_options.output_dir,
                dpi=vis_options.save_dpi,
                format='png',
                close_after_save=vis_options.close_after_save,
            )

        fig_rec = fm.setup_figure(fig_control, axis_control, name="Y Slope Error")
        optic_meas.plot_orthorectified_slope_error(
            optic_ref,
            vis_options.slope_map_resolution,
            type_='y',
            quiver_density=vis_options.slope_map_quiver_density,
            clim=vis_options.slope_error_clim,
            axis=fig_rec.axis,
        )
        if vis_options.to_save:
            fig_rec.save(
                vis_options.output_dir,
                dpi=vis_options.save_dpi,
                format='png',
                close_after_save=vis_options.close_after_save,
            )

    if plot_reference and plot_ray_trace:
        # Draw reference ensemble and traced rays
        fig_rec = fm.setup_figure_for_3d_data(fig_control, axis_control, name='Ray Trace')
        if len(ray_trace_ref.light_paths) < 100:  # Only plot few rays
            ray_trace_ref.draw(fig_rec.view, ray_trace_control)
        optic_ref.draw(fig_rec.view, mirror_control)
        fig_rec.axis.axis('equal')
        if vis_options.to_save:
            fig_rec.save(
                vis_options.output_dir,
                dpi=vis_options.save_dpi,
                format='png',
                close_after_save=vis_options.close_after_save,
            )

        # Draw reference optic sun image on target
        fig_rec = fm.setup_figure(fig_control, axis_control, name='Reference Ray Trace Image')
        fig_rec.axis.imshow(image_ref, cmap='jet', extent=(xv_ref.min(), xv_ref.max(), yv_ref.min(), yv_ref.max()))
        if vis_options.to_save:
            fig_rec.save(
                vis_options.output_dir,
                dpi=vis_options.save_dpi,
                format='png',
                close_after_save=vis_options.close_after_save,
            )

    # Draw measured optic sun image on target
    if plot_ray_trace:
        fig_rec = fm.setup_figure(fig_control, axis_control, name='Measured Ray Trace Image')
        fig_rec.axis.imshow(image_meas, cmap='jet', extent=(xv_meas.min(), xv_meas.max(), yv_meas.min(), yv_meas.max()))
        if vis_options.to_save:
            fig_rec.save(
                vis_options.output_dir,
                dpi=vis_options.save_dpi,
                format='png',
                close_after_save=vis_options.close_after_save,
            )

        # Draw ensquared energy plot
        fig_rec = fm.setup_figure(fig_control, name='Ensquared Energy')
        if plot_reference:
            fig_rec.axis.plot(ws_ref, ee_ref, label='Reference', color='k', linestyle='--')
        fig_rec.axis.plot(ws_meas, ee_meas, label='Measured', color='k', linestyle='-')
        fig_rec.axis.legend()
        fig_rec.axis.grid()
        fig_rec.axis.set_xlabel('Semi-width (meters)')
        fig_rec.axis.set_ylabel('Ensquared Energy')
        fig_rec.axis.set_title('Ensquared Energy')
        if vis_options.to_save:
            fig_rec.save(
                vis_options.output_dir,
                dpi=vis_options.save_dpi,
                format='png',
                close_after_save=vis_options.close_after_save,
            )


def ray_trace_scene(obj: RayTraceable, source: LightSource, obj_resolution=1) -> rt.RayTrace:
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
    scene.add_light_source(source)
    scene.add_object(obj)

    # Trace scene
    trace = rt.trace_scene(scene, obj_resolution=obj_resolution)

    # Calculate intersection with plane
    ray_trace = rt.RayTrace(scene)
    ray_trace.add_many_light_paths(trace.light_paths)

    return ray_trace
