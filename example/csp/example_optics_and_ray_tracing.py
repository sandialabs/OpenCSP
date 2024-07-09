"""Makes sample optics: mirror, facet, and facet ensemble.

1) Draws the optics
2) Performs ray tracing analysis on the optics and a notional receiver
3) Plots sun disc image on receiver for
4) Plots ensquared energy plot

"""

import pytest

import datetime
import os

import numpy as np
import pytz
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
import opencsp.common.lib.csp.RayTrace as rt
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import RegionXY, Resolution
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.figure_management as fm
from opencsp.common.lib.render_control import RenderControlFacet as rcf
from opencsp.common.lib.render_control import RenderControlFacetEnsemble as rcfe
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
from opencsp.common.lib.render_control.RenderControlLightPath import RenderControlLightPath
import opencsp.common.lib.render_control.RenderControlMirror as rcm
from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace


def visualize_mirror() -> None:
    """Draws image of, plots slope, and ray traces mirror"""
    # Define mirror
    mirror = define_mirror(100)

    # Define geometry
    sun_azm = 168.54  # degrees, E of N
    sun_el = 77.79  # degrees, up from horizon
    targ_loc = Vxyz((0, 0, 30))
    targ_norm = Vxyz((0, 1, 0))
    optic_loc = Vxyz((0, 95, 0))

    # Calculate mirror pointing az/el
    v_sun = Vxyz((0, 1, 0)).rotate(Rotation.from_euler("xz", [sun_el, -sun_azm], degrees=True))
    v_optic_targ = (targ_loc - optic_loc).normalize()
    v_pointing = (v_sun + v_optic_targ).normalize()
    rot_pointing = Vxyz((0, 0, 1)).align_to(v_pointing)
    euler_angles = rot_pointing.as_euler("zxz")
    rot_pointing = Rotation.from_euler("xz", euler_angles[1:])

    # Define source
    source = define_source_sun_time()

    # Define scene
    scene = Scene()
    scene.add_light_source(source)
    scene.add_object(mirror)

    # Position optic
    transform = TransformXYZ.from_R_V(rot_pointing, optic_loc)
    scene.set_position_in_space(mirror, transform)

    # Ray trace mirror
    image, trace = ray_trace_obj(scene, targ_loc, targ_norm)

    # Plot ray trace
    plot_ray_trace(scene, image, trace, "Mirror")


def visualize_facet() -> None:
    """Draws image of, plots slope, and ray traces facet"""
    # Define facet
    facet = define_facet(100)

    # Define geometry
    sun_azm = 168.54  # degrees, E of N
    sun_el = 77.79  # degrees, up from horizon
    targ_loc = Vxyz((0, 0, 30))
    targ_norm = Vxyz((0, 1, 0))
    optic_loc = Vxyz((0, 95, 0))

    # Calculate facet pointing az/el
    v_sun = Vxyz((0, 1, 0)).rotate(Rotation.from_euler("xz", [sun_el, -sun_azm], degrees=True))
    v_optic_targ = (targ_loc - optic_loc).normalize()
    v_pointing = (v_sun + v_optic_targ).normalize()
    rot_pointing = Vxyz((0, 0, 1)).align_to(v_pointing)
    euler_angles = rot_pointing.as_euler("zxz")
    rot_pointing = Rotation.from_euler("xz", euler_angles[1:])

    # Point facet
    # facet.set_pointing(rot_pointing)

    # Define source
    source = define_source_sun_time()

    # Define scene
    scene = Scene()
    scene.add_light_source(source)
    scene.add_object(facet)

    # Position optic
    transform = TransformXYZ.from_R_V(rot_pointing, optic_loc)
    scene.set_position_in_space(facet, transform)

    # Ray trace facet
    image, trace = ray_trace_obj(scene, targ_loc, targ_norm, obj_res=Resolution.pixelX(2))

    # Plot ray trace
    plot_ray_trace(scene, image, trace, "Facet", plot_rays=True)


def visualize_mirror_array() -> None:
    """Draws image of, plots slope, and ray traces mirror_array"""
    # Define mirror_array
    mirror_array = define_mirror_array(100)

    # Define geometry
    sun_azm = 168.54  # degrees, E of N
    sun_el = 77.79  # degrees, up from horizon
    targ_loc = Vxyz((0, 0, 30))
    targ_norm = Vxyz((0, 1, 0))
    optic_loc = Vxyz((0, 95, 0))

    # Calculate mirror_array pointing az/el
    v_sun = Vxyz((0, 1, 0)).rotate(Rotation.from_euler("xz", [sun_el, -sun_azm], degrees=True))
    v_optic_targ = (targ_loc - optic_loc).normalize()
    v_pointing = (v_sun + v_optic_targ).normalize()
    rot_pointing = Vxyz((0, 0, 1)).align_to(v_pointing)
    euler_angles = rot_pointing.as_euler("zxz")
    rot_pointing = Rotation.from_euler("xz", euler_angles[1:])

    # Point mirror_array
    # mirror_array.set_pointing(rot_pointing)

    # Define source
    source = define_source_sun_time()

    # Define scene
    scene = Scene()
    scene.add_light_source(source)
    scene.add_object(mirror_array)

    # Position mirror_array
    transform = TransformXYZ.from_R_V(rot_pointing, optic_loc)
    scene.set_position_in_space(mirror_array, transform)

    # Ray trace mirror_array
    image, trace = ray_trace_obj(scene, targ_loc, targ_norm)

    # Plot ray trace
    plot_ray_trace(scene, image, trace, "Heliostat")


def define_mirror(focal_length: float) -> MirrorParametric:
    """Creates parametric mirror with given focal length"""
    region_mirror = RegionXY.from_vertices(Vxy(([-0.6, -0.6, 0.6, 0.6], [-0.6, 0.6, 0.6, -0.6])))
    return MirrorParametric.generate_symmetric_paraboloid(focal_length, region_mirror)


def define_facet(focal_length: float) -> Facet:
    """Creates facet containing a parametric mirror with given focal length"""
    mirror = define_mirror(focal_length)
    return Facet.generate_rotation_defined(mirror)


def define_mirror_array(focal_length: float) -> FacetEnsemble:
    """Creates an array of on-axis canted facets with the given focal length"""
    x_locs = np.array([-2.15, -1, 0, 1, 2.15]) * 1.3
    y_locs = np.array([-2, -1, 0, 1, 2]) * 1.3

    # Define on-axis canting strategy
    v_aim_point = Vxyz((0, 0, focal_length * 2))
    v_init = Vxyz((0, 0, 1))  # +z vector
    facets = [define_facet(focal_length) for _ in range(25)]
    facet_locations = Pxyz.merge([Pxyz([x, y, 0]) for y in y_locs for x in x_locs])
    facet_canting = [v_init.align_to(v_aim_point - v_loc) for v_loc in facet_locations]
    # for x_loc in x_locs:
    #     for y_loc in y_locs:
    #         # Create facet
    #         facet = define_facet(focal_length)
    #         # Position facet in array
    #         v_loc = Vxyz((x_loc, y_loc, 0))
    #         facet.set_position_in_space(v_loc, Rotation.identity())
    #         # Set pointing of facet
    #         v_point = v_aim_point - v_loc
    #         rot = v_init.align_to(v_point)
    #         facet.set_pointing(rot)
    #         # Save facets
    #         facets.append(facet)

    # Build facet ensemble
    facet_ensemble = FacetEnsemble(facets)
    facet_ensemble.set_facet_positions(facet_locations)
    facet_ensemble.set_facet_canting(facet_canting)

    return facet_ensemble  # FacetEnsemble.generate_rotation_defined(facets)


def define_source_sun_time(res: int = 10) -> LightSourceSun:
    # Create source (sun)
    tz = pytz.timezone("US/Mountain")
    time = datetime.datetime(2023, 7, 1, 12, 0, 0, 0, tz)
    loc = (35.0844, -106.6504)  # degrees, Albuquerque, NM
    return LightSourceSun.from_location_time(loc, time, resolution=10)


def ray_trace_obj(
    scene: Scene, v_targ_cent: Vxyz, v_targ_norm: Uxyz, obj_res: Resolution = Resolution.pixelX(20)
) -> tuple[np.ndarray, rt.RayTrace]:
    # Trace scene
    trace = rt.trace_scene(scene, obj_resolution=obj_res)

    # Create ray trace object
    ray_trace = rt.RayTrace(scene)
    ray_trace.add_many_light_paths(trace.light_paths)

    # Calculate intersection with plane
    intersection_points = rt.plane_intersect(ray_trace, v_targ_cent, v_targ_norm)

    # Calculate histogram image
    image = rt.histogram_image(0.04, 2, intersection_points)[0]

    return image, trace


def plot_ray_trace(scene: Scene, image: np.ndarray, trace: rt.RayTrace, title: str, plot_rays: bool = False) -> None:
    """Plots and saves images"""
    # Define save directory
    save_dir = os.path.join(os.path.dirname(__file__), "data/output")

    # Define visualization controls
    figure_control = rcfg.RenderControlFigure(tile_array=(2, 1), tile_square=True)
    mirror_control = rcm.RenderControlMirror(centroid=True, surface_normals=True, norm_res=1)
    facet_control = rcf.RenderControlFacet(draw_mirror_curvature=True, mirror_styles=mirror_control)
    fe_control = rcfe.RenderControlFacetEnsemble(default_style=facet_control)
    axis_control_m = rca.meters()
    if plot_rays:
        light_path_control = RenderControlLightPath(current_length=10)
        ray_trace_control = RenderControlRayTrace(light_path_control=light_path_control)

    controls = {MirrorParametric: mirror_control, Facet: facet_control, FacetEnsemble: fe_control}

    # Plot scenario
    fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, title=title + ": Ray Trace")
    if plot_rays:
        trace.draw(fig_record.view, ray_trace_control)
    scene.draw_objects(fig_record.view, controls)
    fig_record.axis.axis("equal")
    fig_record.view.show(block=True)
    # fig_record.save(save_dir, 'ray_trace_' + title, 'png')

    # Plot image
    fig_record = fm.setup_figure(figure_control, axis_control_m, title=title + ": Sun Image")
    fig_record.axis.imshow(image, cmap="jet")
    fig_record.save(save_dir, "sun_image_" + title, "png")


@pytest.mark.skipif(os.name != 'nt', reason='Test hangs. See https://github.com/sandialabs/OpenCSP/issues/133.')
def example_optics_and_ray_tracing_driver():
    # """A driver for the OpenCSP optics and ray tracing example"""
    visualize_mirror()
    visualize_facet()
    visualize_mirror_array()


if __name__ == "__main__":
    example_optics_and_ray_tracing_driver()
