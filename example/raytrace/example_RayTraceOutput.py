"""
Test generation and rendering of ray trace outputs on opencsp mirrors.
"""

import copy
import logging
import math
import os
import time
from cmath import sin
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from scipy.spatial.transform import Rotation

import opencsp.common.lib.csp.ufacet.Heliostat as helio
import opencsp.common.lib.csp.ufacet.HeliostatConfiguration as hc
import opencsp.common.lib.csp.RayTrace as rt
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.string_tools as st
from opencsp.common.lib.csp.ufacet.Facet import Facet
from opencsp.common.lib.csp.ufacet.Heliostat import Heliostat
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametricRectangular import (
    MirrorParametricRectangular,
)
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.csp.SolarField import SolarField
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
from opencsp.common.lib.render_control.RenderControlEnsemble import (
    RenderControlEnsemble,
)
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlFigureRecord import (
    RenderControlFigureRecord,
)
from opencsp.common.lib.render_control.RenderControlLightPath import (
    RenderControlLightPath,
)
from opencsp.common.lib.render_control.RenderControlRayTrace import (
    RenderControlRayTrace,
)
from opencsp.common.lib.render_control.RenderControlSurface import RenderControlSurface


class ExampleRayTraceOutput(to.TestOutput):
    @classmethod
    def setup_class(
        self,
        source_file_body: str = 'ExampleRayTraceOutput',  # Set these here, because pytest calls
        figure_prefix_root: str = 'trto',  # setup_class() with no arguments.
        interactive: bool = False,
        verify: bool = True,
    ):
        # Generic setup.
        super(ExampleRayTraceOutput, self).setup_class(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
            output_path='raytrace',
        )
        # Domain-specific setup.

        # Mirror, based on a parameteric model.
        self.m1_focal_length = 2.0  # meters
        self.m1_fxn = self.lambda_symmetric_paraboloid(
            self, self.m1_focal_length
        )  # Include self as a parameter, because this setup_class() function is a @classmethod.
        self.m1_len_x = 2.0  # m
        self.m1_len_y = 3.0  # m
        self.m1_rectangle_xy = (self.m1_len_x, self.m1_len_y)
        self.m1 = MirrorParametricRectangular(self.m1_fxn, self.m1_rectangle_xy)
        self.m1_shape_description = (
            'rectangle ' + str(self.m1_len_x) + 'm x ' + str(self.m1_len_y) + 'm'
        )
        self.m1_title = (
            'Mirror ('
            + self.m1_shape_description
            + ', f='
            + str(self.m1_focal_length)
            + 'm), Face Up'
        )
        self.m1_caption = (
            'A single mirror of shape ('
            + self.m1_shape_description
            + '), analytically defined with focal length f='
            + str(self.m1_focal_length)
            + 'm.'
        )
        self.m1_comments = []

        # Facet, based on a parameteric mirror.
        self.f1 = Facet('1', self.m1, [0, 0, 0])
        self.f1_title = 'Facet, from ' + self.m1_title
        self.f1_caption = (
            'A facet defined from a parameteric mirror of shape ('
            + self.m1_shape_description
            + '), with focal length f='
            + str(self.m1_focal_length)
            + 'm.'
        )
        self.f1_comments = []

        # Simple 2x2 heliostat, with parameteric facets.
        self.h2x2_f1 = Facet('1', copy.deepcopy(self.m1), [-1.1, 1.6, 0])
        self.h2x2_f2 = Facet('2', copy.deepcopy(self.m1), [1.1, 1.6, 0])
        self.h2x2_f3 = Facet('3', copy.deepcopy(self.m1), [-1.1, -1.6, 0])
        self.h2x2_f4 = Facet('4', copy.deepcopy(self.m1), [1.1, -1.6, 0])

        # Set canting angles.
        cos5 = np.cos(np.deg2rad(8))
        sin5 = np.sin(np.deg2rad(8))
        tilt_up = Rotation.from_matrix(
            np.asarray([[1, 0, 0], [0, cos5, -sin5], [0, sin5, cos5]])
        )
        tilt_down = Rotation.from_matrix(
            np.asarray([[1, 0, 0], [0, cos5, sin5], [0, -sin5, cos5]])
        )
        tilt_left = Rotation.from_matrix(
            np.asarray([[cos5, 0, sin5], [0, 1, 0], [-sin5, 0, cos5]])
        )
        tilt_right = Rotation.from_matrix(
            np.asarray([[cos5, 0, -sin5], [0, 1, 0], [sin5, 0, cos5]])
        )
        self.h2x2_f1.canting = tilt_left * tilt_up
        self.h2x2_f2.canting = tilt_right * tilt_up
        self.h2x2_f3.canting = tilt_left * tilt_down
        self.h2x2_f4.canting = tilt_right * tilt_down
        self.h2x2_facets = [self.h2x2_f1, self.h2x2_f2, self.h2x2_f3, self.h2x2_f4]
        self.h2x2 = Heliostat(
            'Simple 2x2 Heliostat', [0, 0, 0], 4, 2, 2, self.h2x2_facets, 0, 0
        )
        self.h2x2_title = 'Heliostat with Parametrically Defined Facets'
        self.h2x2_caption = (
            'Heliostat with four facets ('
            + self.m1_shape_description
            + '), with focal length f='
            + str(self.m1_focal_length)
            + 'm.'
        )
        self.h2x2_comments = []

        # Simple solar field, with two simple heliostats.
        self.sf2x2_h1 = Heliostat(
            'Heliostat 1',
            [0, 0, 0],
            4,
            2,
            2,
            copy.deepcopy(self.h2x2_facets),
            4.02,
            0.1778,
        )
        self.sf2x2_h2 = Heliostat(
            'Heliostat 2',
            [0, 10, 0],
            4,
            2,
            2,
            copy.deepcopy(self.h2x2_facets),
            4.02,
            0.1778,
        )
        self.sf2x2_heliostats = [self.sf2x2_h1, self.sf2x2_h2]
        self.sf2x2 = SolarField(
            'Test Field', 'test', [-106.509606, 34.962276], self.sf2x2_heliostats
        )
        self.sf2x2_title = 'Two Heliostats'
        self.sf2x2_caption = 'Two 4-facet heliostats, tracking.'
        self.sf2x2_comments = []

    def lambda_symmetric_paraboloid(
        self, focal_length: float
    ) -> Callable[[float, float], float]:
        """
        Helper function that makes lambdas of paraboloids of a given focal length.
        """
        a = 1.0 / (4 * focal_length)
        return lambda x, y: a * (x**2 + y**2)

    # draw simple rays
    def example_draw_simple_ray(self) -> None:
        self.start_test()

        local_comments = self.m1_comments
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(1),
            title="draw_simple_light_path",
            caption=self.f1_caption,
            comments=local_comments,
            code_tag=self.code_tag,
        )

        view = fig_record.view

        points = Pxyz(
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 1, 0, 3]]
        )  # [np.array([0,0,0]),np.array([0,0,1]),np.array([0,2,0]),np.array([0,0,3])]
        normal_vector = Uxyz([0, 0, 1])
        incoming_vector = Vxyz([0, 1, -1])
        ref_vec = rt.calc_reflected_ray(normal_vector, incoming_vector)
        ray = LightPath(points, incoming_vector, ref_vec)
        light_path_control = RenderControlLightPath(
            line_render_control=rcps.RenderControlPointSeq(color='y')
        )
        ray.draw(view, light_path_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

    # demonstration of ray tracing a single mirror
    def example_mirror_trace(self) -> None:
        self.start_test()

        local_comments = []
        fn = self.lambda_symmetric_paraboloid(2)
        # fn = lambda x, y: 0 * x
        m1 = MirrorParametricRectangular(fn, (2, 2))

        tran = Vxyz([0, 0, 0])

        rot_id = Rotation.identity()
        rot_45_deg = Rotation.from_euler('x', 45, degrees=True)

        light_path_control = RenderControlLightPath(current_length=4)
        mirror_control = rcm.RenderControlMirror()

        ls = LightSourceSun()

        # Face Up, Parallel Beams 3d

        ls.incident_rays = LightPath.many_rays_from_many_vectors(
            None, Vxyz([0, 0, -1])
        )  # straight down

        m1.set_position_in_space(tran, rot_id)

        scene1 = Scene()
        scene1.add_light_source(ls)
        scene1.add_object(m1)

        trace1 = rt.trace_scene(scene1, 8)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(2),
            title="mirror_facing_up",
            caption="mirror_facing_up",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view1 = fig_record.view

        trace1.draw(view1, RenderControlRayTrace(light_path_control=light_path_control))
        m1.draw(view1, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        # Face Up, Parallel Beams yz

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(3),
            title="mirror_facing_up_side_view",
            caption="mirror_facing_up_side_view",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view1_yz = fig_record.view

        trace1.draw(
            view1_yz, RenderControlRayTrace(light_path_control=light_path_control)
        )
        m1.draw(view1_yz, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        # 45 Degree Rotation, Parallel Beams 3d

        ls.incident_rays = LightPath.many_rays_from_many_vectors(
            None, Uxyz([0, 1, -1])
        )  # coming at a 45 degree angle

        m1.set_position_in_space(tran, rot_45_deg)

        scene2 = Scene()
        scene2.add_light_source(ls)
        scene2.add_object(m1)

        trace2 = rt.trace_scene(scene2, 8)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(4),
            title="mirror_tilted",
            caption="mirror_tilted",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view2 = fig_record.view

        trace2.draw(view2, RenderControlRayTrace(light_path_control=light_path_control))
        m1.draw(view2, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        # 45 Degree Rotation, Parallel Beams yz

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(5),
            title="mirror_tilted_side_view",
            caption="mirror_tilted_side_view",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view2_yz = fig_record.view
        trace2.draw(
            view2_yz, RenderControlRayTrace(light_path_control=light_path_control)
        )
        m1.draw(view2_yz, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        # Face Up, Cone of light rays, 3d

        # set of inc vectors to test
        example_vecs = Uxyz(
            [[0, 0, 0, 0.1, -0.1], [0, 0.1, -0.1, 0, 0], [-1, -1, -1, -1, -1]]
        )

        ls.incident_rays = LightPath.many_rays_from_many_vectors(None, example_vecs)

        m1.set_position_in_space(tran, rot_id)

        scene3 = Scene()
        scene3.add_light_source(ls)
        scene3.add_object(m1)
        trace3 = rt.trace_scene(scene3, 8)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(6),
            title="mirror_facing_up_many_rays",
            caption="mirror_facing_up_many_rays",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view3 = fig_record.view

        trace3.draw(view3, RenderControlRayTrace(light_path_control=light_path_control))
        m1.draw(view3, mirror_control)

        # Face Up, Cone of light rays, yz

        # Output.
        self.show_save_and_check_figure(fig_record)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(7),
            title="mirror_facing_up_many_rays_side_view",
            caption="mirror_facing_up_many_rays_side_view",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view3 = fig_record.view

        trace3.draw(view3, RenderControlRayTrace(light_path_control=light_path_control))
        m1.draw(view3, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        # 45 degree rotation, cone of beams 3d

        ls.incident_rays = LightPath.many_rays_from_many_vectors(
            None, example_vecs.rotate(rot_45_deg)
        )

        m1.set_position_in_space(tran, rot_45_deg)

        scene4 = Scene()
        scene4.add_light_source(ls)
        scene4.add_object(m1)
        trace4 = rt.trace_scene(scene4, 8)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(8),
            title="mirror_tilted_many_rays",
            caption="mirror_tilted_many_rays",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view4 = fig_record.view

        trace4.draw(view4, RenderControlRayTrace(light_path_control=light_path_control))
        m1.draw(view4, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        # 45 degree rotation, cone of beams yz

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(9),
            title="mirror_tilted_many_rays_side_view",
            caption="mirror_tilted_many_rays_side_view",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view4_yz = fig_record.view

        trace4.draw(
            view4_yz, RenderControlRayTrace(light_path_control=light_path_control)
        )
        m1.draw(view4_yz, mirror_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        return

    # demonstration of ray tracing a single heliostat
    def example_heliostat_trace(self) -> None:
        self.start_test()

        view_spec = vs.view_spec_3d()
        view_spec_yz = vs.view_spec_yz()

        def flat_func(x, y):
            return x * y * 0

        def h_func(x, y):
            return x**2 / 40 + y**2 / 40

        loc = (-106.509606, 34.962276)

        local_comments = []

        h_flat = helio.h_from_facet_centroids(
            "NSTTF Heliostat 05W01",
            np.asarray([-4.84, 57.93, 3.89]),
            25,
            5,
            5,
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            pivot_height=4.02,
            pivot_offset=0.1778,
            facet_width=1.2192,
            facet_height=1.2192,
            default_mirror_shape=flat_func,
        )
        sf_flat = sf.SolarField(
            "mini Nsttf with 5W1 and 14W1", "mini Field", loc, [h_flat]
        )

        h_curved = helio.h_from_facet_centroids(
            "NSTTF Heliostat 05W01",
            np.asarray([-4.84, 57.93, 3.89]),
            25,
            5,
            5,
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            pivot_height=4.02,
            pivot_offset=0.1778,
            facet_width=1.2192,
            facet_height=1.2192,
            default_mirror_shape=h_func,
        )
        sf_curved = sf.SolarField(
            "mini Nsttf with 5W1 and 14W1", "mini Field", loc, [h_curved]
        )

        h_canted = helio.h_from_facet_centroids(
            "NSTTF Heliostat 05W01",
            np.asarray([-4.84, 57.93, 3.89]),
            25,
            5,
            5,
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            pivot_height=4.02,
            pivot_offset=0.1778,
            facet_width=1.2192,
            facet_height=1.2192,
            default_mirror_shape=h_func,
        )
        h_canted.set_canting_from_equation(h_func)
        sf_canted = sf.SolarField(
            "mini Nsttf with 5W1 and 14W1", "mini Field", loc, [h_canted]
        )

        mirror_control = rcm.RenderControlMirror(
            surface_normals=False, norm_len=8, norm_res=2, resolution=3
        )
        facet_control = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control,
            draw_outline=True,
            draw_surface_normal_at_corners=False,
            draw_name=False,
            draw_centroid=False,
            draw_surface_normal=False,
        )
        heliostat_control = rch.RenderControlHeliostat(
            draw_centroid=True,
            draw_outline=False,
            draw_surface_normal=False,
            draw_surface_normal_at_corners=False,
            facet_styles=facet_control,
            draw_facets=True,
        )
        solar_field_style = rcsf.RenderControlSolarField(
            heliostat_styles=RenderControlEnsemble(default_style=heliostat_control)
        )

        # tracking heliostat -- setup
        #              [   x,   y,    z]
        aimpoint_xyz = [60.0, 8.8, 28.9]
        #              (year, month, day, hour, minute, second, zone)
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)

        h_flat.set_face_up()
        h_curved.set_face_up()
        h_canted.set_face_up()

        # RAY TRACING

        # set of inc vectors to test
        example_vecs = Uxyz(
            [[0, 0, 0, 0.1, -0.1], [0, 0.1, -0.1, 0, 0], [-1, -1, -1, -1, -1]]
        )

        sun = LightSourceSun()
        # sun.set_incident_rays(loc, when_ymdhmsz, 3)
        sun.incident_rays = LightPath.many_rays_from_many_vectors(None, example_vecs)

        path_control = RenderControlLightPath(current_length=50, init_length=4)
        trace_control = RenderControlRayTrace(light_path_control=path_control)

        # FLAT

        scene0 = Scene()
        scene0.add_object(sf_flat)
        scene0.add_light_source(sun)
        trace0 = rt.trace_scene(scene0, obj_resolution=2)

        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(),
        #                                          number_in_name=False, input_prefix=self.figure_prefix(10), # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          title="uncanted_heliostat_trace", caption="uncanted_heliostat_trace", comments=local_comments, code_tag=self.code_tag)
        # view0 = fig_record.view
        # sf_flat.draw(view0, solar_field_style)
        # trace0.draw(view0, trace_control)
        # self.show_save_and_check_figure(fig_record)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(11),
            title="uncanted_heliostat_trace",
            caption="uncanted_heliostat_trace",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view0_yz = fig_record.view
        sf_flat.draw(view0_yz, solar_field_style)
        trace0.draw(view0_yz, trace_control)
        self.show_save_and_check_figure(fig_record)

        # CURVED

        scene1 = Scene()
        scene1.add_object(sf_curved)
        scene1.add_light_source(sun)
        trace1 = rt.trace_scene(scene1, obj_resolution=2)

        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(),
        #                                          number_in_name=False, input_prefix=self.figure_prefix(12), # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          title="curved_heliostat_trace", caption="curved_heliostat_trace", comments=local_comments, code_tag=self.code_tag)
        # view1 = fig_record.view
        # sf_curved.draw(view1, solar_field_style)
        # trace1.draw(view1, trace_control)
        # self.show_save_and_check_figure(fig_record)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(13),
            title="curved_heliostat_trace",
            caption="curved_heliostat_trace",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view1_yz = fig_record.view
        sf_curved.draw(view1_yz, solar_field_style)
        trace1.draw(view1_yz, trace_control)
        self.show_save_and_check_figure(fig_record)

        # CANTED

        scene2 = Scene()
        scene2.add_object(sf_canted)
        scene2.add_light_source(sun)
        trace2 = rt.trace_scene(scene2, obj_resolution=2)

        # fig_record = fm.setup_figure_for_3d_data(self.figure_control, self.axis_control_m, vs.view_spec_3d(),
        #                                          number_in_name=False, input_prefix=self.figure_prefix(14), # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #                                          title="canted_heliostat_trace", caption="canted_heliostat_trace", comments=local_comments, code_tag=self.code_tag)
        # view2 = fig_record.view
        # sf_canted.draw(view2, solar_field_style)
        # trace2.draw(view2, trace_control)
        # self.show_save_and_check_figure(fig_record)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(15),
            title="canted_heliostat_trace",
            caption="canted_heliostat_trace",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view2_yz = fig_record.view
        sf_canted.draw(view2_yz, solar_field_style)
        trace2.draw(view2_yz, trace_control)
        self.show_save_and_check_figure(fig_record)

        # fm.save_all_figures(home_dir() + "/fig_save/ray_trace_visual_proofs/single_heliostat")
        return

    def example_changing_time_of_day(self) -> None:
        # create a figure that shows 5w1 reflecting the sun towards an aimpoint -- TODO tjlarki: sun rays coming from wrong direction
        def _heliostat_at_moment(
            name: str, aimpoint_xyz: tuple, when_ymdhmsz: tuple, i: int
        ) -> None:
            self.start_test()

            local_comments = []

            def fn_5w1(x, y):
                return x**2 / (4 * 65) + y**2 / (4 * 65)

            h_05w01 = helio.h_from_facet_centroids(
                "NSTTF Heliostat 05W01",
                np.asarray([-4.84, 57.93, 3.89]),
                25,
                5,
                5,
                dpft.sandia_nsttf_test_facet_centroidsfile(),
                pivot_height=4.02,
                pivot_offset=0.1778,
                facet_width=1.2192,
                facet_height=1.2192,
                default_mirror_shape=fn_5w1,
            )
            h_05w01.set_canting_from_equation(fn_5w1)

            heliostats = [h_05w01]

            sf1 = sf.SolarField(
                "mini Nsttf with 5W1", "mini Field", lln.NSTTF_ORIGIN, heliostats
            )

            mirror_control = rcm.RenderControlMirror(
                surface_normals=False, norm_len=8, norm_res=2, resolution=3
            )
            facet_control = rcf.RenderControlFacet(
                draw_mirror_curvature=True,
                mirror_styles=mirror_control,
                draw_outline=True,
                draw_surface_normal_at_corners=False,
                draw_name=False,
                draw_centroid=False,
                draw_surface_normal=False,
            )
            heliostat_control = rch.RenderControlHeliostat(
                draw_centroid=True,
                draw_outline=False,
                draw_surface_normal=False,
                draw_surface_normal_at_corners=False,
                facet_styles=facet_control,
                draw_facets=True,
            )
            solar_field_style = rcsf.RenderControlSolarField(
                heliostat_styles=RenderControlEnsemble(default_style=heliostat_control)
            )

            sf1.set_full_field_tracking(aimpoint_xyz, when_ymdhmsz)

            def _draw_helper(view: View3d) -> None:
                sf1.draw(view, solar_field_style)

                sun = LightSourceSun()
                sun.set_incident_rays(lln.NSTTF_ORIGIN, when_ymdhmsz, 3)

                scene = Scene()
                scene.add_object(sf1)
                scene.add_light_source(sun)

                path_control = RenderControlLightPath(
                    current_length=100, init_length=20
                )
                trace_control = RenderControlRayTrace(light_path_control=path_control)

                trace = rt.trace_scene(scene, obj_resolution=1)
                trace.draw(view, trace_control)

                view.draw_xyz(aimpoint_xyz, style=rcps.marker(color='tab:orange'))

                self.show_save_and_check_figure(fig_record)

            fig_record = fm.setup_figure_for_3d_data(
                self.figure_control,
                self.axis_control_m,
                vs.view_spec_3d(),
                # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                number_in_name=False,
                input_prefix=self.figure_prefix(i),
                title=name,
                caption=f"3d view of the ratrace of {name}",
                comments=local_comments,
                code_tag=self.code_tag,
            )
            view_3d = fig_record.view
            _draw_helper(view_3d)

            fig_record = fm.setup_figure_for_3d_data(
                self.figure_control,
                self.axis_control_m,
                vs.view_spec_xz(),
                # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                number_in_name=False,
                input_prefix=self.figure_prefix(i + 1),
                title=name,
                caption=f"xz view of the ratrace of {name}",
                comments=local_comments,
                code_tag=self.code_tag,
            )
            view_xz = fig_record.view
            _draw_helper(view_xz)

        _heliostat_at_moment(
            "5w1 at 11:02", [60.0, 8.8, 28.9], (2021, 5, 13, 11, 2, 0, -6), 16
        )
        # _heliostat_at_moment("5w1 at 12:02", [60.0,8.8,28.9], (2021,5,13,12,2,0,-6), 18)
        # _heliostat_at_moment("5w1 at 13:02", [60.0,8.8,28.9], (2021,5,13,13,2,0,-6), 20)
        # _heliostat_at_moment("5w1 at 14:02", [60.0,8.8,28.9], (2021,5,13,14,2,0,-6), 22)
        _heliostat_at_moment(
            "5w1 at 15:02", [60.0, 8.8, 28.9], (2021, 5, 13, 15, 2, 0, -6), 24
        )
        # test__changing-time_of_day_helper("5w1 at solar noon, pointing at origin",[0,0,28.9], (2021,5,13,13,2,0,-6), figure_control, axis_control_m)
        return

    # partial field test
    def example_partial_field_trace(self) -> None:
        self.start_test()

        local_comments = []
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(26),
            title="partial_field_trace",
            caption="A partial trace of the NSTTF field.",
            comments=local_comments,
            code_tag=self.code_tag,
        )
        view = fig_record.view

        # SOLAR FIELD SETUP
        aimpoint_xyz = np.array([0, 0, 28.9])
        when_ymdhmsz = (2021, 5, 13, 13, 2, 0, -6)  # solar noon

        solar_field: sf.SolarField = sf.sf_from_csv_files(
            name='Sandia NSTTF',
            short_name='NSTTF',
            origin_lon_lat=lln.NSTTF_ORIGIN,
            heliostat_file=dpft.sandia_nsttf_test_heliostats_origin_file(),
            facet_centroids_file=dpft.sandia_nsttf_test_facet_centroidsfile(),
            autoset_canting_and_curvature=aimpoint_xyz,
        )
        solar_field.heliostats = solar_field.heliostats[
            0:215:13
        ]  # only keeps first 15 heliostats

        # Tracking setup
        solar_field.set_full_field_tracking(
            aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz
        )

        # Style setup
        solar_field_style = rcsf.heliostat_outlines(color='b')

        # Comment
        fig_record.comments.append("Partial Solar Field Trace.")
        fig_record.comments.append(
            "Using 1 ray per surface normal and one surface normal per mirror."
        )
        fig_record.comments.append(
            "Mirror curvature and canting is defined per heliostat. They are both parabolic and have focal lengths based on the distance of the heliostat to the tower."
        )
        fig_record.comments.append(
            "Traces one in every 13 heliostats in the NSTTF field."
        )

        # Draw
        mirror_control = rcm.RenderControlMirror(
            surface_normals=False, norm_len=8, norm_res=2, resolution=2
        )
        facet_control = rcf.RenderControlFacet(
            draw_mirror_curvature=False,
            mirror_styles=mirror_control,
            draw_outline=True,
            draw_surface_normal_at_corners=False,
            draw_name=False,
            draw_centroid=False,
            draw_surface_normal=False,
        )
        heliostat_control = rch.RenderControlHeliostat(
            draw_centroid=False,
            draw_outline=True,
            draw_surface_normal=False,
            draw_surface_normal_at_corners=False,
            facet_styles=facet_control,
            draw_facets=True,
        )
        solar_field_style = rcsf.RenderControlSolarField(
            heliostat_styles=RenderControlEnsemble(default_style=heliostat_control)
        )

        solar_field.draw(view, solar_field_style)
        view.draw_xyz(
            aimpoint_xyz, style=rcps.marker(color='tab:orange'), label='aimpoint_xyz'
        )

        sun = LightSourceSun()
        sun.set_incident_rays(lln.NSTTF_ORIGIN, when_ymdhmsz, 1)

        scene = Scene()
        scene.add_light_source(sun)
        scene.add_object(solar_field)

        trace = rt.trace_scene(scene, 1, verbose=True)
        trace.draw(view, RenderControlRayTrace(RenderControlLightPath(15, 200)))

        view.draw_xyz(
            aimpoint_xyz, rcps.RenderControlPointSeq(color='orange', marker='.')
        )

        self.show_save_and_check_figure(fig_record)


# MAIN EXECUTION

if __name__ == "__main__":
    # Control flags.
    interactive = False
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = False
    # Setup.
    example_object = ExampleRayTraceOutput()
    example_object.setup_class(interactive=interactive, verify=verify)
    # Tests.
    lt.info('Beginning tests...')
    example_object.example_draw_simple_ray()
    example_object.example_mirror_trace()
    example_object.example_heliostat_trace()
    example_object.example_changing_time_of_day()
    example_object.example_partial_field_trace()

    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    example_object.teardown_method()
