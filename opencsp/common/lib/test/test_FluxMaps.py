"""
Demonstrate Solar Field Plotting Routines



"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import opencsp.common.lib.csp.RayTrace as rt
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFacetEnsemble as rcfe
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlRayTrace as rcrt
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.csp.LightSourcePoint import LightSourcePoint
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.csp.SolarField import SolarField
from opencsp.common.lib.geometry.Intersection import Intersection
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
from opencsp.common.lib.render_control.RenderControlFunctionXY import RenderControlFunctionXY

UP = Vxyz([0, 0, 1])
DOWN = -UP
NORTH = Vxyz([0, 1, 0])
SOUTH = -NORTH
EAST = Vxyz([1, 0, 0])
WEST = -EAST
STOW = Vxyz([-1, 0, -11.4]).normalize()  # !!!!!
# (az=np.deg2rad(270), el=np.deg2rad(-85))


class TestFluxMaps(to.TestOutput):

    @classmethod
    def setUpClass(
        self,
        source_file_body: str = "TestFluxMaps",  # Set these here, because pytest calls
        figure_prefix_root: str = "tfm",  # setup_class() with no arguments.
        interactive: bool = False,
        verify: bool = True,
    ):

        # Save input.
        # Interactive mode flag.
        # This has two effects:
        #    1. If interactive mode, plots stay displayed.
        #    2. If interactive mode, after several plots have been displayed,
        #       plot contents might change due to Matplotlib memory issues.
        #       This may in turn cause misleading failures in comparing actual
        #       output to expected output.
        # Thus:
        #   - Run pytest in non-interactive mode.
        #   - If viewing run output interactively, be advised that figure content might
        #     change (e.g., garbled plot titles, etc), and actual-vs-expected comparison
        #     might erroneously fail (meaning they might report an error which is not
        #     actually a code error).
        #

        super(TestFluxMaps, self).setUpClass(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
        )

        # Note: It is tempting to put the "Reset rendering" code lines here, to avoid redundant
        # computation and keep all the plots up.  Don't do this, because the plotting system
        # will run low/out of memory, causing adverse effectes on the plots.

    def setUp(self):
        # Load solar field data.
        self.solar_field: SolarField = sf.SolarField.from_csv_files(
            lln.NSTTF_ORIGIN,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            "Sandia NSTTF",
        )

    def test_parabolic_mirror_bullet(self) -> None:
        """
        Produces a mirror with focal length 10 and does a ray trace on it.
        This shows the bullet plot slices at z=4, z=5, and z=6.
        """
        # Initialize test.
        self.start_test()

        FOCAL_LENGTH = 5  # meters
        MIRROR_DIM = (1.2, 1.2)  # meters

        # View setup
        title = "Mirror Facing Up"
        caption = f"A mirror with focal length {FOCAL_LENGTH} facing up for a ray trace."
        comments = []

        # Configuration setup
        mirror = MirrorParametricRectangular.from_focal_length(FOCAL_LENGTH, MIRROR_DIM)

        # Create light source and Scene
        scene = Scene()
        scene.add_object(mirror)
        sun = LightSourceSun.from_given_sun_position(Uxyz([0, 0, -1]), 1)
        scene.add_light_source(sun)

        # Ray Trace Scene
        trace = rt.trace_scene(scene, obj_resolution=Resolution.separation(0.2))

        # bullet map z=4
        z4 = Pxyz([0, 0, 4])
        z5 = Pxyz([0, 0, 5])
        z6 = Pxyz([0, 0, 6])

        bullet4 = Intersection.plane_intersect_from_ray_trace(trace, (z4, DOWN))
        bullet5 = Intersection.plane_intersect_from_ray_trace(trace, (z5, DOWN))
        bullet6 = Intersection.plane_intersect_from_ray_trace(trace, (z6, DOWN))

        sqaure_half_side = 0.15  # half of the side length of a sqaure

        def square(z: float):
            return Vxyz(
                [
                    [sqaure_half_side, sqaure_half_side, -sqaure_half_side, -sqaure_half_side],
                    [sqaure_half_side, -sqaure_half_side, -sqaure_half_side, sqaure_half_side],
                    [z, z, z, z],
                ]
            )

        # comments\
        comments.append("Corresponds to the scene shown in tfm001_Mirror_Facing_Up_3d.png")

        # Draw 3D
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                1
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        mirror_style = rcm.RenderControlMirror()
        mirror.draw(fig_record.view, mirror_style)
        trace_style = rcrt.init_current_lengths(current_len=6)
        trace.draw(fig_record.view, trace_style)
        square(4).draw_line(fig_record.view, close=True, style=rcps.RenderControlPointSeq(color="b", marker=","))
        square(5).draw_line(fig_record.view, close=True, style=rcps.RenderControlPointSeq(color="g", marker=","))
        square(6).draw_line(fig_record.view, close=True, style=rcps.RenderControlPointSeq(color="r", marker=","))
        self.show_save_and_check_figure(fig_record)

        # Draw z=4
        fig_record = fm.setup_figure(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                2
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=f"Intersections at z=4",
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.x_limits = (-sqaure_half_side, sqaure_half_side)
        fig_record.y_limits = (-sqaure_half_side, sqaure_half_side)
        bullet4.draw(fig_record.view, rcps.RenderControlPointSeq(color="b"))
        self.show_save_and_check_figure(fig_record)

        # Draw z=5
        fig_record = fm.setup_figure(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                3
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=f"Intersections at z=5",
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.x_limits = (-sqaure_half_side, sqaure_half_side)
        fig_record.y_limits = (-sqaure_half_side, sqaure_half_side)
        bullet5.draw(fig_record.view, rcps.RenderControlPointSeq(color="g"))
        self.show_save_and_check_figure(fig_record)

        # Draw z=6
        fig_record = fm.setup_figure(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                4
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=f"Intersections at z=6",
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.x_limits = (-sqaure_half_side, sqaure_half_side)
        fig_record.y_limits = (-sqaure_half_side, sqaure_half_side)
        bullet6.draw(fig_record.view, rcps.RenderControlPointSeq(color="r"))
        self.show_save_and_check_figure(fig_record)

    def test_parabolic_mirror_flux_map(self) -> None:
        """
        Produces a mirror with focal length 10 and does a ray trace on it.
        This shows the flux map slices at z=4, z=5, and z=6.
        """
        # Initialize test.
        self.start_test()

        FOCAL_LENGTH = 5  # meters
        MIRROR_DIM = (1.2, 1.2)  # meters

        # View setup
        comments = []

        # Configuration setup
        mirror = MirrorParametricRectangular.from_focal_length(FOCAL_LENGTH, MIRROR_DIM)

        # Create light source and Scene
        scene = Scene()
        scene.add_object(mirror)
        sun = LightSourceSun.from_given_sun_position(Uxyz([0, 0, -1]), 20)
        scene.add_light_source(sun)

        # Ray Trace Scene
        trace = rt.trace_scene(scene, obj_resolution=Resolution.separation(0.1))

        # bullet map z=4
        z4 = Pxyz([0, 0, 4])
        z5 = Pxyz([0, 0, 5])
        z6 = Pxyz([0, 0, 6])

        flux4 = Intersection.plane_intersect_from_ray_trace(trace, (z4, DOWN)).to_flux_mapXY(15)
        flux5 = Intersection.plane_intersect_from_ray_trace(trace, (z5, DOWN)).to_flux_mapXY(15)
        flux6 = Intersection.plane_intersect_from_ray_trace(trace, (z6, DOWN)).to_flux_mapXY(15)

        sqaure_half_side = 0.15  # half of the side length of a sqaure

        # comments\
        comments.append("Corresponds to the scene shown in tfm001_Mirror_Facing_Up_3d_3d.png")

        flux_style = RenderControlFunctionXY(colorbar=True, cmap="OrRd")

        # Draw z=4
        fig_record = fm.setup_figure(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_im(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                12
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=f"Flux at z= 4",
            caption=f"Flux at z= 4",
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.x_limits = (-sqaure_half_side, sqaure_half_side)
        fig_record.y_limits = (-sqaure_half_side, sqaure_half_side)
        flux4.draw(fig_record.view, flux_style)
        self.show_save_and_check_figure(fig_record)

        # Draw z=5
        fig_record = fm.setup_figure(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_im(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                13
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=f"Flux at z= 5",
            caption=f"Flux at z= 5",
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.x_limits = (-sqaure_half_side, sqaure_half_side)
        fig_record.y_limits = (-sqaure_half_side, sqaure_half_side)
        flux5.draw(fig_record.view, flux_style)
        self.show_save_and_check_figure(fig_record)

        # Draw z=6
        fig_record = fm.setup_figure(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_im(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                14
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=f"Flux at z= 6",
            caption=f"Flux at z= 6",
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.x_limits = (-sqaure_half_side, sqaure_half_side)
        fig_record.y_limits = (-sqaure_half_side, sqaure_half_side)
        flux6.draw(fig_record.view, flux_style)
        self.show_save_and_check_figure(fig_record)

    def future_heliostat_flux(self) -> None:
        """
        TODO should be added
        """
        # Initialize test.
        self.start_test()


# MAIN EXECUTION
if __name__ == "__main__":

    # Control flags.
    interactive = False
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = False  # False
    # Setup.
    test_object = TestFluxMaps()
    test_object.setUpClass(interactive=interactive, verify=verify)
    # Tests.
    lt.info("Beginning tests...")
    test_object.test_parabolic_mirror_bullet()
    test_object.test_parabolic_mirror_flux_map()
    lt.info("All tests complete.")
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
