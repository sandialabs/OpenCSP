"""
Demonstrate Motion Based Canting Experiment.

Copyright (c) 2021 Sandia National Laboratories.

"""

# import copy
import csv as csv
import os
import sys as sys
import re


from PIL import Image, ImageTk
import tkinter as tk


# import math
from datetime import datetime, timedelta
from typing import Iterable

# import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rotation


# from sympy import Symbol, diff
# from opencsp.common.lib.geometry.Vxy import Vxy

# import opencsp.common.lib.csp.sun_position as sp
# import opencsp.common.lib.tool.dict_tools as dt
# import opencsp.common.lib.file.CsvColumns as CsvColumns
# import opencsp.common.lib.csp.HeliostatAzEl as Heliostat
import opencsp.common.lib.csp.HeliostatConfiguration as hc
import opencsp.common.lib.csp.RayTrace as rt
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.geo.lon_lat_nsttf as lln

# import opencsp.common.lib.tool.string_tools as st
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFacetEnsemble as rcfe
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlRayTrace as rcrt
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.render_control.RenderControlTower as rct
import opencsp.common.lib.render_control.RenderControlLightPath as rclp

# import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.test.support_test as stest
from opencsp import opencsp_settings
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

# from opencsp.common.lib.csp.Facet import Facet
# from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
# from opencsp.common.lib.csp.HeliostatAbstract import HeliostatAbstract
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl

# from opencsp.common.lib.file.CsvColumns import CsvColumns
from opencsp.common.lib.csp.HeliostatConfiguration import HeliostatConfiguration

# from opencsp.common.lib.csp.LightPath import LightPath
# from opencsp.common.lib.csp.LightSourcePoint import LightSourcePoint
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular

# from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.csp.SolarField import SolarField
from opencsp.common.lib.csp.Tower import Tower
from opencsp.common.lib.geometry.FunctionXYContinuous import FunctionXYContinuous
from opencsp.common.lib.geometry.Intersection import Intersection
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz

# from opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
# from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror
# from opencsp.common.lib.render_control.RenderControlFacet import RenderControlFacet
# from opencsp.common.lib.render_control.RenderControlFacetEnsemble import RenderControlFacetEnsemble
# from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
# from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
# from opencsp.common.lib.render_control.RenderControlLightPath import RenderControlLightPath

# from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace


class TestMotionBasedCanting(to.TestOutput):

    #### TODO MHH, refactor into test canting analysis and test motion based canting
    @classmethod
    def setUpClass(
        self,
        source_file_body: str = 'TestMotionBasedCanting',  # Set these here, because pytest calls
        figure_prefix_root: str = 'tca',  # setUpClass() with no arguments. ####TODO MHH this is for the prefix TestCantingAnalysis
        interactive: bool = False,
        verify: bool = True,
    ):
        self._pivot = 0
        self._az = 0
        self._el = 0
        path, _, _ = ft.path_components(__file__)

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

        super(TestMotionBasedCanting, self).setUpClass(
            source_file_body=source_file_body,
            figure_prefix_root=figure_prefix_root,
            interactive=interactive,
            verify=verify,
            output_path=path,
        )

        # Note: It is tempting to put the "Reset rendering" code lines here, to avoid redundant
        # computation and keep all the plots up.  Don't do this, because the plotting system
        # will run low/out of memory, causing adverse effectes on the plots.

        # Load solar field data.
        self.solar_field: SolarField = sf.SolarField.from_csv_files(
            long_lat=lln.NSTTF_ORIGIN,
            heliostat_attributes_csv=dpft.sandia_nsttf_test_heliostats_origin_file(),
            facet_attributes_csv=dpft.sandia_nsttf_test_facet_centroidsfile(),
            name='Sandia NSTTF',
        )

    def check_heliostat_intersection_point_on_plane(self, target_plane_normal: 'Uxyz', facet_normal: 'Vxyz'):
        """
        Returns boolean value.

        Parameters
        ----------
        target_plane_normal: unit vector perpendicular to plane containing target point
        facet_normal: transposed vector of surface normal of given facet

        """

        if Vxyz.dot(facet_normal, target_plane_normal) < 0:
            print("In check_heliostat_intersection_point_on_plane:        In plane")
            return True
        else:
            print("In check_heliostat_intersection_point_on_plane:        Not in plane")
            return False

    def exit_program(self):
        print("\tExiting program...")
        sys.exit(0)

    def doesnt_hit_plane_with_original_values(self, is_intersect: bool, az: float, el: float):
        """
        If user asks for facet values, returns azimuth and elevation values that face the target plane.

        Parameters
        ----------
        is_intersect = boolean:
            true = surface normal intersects with plane
        az: azimuth in radians
        el: elevation in radians

        """

        if is_intersect:
            return az, el
        else:
            print(
                "The given heliostat azimuth and elevation values do produce facet surface normals that intersect the plane of the target."
            )
            while True:
                message = input("Would you like to find the az el values that hit the target? (y/n): ")
                if message == 'y':
                    az = np.deg2rad(180)
                    el = np.deg2rad(0)
                    return az, el
                elif message == 'n':
                    print("\nAzimuth and elevation values not calculated.")
                    self.exit_program()
                else:
                    print("Input not recognized.")

    def doesnt_hit_plane_when_moving_intersect_point(self, search: float, is_intersect: bool, count: int):
        """
        Returns elevation and azimuth values.

        Parameters
        ----------
        search: jump from original itnersection point in radians
        is_intersect = boolean:
            true = surface normal intersects with plane
        count: integer which counts the number of searches

        """

        if is_intersect:
            pass
        else:
            count = count - 1
            search /= 2

        return search, count
        # divide distance from binary jump by 2

    def projected_normal_comparison(
        self,
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        vector_should_be: Vxyz = None,
    ):
        """
        Returns x and z signed offsets from intersection point to target location on the same plane for heliostat with canting.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        canted_x_angle: canted x angle in radians
        canted_y_angle: canted y angle in radians
        """

        # Set Facet
        facet = self.cant_single_facet(heliostat_name, f_index, az, el, canted_x_angle, canted_y_angle)
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        # Compute offset from target

        # checks for intersection
        is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)

        # computes offset from intersection point to target
        intersection_point = Intersection.plane_lines_intersection(
            (facet_origin, facet_normal), (target_loc, target_plane_normal)
        )
        target_point = Intersection.plane_lines_intersection(
            (facet_origin, vector_should_be), (target_loc, target_plane_normal)
        )

        offset_x = intersection_point.x - target_point.x
        offset_z = intersection_point.z - target_point.z
        print(
            "In TestMotionBasedCanting.projected_facet_normal_intersection_point_offset: x offset from target to intersection=",
            offset_x,
            "    z offset from target to intersection=",
            offset_z,
        )  # TODO mhh

        return is_intersect, offset_x, offset_z, intersection_point, target_point

    def canted_x_binary_search_normal(
        self,
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        vector_should_be: Vxyz = None,
        tolerance: float = None,
    ):
        """
        Returns canted x angle when the offset from the intersection point to the target in the z-direction is 0.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        canted_y_angle: canted y angle in radians
        canted_x_angle: canted x angle in radians
        tolerance: acceptable value of error for azimuth and elevation

        """

        is_intersect, offset_x, offset_z, intersection, target = self.projected_normal_comparison(
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
            vector_should_be,
        )

        count = 0
        # Find canted x angle that corresponds to a negative offset.
        if offset_z < 0:
            # Then current x angle corresponds to negative offset.
            low_x_angle = canted_x_angle
            low_offset_z = offset_z
            # Now search for an x angle corresponding to a positive offset.
            search_x_angle = canted_x_angle
            while offset_z < 0:
                search_x_angle -= 0.05
                is_intersect, offset_x, offset_z, intersection, target = self.projected_normal_comparison(
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    search_x_angle,
                    canted_y_angle,
                    vector_should_be,
                )
                search_x_angle, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_x_angle, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.canted_x_binary_search(): offset_z = ",
                    offset_z,
                    "   search_x_angle= ",
                    search_x_angle,
                )  # TODO MHH delete!
                if offset_z > 0:
                    high_x_angle = search_x_angle
                    high_offset_z = offset_z
                    print(
                        "In TestMotionBasedCanting.canted_x_binary_search(): offset_z = ",
                        offset_z,
                        "   low_x_angle = ",
                        low_x_angle,
                        "   high_x_angle = ",
                        high_x_angle,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.canted_x_binary_search(): low_offset_z = ",
                low_offset_z,
                "   high_offset_z = ",
                high_offset_z,
            )

        elif offset_z > 0:
            # Else we need to find a canted x angle that corresponds to negative offset.
            high_x_angle = canted_x_angle
            high_offset_z = offset_z
            # Now search for x angle value corresponding to a positive offset.
            search_x_angle = canted_x_angle
            while offset_z > 0:
                search_x_angle += 0.05
                is_intersect, offset_x, offset_z, intersection, target = self.projected_normal_comparison(
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    search_x_angle,
                    canted_y_angle,
                    vector_should_be,
                )
                search_x_angle, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_x_angle, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.canted_x_binary_search(): offset_z = ",
                    offset_z,
                    "   search_x_angle= ",
                    search_x_angle,
                )  # TODO MHH delete!
                if offset_z < 0:
                    low_x_angle = search_x_angle
                    low_offset_z = offset_z
                    print(
                        "In TestMotionBasedCanting.canted_x_binary_search(): offset_z = ",
                        offset_z,
                        "   low_x_angle = ",
                        low_x_angle,
                        "   high_x_angle = ",
                        high_x_angle,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.canted_x_binary_search(): low_offset_z = ",
                low_offset_z,
                "   high_offset_z = ",
                high_offset_z,
            )

        else:
            middle_x_angle = canted_x_angle
            offset_z_points = offset_z
            # If the current x angle corresponds with an offset_z of 0.
            print(
                "\nIn TestMotionBasedCanting.canted_x_binary_search(): offset_z = 0,    canted_x_angle = ",
                canted_x_angle,
            )

        offset_z_points = []

        while low_offset_z <= high_offset_z:
            middle_x_angle = low_x_angle + (high_x_angle - low_x_angle) / 2
            is_intersect, offset_x, offset_z, intersection, target = self.projected_normal_comparison(
                target_loc,
                target_plane_normal,
                heliostat_name,
                f_index,
                az,
                el,
                middle_x_angle,
                canted_y_angle,
                vector_should_be,
            )
            print(
                "In TestMotionBasedCanting.canted_x_binary_search(): offset=",
                offset_z,
                "    middle_x_angle=",
                middle_x_angle,
            )
            offset_z_points.append(offset_z)
            # ignore left half
            if offset_z < -(tolerance / np.sqrt(2)):
                low_x_angle = middle_x_angle
                print("In TestMotionBasedCanting.canted_x_binary_search: new low_x_angle = ", low_x_angle)
            # ignore right half
            elif offset_z > tolerance / np.sqrt(2):
                high_x_angle = middle_x_angle
                print("In TestMotionBasedCanting.canted_x_binary_search: new high_x_angle = ", high_x_angle)
            # middle_azimuth value hits target
            else:
                return middle_x_angle, offset_z_points

        # couldn't find the target x angle
        print("In TestMotionBasedCanting.canted_x_binary_search: x angle not calculated")

    def canted_y_binary_search_normal(
        self,
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        vector_should_be: Vxyz = None,
        tolerance: float = None,
    ):
        """
        Returns canted y angle when the offset from the intersection point to the target in the x-direction is 0.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        tolerance: acceptable value of error for azimuth and elevation
        canted_x_angle: canted x angle in radians
        canted_y_angle: canted y angle in radians

        """

        is_intersect, offset_x, offset_z, intersection, target = self.projected_normal_comparison(
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
            vector_should_be,
        )

        count = 0
        # Find y angle that corresponds to a negative offset.
        if offset_x < 0:
            # Then current y angle corresponds to negative offset.
            low_y_angle = canted_y_angle
            low_offset_x = offset_x
            # Now search for an y angle corresponding to a positive offset.
            search_y_angle = canted_y_angle
            while offset_x < 0:
                search_y_angle += 0.05
                is_intersect, offset_x, offset_z, intersection, target = self.projected_normal_comparison(
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    search_y_angle,
                    vector_should_be,
                )
                search_y_angle, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_y_angle, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.canted_y_binary_search(): offset_x = ",
                    offset_x,
                    "   search_y_angle= ",
                    search_y_angle,
                )  # TODO MHH delete!
                if offset_x > 0:
                    high_y_angle = search_y_angle
                    high_offset_x = offset_x
                    print(
                        "In TestMotionBasedCanting.canted_y_binary_search(): offset_x = ",
                        offset_x,
                        "   low_y_angle = ",
                        low_y_angle,
                        "   high_y_angle = ",
                        high_y_angle,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.canted_y_binary_search(): low_offset_x = ",
                low_offset_x,
                "   high_offset_x = ",
                high_offset_x,
            )

        elif offset_x > 0:
            # Else we need to find a y angle that corresponds to negative offset.
            high_y_angle = canted_y_angle
            high_offset_x = offset_x
            # Now search for a y angle corresponding to a positive offset.
            search_y_angle = canted_y_angle
            while offset_x > 0:
                search_y_angle -= 0.05
                is_intersect, offset_x, offset_z, intersection, target = self.projected_normal_comparison(
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    search_y_angle,
                    vector_should_be,
                )
                search_y_angle, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_y_angle, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.canted_y_binary_search(): offset_x = ",
                    offset_x,
                    "   search_y_angle= ",
                    search_y_angle,
                )  # TODO MHH delete!
                if offset_x < 0:
                    low_y_angle = search_y_angle
                    low_offset_x = offset_x
                    print(
                        "In TestMotionBasedCanting.canted_y_binary_search(): offset_x = ",
                        offset_x,
                        "   low_y_angle = ",
                        low_y_angle,
                        "   high_y_angle = ",
                        high_y_angle,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.canted_y_binary_search(): low_offset_x = ",
                low_offset_x,
                "   high_offset_x = ",
                high_offset_x,
            )

        else:
            middle_y_angle = canted_y_angle
            offset_x_points = offset_x
            # If the current y angle corresponds with an offset_x of 0.
            print(
                "\nIn TestMotionBasedCanting.canted_y_binary_search(): offset_x = 0,    canted_y_angle = ",
                canted_y_angle,
            )

        offset_x_points = []

        while low_offset_x <= high_offset_x:
            middle_y_angle = low_y_angle + (high_y_angle - low_y_angle) / 2
            is_intersect, offset_x, offset_z, intersection, target = self.projected_normal_comparison(
                target_loc,
                target_plane_normal,
                heliostat_name,
                f_index,
                az,
                el,
                canted_x_angle,
                middle_y_angle,
                vector_should_be,
            )
            print(
                "In TestMotionBasedCanting.canted_y_binary_search(): offset=",
                offset_x,
                "    middle_y_angle=",
                middle_y_angle,
            )
            offset_x_points.append(offset_x)
            # ignore left half
            if offset_x < -(tolerance / np.sqrt(2)):
                low_y_angle = middle_y_angle
                print("In TestMotionBasedCanting.canted_y_binary_search: new low_y_angle = ", low_y_angle)
            # ignore right half
            elif offset_x > tolerance / np.sqrt(2):
                high_y_angle = middle_y_angle
                print("In TestMotionBasedCanting.canted_y_binary_search: new high_y_angle = ", high_y_angle)
            # middle_azimuth value hits target
            else:
                return middle_y_angle, offset_x_points

        # couldn't find the target y angle
        print("In TestMotionBasedCanting.canted_y_binary_search: y angle not calculated")

    def find_single_facet_canting_angles_normal(
        self,
        target_loc: 'Pxyz' = None,
        target_plane_normal: 'Uxyz' = None,
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        vector_should_be: Vxyz = None,
        tolerance: float = None,
    ):
        """
        Returns canted x and y angles for single facet.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        canted_x_angle: canted x angle in degrees
        canted_y_angle: canted y angle in degrees
        tolerance: acceptable value of error for azimuth and elevation

        """

        is_intersect, offset_x, offset_z, intersection_point, target_point = self.projected_normal_comparison(
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
            vector_should_be,
        )

        # iterates through finding x and y angle values, by first finding x value, then y value given x, etc. until within tolerance
        if Pxyz.distance(intersection_point, target_point) > tolerance:
            for i in range(1, 21):
                canted_x_angle, __ = self.canted_x_binary_search_normal(
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                    vector_should_be,
                    tolerance,
                )

                canted_y_angle, __ = self.canted_y_binary_search_normal(
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                    vector_should_be,
                    tolerance,
                )
                __, offset_x, offset_z, intersection_point, target_point = self.projected_normal_comparison(
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                    vector_should_be,
                )
                print(
                    "\n\tIn TestMotionBasedCanting.find_single_facet_canting_angles Not in tolerance = ",
                    tolerance,
                    "distance = ",
                    Pxyz.distance(intersection_point, target_point),
                    "canted_x_angle = ",
                    canted_x_angle,
                    "canted_y_angle = ",
                    canted_y_angle,
                )  # TODO MHH

                if Pxyz.distance(intersection_point, target_point) < tolerance:
                    print("\n\tIN TOLERANCE")
                    print(
                        "In TestMotionBasedCanting.find_single_single_facet_canting_angles: Values in tolerance = ",
                        tolerance,
                        "distance = ",
                        Pxyz.distance(intersection_point, target_point),
                        "canted_x_angle = ",
                        canted_x_angle,
                        "     canted_y_angle = ",
                        canted_y_angle,
                    )  # TODO MHH
                    break

        # x and y canted angles not computed within 20 iterations
        else:
            print("Did not find canting angle values within tolerance before reaching maximum number of iterations")

        return canted_x_angle, canted_y_angle, intersection_point, offset_z, offset_x

    def find_all_canting_angle_values_normal(
        self,
        time: tuple = None,
        aimpoint: Pxyz = None,
        target_loc: Pxyz = None,
        target_plane_normal: Uxyz = None,
        heliostat_name: str = None,
        azimuth: float = None,
        elevation: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        tolerance: float = None,
    ):
        """
                Returns all canting angle values, x and y, for every facet if within tolerance.

                Parameters
                ----------
        .       target_loc: xyz location of target on given plane
                target_plane_normal: unit vector perpendicular to plane containing target point
                heliostat_name: abbreviated name of heliostat
                az: azimuth in radians
                el: elevation in radians
                canted_x_angle: arbritary canting angle for x rotation to start binary search
                canted_y_angle: arbritary canting angle for x rotation to start binary search
                tolerance: acceptable value of error for azimuth and elevation

        """
        # Configuration setup
        solar_field = self.solar_field
        heliostat = solar_field.lookup_heliostat(heliostat_name)

        UP = Vxyz([0, 0, 1])

        if time:
            self.set_tracking(heliostat, aimpoint, lln.NSTTF_ORIGIN, time)

            az = heliostat._az
            el = heliostat._el
            # sets solar field
            test_config = hc.HeliostatConfiguration("az-el", az, el)
            solar_field = self.solar_field
            heliostat = solar_field.lookup_heliostat(heliostat_name)
            heliostat.set_orientation(test_config)

            all_facets = []
            canted_x_angles = []
            canted_y_angles = []
            offset_x_values = []
            offset_z_values = []
            f_idx = []

            # checks az and el values to see if facet surface normal for facet 13 intersects with given plane.
            f_index_temp = '13'
            is_intersect, offset_x, offset_z, intersection_point = (
                self.projected_facet_normal_intersection_point_offset(
                    target_loc, target_plane_normal, heliostat_name, f_index_temp, az, el
                )
            )
            az, el = self.doesnt_hit_plane_with_original_values(is_intersect, az, el)

            # iterates through every facet to find the azmimuth and elevation values that correspond with a surface normal that intersects with the target.
            for facet in heliostat.facet_ensemble.facets:
                print("\nIn TestMotionBasedCanting.find_all_canting_angle_values:    facet=", facet.name)
                facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
                vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz(
                    facet_origin, aimpoint, lln.NSTTF_ORIGIN, time
                )
                facet_canted_x_angle, facet_canted_y_angle, __, offset_z, offset_x = (
                    self.find_single_facet_canting_angles_normal(
                        target_loc=target_loc,
                        target_plane_normal=target_plane_normal,
                        heliostat_name=heliostat_name,
                        f_index=facet.name,
                        az=az,
                        el=el,
                        canted_x_angle=canted_x_angle,
                        canted_y_angle=canted_y_angle,
                        vector_should_be=vector_the_normal_should_be,
                        tolerance=tolerance,
                    )
                )
                canting_angles_found = {
                    'facet_number': facet.name,
                    'canting data': {'canted x angle': facet_canted_x_angle, 'canted y angle': facet_canted_y_angle},
                }
                f_idx.append(facet.name)
                canted_x_angles.append(facet_canted_x_angle)
                canted_y_angles.append(facet_canted_y_angle)
                offset_x_values.append(offset_x)
                offset_z_values.append(offset_z)
                all_facets.append(canting_angles_found)

        else:  # for on-axis computation
            az = azimuth
            el = elevation

            # sets solar field
            test_config = hc.HeliostatConfiguration("az-el", az, el)
            solar_field = self.solar_field
            heliostat = solar_field.lookup_heliostat(heliostat_name)
            heliostat.set_orientation(test_config)

            all_facets = []
            canted_x_angles = []
            canted_y_angles = []
            offset_x_values = []
            offset_z_values = []
            f_idx = []

            # checks az and el values to see if facet surface normal for facet 13 intersects with given plane.
            f_index_temp = '13'
            is_intersect, offset_x, offset_z, intersection_point = (
                self.projected_facet_normal_intersection_point_offset(
                    target_loc, target_plane_normal, heliostat_name, f_index_temp, az, el
                )
            )
            az, el = self.doesnt_hit_plane_with_original_values(is_intersect, az, el)

            test_facet = heliostat.facet_ensemble.lookup_facet("13")
            __, trc_sur_norm = test_facet.survey_of_points(Resolution.center())
            sun_ray = -trc_sur_norm

            # iterates through every facet to find the azmimuth and elevation values that correspond with a surface normal that intersects with the target.
            for facet in heliostat.facet_ensemble.facets:
                print(f"\nIn TestMotionBasedCanting.find_all_canting_angle_values:    facet= {facet.name}")
                facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
                vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz_given_sun(
                    facet_origin, aimpoint, sun_ray
                )

                facet_canted_x_angle, facet_canted_y_angle, __, offset_z, offset_x = (
                    self.find_single_facet_canting_angles_normal(
                        target_loc=target_loc,
                        target_plane_normal=target_plane_normal,
                        heliostat_name=heliostat_name,
                        f_index=facet.name,
                        az=az,
                        el=el,
                        canted_x_angle=canted_x_angle,
                        canted_y_angle=canted_y_angle,
                        vector_should_be=vector_the_normal_should_be,
                        tolerance=tolerance,
                    )
                )
                canting_angles_found = {
                    'facet_number': facet.name,
                    'canting data': {'canted x angle': facet_canted_x_angle, 'canted y angle': facet_canted_y_angle},
                }
                f_idx.append(facet.name)
                canted_x_angles.append(facet_canted_x_angle)
                canted_y_angles.append(facet_canted_y_angle)
                offset_x_values.append(offset_x)
                offset_z_values.append(offset_z)
                all_facets.append(canting_angles_found)

        for facet in all_facets:
            print(facet)  # TODO mhh

        return canted_x_angles, canted_y_angles, f_idx, offset_x_values, offset_z_values

    def projected_facet_normal_intersection_point_offset(
        self, target_loc: 'Pxyz', target_plane_normal: 'Uxyz', heliostat_name: str, f_index: int, az: float, el: float
    ):
        """
        Returns x and y signed offsets from intersection point to target location on the same plane.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians

        """

        # sets solar field
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)

        # configuration setup
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        test_heliostat.set_orientation(test_config)
        facet = test_heliostat.lookup_facet(f_index)
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())

        # checks for intersection
        is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)

        # computes offset from intersection point to target
        intersection_point = Intersection.plane_lines_intersection(
            (facet_origin, facet_normal), (target_loc, target_plane_normal)
        )
        offset_x = intersection_point.x - target_loc.x
        offset_z = intersection_point.z - target_loc.z
        print(
            "In TestMotionBasedCanting.projected_facet_normal_intersection_point_offset: x offset from target to intersection=",
            offset_x,
            "    z offset from target to intersection=",
            offset_z,
        )  # TODO mhh

        return is_intersect, offset_x, offset_z, intersection_point

    def azimuth_binary_search(
        self,
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str,
        f_index: int,
        az: float,
        el: float,
        tolerance: float,
    ):
        """
        Returns azimuth value when the offset from the intersection point to the target in the x-direction is 0.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        tolerance: acceptable value of error for azimuth and elevation

        """

        is_intersect, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
            target_loc, target_plane_normal, heliostat_name, f_index, az, el
        )

        count = 0
        # Find azimuth value that corresponds to a negative offset.
        if offset_x < 0:
            # Then current azimuth corresponds to negative offset.
            low_azimuth = az
            low_offset_x = offset_x
            # Now search for an azimuth value corresponding to a positive offset.
            search_azimuth = az
            while offset_x < 0:
                search_azimuth -= 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
                    target_loc, target_plane_normal, heliostat_name, f_index, search_azimuth, el
                )
                search_azimuth, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_azimuth, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.azimuth_binary_search(): offset_x = ",
                    offset_x,
                    "   search_azimuth= ",
                    search_azimuth,
                )  # TODO MHH delete!
                if offset_x > 0:
                    high_azimuth = search_azimuth
                    high_offset_x = offset_x
                    print(
                        "In TestMotionBasedCanting.azimuth_binary_search(): offset_x = ",
                        offset_x,
                        "   low_azimuth = ",
                        low_azimuth,
                        "   high_azimuth = ",
                        high_azimuth,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.azimuth_binary_search(): low_offset_x = ",
                low_offset_x,
                "   high_offset_x = ",
                high_offset_x,
            )

        elif offset_x > 0:
            # Else we need to find an azimuth that corresponds to negative offset.
            high_azimuth = az
            high_offset_x = offset_x
            # Now search for an azimuth value corresponding to a positive offset.
            search_azimuth = az
            while offset_x > 0:
                search_azimuth += 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
                    target_loc, target_plane_normal, heliostat_name, f_index, search_azimuth, el
                )
                search_azimuth, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_azimuth, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.azimuth_binary_search(): offset_x = ",
                    offset_x,
                    "   search_azimuth= ",
                    search_azimuth,
                )  # TODO MHH delete!
                if offset_x < 0:
                    low_azimuth = search_azimuth
                    low_offset_x = offset_x
                    print(
                        "In TestMotionBasedCanting.azimuth_binary_search(): offset_x = ",
                        offset_x,
                        "   low_azimuth = ",
                        low_azimuth,
                        "   high_azimuth = ",
                        high_azimuth,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.azimuth_binary_search(): low_offset_x = ",
                low_offset_x,
                "   high_offset_x = ",
                high_offset_x,
            )

        else:
            middle_azimuth = az
            offset_x_points = offset_x
            # If the current azimuth value corresponds with an offset_x of 0.
            print("\nIn TestMotionBasedCanting.azimuth_binary_search(): offset_x = 0,    az = ", az)

        offset_x_points = []

        while low_offset_x <= high_offset_x:
            middle_azimuth = low_azimuth + (high_azimuth - low_azimuth) / 2
            is_intersect, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
                target_loc, target_plane_normal, heliostat_name, f_index, middle_azimuth, el
            )
            print(
                "In TestMotionBasedCanting.azimuth_binary_search(): offset=", offset_x, "    azimuth=", middle_azimuth
            )
            offset_x_points.append(offset_x)
            # ignore left half
            if offset_x < -(tolerance / np.sqrt(2)):
                low_azimuth = middle_azimuth
                print("In TestMotionBasedCanting.azimuth_binary_search: new low_azimuth = ", low_azimuth)
            # ignore right half
            elif offset_x > tolerance / np.sqrt(2):
                high_azimuth = middle_azimuth
                print("In TestMotionBasedCanting.azimuth_binary_search: new high_azimuth = ", high_azimuth)
            # middle_azimuth value hits target
            else:
                return middle_azimuth, offset_x_points

        # couldn't find the target az-values
        print("In TestMotionBasedCanting.azimuth_binary_search: azimuth value not calculated")

    def elevation_binary_search(
        self,
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str,
        f_index: int,
        az: float,
        el: float,
        tolerance: float,
    ):
        """
        Returns elevation value when the offset from the intersection point to the target in the z-direction is 0.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        tolerance: acceptable value of error for azimuth and elevation

        """

        is_intersect, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
            target_loc, target_plane_normal, heliostat_name, f_index, az, el
        )

        count = 0
        # Find elevation value that corresponds to a negative offset.
        if offset_z < 0:
            # Then current elevation corresponds to negative offset.
            low_el = el
            low_offset_z = offset_z
            # Now search for an elevation value corresponding to a positive offset.
            search_el = el
            while offset_z < 0:
                search_el += 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
                    target_loc, target_plane_normal, heliostat_name, f_index, az, search_el
                )
                search_el, count = self.doesnt_hit_plane_when_moving_intersect_point(search_el, is_intersect, count)
                count += 1
                print(
                    "In TestMotionBasedCanting.elevation_binary_search(): offset_z = ",
                    offset_z,
                    "   search_el= ",
                    search_el,
                )  # TODO MHH delete!
                if offset_z > 0:
                    high_el = search_el
                    high_offset_z = offset_z
                    print(
                        "In TestMotionBasedCanting.elevation_binary_search(): offset_z = ",
                        offset_z,
                        "   low_el = ",
                        low_el,
                        "   high_el = ",
                        high_el,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.elevation_binary_search(): low_offset_z = ",
                low_offset_z,
                "   high_offset_z = ",
                high_offset_z,
            )

        elif offset_z > 0:
            # Else we need to find an elevation that corresponds to negative offset.
            high_el = el
            high_offset_z = offset_z
            # Now search for an elevation value corresponding to a positive offset.
            search_el = el
            while offset_z > 0:
                search_el -= 0.05
                is_intersect, offset_z, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
                    target_loc, target_plane_normal, heliostat_name, f_index, az, search_el
                )
                search_el, count = self.doesnt_hit_plane_when_moving_intersect_point(search_el, is_intersect, count)
                count += 1
                print(
                    "In TestMotionBasedCanting.elevation_binary_search(): offset_z = ",
                    offset_z,
                    "   search_el= ",
                    search_el,
                )  # TODO MHH delete!
                if offset_z < 0:
                    low_el = search_el
                    low_offset_z = offset_z
                    print(
                        "In TestMotionBasedCanting.elevation_binary_search(): offset_z = ",
                        offset_z,
                        "   low_el = ",
                        low_el,
                        "   high_el = ",
                        high_el,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.elevation_binary_search(): low_offset_z = ",
                low_offset_z,
                "   high_offset_z = ",
                high_offset_z,
            )

        else:
            # If the current elevation value corresponds with an offset_x of 0.
            print("\nIn TestMotionBasedCanting.elevation_binary_search(): offset_z = 0,    el = ", el)
            middle_el = el
            offset_z_points = offset_z

        offset_z_points = []

        while low_offset_z <= high_offset_z:
            middle_el = low_el + (high_el - low_el) / 2
            is_intersect, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
                target_loc, target_plane_normal, heliostat_name, f_index, az, middle_el
            )
            print("In TestMotionBasedCanting.elevation_binary_search(): offset=", offset_z, "    elevation=", middle_el)
            offset_z_points.append(offset_z)
            # ignore left half
            if offset_z < -(tolerance / np.sqrt(2)):
                low_el = middle_el
                print("In TestMotionBasedCanting.elevation_binary_search: new low_el = ", low_el)
            # ignore right half
            elif offset_z > tolerance / np.sqrt(2):
                high_el = middle_el
                print("In TestMotionBasedCanting.elevation_binary_search: new high_el = ", high_el)
            # middle_el value hits target
            else:
                return middle_el, offset_z_points

        # couldn't find the target el-values
        print("In TestMotionBasedCanting.elevation_binary_search: elevation value not calculated")

    def find_single_facet_azimuth_el_value(
        self,
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str,
        f_index: int,
        az: float,
        el: float,
        tolerance: float,
    ):
        """
        Returns azimuth and elevation values for single facet.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        tolerance: acceptable value of error for azimuth and elevation

        """

        is_intersect, offset_x, offset_z, intersection_point = self.projected_facet_normal_intersection_point_offset(
            target_loc, target_plane_normal, heliostat_name, f_index, az, el
        )

        # iterates through finding az el values, by first finding az value, then el value given az, etc. until within tolerance
        if Pxyz.distance(intersection_point, target_loc) > tolerance:
            for i in range(1, 21):
                az, offset_x_points = self.azimuth_binary_search(
                    target_loc, target_plane_normal, heliostat_name, f_index, az, el, tolerance
                )
                el, offset_z_points = self.elevation_binary_search(
                    target_loc, target_plane_normal, heliostat_name, f_index, az, el, tolerance
                )
                is_intersect, offset_x, offset_z, intersection_point = (
                    self.projected_facet_normal_intersection_point_offset(
                        target_loc, target_plane_normal, heliostat_name, f_index, az, el
                    )
                )
                print(
                    "In TestMotionBasedCanting.find_single_facet_azimuth_el_value: Not in tolerance = ",
                    tolerance,
                    "distance = ",
                    Pxyz.distance(intersection_point, target_loc),
                    "azimuth = ",
                    az,
                    "     elevation = ",
                    el,
                )  # TODO MHH
                if Pxyz.distance(intersection_point, target_loc) < tolerance:
                    print("\n\tIN TOLERANCE")
                    break

        else:
            print(
                "In TestMotionBasedCanting.find_single_facet_azimuth_el_value: Values already in tolerance = ",
                tolerance,
                "distance = ",
                Pxyz.distance(intersection_point, target_loc),
                "azimuth = ",
                az,
                "     elevation = ",
                el,
            )  # TODO MHH

        return az, el, intersection_point

    def azimuth_binary_search_bcs(
        self,
        time: tuple = None,
        sun_ray: Vxyz = None,
        target_loc: 'Pxyz' = None,
        target_plane_normal: 'Uxyz' = None,
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        tolerance: float = None,
    ):
        """
        Returns azimuth value when the offset from the intersection point to the target in the x-direction is 0.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        tolerance: acceptable value of error for azimuth and elevation

        """
        is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
            time,
            sun_ray,
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
        )

        count = 0
        # Find azimuth value that corresponds to a negative offset.
        if offset_x < 0:
            # Then current azimuth corresponds to negative offset.
            low_azimuth = az
            low_offset_x = offset_x
            # Now search for an azimuth value corresponding to a positive offset.
            search_azimuth = az
            while offset_x < 0:
                search_azimuth -= 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    search_azimuth,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                )
                search_azimuth, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_azimuth, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.azimuth_binary_search(): offset_x = ",
                    offset_x,
                    "   search_azimuth= ",
                    search_azimuth,
                )  # TODO MHH delete!
                if offset_x > 0:
                    high_azimuth = search_azimuth
                    high_offset_x = offset_x
                    print(
                        "In TestMotionBasedCanting.azimuth_binary_search(): offset_x = ",
                        offset_x,
                        "   low_azimuth = ",
                        low_azimuth,
                        "   high_azimuth = ",
                        high_azimuth,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.azimuth_binary_search(): low_offset_x = ",
                low_offset_x,
                "   high_offset_x = ",
                high_offset_x,
            )

        elif offset_x > 0:
            # Else we need to find an azimuth that corresponds to negative offset.
            high_azimuth = az
            high_offset_x = offset_x
            # Now search for an azimuth value corresponding to a positive offset.
            search_azimuth = az
            while offset_x > 0:
                search_azimuth += 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    search_azimuth,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                )
                search_azimuth, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_azimuth, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.azimuth_binary_search(): offset_x = ",
                    offset_x,
                    "   search_azimuth= ",
                    search_azimuth,
                )  # TODO MHH delete!
                if offset_x < 0:
                    low_azimuth = search_azimuth
                    low_offset_x = offset_x
                    print(
                        "In TestMotionBasedCanting.azimuth_binary_search(): offset_x = ",
                        offset_x,
                        "   low_azimuth = ",
                        low_azimuth,
                        "   high_azimuth = ",
                        high_azimuth,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.azimuth_binary_search(): low_offset_x = ",
                low_offset_x,
                "   high_offset_x = ",
                high_offset_x,
            )

        else:
            middle_azimuth = az
            offset_x_points = offset_x
            # If the current azimuth value corresponds with an offset_x of 0.
            print("\nIn TestMotionBasedCanting.azimuth_binary_search(): offset_x = 0,    az = ", az)

        offset_x_points = []

        while low_offset_x <= high_offset_x:
            middle_azimuth = low_azimuth + (high_azimuth - low_azimuth) / 2
            is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                time,
                sun_ray,
                target_loc,
                target_plane_normal,
                heliostat_name,
                f_index,
                middle_azimuth,
                el,
                canted_x_angle,
                canted_y_angle,
            )
            print(
                "In TestMotionBasedCanting.azimuth_binary_search(): offset=", offset_x, "    azimuth=", middle_azimuth
            )
            offset_x_points.append(offset_x)
            # ignore left half
            if offset_x < -(tolerance / np.sqrt(2)):
                low_azimuth = middle_azimuth
                print("In TestMotionBasedCanting.azimuth_binary_search: new low_azimuth = ", low_azimuth)
            # ignore right half
            elif offset_x > tolerance / np.sqrt(2):
                high_azimuth = middle_azimuth
                print("In TestMotionBasedCanting.azimuth_binary_search: new high_azimuth = ", high_azimuth)
            # middle_azimuth value hits target
            else:
                return middle_azimuth, offset_x_points

        # couldn't find the target az-values
        print("In TestMotionBasedCanting.azimuth_binary_search: azimuth value not calculated")

    def elevation_binary_search_bcs(
        self,
        time: tuple = None,
        sun_ray: Vxyz = None,
        target_loc: 'Pxyz' = None,
        target_plane_normal: 'Uxyz' = None,
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        tolerance: float = None,
    ):
        """
        Returns elevation value when the offset from the intersection point to the target in the z-direction is 0.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        tolerance: acceptable value of error for azimuth and elevation

        """

        is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
            time,
            sun_ray,
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
        )

        count = 0
        # Find elevation value that corresponds to a negative offset.
        if offset_z < 0:
            # Then current elevation corresponds to negative offset.
            low_el = el
            low_offset_z = offset_z
            # Now search for an elevation value corresponding to a positive offset.
            search_el = el
            while offset_z < 0:
                search_el += 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    search_el,
                    canted_x_angle,
                    canted_y_angle,
                )
                search_el, count = self.doesnt_hit_plane_when_moving_intersect_point(search_el, is_intersect, count)
                count += 1
                print(
                    "In TestMotionBasedCanting.elevation_binary_search(): offset_z = ",
                    offset_z,
                    "   search_el= ",
                    search_el,
                )  # TODO MHH delete!
                if offset_z > 0:
                    high_el = search_el
                    high_offset_z = offset_z
                    print(
                        "In TestMotionBasedCanting.elevation_binary_search(): offset_z = ",
                        offset_z,
                        "   low_el = ",
                        low_el,
                        "   high_el = ",
                        high_el,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.elevation_binary_search(): low_offset_z = ",
                low_offset_z,
                "   high_offset_z = ",
                high_offset_z,
            )

        elif offset_z > 0:
            # Else we need to find an elevation that corresponds to negative offset.
            high_el = el
            high_offset_z = offset_z
            # Now search for an elevation value corresponding to a positive offset.
            search_el = el
            while offset_z > 0:
                search_el -= 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    search_el,
                    canted_x_angle,
                    canted_y_angle,
                )
                search_el, count = self.doesnt_hit_plane_when_moving_intersect_point(search_el, is_intersect, count)
                count += 1
                print(
                    "In TestMotionBasedCanting.elevation_binary_search(): offset_z = ",
                    offset_z,
                    "   search_el= ",
                    search_el,
                )  # TODO MHH delete!
                if offset_z < 0:
                    low_el = search_el
                    low_offset_z = offset_z
                    print(
                        "In TestMotionBasedCanting.elevation_binary_search(): offset_z = ",
                        offset_z,
                        "   low_el = ",
                        low_el,
                        "   high_el = ",
                        high_el,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.elevation_binary_search(): low_offset_z = ",
                low_offset_z,
                "   high_offset_z = ",
                high_offset_z,
            )

        else:
            # If the current elevation value corresponds with an offset_x of 0.
            print("\nIn TestMotionBasedCanting.elevation_binary_search(): offset_z = 0,    el = ", el)
            middle_el = el
            offset_z_points = offset_z

        offset_z_points = []

        while low_offset_z <= high_offset_z:
            middle_el = low_el + (high_el - low_el) / 2
            is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                time,
                sun_ray,
                target_loc,
                target_plane_normal,
                heliostat_name,
                f_index,
                az,
                middle_el,
                canted_x_angle,
                canted_y_angle,
            )
            print("In TestMotionBasedCanting.elevation_binary_search(): offset=", offset_z, "    elevation=", middle_el)
            offset_z_points.append(offset_z)
            # ignore left half
            if offset_z < -(tolerance / np.sqrt(2)):
                low_el = middle_el
                print("In TestMotionBasedCanting.elevation_binary_search: new low_el = ", low_el)
            # ignore right half
            elif offset_z > tolerance / np.sqrt(2):
                high_el = middle_el
                print("In TestMotionBasedCanting.elevation_binary_search: new high_el = ", high_el)
            # middle_el value hits target
            else:
                return middle_el, offset_z_points

        # couldn't find the target el-values
        print("In TestMotionBasedCanting.elevation_binary_search: elevation value not calculated")

    def find_single_facet_azimuth_el_value_bcs(
        self,
        time: tuple = None,
        sun_ray: Vxyz = None,
        target_loc: 'Pxyz' = None,
        target_plane_normal: 'Uxyz' = None,
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        tolerance: float = None,
    ):
        """
        Returns azimuth and elevation values for single facet.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        tolerance: acceptable value of error for azimuth and elevation

        """

        __, offset_x, offset_z, intersection_point = self.projected_ray_trace_intersection_offset(
            time,
            sun_ray,
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
        )

        # iterates through finding az el values, by first finding az value, then el value given az, etc. until within tolerance
        if Pxyz.distance(intersection_point, target_loc) > tolerance:
            for i in range(1, 21):
                az, offset_x_points = self.azimuth_binary_search_bcs(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                    tolerance,
                )
                el, offset_z_points = self.elevation_binary_search_bcs(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                    tolerance,
                )
                is_intersect, offset_x, offset_z, intersection_point = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                )
                print(
                    "In TestMotionBasedCanting.find_single_facet_azimuth_el_value: Not in tolerance = ",
                    tolerance,
                    "distance = ",
                    Pxyz.distance(intersection_point, target_loc),
                    "azimuth = ",
                    az,
                    "     elevation = ",
                    el,
                )  # TODO MHH
                if Pxyz.distance(intersection_point, target_loc) < tolerance:
                    print("\n\tIN TOLERANCE")
                    break

        else:
            print(
                "In TestMotionBasedCanting.find_single_facet_azimuth_el_value: Values already in tolerance = ",
                tolerance,
                "distance = ",
                Pxyz.distance(intersection_point, target_loc),
                "azimuth = ",
                az,
                "     elevation = ",
                el,
            )  # TODO MHH

        return az, el, intersection_point

    def find_all_azimuth_el_test_values(
        self, target_loc: Pxyz, target_plane_normal: Uxyz, heliostat_name: str, az: float, el: float, tolerance: float
    ):
        """
                Returns all azimuth and elevation values for every facet if within tolerance.

                Parameters
                ----------
        .       target_loc: xyz location of target on given plane
                target_plane_normal: unit vector perpendicular to plane containing target point
                heliostat_name: abbreviated name of heliostat
                az: azimuth in radians
                el: elevation in radians
                tolerance: acceptable value of error for azimuth and elevation

        """

        # sets solar field
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)

        # facet_index = test_heliostat.facet_ensemble.children(dict)
        all_facets = []

        # checks az and el values to see if facet surface normal for facet 13 intersects with given plane.
        f_index_temp = '13'

        is_intersect, offset_x, offset_z, intersection_point = self.projected_facet_normal_intersection_point_offset(
            target_loc, target_plane_normal, heliostat_name, f_index_temp, az, el
        )
        az, el = self.doesnt_hit_plane_with_original_values(is_intersect, az, el)

        # iterates through every facet to find the azmimuth and elevation values that correspond with a surface normal that intersects with the target.
        for f_index in range(test_heliostat.facet_ensemble.num_facets):
            f_index = str(f_index + 1)
            print("\nIn TestMotionBasedCanting.find_all_facet_azimuth_el_values:    facet=", f_index)
            facet_azimuth, facet_el, intersection = self.find_single_facet_azimuth_el_value(
                target_loc, target_plane_normal, heliostat_name, f_index, az, el, tolerance
            )
            azimuth_el_values_found = {
                'facet_number': f_index,
                'canting data': {
                    'center_normal_to_target_azimuth': facet_azimuth,
                    'center_normal_to_target_elevation': facet_el,
                },
            }
            all_facets.append(azimuth_el_values_found)

        for facet in all_facets:
            print(facet)  # TODO mhh

    def spherical_canting_facet(
        self, target_loc: 'Pxyz', target_plane_normal: 'Uxyz', heliostat_name: str, az: float, el: float
    ):
        """
        Returns x and y canting angles of spherically canted heliostat.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        az: azimuth in radians
        el: elevation in radians
        """
        # sets solar field
        # test_config = hc.HeliostatConfiguration("az-el", az, el)
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        # test_heliostat.set_orientation(test_config)

        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = (Pxyz.distance(target_loc, facet_origin)) / 2

        # set canting
        curved_equation = FunctionXYContinuous(f"(x**2 + y**2)/(4*{focal_length})")
        mirror_template = MirrorParametricRectangular(curved_equation, (1.2, 1.2))
        test_heliostat, __ = HeliostatAzEl.from_csv_files(
            heliostat_name,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            mirror_template,
        )
        test_heliostat.set_canting_from_equation(curved_equation)
        canting_rotations = test_heliostat.set_canting_from_equation(curved_equation)

        # find canting angles
        for facet in test_heliostat.facet_ensemble.facets:
            canting = canting_rotations[int(facet.name) - 1]
            rot = Rotation.from_matrix(canting.as_matrix())
            angles = rot.as_euler('xyz')
            canted_x, canted_y = angles[0], angles[1]
            print(
                f"facet:{facet.name} x_canting_angle = {canted_x:.5f} rad, y_canting_angle = {canted_y:.5f} rad"
            )  # TODO MHH

        is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
        # intersection_point = Intersection.plane_lines_intersection(
        #     (facet_origin, facet_normal), (target_loc, target_plane_normal)
        # )

        return is_intersect, angles

    def cant_single_facet(
        self, heliostat_name: str, f_index: int, az: float, el: float, canted_x_angle: float, canted_y_angle: float
    ):
        """
        Returns facet with canting.

        Parameters
        ----------
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        canted_x_angle: canted x angle in radians
        canted_y_angle: canted y angle in radians
        """

        # Configuration Setup
        solar_field = self.solar_field
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)
        facet = test_heliostat.facet_ensemble.lookup_facet(f_index)

        # set canting
        position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
        canting = Rotation.from_euler('xyz', [canted_x_angle, canted_y_angle, 0], degrees=False)
        facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)

        return facet

    def projected_ray_trace_intersection_offset(
        self,
        time: tuple = None,
        sun_ray: Vxyz = None,
        target_loc: 'Pxyz' = None,
        target_plane_normal: 'Uxyz' = None,
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
    ):
        """
        Returns x and z signed offsets from intersection point to target location on the same plane for heliostat with canting.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        canted_x_angle: canted x angle in radians
        canted_y_angle: canted y angle in radians
        """
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc

        # Set Facet
        solar_field = self.solar_field
        facet = self.cant_single_facet(heliostat_name, f_index, az, el, canted_x_angle, canted_y_angle)
        facet_no_parent = facet.no_parent_copy()
        facet_location = facet.self_to_global_tranformation

        # Set Scene
        scene = Scene()
        scene.add_object(facet_no_parent)
        scene.set_position_in_space(facet_no_parent, facet_location)

        if time:
            # Add ray trace
            solar_field.set_full_field_tracking(AIMPOINT, TIME)
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center())

            place_to_intersect = (AIMPOINT, target_plane_normal)
            __, facet_normal = facet.survey_of_points(Resolution.center())

            # Compute offset from target
            is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
            intersection_points = Intersection.plane_intersect_from_ray_trace(
                trace, place_to_intersect
            ).intersection_points

            offset_x = intersection_points.x - target_loc.x
            offset_z = intersection_points.z - target_loc.z
            print(
                f"In TestMotionBasedCanting.projected_ray_trace_intersection_offfset: for facet: {f_index} \n\t x offset from target to intersection= {offset_x}, z offset from target to intersection= {offset_z}"
            )  # TODO mhh

        else:
            # Add ray trace
            test_config = hc.HeliostatConfiguration("az-el", az, el)
            solar_field = self.solar_field
            test_heliostat = solar_field.lookup_heliostat(heliostat_name)
            test_heliostat.set_orientation(test_config)
            sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center())

            place_to_intersect = (AIMPOINT, target_plane_normal)
            __, facet_normal = facet.survey_of_points(Resolution.center())

            # Compute offset from target
            is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
            intersection_points = Intersection.plane_intersect_from_ray_trace(
                trace, place_to_intersect
            ).intersection_points

            offset_x = intersection_points.x - target_loc.x
            offset_z = intersection_points.z - target_loc.z
            print(
                f"In TestMotionBasedCanting.projected_ray_trace_intersection_offfset: for facet: {f_index} \n\t x offset from target to intersection= {offset_x}, z offset from target to intersection= {offset_z}"
            )  # TODO mhh

        return is_intersect, offset_x, offset_z, intersection_points

    def canted_x_binary_search(
        self,
        time: tuple = None,
        sun_ray: Vxyz = None,
        target_loc: 'Pxyz' = None,
        target_plane_normal: 'Uxyz' = None,
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        tolerance: float = None,
    ):
        """
        Returns canted x angle when the offset from the intersection point to the target in the z-direction is 0.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        canted_y_angle: canted y angle in radians
        canted_x_angle: canted x angle in radians
        tolerance: acceptable value of error for azimuth and elevation

        """

        is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
            time,
            sun_ray,
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
        )

        count = 0
        # Find canted x angle that corresponds to a negative offset.
        if offset_z < 0:
            # Then current x angle corresponds to negative offset.
            low_x_angle = canted_x_angle
            low_offset_z = offset_z
            # Now search for an x angle corresponding to a positive offset.
            search_x_angle = canted_x_angle
            while offset_z < 0:
                search_x_angle -= 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    search_x_angle,
                    canted_y_angle,
                )
                search_x_angle, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_x_angle, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.canted_x_binary_search(): offset_z = ",
                    offset_z,
                    "   search_x_angle= ",
                    search_x_angle,
                )  # TODO MHH delete!
                if offset_z > 0:
                    high_x_angle = search_x_angle
                    high_offset_z = offset_z
                    print(
                        "In TestMotionBasedCanting.canted_x_binary_search(): offset_z = ",
                        offset_z,
                        "   low_x_angle = ",
                        low_x_angle,
                        "   high_x_angle = ",
                        high_x_angle,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.canted_x_binary_search(): low_offset_z = ",
                low_offset_z,
                "   high_offset_z = ",
                high_offset_z,
            )

        elif offset_z > 0:
            # Else we need to find a canted x angle that corresponds to negative offset.
            high_x_angle = canted_x_angle
            high_offset_z = offset_z
            # Now search for x angle value corresponding to a positive offset.
            search_x_angle = canted_x_angle
            while offset_z > 0:
                search_x_angle += 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    search_x_angle,
                    canted_y_angle,
                )
                search_x_angle, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_x_angle, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.canted_x_binary_search(): offset_z = ",
                    offset_z,
                    "   search_x_angle= ",
                    search_x_angle,
                )  # TODO MHH delete!
                if offset_z < 0:
                    low_x_angle = search_x_angle
                    low_offset_z = offset_z
                    print(
                        "In TestMotionBasedCanting.canted_x_binary_search(): offset_z = ",
                        offset_z,
                        "   low_x_angle = ",
                        low_x_angle,
                        "   high_x_angle = ",
                        high_x_angle,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.canted_x_binary_search(): low_offset_z = ",
                low_offset_z,
                "   high_offset_z = ",
                high_offset_z,
            )

        else:
            middle_x_angle = canted_x_angle
            offset_z_points = offset_z
            # If the current x angle corresponds with an offset_z of 0.
            print(
                "\nIn TestMotionBasedCanting.canted_x_binary_search(): offset_z = 0,    canted_x_angle = ",
                canted_x_angle,
            )

        offset_z_points = []

        while low_offset_z <= high_offset_z:
            middle_x_angle = low_x_angle + (high_x_angle - low_x_angle) / 2
            is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                time,
                sun_ray,
                target_loc,
                target_plane_normal,
                heliostat_name,
                f_index,
                az,
                el,
                middle_x_angle,
                canted_y_angle,
            )
            print(
                "In TestMotionBasedCanting.canted_x_binary_search(): offset=",
                offset_z,
                "    middle_x_angle=",
                middle_x_angle,
            )
            offset_z_points.append(offset_z)
            # ignore left half
            if offset_z < -(tolerance / np.sqrt(2)):
                low_x_angle = middle_x_angle
                print("In TestMotionBasedCanting.canted_x_binary_search: new low_x_angle = ", low_x_angle)
            # ignore right half
            elif offset_z > tolerance / np.sqrt(2):
                high_x_angle = middle_x_angle
                print("In TestMotionBasedCanting.canted_x_binary_search: new high_x_angle = ", high_x_angle)
            # middle_azimuth value hits target
            else:
                return middle_x_angle, offset_z_points

        # couldn't find the target x angle
        print("In TestMotionBasedCanting.canted_x_binary_search: x angle not calculated")

    def canted_y_binary_search(
        self,
        time: tuple = None,
        sun_ray: Vxyz = None,
        target_loc: 'Pxyz' = None,
        target_plane_normal: 'Uxyz' = None,
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        tolerance: float = None,
    ):
        """
        Returns canted y angle when the offset from the intersection point to the target in the x-direction is 0.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        tolerance: acceptable value of error for azimuth and elevation
        canted_x_angle: canted x angle in radians
        canted_y_angle: canted y angle in radians

        """

        is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
            time,
            sun_ray,
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
        )

        count = 0
        # Find y angle that corresponds to a negative offset.
        if offset_x < 0:
            # Then current y angle corresponds to negative offset.
            low_y_angle = canted_y_angle
            low_offset_x = offset_x
            # Now search for an y angle corresponding to a positive offset.
            search_y_angle = canted_y_angle
            while offset_x < 0:
                search_y_angle += 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    search_y_angle,
                )
                search_y_angle, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_y_angle, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.canted_y_binary_search(): offset_x = ",
                    offset_x,
                    "   search_y_angle= ",
                    search_y_angle,
                )  # TODO MHH delete!
                if offset_x > 0:
                    high_y_angle = search_y_angle
                    high_offset_x = offset_x
                    print(
                        "In TestMotionBasedCanting.canted_y_binary_search(): offset_x = ",
                        offset_x,
                        "   low_y_angle = ",
                        low_y_angle,
                        "   high_y_angle = ",
                        high_y_angle,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.canted_y_binary_search(): low_offset_x = ",
                low_offset_x,
                "   high_offset_x = ",
                high_offset_x,
            )

        elif offset_x > 0:
            # Else we need to find a y angle that corresponds to negative offset.
            high_y_angle = canted_y_angle
            high_offset_x = offset_x
            # Now search for a y angle corresponding to a positive offset.
            search_y_angle = canted_y_angle
            while offset_x > 0:
                search_y_angle -= 0.05
                is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    search_y_angle,
                )
                search_y_angle, count = self.doesnt_hit_plane_when_moving_intersect_point(
                    search_y_angle, is_intersect, count
                )
                count += 1
                print(
                    "In TestMotionBasedCanting.canted_y_binary_search(): offset_x = ",
                    offset_x,
                    "   search_y_angle= ",
                    search_y_angle,
                )  # TODO MHH delete!
                if offset_x < 0:
                    low_y_angle = search_y_angle
                    low_offset_x = offset_x
                    print(
                        "In TestMotionBasedCanting.canted_y_binary_search(): offset_x = ",
                        offset_x,
                        "   low_y_angle = ",
                        low_y_angle,
                        "   high_y_angle = ",
                        high_y_angle,
                    )  # TODO MHH
                    break
            print(
                "In TestMotionBasedCanting.canted_y_binary_search(): low_offset_x = ",
                low_offset_x,
                "   high_offset_x = ",
                high_offset_x,
            )

        else:
            middle_y_angle = canted_y_angle
            offset_x_points = offset_x
            # If the current y angle corresponds with an offset_x of 0.
            print(
                "\nIn TestMotionBasedCanting.canted_y_binary_search(): offset_x = 0,    canted_y_angle = ",
                canted_y_angle,
            )

        offset_x_points = []

        while low_offset_x <= high_offset_x:
            middle_y_angle = low_y_angle + (high_y_angle - low_y_angle) / 2
            is_intersect, offset_x, offset_z, intersection = self.projected_ray_trace_intersection_offset(
                time,
                sun_ray,
                target_loc,
                target_plane_normal,
                heliostat_name,
                f_index,
                az,
                el,
                canted_x_angle,
                middle_y_angle,
            )
            print(
                "In TestMotionBasedCanting.canted_y_binary_search(): offset=",
                offset_x,
                "    middle_y_angle=",
                middle_y_angle,
            )
            offset_x_points.append(offset_x)
            # ignore left half
            if offset_x < -(tolerance / np.sqrt(2)):
                low_y_angle = middle_y_angle
                print("In TestMotionBasedCanting.canted_y_binary_search: new low_y_angle = ", low_y_angle)
            # ignore right half
            elif offset_x > tolerance / np.sqrt(2):
                high_y_angle = middle_y_angle
                print("In TestMotionBasedCanting.canted_y_binary_search: new high_y_angle = ", high_y_angle)
            # middle_azimuth value hits target
            else:
                return middle_y_angle, offset_x_points

        # couldn't find the target y angle
        print("In TestMotionBasedCanting.canted_y_binary_search: y angle not calculated")

    def find_single_facet_canting_angles(
        self,
        time: tuple = None,
        sun_ray: Vxyz = None,
        target_loc: 'Pxyz' = None,
        target_plane_normal: 'Uxyz' = None,
        heliostat_name: str = None,
        f_index: int = None,
        az: float = None,
        el: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        tolerance: float = None,
    ):
        """
        Returns canted x and y angles for single facet.

        Parameters
        ----------
        target_loc: xyz location of target on given plane
        target_plane_normal: unit vector perpendicular to plane containing target point
        heliostat_name: abbreviated name of heliostat
        f_index: facet index, 1-25
        az: azimuth in radians
        el: elevation in radians
        canted_x_angle: canted x angle in degrees
        canted_y_angle: canted y angle in degrees
        tolerance: acceptable value of error for azimuth and elevation

        """

        __, offset_x, offset_z, intersection_point = self.projected_ray_trace_intersection_offset(
            time,
            sun_ray,
            target_loc,
            target_plane_normal,
            heliostat_name,
            f_index,
            az,
            el,
            canted_x_angle,
            canted_y_angle,
        )

        # iterates through finding x and y angle values, by first finding x value, then y value given x, etc. until within tolerance
        if Pxyz.distance(intersection_point, target_loc) > tolerance:
            for i in range(1, 21):
                canted_x_angle, __ = self.canted_x_binary_search(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                    tolerance,
                )
                canted_y_angle, __ = self.canted_y_binary_search(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                    tolerance,
                )
                __, offset_x, offset_z, intersection_point = self.projected_ray_trace_intersection_offset(
                    time,
                    sun_ray,
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    canted_x_angle,
                    canted_y_angle,
                )
                print(
                    "\n\tIn TestMotionBasedCanting.find_single_facet_canting_angles Not in tolerance = ",
                    tolerance,
                    "distance = ",
                    Pxyz.distance(intersection_point, target_loc),
                    "canted_x_angle = ",
                    canted_x_angle,
                    "canted_y_angle = ",
                    canted_y_angle,
                )  # TODO MHH

                if Pxyz.distance(intersection_point, target_loc) < tolerance:
                    print("\n\tIN TOLERANCE")
                    print(
                        "In TestMotionBasedCanting.find_single_single_facet_canting_angles: Values in tolerance = ",
                        tolerance,
                        "distance = ",
                        Pxyz.distance(intersection_point, target_loc),
                        "canted_x_angle = ",
                        canted_x_angle,
                        "     canted_y_angle = ",
                        canted_y_angle,
                    )  # TODO MHH
                    break

        # x and y canted angles not computed within 20 iterations
        else:
            print("Did not find canting angle values within tolerance before reaching maximum number of iterations")

        return canted_x_angle, canted_y_angle, intersection_point, offset_z, offset_x

    def find_all_canting_angle_values(
        self,
        time: tuple = None,
        aimpoint: Pxyz = None,
        sun_ray: Vxyz = None,
        target_loc: Pxyz = None,
        target_plane_normal: Uxyz = None,
        heliostat_name: str = None,
        azimuth: float = None,
        elevation: float = None,
        canted_x_angle: float = None,
        canted_y_angle: float = None,
        tolerance: float = None,
    ):
        """
                Returns all canting angle values, x and y, for every facet if within tolerance.

                Parameters
                ----------
        .       target_loc: xyz location of target on given plane
                target_plane_normal: unit vector perpendicular to plane containing target point
                heliostat_name: abbreviated name of heliostat
                az: azimuth in radians
                el: elevation in radians
                canted_x_angle: arbritary canting angle for x rotation to start binary search
                canted_y_angle: arbritary canting angle for x rotation to start binary search
                tolerance: acceptable value of error for azimuth and elevation

        """

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)

        if time:
            self.set_tracking(test_heliostat, aimpoint, lln.NSTTF_ORIGIN, time)

            az = test_heliostat._az
            el = test_heliostat._el

        else:  # for on-axis computation
            az = azimuth
            el = elevation

        # sets solar field
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)

        all_facets = []
        canted_x_angles = []
        canted_y_angles = []
        offset_x_values = []
        offset_z_values = []
        f_idx = []

        # checks az and el values to see if facet surface normal for facet 13 intersects with given plane.
        f_index_temp = '13'
        is_intersect, offset_x, offset_z, intersection_point = self.projected_facet_normal_intersection_point_offset(
            target_loc, target_plane_normal, heliostat_name, f_index_temp, az, el
        )
        az, el = self.doesnt_hit_plane_with_original_values(is_intersect, az, el)

        # iterates through every facet to find the azmimuth and elevation values that correspond with a surface normal that intersects with the target.
        for f_index in range(test_heliostat.facet_ensemble.num_facets):
            f_index = str(f_index + 1)
            print("\nIn TestMotionBasedCanting.find_all_canting_angle_values:    facet=", f_index)
            facet_canted_x_angle, facet_canted_y_angle, __, offset_z, offset_x = self.find_single_facet_canting_angles(
                time=time,
                sun_ray=sun_ray,
                target_loc=target_loc,
                target_plane_normal=target_plane_normal,
                heliostat_name=heliostat_name,
                f_index=f_index,
                az=az,
                el=el,
                canted_x_angle=canted_x_angle,
                canted_y_angle=canted_y_angle,
                tolerance=tolerance,
            )
            canting_angles_found = {
                'facet_number': f_index,
                'canting data': {'canted x angle': facet_canted_x_angle, 'canted y angle': facet_canted_y_angle},
            }
            f_idx.append(f_index)
            canted_x_angles.append(facet_canted_x_angle)
            canted_y_angles.append(facet_canted_y_angle)
            offset_x_values.append(offset_x)
            offset_z_values.append(offset_z)
            all_facets.append(canting_angles_found)

        for facet in all_facets:
            print(facet)  # TODO mhh

        return canted_x_angles, canted_y_angles, f_idx, offset_x_values, offset_z_values

    @property
    def pivot(self):
        return self._pivot

    @pivot.setter
    def pivot(self, new_pivot):
        self._pivot = new_pivot
        HeliostatAzEl.set_orientation_from_az_el(self._az, self._el)

    def set_tracking(self, heliostat, aimpoint: Pxyz, location_lon_lat: Iterable, when_ymdhmsz: tuple):
        heliostat_origin = heliostat.self_to_global_tranformation.apply(Pxyz.origin())

        pointing_vector = sun_track.tracking_surface_normal_xyz(
            heliostat_origin, aimpoint, location_lon_lat, when_ymdhmsz
        )

        for _ in range(10):
            pivot_vector = pointing_vector.normalize() * heliostat.pivot
            pointing_vector = sun_track.tracking_surface_normal_xyz(
                heliostat_origin + pivot_vector, aimpoint, location_lon_lat, when_ymdhmsz
            )

        heliostat.set_orientation_from_pointing_vector(pointing_vector)
        return pointing_vector

    def off_axis_canting(self, heliostat: HeliostatAzEl, aimpoint: Pxyz, long_lat: tuple, time: tuple):

        ## TODO MHH doesn't work, need to figure out what's happening with the rotations
        self.set_tracking(heliostat, aimpoint, long_lat, time)

        cantings: list[Rotation] = []
        cantings_1: list[Rotation] = []
        vectors: list[Vxyz] = []

        UP = Vxyz([0, 0, 1])

        ### Unnecessary code and computation since heliostat in new classes remembers orientation
        # R2 = heliostat.facet_ensemble.self_to_global_tranformation.R
        # R2_inv = R2.inv()
        # h_pointing_vector = UP.rotate(heliostat.self_to_global_tranformation.R)
        # heliostat.set_orientation_from_pointing_vector(h_pointing_vector)
        ###

        # ### test to see if it works with other facets besides 13
        # test_facet = heliostat.facet_ensemble.lookup_facet("11")
        # test_facet_origin, test_facet_normal = test_facet.survey_of_points(Resolution.center())
        # # test_facet_origin = test_facet.self_to_global_tranformation.apply(Pxyz.origin())     ## same thing as survey of points
        # vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz(test_facet_origin, aimpoint, long_lat, time)
        # ###

        # for facet in heliostat.facet_ensemble.facets:
        #     # facet_origin = facet._self_to_parent_transform.apply(Pxyz.origin())  ## same thing as survey of points
        #     facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        #     # facet_origin = facet.get_transform_relative_to(heliostat.facet_ensemble).apply(Pxyz.origin())
        #     facet_origin_y = np.round(facet_origin.y, 7)
        #     facet_origin_ = Pxyz([facet_origin.x, facet_origin_y, facet_origin.z])
        #     # facet_normal = UP.rotate(facet.get_transform_relative_to(heliostat.facet_ensemble).R)
        #     print(f"fac_norm x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")

        #     # facet_normal = UP.rotate(facet._self_to_parent_transform.R)  ## same thing as facet_normal
        #     vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz(facet_origin_, aimpoint, long_lat, time)
        #     # print(f"fac_ori x: {facet_origin.x}, y: {facet_origin_y}, z: {facet_origin.z}")

        #     # vector_the_normal_should_be = Vxyz(np.array([[1], [0], [0]]))
        #     # facet_normal = Vxyz(np.array([[0], [0], [1]]))
        #     # print(
        #     #     f"fac_ori x: {vector_the_normal_should_be.x}, y: {vector_the_normal_should_be.y}, z: {vector_the_normal_should_be.z}"
        #     # )
        #     # print(f"Vector_should_be{vector_the_normal_should_be}")
        #     # print(f"facet_normal")
        #     # sun = Vxyz(sp.sun_position(long_lat, time))
        #     # ref_ray = rt.calc_reflected_ray(facet_normal[0], sun[0])
        #     # ref_ray_2 = rt.calc_reflected_ray(vector_the_normal_should_be[0], sun[0])
        #     # vector.append(vector_the_normal_should_be)
        #     # R1_prime = ref_ray.align_to(ref_ray_2)
        #     R3 = facet_normal.align_to(vector_the_normal_should_be)
        #     # Rotation.from_matrix(R3)
        #     # new = facet_normal.rotate(R3)
        #     # print(f"new x: {new.x}, y: {new.y}, z: {new.z}")

        #     # R1_prime = R1_prime_
        #     # canting_angles = []
        #     # angles = R1_prime.as_euler('xyz')
        #     # canted_x, canted_y = angles[0], angles[1]
        #     # canting_angles.append((canted_x, canted_y))
        #     # # print(
        #     #     f"Facet {facet.name}: x_canting_angle = {canted_x:.5f} degrees, y_canting_angle = {canted_y:.5f} degrees"
        #     # )
        #     # # print(f"fac_norm x: {new.x}, y: {new.y}, z: {new.z}")
        #     # facet_normal.rotate_in_place(R3)
        #     # # cantings.append(R2_inv * R1_prime * R2)    ## unnecessary code
        #     canting_rotations = R3
        #     # print(f"{R3.as_matrix()}")
        #     # # set canting
        #     # position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
        #     # # canting = Rotation.from_euler('xyz', [canted_x_angle, canted_y_angle, 0], degrees=False)
        #     # facet._self_to_parent_transform = TransformXYZ.from_R_V(R1_prime, position)
        #     cantings.append(canting_rotations)

        # vector_the_normal_should_be = [
        #     [[0.03345813], [-0.73743583], [0.67458799]],
        #     [[0.02681706], [-0.7377056], [0.67458972]],
        #     [[0.02127552], [-0.7378953], [0.67457977]],
        #     [[0.0157331], [-0.73805431], [0.67455786]],
        #     [[0.00907939], [-0.73820315], [0.67451737]],
        #     [[0.03339566], [-0.73388238], [0.67845515]],
        #     [[0.02673908], [-0.73415306], [0.6784573]],
        #     [[0.02119121], [-0.73434315], [0.67844754]],
        #     [[0.01564246], [-0.73450245], [0.67842573]],
        #     [[0.00898231], [-0.7346494], [0.67838748]],
        #     [[0.03332687], [-0.73022762], [0.68239061]],
        #     [[0.02666405], [-0.7304968], [0.68239538]],
        #     [[0.02111002], [-0.7306873], [0.68238584]],
        #     [[0.01554893], [-0.73084498], [0.68236635]],
        #     [[0.0088827], [-0.73099627], [0.68232364]],
        #     [[0.03325311], [-0.7265184], [0.68634193]],
        #     [[0.02658119], [-0.72678615], [0.68634928]],
        #     [[0.0210212], [-0.72697912], [0.68633772]],
        #     [[0.0154602], [-0.72713692], [0.68631835]],
        #     [[0.00878181], [-0.72728646], [0.68627785]],
        #     [[0.0331802], [-0.72285102], [0.69020684]],
        #     [[0.02650554], [-0.72311919], [0.69021452]],
        #     [[0.02093347], [-0.72330855], [0.6902076]],
        #     [[0.01536673], [-0.72347082], [0.69018391]],
        #     [[0.00868253], [-0.72362054], [0.69014341]],
        # ]
        count = int(0)
        # for facet in heliostat.facet_ensemble.facets:
        #     facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        #     # canted_x_angle, canted_y_angle = facet_normal.align_to_wihout_z(Vxyz(vector_the_normal_should_be[count]))
        #     R3_x = facet_normal.align_to_keep_x(Vxyz(vector_the_normal_should_be[count]))
        #     # x_cantings = R3_x.as_euler('xyz')
        #     # R3_y = facet_normal.align_to_keep_y(Vxyz(vector_the_normal_should_be[count]))
        #     # y_cantings = R3_y.as_euler('xyz')
        #     # canting_rotations = Rotation.from_euler('xyz', [x_cantings[0], y_cantings[1], 0], degrees=False)
        #     # rotated_R3 = facet_normal.rotate(R3)
        #     # normalized = rotated_R3.normalize()
        #     # canting_rotations = Rotation.from_euler('xyz', [canted_x_angle, canted_y_angle, 0], degrees=False)
        #     canting_rotations = R3_x
        #     # vec = Vxyz([1, 2, 3])
        #     # R = vec.align_to(Vxyz([1, 0, 0]))
        #     # vec_r = vec.rotate(R)
        #     # vec_r_n = vec_r.normalize()
        #     cantings.append(canting_rotations)
        #     # print(canting_rotations.as_euler('xyz'))
        #     # print(rotated_R3)
        #     count = int(count + 1)

        # print("\n")
        # heliostat.facet_ensemble.set_facet_cantings(cantings)

        for facet in heliostat.facet_ensemble.facets:
            facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz(facet_origin, aimpoint, long_lat, time)
            # print(f"{facet.name} x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            print(
                f"{facet.name} x: {vector_the_normal_should_be.x}, y: {vector_the_normal_should_be.y}, z: {vector_the_normal_should_be.z}"
            )
            R3 = facet_normal.align_to(vector_the_normal_should_be)
            canting_rotations = R3
            cantings.append(canting_rotations)

        print("\n")
        heliostat.set_facet_cantings(cantings)

        for facet in heliostat.facet_ensemble.facets:
            facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            count = int(count + 1)

            # print(f"{facet.name} x: {facet_origin.x}, y: {facet_origin.y}, z: {facet_origin.z}")

        canted_x_angle = [
            0.010546875,
            0.010598754882812499,
            0.010626221,
            0.010641479492187498,
            0.010641479492187503,
            0.0052948,
            0.00534668,
            0.005374146,
            0.005389404,
            0.005386353,
            -7.63e-05,
            -2.75e-05,
            0,
            1.2207031250000013e-05,
            1.5258789062500024e-05,
            -0.005496216,
            -0.005450439,
            -0.005419922,
            -0.005407715,
            -0.005407715,
            -0.010824585,
            -0.010778809,
            -0.010754395,
            -0.010736084,
            -0.010736084,
        ]
        canted_y_angle = [
            0.012149048,
            0.005502319,
            -4.27e-05,
            -0.005587769,
            -0.012243652,
            0.012188721,
            0.005526733,
            -2.44e-05,
            -0.005575562,
            -0.012237549,
            0.012225341796874999,
            0.005557251,
            0,
            -0.005563354,
            -0.012231445,
            0.012258911132812501,
            0.005581665,
            1.8310546875000003e-05,
            -0.005545044,
            -0.012225342,
            0.01229248,
            0.005612183,
            3.6621093750000005e-05,
            -0.005532837,
            -0.012219238,
        ]

        # # position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
        # for angles in canted_x_angle:

        #     canting_rotations = Rotation.from_euler('xyz', [canted_x_angle[count], 0, 0], degrees=False)
        #     # facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)
        #     count = int(count + 1)
        #     cantings.append(canting_rotations)

        # print("\n")
        # heliostat.facet_ensemble.set_facet_cantings(cantings)

        # for i in range(10):
        #     canting = i
        #     canting: list[Rotation] = []
        #     for facet in heliostat.facet_ensemble.facets:
        #         facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        #         vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz(
        #             facet_origin, aimpoint, long_lat, time
        #         )
        #         print(f"fac_ori x: {facet_origin.x}, y: {facet_origin.y}, z: {facet_origin.z}")
        #         R3 = facet_normal.align_to(vector_the_normal_should_be)
        #         canting_rotations = R3
        #         # print(
        #         #     f"fac_ori x: {vector_the_normal_should_be.x}, y: {vector_the_normal_should_be.y}, z: {vector_the_normal_should_be.z}"
        #         # )
        #         canting.append(canting_rotations)
        #     heliostat.facet_ensemble.set_facet_cantings(canting)
        #     print("\n")

        # sun = Vxyz(sp.sun_position(long_lat, time))
        # surface_norm = rt.calc_reflected_ray(facet_normal[0], sun[0])
        # print(f"x: {facet_origin.x}, y: {facet_origin.y}, z: {facet_origin.z}")

        self.set_tracking(heliostat, aimpoint, long_lat, time)

    # def create_folder(self, folder_path):
    #     try:
    #         os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist
    #         print(f"Folder '{folder_path}' created successfully.")
    #     except Exception as e:
    #         print(f"Error creating folder: {e}")

    def save_to_csv(
        self,
        canting_details: str,
        target: str,
        target_loc: Pxyz,
        heliostat_name: str,
        f_index: str,
        up_f_origin_x: float,
        up_f_origin_y: float,
        up_f_origin_z: float,
        up_canted_x_rotation: float,
        up_canted_y_rotation: float,
        up_x_sur_norm: float,
        up_y_sur_norm: float,
        up_z_sur_norm: float,
        azimuth: float,
        elevation: float,
        tracking_f_origin_x: float,
        tracking_f_origin_y: float,
        tracking_f_origin_z: float,
        tracking_x_sur_norm: float,
        tracking_y_sur_norm: float,
        tracking_z_sur_norm: float,
        offset_x: float,
        offset_z: float,
        actual_output_dir: str = None,
    ):
        """
        Outputs a CSV file with canting information.

        Parameters
        ----------
        heliostat_name: abbreviated name of heliostat
        f_index : list of facet indexes
        up_f_origin_x: x- component of facet origin in respect to facet ensemble when center facet is at 0m
        up_f_origin_y: y- component of facet origin in respect to facet ensemble when center facet is at 0m
        up_f_origin_z: z- component of facet origin in respect to facet ensemble when center facet is at 0m
        up_canted_x_rotation: list of canted x angle rotations when heliostat is in UP orientation
        up_canted_y_rotation: list of canted y angle rotations when heliostat is in UP orientation
        up_x_sur_norm: x-component of the surface normal when heliostat is in UP orientation
        up_y_sur_norm: y-component of the surface normal when heliostat is in UP orientation
        up_z_sur_norm: z-component of the surface normal when heliostat is in UP orientation
        azimuth: tracking azimuth orientation in radians
        elevation: tracking elevation orientation in radians
        tracking_f_origin_x: x- component of facet origin in respect to facet ensemble when heliostat is tracking target
        tracking_f_origin_y: y- component of facet origin in respect to facet ensemble when heliostat is tracking target
        tracking_f_origin_z: z- component of facet origin in respect to facet ensemble when heliostat is tracking target
        tracking_x_sur_norm: x-component of the surface normal when heliostat is tracking target
        tracking_y_sur_norm: y-component of the surface normal when heliostat is tracking target
        tracking_z_sur_norm: z-component of the surface normal when heliostat is tracking target
        offset_x: offset in x direction of reflected ray intersection point on target plane from target
        offset_z: offset in z direction of reflected ray intersection point on target plane from target

        """

        # Both lists have same length
        assert len(up_canted_y_rotation) == len(
            up_canted_x_rotation
        ), "canted_x_angles and canted_y_angles must have the same length"

        # Creates list of rows, each row is a list of [facet, canted_x_angle, canted_y_angle, offset_z, offset_x]
        rows = []
        for facet in range(len(f_index)):
            row = [facet + 1]
            if facet < len(up_f_origin_x):
                row.append(str(up_f_origin_x[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_f_origin_y):
                row.append(str(up_f_origin_y[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_f_origin_z):
                row.append(str(up_f_origin_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_canted_x_rotation):
                row.append(up_canted_x_rotation[facet])
            else:
                row.append('')  # If there are fewer angles than facets, fill with empty string
            if facet < len(up_canted_y_rotation):
                row.append(up_canted_y_rotation[facet])
            else:
                row.append('')
            if facet < len(up_x_sur_norm):
                row.append(str(up_x_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_y_sur_norm):
                row.append(str(up_y_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_z_sur_norm):
                row.append(str(up_z_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            row.append(azimuth)
            row.append(elevation)
            if facet < len(tracking_f_origin_x):
                row.append(str(tracking_f_origin_x[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_f_origin_y):
                row.append(str(tracking_f_origin_y[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_f_origin_z):
                row.append(str(tracking_f_origin_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_x_sur_norm):
                row.append(str(tracking_x_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_y_sur_norm):
                row.append(str(tracking_y_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_z_sur_norm):
                row.append(str(tracking_z_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(offset_z):
                row.append(str(offset_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(offset_x):
                row.append(str(offset_x[facet]).strip("[]"))
            else:
                row.append('')
            rows.append(row)

        heliostat_name.strip('""')
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")

        file_name = (
            heliostat_name
            + '_facet_details_canted_'
            + canting_details.lower()
            + '_axis_'
            + target
            + '_'
            + f'{float(target_x): .2f}'
            + '_'
            + f'{float(target_y): .2f}'
            + '_'
            + f'{float(target_z): .2f}'
            + '.csv'
        )
        # output_directory = os.path.join(self.output_path, 'data', 'output', self.source_file_body)
        file_path = os.path.join(actual_output_dir, file_name)

        os.makedirs(actual_output_dir, exist_ok=True)

        # Write to CSV file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    'Facet Index',
                    'Face Up Facet Origin X (m)',
                    'Face Up Facet Origin Y (m)',
                    'Face Up Facet Origin Z (m)',
                    'Face Up Canting Rotation about X (rad)',
                    'Face Up Canting Rotation about Y (rad)',
                    'Face Up Surface Normal X',
                    'Face Up Surface Normal Y',
                    'Face Up Surface Normal Z',
                    'Tracking Az (rad)',
                    'Tracking El (rad)',
                    'Tracking Facet Origin X (m)',
                    'Tracking Facet Origin Y (m)',
                    'Tracking Facet Origin Z (m)',
                    'Tracking Surface Normal X',
                    'Tracking Surface Normal Y',
                    'Tracking Surface Normal Z',
                    'Reflected Ray Target Plane Intersection X (m)',
                    'Reflected Ray Target Plane Intersection Z (m)',
                ]
            )
            writer.writerows(rows)

    def save_to_csv_time(
        self,
        canting_details: str,
        time: str,
        target: str,
        target_loc: Pxyz,
        heliostat_name: str,
        f_index: str,
        up_f_origin_x: float,
        up_f_origin_y: float,
        up_f_origin_z: float,
        up_canted_x_rotation: float,
        up_canted_y_rotation: float,
        up_x_sur_norm: float,
        up_y_sur_norm: float,
        up_z_sur_norm: float,
        azimuth: float,
        elevation: float,
        tracking_f_origin_x: float,
        tracking_f_origin_y: float,
        tracking_f_origin_z: float,
        tracking_x_sur_norm: float,
        tracking_y_sur_norm: float,
        tracking_z_sur_norm: float,
        sun_ray_x,
        sun_ray_y,
        sun_ray_z,
        reflected_ray_x,
        reflected_ray_y,
        reflected_ray_z,
        offset_x: float,
        offset_z: float,
        actual_output_dir: str = None,
    ):
        """
        Outputs a CSV file with canting information.

        Parameters
        ----------
        heliostat_name: abbreviated name of heliostat
        f_index : list of facet indexes
        up_f_origin_x: x- component of facet origin in respect to facet ensemble when center facet is at 0m
        up_f_origin_y: y- component of facet origin in respect to facet ensemble when center facet is at 0m
        up_f_origin_z: z- component of facet origin in respect to facet ensemble when center facet is at 0m
        up_canted_x_rotation: list of canted x angle rotations when heliostat is in UP orientation
        up_canted_y_rotation: list of canted y angle rotations when heliostat is in UP orientation
        up_x_sur_norm: x-component of the surface normal when heliostat is in UP orientation
        up_y_sur_norm: y-component of the surface normal when heliostat is in UP orientation
        up_z_sur_norm: z-component of the surface normal when heliostat is in UP orientation
        azimuth: tracking azimuth orientation in radians
        elevation: tracking elevation orientation in radians
        tracking_f_origin_x: x- component of facet origin in respect to facet ensemble when heliostat is tracking target
        tracking_f_origin_y: y- component of facet origin in respect to facet ensemble when heliostat is tracking target
        tracking_f_origin_z: z- component of facet origin in respect to facet ensemble when heliostat is tracking target
        tracking_x_sur_norm: x-component of the surface normal when heliostat is tracking target
        tracking_y_sur_norm: y-component of the surface normal when heliostat is tracking target
        tracking_z_sur_norm: z-component of the surface normal when heliostat is tracking target
        offset_x: offset in x direction of reflected ray intersection point on target plane from target
        offset_z: offset in z direction of reflected ray intersection point on target plane from target

        """

        # Both lists have same length
        assert len(up_canted_y_rotation) == len(
            up_canted_x_rotation
        ), "canted_x_angles and canted_y_angles must have the same length"

        # Creates list of rows, each row is a list of [facet, canted_x_angle, canted_y_angle, offset_z, offset_x]
        rows = []
        for facet in range(len(f_index)):
            row = [facet + 1]
            if facet < len(up_f_origin_x):
                row.append(str(up_f_origin_x[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_f_origin_y):
                row.append(str(up_f_origin_y[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_f_origin_z):
                row.append(str(up_f_origin_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_canted_x_rotation):
                row.append(up_canted_x_rotation[facet])
            else:
                row.append('')  # If there are fewer angles than facets, fill with empty string
            if facet < len(up_canted_y_rotation):
                row.append(up_canted_y_rotation[facet])
            else:
                row.append('')
            if facet < len(up_x_sur_norm):
                row.append(str(up_x_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_y_sur_norm):
                row.append(str(up_y_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(up_z_sur_norm):
                row.append(str(up_z_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            row.append(azimuth)
            row.append(elevation)
            if facet < len(sun_ray_x):
                row.append(str(sun_ray_x[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(sun_ray_y):
                row.append(str(sun_ray_y[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(sun_ray_z):
                row.append(str(sun_ray_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(reflected_ray_x):
                row.append(str(reflected_ray_x[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(reflected_ray_y):
                row.append(str(reflected_ray_y[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(reflected_ray_z):
                row.append(str(reflected_ray_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_f_origin_x):
                row.append(str(tracking_f_origin_x[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_f_origin_y):
                row.append(str(tracking_f_origin_y[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_f_origin_z):
                row.append(str(tracking_f_origin_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_x_sur_norm):
                row.append(str(tracking_x_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_y_sur_norm):
                row.append(str(tracking_y_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(tracking_z_sur_norm):
                row.append(str(tracking_z_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(offset_z):
                row.append(str(offset_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(offset_x):
                row.append(str(offset_x[facet]).strip("[]"))
            else:
                row.append('')
            rows.append(row)

        heliostat_name.strip('""')
        time.strip('""')
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")

        file_name = (
            heliostat_name
            + '_facet_details_canted_'
            + canting_details.lower()
            # +'time_of_day'
            + '_axis_'
            + target
            + '_'
            + f'{float(target_x): .2f}'
            + '_'
            + f'{float(target_y): .2f}'
            + '_'
            + f'{float(target_z): .2f}'
            + '_time_'
            + time
            + '.csv'
        )
        # output_directory = os.path.join(self.output_path, 'data', 'output', self.source_file_body)
        file_path = os.path.join(actual_output_dir, file_name)

        os.makedirs(actual_output_dir, exist_ok=True)

        # Write to CSV file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    'Facet Index',
                    'Face Up Facet Origin X (m)',
                    'Face Up Facet Origin Y (m)',
                    'Face Up Facet Origin Z (m)',
                    'Face Up Canting Rotation about X (rad)',
                    'Face Up Canting Rotation about Y (rad)',
                    'Face Up Surface Normal X',
                    'Face Up Surface Normal Y',
                    'Face Up Surface Normal Z',
                    'Tracking Az (rad)',
                    'Tracking El (rad)',
                    'Sun Ray X',
                    'Sun Ray Y',
                    'Sun Ray Z',
                    'Reflected Ray X',
                    'Reflected Ray Y',
                    'Reflected Ray Z',
                    'Tracking Facet Origin X (m)',
                    'Tracking Facet Origin Y (m)',
                    'Tracking Facet Origin Z (m)',
                    'Tracking Surface Normal X',
                    'Tracking Surface Normal Y',
                    'Tracking Surface Normal Z',
                    'Reflected Ray Target Plane Intersection X (m)',
                    'Reflected Ray Target Plane Intersection Z (m)',
                ]
            )
            writer.writerows(rows)

    def save_az_el_to_csv(
        self,
        output_dir: str,
        heliostat_name: str,
        azimuth_rad: float,
        elevation_rad: float,
        azimuth_deg: float,
        elevation_deg: float,
    ):
        """
        Outputs a CSV file with canting information.

        Parameters
        ----------
        heliostat_name: abbreviated name of heliostat
        azimuth_rad: tracking azimuth orientation in radians
        elevation_rad: tracking elevation orientation in radians
        azimuth_deg: tracking azimuth orientation in degrees
        elevation_deg: tracking elevation orientation in degrees
        """

        # Creates list of rows, each row is a list of [facet, canted_x_angle, canted_y_angle, offset_z, offset_x]
        rows = [
            ['Heliostat', 'Azimuth (rad)', 'Elevation (rad)', 'Azimuth (deg)', 'Elevation (deg)'],  # Header row
            [
                heliostat_name.strip('"'),
                azimuth_rad,
                elevation_rad,
                azimuth_deg.strip('"'),
                elevation_deg.strip('"'),
            ],  # Data row for azimuth
        ]

        file_name = heliostat_name + '_SOFAST_Tower_Azimuth_Elevation.csv'
        file_path = os.path.join(output_dir, file_name)

        os.makedirs(output_dir, exist_ok=True)

        # Write to CSV file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def save_master_csv(self):
        """
        Compiles data from multiple heliostat CSV files into a master CSV file.

        Parameters
        ----------
        output_dir: directory where the individual heliostat CSV files are stored
        """
        # Initialize a list to hold all rows
        master_rows = [
            ['Heliostat', 'Azimuth (rad)', 'Elevation (rad)', 'Azimuth (deg)', 'Elevation (deg)']
        ]  # Header row

        # Loop through each file in the output directory
        parent_folder_name = os.path.join(self.actual_output_dir, 'SofastConfigurations')
        actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name)
        heliostat_spec_list = self.solar_field.heliostat_name_list()
        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            heliostat_dir = os.path.join(actual_output_dir, helio)

            # Check if it's a directory
            if os.path.isdir(heliostat_dir):
                # Construct the expected CSV file path
                csv_file_path = os.path.join(heliostat_dir, f'{heliostat_name}_SOFAST_Tower_Azimuth_Elevation.csv')

                # Check if the CSV file exists
                if os.path.isfile(csv_file_path):
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(csv_file_path)

                    # Append the DataFrame to master_rows, excluding the header
                    for index, row in df.iterrows():
                        master_rows.append(
                            [
                                row['Heliostat'],
                                row['Azimuth (rad)'],
                                row['Elevation (rad)'],
                                row['Azimuth (deg)'],
                                row['Elevation (deg)'],
                            ]
                        )

        # Define the master CSV file path
        master_file_path = os.path.join(actual_output_dir, 'NSTTF_SOFAST_Tower_Azimuth_Elevation.csv')

        # Write to master CSV file
        with open(master_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(master_rows)

        print("Master CSV file created successfully.")

    def read_csv_float(self, input_dir_body_ext: str, column_name: str):
        # Check if the file exists at the specified path
        if not ft.file_exists(input_dir_body_ext):
            raise OSError('In FrameNameXyList.load(), file does not exist: ' + str(input_dir_body_ext))
        # Open and read the file.
        with open(input_dir_body_ext, mode='r', newline='') as input_stream:
            reader = csv.reader(input_stream, delimiter=',')

            # Read the header to find the index of the desired column
            header = next(reader)
            column_index = header.index(column_name)

            # Extract the column into a list, excluding the header
            # column_data = [float(row[column_index]) for row in reader]
            column_data = [str(row[column_index]) for row in reader]

        return column_data

    def test_on_axis_canting_1(self) -> None:
        """
        Draws binary search algorithm with the target using both elevation and azimuth.
        """
        # Initialize test.
        self.start_test()

        f_index = '13'
        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        # View setup
        title = 'On-Axis Canted Heliostat ' + heliostat_name
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'using facet 1.'
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177.5)
        test_el = np.deg2rad(43.2)

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        test_canted_x_angle = 0
        test_canted_y_angle = 0
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])
        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_config = hc.HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
        test_heliostat.set_orientation(test_config)

        # style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        solar_field_style = rcsf.heliostat_blanks()
        solar_field_style.add_special_names(test_heliostat, surface_normal_facet_style)
        tower_control = rct.normal_tower()

        az, el, intersection = self.find_single_facet_azimuth_el_value(
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )
        test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
        test_heliostat.set_orientation(test_config)

        test_facet = test_heliostat.facet_ensemble.lookup_facet("13")
        __, trc_sur_norm = test_facet.survey_of_points(Resolution.center())
        sun_ray = -trc_sur_norm

        #   dimensions for intersection drawing
        x_min = -0.01
        x_max = 0.01
        exaggerated_x = [x_min, x_max]
        z_min = 63.55 - 0.01
        z_max = 63.55 + 0.01
        exaggerated_z = [z_max, z_min]
        x_lim = [-10, 10]  ##TODO mhh figure out limits that are automated
        y_lim = [0, 90]
        z_lim = [-10, 70]
        count = int(0)

        heliostat = test_heliostat.no_parent_copy()
        heliostat_loc = test_heliostat._self_to_parent_transform

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
        test_heliostat.set_orientation(test_config)
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 120

        scene = Scene()
        scene.add_object(heliostat)
        scene.set_position_in_space(heliostat, heliostat_loc)
        # Add ray trace to field
        # sun_1ray = LightSourceSun()
        # sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(51),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XY Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(52),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XY Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(53),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="YZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(54),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(55),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + " Intersection at Aimpoint All Facets",
            caption='A single Sandia NSTTF heliostat ' + heliostat_name,
            comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        AIMPOINT.draw_points(figure_control.view, rcps.marker(color='r'))
        self.show_save_and_check_figure(figure_control)

    def test_on_axis_canting_angles(self) -> None:
        canting_details = 'on'
        heliostat_spec_list: list[str]
        heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']

        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            folder_name = os.path.join(self.actual_output_dir, helio)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            actual_output_dir = os.path.join(self.actual_output_dir, helio)
            expected_output_dir = os.path.join(self.expected_output_dir, helio)

            self.find_all_on_axis_canting_angles(heliostat_name, actual_output_dir)
            self.create_on_axis_canted_heliostat(heliostat_name, actual_output_dir, expected_output_dir)
            self.example_canting_calculated(heliostat_name, canting_details, actual_output_dir, expected_output_dir)
            self.example_canting_bar_charts_calculated(
                heliostat_name, canting_details, actual_output_dir, expected_output_dir
            )

    def find_all_on_axis_canting_angles(self, heliostat_name, actual_output_dir: str = None) -> None:
        """
        Draws binary search algorithm with the target using both elevation and azimuth.
        """
        # Initialize test.
        self.start_test()

        f_index = '13'
        canting_details = 'on'
        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        test_heliostat = [heliostat_name]

        # View setup
        title = 'On-Axis Canted Heliostat ' + heliostat_name
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'using facet 1.'
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177.5)
        test_el = np.deg2rad(43.2)

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        test_canted_x_angle = 0
        test_canted_y_angle = 0
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])
        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_config = hc.HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
        test_heliostat.set_orientation(test_config)

        # style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        solar_field_style = rcsf.heliostat_blanks()
        solar_field_style.add_special_names(test_heliostat, surface_normal_facet_style)
        tower_control = rct.normal_tower()

        azim, elev, intersection = self.find_single_facet_azimuth_el_value(
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )
        test_config = hc.HeliostatConfiguration('az-el', az=azim, el=elev)
        test_heliostat.set_orientation(test_config)

        test_facet = test_heliostat.facet_ensemble.lookup_facet("13")
        __, trc_sur_norm = test_facet.survey_of_points(Resolution.center())
        sun_ray = -trc_sur_norm

        canted_x_angles, canted_y_angles, f_idx, offset_x, offset_z = self.find_all_canting_angle_values(
            sun_ray=sun_ray,
            aimpoint=AIMPOINT,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            azimuth=azim,
            elevation=elev,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        canting_rotations = []

        for facet in range(0, 25):
            # position = Pxyz.merge(test_heliostat._self_to_parent_transform.apply(Pxyz.origin()))
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
            print(f"rotation: {canting.as_euler('xyz')}")
        test_heliostat.set_facet_canting(canting_rotations)

        trc_f_origin_x = []
        trc_f_origin_y = []
        trc_f_origin_z = []
        trc_x_sur_norm = []
        trc_y_sur_norm = []
        trc_z_sur_norm = []

        # tracking
        for facet in test_heliostat.facet_ensemble.facets:
            trc_facet_origin, trc_sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            trc_f_origin_x.append(trc_facet_origin.x)
            trc_f_origin_y.append(trc_facet_origin.y)
            trc_f_origin_z.append(trc_facet_origin.z)
            trc_x_sur_norm.append(trc_sur_norm.x)
            trc_y_sur_norm.append(trc_sur_norm.y)
            trc_z_sur_norm.append(trc_sur_norm.z)

        az = np.deg2rad(180)
        el = np.deg2rad(90)
        test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
        test_heliostat.set_orientation(test_config)

        f_origin_x = []
        f_origin_y = []
        f_origin_z = []
        x_sur_norm = []
        y_sur_norm = []
        z_sur_norm = []

        # face-up
        for facet in test_heliostat.facet_ensemble.facets:
            _, sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            x_sur_norm.append(sur_norm.x)
            y_sur_norm.append(sur_norm.y)
            z_sur_norm.append(sur_norm.z)

        for facet in test_heliostat.facet_ensemble.no_parent_copy().facets:
            facet_origin, _ = facet.survey_of_points(Resolution.center())
            f_origin_x.append(facet_origin.x)
            f_origin_y.append(facet_origin.y)
            f_origin_z.append(facet_origin.z)

        self.save_to_csv(
            canting_details,
            heliostat_name,
            f_idx,
            f_origin_x,
            f_origin_y,
            f_origin_z,
            canted_x_angles,
            canted_y_angles,
            x_sur_norm,
            y_sur_norm,
            z_sur_norm,
            azim,
            elev,
            trc_f_origin_x,
            trc_f_origin_y,
            trc_f_origin_z,
            trc_x_sur_norm,
            trc_y_sur_norm,
            trc_z_sur_norm,
            offset_z,
            offset_x,
            actual_output_dir,
        )

    def test_off_axis_code(
        self,
        heliostat_name: str = None,
        target: str = 'target',
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:

        # Initialize test.
        self.start_test()

        # View Setup
        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # View Setup
        canting_details = 'Off'
        # heliostat_name = "9W1"
        f_index = "13"
        title = "Off-Axis Canted " + heliostat_name
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []
        count = int(0)

        if target == "G3P3":
            wire_frame = 'G3P3 Tower'
        elif target == 'BCS':
            wire_frame = 'BCS Tower'
        else:
            wire_frame = None

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target, wire_frame])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        azimuth = test_heliostat._az
        elevation = test_heliostat._el
        print(f"az and el = {azimuth}, {elevation}")
        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        canted_x_angles, canted_y_angles, f_idx, offset_x, offset_z = self.find_all_canting_angle_values_normal(
            time=TIME,
            aimpoint=AIMPOINT,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            azimuth=azimuth,
            elevation=elevation,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)

        trace = rt.trace_scene(scene, Resolution.center())

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        heliostat_origin, __ = test_heliostat.survey_of_points(Resolution.center())

        # array = sun_1ray.get_incident_rays(heliostat_origin)[0]
        # print(array.current_direction.x)
        # print('done')

        focal_length = Pxyz.distance(target_loc, heliostat_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80
        x_min = float(target_loc.x[0]) - 0.01
        x_max = float(target_loc.x[0]) + 0.01
        exaggerated_x = [x_min, x_max]
        z_min = f'{(float(target_loc.z[0]) - 0.01): .2f}'
        z_max = f'{(float(target_loc.z[0]) + 0.01): .2f}'
        exaggerated_z = [float(z_max), float(z_min)]

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )
        target_origin = AIMPOINT
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
        )
        target_file_name = (
            target + ' ' + f'{float(target_x): .2f}' + ' ' + f'{float(target_y): .2f}' + ' ' + f'{float(target_z): .2f}'
        )
        line_thickness: float = rcps.thick()

        # DRAWING
        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
            rca.meters(grid=False),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace 3D ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        # trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Birds View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Looking North View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Side View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Intersection at Aimpoint ' + target_file_name,
            title=title + ', ' + target_plot_name,
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count_ = int(15)

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            # Add ray trace to field
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
            count_ = int(count_ + 1)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
                rca.meters(grid=False),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Ray Trace 3D ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            target_origin.draw_points(figure_control.view)
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count_ = int(count_ + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Intersection at Aimpoint ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

        trc_f_origin_x = []
        trc_f_origin_y = []
        trc_f_origin_z = []
        trc_x_sur_norm = []
        trc_y_sur_norm = []
        trc_z_sur_norm = []

        # tracking
        for facet in test_heliostat.facet_ensemble.facets:
            trc_facet_origin, trc_sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            trc_f_origin_x.append(trc_facet_origin.x)
            trc_f_origin_y.append(trc_facet_origin.y)
            trc_f_origin_z.append(trc_facet_origin.z)
            trc_x_sur_norm.append(trc_sur_norm.x)
            trc_y_sur_norm.append(trc_sur_norm.y)
            trc_z_sur_norm.append(trc_sur_norm.z)

        az = np.deg2rad(180)
        el = np.deg2rad(90)
        test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
        test_heliostat.set_orientation(test_config)

        f_origin_x = []
        f_origin_y = []
        f_origin_z = []
        x_sur_norm = []
        y_sur_norm = []
        z_sur_norm = []

        # face-up
        for facet in test_heliostat.facet_ensemble.facets:
            _, sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            x_sur_norm.append(sur_norm.x)
            y_sur_norm.append(sur_norm.y)
            z_sur_norm.append(sur_norm.z)

        for facet in test_heliostat.facet_ensemble.no_parent_copy().facets:
            facet_origin, _ = facet.survey_of_points(Resolution.center())
            f_origin_x.append(facet_origin.x)
            f_origin_y.append(facet_origin.y)
            f_origin_z.append(facet_origin.z)

        self.save_to_csv(
            canting_details,
            target,
            target_loc,
            heliostat_name,
            f_idx,
            f_origin_x,
            f_origin_y,
            f_origin_z,
            canted_x_angles,
            canted_y_angles,
            x_sur_norm,
            y_sur_norm,
            z_sur_norm,
            azimuth,
            elevation,
            trc_f_origin_x,
            trc_f_origin_y,
            trc_f_origin_z,
            trc_x_sur_norm,
            trc_y_sur_norm,
            trc_z_sur_norm,
            offset_z,
            offset_x,
            actual_output_dir,
        )

    def test_off_axis_code_time_of_day(
        self,
        heliostat_name: str = None,
        target: str = 'target',
        actual_output_dir: str = None,
        expected_output_dir: str = None,
        TIME: tuple = None,
        time_name: str = None,
    ) -> None:

        # Initialize test.
        self.start_test()

        # View Setup
        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # View Setup
        canting_details = 'TOD'
        # heliostat_name = "9W1"
        f_index = "13"
        title = "Time Canted " + heliostat_name
        title_file = "Time Of Day Canted " + heliostat_name
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []
        count = int(0)

        if target == "G3P3":
            wire_frame = 'G3P3 Tower'
        elif target == 'BCS':
            wire_frame = 'BCS Tower'
        else:
            wire_frame = None

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target, wire_frame])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        azimuth = test_heliostat._az
        elevation = test_heliostat._el
        print(f"az and el = {azimuth}, {elevation}")
        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        canted_x_angles, canted_y_angles, f_idx, offset_x, offset_z = self.find_all_canting_angle_values_normal(
            time=TIME,
            aimpoint=AIMPOINT,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            azimuth=azimuth,
            elevation=elevation,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center())

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        heliostat_origin, __ = test_heliostat.survey_of_points(Resolution.center())

        focal_length = Pxyz.distance(target_loc, heliostat_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80
        x_min = float(target_loc.x[0]) - 6
        x_max = float(target_loc.x[0]) + 6
        exaggerated_x = [x_min, x_max]
        z_min = f'{(float(target_loc.z[0]) - 6): .2f}'
        z_max = f'{(float(target_loc.z[0]) + 6): .2f}'
        exaggerated_z = [float(z_max), float(z_min)]

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )
        time_name_new = time_name.replace('_', ':')
        target_origin = AIMPOINT
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
            + ' Time '
            + time_name_new
        )
        target_file_name = (
            target
            + ' '
            + f'{float(target_x): .2f}'
            + ' '
            + f'{float(target_y): .2f}'
            + ' '
            + f'{float(target_z): .2f}'
            + ' Time '
            + time_name
        )

        # DRAWING
        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
            rca.meters(grid=False),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title_file + ' Ray Trace 3D ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title_file + ' Ray Trace Birds View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title_file + ' Ray Trace Looking North View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title_file + ' Ray Trace Side View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title_file + ' Intersection at Aimpoint ' + target_file_name,
            title=title + ', Intersection at Aimpoint',
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count_ = int(15)

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            # Add ray trace to field
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
            count_ = int(count_ + 1)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
                rca.meters(grid=False),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title_file + ' Facet ' + facet.name + ' Ray Trace 3D ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            target_origin.draw_points(figure_control.view)
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count_ = int(count_ + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title_file + ' Intersection at Aimpoint ' + target_file_name,
                title=title + ', Intersection at Aimpoint',
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

        trc_f_origin_x = []
        trc_f_origin_y = []
        trc_f_origin_z = []
        trc_x_sur_norm = []
        trc_y_sur_norm = []
        trc_z_sur_norm = []

        # tracking
        for facet in test_heliostat.facet_ensemble.facets:
            trc_facet_origin, trc_sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            trc_f_origin_x.append(trc_facet_origin.x)
            trc_f_origin_y.append(trc_facet_origin.y)
            trc_f_origin_z.append(trc_facet_origin.z)
            trc_x_sur_norm.append(trc_sur_norm.x)
            trc_y_sur_norm.append(trc_sur_norm.y)
            trc_z_sur_norm.append(trc_sur_norm.z)

        az = np.deg2rad(180)
        el = np.deg2rad(90)
        test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
        test_heliostat.set_orientation(test_config)

        f_origin_x = []
        f_origin_y = []
        f_origin_z = []
        x_sur_norm = []
        y_sur_norm = []
        z_sur_norm = []

        # face-up
        for facet in test_heliostat.facet_ensemble.facets:
            _, sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            x_sur_norm.append(sur_norm.x)
            y_sur_norm.append(sur_norm.y)
            z_sur_norm.append(sur_norm.z)

        for facet in test_heliostat.facet_ensemble.no_parent_copy().facets:
            facet_origin, _ = facet.survey_of_points(Resolution.center())
            f_origin_x.append(facet_origin.x)
            f_origin_y.append(facet_origin.y)
            f_origin_z.append(facet_origin.z)

        self.save_to_csv_time(
            canting_details,
            time_name,
            target,
            target_loc,
            heliostat_name,
            f_idx,
            f_origin_x,
            f_origin_y,
            f_origin_z,
            canted_x_angles,
            canted_y_angles,
            x_sur_norm,
            y_sur_norm,
            z_sur_norm,
            azimuth,
            elevation,
            trc_f_origin_x,
            trc_f_origin_y,
            trc_f_origin_z,
            trc_x_sur_norm,
            trc_y_sur_norm,
            trc_z_sur_norm,
            offset_z,
            offset_x,
            actual_output_dir,
        )

    def test_off_axis_code_time_of_day_not_calculated(
        self,
        heliostat_name: str = None,
        target: str = 'target',
        canting_dir: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
        TIME: tuple = None,
        time_name: str = None,
    ) -> None:

        # Initialize test.
        self.start_test()

        # View Setup
        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # View Setup
        canting_details = 'Off'
        # heliostat_name = "9W1"
        f_index = "13"
        title = "Off-Axis Canted " + heliostat_name
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []
        count = int(0)

        if target == "G3P3":
            wire_frame = 'G3P3 Tower'
        elif target == 'BCS':
            wire_frame = 'BCS Tower'
        else:
            wire_frame = None

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target, wire_frame])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        target_plane_normal = Uxyz([0, 1, 0])
        target_plane_normal_y = Uxyz([1, 0, 0])
        test_tolerance = 0.0001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        time_name_new = time_name.replace('_', ':')
        target_origin = AIMPOINT
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
            + ' Time '
            + time_name_new
        )
        target_file_name = (
            target
            + ' '
            + f'{float(target_x): .2f}'
            + ' '
            + f'{float(target_y): .2f}'
            + ' '
            + f'{float(target_z): .2f}'
            + ' Time '
            + time_name
        )

        opencsp_dir = canting_dir

        canted_x_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about X (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about Y (rad)',
        )

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        azimuth = test_heliostat._az
        elevation = test_heliostat._el
        print(f"az and el = {azimuth}, {elevation}")
        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        canting_rotations: list[TransformXYZ] = []

        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
            print(canted_x_angles[facet])
        test_heliostat.set_facet_cantings(canting_rotations)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)

        trace = rt.trace_scene(scene, Resolution.center())

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)
        place_to_intersect_y = (AIMPOINT, target_plane_normal_y)
        intersection_point_y = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect_y)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        heliostat_origin, __ = test_heliostat.survey_of_points(Resolution.center())

        # array = sun_1ray.get_incident_rays(heliostat_origin)[0]
        # print(array.current_direction.x)
        # print('done')

        focal_length = Pxyz.distance(target_loc, heliostat_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80
        x_min = float(target_loc.x[0]) - 6
        x_max = float(target_loc.x[0]) + 6
        exaggerated_x = [x_max, x_min]
        y_min = f'{(float(target_loc.y[0]) - 6): .2f}'
        y_max = f'{(float(target_loc.y[0]) + 6): .2f}'
        exaggerated_y = [float(y_min), float(y_max)]
        z_min = f'{(float(target_loc.z[0]) - 6): .2f}'
        z_max = f'{(float(target_loc.z[0]) + 6): .2f}'
        exaggerated_z = [float(z_min), float(z_max)]

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )

        # DRAWING
        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
            rca.meters(grid=False),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace 3D ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Birds View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Looking North View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Side View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' South Intersection at Aimpoint ' + target_file_name,
            title=title + ', South Intersection at Aimpoint, Time ' + time_name_new,
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' West Intersection at Aimpoint ' + target_file_name,
            title=title + ', West Intersection at Aimpoint, Time ' + time_name_new,
        )
        figure_control.y_limits = exaggerated_y
        figure_control.z_limits = exaggerated_z
        intersection_point_y.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count_ = int(16)

        print("sun_ray\n")
        offset_x = []
        offset_z = []
        sun_ray_x = []
        sun_ray_y = []
        sun_ray_z = []

        reflected_ray_x = []
        reflected_ray_y = []
        reflected_ray_z = []

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            # Add ray trace to field
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)
            place_to_intersect_y = (AIMPOINT, target_plane_normal_y)
            intersection_point_y = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect_y)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
            count_ = int(count_ + 1)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
                rca.meters(grid=False),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Ray Trace 3D',
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            target_origin.draw_points(figure_control.view)
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count_ = int(count_ + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' South Intersection ' + target_file_name,
                title=title + ', Facet ' + facet.name + ', South Intersection, Time ' + time_name_new,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count_ = int(count_ + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_yz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' West Intersection ' + target_file_name,
                title=title + ', Facet ' + facet.name + ', West Intersection, Time ' + time_name_new,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.y_limits = exaggerated_y
            figure_control.z_limits = exaggerated_z
            intersection_point_y.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            x_intersect = intersection_point.intersection_points.x
            offset_x.append(x_intersect)
            z_intersect = intersection_point.intersection_points.z
            offset_z.append(z_intersect)

            sun_ray_x.append(sun_1ray.incident_rays[0].init_direction.x)
            sun_ray_y.append(sun_1ray.incident_rays[0].init_direction.y)
            sun_ray_z.append(sun_1ray.incident_rays[0].init_direction.z)

            reflected_ray_x.append(trace.light_paths[0].current_direction.x)
            reflected_ray_y.append(trace.light_paths[0].current_direction.y)
            reflected_ray_z.append(trace.light_paths[0].current_direction.z)

        trc_f_origin_x = []
        trc_f_origin_y = []
        trc_f_origin_z = []
        trc_x_sur_norm = []
        trc_y_sur_norm = []
        trc_z_sur_norm = []

        # tracking
        for facet in test_heliostat.facet_ensemble.facets:
            trc_facet_origin, trc_sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            trc_f_origin_x.append(trc_facet_origin.x)
            trc_f_origin_y.append(trc_facet_origin.y)
            trc_f_origin_z.append(trc_facet_origin.z)
            trc_x_sur_norm.append(trc_sur_norm.x)
            trc_y_sur_norm.append(trc_sur_norm.y)
            trc_z_sur_norm.append(trc_sur_norm.z)

        az = np.deg2rad(180)
        el = np.deg2rad(90)
        test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
        test_heliostat.set_orientation(test_config)

        f_origin_x = []
        f_origin_y = []
        f_origin_z = []
        x_sur_norm = []
        y_sur_norm = []
        z_sur_norm = []

        # face-up
        for facet in test_heliostat.facet_ensemble.facets:
            _, sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            x_sur_norm.append(sur_norm.x)
            y_sur_norm.append(sur_norm.y)
            z_sur_norm.append(sur_norm.z)

        for facet in test_heliostat.facet_ensemble.no_parent_copy().facets:
            facet_origin, _ = facet.survey_of_points(Resolution.center())
            f_origin_x.append(facet_origin.x)
            f_origin_y.append(facet_origin.y)
            f_origin_z.append(facet_origin.z)

        f_idx = [
            '1',
            '2',
            '3',
            '4',
            '5',
            '7',
            '8',
            '9',
            '10',
            '11',
            '12',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21',
            '22',
            '23',
            '24',
            '25',
        ]

        self.save_to_csv_time(
            canting_details,
            time_name,
            target,
            target_loc,
            heliostat_name,
            f_idx,
            f_origin_x,
            f_origin_y,
            f_origin_z,
            canted_x_angles,
            canted_y_angles,
            x_sur_norm,
            y_sur_norm,
            z_sur_norm,
            azimuth,
            elevation,
            trc_f_origin_x,
            trc_f_origin_y,
            trc_f_origin_z,
            trc_x_sur_norm,
            trc_y_sur_norm,
            trc_z_sur_norm,
            sun_ray_x,
            sun_ray_y,
            sun_ray_z,
            reflected_ray_x,
            reflected_ray_y,
            reflected_ray_z,
            offset_z,
            offset_x,
            actual_output_dir,
        )

    def test_off_axis_code_individual_facets(
        self,
        heliostat_name: str,
        target: str = 'target',
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:

        # Initialize test.
        self.start_test()

        # View Setup
        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # View Setup
        canting_details = 'Off'
        # heliostat_name = "9W1"
        f_index = "13"
        title = "Off-Axis Canted " + heliostat_name
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []
        count = int(15)
        if target == "G3P3":
            wire_frame = 'G3P3 Tower'
        elif target == 'BCS':
            wire_frame = 'BCS Tower'
        else:
            wire_frame = None

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target, wire_frame])

        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        azimuth = test_heliostat._az
        elevation = test_heliostat._el
        print(f"az and el = {azimuth}, {elevation}")
        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        canted_x_angles, canted_y_angles, f_idx, offset_x, offset_z = self.find_all_canting_angle_values_normal(
            time=TIME,
            aimpoint=AIMPOINT,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            azimuth=azimuth,
            elevation=elevation,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center())

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )
        target_origin = AIMPOINT
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
        )
        target_file_name = (
            target + ' ' + f'{float(target_x): .2f}' + ' ' + f'{float(target_y): .2f}' + ' ' + f'{float(target_z): .2f}'
        )

        # DRAWING
        heliostat_origin, __ = test_heliostat.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, heliostat_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80
        x_min = float(target_loc.x) - 0.01
        x_max = float(target_loc.x) + 0.01
        exaggerated_x = [x_min, x_max]
        z_min = float(target_loc.z) - 0.01
        z_max = float(target_loc.z) + 0.01
        exaggerated_z = [z_max, z_min]

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            # Add ray trace to field
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
            count = int(count + 1)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
                rca.meters(grid=False),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                name=title + ' Facet ' + facet.name + ' Ray Trace 3D ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            target_origin.draw_points(figure_control.view)
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count = int(count + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                name=title + ' Facet ' + facet.name + ' Intersection at Aimpoint ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

    def test_on_axis_code_individual_facets(
        self,
        heliostat_name: str,
        target: str = 'target',
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:

        # Initialize test.
        self.start_test()

        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # View Setup
        canting_details = 'On'
        # heliostat_name = "9W1"
        f_index = "13"
        title = "On-Axis Canted " + heliostat_name
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []
        count = int(15)

        if target == "G3P3":
            wire_frame = 'G3P3 Tower'
        else:
            wire_frame = None

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        # Initial positions
        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        test_azimuth = test_heliostat._az
        test_elevation = test_heliostat._el

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        azim, elev, intersection = self.find_single_facet_azimuth_el_value(
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_elevation,
            tolerance=test_tolerance,
        )

        print(f"az and el = {azim}, {elev}")

        canted_x_angles, canted_y_angles, f_idx, offset_x, offset_z = self.find_all_canting_angle_values_normal(
            aimpoint=AIMPOINT,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            azimuth=azim,
            elevation=elev,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        canting_rotations: list[TransformXYZ] = []

        test_config = hc.HeliostatConfiguration('az-el', az=azim, el=elev)
        test_heliostat.set_orientation(test_config)

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )
        target_origin = AIMPOINT

        test_facet = test_heliostat.facet_ensemble.lookup_facet("13")
        facet_origin, trc_sur_norm = test_facet.survey_of_points(Resolution.center())
        sun_ray = -trc_sur_norm

        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80
        x_min = -0.01
        x_max = 0.01
        exaggerated_x = [x_min, x_max]
        z_min = float(target_loc.z) - 0.01
        z_max = float(target_loc.z) + 0.01
        exaggerated_z = [z_max, z_min]

        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
        )
        target_file_name = (
            target + ' ' + f'{float(target_x): .2f}' + ' ' + f'{float(target_y): .2f}' + ' ' + f'{float(target_z): .2f}'
        )

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            # Add ray trace to field
            sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
            count = int(count + 1)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
                rca.meters(grid=False),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                name=title + ' Facet ' + facet.name + ' Ray Trace 3D ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            target_origin.draw_points(figure_control.view)
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count = int(count + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                name=title + ' Facet ' + facet.name + ' Intersection at Aimpoint ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

    def test_on_axis_code(
        self,
        heliostat_name: str = None,
        target: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:
        """
        Draws ray trace intersection for single facet with the target.

        """
        ## TODO MHH fix off-axis code
        # Initialize test.
        self.start_test()

        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # View Setup
        canting_details = 'On'
        # heliostat_name = "9W1"
        f_index = "13"
        title = "On-Axis Canted " + heliostat_name
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []
        count = int(0)

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        # Initial positions
        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        test_azimuth = test_heliostat._az
        test_elevation = test_heliostat._el

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        azim, elev, intersection = self.find_single_facet_azimuth_el_value(
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_elevation,
            tolerance=test_tolerance,
        )

        print(f"az and el = {azim}, {elev}")

        canted_x_angles, canted_y_angles, f_idx, offset_x, offset_z = self.find_all_canting_angle_values_normal(
            aimpoint=AIMPOINT,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            azimuth=azim,
            elevation=elev,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        canting_rotations: list[TransformXYZ] = []

        test_config = hc.HeliostatConfiguration('az-el', az=azim, el=elev)
        test_heliostat.set_orientation(test_config)

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )
        target_origin = AIMPOINT

        test_facet = test_heliostat.facet_ensemble.lookup_facet("13")
        facet_origin, trc_sur_norm = test_facet.survey_of_points(Resolution.center())
        sun_ray = -trc_sur_norm

        sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)

        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80
        x_min = -0.01
        x_max = 0.01
        exaggerated_x = [x_min, x_max]
        z_min = f'{(float(target_loc.z[0]) - 0.01): .2f}'
        z_max = f'{(float(target_loc.z[0]) + 0.01): .2f}'
        exaggerated_z = [float(z_max), float(z_min)]

        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
        )
        target_file_name = (
            target + ' ' + f'{float(target_x): .2f}' + ' ' + f'{float(target_y): .2f}' + ' ' + f'{float(target_z): .2f}'
        )

        # DRAWING
        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
            rca.meters(grid=False),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace 3D ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Birds View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Looking North View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Side View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Intersection at Aimpoint ' + target_file_name,
            title=title + ', ' + target_plot_name,
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count_ = int(15)
        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
            count_ = int(count_ + 1)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
                rca.meters(grid=False),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Ray Trace 3D ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            target_origin.draw_points(figure_control.view)
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count_ = int(count_ + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Intersection at Aimpoint ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

        trc_f_origin_x = []
        trc_f_origin_y = []
        trc_f_origin_z = []
        trc_x_sur_norm = []
        trc_y_sur_norm = []
        trc_z_sur_norm = []

        # tracking
        for facet in test_heliostat.facet_ensemble.facets:
            trc_facet_origin, trc_sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            trc_f_origin_x.append(trc_facet_origin.x)
            trc_f_origin_y.append(trc_facet_origin.y)
            trc_f_origin_z.append(trc_facet_origin.z)
            trc_x_sur_norm.append(trc_sur_norm.x)
            trc_y_sur_norm.append(trc_sur_norm.y)
            trc_z_sur_norm.append(trc_sur_norm.z)

        az = np.deg2rad(180)
        el = np.deg2rad(90)
        test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
        test_heliostat.set_orientation(test_config)

        f_origin_x = []
        f_origin_y = []
        f_origin_z = []
        x_sur_norm = []
        y_sur_norm = []
        z_sur_norm = []

        # face-up
        for facet in test_heliostat.facet_ensemble.facets:
            _, sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            x_sur_norm.append(sur_norm.x)
            y_sur_norm.append(sur_norm.y)
            z_sur_norm.append(sur_norm.z)

        for facet in test_heliostat.facet_ensemble.no_parent_copy().facets:
            facet_origin, _ = facet.survey_of_points(Resolution.center())
            f_origin_x.append(facet_origin.x)
            f_origin_y.append(facet_origin.y)
            f_origin_z.append(facet_origin.z)

        self.save_to_csv(
            canting_details,
            target,
            target_loc,
            heliostat_name,
            f_idx,
            f_origin_x,
            f_origin_y,
            f_origin_z,
            canted_x_angles,
            canted_y_angles,
            x_sur_norm,
            y_sur_norm,
            z_sur_norm,
            azim,
            elev,
            trc_f_origin_x,
            trc_f_origin_y,
            trc_f_origin_z,
            trc_x_sur_norm,
            trc_y_sur_norm,
            trc_z_sur_norm,
            offset_z,
            offset_x,
            actual_output_dir,
        )

    def test_9W1_heliostat(self) -> None:
        """
        Draws 9W1 and surrounding heliostats
        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        heliostat_name = '9W1'
        up_heliostats = ['9W2', '9W3', '9E1', '9E2', '8W2', '8W1', '8E1', '8E2', '10W2', '10W1', '10E1', '10E2']
        test_heliostat = [heliostat_name]

        # View setup
        title = 'Heliostat ' + heliostat_name + ', and Surrounding Outlines'
        caption = ' Sandia NSTTF heliostats ' + heliostat_name + '.'
        comments = []

        # # Define test heliostat orientation
        # test_azimuth = np.deg2rad(45)
        # test_el = np.deg2rad(0)
        test_azimuth = np.deg2rad(177)
        test_el = np.deg2rad(43)

        # # # Define upward-facing heliostat orientation.
        UP = Vxyz([0, 0, 1])

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        for h_name in up_heliostats:
            solar_field.lookup_heliostat(h_name).set_orientation_from_pointing_vector(UP)
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        intersection_points = Intersection.plane_lines_intersection(
            (facet_origin, facet_normal), (tower.target_loc, Uxyz([0, 1, 0]))
        )

        # Style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        solar_field_style = rcsf.heliostat_blanks()
        solar_field_style.add_special_names(up_heliostats, rch.facet_outlines(color='g'))
        solar_field_style.add_special_names(test_heliostat, surface_normal_facet_style)
        tower_control = rct.normal_tower()

        # Comment
        comments.append("A subset of heliostats selected, and towers.")
        comments.append("Blue heliostat is test heliostat 9W1 with vector aiming into target on reciever tower")
        comments.append("Green heliostats are face up.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                1  # 'f_index'
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        solar_field.draw(fig_record.view, solar_field_style)
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(intersection_points, rcps.RenderControlPointSeq(marker='+'))
        self.show_save_and_check_figure(fig_record)

        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                2  # 'f_index'
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        fig_record.view.draw_single_Pxyz(intersection_points, rcps.RenderControlPointSeq(marker='+'))
        tower.draw(fig_record.view, tower_control)

        # Output.
        self.show_save_and_check_figure(fig_record)

        return

    def test_azimuth_binary_search(self) -> None:
        """
        Draws azimuth binary search algorithm with the target.
        """
        # Initialize test.
        self.start_test()

        f_index = '1'
        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        # View setup
        title = 'Heliostat ' + heliostat_name + ', Azimuth Binary Search'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'using facet 1.'
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177.5)
        test_el = np.deg2rad(43.2)

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc_1 = tower.target_loc
        target_plane_normal_1 = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        is_intersect, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
        )

        middle_azimuth, offset_x_points = self.azimuth_binary_search(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )

        z = [(offset_z + target_loc_1.z) for x in enumerate(offset_x_points)]
        y = [target_loc_1.y for x in enumerate(offset_x_points)]

        points = Pxyz((offset_x_points, y, z))

        # Style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        tower_control = rct.normal_tower()

        # Comment
        comments.append(
            "The tower with a target in red and intersection points in blue from the surface normal of the facet."
        )
        comments.append("Draws binary search algorithm for azimuth.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                2
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xyz')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xy
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                3
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xy')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                4
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xz')
        self.show_save_and_check_figure(fig_record)

        return

    def test_el_binary_search(self) -> None:
        """
        Draws elevation binary search algorithm with the target.
        """
        # Initialize test.
        self.start_test()

        f_index = '1'
        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        # View setup
        title = 'Heliostat ' + heliostat_name + ', Elevation Binary Search'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'using facet 1.'
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177.5)
        test_el = np.deg2rad(43.2)

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc_1 = tower.target_loc
        target_plane_normal_1 = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        is_intersec, offset_x, offset_z, intersection = self.projected_facet_normal_intersection_point_offset(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
        )

        middle_el, offset_z_points = self.elevation_binary_search(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )
        offset_z_points = np.array(offset_z_points) + target_loc_1.z

        x = [offset_x for x in enumerate(offset_z_points)]
        y = [target_loc_1.y for y in enumerate(offset_z_points)]

        points = Pxyz((x, y, offset_z_points))

        # Style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        tower_control = rct.normal_tower()

        # Comment
        comments.append(
            "The tower with a target in red and intersection points in blue from the surface normal of the facet."
        )
        comments.append("Draws binary search algorithm for elevation.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                5
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xyz')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xy
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                6
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xy')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for yz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                7
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_yz')
        self.show_save_and_check_figure(fig_record)

        return

    def test_single_facet(self) -> None:
        """
        Draws binary search algorithm with the target using both elevation and azimuth.
        """
        # Initialize test.
        self.start_test()

        f_index = '13'
        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        # View setup
        title = 'Heliostat ' + heliostat_name + ', Single Binary Search'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'using facet 1.'
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177.5)
        test_el = np.deg2rad(43.2)

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc_1 = tower.target_loc
        target_plane_normal_1 = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        # style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        solar_field_style = rcsf.heliostat_blanks()
        solar_field_style.add_special_names(test_heliostat, surface_normal_facet_style)
        tower_control = rct.normal_tower()

        az, el, intersection = self.find_single_facet_azimuth_el_value(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )

        # Comment
        comments.append(
            "The tower with a target in red and intersection point in blue from the surface normal of the facet."
        )

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                8
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        solar_field.draw(fig_record.view, solar_field_style)
        fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xyz')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xy
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                9
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        solar_field.draw(fig_record.view, solar_field_style)
        fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xy')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                10
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        solar_field.draw(fig_record.view, solar_field_style)
        fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xz')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for yz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                11
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        solar_field.draw(fig_record.view, solar_field_style)
        fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_yz')
        self.show_save_and_check_figure(fig_record)

        return

    def test_9W1_with_canting(self) -> None:
        """
        Draws binary search algorithm with the target using both elevation and azimuth.
        """
        # Initialize test.
        self.start_test()

        f_index = '13'
        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        # View setup
        title = 'Heliostat ' + heliostat_name + ', with Spherical Canting'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'using facet 13.'
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177.5)
        test_el = np.deg2rad(43.2)

        # define canting
        focal_length = 68.62
        curved_equation = FunctionXYContinuous(f"(x**2 + y**2)/(4*{focal_length})")

        mirror_template = MirrorParametricRectangular(curved_equation, (1.2, 1.2))
        h_9W1, h5W1_location = HeliostatAzEl.from_csv_files(
            "9W1",
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            mirror_template,
        )

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc_1 = tower.target_loc
        target_plane_normal_1 = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            solar_field.lookup_heliostat(h_name).set_canting_from_equation(curved_equation)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        # intersection_points = Intersection.plane_lines_intersection(
        #         (facet_origin, facet_normal), (tower.target_loc, Uxyz([0, 1, 0]))
        #     )

        # style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        solar_field_style = rcsf.heliostat_blanks()
        solar_field_style.add_special_names(test_heliostat, surface_normal_facet_style)
        tower_control = rct.normal_tower()

        az, el, intersection = self.find_single_facet_azimuth_el_value(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )

        # Comment
        comments.append(
            "The tower with a target in red and intersection point in blue from the surface normal of the facet."
        )

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                8
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        solar_field.draw(fig_record.view, solar_field_style)
        fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xyz')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xy
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                9
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        solar_field.draw(fig_record.view, solar_field_style)
        fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xy')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                10
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xz')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for yz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                11
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        solar_field.draw(fig_record.view, solar_field_style)
        fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_yz')
        self.show_save_and_check_figure(fig_record)

        return

    def test_all_facets(self) -> None:
        """
        Draws binary search algorithm with the target using both elevation and azimuth.
        """
        # Initialize test.
        self.start_test()

        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177.5)
        test_el = np.deg2rad(43.2)

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc_1 = tower.target_loc
        target_plane_normal_1 = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        for f_index in range(solar_field.lookup_heliostat(heliostat_name).facet_ensemble.num_facets):
            # View setup
            f_index = str(f_index + 1)
            title = 'Heliostat ' + heliostat_name + ', Facet ' + '{0:02d}'.format(int(f_index))
            caption = (
                'A single Sandia NSTTF heliostat ' + heliostat_name + ', with facet ' + '{0:02d}'.format(int(f_index))
            )
            comments = []

            # style setup
            surface_normal_facet_style = rch.facet_outlines_normals(color='b')
            surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
            solar_field_style = rcsf.heliostat_blanks()
            solar_field_style.add_special_names(test_heliostat, surface_normal_facet_style)
            tower_control = rct.normal_tower()

            az, el, intersection = self.find_single_facet_azimuth_el_value(
                target_loc=target_loc_1,
                target_plane_normal=target_plane_normal_1,
                heliostat_name=heliostat_name,
                f_index=f_index,
                az=test_azimuth,
                el=test_el,
                tolerance=test_tolerance,
            )

            # Comment
            comments.append(
                "The tower with a target in red and intersection point in blue from the surface normal of the facet: "
            )

            # Draw
            fig_record = fm.setup_figure_for_3d_data(
                self.figure_control,
                self.axis_control_m,
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(
                    12
                ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                title=title,
                caption=caption,
                comments=comments,
                code_tag=self.code_tag,
            )
            tower.draw(fig_record.view, tower_control)
            solar_field.draw(fig_record.view, solar_field_style)
            fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xyz')
            self.show_save_and_check_figure(fig_record)

            # Draw and produce output for xy
            fig_record = fm.setup_figure_for_3d_data(
                self.figure_control,
                self.axis_control_m,
                vs.view_spec_xy(),
                number_in_name=False,
                input_prefix=self.figure_prefix(
                    13
                ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                title=title,
                caption=caption,
                comments=comments,
                code_tag=self.code_tag,
            )
            tower.draw(fig_record.view, tower_control)
            solar_field.draw(fig_record.view, solar_field_style)
            fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xy')
            self.show_save_and_check_figure(fig_record)

            # Draw and produce output for xz
            fig_record = fm.setup_figure_for_3d_data(
                self.figure_control,
                self.axis_control_m,
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(
                    14
                ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                title=title,
                caption=caption,
                comments=comments,
                code_tag=self.code_tag,
            )
            tower.draw(fig_record.view, tower_control)
            solar_field.draw(fig_record.view, solar_field_style)
            fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xz')
            self.show_save_and_check_figure(fig_record)

            # Draw and produce output for yz
            fig_record = fm.setup_figure_for_3d_data(
                self.figure_control,
                self.axis_control_m,
                vs.view_spec_yz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(
                    15
                ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
                title=title,
                caption=caption,
                comments=comments,
                code_tag=self.code_tag,
            )
            tower.draw(fig_record.view, tower_control)
            solar_field.draw(fig_record.view, solar_field_style)
            fig_record.view.draw_single_Pxyz(intersection, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_yz')
            self.show_save_and_check_figure(fig_record)

        return

    def test_when_initial_position_not_on_target(self) -> None:
        """
        Computes facet values when given initial position not on target plane.

        """

        # Initialize test.
        self.start_test()

        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        title = 'Heliostat ' + heliostat_name + 'when surface normals parallel with target plane.'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(0)
        test_el = np.deg2rad(90)

        # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        test_target_loc = tower.target_loc
        test_target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        # Comment
        comments.append(
            "The tower with a target in red and intersection point in blue from the surface normal of the facet: "
        )

        self.find_all_azimuth_el_test_values(
            test_target_loc, test_target_plane_normal, heliostat_name, test_azimuth, test_el, test_tolerance
        )

        return

    ## Canting tests ###

    def test_canted_x_angle_binary_search(self) -> None:
        """
        Draws canted x binary search algorithm with the target.
        """
        # Initialize test.
        self.start_test()

        f_index = '1'
        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        # View setup
        title = 'Heliostat ' + heliostat_name + ', Canted X Binary Search'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'using facet 1.'
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177)
        test_el = np.deg2rad(50)

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        test_target_loc = tower.target_loc
        test_target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        test_canted_x_angle = 2
        test_canted_y_angle = 4
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        is_intersect, offset_x, offset_z, intersection = test_object.projected_facet_normal_intersection_point_offset(
            target_loc=test_target_loc,
            target_plane_normal=test_target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
        )

        middle_x_angle, offset_z_points = self.canted_x_binary_search(
            time=time,
            target_loc=test_target_loc,
            target_plane_normal=test_target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        offset_z_points = np.array(offset_z_points) + target_loc_1.z

        x = [offset_x for x in enumerate(offset_z_points)]
        y = [target_loc_1.y for y in enumerate(offset_z_points)]

        points = Pxyz((x, y, offset_z_points))

        # Style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        tower_control = rct.normal_tower()

        # Comment
        comments.append(
            "The tower with a target in red and intersection points in blue from the reflected ray trace of the facet."
        )
        comments.append("Draws binary search algorithm for canted x angle.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                2
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xyz')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xy
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                3
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xy')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for yz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                4
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_yz')
        self.show_save_and_check_figure(fig_record)

        return

    def test_canted_y_angle_binary_search(self) -> None:
        """
        Draws canted y binary search algorithm with the target.
        """
        # Initialize test.
        self.start_test()

        f_index = '1'
        heliostat_name = "9W1"
        test_heliostat = [heliostat_name]

        # View setup
        title = 'Heliostat ' + heliostat_name + ', Canted Y Binary Search'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'using facet 1.'
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177)
        test_el = np.deg2rad(50)

        # # Configuration setup
        solar_field = self.solar_field
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        test_target_loc = tower.target_loc
        test_target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        test_canted_x_angle = 2
        test_canted_y_angle = 4
        for h_name in test_heliostat:
            config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
            solar_field.lookup_heliostat(h_name).set_orientation(config)
            facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
                Resolution.center()
            )

        is_intersect, offset_x, offset_z, intersection = test_object.projected_facet_normal_intersection_point_offset(
            target_loc=test_target_loc,
            target_plane_normal=test_target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
        )

        middle_y_angle, offset_x_points = self.canted_y_binary_search(
            time=time,
            target_loc=test_target_loc,
            target_plane_normal=test_target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        z = [(offset_z + target_loc_1.z) for x in enumerate(offset_x_points)]
        y = [target_loc_1.y for x in enumerate(offset_x_points)]

        points = Pxyz((offset_x_points, y, z))

        # Style setup
        surface_normal_facet_style = rch.facet_outlines_normals(color='b')
        surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
        tower_control = rct.normal_tower()

        # Comment
        comments.append(
            "The tower with a target in red and intersection points in blue from the reflected ray trace of the facet."
        )
        comments.append("Draws binary search algorithm for canted y angle.")

        # Draw
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                2
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xyz')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xy
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                3
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xy')
        self.show_save_and_check_figure(fig_record)

        # Draw and produce output for xz
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            self.axis_control_m,
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(
                4
            ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            title=title,
            caption=caption,
            comments=comments,
            code_tag=self.code_tag,
        )
        tower.draw(fig_record.view, tower_control)
        fig_record.view.draw_single_Pxyz(points, rcps.RenderControlPointSeq(marker='+'), 'aimpoint_xz')
        self.show_save_and_check_figure(fig_record)

        return

    def test_find_single_canting_angles(self) -> None:
        """
        Draws ray trace intersection for single facet with the target.

        """
        # Initialize test.
        self.start_test()

        # View Setup
        heliostat_name = "9W1"
        f_index = "13"
        title = 'Heliostat ' + heliostat_name + 'when ray trace intersects with target.'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []

        # Initial positions
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        az = np.deg2rad(178.34515)
        el = np.deg2rad(43.03036)
        test_tolerance = 0.001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc

        # Determine canting angles for single facet
        canted_x_angle, canted_y_angle, intersection_point, offset_z, offset_x = self.find_single_facet_canting_angles(
            time=time,
            sun_ray=sun_ray,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=az,
            el=el,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        # Configuration setup
        solar_field = self.solar_field
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)
        facet = self.cant_single_facet(heliostat_name, f_index, az, el, canted_x_angle, canted_y_angle)
        facet_no_parent = facet.no_parent_copy()
        facet_location = facet.self_to_global_tranformation

        # Set canting
        position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
        canting = Rotation.from_euler('xyz', [canted_x_angle, canted_y_angle, 0], degrees=False)
        facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)
        facet_no_parent = facet.no_parent_copy()

        # Set solar field scene
        scene = Scene()
        scene.add_object(facet_no_parent)
        scene.set_position_in_space(facet_no_parent, facet_location)
        solar_field.set_full_field_tracking(AIMPOINT, TIME)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center())

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        # # Calculate Reflected Ray
        # sun = Vxyz(sp.sun_position(lln.NSTTF_ORIGIN, TIME))
        # surface_norm = rt.calc_reflected_ray(facet_normal[0], sun[0])
        # print(surface_norm)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )

        # DRAWING
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(4, 2), figsize=(10.8, 5.4)),
            rca.meters(),
            vs.view_spec_3d(),
            title=f"9W1 Ray Trace 3D through Target",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=120))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        figure_control.view.show(x_limits=[-50, 100], y_limits=[0, 80], z_limits=[-10, 80])
        self.show_save_and_check_figure(figure_control)

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            title=f"9W1 Ray Trace Intersection at Aimpoint XZ View",
        )
        # figure_control.equal = False
        figure_control.x_limits = (-test_tolerance, test_tolerance)
        figure_control.z_limits = (63.5508 - test_tolerance, 63.5508 + test_tolerance)
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        AIMPOINT.draw_points(figure_control.view, rcps.marker(color='r'))
        self.show_save_and_check_figure(figure_control)

    # def test_find_all_canting_angles(self) -> None:
    #     """
    #     Draws ray trace intersection for 25 facets with the target.

    #     """
    #     # Initialize test.
    #     self.start_test()

    #     # View Setup
    #     heliostat_name = "5W2"

    #     # Initial positions
    #     tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
    #     target_loc = tower.target_loc
    #     target_plane_normal = Uxyz([0, 1, 0])
    #     test_tolerance = 0.001
    #     tower_control = rct.normal_tower()
    #     test_canted_x_angle = 0
    #     test_canted_y_angle = 0

    #     # Define tracking time
    #     # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
    #     TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
    #     AIMPOINT = target_loc
    #     UP = Vxyz([0, 0, 1])

    #     # Configuration setup
    #     solar_field = self.solar_field
    #     test_heliostat = solar_field.lookup_heliostat(heliostat_name)

    #     self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

    #     az = test_heliostat._az
    #     el = test_heliostat._el

    #     # Determine canting angles for single facet
    #     test_config = hc.HeliostatConfiguration("az-el", az, el)
    #     solar_field = self.solar_field
    #     test_heliostat = solar_field.lookup_heliostat(heliostat_name)
    #     test_heliostat.set_orientation(test_config)

    #     # facet_index = test_heliostat.facet_ensemble.children(dict)
    #     all_facets = []
    #     canted_x_angles = []
    #     canted_y_angles = []
    #     offset_x_found = []
    #     offset_z_found = []
    #     trc_x_sur_norm = []
    #     trc_y_sur_norm = []
    #     trc_z_sur_norm = []
    #     trc_f_origin_x = []
    #     trc_f_origin_y = []
    #     trc_f_origin_z = []
    #     f_idx = []
    #     cantings = []

    #     x_min = -0.01
    #     x_max = 0.01
    #     exaggerated_x = [x_min, x_max]
    #     z_min = 63.55 - 0.01
    #     z_max = 63.55 + 0.01
    #     exaggerated_z = [z_max, z_min]

    #     # checks az and el values to see if facet surface normal for facet 13 intersects with given plane.
    #     f_index_temp = '13'

    #     is_intersect, offset_x, offset_z, intersection_point = self.projected_facet_normal_intersection_point_offset(
    #         target_loc, target_plane_normal, heliostat_name, f_index_temp, az, el
    #     )
    #     az, el = self.doesnt_hit_plane_with_original_values(is_intersect, az, el)

    #     # iterates through every facet to find the azmimuth and elevation values that correspond with a surface normal that intersects with the target.
    #     for f_index in range(test_heliostat.facet_ensemble.num_facets):
    #         # for f_index in range(12, 13):
    #         f_index = str(f_index + 1)
    #         print("\nIn TestMotionBasedCanting.find_all_canting_angle_values:    facet=", f_index)
    #         facet_canted_x_angle, facet_canted_y_angle, __, offset_z, offset_x = self.find_single_facet_canting_angles(
    #             target_loc,
    #             target_plane_normal,
    #             heliostat_name,
    #             f_index,
    #             az,
    #             el,
    #             test_canted_x_angle,
    #             test_canted_y_angle,
    #             test_tolerance,
    #         )
    #         canting_angles_found = {
    #             'facet_number': f_index,
    #             'canting data': {'canted x angle': facet_canted_x_angle, 'canted y angle': facet_canted_y_angle},
    #         }
    #         f_idx.append(f_index)
    #         canted_x_angles.append(facet_canted_x_angle)
    #         canted_y_angles.append(facet_canted_y_angle)
    #         offset_x_found.append(offset_x)
    #         offset_z_found.append(offset_z)
    #         all_facets.append(canting_angles_found)

    #         # Configuration setup
    #         facet = self.cant_single_facet(heliostat_name, f_index, az, el, facet_canted_x_angle, facet_canted_y_angle)
    #         facet_no_parent = facet.no_parent_copy()
    #         facet_location = facet.self_to_global_tranformation

    #         # Set canting
    #         position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
    #         canting = Rotation.from_euler('xyz', [facet_canted_x_angle, facet_canted_y_angle, 0], degrees=False)
    #         facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)
    #         facet_no_parent = facet.no_parent_copy()

    #         cantings.append(canting)

    #         # # Set solar field scene
    #         # scene = Scene()
    #         # scene.add_object(facet_no_parent)
    #         # scene.set_position_in_space(facet_no_parent, facet_location)
    #         # solar_field.set_full_field_tracking(AIMPOINT, TIME)

    #         # # Add ray trace to field
    #         # sun_1ray = LightSourceSun()
    #         # sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
    #         # scene.add_light_source(sun_1ray)
    #         # trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

    #         # # Add intersection point
    #         # place_to_intersect = (AIMPOINT, target_plane_normal)
    #         # facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
    #         # is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
    #         # intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

    #         # # render control
    #         # facet_ensemble_control = rcfe.facet_outlines_thin()
    #         # heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
    #         # heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

    #         # # DRAWING
    #         # figure_control = fm.setup_figure_for_3d_data(
    #         #     rcfg.RenderControlFigure(tile_array=(1, 1)),
    #         #     rca.meters(),
    #         #     vs.view_spec_3d(),
    #         #     title='Heliostat ' + heliostat_name + ', Facet ' + f_index,
    #         #     caption='A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index,
    #         #     comments="3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
    #         # )
    #         # test_heliostat.draw(figure_control.view, heliostat_style)
    #         # tower.draw(figure_control.view, tower_control)
    #         # trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=80, current_len=250))
    #         # figure_control.view.draw_single_Pxyz(AIMPOINT)
    #         # figure_control.view.show(x_limits=[-50, 100], y_limits=[0, 170], z_limits=[-10, 70])
    #         # self.show_save_and_check_figure(figure_control)

    #         # figure_control = fm.setup_figure_for_3d_data(
    #         #     rcfg.RenderControlFigure(),
    #         #     rca.meters(),
    #         #     vs.view_spec_xz(),
    #         #     title='Heliostat ' + heliostat_name + ', Facet ' + f_index,
    #         #     caption='A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index,
    #         #     comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
    #         # )
    #         # figure_control.x_limits = exaggerated_x
    #         # figure_control.z_limits = exaggerated_z
    #         # intersection_point.draw(figure_control.view)
    #         # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
    #         # self.show_save_and_check_figure(figure_control)

    #     test_heliostat.set_facet_canting(cantings)

    #     for facet in test_heliostat.facet_ensemble.facets:
    #         facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
    #         print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
    #         trc_x_sur_norm.append(facet_normal.x)
    #         trc_y_sur_norm.append(facet_normal.y)
    #         trc_z_sur_norm.append(facet_normal.z)
    #         trc_f_origin_x.append(facet_origin.x)
    #         trc_f_origin_y.append(facet_origin.y)
    #         trc_f_origin_z.append(facet_origin.z)

    #     # heliostat = test_heliostat.no_parent_copy()
    #     # heliostat_loc = test_heliostat._self_to_parent_transform

    #     # # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
    #     # scene = Scene()
    #     # scene.add_object(heliostat)
    #     # scene.set_position_in_space(heliostat, heliostat_loc)

    #     # # Add ray trace to field
    #     # sun_1ray = LightSourceSun()
    #     # sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
    #     # scene.add_light_source(sun_1ray)
    #     # trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

    #     # Add intersection point
    #     # place_to_intersect = (AIMPOINT, target_plane_normal)

    #     # facet_ensemble_control = rcfe.facet_outlines_thin()
    #     # heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)

    #     # figure_control = fm.setup_figure_for_3d_data(
    #     #     rcfg.RenderControlFigure(tile_array=(1, 1)),
    #     #     rca.meters(),
    #     #     vs.view_spec_3d(),
    #     #     title='Off-Axis Canted Heliostat ' + heliostat_name + 'with Ray-Trace',
    #     #     caption='A single Sandia NSTTF heliostat ',
    #     #     comments="3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
    #     # )
    #     # test_heliostat.draw(figure_control.view, heliostat_style)
    #     # tower.draw(figure_control.view, tower_control)
    #     # trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=250))
    #     # figure_control.view.draw_single_Pxyz(AIMPOINT)
    #     # figure_control.view.show(x_limits=[-50, 100], y_limits=[0, 170], z_limits=[-10, 80])
    #     # self.show_save_and_check_figure(figure_control)
    #     # figure_control.view.show()

    #     az = np.deg2rad(180)
    #     el = np.deg2rad(90)
    #     test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
    #     test_heliostat.set_orientation(test_config)

    #     f_origin_x = []
    #     f_origin_y = []
    #     f_origin_z = []
    #     x_sur_norm = []
    #     y_sur_norm = []
    #     z_sur_norm = []

    #     # face-up
    #     for facet in test_heliostat.facet_ensemble.facets:
    #         _, sur_norm = facet.survey_of_points(Resolution.center())
    #         # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
    #         x_sur_norm.append(sur_norm.x)
    #         y_sur_norm.append(sur_norm.y)
    #         z_sur_norm.append(sur_norm.z)

    #     for facet in test_heliostat.facet_ensemble.no_parent_copy().facets:
    #         facet_origin, _ = facet.survey_of_points(Resolution.center())
    #         f_origin_x.append(facet_origin.x)
    #         f_origin_y.append(facet_origin.y)
    #         f_origin_z.append(facet_origin.z)

    #     # Prints all facet canting angles
    #     for facet in all_facets:
    #         print(facet)  # TODO mhh

    #     # saves canting angles to CSV
    #     self.save_to_csv(
    #         heliostat_name,
    #         f_index,
    #         f_origin_x,
    #         f_origin_y,
    #         f_origin_z,
    #         canted_x_angles,
    #         canted_y_angles,
    #         x_sur_norm,
    #         y_sur_norm,
    #         z_sur_norm,
    #         az,
    #         el,
    #         trc_f_origin_x,
    #         trc_f_origin_y,
    #         trc_f_origin_z,
    #         trc_x_sur_norm,
    #         trc_y_sur_norm,
    #         trc_z_sur_norm,
    #         offset_z,
    #         offset_x,
    #     )

    def test_off_axis_canting_angles(self) -> None:
        canting_details = 'off'
        heliostat_spec_list: list[str]
        # heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        heliostat_spec_list = self.solar_field.heliostat_name_list()
        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            folder_name = os.path.join(self.actual_output_dir, helio)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            actual_output_dir = os.path.join(self.actual_output_dir, helio)
            expected_output_dir = os.path.join(self.expected_output_dir, helio)
            # self.find_all_off_axis_canting_angles(heliostat_name, actual_output_dir)
            # self.create_off_axis_canted_heliostat(heliostat_name, actual_output_dir, expected_output_dir)
            self.example_canting_calculated(heliostat_name, canting_details, actual_output_dir, expected_output_dir)
            self.example_canting_bar_charts_calculated(
                heliostat_name, canting_details, actual_output_dir, expected_output_dir
            )

    def test_off_axis_canting_angles_vector_g3p3(self) -> None:
        canting_details = 'Off'
        heliostat_spec_list: list[str]
        # heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        heliostat_spec_list = self.solar_field.heliostat_name_list()
        target = 'G3P3'
        parent_folder_name = os.path.join(self.actual_output_dir, 'OffAxis_' + target + '_(-40.00,8.50,45.48)')
        if not os.path.exists(parent_folder_name):
            os.mkdir(parent_folder_name)
        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            folder_name = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            expected_output_dir = os.path.join(self.expected_output_dir, parent_folder_name, helio)
            self.test_off_axis_code(heliostat_name, target, actual_output_dir, expected_output_dir)
            self.example_canting_calculated(
                heliostat_name, target, canting_details, actual_output_dir, expected_output_dir
            )
            self.example_canting_bar_charts_calculated(
                heliostat_name, target, canting_details, actual_output_dir, expected_output_dir
            )
            # self.test_off_axis_code_individual_facets(heliostat_name, target, actual_output_dir, expected_output_dir)

    def test_off_axis_canting_angles_vector_towertop_time(self) -> None:
        canting_details = 'tod'
        heliostat_spec_list: list[str]
        heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        target = 'TowerTop'
        # Initial TIME tuple
        TIME = (2024, 3, 21, 7, 35, 5, -6)
        # Convert the TIME tuple to a datetime object
        time_obj = datetime(*TIME[:6])  # Exclude the last element (-6)
        # Number of iterations
        iterations = 17  # Change this to however many times you want to add the time
        # Loop to add 38.5 minutes
        for _ in range(iterations):
            # Add 38.5 minutes
            time_obj += timedelta(minutes=38.5)
            # Convert back to a tuple
            new_time = (
                time_obj.year,
                time_obj.month,
                time_obj.day,
                time_obj.hour,
                time_obj.minute,
                time_obj.second,
                TIME[6],
            )

            hour = new_time[3]
            minute = new_time[4]
            # Format the time as 'HH_MM'
            time = f'{hour:02}_{minute:02}'

            parent_folder_name = os.path.join(
                self.actual_output_dir, 'TimeOfDay_Canting_' + target + '_(0.00,6.25,63.55)_Time_' + time
            )
            if not os.path.exists(parent_folder_name):
                os.mkdir(parent_folder_name)
            for heliostat_name in heliostat_spec_list:
                helio = re.sub(
                    r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name
                )
                folder_name = os.path.join(self.actual_output_dir, parent_folder_name, helio)
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)
                actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name, helio)
                expected_output_dir = os.path.join(self.expected_output_dir, parent_folder_name, helio)
                self.test_off_axis_code_time_of_day(
                    heliostat_name, target, actual_output_dir, expected_output_dir, new_time, time
                )
                self.example_canting_calculated_time(
                    heliostat_name, target, canting_details, actual_output_dir, expected_output_dir, time
                )
                self.example_canting_bar_charts_calculated_time(
                    heliostat_name, target, canting_details, actual_output_dir, expected_output_dir, time
                )

    def test_off_axis_not_calculated_canting_angles_vector_towertop_time(self) -> None:
        canting_details = 'Off'
        heliostat_spec_list: list[str]
        heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        target = 'TowerTop'
        # Initial TIME tuple
        TIME = (2024, 3, 21, 7, 35, 5, -6)
        # Convert the TIME tuple to a datetime object
        time_obj = datetime(*TIME[:6])  # Exclude the last element (-6)
        # Number of iterations
        iterations = 17  # Change this to however many times you want to add the time
        # Loop to add 38.5 minutes
        for _ in range(iterations):
            # Add 38.5 minutes
            time_obj += timedelta(minutes=38.5)
            # Convert back to a tuple
            new_time = (
                time_obj.year,
                time_obj.month,
                time_obj.day,
                time_obj.hour,
                time_obj.minute,
                time_obj.second,
                TIME[6],
            )

            hour = new_time[3]
            minute = new_time[4]
            # Format the time as 'HH_MM'
            time = f'{hour:02}_{minute:02}'
            parent_folder_name = os.path.join(
                self.actual_output_dir, 'OffAxis_' + target + '_(0.00,6.25,63.55)_Time_' + time
            )
            canting_folder_name = os.path.join(self.actual_output_dir, 'OffAxis_' + target + '_(0.00,6.25,63.55)')
            if not os.path.exists(parent_folder_name):
                os.mkdir(parent_folder_name)
            for heliostat_name in heliostat_spec_list:
                helio = re.sub(
                    r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name
                )
                folder_name = os.path.join(self.actual_output_dir, parent_folder_name, helio)
                if not os.path.exists(folder_name):
                    os.mkdir(folder_name)
                canting_dir = os.path.join(self.actual_output_dir, canting_folder_name, helio)
                actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name, helio)
                expected_output_dir = os.path.join(self.expected_output_dir, parent_folder_name, helio)
                self.test_off_axis_code_time_of_day_not_calculated(
                    heliostat_name, target, canting_dir, actual_output_dir, expected_output_dir, new_time, time
                )
                self.example_canting_calculated_time(
                    heliostat_name, target, canting_details, canting_dir, actual_output_dir, expected_output_dir, time
                )
                self.example_canting_bar_charts_calculated_time(
                    heliostat_name, target, canting_details, canting_dir, actual_output_dir, expected_output_dir, time
                )

    def test_on_axis_canting_in_off_axis_orientation_vector_towertop(self) -> None:
        canting_details = 'On'
        heliostat_spec_list: list[str]
        heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        # heliostat_spec_list = self.solar_field.heliostat_name_list()
        target = 'TowerTop'
        on_axis_folder_name = os.path.join(self.actual_output_dir, 'OnAxis_' + target + '_(0.00,6.25,63.55)')
        parent_folder_name = os.path.join(
            self.actual_output_dir, 'OnAxisInOffAxisConfig_' + target + '_(0.00,6.25,63.55)'
        )
        if not os.path.exists(parent_folder_name):
            os.mkdir(parent_folder_name)
        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            folder_name = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            on_output_dir = os.path.join(self.actual_output_dir, on_axis_folder_name, helio)
            actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            expected_output_dir = os.path.join(self.expected_output_dir, parent_folder_name, helio)
            self.test_canting_angles_in_off_axis_config(
                heliostat_name, target, on_output_dir, actual_output_dir, expected_output_dir
            )

    def test_off_axis_canting_angles_vector_towertop(self) -> None:
        canting_details = 'Off'
        heliostat_spec_list: list[str]
        # heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        heliostat_spec_list = self.solar_field.heliostat_name_list()
        target = 'TowerTop'
        parent_folder_name = os.path.join(self.actual_output_dir, 'OffAxis_' + target + '_(0.00,6.25,63.55)')
        if not os.path.exists(parent_folder_name):
            os.mkdir(parent_folder_name)
        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            folder_name = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            expected_output_dir = os.path.join(self.expected_output_dir, parent_folder_name, helio)
            self.test_off_axis_code(heliostat_name, target, actual_output_dir, expected_output_dir)
            self.example_canting_calculated(
                heliostat_name, target, canting_details, actual_output_dir, expected_output_dir
            )
            self.example_canting_bar_charts_calculated(
                heliostat_name, target, canting_details, actual_output_dir, expected_output_dir
            )
            # self.test_off_axis_code_individual_facets(heliostat_name, target, actual_output_dir, expected_output_dir)

    def test_off_axis_canting_angles_vector_bcs(self) -> None:
        canting_details = 'Off'
        heliostat_spec_list: list[str]
        # heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        heliostat_spec_list = self.solar_field.heliostat_name_list()
        target = 'BCS'
        parent_folder_name = os.path.join(self.actual_output_dir, 'OffAxis_' + target + '_(0.00,8.80,28.90)')
        if not os.path.exists(parent_folder_name):
            os.mkdir(parent_folder_name)
        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            folder_name = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            expected_output_dir = os.path.join(self.expected_output_dir, parent_folder_name, helio)
            self.test_off_axis_code(heliostat_name, target, actual_output_dir, expected_output_dir)
            self.example_canting_calculated(
                heliostat_name, target, canting_details, actual_output_dir, expected_output_dir
            )
            self.example_canting_bar_charts_calculated(
                heliostat_name, target, canting_details, actual_output_dir, expected_output_dir
            )
            # self.test_off_axis_code_individual_facets(heliostat_name, target, actual_output_dir, expected_output_dir)

    def test_on_axis_canting_angles_vector_towertop(self) -> None:
        canting_details = 'On'
        heliostat_spec_list: list[str]
        # heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        heliostat_spec_list = self.solar_field.heliostat_name_list()
        target = 'TowerTop'
        parent_folder_name = os.path.join(self.actual_output_dir, 'OnAxis_' + target + '_(0.00,6.25,63.55)')
        if not os.path.exists(parent_folder_name):
            os.mkdir(parent_folder_name)
        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            folder_name = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            expected_output_dir = os.path.join(self.expected_output_dir, parent_folder_name, helio)
            # self.test_on_axis_code(heliostat_name, target, actual_output_dir, expected_output_dir)
            # self.example_canting_calculated(
            #     heliostat_name, target, canting_details, actual_output_dir, expected_output_dir
            # )
            self.example_canting_bar_charts_calculated(
                heliostat_name, target, canting_details, actual_output_dir, expected_output_dir
            )
            # self.test_on_axis_code_individual_facets(heliostat_name, target, actual_output_dir, expected_output_dir)

    def test_sofast_axis_canting_angles_vector_towertop(self) -> None:
        canting_details = 'Off'
        heliostat_spec_list: list[str]
        # heliostat_spec_list = ['5E9', '5W1', '5W9', '9E11', '9W1', '9W11', '14E6', '14W1', '14W6']
        heliostat_spec_list = self.solar_field.heliostat_name_list()
        target = 'TowerTop'
        off_axis_folder_name = os.path.join(self.actual_output_dir, 'OffAxis_' + target + '_(0.00,6.25,63.55)')
        parent_folder_name = os.path.join(self.actual_output_dir, 'SofastConfigurations')
        if not os.path.exists(parent_folder_name):
            os.mkdir(parent_folder_name)
        for heliostat_name in heliostat_spec_list:
            helio = re.sub(r'(\d+)([WE])(\d)(?!\d)', lambda m: f"{m.group(1)}{m.group(2)}0{m.group(3)}", heliostat_name)
            folder_name = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)
            off_axis_output_dir = os.path.join(self.actual_output_dir, off_axis_folder_name, helio)
            actual_output_dir = os.path.join(self.actual_output_dir, parent_folder_name, helio)
            expected_output_dir = os.path.join(self.expected_output_dir, parent_folder_name, helio)
            self.test_create_sofast_axis_canted_heliostat(
                heliostat_name, target, canting_details, off_axis_output_dir, actual_output_dir, expected_output_dir
            )
        self.save_master_csv()

    def find_all_off_axis_canting_angles(self, heliostat_name, actual_output_dir: str = None) -> None:
        """
        Draws ray trace intersection for 25 facets with the target.

        """
        # Initialize test.
        self.start_test()

        # Heliostat selection
        canting_details = 'off'
        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)

        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        azimuth = test_heliostat._az
        elevation = test_heliostat._el

        canted_x_angles, canted_y_angles, f_idx, offset_x, offset_z = self.find_all_canting_angle_values(
            time=TIME,
            aimpoint=AIMPOINT,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            azimuth=azimuth,
            elevation=elevation,
            canted_x_angle=test_canted_x_angle,
            canted_y_angle=test_canted_y_angle,
            tolerance=test_tolerance,
        )

        canting_rotations = []

        for facet in range(0, 25):
            # position = Pxyz.merge(test_heliostat._self_to_parent_transform.apply(Pxyz.origin()))
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
            print(f"rotation: {canting.as_euler('xyz')}")
        test_heliostat.set_facet_canting(canting_rotations)

        trc_f_origin_x = []
        trc_f_origin_y = []
        trc_f_origin_z = []
        trc_x_sur_norm = []
        trc_y_sur_norm = []
        trc_z_sur_norm = []

        # tracking
        for facet in test_heliostat.facet_ensemble.facets:
            trc_facet_origin, trc_sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            trc_f_origin_x.append(trc_facet_origin.x)
            trc_f_origin_y.append(trc_facet_origin.y)
            trc_f_origin_z.append(trc_facet_origin.z)
            trc_x_sur_norm.append(trc_sur_norm.x)
            trc_y_sur_norm.append(trc_sur_norm.y)
            trc_z_sur_norm.append(trc_sur_norm.z)

        az = np.deg2rad(180)
        el = np.deg2rad(90)
        test_config = hc.HeliostatConfiguration('az-el', az=az, el=el)
        test_heliostat.set_orientation(test_config)

        f_origin_x = []
        f_origin_y = []
        f_origin_z = []
        x_sur_norm = []
        y_sur_norm = []
        z_sur_norm = []

        # face-up
        for facet in test_heliostat.facet_ensemble.facets:
            _, sur_norm = facet.survey_of_points(Resolution.center())
            # print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            x_sur_norm.append(sur_norm.x)
            y_sur_norm.append(sur_norm.y)
            z_sur_norm.append(sur_norm.z)

        for facet in test_heliostat.facet_ensemble.no_parent_copy().facets:
            facet_origin, _ = facet.survey_of_points(Resolution.center())
            f_origin_x.append(facet_origin.x)
            f_origin_y.append(facet_origin.y)
            f_origin_z.append(facet_origin.z)

        self.save_to_csv(
            canting_details,
            heliostat_name,
            f_idx,
            f_origin_x,
            f_origin_y,
            f_origin_z,
            canted_x_angles,
            canted_y_angles,
            x_sur_norm,
            y_sur_norm,
            z_sur_norm,
            azimuth,
            elevation,
            trc_f_origin_x,
            trc_f_origin_y,
            trc_f_origin_z,
            trc_x_sur_norm,
            trc_y_sur_norm,
            trc_z_sur_norm,
            offset_z,
            offset_x,
            actual_output_dir,
        )

    def create_on_axis_canted_heliostat(
        self, heliostat_name: str, actual_output_dir: str = None, expected_output_dir: str = None
    ) -> None:

        # Initialize test.
        self.start_test()

        # View Setup

        comments = []

        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir
        # Initial positions
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        tower_control = rct.normal_tower()
        f_index = '13'
        test_heliostat = [heliostat_name]
        test_tolerance = 0.0001

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(177.5)
        test_el = np.deg2rad(43.2)

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # facet_index = test_heliostat.facet_ensemble.children(dict)
        opencsp_dir = os.path.join(
            orp.opencsp_code_dir(),
            'common',
            'lib',
            'test',
            'data',
            'input',
            'sandia_nsttf_test_definition',
            'NSTTF_Canting_Prescriptions',
        )

        canted_x_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_on_axis.csv'),
            'Face Up Canting Rotation about X',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_on_axis.csv'),
            'Face Up Canting Rotation about Y',
        )

        azim = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_on_axis.csv'), 'Tracking Az (rad)'
        )
        elev = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_on_axis.csv'), 'Tracking El (rad)'
        )

        surface_norm_x = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_on_axis.csv'), 'Tracking Surface Normal X'
        )
        surface_norm_y = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_on_axis.csv'), 'Tracking Surface Normal Y'
        )
        surface_norm_z = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_on_axis.csv'), 'Tracking Surface Normal Z'
        )

        canting_rotations: list[TransformXYZ] = []

        # Determine canting angles for single facet
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_config = hc.HeliostatConfiguration('az-el', az=float(azim[4]), el=float(elev[4]))
        test_heliostat.set_orientation(test_config)

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        heliostat = test_heliostat.no_parent_copy()
        heliostat_loc = test_heliostat._self_to_parent_transform

        # # Configuration setup
        # facet = self.cant_single_facet(heliostat_name, f_index, az, el, facet_canted_x_angle, facet_canted_y_angle)
        # facet_no_parent = facet.no_parent_copy()
        # facet_location = facet.self_to_global_tranformation

        # # Set canting
        # position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
        # canting = Rotation.from_euler('xyz', [facet_canted_x_angle, facet_canted_y_angle, 0], degrees=True)
        # facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)
        # facet_no_parent = facet.no_parent_copy()

        # solar_field.set_full_field_tracking(AIMPOINT, TIME)

        # h_pointing_vector = UP.rotate(test_heliostat._self_to_parent_transform.R)
        # test_heliostat.set_orientation_from_pointing_vector(h_pointing_vector)

        # dimensions for intersection drawing
        x_min = -0.01
        x_max = 0.01
        exaggerated_x = [x_min, x_max]
        z_min = 63.55 - 0.01
        z_max = 63.55 + 0.01
        exaggerated_z = [z_max, z_min]
        x_lim = [-20, 90]  ##TODO mhh figure out limits that are automated
        y_lim = [0, 80]
        z_lim = [-10, 80]
        count = int(0)

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_config = hc.HeliostatConfiguration('az-el', az=float(azim[4]), el=float(elev[4]))
        test_heliostat.set_orientation(test_config)
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80

        test_facet = test_heliostat.facet_ensemble.lookup_facet("13")
        __, trc_sur_norm = test_facet.survey_of_points(Resolution.center())
        sun_ray = -trc_sur_norm

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))

            count = int(count + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile_array=(1, 1)),
                rca.meters(),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                title='On-Axis Canted NSTTF Heliostat' + heliostat_name + ', Facet ' + facet.name + ' with Ray-Trace',
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ', Facet ' + facet.name,
                comments=comments.append(
                    "3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            AIMPOINT.draw_points(figure_control.view)
            # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count = int(count + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                title='On-Axis Canted NSTTF Heliostat '
                + heliostat_name
                + ', Facet '
                + facet.name
                + " Intersection at Aimpoint",
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            AIMPOINT.draw_points(figure_control.view, rcps.marker(color='r'))
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

        scene = Scene()
        scene.add_object(heliostat)
        scene.set_position_in_space(heliostat, heliostat_loc)
        sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(51),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XY Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(52),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XY Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(53),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="YZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(54),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(55),
            title='On-Axis Canted NSTTF Heliostat ' + heliostat_name + " Intersection at Aimpoint All Facets",
            caption='A single Sandia NSTTF heliostat ' + heliostat_name,
            comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        AIMPOINT.draw_points(figure_control.view, rcps.marker(color='r'))
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

    def test_canting_angles_in_off_axis_config(
        self,
        heliostat_name: str = None,
        target: str = 'target',
        test_output_dir: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:

        # Initialize test.
        self.start_test()

        # View Setup
        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # View Setup
        canting_details = 'On'
        # heliostat_name = "9W1"
        f_index = "13"
        title = "On-Axis Canted " + heliostat_name + " in Off-Axis Config"
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []
        count = int(0)

        if target == "G3P3":
            wire_frame = 'G3P3 Tower'
        elif target == 'BCS':
            wire_frame = 'BCS Tower'
        else:
            wire_frame = None

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target, wire_frame])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        target_origin = AIMPOINT
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
        )
        target_file_name = (
            target + ' ' + f'{float(target_x): .2f}' + ' ' + f'{float(target_y): .2f}' + ' ' + f'{float(target_z): .2f}'
        )

        opencsp_dir = test_output_dir

        canted_x_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about X (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about Y (rad)',
        )

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        azimuth = test_heliostat._az
        elevation = test_heliostat._el
        print(f"az and el = {azimuth}, {elevation}")
        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        canting_rotations: list[TransformXYZ] = []

        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
            print(canted_x_angles[facet])
        test_heliostat.set_facet_cantings(canting_rotations)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)

        trace = rt.trace_scene(scene, Resolution.center())

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        heliostat_origin, __ = test_heliostat.survey_of_points(Resolution.center())

        # array = sun_1ray.get_incident_rays(heliostat_origin)[0]
        # print(array.current_direction.x)
        # print('done')

        focal_length = Pxyz.distance(target_loc, heliostat_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80
        x_min = float(target_loc.x[0]) - 1.5
        x_max = float(target_loc.x[0]) + 1.5
        exaggerated_x = [x_min, x_max]
        z_min = f'{(float(target_loc.z[0]) - 1.5): .2f}'
        z_max = f'{(float(target_loc.z[0]) + 1.5): .2f}'
        exaggerated_z = [float(z_max), float(z_min)]

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )

        # DRAWING
        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
            rca.meters(grid=False),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace 3D ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Birds View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Looking North View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Side View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Intersection',
            title=title + ' Intersection',
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count_ = int(15)

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            # Add ray trace to field
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
            count_ = int(count_ + 1)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
                rca.meters(grid=False),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Ray Trace 3D ' + target_file_name,
                title=title + ', Facet ' + facet.name,
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            target_origin.draw_points(figure_control.view)
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count_ = int(count_ + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Intersection',
                title=title + ', Facet ' + facet.name,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

    def test_canting_angles_in_off_axis_config_code(
        self,
        heliostat_name: str = None,
        target: str = 'target',
        canting_details: str = None,
        test_output_dir: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:

        # Initialize test.
        self.start_test()

        # View Setup
        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # View Setup
        canting_details = canting_details
        # heliostat_name = "9W1"
        f_index = "13"
        title = canting_details + "-Axis Canted " + heliostat_name + " in Off-Axis configuration"
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index
        comments = []
        count = int(0)

        if target == "G3P3":
            wire_frame = 'G3P3 Tower'
        elif target == 'BCS':
            wire_frame = 'BCS Tower'
        else:
            wire_frame = None

        # Initial positions
        print(f"In test_find_all_canting_angles: heliostat = {heliostat_name}")
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target, wire_frame])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        target_plane_normal = Uxyz([0, 1, 0])
        test_tolerance = 0.0001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        azimuth = test_heliostat._az
        elevation = test_heliostat._el
        print(f"az and el = {azimuth}, {elevation}")
        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        opencsp_dir = test_output_dir

        canted_x_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about X (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about Y (rad)',
        )

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters
        z = facet_origin.z

        canting_rotations: list[TransformXYZ] = []

        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center())

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        heliostat_origin, __ = test_heliostat.survey_of_points(Resolution.center())

        # array = sun_1ray.get_incident_rays(heliostat_origin)[0]
        # print(array.current_direction.x)
        # print('done')

        focal_length = Pxyz.distance(target_loc, heliostat_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80
        x_min = float(target_loc.x[0]) - 1.5
        x_max = float(target_loc.x[0]) + 1.5
        exaggerated_x = [x_min, x_max]
        z_min = f'{(float(target_loc.z[0]) - 1.5): .2f}'
        z_max = f'{(float(target_loc.z[0]) + 1.5): .2f}'
        exaggerated_z = [float(z_max), float(z_min)]

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )
        target_origin = AIMPOINT
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
        )
        target_file_name = (
            target + ' ' + f'{float(target_x): .2f}' + ' ' + f'{float(target_y): .2f}' + ' ' + f'{float(target_z): .2f}'
        )

        # DRAWING
        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
            rca.meters(grid=False),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace 3D ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        # trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Birds View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Looking North View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Ray Trace Side View ' + target_file_name,
            title=title + ", " + target_plot_name,
        )
        test_heliostat.draw(figure_control.view)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view)
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count = count + 1
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(count),
            name=title + ' Intersection at Aimpoint ' + target_file_name,
            title=title + ', ' + target_plot_name,
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
        # figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, actual_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        count_ = int(15)

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            # Add ray trace to field
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
            count_ = int(count_ + 1)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
                rca.meters(grid=False),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Ray Trace 3D ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            target_origin.draw_points(figure_control.view)
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count_ = int(count_ + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count_),
                name=title + ' Facet ' + facet.name + ' Intersection at Aimpoint ' + target_file_name,
                title=title + ', Facet ' + facet.name + ", " + target_plot_name,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
            # figure_control.view.show()
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

    def example_canting_calculated(
        self,
        heliostat_name: str = None,
        target: str = None,
        canting_details: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target == 'target'
            target_loc = Pxyz([[0], [0], [0]])

        # Define scenario.
        nsttf_pivot_height = 4.02  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_pivot_offset = 0.1778  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        focal_length_max = 210  # meters

        opencsp_dir = actual_output_dir

        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
        )
        target_file_name = (
            target + ' ' + f'{float(target_x): .2f}' + ' ' + f'{float(target_y): .2f}' + ' ' + f'{float(target_z): .2f}'
        )

        canted_x_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about X (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about Y (rad)',
        )

        # heliostat z position and focal_length
        # z = 5.63  # meters
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters
        z = facet_origin.z

        name = heliostat_name
        title = canting_details + '-Axis Canted ' + heliostat_name
        caption = heliostat_name + ' modeled as a symmetric paraboloid with focal length f=' + str(focal_length) + 'm.'
        # Solar field.
        short_name_sf = 'Mini NSTTF'
        name_sf = 'Mini NSTTF with ' + canting_details + '-Axis Canted' + heliostat_name
        caption_sf = (
            'NSTTF heliostat.' + '  ' + caption + '  ' + 'Facet surfaces and canting have the same focal length.'
        )
        comments = []

        # Construct heliostat objects and solar field object.
        def fn(x, y):
            return (x**2) / (4 * focal_length) + (y**2) / (4 * focal_length)

        h1_mirror = MirrorParametricRectangular(fn, (nsttf_facet_width, nsttf_facet_height))
        h1, location1 = HeliostatAzEl.from_csv_files(
            heliostat_name,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            h1_mirror,
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)

        h1.set_facet_cantings(canting_rotations)

        heliostats = [h1]
        heliostat_locations = [location1]

        UP = Pxyz([0, 0, 1])
        h1.set_orientation_from_az_el(0, np.pi / 2)
        # h_14W01.set_orientation_from_az_el(0, np.pi / 2)

        sf = SolarField(heliostats, lln.NSTTF_ORIGIN, name_sf)
        sf.set_heliostat_positions(heliostat_locations)

        comments_long = comments.copy()  # We'll add a different comment for the plots with long normals.
        comments_very_long = comments.copy()  # We'll add a different comment for the plots with very long normals.
        comments_exaggerated_z = (
            comments.copy()
        )  # We'll add a different comment for the plots with an exaggerated z axis.
        comments.append('Render mirror surfaces and normals, facet outlines, and heliostat centroid.')

        # Setup render control (long normals).
        mirror_control_long = rcm.RenderControlMirror(surface_normals=True, norm_len=15, norm_res=3)
        facet_control_long = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_long,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control_long = rcfe.RenderControlFacetEnsemble(facet_control_long, draw_outline=False)
        heliostat_control_long = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_long, draw_facet_ensemble=True
        )

        comments_long.append('Render mirror surfaces and long normals, facet outlines, and heliostat centroid.')

        # Draw and output 5W01 figure (long normals, xy view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(grid=False),
            vs.view_spec_xy(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(6),
            name=title + ' (long normals, 9 for each facet) ' + target_file_name,
            title=title + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_long)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Setup render control (long normals).
        mirror_control_long_1 = rcm.RenderControlMirror(surface_normals=True, norm_len=65, norm_res=1)
        facet_control_long_1 = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_long_1,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control_long_1 = rcfe.RenderControlFacetEnsemble(facet_control_long_1, draw_outline=False)
        heliostat_control_long_1 = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_long_1, draw_facet_ensemble=True
        )

        comments_long.append('Render mirror surfaces and long normals, facet outlines, and heliostat centroid.')

        # Draw and output 5W01 figure (long normals, xy view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(grid=False),
            vs.view_spec_xy(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(7),
            name=title + ' (long normals, 1 for each facet) ' + target_file_name,
            title=title + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_long_1)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Setup render control (very long normals).
        mirror_control_very_long = rcm.RenderControlMirror(
            surface_normals=True,
            norm_len=(2 * focal_length_max),  # Twice the focal length is the center of curvature.
            norm_res=2,
        )
        facet_control_very_long = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_very_long,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control_very_long = rcfe.RenderControlFacetEnsemble(facet_control_very_long)
        heliostat_control_very_long = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_very_long, draw_facet_ensemble=True
        )
        solar_field_control_very_long = rcsf.RenderControlSolarField(heliostat_styles=heliostat_control_very_long)
        comments_very_long.append(
            'Render mirror surfaces and very long normals, facet outlines, and heliostat centroid.'
        )

        # Draw and output solar_field figure (very long normals, yz view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(8),
            name=title + ' (very long normals) ' + target_file_name,
            title=title + ', ' + target_plot_name,
            caption=caption_sf,
            comments=comments,
        )
        sf.draw(fig_record.view, solar_field_control_very_long)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Setup render control (exaggerated z).
        z_exaggerated_margin = 0.35  # meters, plus or minus reference height.
        decimal_factor = 100.0
        # Different z limits for each heliostat, because they are at different elevations on the sloped field.
        z_min = np.floor(decimal_factor * ((z + nsttf_pivot_offset) - z_exaggerated_margin)) / decimal_factor
        z_max = np.ceil(decimal_factor * ((z + nsttf_pivot_offset) + z_exaggerated_margin)) / decimal_factor
        z_min_str = str(str(z_min).strip("[]"))
        z_max_str = str(str(z_max).strip("[]"))
        exaggerated_z_limits = [float(z_min_str), float(z_max_str)]

        mirror_control_exaggerated_z = rcm.RenderControlMirror(surface_normals=False)
        facet_control_exaggerated_z = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_exaggerated_z,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=True,
        )
        facet_ensemble_control_exaggerated_z = rcfe.RenderControlFacetEnsemble(
            default_style=facet_control_exaggerated_z, draw_outline=False
        )
        heliostat_control_exaggerated_z = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_exaggerated_z, draw_facet_ensemble=True
        )
        comments_exaggerated_z.append('Render heliostat with exaggerated z axis.')

        # Draw and output 5W01 figure (exaggerated z).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_3d(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(9),
            name=title + ' (exaggerated z) ' + target_file_name,
            title=title + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        fig_record.z_limits = exaggerated_z_limits
        h1.draw(fig_record.view, heliostat_control_exaggerated_z)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        mirror_control_exaggerated_z_normal = rcm.RenderControlMirror(surface_normals=True, norm_len=5, norm_res=1)
        facet_control_exaggerated_z_normal = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_exaggerated_z_normal,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=True,
        )
        facet_ensemble_control_exaggerated_z_normal = rcfe.RenderControlFacetEnsemble(
            default_style=facet_control_exaggerated_z_normal
        )
        heliostat_control_exaggerated_z_normal = rch.RenderControlHeliostat(
            draw_centroid=False,
            facet_ensemble_style=facet_ensemble_control_exaggerated_z_normal,
            draw_facet_ensemble=True,
        )

        # Draw and output 5W01 figure (exaggerated z).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_3d(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(10),
            name=title + ' (with normals) ' + target_file_name,
            title=title + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z_normal)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Draw and output 5W01 figure (exaggerated z, yz view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(11),
            name=title + ' (exaggerated z) ' + target_file_name,
            title=title + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        fig_record.equal = False  # Asserting equal axis scales contradicts exaggerated z limits in 2-d plots.
        fig_record.z_limits = (
            exaggerated_z_limits  # Limits are on z values, even though the plot is 2-d.  View3d.py handles this.
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Draw and output 5W01 figure (exaggerated z, yz view with normals).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(12),
            name=title + ' (with normals) ' + target_file_name,
            title=title + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z_normal)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

    def example_canting_bar_charts_calculated(
        self,
        heliostat_name: str = None,
        target: str = None,
        canting_details: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        # Define scenario.
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters

        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        title = canting_details + '-Axis Canted ' + heliostat_name
        titled = canting_details + '-Axis_Canted_' + heliostat_name
        caption = heliostat_name + ' modeled as a symmetric paraboloid with focal length f=' + str(focal_length) + 'm.'

        # Construct heliostat objects and solar field object.
        def fn(x, y):
            return (x**2) / (4 * focal_length) + (y**2) / (4 * focal_length)

        h1_mirror = MirrorParametricRectangular(fn, (nsttf_facet_width, nsttf_facet_height))
        h1, location1 = HeliostatAzEl.from_csv_files(
            heliostat_name,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            h1_mirror,
        )
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
        )
        target_file_name = (
            target
            + '_'
            + str(f'{float(target_x): .2f}').strip(' ')
            + '_'
            + str(f'{float(target_y): .2f}').strip(' ')
            + '_'
            + str(f'{float(target_z): .2f}').strip(' ')
        )

        opencsp_dir = actual_output_dir

        canted_x_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about X (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about Y (rad)',
        )

        anglesx = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Surface Normal X',
        )
        anglesy = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Surface Normal Y',
        )
        anglesz = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Surface Normal Z',
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, h1.facet_ensemble.num_facets):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)

        h1.set_facet_cantings(canting_rotations)

        ####plot canting angles bar chart

        canting_angles = {}
        # ?? SCAFFOLDING RCB -- THIS IS CRUFTY.  THIS SHOULD GO IN AN ANALYSIS PLACE, AND ONLY COMPUTE ONCE.  SEE ALSO SIMILAR CODE IN PLOT NORMALS FUNCTION.
        # Compute average canting angle x and y components.\
        angles_x = []
        for x_angles in anglesx:
            x_angles = float(x_angles)
            angles_x.append(x_angles)
        angles_y = []
        for y_angles in anglesy:
            y_angles = float(y_angles)
            angles_y.append(y_angles)
        angles_z = []
        for z_angles in anglesz:
            z_angles = float(z_angles)
            angles_z.append(z_angles)

        average_ax = sum(angles_x) / len(angles_x)
        average_ay = sum(angles_y) / len(angles_y)
        # Compute offsets.
        offset_anglesx = [(x - average_ax) for x in angles_x]
        offset_anglesy = [(y - average_ay) for y in angles_y]
        offset_anglesz = []
        for facet in range(h1.facet_ensemble.num_facets):
            canting_angles[facet] = [angles_x[facet], angles_y[facet], angles_z[facet]]

        for offset_ax, offset_ay in zip(offset_anglesx, offset_anglesy):
            offset_az = np.sqrt(1.0 - (offset_ax**2 + offset_ay**2))
            offset_anglesz.append(offset_az)
        offset_canting_angles = {}
        for key in range(h1.facet_ensemble.num_facets):
            axyz = canting_angles[key]
            ax2 = axyz[0]
            ay2 = axyz[1]
            az2 = axyz[2]
            offset_ax2 = ax2 - average_ax
            offset_ay2 = ay2 - average_ay
            offset_az2 = np.sqrt(1.0 - (offset_ax2**2 + offset_ay2**2))
            offset_canting_angles[key] = [offset_ax2, offset_ay2, offset_az2]
        # Set output.
        anglesx = offset_anglesx
        anglesy = offset_anglesy
        anglesz = offset_anglesz
        canting_angles = offset_canting_angles
        if actual_output_dir == None:
            actual_output_dir = os.path.join(
                orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'output', 'TestMotionBasedCanting'
            )

        df = pd.DataFrame({'Nx': anglesx}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Nx": "tab:blue"}, figsize=(15, 10))
        titles = title + ' Facet Normal X Component, ' + target_plot_name
        plt.title(titles)
        plt.xlabel('Facet id')
        plt.ylabel('X-component in units of Surface Normal')
        y_axis_max = 0.030  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        plt.ylim(
            -y_axis_max, y_axis_max
        )  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        #        plt.ylim(min(anglesx)-0.005, max(anglesx)+0.005)
        plt.grid(axis='y')
        figure_name = 'tca013_' + titled + '_CanglesX_' + target_file_name + '.png'
        plt.savefig(actual_output_dir + '/' + figure_name)

        if not self.interactive:
            plt.close('all')
        plt.close()

        df = pd.DataFrame({'Ny': anglesy}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Ny": "tab:orange"}, figsize=(15, 10))
        titles = title + ' Facet Normal Y Component, ' + target_plot_name
        plt.title(titles)
        plt.xlabel('Facet id')
        plt.ylabel('Y-component in units of surface normal')
        plt.ylim(
            -y_axis_max, y_axis_max
        )  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        #        plt.ylim(min(anglesy)-0.001, max(anglesy)+0.001)
        plt.grid(axis='y')
        figure_name = 'tca014_' + titled + '_CanglesY_' + target_file_name + '.png'
        plt.savefig(actual_output_dir + '/' + figure_name)
        plt.close()

        df = pd.DataFrame({'Nz': anglesz}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Nz": "magenta"}, figsize=(15, 10))
        titles = title + ' Facet Normal Z Component, ' + target_plot_name
        plt.title(titles)
        plt.xlabel('Facet id')
        plt.ylabel('Z-component in units of Surface Normal')
        y_axis_min_for_z = 0.9995
        plt.ylim(y_axis_min_for_z, 1)
        plt.grid(axis='y')
        figure_name = 'tca015_' + titled + '_CanglesZ_' + target_file_name + '.png'
        plt.savefig(actual_output_dir + '/' + figure_name)
        plt.close()

        # plt.show()
        plt.close()

    def example_canting_calculated_time(
        self,
        heliostat_name: str = None,
        target: str = None,
        canting_details: str = None,
        canting_dir: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
        time: str = None,
    ) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target == 'target'
            target_loc = Pxyz([[0], [0], [0]])

        time_name = time.replace('_', ':')

        # Define scenario.
        nsttf_pivot_height = 4.02  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_pivot_offset = 0.1778  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        focal_length_max = 210  # meters

        opencsp_dir = canting_dir

        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
            + ' Time '
            + time_name
        )
        target_file_name = (
            target
            + ' '
            + f'{float(target_x): .2f}'
            + ' '
            + f'{float(target_y): .2f}'
            + ' '
            + f'{float(target_z): .2f}'
            + ' Time '
            + time
        )

        canted_x_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about X (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about Y (rad)',
        )

        # heliostat z position and focal_length
        # z = 5.63  # meters
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters
        z = facet_origin.z

        name = heliostat_name
        title = 'Off-Axis Canted ' + heliostat_name
        title_file = 'Off-Axis Canted ' + heliostat_name

        # title = canting_details + '-Axis Canted ' + heliostat_name
        caption = heliostat_name + ' modeled as a symmetric paraboloid with focal length f=' + str(focal_length) + 'm.'
        # Solar field.
        short_name_sf = 'Mini NSTTF'
        name_sf = 'Mini NSTTF with ' + canting_details + '-Axis Canted' + heliostat_name
        caption_sf = (
            'NSTTF heliostat.' + '  ' + caption + '  ' + 'Facet surfaces and canting have the same focal length.'
        )
        comments = []

        # Construct heliostat objects and solar field object.
        def fn(x, y):
            return (x**2) / (4 * focal_length) + (y**2) / (4 * focal_length)

        h1_mirror = MirrorParametricRectangular(fn, (nsttf_facet_width, nsttf_facet_height))
        h1, location1 = HeliostatAzEl.from_csv_files(
            heliostat_name,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            h1_mirror,
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)

        h1.set_facet_cantings(canting_rotations)

        heliostats = [h1]
        heliostat_locations = [location1]

        UP = Pxyz([0, 0, 1])
        h1.set_orientation_from_az_el(0, np.pi / 2)
        # h_14W01.set_orientation_from_az_el(0, np.pi / 2)

        sf = SolarField(heliostats, lln.NSTTF_ORIGIN, name_sf)
        sf.set_heliostat_positions(heliostat_locations)

        comments_long = comments.copy()  # We'll add a different comment for the plots with long normals.
        comments_very_long = comments.copy()  # We'll add a different comment for the plots with very long normals.
        comments_exaggerated_z = (
            comments.copy()
        )  # We'll add a different comment for the plots with an exaggerated z axis.
        comments.append('Render mirror surfaces and normals, facet outlines, and heliostat centroid.')

        # Setup render control (long normals).
        mirror_control_long = rcm.RenderControlMirror(surface_normals=True, norm_len=15, norm_res=3)
        facet_control_long = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_long,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control_long = rcfe.RenderControlFacetEnsemble(facet_control_long, draw_outline=False)
        heliostat_control_long = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_long, draw_facet_ensemble=True
        )

        comments_long.append('Render mirror surfaces and long normals, facet outlines, and heliostat centroid.')

        # Draw and output 5W01 figure (long normals, xy view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(grid=False),
            vs.view_spec_xy(),
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            number_in_name=False,
            input_prefix=self.figure_prefix(6),
            name=title + ' (long normals, 9 for each facet) ' + target_file_name,
            title=title_file + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_long)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Setup render control (long normals).
        mirror_control_long_1 = rcm.RenderControlMirror(surface_normals=True, norm_len=65, norm_res=1)
        facet_control_long_1 = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_long_1,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control_long_1 = rcfe.RenderControlFacetEnsemble(facet_control_long_1, draw_outline=False)
        heliostat_control_long_1 = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_long_1, draw_facet_ensemble=True
        )

        comments_long.append('Render mirror surfaces and long normals, facet outlines, and heliostat centroid.')

        # Draw and output 5W01 figure (long normals, xy view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(grid=False),
            vs.view_spec_xy(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(7),
            name=title + ' (long normals, 1 for each facet) ' + target_file_name,
            title=title_file + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_long_1)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Setup render control (very long normals).
        mirror_control_very_long = rcm.RenderControlMirror(
            surface_normals=True,
            norm_len=(2 * focal_length_max),  # Twice the focal length is the center of curvature.
            norm_res=2,
        )
        facet_control_very_long = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_very_long,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=False,
        )
        facet_ensemble_control_very_long = rcfe.RenderControlFacetEnsemble(facet_control_very_long)
        heliostat_control_very_long = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_very_long, draw_facet_ensemble=True
        )
        solar_field_control_very_long = rcsf.RenderControlSolarField(heliostat_styles=heliostat_control_very_long)
        comments_very_long.append(
            'Render mirror surfaces and very long normals, facet outlines, and heliostat centroid.'
        )

        # Draw and output solar_field figure (very long normals, yz view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(8),
            name=title + ' (very long normals) ' + target_file_name,
            title=title_file + ', ' + target_plot_name,
            caption=caption_sf,
            comments=comments,
        )
        sf.draw(fig_record.view, solar_field_control_very_long)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Setup render control (exaggerated z).
        z_exaggerated_margin = 0.35  # meters, plus or minus reference height.
        decimal_factor = 100.0
        # Different z limits for each heliostat, because they are at different elevations on the sloped field.
        z_min = np.floor(decimal_factor * ((z + nsttf_pivot_offset) - z_exaggerated_margin)) / decimal_factor
        z_max = np.ceil(decimal_factor * ((z + nsttf_pivot_offset) + z_exaggerated_margin)) / decimal_factor
        z_min_str = str(str(z_min).strip("[]"))
        z_max_str = str(str(z_max).strip("[]"))
        exaggerated_z_limits = [float(z_min_str), float(z_max_str)]

        mirror_control_exaggerated_z = rcm.RenderControlMirror(surface_normals=False)
        facet_control_exaggerated_z = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_exaggerated_z,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=True,
        )
        facet_ensemble_control_exaggerated_z = rcfe.RenderControlFacetEnsemble(
            default_style=facet_control_exaggerated_z, draw_outline=False
        )
        heliostat_control_exaggerated_z = rch.RenderControlHeliostat(
            draw_centroid=False, facet_ensemble_style=facet_ensemble_control_exaggerated_z, draw_facet_ensemble=True
        )
        comments_exaggerated_z.append('Render heliostat with exaggerated z axis.')

        # Draw and output 5W01 figure (exaggerated z).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_3d(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(9),
            name=title + ' (exaggerated z) ' + target_file_name,
            title=title_file + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        fig_record.z_limits = exaggerated_z_limits
        h1.draw(fig_record.view, heliostat_control_exaggerated_z)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        mirror_control_exaggerated_z_normal = rcm.RenderControlMirror(surface_normals=True, norm_len=5, norm_res=1)
        facet_control_exaggerated_z_normal = rcf.RenderControlFacet(
            draw_mirror_curvature=True,
            mirror_styles=mirror_control_exaggerated_z_normal,
            draw_outline=True,
            draw_surface_normal=False,
            draw_name=False,
            draw_centroid=True,
        )
        facet_ensemble_control_exaggerated_z_normal = rcfe.RenderControlFacetEnsemble(
            default_style=facet_control_exaggerated_z_normal
        )
        heliostat_control_exaggerated_z_normal = rch.RenderControlHeliostat(
            draw_centroid=False,
            facet_ensemble_style=facet_ensemble_control_exaggerated_z_normal,
            draw_facet_ensemble=True,
        )

        # Draw and output 5W01 figure (exaggerated z).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_3d(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(10),
            name=title + ' (with normals) ' + target_file_name,
            title=title_file + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z_normal)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Draw and output 5W01 figure (exaggerated z, yz view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(11),
            name=title + ' (exaggerated z) ' + target_file_name,
            title=title_file + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        fig_record.equal = False  # Asserting equal axis scales contradicts exaggerated z limits in 2-d plots.
        fig_record.z_limits = (
            exaggerated_z_limits  # Limits are on z values, even though the plot is 2-d.  View3d.py handles this.
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

        # Draw and output 5W01 figure (exaggerated z, yz view with normals).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(12),
            name=title + ' (with normals) ' + target_file_name,
            title=title + ', ' + target_plot_name,
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z_normal)
        stest.show_save_and_check_figure(fig_record, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(fig_record)

    def example_canting_bar_charts_calculated_time(
        self,
        heliostat_name: str = None,
        target: str = None,
        canting_details: str = None,
        canting_dir: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
        time: str = None,
    ) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        # Define scenario.
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", target])
        if target == 'TowerTop':
            target_loc = tower.target_loc

        elif target == 'BCS':
            target_loc = tower.bcs

        elif target == "G3P3":
            target_loc = tower.g3p3
        else:
            target_loc = Pxyz([[0], [0], [0]])

        time_name = time.replace('_', ':')

        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters

        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # title = canting_details + '-Axis Canted ' + heliostat_name
        title = 'Off-Axis Canted ' + heliostat_name
        title_file = 'Off-Axis Canted ' + heliostat_name
        caption = heliostat_name + ' modeled as a symmetric paraboloid with focal length f=' + str(focal_length) + 'm.'

        # Construct heliostat objects and solar field object.
        def fn(x, y):
            return (x**2) / (4 * focal_length) + (y**2) / (4 * focal_length)

        h1_mirror = MirrorParametricRectangular(fn, (nsttf_facet_width, nsttf_facet_height))
        h1, location1 = HeliostatAzEl.from_csv_files(
            heliostat_name,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            h1_mirror,
        )
        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")
        target_plot_name = (
            target
            + ' Target ('
            + f'{float(target_x): .2f}'
            + ', '
            + f'{float(target_y): .2f}'
            + ', '
            + f'{float(target_z): .2f}'
            + ')'
            + ' Time '
            + time_name
        )
        target_file_name = (
            target
            + '_'
            + str(f'{float(target_x): .2f}').strip(' ')
            + '_'
            + str(f'{float(target_y): .2f}').strip(' ')
            + '_'
            + str(f'{float(target_z): .2f}').strip(' ')
            + '_Time_'
            + time
        )

        opencsp_dir = canting_dir

        canted_x_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about X (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about Y (rad)',
        )

        anglesx = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Surface Normal X',
        )
        anglesy = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Surface Normal Y',
        )
        anglesz = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Surface Normal Z',
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, h1.facet_ensemble.num_facets):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)

        h1.set_facet_cantings(canting_rotations)

        ####plot canting angles bar chart

        canting_angles = {}
        # ?? SCAFFOLDING RCB -- THIS IS CRUFTY.  THIS SHOULD GO IN AN ANALYSIS PLACE, AND ONLY COMPUTE ONCE.  SEE ALSO SIMILAR CODE IN PLOT NORMALS FUNCTION.
        # Compute average canting angle x and y components.\
        angles_x = []
        for x_angles in anglesx:
            x_angles = float(x_angles)
            angles_x.append(x_angles)
        angles_y = []
        for y_angles in anglesy:
            y_angles = float(y_angles)
            angles_y.append(y_angles)
        angles_z = []
        for z_angles in anglesz:
            z_angles = float(z_angles)
            angles_z.append(z_angles)

        average_ax = sum(angles_x) / len(angles_x)
        average_ay = sum(angles_y) / len(angles_y)
        # Compute offsets.
        offset_anglesx = [(x - average_ax) for x in angles_x]
        offset_anglesy = [(y - average_ay) for y in angles_y]
        offset_anglesz = []
        for facet in range(h1.facet_ensemble.num_facets):
            canting_angles[facet] = [angles_x[facet], angles_y[facet], angles_z[facet]]

        for offset_ax, offset_ay in zip(offset_anglesx, offset_anglesy):
            offset_az = np.sqrt(1.0 - (offset_ax**2 + offset_ay**2))
            offset_anglesz.append(offset_az)
        offset_canting_angles = {}
        for key in range(h1.facet_ensemble.num_facets):
            axyz = canting_angles[key]
            ax2 = axyz[0]
            ay2 = axyz[1]
            az2 = axyz[2]
            offset_ax2 = ax2 - average_ax
            offset_ay2 = ay2 - average_ay
            offset_az2 = np.sqrt(1.0 - (offset_ax2**2 + offset_ay2**2))
            offset_canting_angles[key] = [offset_ax2, offset_ay2, offset_az2]
        # Set output.
        anglesx = offset_anglesx
        anglesy = offset_anglesy
        anglesz = offset_anglesz
        canting_angles = offset_canting_angles
        if actual_output_dir == None:
            actual_output_dir = os.path.join(
                orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'output', 'TestMotionBasedCanting'
            )

        df = pd.DataFrame({'Nx': anglesx}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Nx": "tab:blue"}, figsize=(15, 10))
        titles = title_file + ' Facet Normal X Component, ' + target_plot_name
        plt.title(titles)
        plt.xlabel('Facet id')
        plt.ylabel('X-component in units of Surface Normal')
        y_axis_max = 0.030  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        plt.ylim(
            -y_axis_max, y_axis_max
        )  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        #        plt.ylim(min(anglesx)-0.005, max(anglesx)+0.005)
        plt.grid(axis='y')
        figure_name = 'tca013_' + title + '_canglesX_' + target_file_name + '.png'
        plt.savefig(actual_output_dir + '/' + figure_name)

        if not self.interactive:
            plt.close('all')
        plt.close()

        df = pd.DataFrame({'Ny': anglesy}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Ny": "tab:orange"}, figsize=(15, 10))
        titles = title_file + ' Facet Normal Y Component, ' + target_plot_name
        plt.title(titles)
        plt.xlabel('Facet id')
        plt.ylabel('Y-component in units of surface normal')
        plt.ylim(
            -y_axis_max, y_axis_max
        )  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        #        plt.ylim(min(anglesy)-0.001, max(anglesy)+0.001)
        plt.grid(axis='y')
        figure_name = 'tca014_' + title + '_canglesY_' + target_file_name + '.png'
        plt.savefig(actual_output_dir + '/' + figure_name)
        plt.close()

        df = pd.DataFrame({'Nz': anglesz}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Nz": "magenta"}, figsize=(15, 10))
        titles = title_file + ' Facet Normal Z Component, ' + target_plot_name
        plt.title(titles)
        plt.xlabel('Facet id')
        plt.ylabel('Z-component in units of Surface Normal')
        y_axis_min_for_z = 0.9995
        plt.ylim(y_axis_min_for_z, 1)
        plt.grid(axis='y')
        figure_name = 'tca015_' + title + '_canglesZ_' + target_file_name + '.png'
        plt.savefig(actual_output_dir + '/' + figure_name)
        plt.close()

        # plt.show()
        plt.close()

    def test_create_sofast_axis_canted_heliostat(
        self,
        heliostat_name: str = None,
        target: str = None,
        canting_details: str = None,
        off_axis_output_dir: str = None,
        actual_output_dir: str = None,
        expected_output_dir: str = None,
    ) -> None:

        #### need to change location of tower target when using this code!###

        # Initialize test.
        self.start_test()

        # View Setup

        comments = []

        # Initial positions
        tower = Tower(
            name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "BCS Tower", "target", "BCS"]
        )
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        tower_control = rct.normal_tower()
        bcs = tower.bcs
        f_index = '13'
        test_heliostat = [heliostat_name]
        test_tolerance = 0.001
        canted_x_angle = 0
        canted_y_angle = 0

        opencsp_dir = off_axis_output_dir

        target_x = str(target_loc.x).strip("[]")
        target_y = str(target_loc.y).strip("[]")
        target_z = str(target_loc.z).strip("[]")

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(180)
        test_el = np.deg2rad(0)

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        canted_x_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about X (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(
                opencsp_dir,
                heliostat_name
                + '_facet_details_canted_'
                + canting_details
                + '_axis_'
                + target
                + '_'
                + f'{float(target_x): .2f}'
                + '_'
                + f'{float(target_y): .2f}'
                + '_'
                + f'{float(target_z): .2f}'
                + '.csv',
            ),
            'Face Up Canting Rotation about Y (rad)',
        )

        canting_rotations: list[TransformXYZ] = []

        # Configuration setup
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_config = hc.HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
        test_heliostat.set_orientation(test_config)

        azim, elev, intersection = self.find_single_facet_azimuth_el_value(
            target_loc=bcs,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )

        test_config = hc.HeliostatConfiguration('az-el', az=azim, el=elev)
        test_heliostat.set_orientation(test_config)

        test_facet = test_heliostat.facet_ensemble.lookup_facet("13")
        __, trc_sur_norm = test_facet.survey_of_points(Resolution.center())
        sun_ray = -trc_sur_norm

        azimuth, elevation, intersection = self.find_single_facet_azimuth_el_value_bcs(
            sun_ray=sun_ray,
            target_loc=target_loc,
            target_plane_normal=target_plane_normal,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=azim,
            el=elev,
            canted_x_angle=canted_x_angle,
            canted_y_angle=canted_y_angle,
            tolerance=test_tolerance,
        )

        azimuth_deg = round(np.rad2deg(azimuth), 3)
        azimuth_degrees = f"{azimuth_deg: .3f}"
        elevation_deg = round(np.rad2deg(elevation), 3)
        elevation_degrees = f"{elevation_deg: .3f}"

        count = int(0)

        x_min = -0.01
        x_max = 0.01
        exaggerated_x = [x_min, x_max]
        z_min = float(tower.target_loc.z[0]) - 0.01
        z_max = float(tower.target_loc.z[0]) + 0.01
        exaggerated_z = [z_max, z_min]

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_config = hc.HeliostatConfiguration('az-el', az=azimuth, el=elevation)
        test_heliostat.set_orientation(test_config)
        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length_tar = Pxyz.distance(target_loc, facet_origin)  # meters
        focal_length_bcs = Pxyz.distance(bcs, facet_origin)  # meters
        current_length = focal_length_tar + (focal_length_tar * 0.15)
        init = focal_length_bcs + (focal_length_bcs * 0.15)

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        heliostat = test_heliostat.no_parent_copy()
        heliostat_loc = test_heliostat._self_to_parent_transform

        ##draw facet 13 on target_location
        facet = test_heliostat.lookup_facet('13')
        facet_no_parent = facet.no_parent_copy()
        facet_loc = facet.self_to_global_tranformation

        # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
        scene = Scene()
        scene.add_object(facet_no_parent)
        scene.set_position_in_space(facet_no_parent, facet_loc)
        sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center(), verbose=True)
        line_thickness: float = rcps.thick()

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(1),
            title=heliostat_name
            + ' SOFAST Tower Configuration, (az, el) = ('
            + azimuth_degrees
            + ' deg, '
            + elevation_degrees
            + ' deg)',
            caption='A single Sandia NSTTF heliostat ',
            comments='3D Ray trace with reflection from BCS target on tower to center of heliostat facet to target on tower.',
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(
            figure_control.view,
            rcrt.init_current_lengths(init_len=init, current_len=current_length, line_render=line_thickness),
        )
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        bcs.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(figure_control)

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(2),
            title=heliostat_name
            + ' SOFAST Tower Configuration, (az, el) = ('
            + azimuth_degrees
            + ' deg, '
            + elevation_degrees
            + ' deg)',
            caption='A single Sandia NSTTF heliostat ',
            comments='XY Ray trace with reflection from BCS target on tower to center of heliostat facet to target on tower.',
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(
            figure_control.view,
            rcrt.init_current_lengths(init_len=init, current_len=current_length, line_render=line_thickness),
        )
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        bcs.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(3),
            title=heliostat_name
            + ' SOFAST Tower Configuration, (az, el) = ('
            + azimuth_degrees
            + ' deg, '
            + elevation_degrees
            + ' deg)',
            caption='A single Sandia NSTTF heliostat ',
            comments='YZ Ray trace with reflection from BCS target on tower to center of heliostat facet to target on tower.',
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(
            figure_control.view,
            rcrt.init_current_lengths(init_len=init, current_len=current_length, line_render=rcps.thick()),
        )
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        bcs.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(4),
            title=heliostat_name
            + ' SOFAST Tower Configuration, (az, el) = ('
            + azimuth_degrees
            + ' deg, '
            + elevation_degrees
            + ' deg)',
            caption='A single Sandia NSTTF heliostat',
            comments='XZ Ray trace with reflection from BCS target on tower to center of heliostat facet to target on tower.',
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(
            figure_control.view,
            rcrt.init_current_lengths(init_len=init, current_len=current_length, line_render=rcps.thick()),
        )
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        bcs.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        # for facet in test_heliostat.facet_ensemble.facets:
        #     facet_no_parent = facet.no_parent_copy()
        #     facet_loc = facet.self_to_global_tranformation

        #     # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scen

        #     scene = Scene()
        #     scene.add_object(facet_no_parent)
        #     scene.set_position_in_space(facet_no_parent, facet_loc)
        #     sun_1ray = LightSourceSun.from_given_sun_position(Vxyz(sun_ray), 1)
        #     scene.add_light_source(sun_1ray)
        #     trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

        #     facet_ensemble_control = rcfe.facet_outlines_thin()
        #     heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        #     heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))
        #     count = int(count + 1)

        #     figure_control = fm.setup_figure_for_3d_data(
        #         rcfg.RenderControlFigure(tile_array=(1, 1)),
        #         rca.meters(),
        #         vs.view_spec_3d(),
        #         number_in_name=False,
        #         input_prefix=self.figure_prefix(1),
        #         title=heliostat_name
        #         + ' SOFAST Tower Configuration, Facet '
        #         + facet.name
        #         + ', (az,el)= ('
        #         + azimuth_degrees
        #         + ' deg, '
        #         + elevation_degrees
        #         + ' deg)',
        #         caption='A single Sandia NSTTF heliostat ',
        #         comments='3D Ray trace with reflection from BCS target on tower to center of heliostat facet to target on tower.',
        #     )
        #     test_heliostat.draw(figure_control.view, heliostat_style)
        #     tower.draw(figure_control.view, tower_control)
        #     trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        #     # figure_control.view.draw_single_Pxyz(AIMPOINT)
        #     AIMPOINT.draw_points(figure_control.view)
        #     # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        #     stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        #     if not self.interactive:
        #         plt.close('all')
        # self.show_save_and_check_figure(figure_control)

        # count = int(count + 1)
        # figure_control = fm.setup_figure_for_3d_data(
        #     rcfg.RenderControlFigure(),
        #     rca.meters(),
        #     vs.view_spec_xz(),
        #     number_in_name=False,
        #     input_prefix=self.figure_prefix(count),
        #     name=title + ' Facet ' + facet.name + ' Intersection at Aimpoint ' + target_file_name,
        #     title=title + ', Facet ' + facet.name + ", " + target_plot_name,
        #     caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
        #     comments=comments.append(
        #         "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        #     ),
        # )
        # figure_control.x_limits = exaggerated_x
        # figure_control.z_limits = exaggerated_z
        # intersection_point.draw(figure_control.view)
        # # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        # target_origin.draw_points(figure_control.view, rcps.marker(color='r'))
        # # figure_control.view.show()
        # # self.show_save_and_check_figure(figure_control)
        # stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        # if not self.interactive:
        #     plt.close('all')

        self.save_az_el_to_csv(
            actual_output_dir, heliostat_name, azimuth, elevation, azimuth_degrees, elevation_degrees
        )

    def create_off_axis_canted_heliostat(
        self, heliostat_name, actual_output_dir: str = None, expected_output_dir: str = None
    ) -> None:

        # Initialize test.
        self.start_test()

        # View Setup

        comments = []

        if actual_output_dir == None:
            actual_output_dir = self.actual_output_dir
            expected_output_dir = self.expected_output_dir

        # Initial positions
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        tower_control = rct.normal_tower()

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Determine canting angles for single facet
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        # facet_index = test_heliostat.facet_ensemble.children(dict)
        opencsp_dir = os.path.join(
            orp.opencsp_code_dir(),
            'common',
            'lib',
            'test',
            'data',
            'input',
            'sandia_nsttf_test_definition',
            'NSTTF_Canting_Prescriptions',
        )

        canted_x_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'),
            'Face Up Canting Rotation about X',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'),
            'Face Up Canting Rotation about Y',
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_cantings(canting_rotations)

        heliostat = test_heliostat.no_parent_copy()
        heliostat_loc = test_heliostat._self_to_parent_transform

        # # Configuration setup
        # facet = self.cant_single_facet(heliostat_name, f_index, az, el, facet_canted_x_angle, facet_canted_y_angle)
        # facet_no_parent = facet.no_parent_copy()
        # facet_location = facet.self_to_global_tranformation

        # # Set canting
        # position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
        # canting = Rotation.from_euler('xyz', [facet_canted_x_angle, facet_canted_y_angle, 0], degrees=True)
        # facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)
        # facet_no_parent = facet.no_parent_copy()

        # solar_field.set_full_field_tracking(AIMPOINT, TIME)

        # h_pointing_vector = UP.rotate(test_heliostat._self_to_parent_transform.R)
        # test_heliostat.set_orientation_from_pointing_vector(h_pointing_vector)

        # dimensions for intersection drawing
        x_min = -0.01
        x_max = 0.01
        exaggerated_x = [x_min, x_max]
        z_min = 63.55 - 0.01
        z_max = 63.55 + 0.01
        exaggerated_z = [z_max, z_min]
        x_lim = [-10, 30]  ##TODO mhh figure out limits that are automated
        y_lim = [0, 160]
        z_lim = [-10, 80]
        count = int(0)

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        # focal length
        facet = test_heliostat.lookup_facet('13')
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
        focal_length = Pxyz.distance(target_loc, facet_origin)  # meters
        current_length = focal_length + (focal_length * 0.15)
        init = 80

        for facet in test_heliostat.facet_ensemble.facets:
            facet_no_parent = facet.no_parent_copy()
            facet_loc = facet.self_to_global_tranformation

            # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_loc)
            # Add ray trace to field
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(facet.name, rcf.outline('blue'))

            count = int(count + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile_array=(1, 1)),
                rca.meters(),
                vs.view_spec_3d(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                title='Off-Axis Canted NSTTF Heliostat' + heliostat_name + ', Facet ' + facet.name + ' with Ray-Trace',
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ', Facet ' + facet.name,
                comments=comments.append(
                    "3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            # figure_control.view.draw_single_Pxyz(AIMPOINT)
            AIMPOINT.draw_points(figure_control.view)
            # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

            count = int(count + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                title='Off-Axis Canted NSTTF Heliostat '
                + heliostat_name
                + ', Facet '
                + facet.name
                + " Intersection at Aimpoint",
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments=comments.append(
                    "XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
                ),
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            AIMPOINT.draw_points(figure_control.view, rcps.marker(color='r'))
            # self.show_save_and_check_figure(figure_control)
            stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
            if not self.interactive:
                plt.close('all')

        scene = Scene()
        scene.add_object(heliostat)
        scene.set_position_in_space(heliostat, heliostat_loc)
        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_3d(),
            number_in_name=False,
            input_prefix=self.figure_prefix(51),
            title='Off-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XY Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(52),
            title='Off-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XY Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(53),
            title='Off-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="YZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(54),
            title='Off-Axis Canted NSTTF Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        AIMPOINT.draw_points(figure_control.view)
        # figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        # self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(55),
            title='Off-Axis Canted NSTTF Heliostat ' + heliostat_name + " Intersection at Aimpoint All Facets",
            caption='A single Sandia NSTTF heliostat ' + heliostat_name,
            comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        AIMPOINT.draw_points(figure_control.view, rcps.marker(color='r'))
        # self.show_save_and_check_figure(figure_control)
        stest.show_save_and_check_figure(figure_control, actual_output_dir, expected_output_dir, self.verify)
        if not self.interactive:
            plt.close('all')

    ########## values provided for outputs ###########

    # heliostat_name = ("9W1")
    # tower= Tower(name='Sandia NSTTF', origin=np.array([0,0,0]), parts = ["whole tower", "target"])
    # plane = tuple(tower.target_loc, Uxyz([0, 1, 0]))

    # target_loc_1 = tower.target_loc
    # target_plane_normal_1 = Uxyz([0,1,0])
    # heliostat_name_1=("9W1")
    # f_index_1 = 13

    # #close to where the target is for facet 1
    # azimuth_1 = np.deg2rad(177.5)
    # el_1 = np.deg2rad(43.2)

    # when canting is 0 for facet 13 ray trace given solar noon TIME
    # azimuth_1 = np.deg2rad(178.4312)  # 177
    # el_1 = np.deg2rad(49.9287)  # 43

    # #facing plane
    # azimuth_1 = np.deg2rad(180) #or 0
    # el_1 = np.deg2rad(0) #or 180

    # #Doesn't hit plane
    # azimuth_1 = np.deg2rad(0)
    # el_1 = np.deg2rad(0) or 90

    # test_tolerance =.001

    # result = test_object.projected_facet_normal_intersection_point_offset(target_loc=target_loc_1, target_plane_normal=target_plane_normal_1, heliostat_name=heliostat_name_1, f_index=f_index_1, az=azimuth_1, el=el_1)
    # result = test_object.azimuth_binary_search(target_loc=target_loc_1, target_plane_normal=target_plane_normal_1, heliostat_name=heliostat_name_1, f_index=f_index_1, az=azimuth_1, el=el_1, tolerance=test_tolerance)
    # result = test_object.find_single_facet_azimuth_el_value(target_loc=target_loc_1, target_plane_normal=target_plane_normal_1, heliostat_name=heliostat_name_1, f_index=f_index_1, az=azimuth_1, el=el_1, tolerance=test_tolerance)
    # result = test_object.elevation_binary_search(target_loc=target_loc_1, target_plane_normal=target_plane_normal_1, heliostat_name=heliostat_name_1, f_index=f_index_1, az=azimuth_1, el=el_1, tolerance=test_tolerance)
    # result = test_object.find_all_azimuth_el_test_values(target_loc=target_loc_1, target_plane_normal=target_plane_normal_1, heliostat_name=heliostat_name_1, az=azimuth_1, el=el_1, tolerance=test_tolerance)

    # # Save figures.
    # if save_figures:
    #     print('\n\nSaving figures...')
    #     # Output directory.
    #     output_path = os.path.join('..', ('output_' + datetime.now().strftime('%Y_%m_%d_%H%M')))
    #     if not(os.path.exists(output_path)):
    #         os.makedirs(output_path)
    #     fm.save_all_figures(output_path)

    # if show_figures:
    #     input("Press 'Enter' to close the figures...")
    #     plt.close('all')


# MAIN EXECUTION

if __name__ == "__main__":

    # Control flags.
    interactive = True
    # Set verify to False when you want to generate all figures and then copy
    # them into the expected_output directory.
    # (Does not affect pytest, which uses default value.)
    verify = False  # False
    # Setup.
    test_object = TestMotionBasedCanting()
    test_object.setUpClass(interactive=interactive, verify=verify)
    test_object.setUp()
    # Tests.
    lt.info('Beginning tests...')

    tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
    target_loc_1 = tower.target_loc
    target_plane_normal_1 = Uxyz([0, 1, 0])
    heliostat_name_1 = "9W1"
    f_index_1 = "13"
    azimuth_1 = np.deg2rad(178.4312)  # 177
    el_1 = np.deg2rad(49.9287)  # 43
    tolerance_1 = 0.001
    canted_x_angle_1 = 0  # 2
    canted_y_angle_1 = 0

    # result = test_object.projected_facet_normal_intersection_point_offset(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    # )
    # print("next test")

    # result = test_object.spherical_canting_facet(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     az=azimuth_1,
    #     el=el_1,
    # )
    # print("next test")

    # result = test_object.projected_ray_trace_intersection_offset(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     canted_x_angle=canted_x_angle_1,
    #     canted_y_angle=canted_y_angle_1,
    # )
    # print("next test")

    # result = test_object.canted_x_binary_search(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     canted_x_angle=canted_x_angle_1,
    #     canted_y_angle=canted_y_angle_1,
    #     tolerance=tolerance_1,
    # )
    # print("next test")

    # result = test_object.canted_y_binary_search(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     canted_x_angle=canted_x_angle_1,
    #     canted_y_angle=canted_y_angle_1,
    #     tolerance=tolerance_1,
    # )
    # print("next test")

    # result = test_object.azimuth_binary_search(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     tolerance=tolerance_1,
    # )
    # print("next test")

    # result = test_object.elevation_binary_search(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     tolerance=tolerance_1,
    # )
    # print("next test")

    # result = test_object.find_single_facet_azimuth_el_value(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     tolerance=tolerance_1,
    # )
    # print("next test")

    # result = test_object.find_single_facet_canting_angles(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     canted_x_angle=canted_x_angle_1,
    #     canted_y_angle=canted_y_angle_1,
    #     tolerance=tolerance_1,
    # )
    # print("next test")

    # result = test_object.find_all_canting_angle_values(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     canted_x_angle=canted_x_angle_1,
    #     canted_y_angle=canted_y_angle_1,
    #     tolerance=tolerance_1,
    # )
    # print("next test")

    # result = test_object.find_all_azimuth_el_test_values(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     tolerance=tolerance_1,
    # )
    # print("next test")

    # test_object.test_9W1_heliostat()
    # test_object.test_9W1_with_canting()
    # test_object.test_azimuth_binary_search()
    # test_object.test_el_binary_search()
    # test_object.test_single_facet()
    # test_object.test_all_facets()
    # test_object.test_when_initial_position_not_on_target()
    # test_object.test_canted_x_angle_binary_search()
    # test_object.test_canted_y_angle_binary_search()
    # test_object.test_find_single_canting_angles()

    # count = int(0)

    # opencsp_dir = os.path.join(
    #     orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'input', 'sandia_nsttf_test_definition'
    # )

    # heliostat_names = test_object.read_csv_float(
    #     os.path.join(opencsp_dir, 'NSTTF_Heliostats_origin_at_torque_tube.csv'), 'Name'
    # )

    # for heliostat_name in heliostat_names:
    # test_object.test_off_axis_code(heliostat_name)
    #     test_object.test_on_axis_code(heliostat_name)

    # heliostat_name = '5W1'
    # test_object.test_on_axis_code(heliostat_name)

    # test_object.test_off_axis_canting_angles_vector_g3p3()
    # test_object.test_off_axis_canting_angles_vector_towertop()
    # test_object.test_off_axis_canting_angles_vector_bcs()
    # test_object.test_on_axis_canting_angles_vector_towertop()
    # test_object.test_off_axis_canting_angles_vector_towertop_time()
    test_object.test_off_axis_not_calculated_canting_angles_vector_towertop_time()

    # test_object.test_sofast_axis_canting_angles_vector_towertop()

    test_object.test_on_axis_canting_in_off_axis_orientation_vector_towertop()

    # test_object.test_off_axis_code(heliostat_name)
    # test_object.test_create_sofast_axis_canted_heliostat()
    # test_object.test_on_axis_canting_angles()
    # test_object.find_all_on_axis_canting_angles(heliostat_name)
    # test_object.create_on_axis_canted_heliostat(heliostat_name)
    # test_object.example_canting_bar_charts_calculated(heliostat_name, canting_details)
    # test_object.example_canting_calculated(heliostat_name, canting_details)
    # test_object.test_off_axis_canting_angles()
    # test_object.find_all_off_axis_canting_angles(heliostat_name)
    # test_object.create_off_axis_canted_heliostat(heliostat_name)

    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
