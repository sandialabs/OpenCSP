"""
Demonstrate Motion Based Canting Experiment.

Copyright (c) 2021 Sandia National Laboratories.

"""

import copy
import csv as csv
import os
import sys as sys
import math
from datetime import datetime
from typing import Iterable
import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rotation
from sympy import Symbol, diff
from opencsp.common.lib.geometry.Vxy import Vxy

import opencsp.common.lib.csp.sun_position as sp
import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.file.CsvColumns as CsvColumns
import opencsp.common.lib.csp.HeliostatAzEl as Heliostat
import opencsp.common.lib.csp.HeliostatConfiguration as hc
import opencsp.common.lib.csp.RayTrace as rt
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.tool.string_tools as st
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
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
from opencsp.common.lib.opencsp_path import opencsp_settings
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.HeliostatAbstract import HeliostatAbstract
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.file.CsvColumns import CsvColumns
from opencsp.common.lib.csp.HeliostatConfiguration import HeliostatConfiguration
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.LightSourcePoint import LightSourcePoint
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
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
from opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror
from opencsp.common.lib.render_control.RenderControlFacet import RenderControlFacet
from opencsp.common.lib.render_control.RenderControlFacetEnsemble import RenderControlFacetEnsemble
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
from opencsp.common.lib.render_control.RenderControlLightPath import RenderControlLightPath
from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace


class TestMotionBasedCanting(to.TestOutput):

    @classmethod
    def setUpClass(
        self,
        source_file_body: str = 'TestMotionBasedCanting',  # Set these here, because pytest calls
        figure_prefix_root: str = 'tca',  # setUpClass() with no arguments.
        interactive: bool = False,
        verify: bool = True,
    ):
        self._pivot = 0
        self._az = 0
        self._el = 0
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
        self,
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str,
        az: float,
        el: float,
        tolerance: float,
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
        tolerance: acceptable value of error for azimuth and elevation
        """
        # sets solar field
        test_config = hc.HeliostatConfiguration("az-el", az, el)

        # # # Define upward-facing heliostat orientation.

        UP = Vxyz([0, 0, 1])

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)

        facet = test_heliostat.lookup_facet('13')

        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())

        focal_length = (Pxyz.distance(target_loc, facet_origin)) / 2
        print(focal_length)

        curved_equation = FunctionXYContinuous(f"(x**2 + y**2)/(4*{focal_length})")

        mirror_template = MirrorParametricRectangular(curved_equation, (1.2, 1.2))
        test_heliostat, test_heliostat_location = HeliostatAzEl.from_csv_files(
            heliostat_name,
            dpft.sandia_nsttf_test_heliostats_origin_file(),
            dpft.sandia_nsttf_test_facet_centroidsfile(),
            mirror_template,
        )
        test_heliostat.set_canting_from_equation(curved_equation)

        is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
        intersection_point = Intersection.plane_lines_intersection(
            (facet_origin, facet_normal), (target_loc, target_plane_normal)
        )
        angles = test_heliostat.extract_angles(curved_equation)

        offset_x = intersection_point.x - target_loc.x
        offset_z = intersection_point.z - target_loc.z
        print(
            "In TestMotionBasedCanting.projected_facet_normal_intersection_point_offset: x offset from target to intersection=",
            offset_x,
            "    z offset from target to intersection=",
            offset_z,
        )  # TODO mhh

        return is_intersect, offset_x, offset_z, intersection_point, angles

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
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str,
        f_index: int,
        az: float,
        el: float,
        canted_x_angle: float,
        canted_y_angle: float,
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

        # Add ray trace
        solar_field.set_full_field_tracking(AIMPOINT, TIME)
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center())

        place_to_intersect = (AIMPOINT, target_plane_normal)
        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())

        # Compute offset from target
        is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
        intersection_points = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect).intersection_points

        offset_x = intersection_points.x - target_loc.x
        offset_z = intersection_points.z - target_loc.z
        print(
            f"In TestMotionBasedCanting.projected_ray_trace_intersection_offfset: for facet: {f_index} \n\t x offset from target to intersection= {offset_x}, z offset from target to intersection= {offset_z}"
        )  # TODO mhh

        return is_intersect, offset_x, offset_z, intersection_points

    def canted_x_binary_search(
        self,
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str,
        f_index: int,
        az: float,
        el: float,
        canted_x_angle: float,
        canted_y_angle: float,
        tolerance: float,
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
            target_loc, target_plane_normal, heliostat_name, f_index, az, el, canted_x_angle, canted_y_angle
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
                    target_loc, target_plane_normal, heliostat_name, f_index, az, el, search_x_angle, canted_y_angle
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
                    target_loc, target_plane_normal, heliostat_name, f_index, az, el, search_x_angle, canted_y_angle
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
                target_loc, target_plane_normal, heliostat_name, f_index, az, el, middle_x_angle, canted_y_angle
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
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str,
        f_index: int,
        az: float,
        el: float,
        canted_x_angle: float,
        canted_y_angle: float,
        tolerance: float,
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
            target_loc, target_plane_normal, heliostat_name, f_index, az, el, canted_x_angle, canted_y_angle
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
                    target_loc, target_plane_normal, heliostat_name, f_index, az, el, canted_x_angle, search_y_angle
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
                    target_loc, target_plane_normal, heliostat_name, f_index, az, el, canted_x_angle, search_y_angle
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
                target_loc, target_plane_normal, heliostat_name, f_index, az, el, canted_x_angle, middle_y_angle
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
        target_loc: 'Pxyz',
        target_plane_normal: 'Uxyz',
        heliostat_name: str,
        f_index: int,
        az: float,
        el: float,
        canted_x_angle: float,
        canted_y_angle: float,
        tolerance: float,
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

        is_intersect, offset_x, offset_z, intersection_point = self.projected_ray_trace_intersection_offset(
            target_loc, target_plane_normal, heliostat_name, f_index, az, el, canted_x_angle, canted_y_angle
        )

        # iterates through finding x and y angle values, by first finding x value, then y value given x, etc. until within tolerance
        if Pxyz.distance(intersection_point, target_loc) > tolerance:
            for i in range(1, 21):
                canted_x_angle, offset_z_points = self.canted_x_binary_search(
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
                canted_y_angle, offset_x_points = self.canted_y_binary_search(
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
                    target_loc, target_plane_normal, heliostat_name, f_index, az, el, canted_x_angle, canted_y_angle
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
        target_loc: Pxyz,
        target_plane_normal: Uxyz,
        heliostat_name: str,
        az: float,
        el: float,
        canted_x_angle: float,
        canted_y_angle: float,
        tolerance: float,
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
                tolerance: acceptable value of error for azimuth and elevation

        """

        # sets solar field
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)

        all_facets = []
        canted_x_angles = []
        canted_y_angles = []
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
            facet_canted_x_angle, facet_canted_y_angle, intersection, offset_z, offset_x = (
                self.find_single_facet_canting_angles(
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
            )
            canting_angles_found = {
                'facet_number': f_index,
                'canting data': {'canted x angle': facet_canted_x_angle, 'canted y angle': facet_canted_y_angle},
            }
            f_idx.append(f_index)
            canted_x_angles.append(facet_canted_x_angle)
            canted_y_angles.append(facet_canted_y_angle)
            all_facets.append(canting_angles_found)

        for facet in all_facets:
            print(facet)  # TODO mhh

        return canted_x_angles, canted_y_angles, f_idx

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
        # old_positions = heliostat.facet_ensemble.facet_positions
        self.set_tracking(heliostat, aimpoint, long_lat, time)

        cantings: list[Rotation] = []
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
        canting_x = []
        canting_y = []
        canting_z = []
        canting_angles = []

        for facet in heliostat.facet_ensemble.facets:
            # facet_origin = facet._self_to_parent_transform.apply(Pxyz.origin())  ## same thing as survey of points
            facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            # facet_origin = facet.get_transform_relative_to(heliostat.facet_ensemble).apply(Pxyz.origin())
            # print(f"fac_ori x: {facet_origin.x}, y: {facet_origin.y}, z: {facet_origin.z}")

            # facet_normal = UP.rotate(facet.get_transform_relative_to(heliostat.facet_ensemble).R)
            # print(f"fac_norm x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")

            # facet_normal = UP.rotate(facet._self_to_parent_transform.R)  ## same thing as facet_normal
            vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz(facet_origin, aimpoint, long_lat, time)
            # vector_the_normal_should_be = Vxyz(np.array([[1], [0], [0]]))
            # facet_normal = Vxyz(np.array([[0], [0], [1]]))
            # print(
            #     f"fac_ori x: {vector_the_normal_should_be.x}, y: {vector_the_normal_should_be.y}, z: {vector_the_normal_should_be.z}"
            # )
            # print(f"Vector_should_be{vector_the_normal_should_be}")
            # print(f"facet_normal")
            # sun = Vxyz(sp.sun_position(long_lat, time))
            # ref_ray = rt.calc_reflected_ray(facet_normal[0], sun[0])
            # ref_ray_2 = rt.calc_reflected_ray(vector_the_normal_should_be[0], sun[0])
            # vector.append(vector_the_normal_should_be)
            # R1_prime = ref_ray.align_to(ref_ray_2)
            R3 = facet_normal.align_to(vector_the_normal_should_be)
            # Rotation.from_matrix(R3)
            # new = facet_normal.rotate(R3)
            # print(f"new x: {new.x}, y: {new.y}, z: {new.z}")

            # R1_prime = R1_prime_
            # canting_angles = []
            # angles = R1_prime.as_euler('xyz')
            # canted_x, canted_y = angles[0], angles[1]
            # canting_angles.append((canted_x, canted_y))
            # # print(
            #     f"Facet {facet.name}: x_canting_angle = {canted_x:.5f} degrees, y_canting_angle = {canted_y:.5f} degrees"
            # )
            # # print(f"fac_norm x: {new.x}, y: {new.y}, z: {new.z}")
            # facet_normal.rotate_in_place(R3)
            # # cantings.append(R2_inv * R1_prime * R2)    ## unnecessary code
            canting_rotations = R3
            print(f"{R3.as_matrix()}")
            # # set canting
            # position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
            # # canting = Rotation.from_euler('xyz', [canted_x_angle, canted_y_angle, 0], degrees=False)
            # facet._self_to_parent_transform = TransformXYZ.from_R_V(R1_prime, position)
            cantings.append(canting_rotations)

            # heliostat.set_orientation_from_pointing_vector(h_pointing_vector)

            # for facet in heliostat.facet_ensemble.facets:
            #     facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            #     # sun = Vxyz(sp.sun_position(long_lat, time))
            #     # surface_norm = rt.calc_reflected_ray(facet_normal[0], sun[0])
            #     print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            # set canting

        heliostat.facet_ensemble.set_facet_canting(cantings)
        for facet in heliostat.facet_ensemble.facets:
            facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            # sun = Vxyz(sp.sun_position(long_lat, time))
            # surface_norm = rt.calc_reflected_ray(facet_normal[0], sun[0])
            # print(f"x: {facet_origin.x}, y: {facet_origin.y}, z: {facet_origin.z}")

        self.set_tracking(heliostat, aimpoint, long_lat, time)

    def save_to_csv(
        self,
        heliostat_name,
        f_index,
        up_f_origin_x,
        up_f_origin_y,
        up_f_origin_z,
        canted_x_angles,
        canted_y_angles,
        up_x_sur_norm,
        up_y_sur_norm,
        up_z_sur_norm,
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
    ) -> None:
        """
        Outputs a CSV file with x and y canting angle values for each facet.

        Parameters
        ----------
        canted_x_angle rotations : list of canted x angle rotations in radians
        canted_y_angle rotations : list of canted y angle rotations in radians
        f_index : list of facet indexes

        """

        # Both lists have same length
        assert len(canted_x_angles) == len(
            canted_y_angles
        ), "canted_x_angles and canted_y_angles must have the same length"

        # Creates list of rows, each row is a list of [facet, canted_x_angle, canted_y_angle, offset_z, offset_x]
        rows = []
        for facet in range(len(f_index)):
            row = [facet + 1]
            # if facet < len(f_origin):
            #     row.append(extract_coordinates(str(f_origin[facet])))
            # else:
            #     row.append('')
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
            if facet < len(canted_x_angles):
                row.append(canted_x_angles[facet])
            else:
                row.append('')  # If there are fewer angles than facets, fill with empty string
            if facet < len(canted_y_angles):
                row.append(canted_y_angles[facet])
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
            if facet < len(trc_f_origin_x):
                row.append(str(trc_f_origin_x[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(trc_f_origin_y):
                row.append(str(trc_f_origin_y[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(trc_f_origin_z):
                row.append(str(trc_f_origin_z[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(trc_x_sur_norm):
                row.append(str(trc_x_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(trc_y_sur_norm):
                row.append(str(trc_y_sur_norm[facet]).strip("[]"))
            else:
                row.append('')
            if facet < len(trc_z_sur_norm):
                row.append(str(trc_z_sur_norm[facet]).strip("[]"))
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

        file_name = heliostat_name + '_facet_details_canted_off_axis.csv'
        output_directory = os.path.join(self.output_path, 'data', 'output', self.source_file_body)
        file_path = os.path.join(output_directory, file_name)

        os.makedirs(output_directory, exist_ok=True)

        # Write to CSV file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    'Facet Index',
                    'Face Up Facet Origin X',
                    'Face Up Facet Origin Y',
                    'Face Up Facet Origin Z',
                    'Face Up Canting Rotation about X',
                    'Face Up Canting Rotation about Y',
                    'Face Up Surface Normal X',
                    'Face Up Surface Normal Y',
                    'Face Up Surface Normal Z',
                    'Tracking Az (rad)',
                    'Tracking El (rad)',
                    'Tracking Facet Origin X',
                    'Tracking Facet Origin Y',
                    'Tracking Facet Origin Z',
                    'Tracking Surface Normal X',
                    'Tracking Surface Normal Y',
                    'Tracking Surface Normal Z',
                    'Reflected Ray Target Plane Intersection X',
                    'Reflected Ray Target Plane Intersection Z',
                ]
            )
            writer.writerows(rows)

    def read_csv_float(self, input_dir_body_ext: str, column_name: str):
        # print('In FrameNameXyList.load(), loading input file: ', input_dir_body_ext)
        # Check if the input file exists.

        # # Check if the file exists at the specified path
        if not ft.file_exists(input_dir_body_ext):
            raise OSError('In FrameNameXyList.load(), file does not exist: ' + str(input_dir_body_ext))
        # Open and read the file.
        with open(input_dir_body_ext, mode='r', newline='') as input_stream:
            reader = csv.reader(input_stream, delimiter=',')

            # Read the header to find the index of the desired column
            header = next(reader)
            column_index = header.index(column_name)

            # Extract the column into a list, excluding the header
            column_data = [float(row[column_index]) for row in reader]
            # column_data = [f'[{row[column_index]}]' for row in reader]

        return column_data

    def test_off_axis_code(self) -> None:
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
        az = np.deg2rad(178.34514913)
        el = np.deg2rad(43.0303633698)
        test_tolerance = 0.001
        tower_control = rct.normal_tower()

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Configuration setup
        solar_field = self.solar_field
        # test_config = hc.HeliostatConfiguration("az-el", az, el)
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        # HeliostatAzEl.set_orientation_from_az_el(test_heliostat, az, el)
        # test_heliostat.set_orientation(test_config)

        # h_pointing_vector = UP.rotate(test_heliostat._self_to_parent_transform.R)
        # test_heliostat.set_orientation_from_pointing_vector(h_pointing_vector)
        # test_heliostat.set_tracking_configuration(AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        # self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        # HeliostatAzEl.set_orientation_from_az_el(test_heliostat, az, el)

        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        # h_pointing_vector = UP.rotate(test_heliostat._self_to_parent_transform.R)
        # test_heliostat.set_orientation_from_pointing_vector(h_pointing_vector)

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        # self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        # set canting

        for facet in test_heliostat.facet_ensemble.facets:
            facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            # sun = Vxyz(sp.sun_position(lln.NSTTF_ORIGIN, TIME))
            # surface_norm = rt.calc_reflected_ray(facet_normal[0], sun[0])
            # print(f"{facet_origin.x}, {facet_origin.y}, {facet_origin.z}")

        az = test_heliostat._az
        el = test_heliostat._el
        print(f"az and el = {az}, {el}")
        # test_heliostat.set_tracking_configuration(AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        # test_heliostat.set_orientation(test_config)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center())

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        # test_heliostat_origin, test_heliostat_normal = test_heliostat.survey_of_points(Resolution.center())
        # is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, test_heliostat_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )

        # DRAWING
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(1, 1)),
            rca.meters(grid=False),
            vs.view_spec_3d(),
            title=f"9W1 Ray Trace 3D through Target",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=200, current_len=220))
        figure_control.view.draw_single_Pxyz(AIMPOINT)
        figure_control.view.show()
        # self.show_save_and_check_figure(figure_control)

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(), rca.meters(), vs.view_spec_xy(), title="5W1 Ray Trace Birds View"
        )
        test_heliostat.draw(figure_control.view)
        # trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=300))
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(), rca.meters(), vs.view_spec_xz(), title="5W1 Ray Trace Looking North View"
        )
        test_heliostat.draw(figure_control.view)
        # trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=300))
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(), rca.meters(), vs.view_spec_yz(), title="5W1 Ray Trace Side View"
        )
        test_heliostat.draw(figure_control.view)
        # trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=300))
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(), rca.meters(), vs.view_spec_xz(), title="Intersection at aimpoint"
        )
        intersection_point.draw(figure_control.view)
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        figure_control.view.show()

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

        f_index = '1'
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

        # # heliostat selection index
        # heliostat_id = solar_field.heliostat_dict[heliostat_name]  # for 9W1: 87
        # heliostat = solar_field.heliostats[heliostat_id]

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

        result = test_object.find_all_azimuth_el_test_values(
            target_loc=test_target_loc,
            target_plane_normal=test_target_plane_normal,
            heliostat_name=heliostat_name,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )

        return

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
        figure_control.view.draw_single_Pxyz(AIMPOINT)
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
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        self.show_save_and_check_figure(figure_control)

    def test_find_all_canting_angles(self) -> None:
        """
        Draws ray trace intersection for 25 facets with the target.

        """
        # Initialize test.
        self.start_test()

        # View Setup
        heliostat_name = "14E6"

        # Initial positions
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

        az = test_heliostat._az
        el = test_heliostat._el

        # Determine canting angles for single facet
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)

        # facet_index = test_heliostat.facet_ensemble.children(dict)
        all_facets = []
        canted_x_angles = []
        canted_y_angles = []
        offset_x_found = []
        offset_z_found = []
        x_sur_norm = []
        y_sur_norm = []
        z_sur_norm = []
        f_origin = []
        f_idx = []
        cantings = []

        # checks az and el values to see if facet surface normal for facet 13 intersects with given plane.
        f_index_temp = '13'

        is_intersect, offset_x, offset_z, intersection_point = self.projected_facet_normal_intersection_point_offset(
            target_loc, target_plane_normal, heliostat_name, f_index_temp, az, el
        )
        az, el = self.doesnt_hit_plane_with_original_values(is_intersect, az, el)

        # iterates through every facet to find the azmimuth and elevation values that correspond with a surface normal that intersects with the target.
        for f_index in range(test_heliostat.facet_ensemble.num_facets):
            # for f_index in range(12, 13):
            f_index = str(f_index + 1)
            print("\nIn TestMotionBasedCanting.find_all_canting_angle_values:    facet=", f_index)
            facet_canted_x_angle, facet_canted_y_angle, intersection, offset_z, offset_x = (
                self.find_single_facet_canting_angles(
                    target_loc,
                    target_plane_normal,
                    heliostat_name,
                    f_index,
                    az,
                    el,
                    test_canted_x_angle,
                    test_canted_y_angle,
                    test_tolerance,
                )
            )
            canting_angles_found = {
                'facet_number': f_index,
                'canting data': {'canted x angle': facet_canted_x_angle, 'canted y angle': facet_canted_y_angle},
            }
            f_idx.append(f_index)
            canted_x_angles.append(facet_canted_x_angle)
            canted_y_angles.append(facet_canted_y_angle)
            offset_x_found.append(offset_x)
            offset_z_found.append(offset_z)
            all_facets.append(canting_angles_found)

            # Configuration setup
            facet = self.cant_single_facet(heliostat_name, f_index, az, el, facet_canted_x_angle, facet_canted_y_angle)
            facet_no_parent = facet.no_parent_copy()
            facet_location = facet.self_to_global_tranformation

            # Set canting
            position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
            canting = Rotation.from_euler('xyz', [facet_canted_x_angle, facet_canted_y_angle, 0], degrees=False)
            facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)
            facet_no_parent = facet.no_parent_copy()

            cantings.append(canting)

            # Set solar field scene
            scene = Scene()
            scene.add_object(facet_no_parent)
            scene.set_position_in_space(facet_no_parent, facet_location)
            solar_field.set_full_field_tracking(AIMPOINT, TIME)

            # Add ray trace to field
            sun_1ray = LightSourceSun()
            sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
            scene.add_light_source(sun_1ray)
            trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

            # Add intersection point
            place_to_intersect = (AIMPOINT, target_plane_normal)
            facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            f_origin.append(facet_origin)
            is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
            intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

            # # Calculate Reflected Ray
            # sun = Vxyz(sp.sun_position(lln.NSTTF_ORIGIN, TIME))
            # surface_norm = rt.calc_reflected_ray(facet_normal[0], sun[0])
            # print(f"x: {surface_norm.x}, y: {surface_norm.y}, z: {surface_norm.z}")
            # x_sur_norm.append(surface_norm.x)
            # y_sur_norm.append(surface_norm.y)
            # z_sur_norm.append(surface_norm.z)

            # render control
            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

            # DRAWING
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(tile_array=(1, 1)),
                rca.meters(),
                vs.view_spec_3d(),
                title='Heliostat ' + heliostat_name + ', Facet ' + f_index,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index,
                comments="3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=80, current_len=250))
            figure_control.view.draw_single_Pxyz(AIMPOINT)
            figure_control.view.show(x_limits=[-50, 100], y_limits=[0, 170], z_limits=[-10, 70])
            self.show_save_and_check_figure(figure_control)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                title='Heliostat ' + heliostat_name + ', Facet ' + f_index,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index,
                comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
            )
            figure_control.x_limits = (-test_tolerance, test_tolerance)
            figure_control.z_limits = (63.5508 - test_tolerance, 63.5508 + test_tolerance)
            intersection_point.draw(figure_control.view)
            figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            self.show_save_and_check_figure(figure_control)

        test_heliostat.set_facet_canting(cantings)

        heliostat = test_heliostat.no_parent_copy()
        heliostat_loc = test_heliostat._self_to_parent_transform

        # # # FOR CREATING PHOTO OF CANTING ANGLES ALL  Set solar field scene
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

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_3d(),
            title='Off-Axis Canted Heliostat ' + heliostat_name + 'with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=250))
        figure_control.view.draw_single_Pxyz(AIMPOINT)
        figure_control.view.show(x_limits=[-50, 100], y_limits=[0, 170], z_limits=[-10, 80])
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        h_pointing_vector = UP.rotate(test_heliostat._self_to_parent_transform.R)
        test_heliostat.set_orientation_from_pointing_vector(h_pointing_vector)

        for facet in test_heliostat.facet_ensemble.facets:
            facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            print(f"x: {facet_normal.x}, y: {facet_normal.y}, z: {facet_normal.z}")
            x_sur_norm.append(facet_normal.x)
            y_sur_norm.append(facet_normal.y)
            z_sur_norm.append(facet_normal.z)

        # Prints all facet canting angles
        for facet in all_facets:
            print(facet)  # TODO mhh

        # saves canting angles to CSV
        self.save_to_csv(
            f_idx,
            f_origin,
            az,
            el,
            canted_x_angles,
            canted_y_angles,
            x_sur_norm,
            y_sur_norm,
            z_sur_norm,
            offset_z_found,
            offset_x_found,
        )

    def generate_csv(self):

        # Initialize test.
        self.start_test()

        # View Setup
        heliostat_name = "5W1"

        # Initial positions
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc
        UP = Vxyz([0, 0, 1])

        # Determine canting angles for single facet
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        self.set_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)

        azimuth = test_heliostat._az
        elevation = test_heliostat._el

        # facet_index = test_heliostat.facet_ensemble.children(dict)
        opencsp_dir = os.path.join(
            orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'input', 'TstMotionBasedCanting'
        )

        canted_x_angles = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'),
            'Canted X Angle Rotations (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'),
            'Canted Y Angle Rotations (rad)',
        )
        offset_x = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'), 'Offset X'
        )
        offset_z = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'), 'Offset Z'
        )

        # f_idx = self.read_csv(
        #     os.path.join(opencsp_dir, 'canting_angles_computed_9W1.csv'), 'Facet Index'
        # )
        f_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

        canting_rotations: list[TransformXYZ] = []

        # checks az and el values to see if facet surface normal for facet 13 intersects with given plane.
        f_index_temp = '13'

        # is_intersect, offset_x, offset_z, intersection_point = self.projected_facet_normal_intersection_point_offset(
        #     target_loc, target_plane_normal, heliostat_name, f_index_temp, az, el
        # )
        # az, el = self.doesnt_hit_plane_with_original_values(is_intersect, az, el)

        # set canting
        for facet in range(0, 25):
            # position = Pxyz.merge(test_heliostat._self_to_parent_transform.apply(Pxyz.origin()))
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
            # print(f"rotation: {canting.as_euler('xyz')}")
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
            heliostat_name,
            f_index,
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
        )

    def test_create_canted_heliostat(self) -> None:

        # Initialize test.
        self.start_test()

        # View Setup
        heliostat_name = "5W1"

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
            orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'input', 'TstMotionBasedCanting'
        )

        canted_x_angles = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'),
            'Canted X Angle Rotations (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'),
            'Canted Y Angle Rotations (rad)',
        )
        az = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'), 'Azimuth (rad)'
        )
        el = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'), 'Elevation (rad)'
        )

        f_idx = [test_heliostat.facet_ensemble.num_facets]

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        test_heliostat.set_facet_canting(canting_rotations)

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
        current_length = 90
        init = 75
        x_lim = [-10, 10]
        y_lim = [0, 60]
        z_lim = [-10, 70]
        count = int(0)

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
                title='Off-Axis Canted Heliostat ' + heliostat_name + ', Facet ' + facet.name + ' with Ray-Trace',
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ', Facet ' + facet.name,
                comments="3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
            figure_control.view.draw_single_Pxyz(AIMPOINT)
            figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
            self.show_save_and_check_figure(figure_control)

            count = int(count + 1)
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                number_in_name=False,
                input_prefix=self.figure_prefix(count),
                title='Off-Axis Canted Heliostat '
                + heliostat_name
                + ', Facet '
                + facet.name
                + " Intersection at Aimpoint",
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + ' facet ' + facet.name,
                comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
            )
            figure_control.x_limits = exaggerated_x
            figure_control.z_limits = exaggerated_z
            intersection_point.draw(figure_control.view)
            figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            self.show_save_and_check_figure(figure_control)

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
            title='Off-Axis Canted Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XY Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        figure_control.view.draw_single_Pxyz(AIMPOINT)
        figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xy(),
            number_in_name=False,
            input_prefix=self.figure_prefix(52),
            title='Off-Axis Canted Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XY Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        figure_control.view.draw_single_Pxyz(AIMPOINT)
        figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(53),
            title='Off-Axis Canted Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="YZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        figure_control.view.draw_single_Pxyz(AIMPOINT)
        figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile_array=(1, 1)),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(54),
            title='Off-Axis Canted Heliostat ' + heliostat_name + ' with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=init, current_len=current_length))
        figure_control.view.draw_single_Pxyz(AIMPOINT)
        figure_control.view.show(x_limits=x_lim, y_limits=y_lim, z_limits=z_lim)
        self.show_save_and_check_figure(figure_control)
        # figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_xz(),
            number_in_name=False,
            input_prefix=self.figure_prefix(55),
            title='Off-Axis Canted Heliostat ' + heliostat_name + " Intersection at Aimpoint All Facets",
            caption='A single Sandia NSTTF heliostat ' + heliostat_name,
            comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        figure_control.x_limits = exaggerated_x
        figure_control.z_limits = exaggerated_z
        intersection_point.draw(figure_control.view)
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        self.show_save_and_check_figure(figure_control)

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

    #     # Save figures.
    #     if save_figures:
    #         print('\n\nSaving figures...')
    #         # Output directory.
    #         output_path = os.path.join('..', ('output_' + datetime.now().strftime('%Y_%m_%d_%H%M')))
    #         if not(os.path.exists(output_path)):
    #             os.makedirs(output_path)
    #         fm.save_all_figures(output_path)

    #     if show_figures:
    #         input("Press 'Enter' to close the figures...")
    #         plt.close('all')

    def example_canting_calculated(self) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        heliostat_name = "5W1"

        # Define scenario.
        nsttf_pivot_height = 4.02  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_pivot_offset = 0.1778  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        focal_length_max = 210  # meters

        # # 5W01.
        # x_5W01 = -4.84  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        # y_5W01 = 57.93  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        # z_5W01 = 3.89  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        # focal_length_5W01 = 55  # meters
        # name_5W01 = '5W01'
        # title_5W01 = 'Canted NSTTF Heliostat ' + heliostat_name_1
        # caption_5W01 = '5W01 modeled as a symmetric paraboloid with focal length f=' + str(focal_length_5W01) + 'm.'
        # # 14W01.
        # x_14W01 = -4.88  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        # y_14W01 = 194.71  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        # z_14W01 = 4.54  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        # focal_length_14W01
        # = 186.8  # meters

        # heliostat z position and focal_length
        z = 3.89  # meters  # TODO RCB: FETCH FROM DEFINITION FILE
        focal_length = 55  # meters
        name = heliostat_name
        title = 'Off-Axis Canted NSTTF Heliostat ' + heliostat_name
        caption = heliostat_name + ' modeled as a symmetric paraboloid with focal length f=' + str(focal_length) + 'm.'
        # Solar field.
        short_name_sf = 'Mini NSTTF'
        name_sf = 'Mini NSTTF with Off-Axis Canted' + heliostat_name
        title_sf = 'Off-Axis Canted NSTTF Heliostat: ' + heliostat_name
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

        opencsp_dir = os.path.join(
            orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'input', 'TstMotionBasedCanting'
        )

        canted_x_angles = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'),
            'Canted X Angle Rotations (rad)',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(opencsp_dir, 'canting_angles_computed_' + heliostat_name + '.csv'),
            'Canted Y Angle Rotations (rad)',
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, 25):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)

        h1.set_facet_canting(canting_rotations)

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
            input_prefix=self.figure_prefix(56),
            title=title + ' (long normals, 9 for each facet)',
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_long)
        self.show_save_and_check_figure(fig_record)

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
            input_prefix=self.figure_prefix(57),
            title=title + ' (long normals, 1 for each facet)',
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_long_1)
        self.show_save_and_check_figure(fig_record)

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
            input_prefix=self.figure_prefix(58),
            title=title_sf + ' (very long normals)',
            caption=caption_sf,
            comments=comments,
        )
        sf.draw(fig_record.view, solar_field_control_very_long)
        self.show_save_and_check_figure(fig_record)

        # Setup render control (exaggerated z).
        z_exaggerated_margin = 0.35  # meters, plus or minus reference height.
        decimal_factor = 100.0
        # Different z limits for each heliostat, because they are at different elevations on the sloped field.
        z_min = np.floor(decimal_factor * ((z + nsttf_pivot_offset) - z_exaggerated_margin)) / decimal_factor
        z_max = np.ceil(decimal_factor * ((z + nsttf_pivot_offset) + z_exaggerated_margin)) / decimal_factor
        exaggerated_z_limits = [z_min, z_max]

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
            input_prefix=self.figure_prefix(59),
            name=title + ' (exaggerated z)',
            title=title,
            caption=caption,
            comments=comments,
        )
        fig_record.z_limits = exaggerated_z_limits
        h1.draw(fig_record.view, heliostat_control_exaggerated_z)
        self.show_save_and_check_figure(fig_record)

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
            input_prefix=self.figure_prefix(60),
            name=title + ' with normals',
            title=title + ' with normals',
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z_normal)
        self.show_save_and_check_figure(fig_record)

        # Draw and output 5W01 figure (exaggerated z, yz view).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(61),
            name=title + ' exaggerated z',
            title=title,
            caption=caption,
            comments=comments,
        )
        fig_record.equal = False  # Asserting equal axis scales contradicts exaggerated z limits in 2-d plots.
        fig_record.z_limits = (
            exaggerated_z_limits  # Limits are on z values, even though the plot is 2-d.  View3d.py handles this.
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z)
        self.show_save_and_check_figure(fig_record)

        # Draw and output 5W01 figure (exaggerated z, yz view with normals).
        fig_record = fm.setup_figure_for_3d_data(
            self.figure_control,
            rca.meters(),
            vs.view_spec_yz(),
            number_in_name=False,
            # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
            input_prefix=self.figure_prefix(62),
            name=title + ' with normals',
            title=title + ' with normals',
            caption=caption,
            comments=comments,
        )
        h1.draw(fig_record.view, heliostat_control_exaggerated_z_normal)
        self.show_save_and_check_figure(fig_record)

    def example_canting_bar_charts_calculated(self) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        heliostat_name = "5W1"

        # Define scenario.
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE

        focal_length = 55  # meters
        # title = 'Off-Axis Canted NSTTF Heliostat ' + heliostat_name
        # caption = heliostat_name + ' modeled as a symmetric paraboloid with focal length f=' + str(focal_length) + 'm.'

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

        opencsp_dir = os.path.join(
            orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'input', 'TstMotionBasedCanting'
        )
        canted_x_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'),
            'Face Up Canting Rotation about X',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'),
            'Face Up Canting Rotation about Y',
        )
        anglesx = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal X'
        )
        anglesy = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal Y'
        )
        anglesz = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal Z'
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, h1.facet_ensemble.num_facets):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)

        h1.set_facet_canting(canting_rotations)

        ####plot canting angles bar chart

        canting_angles = {}
        # ?? SCAFFOLDING RCB -- THIS IS CRUFTY.  THIS SHOULD GO IN AN ANALYSIS PLACE, AND ONLY COMPUTE ONCE.  SEE ALSO SIMILAR CODE IN PLOT NORMALS FUNCTION.
        # Compute average canting angle x and y components.
        average_ax = sum(anglesx) / len(anglesx)
        average_ay = sum(anglesy) / len(anglesy)
        # Compute offsets.
        offset_anglesx = [(x - average_ax) for x in anglesx]
        offset_anglesy = [(y - average_ay) for y in anglesy]
        offset_anglesz = []
        for facet in range(h1.facet_ensemble.num_facets):
            canting_angles[facet] = [anglesx[facet], anglesy[facet], anglesz[facet]]

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

        y_limits = [0.03, -0.03]

        # ax = df.plot.bar(rot=0, color={"Nx": "tab:blue"}, figsize=(15, 10))
        ##y surface normal plot
        # fig_record = fm.setup_figure(
        #     rcfg.RenderControlFigure(),
        #     self.axis_control_m,
        #     vs.view_spec_xy(),
        #     # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
        #     input_prefix=self.figure_prefix(0),
        #     name=title + ' exaggerated z',
        #     title=title,
        #     caption=caption,
        #     comments=comments,
        #     code_tag=self.code_tag,
        # )
        # fig_record.equal = False  # Asserting equal axis scales contradicts exaggerated z limits in 2-d plots.
        # fig_record.y_limits = (
        #     y_limits  # Limits are on z values, even though the plot is 2-d.  View3d.py handles this.
        # )
        # ax.draw(fig_record.view, bar_control)
        # self.show_save_and_check_figure(fig_record)

        df = pd.DataFrame({'Nx': anglesx}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Nx": "tab:blue"}, figsize=(15, 10))
        title = heliostat_name + ': Canted Facet Normal X Component'
        plt.title(title)
        plt.xlabel('Facet id')
        plt.ylabel('X-component in units of Surface Normal')
        y_axis_max = 0.030  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        plt.ylim(
            -y_axis_max, y_axis_max
        )  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        #        plt.ylim(min(anglesx)-0.005, max(anglesx)+0.005)
        plt.grid(axis='y')
        figure_name = 'tca063_' + heliostat_name + '_canglesX_' + '.png'
        plt.savefig(opencsp_dir + '/' + figure_name)
        plt.close()

        df = pd.DataFrame({'Ny': anglesy}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Ny": "tab:orange"}, figsize=(15, 10))
        title = heliostat_name + ': Canted Facet Normal Y Component'
        plt.title(title)
        plt.xlabel('Facet id')
        plt.ylabel('Y-component in units of surface normal')
        plt.ylim(
            -y_axis_max, y_axis_max
        )  # ?? SCAFFOLDING RCB -- THIS MAKES SCALES SIMILAR ACROSS HELIOSTATS, BUT WILL BE TOO SMALL FOR SOME EXTREME ERRORS, AND ALSO DOESN'T ADJUST FOR DIFFERENT SOALR FIELDS.  RECTIFY THIS.
        #        plt.ylim(min(anglesy)-0.001, max(anglesy)+0.001)
        plt.grid(axis='y')
        figure_name = 'tca064_' + heliostat_name + '_canglesY_' + '.png'
        plt.savefig(opencsp_dir + '/' + figure_name)
        plt.close()

        df = pd.DataFrame({'Nz': anglesz}, index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)])
        ax = df.plot.bar(rot=0, color={"Nz": "magenta"}, figsize=(15, 10))
        title = heliostat_name + ': Canted Facet Normal X Component'
        plt.title(title)
        plt.xlabel('Facet id')
        plt.ylabel('Z-component in units of Surface Normal')
        plt.ylim(min(anglesz) - 0.000001, 1)
        plt.grid(axis='y')
        figure_name = 'tca065_' + heliostat_name + '_canglesZ_' + '.png'
        plt.savefig(opencsp_dir + '/' + figure_name)
        plt.close()

        # plt.show()
        plt.close()

    def example_canting_square_diagonal_offset_calculated(self) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        heliostat_name = "9W1"

        # Define scenario.
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE

        focal_length = 102  # meters
        # title = 'Off-Axis Canted NSTTF Heliostat ' + heliostat_name
        # caption = heliostat_name + ' modeled as a symmetric paraboloid with focal length f=' + str(focal_length) + 'm.'

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

        opencsp_dir = os.path.join(
            orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'input', 'TstMotionBasedCanting'
        )
        canted_x_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'),
            'Face Up Canting Rotation about X',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'),
            'Face Up Canting Rotation about Y',
        )
        anglesx = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal X'
        )
        anglesy = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal Y'
        )
        anglesz = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal Z'
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, h1.facet_ensemble.num_facets):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)

        h1.set_facet_canting(canting_rotations)

        diagonals_distance = []
        diagonals_distance_dict = {}

        for facet in h1.facet_ensemble.facets:
            left, right, bottom, top = facet.axis_aligned_bounding_box
            origin, _ = facet.survey_of_points(Resolution.center())
            top_left_corner = Pxyz([[origin.x + left], [origin.y + top], [origin.z]])
            top_right_corner = Pxyz([[origin.x + right], [origin.y + top], [origin.z]])
            bottom_left_corner = Pxyz([[origin.x + left], [origin.y + bottom], [origin.z]])
            bottom_right_corner = Pxyz([[origin.x + right], [origin.y + bottom], [origin.z]])
            # diagonals direction vectors
            e1 = Vxyz.__sub__(bottom_right_corner, top_left_corner)
            e2 = Vxyz.__sub__(bottom_left_corner, top_right_corner)
            n = Vxyz.cross(e1, e2)
            r1r2 = Vxyz.__sub__(top_left_corner, top_right_corner)
            distance = np.dot(n, r1r2) / np.linalg.norm(n)
            diagonals_distance.append(abs(distance))
            diagonals_distance_dict[facet.name] = abs(distance)

        df = pd.DataFrame(
            {'Diagonal Offset Distance': diagonals_distance},
            index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)],
        )
        ax = df.plot.bar(rot=0, color={"Diagonal Offset Distance": "tab:blue"})
        title = heliostat_name + ': Diagonal Offset'
        figure_name = 'tca066_' + heliostat_name + '_diagonal_offset_' + '.png'
        plt.title(title)
        plt.xlabel('Facet id')
        plt.ylabel('Y (m)')
        plt.savefig(opencsp_dir + '/' + figure_name)
        plt.close()

    def example_canting_square_sides_quality_calculated(self) -> None:
        """
        Displays two NSTTF heliostats with an approximate, realistic focal length.
        """
        # Initialize test.
        self.start_test()

        heliostat_name = "9W1"

        # Define scenario.
        nsttf_facet_width = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE
        nsttf_facet_height = 1.2192  # TODO RCB: FETCH FROM DEFINITION FILE

        focal_length = 102  # meters
        # title = 'Off-Axis Canted NSTTF Heliostat ' + heliostat_name
        # caption = heliostat_name + ' modeled as a symmetric paraboloid with focal length f=' + str(focal_length) + 'm.'

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

        opencsp_dir = os.path.join(
            orp.opencsp_code_dir(), 'common', 'lib', 'test', 'data', 'input', 'TstMotionBasedCanting'
        )
        canted_x_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'),
            'Face Up Canting Rotation about X',
        )
        canted_y_angles = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'),
            'Face Up Canting Rotation about Y',
        )
        anglesx = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal X'
        )
        anglesy = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal Y'
        )
        anglesz = self.read_csv_float(
            os.path.join(opencsp_dir, heliostat_name + '_facet_details_canted_off_axis.csv'), 'Face Up Surface Normal Z'
        )

        canting_rotations: list[TransformXYZ] = []

        # set canting
        for facet in range(0, h1.facet_ensemble.num_facets):
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)

        h1.set_facet_canting(canting_rotations)
        left, right, bottom, top = h1.facet_ensemble.axis_aligned_bounding_box
        facet_ensemble_corners = Pxyz([[left, left, right, right], [top, bottom, bottom, top], [0, 0, 0, 0]])

        rmse = 0
        residual = 0
        top_side, top_error = [], []
        right_side, right_error = [], []
        bottom_side, bottom_error = [], []
        left_side, left_error = [], []
        square_sides_errors = {}
        for facet in h1.facet_ensemble.facets:
            left, right, bottom, top = facet.axis_aligned_bounding_box
            left_side.append(left)
            left_error.append(top - nsttf_facet_height)
            right_side.append(right)
            right_error.append(right - nsttf_facet_height)
            bottom_side.append(bottom)
            bottom_error.append(bottom - nsttf_facet_width)
            top_side.append(top)
            top_error.append(top - nsttf_facet_width)

            square_sides_errors[facet.name] = [
                top - nsttf_facet_width,
                right - nsttf_facet_height,
                bottom - nsttf_facet_width,
                left - nsttf_facet_height,
            ]

            rmse += (
                (top - nsttf_facet_width) ** 2
                + (right - nsttf_facet_height) ** 2
                + (bottom - nsttf_facet_width) ** 2
                + (left - nsttf_facet_height) ** 2
            )
            mean_error = (
                abs(top - nsttf_facet_width)
                + abs(right - nsttf_facet_height)
                + abs(bottom - nsttf_facet_width)
                + abs(left - nsttf_facet_height)
            )

        rmse /= len(facet_ensemble_corners)
        mean_error /= len(facet_ensemble_corners)
        rmse = np.sqrt(rmse)
        df = pd.DataFrame(
            {
                'Top Side Error': top_error,
                'Bottom Side Error': bottom_error,
                'Right Side Error': right_error,
                'Left Side Error': left_error,
            },
            index=[i + 1 for i in range(0, h1.facet_ensemble.num_facets)],
        )
        ax = df.plot.bar(
            rot=0,
            color={
                "Top Side Error": "tab:blue",
                "Right Side Error": "tab:orange",
                "Bottom Side Error": "tab:olive",
                "Left Side Error": "tab:purple",
            },
            figsize=(15, 10),
        )
        title = heliostat_name + ': Square Sides Quality'
        plt.title(title)
        plt.xlabel('Facet id')
        plt.ylabel('Meters')
        bottom = min([min(top_error), min(right_error), min(bottom_error), min(left_error)])
        top = max([max(top_error), max(right_error), max(bottom_error), max(left_error)])
        x = [indx for indx in range(-1, h1.facet_ensemble.num_facets + 1)]
        y = [rmse for _ in range(-1, h1.facet_ensemble.num_facets + 1)]
        plt.plot(x, y, linewidth=1.5, color='red', linestyle='dashed')
        y = [-rmse for _ in range(-1, h1.facet_ensemble.num_facets + 1)]
        plt.plot(x, y, linewidth=1.5, color='red', linestyle='dashed', label='RMSE')
        plt.ylim(bottom + (10e-2 * bottom), top + 10e-2 * top)
        plt.grid(axis='y')
        plt.legend()
        # plt.show()
        figure_name = heliostat_name + '_square_sides_errors' + '.png'
        plt.savefig(opencsp_dir + '/' + figure_name)
        plt.close()


# MAIN EXECUTION

if __name__ == "__main__":
    # Control flags.
    interactive = False
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
    #     tolerance=tolerance_1,
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
    # test_object.test_off_axis_code()
    # test_object.test_find_all_canting_angles()
    # test_object.test_create_canted_heliostat()
    # test_object.example_canting_calculated()
    # test_object.example_canting_bar_charts_calculated()
    # test_object.example_canting_square_sides_quality_calculated()
    test_object.example_canting_square_diagonal_offset_calculated()
    # test_object.generate_csv()

    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
