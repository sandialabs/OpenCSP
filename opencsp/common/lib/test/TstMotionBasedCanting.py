"""
Demonstrate Motion Based Canting Experiment.

Copyright (c) 2021 Sandia National Laboratories.

"""

import copy
import csv as csv
import os
import sys as sys
from datetime import datetime
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
from sympy import Symbol, diff

import opencsp.common.lib.csp.HeliostatAzEl as Heliostat
import opencsp.common.lib.csp.HeliostatConfiguration as hc
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
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlRayTrace as rcrt
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.render_control.RenderControlTower as rct
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.HeliostatAbstract import HeliostatAbstract
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
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
        figure_prefix_root: str = 'tmbc',  # setUpClass() with no arguments.
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

        # Computer offset from target
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

    def better_tracking(self, heliostat, aimpoint: Pxyz, location_lon_lat: Iterable, when_ymdhmsz: tuple):
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
        self.better_tracking(heliostat, aimpoint, lln.NSTTF_ORIGIN, time)
        # az = heliostat._az
        # el = heliostat._el
        # print(f"az and el = {az}, {el}")

        cantings: list[Rotation] = []

        UP = Vxyz([0, 0, 1])

        ### Unnecessary code and computation since heliostat in new classes remembers orientation
        # R2 = heliostat.facet_ensemble.self_to_global_tranformation.R
        # R2_inv = R2.inv()
        # h_pointing_vector = UP.rotate(heliostat.self_to_global_tranformation.R)
        ###

        ### test to see if it works with other facets besides 13
        test_facet = heliostat.facet_ensemble.lookup_facet("3")
        test_facet_origin, test_facet_normal = test_facet.survey_of_points(Resolution.center())
        # test_facet_origin = test_facet.self_to_global_tranformation.apply(Pxyz.origin())     ## same thing as survey of points
        vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz(test_facet_origin, aimpoint, long_lat, time)
        ###

        for facet in heliostat.facet_ensemble.facets:
            # facet_origin = facet.self_to_global_tranformation.apply(Pxyz.origin())     ## same thing as survey of points
            facet_origin, facet_normal = facet.survey_of_points(Resolution.center())
            # h_pointing_vector = UP.rotate(facet.self_to_global_tranformation.R)  ## same thing as facet_normal
            # vector_the_normal_should_be = sun_track.tracking_surface_normal_xyz(facet_origin, aimpoint, long_lat, time)
            R1_prime = facet_normal.align_to(vector_the_normal_should_be)
            # cantings.append(R2_inv * R1_prime * R2)    ## unnecessary code
            canting_rotations = R1_prime
            cantings.append(canting_rotations)

        heliostat.facet_ensemble.set_facet_canting(cantings)

    def save_to_csv(
        self, f_index, f_origin, azimuth, elevation, canted_x_angles, canted_y_angles, offset_z, offset_x
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
            if facet < len(f_origin):
                row.append(f_origin[facet])
            else:
                row.append('')
            row.append(azimuth)
            row.append(elevation)
            if facet < len(canted_x_angles):
                row.append(canted_x_angles[facet])
            else:
                row.append('')  # If there are fewer angles than facets, fill with empty string
            if facet < len(canted_y_angles):
                row.append(canted_y_angles[facet])
            else:
                row.append('')
            if facet < len(offset_z):
                row.append(offset_z[facet])
            else:
                row.append('')
            if facet < len(offset_x):
                row.append(offset_x[facet])
            else:
                row.append('')
            rows.append(row)

        file_name = 'canting_angles_computed.csv'
        output_directory = os.path.join(self.output_path, 'data', 'output', self.source_file_body)
        file_path = os.path.join(output_directory, file_name)

        os.makedirs(output_directory, exist_ok=True)

        # Write to CSV file
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    'Facet Index',
                    'Facet Origin',
                    'Azimuth (rad)',
                    'Elevation (rad)',
                    'Canted X Angle Rotations (rad)',
                    'Canted Y Angle Rotations (rad)',
                    'Offset Z',
                    'Offset X',
                ]
            )
            writer.writerows(rows)

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
        az = np.deg2rad(178.34515)
        el = np.deg2rad(43.03037)
        test_tolerance = 0.001
        tower_control = rct.normal_tower()

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc

        # FOCAL_LENGTH = 75.26

        # curved_equation = FunctionXYContinuous(f"(x**2 + y**2)/(4*{FOCAL_LENGTH})")

        # mirror_template = MirrorParametricRectangular(curved_equation, (1.2, 1.2))
        # h5W1, h5W1_location = HeliostatAzEl.from_csv_files(
        #     "5W1",
        #     dpft.sandia_nsttf_test_heliostats_origin_file(),
        #     dpft.sandia_nsttf_test_facet_centroidsfile(),
        #     mirror_template,
        # )
        # h5W1.set_canting_from_equation(curved_equation)

        # Configuration setup
        solar_field = self.solar_field
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        HeliostatAzEl.set_orientation_from_az_el(test_heliostat, az, el)
        # test_heliostat.set_orientation(test_config)
        test_heliostat_loc = test_heliostat._self_to_parent_transform
        test_heliostat = test_heliostat.no_parent_copy()

        scene = Scene()
        scene.add_object(test_heliostat)
        scene.set_position_in_space(test_heliostat, test_heliostat_loc)

        # set canting
        self.better_tracking(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        az = test_heliostat._az
        el = test_heliostat._el
        print(f"az and el = {az}, {el}")
        # test_heliostat.set_tracking_configuration(AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        # self.off_axis_canting(test_heliostat, AIMPOINT, lln.NSTTF_ORIGIN, TIME)
        # test_heliostat.set_orientation(test_config)

        # facet = self.cant_single_facet(heliostat_name, f_index, az, el, canted_x_angle, canted_y_angle)
        # facet_no_parent = facet.no_parent_copy()
        # facet_location = facet.self_to_global_tranformation

        # # Set canting
        # position = Pxyz.merge(facet._self_to_parent_transform.apply(Pxyz.origin()))
        # canting = Rotation.from_euler('xyz', [canted_x_angle, canted_y_angle, 0], degrees=False)
        # facet._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)
        # facet_no_parent = facet.no_parent_copy()
        # Set solar field scene
        # scene = Scene()
        # scene.add_object(test_heliostat)
        # scene.set_position_in_space(test_heliostat, test_heliostat_loc)
        # solar_field.set_full_field_tracking(AIMPOINT, TIME)

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

        # # DRAWING
        # figure_control = fm.setup_figure_for_3d_data(
        #     rcfg.RenderControlFigure(tile=True, tile_array=(4, 2), grid=False),
        #     rca.meters(grid=False),
        #     vs.view_spec_3d(),
        #     title=f"9W1 Ray Trace 3D through Target",
        # )
        # test_heliostat.draw(figure_control.view, heliostat_style)
        # tower.draw(figure_control.view, tower_control)
        # trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=120))
        # figure_control.view.draw_single_Pxyz(AIMPOINT)
        # figure_control.view.show(x_limits=[-40, 40], y_limits=[0, 80], z_limits=[-10, 80], grid=False)
        # self.show_save_and_check_figure(figure_control)

        # figure_control = fm.setup_figure_for_3d_data(
        #     rcfg.RenderControlFigure(),
        #     rca.meters(),
        #     vs.view_spec_xz(),
        #     title=f"9W1 Ray Trace Intersection at Aimpoint XZ View",
        # )
        # figure_control.equal = False
        # figure_control.x_limits = (-test_tolerance, test_tolerance)
        # figure_control.z_limits = (63.5508 - test_tolerance, 63.5508 + test_tolerance)
        # intersection_point.draw(figure_control.view)
        # figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        # self.show_save_and_check_figure(figure_control)

        # trace = rt.trace_scene(scene, Resolution.center())
        # intersection = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        # DRAWING
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(4, 2), grid=False),
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
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=300))
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(), rca.meters(), vs.view_spec_xz(), title="5W1 Ray Trace Birds View"
        )
        test_heliostat.draw(figure_control.view)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=300))
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        figure_control.view.show()

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(), rca.meters(), vs.view_spec_yz(), title="5W1 Ray Trace Birds View"
        )
        test_heliostat.draw(figure_control.view)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=300))
        figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
        figure_control.view.show()

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
        el = np.deg2rad(43.03037)
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

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
        heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

        # Comment
        comments.append(
            "Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower."
        )

        # DRAWING
        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(tile=True, tile_array=(4, 2)),
            rca.meters(),
            vs.view_spec_3d(),
            title=f"9W1 Ray Trace 3D through Target",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=120))
        figure_control.view.draw_single_Pxyz(AIMPOINT)
        figure_control.view.show(x_limits=[-40, 40], y_limits=[0, 80], z_limits=[-10, 80])
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
        heliostat_name = "9W1"

        # Initial positions
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        az = np.deg2rad(178.34515)
        el = np.deg2rad(43.03037)
        test_tolerance = 0.001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc

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
        f_origin = []
        f_idx = []

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

            facet_ensemble_control = rcfe.facet_outlines_thin()
            heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)
            heliostat_style.facet_ensemble_style.add_special_style(f_index, rcf.outline('blue'))

            # DRAWING
            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_3d(),
                title='Heliostat ' + heliostat_name + ', Facet ' + f_index,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index,
                comments="3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
            )
            test_heliostat.draw(figure_control.view, heliostat_style)
            tower.draw(figure_control.view, tower_control)
            trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=80, current_len=120))
            figure_control.view.draw_single_Pxyz(AIMPOINT)
            figure_control.view.show(x_limits=[-40, 40], y_limits=[0, 80], z_limits=[-10, 70])
            self.show_save_and_check_figure(figure_control)

            figure_control = fm.setup_figure_for_3d_data(
                rcfg.RenderControlFigure(),
                rca.meters(),
                vs.view_spec_xz(),
                title='Heliostat ' + heliostat_name + ', Facet ' + f_index,
                caption='A single Sandia NSTTF heliostat ' + heliostat_name + 'facet' + f_index,
                comments="XZ Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
            )
            intersection_point.draw(figure_control.view)
            figure_control.view.draw_single_Pxyz(AIMPOINT, rcps.marker(color='r'))
            self.show_save_and_check_figure(figure_control)

        # Prints all facet canting angles
        for facet in all_facets:
            print(facet)  # TODO mhh

        # saves canting angles to CSV
        self.save_to_csv(f_idx, f_origin, az, el, canted_x_angles, canted_y_angles, offset_z_found, offset_x_found)

    def test_create_canted_heliostat(self):

        # Initialize test.
        self.start_test()

        # View Setup
        heliostat_name = "9W1"

        # Initial positions
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        az = np.deg2rad(178.4312)
        el = np.deg2rad(49.9287)
        test_tolerance = 0.001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 2
        test_canted_y_angle = 0

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc

        # Determine canting angles for single facet
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)

        # facet_index = test_heliostat.facet_ensemble.children(dict)
        canted_x_angles = [
            0.529907227,
            0.531005859,
            0.531494141,
            0.531738281,
            0.531860352,
            0.266845703,
            0.267944336,
            0.268554688,
            0.268798828,
            0.268798828,
            -0.001464844,
            -0.000610352,
            -6.10e-05,
            0.000244141,
            0.000366211,
            -0.270996094,
            -0.270080566,
            -0.269592285,
            -0.269348145,
            -0.269226074,
            -0.535339355,
            -0.534484863,
            -0.533996582,
            -0.533752441,
            -0.533630371,
        ]
        canted_y_angles = [
            0.584228516,
            0.265258789,
            -0.000976563,
            -0.267211914,
            -0.586547852,
            0.585205078,
            0.265869141,
            -0.000488281,
            -0.266723633,
            -0.586547852,
            0.5859375,
            0.266479492,
            0.00012207,
            -0.266601563,
            -0.586181641,
            0.586425781,
            0.266967773,
            0.000488281,
            -0.266235352,
            -0.58605957,
            0.587158203,
            0.267456055,
            0.000732422,
            -0.265991211,
            -0.58605957,
        ]
        f_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

        canting_rotations: list[TransformXYZ] = []

        # checks az and el values to see if facet surface normal for facet 13 intersects with given plane.
        f_index_temp = '13'

        is_intersect, offset_x, offset_z, intersection_point = self.projected_facet_normal_intersection_point_offset(
            target_loc, target_plane_normal, heliostat_name, f_index_temp, az, el
        )
        az, el = self.doesnt_hit_plane_with_original_values(is_intersect, az, el)

        # set canting
        for facet in range(0, 25):
            # position = Pxyz.merge(test_heliostat._self_to_parent_transform.apply(Pxyz.origin()))
            canting = Rotation.from_euler('xyz', [canted_x_angles[facet], canted_y_angles[facet], 0], degrees=False)
            canting_rotations.append(canting)
        # test_heliostat._self_to_parent_transform = TransformXYZ.from_R_V(canting, position)
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

        # Set solar field scene
        scene = Scene()
        scene.add_object(heliostat)
        scene.set_position_in_space(heliostat, heliostat_loc)
        solar_field.set_full_field_tracking(AIMPOINT, TIME)

        # Add ray trace to field
        sun_1ray = LightSourceSun()
        sun_1ray.set_incident_rays(lln.NSTTF_ORIGIN, TIME, 1)
        scene.add_light_source(sun_1ray)
        trace = rt.trace_scene(scene, Resolution.center(), verbose=True)

        # Add intersection point
        place_to_intersect = (AIMPOINT, target_plane_normal)
        # heliostat_origin, heliostat_normal = test_heliostat.survey_of_points(Resolution.center())
        # is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, heliostat_normal)
        intersection_point = Intersection.plane_intersect_from_ray_trace(trace, place_to_intersect)

        facet_ensemble_control = rcfe.facet_outlines_thin()
        heliostat_style = rch.RenderControlHeliostat(facet_ensemble_style=facet_ensemble_control)

        figure_control = fm.setup_figure_for_3d_data(
            rcfg.RenderControlFigure(),
            rca.meters(),
            vs.view_spec_3d(),
            title='Off-Axis Canted Heliostat ' + heliostat_name + 'with Ray-Trace',
            caption='A single Sandia NSTTF heliostat ',
            comments="3D Ray trace at NSTTF spring equinox solar noon from center of heliostat facet to target on tower.",
        )
        test_heliostat.draw(figure_control.view, heliostat_style)
        tower.draw(figure_control.view, tower_control)
        trace.draw(figure_control.view, rcrt.init_current_lengths(init_len=100, current_len=120))
        figure_control.view.draw_single_Pxyz(AIMPOINT)
        figure_control.view.show(x_limits=[-40, 40], y_limits=[0, 80], z_limits=[-10, 80])
        self.show_save_and_check_figure(figure_control)

    def test_generate_plots(self):
        file_name = 'generating plots'
        heliostat = FacetEnsemble.read_txt_file_to_heliostat(filename=file_name)
        output_directory = os.path.join(self.output_path, 'data', 'output', self.source_file_body)
        file_path = os.path.join(output_directory, file_name)
        FacetEnsemble.generate_plots(file_name, output_directory, heliostat)

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

    def test_facet_ensemble(self):

        left, right, top, bottom = 1.2, 1.3, 1.4, 1.5
        corners = Pxyz([[left, left, right, right], [top, bottom, bottom, top], [0, 0, 0, 0]])
        print(f"corners length {len(corners)}")
        corners_per_heliostat = (len(corners)) * 25
        print(f"corners length per helio {corners_per_heliostat}")
        print(f"division for single / {25/2}")
        print(f"division for double // {(1+24)//2}")

    def test_get_width(self):

        heliostat_name = "9W1"

        # Initial positions
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc = tower.target_loc
        target_plane_normal = Uxyz([0, 1, 0])
        az = np.deg2rad(178.34515)
        el = np.deg2rad(43.03037)
        test_tolerance = 0.001
        tower_control = rct.normal_tower()
        test_canted_x_angle = 0
        test_canted_y_angle = 0
        f_index = '1'
        facet_canted_x_angle = 0.005
        facet_canted_y_angle = 0.005

        # Define tracking time
        # https://gml.noaa.gov/grad/solcalc/ using NSTTF lon: -106.509 and lat: 34.96
        TIME = (2024, 3, 21, 13, 13, 5, -6)  # NSTTF spring equinox, solar noon
        AIMPOINT = target_loc

        # Determine canting angles for single facet
        test_config = hc.HeliostatConfiguration("az-el", az, el)
        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)
        facet = test_heliostat.facet_ensemble.lookup_facet(f_index)

        # facet = self.cant_single_facet(heliostat_name, f_index, az, el, facet_canted_x_angle, facet_canted_y_angle)

        width, height = Facet.get_2D_dimensions(facet)

        print(f"width: {width}, heigh: {height}")

    def test_2D(self):
        theoretical_flat_heliostat_dir_body_ext = (
            experiment_dir()
            + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/010_HeliostatModel/data/NSTTF_Facet_Corners.csv'
        )
        output_heliostat_3d_dir = (
            experiment_dir()
            + '2020-12-03_FastScan1/9_Analysis/2021-10-02_Study_Heliostat3dInfer/01_Study_5W1_solo/output_reconstructed_heliostats/5W1all_confirmed_undistorted_corners_3d.txt'
        )
        output_directory = os.path.join(self.output_path, 'data', 'output', self.source_file_body)

        heliostat = FacetEnsemble.read_txt_file_to_heliostat(filename=theoretical_flat_heliostat_dir_body_ext)
        heliostat = FacetEnsemble.read_txt_file_to_heliostat(filename=output_heliostat_3d_dir)
        if heliostat is None:
            return None
        hel_name = FacetEnsemble.heliostat_name_given_heliostat_3d_dir_body_ext(
            filename=output_heliostat_3d_dir
        )  # ?? SCAFFOLDING RCB -- CRUFTY.  FIX THIS.
        heliostat_path = output_directory  # ?? SCAFFOLDING RCB -- CLEAN THIS UP
        ft.create_directories_if_necessary(heliostat_path)
        FacetEnsemble.plot_heliostat_2d(
            heliostat,
            heliostat_theoretical_dict=theoretical_flat_heliostat_dir_body_ext,
            theoretical_flag=True,
            plot_surface_normals=False,
            saving_path=heliostat_path,
            hel_name=hel_name,
            title_prefix='2d',
            option='noRotate',
        )


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
    test_object.test_off_axis_code()
    # test_object.test_find_all_canting_angles()
    # test_object.test_create_canted_heliostat()
    # test_object.test_generate_plots()
    # test_object.test_facet_ensemble()
    # test_object.test_get_width()

    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
