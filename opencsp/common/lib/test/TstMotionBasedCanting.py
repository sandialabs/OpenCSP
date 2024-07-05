"""
Demonstrate Motion Based Canting Experiment.

Copyright (c) 2021 Sandia National Laboratories.

"""

import copy
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import sys as sys

from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.geometry.FunctionXYContinuous import FunctionXYContinuous
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.csp.LightSourcePoint import LightSourcePoint
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.HeliostatAzEl as Heliostat
import opencsp.common.lib.csp.HeliostatConfiguration as hc
import opencsp.common.lib.csp.RayTrace as rt
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.opencsp_path.data_path_for_test as dpft
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf
import opencsp.common.lib.render_control.RenderControlTower as rct
import opencsp.common.lib.test.support_test as stest
import opencsp.common.lib.test.TestOutput as to
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.geometry.Intersection import Intersection
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.Scene import Scene
from opencsp.common.lib.csp.SolarField import SolarField
from opencsp.common.lib.csp.Tower import Tower
from opencsp.common.lib.csp.HeliostatConfiguration import HeliostatConfiguration
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render_control.RenderControlAxis import RenderControlAxis
from opencsp.common.lib.render_control.RenderControlFigure import RenderControlFigure
from opencsp.common.lib.render_control.RenderControlFigureRecord import RenderControlFigureRecord
from opencsp.common.lib.render_control.RenderControlLightPath import RenderControlLightPath
from opencsp.common.lib.render_control.RenderControlRayTrace import RenderControlRayTrace
from opencsp.common.lib.render_control.RenderControlFacetEnsemble import RenderControlFacetEnsemble
from opencsp.common.lib.render_control.RenderControlFacet import RenderControlFacet
from opencsp.common.lib.render_control.RenderControlFacet import RenderControlFacet


class TestMotionBasedCanting(to.TestOutput):

    @classmethod
    def setUpClass(
        self,
        source_file_body: str = 'TestMotionBasedCanting',  # Set these here, because pytest calls
        figure_prefix_root: str = 'tmbc',  # setUpClass() with no arguments.
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
        # divide distance from jump by 2

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
        test_config = hc.HeliostatConfiguration("az-el", az, el)

        solar_field = self.solar_field
        test_heliostat = solar_field.lookup_heliostat(heliostat_name)
        test_heliostat.set_orientation(test_config)

        facet = test_heliostat.lookup_facet(f_index)

        facet_origin, facet_normal = facet.survey_of_points(Resolution.center())

        is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
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
        for i in range(1, 21):
            if Pxyz.distance(intersection_point, target_loc) > tolerance:
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
                    "In TestMotionBasedCanting.find_single_facet_azimuth_el_value:  azimuth = ",
                    az,
                    "     elevation = ",
                    el,
                )  # TODO MHH
                if i <= 21:
                    print(
                        "In TestMotionBasedCanting.find_single_facet_azimuth_el_value: In tolerance = ",
                        tolerance,
                        "distance = ",
                        Pxyz.distance(intersection_point, target_loc),
                        "azimuth = ",
                        az,
                        "     elevation = ",
                        el,
                    )  # TODO MHH

                    break

            # else: #Pxyz.distance(intersection_point, target_loc) <= tolerance:
            #     print("In TestMotionBasedCanting.find_single_facet_azimuth_el_value: Values already in tolerance = ", tolerance, "distance = ", Pxyz.distance(intersection_point, target_loc), "azimuth = ",  az, "     elevation = ",el) #TODO MHH

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

    # def reflected_facet_intersection_point_offset(self,
    #     target_loc: 'Pxyz',
    #     target_plane_normal: 'Uxyz',
    #     heliostat_name: str,
    #     f_index: int,
    #     az: float,
    #     el: float,
    #     tolerance: float,):
    #     """
    #     Returns reflected x and y signed offsets from intersection point to target location on the same plane.

    #     Parameters
    #     ----------
    #     target_loc: xyz location of target on given plane
    #     target_plane_normal: unit vector perpendicular to plane containing target point
    #     heliostat_name: abbreviated name of heliostat
    #     f_index: facet index, 1-25
    #     az: azimuth in radians
    #     el: elevation in radians
    #     tolerance: acceptable value of error for azimuth and elevation
    #     """

    #     # sets solar field
    #     mirror_template = MirrorParametricRectangular(0, (1.2, 1.2))
    #     test_heliostat, test_heliostat_location = HeliostatAzEl.from_csv_files(
    #         heliostat_name,
    #         dpft.sandia_nsttf_test_heliostats_origin_file(),
    #         dpft.sandia_nsttf_test_facet_centroidsfile(),
    #         mirror_template,
    #     )

    #     # test_heliostat = test_heliostat.set_orientation_from_az_el(az, el)

    #     scene = Scene()
    #     scene.add_object(test_heliostat)
    #     scene.set_position_in_space(test_heliostat, TransformXYZ.from_V(test_heliostat_location))

    #     test_heliostat.set_orientation_from_az_el(az, el)

    #     light_source = LightSourcePoint()
    #     light_source.get_incident_rays(target_loc)

    #     scene.add_light_source(light_source)

    #     trace = rt.trace_scene(scene)

    #     #find facet
    #     facet = #given f_index

    #     is_intersect = self.check_heliostat_intersection_point_on_plane(target_plane_normal, facet_normal)
    #     intersection_point = Intersection.plane_intersect_from_ray_trace(trace, target_loc)

    #     offset_x = intersection_point.x- target_loc.x
    #     offset_z = intersection_point.z- target_loc.z
    #     print("In TestMotionBasedCanting.projected_facet_normal_intersection_point_offset: x offset from target to intersection=", offset_x, "    z offset from target to intersection=", offset_z) #TODO mhh

    #     return is_intersect, offset_x, offset_z, intersection_point

    # def find_single_facet_x_y_value(
    #     self,
    #     target_loc: 'Pxyz',
    #     target_plane_normal: 'Uxyz',
    #     heliostat_name: str,
    #     f_index: int,
    #     az: float,
    #     el: float,
    #     tolerance: float,
    #     ):
    #     """
    #     Returns x and y components for single facet.

    #     Parameters
    #     ----------
    #     target_loc: xyz location of target on given plane
    #     target_plane_normal: unit vector perpendicular to plane containing target point
    #     heliostat_name: abbreviated name of heliostat
    #     f_index: facet index, 1-25
    #     az: azimuth in radians
    #     el: elevation in radians
    #     tolerance: acceptable value of error for azimuth and elevation

    #     """
    #     is_intersect, offset_x, offset_z, intersection_point = self.reflected_facet_intersection_point_offset(
    #         target_loc, target_plane_normal, heliostat_name, f_index, az, el
    #     )

    #     # iterates through finding x and y components, by first finding x component, then y component given x, etc. until within tolerance
    #     for i in range(1, 21):
    #         if Pxyz.distance(intersection_point, target_loc) > tolerance:
    #             x-comp, offset_x_points = self.azimuth_binary_search(
    #                 target_loc, target_plane_normal, heliostat_name, f_index, az, el, tolerance
    #             )
    #             y-comp, offset_z_points = self.elevation_binary_search(
    #                 target_loc, target_plane_normal, heliostat_name, f_index, az, el, tolerance
    #             )
    #             is_intersect, offset_x, offset_z, intersection_point = (
    #                 self.projected_facet_normal_intersection_point_offset(
    #                     target_loc, target_plane_normal, heliostat_name, f_index, az, el
    #                 )
    #             )
    #             print(
    #                 "In TestMotionBasedCanting.find_single_facet_azimuth_el_value:  azimuth = ",
    #                 x-comp,
    #                 "     elevation = ",
    #                 y-comp,
    #             )  # TODO MHH
    #             if i <= 21:
    #                 print(
    #                     "In TestMotionBasedCanting.find_single_facet_x_y_value: In tolerance = ",
    #                     tolerance,
    #                     "distance = ",
    #                     Pxyz.distance(intersection_point, target_loc),
    #                     "x-component = ",
    #                     x-comp,
    #                     "   y-component = ",
    #                     y-comp,
    #                 )  # TODO MHH

    #                 break

    #         # else: #Pxyz.distance(intersection_point, target_loc) <= tolerance:
    #         #     print("In TestMotionBasedCanting.find_single_facet_azimuth_el_value: Values already in tolerance = ", tolerance, "distance = ", Pxyz.distance(intersection_point, target_loc), "azimuth = ",  az, "     elevation = ",el) #TODO MHH

    #     return x, y, intersection_point

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

    # def test_9W1_heliostat_with_canting(self) -> None:
    #     """
    #     Draws 9W1 and surrounding heliostats
    #     """
    #     # Initialize test.
    #     self.start_test()

    #     # Heliostat selection
    #     heliostat_name = '9W1'
    #     up_heliostats = ['9W2', '9W3', '9E1', '9E2', '8W2', '8W1', '8E1', '8E2', '10W2', '10W1', '10E1', '10E2']
    #     test_heliostat = [heliostat_name]

    #     # View setup
    #     title = 'Heliostat ' + heliostat_name + ', with Spherical Canting'
    #     caption = ' Sandia NSTTF heliostats ' + heliostat_name + '.'
    #     comments = []

    #     # # Define test heliostat orientation
    #     # test_azimuth = np.deg2rad(45)
    #     # test_el = np.deg2rad(0)
    #     test_azimuth = np.deg2rad(177)
    #     test_el = np.deg2rad(43)

    #     # # # Define upward-facing heliostat orientation.
    #     UP = Vxyz([0, 0, 1])
    #     focal_length = 68

    #     curved_equation = FunctionXYContinuous(f"(x**2 + y**2)/(4*{focal_length})")

    #     mirror_template = MirrorParametricRectangular(curved_equation, (1.2, 1.2))
    #     h_9W1, h5W1_location = HeliostatAzEl.from_csv_files(
    #         "9W1",
    #         dpft.sandia_nsttf_test_heliostats_origin_file(),
    #         dpft.sandia_nsttf_test_facet_centroidsfile(),
    #         mirror_template,
    #     )

    #     # # Configuration setup
    #     solar_field = self.solar_field
    #     tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
    #     for h_name in up_heliostats:
    #         solar_field.lookup_heliostat(h_name).set_orientation_from_pointing_vector(UP)
    #     for h_name in test_heliostat:
    #         config = HeliostatConfiguration('az-el', az=test_azimuth, el=test_el)
    #         solar_field.lookup_heliostat(h_name).set_orientation(config)
    #         solar_field.lookup_heliostat(h_name).set_canting_from_equation(curved_equation)
    #         facet_origin, facet_normal = solar_field.lookup_heliostat(h_name).facet_ensemble.survey_of_points(
    #             Resolution.center()
    #         )

    #     intersection_points = Intersection.plane_lines_intersection(
    #         (facet_origin, facet_normal), (tower.target_loc, Uxyz([0, 1, 0]))
    #     )

    #     # Style setup
    #     surface_normal_facet_style = rch.facet_outlines_normals(color='b')
    #     surface_normal_facet_style.facet_ensemble_style.default_style.surface_normal_length = 200
    #     solar_field_style = rcsf.heliostat_blanks()
    #     solar_field_style.add_special_names(up_heliostats, rch.facet_outlines(color='g'))
    #     solar_field_style.add_special_names(test_heliostat, surface_normal_facet_style)
    #     tower_control = rct.normal_tower()

    #     # Comment
    #     comments.append("A subset of heliostats selected, and towers.")
    #     comments.append("Blue heliostat is test heliostat 9W1 with vector aiming into target on reciever tower")
    #     comments.append("Green heliostats are face up.")

    #     # Draw
    #     fig_record = fm.setup_figure_for_3d_data(
    #         self.figure_control,
    #         self.axis_control_m,
    #         vs.view_spec_3d(),
    #         number_in_name=False,
    #         input_prefix=self.figure_prefix(
    #             3  # 'f_index'
    #         ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
    #         title=title,
    #         caption=caption,
    #         comments=comments,
    #         code_tag=self.code_tag,
    #     )
    #     solar_field.draw(fig_record.view, solar_field_style)
    #     tower.draw(fig_record.view, tower_control)
    #     fig_record.view.draw_single_Pxyz(intersection_points, rcps.RenderControlPointSeq(marker='+'))
    #     self.show_save_and_check_figure(fig_record)

    #     fig_record = fm.setup_figure_for_3d_data(
    #         self.figure_control,
    #         self.axis_control_m,
    #         vs.view_spec_xz(),
    #         number_in_name=False,
    #         input_prefix=self.figure_prefix(
    #             4  # 'f_index'
    #         ),  # Figure numbers needed because titles may be identical. Hard-code number because test order is unpredictable.
    #         title=title,
    #         caption=caption,
    #         comments=comments,
    #         code_tag=self.code_tag,
    #     )
    #     fig_record.view.draw_single_Pxyz(intersection_points, rcps.RenderControlPointSeq(marker='+'))
    #     tower.draw(fig_record.view, tower_control)

    #     # Output.
    #     self.show_save_and_check_figure(fig_record)

    #     return

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

        is_intersect, offset_x, offset_z, intersection = test_object.projected_facet_normal_intersection_point_offset(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=azimuth_1,
            el=el_1,
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

        is_intersec, offset_x, offset_z, intersection = test_object.projected_facet_normal_intersection_point_offset(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            f_index=f_index,
            az=azimuth_1,
            el=el_1,
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
        focal_length = 68
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

        # heliostat selection index
        heliostat_id = solar_field.heliostat_dict[heliostat_name]  # for 9W1: 87
        heliostat = solar_field.heliostats[heliostat_id]

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

        title = 'Heliostat ' + heliostat_name + 'when surface normals parallel with target plane.'
        caption = 'A single Sandia NSTTF heliostat ' + heliostat_name
        comments = []

        # choose initial position of azimuth and elevation
        test_azimuth = np.deg2rad(0)
        test_el = np.deg2rad(90)

        # Configuration setup
        test_heliostat = [heliostat_name]
        solar_field = self.solar_field
        test_configuration = hc.HeliostatConfiguration(az=test_azimuth, el=test_el)
        solar_field.set_heliostats_configuration(test_heliostat, test_configuration)
        tower = Tower(name='Sandia NSTTF', origin=np.array([0, 0, 0]), parts=["whole tower", "target"])
        target_loc_1 = tower.target_loc
        target_plane_normal_1 = Uxyz([0, 1, 0])
        test_tolerance = 0.001

        # heliostat selection index
        heliostat_id = solar_field.heliostat_dict[heliostat_name]  # for 9W1: 87
        heliostat = solar_field.heliostats[heliostat_id]

        # Comment
        comments.append(
            "The tower with a target in red and intersection point in blue from the surface normal of the facet: "
        )

        result = test_object.find_all_azimuth_el_test_values(
            target_loc=target_loc_1,
            target_plane_normal=target_plane_normal_1,
            heliostat_name=heliostat_name,
            az=test_azimuth,
            el=test_el,
            tolerance=test_tolerance,
        )

        return

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
    azimuth_1 = np.deg2rad(177)
    el_1 = np.deg2rad(43)
    test_tolerance = 0.001

    # result = test_object.projected_facet_normal_intersection_point_offset(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    # )
    # print("next test")
    # result = test_object.azimuth_binary_search(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     tolerance=test_tolerance,
    # )
    # print("next test")
    # result = test_object.elevation_binary_search(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     tolerance=test_tolerance,
    # )
    # print("next test")
    # result = test_object.find_single_facet_azimuth_el_value(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     f_index=f_index_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     tolerance=test_tolerance,
    # )
    # print("next test")
    # result = test_object.find_all_azimuth_el_test_values(
    #     target_loc=target_loc_1,
    #     target_plane_normal=target_plane_normal_1,
    #     heliostat_name=heliostat_name_1,
    #     az=azimuth_1,
    #     el=el_1,
    #     tolerance=test_tolerance,
    # )
    # print("next test")

    # test_object.test_9W1_heliostat()
    # test_object.test_9W1_heliostat_with_canting()
    test_object.test_9W1_with_canting()
    # test_object.test_azimuth_binary_search()
    # test_object.test_el_binary_search()
    # test_object.test_single_facet()
    # test_object.test_all_facets()
    # test_object.test_when_initial_position_not_on_target()

    lt.info('All tests complete.')
    # Cleanup.
    if interactive:
        input("Press Enter...")
    test_object.tearDown()
