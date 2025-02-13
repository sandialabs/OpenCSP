"""
Computing our best estimate of the UAS trajectory and heliostat motion trajectories.



"""

import copy
import csv
from cv2 import cv2 as cv
from datetime import datetime
import logging
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
from numpy.core.numeric import cross
from numpy.linalg.linalg import inv
import pandas as pd
import pickle
import subprocess

import opencsp.common.lib.csp.Heliostat as Heliostat
import opencsp.common.lib.csp.HeliostatConfiguration as hc
import opencsp.common.lib.csp.SolarField as sf
import opencsp.common.lib.csp.sun_track as sun_track  # "st" is taken by string_tools.
import opencsp.common.lib.geo.lon_lat_nsttf as lln
import opencsp.common.lib.geometry.angle as angle
import opencsp.common.lib.geometry.geometry_3d as g3d
import opencsp.common.lib.geometry.transform_3d as t3d
import opencsp.common.lib.render.color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.general_plot as gp
import opencsp.common.lib.render.pandas_plot as pp
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlSolarField as rcsf


import lib.DEPRECATED_specifications as Dspec  # ?? SCAFFOLDING RCB -- TEMPORARY
import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.list_tools as lt
import opencsp.common.lib.tool.math_tools as mt
import opencsp.common.lib.tool.time_date_tools as tdt

# import lib.FrameNameXyList as fnxl
# import lib.HeliostatInfer3d as hi3d
# import common.lib.tool.log_tools as logt
# import lib.NameFrameXyList as nfxl
# import common.lib.render_control.RenderControlKeyFramesGivenManual as rckfgm
# import common.lib.render_control.RenderControlPointSeq as rcps
# import common.lib.render_control.RenderControlText     as rctxt
# import common.lib.render_control.RenderControlTrajectoryAnalysis as rcta
# import lib.ufacet_heliostat_3d_analysis as uh3a
# import lib.ufacet_pipeline_clear as upc
# import lib.ufacet_pipeline_frame as upf
# import lib.DEPRECATED_utils as utils  # ?? SCAFFOLDING RCB -- TEMPORARY
# import common.lib.render.video_manipulation as vm


class TrajectoryAnalysis:
    """
    Class for constructing our best estimate of UAS and heliostat motion trajectories.

    """

    def __init__(
        self,
        #  # Execution control.
        log_dir_body_ext,  # File to add log messages to.  Only used when single_processor == False.
        velocity_calculation_offset_fwd_bwd,  # Integer number of steps to skip to fetch a value for velocity computation.  Must be at least 1.  # ?? SCAFFOLDING RCB -- SHOULD BE UNITS OF TIME?
        delta_velocity_angle_xy_peak_threshold,  # radians.  For synchronization, this is the minimum value required for a peak.
        delta_velocity_angle_xy_non_peak_threshold,  # radians.  For synchronization, this is the maximum value allowed for a peak's neighbor.
        turn_overshoot_skip_time,  # sec.   Time window to allow turn correction overshoot without inferring an interval varying from turn min to turn max.
        scan_establish_velocity_time,  # sec.   Time for the UAS to reach a "reasonably stable " velocity at the beginning of a scan pass.  The UAS is "establishing" its constant scan velocity.
        scan_discard_velocity_time,  # sec.   Time prior to the end point when a UAS bgeins changing velocity prior to the pass endpoint.  The UAS is "discarding" its constant scan velocity.
        minimum_scan_pass_time,  # sec.   Duration of the shortest possible scan pass, after already trimming away the times to establish and discard the scan velocity.
        nominal_scan_speed,  # m/sec. Nominal speed of UAS flight during a linear scan pass.
        # m/sec. Tolerance to use wheen deciding that the average speed fo a cnadidate pass is consistent with a possible scan.  Not the scan speed control tolerance; larger than that.
        scan_speed_tolerance,
        nominal_scan_velocity_z,  # m/sec. Nominal vertical speed of UAS flight during a linear scan pass.
        # m/sec. Tolerance to use wheen deciding that the average vertical speed of a candidate pass is consistent with a possible scan.  Not the scan speed control tolerance; larger than that.
        scan_velocity_z_tolerance,
        maximum_n_missing,  # Maximum number of missing points allowed for a frame to be used to infer GPS error.
        minimum_gps_pass_inter_point_speed,  # m/sec. Minimum observed inter-point speed allowable along a GPS scan pass.
        minimum_gps_pass_number_of_points,  # Minmum number of points required to constitue a GPS pass.
        gps_pass_start_margin,  # Number of points to shrink the start of a camera pass after removing points corresponding to excess missing corners.
        gps_pass_stop_margin,  # Number of points to shrink the end of a camera pass after removing points corresponding to excess missing corners.
        # Maximum distance between estiamted camera trajectory points (expressed in heliostat coordiantes), to consider part of a connected trajectory.
        maximum_camera_pass_inter_point_distance,
        minimum_camera_pass_inter_point_speed,  # m/sec. Minimum observed inter-point speed allowable along a GPS scan pass.
        minimum_camera_pass_number_of_points,  # Minmum number of points required to constitue a camera pass.
        camera_pass_start_margin,  # Number of points to shrink the start of a camera pass after removing points corresponding to excess missing corners.
        camera_pass_stop_margin,  # Number of points to shrink the end of a camera pass after removing points corresponding to excess missing corners.
        # Input/output sources.
        # Solar field parameters.  # ?? SCAFFOLDING  RCB -- RENAME THIS TO "input_specificaitons" THROUGHOUT CODE, OR ELIMINATE AS PART OF LARGER INPUT DEFINITION SOLUTION.
        specifications,
        aimpoint_xyz,  # Tracking aim point for those heliostats that are tracking.  # ?? SCAFFOLDING RCB -- GENERALIZE THIS TO ALLOW PER-HELIOSTAT AIM POINTS.
        when_ymdhmsz,  # Nominal time.  For example, mid point of the flight, or takeoff time.
        up_heliostats,  # List of heliostat names that are face up.
        up_configuration,  # Configuration of face-up heliostats.  A HeliostatConfiguration object.
        down_heliostats,  # List of heliostat names that are face down.
        down_configuration,  # Configuration of face-down heliostats.  A HeliostatConfiguration object.
        input_video_dir_body_ext,  # Where to find the video file.
        input_flight_log_dir_body_ext,  # Where to find flight log file.
        input_reconstructed_heliostats_dir,  # Where to find directories containing heliostat reconstruction results.
        output_data_dir,  # Where to save the resulting summary data.
        output_render_dir,  # Where to save the resulting plots showing final heliostat tracks.
        output_construction_dir,  # Where to save the detailed image processing step-by-step plots.
        #  # Render control.
        #  render_control_projected_distorted,            # Flags to control rendering on this run, for the projected, distorted data.
        #  render_control_projected_undistorted,          # Flags to control rendering on this run, for the projected, undistorted data.
        #  render_control_confirmed_distorted,            # Flags to control rendering on this run, for the confirmed, distorted data.
        #  render_control_confirmed_undistorted,          # Flags to control rendering on this run, for the confirmed, undistorted data.
    ):
        # Check input.
        if (input_video_dir_body_ext == None) or (len(input_video_dir_body_ext) == 0):
            raise ValueError("In TrajectoryAnalysis.__init__(), null input_video_dir_body_ext encountered.")
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In TrajectoryAnalysis.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In TrajectoryAnalysis.__init__(), null output_render_dir encountered.")

        # Parse input video path components.
        input_video_dir, input_video_body, input_video_ext = ft.path_components(input_video_dir_body_ext)

        # Store input.
        # Execution control.
        self.log_dir_body_ext = log_dir_body_ext
        self.velocity_calculation_offset_fwd_bwd = velocity_calculation_offset_fwd_bwd
        self.delta_velocity_angle_xy_peak_threshold = delta_velocity_angle_xy_peak_threshold
        self.delta_velocity_angle_xy_non_peak_threshold = delta_velocity_angle_xy_non_peak_threshold
        self.turn_overshoot_skip_time = turn_overshoot_skip_time
        self.scan_establish_velocity_time = scan_establish_velocity_time
        self.scan_discard_velocity_time = scan_discard_velocity_time
        self.minimum_scan_pass_time = minimum_scan_pass_time
        self.nominal_scan_speed = nominal_scan_speed
        self.scan_speed_tolerance = scan_speed_tolerance
        self.nominal_scan_velocity_z = nominal_scan_velocity_z
        self.scan_velocity_z_tolerance = scan_velocity_z_tolerance
        self.maximum_n_missing = maximum_n_missing
        self.minimum_gps_pass_inter_point_speed = minimum_gps_pass_inter_point_speed
        self.minimum_gps_pass_number_of_points = minimum_gps_pass_number_of_points
        self.gps_pass_start_margin = gps_pass_start_margin
        self.gps_pass_stop_margin = gps_pass_stop_margin
        self.maximum_camera_pass_inter_point_distance = maximum_camera_pass_inter_point_distance
        self.minimum_camera_pass_inter_point_speed = minimum_camera_pass_inter_point_speed
        self.minimum_camera_pass_number_of_points = minimum_camera_pass_number_of_points
        self.camera_pass_start_margin = camera_pass_start_margin
        self.camera_pass_stop_margin = camera_pass_stop_margin

        # Input/output sources.
        self.specifications = specifications
        self.aimpoint_xyz = aimpoint_xyz
        self.when_ymdhmsz = when_ymdhmsz
        self.up_heliostats = up_heliostats
        self.up_configuration = up_configuration
        self.down_heliostats = down_heliostats
        self.down_configuration = down_configuration
        self.input_video_dir_body_ext = input_video_dir_body_ext
        self.input_video_dir = input_video_dir
        self.input_video_body = input_video_body
        self.input_video_ext = input_video_ext
        self.input_flight_log_dir_body_ext = input_flight_log_dir_body_ext
        self.input_reconstructed_heliostats_dir = input_reconstructed_heliostats_dir
        self.output_data_dir = output_data_dir
        # self.output_data_dir = os.path.join(<home_dir>, ('output_' + datetime.now().strftime('%Y_%m_%d_%H%M')))  # ?? SCAFFOLDING RCB -- TEMPORARY
        self.output_render_dir = output_render_dir
        self.output_construction_dir = output_construction_dir
        #         # Render control.
        #         self.render_control_projected_distorted            = render_control_projected_distorted
        #         self.render_control_projected_undistorted          = render_control_projected_undistorted
        #         self.render_control_confirmed_distorted            = render_control_confirmed_distorted
        #         self.render_control_confirmed_undistorted          = render_control_confirmed_undistorted

        # Other times.
        # An hour before flight.
        self.when_ymdhmsz_minus_1_hour = copy.deepcopy(self.when_ymdhmsz)
        self.when_ymdhmsz_minus_1_hour[3] -= 1  # Assume not near midnight, so don't worry about rollover.
        # A half-hour before flight.
        self.when_ymdhmsz_minus_30_minutes = copy.deepcopy(self.when_ymdhmsz)
        if self.when_ymdhmsz_minus_30_minutes[4] > 30:
            self.when_ymdhmsz_minus_30_minutes[4] -= 30
        else:
            self.when_ymdhmsz_minus_30_minutes[3] -= 1  # Assume not near midnight, so don't worry about rollover.
            self.when_ymdhmsz_minus_30_minutes[4] += 30

        # Load solar field data.
        print("In TrajectoryAnalysis.__init__(), loading solar field...")
        self.solar_field = sf.SolarField(
            name="Sandia NSTTF",
            short_name="NSTTF",
            origin_lon_lat=lln.NSTTF_ORIGIN,
            heliostat_file=self.specifications.heliostat_locations_file,  # ?? SCAFFOLDING RCB -- RENAME "_file" TO "_dir_body_ext" THROUGHOUT CODE.
            facet_centroids_file=self.specifications.facets_centroids_file,
        )  # ?? SCAFFOLDING RCB -- RENAME "_file" TO "_dir_body_ext" THROUGHOUT CODE.

        # Configuration setup
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)

        # Drone GPS trajectory from flight log.
        print("In TrajectoryAnalysis.__init__(), loading GPS log...")
        # ?? SCAFFOLDING RCB -- FLIGHT_LOG SHOULD BE A CLASS, SUPPORTING FUNCTIONS SUCH A MID-TIME, ALTITUDE AND POSITION LIMITS, ETC.  SHOULD ALSO HANDLE LOGS FROM DIFFERENT UAS SOURCES.
        self.flight_log_df = pd.read_csv(self.input_flight_log_dir_body_ext)
        self.add_velocity_columns_to_flight_log_df()
        self.print_flight_log_df()
        # Save enhanced GPS flight log.
        self.output_flight_log_plus_dir_body_ext = self.save_enhanced_flight_log()

        # Determine GPS time corresponding to t=0 in the flight log seconds column, and store
        # in the format used by heliostat aim point calculations.
        print("In TrajectoryAnalysis.__init__(), finding GPS time corresponding to flight log time=0...")
        self.gps_ymdhmsz_given_flight_log_zero_seconds = self.compute_gps_ymdhmsz_given_flight_log_zero_seconds()
        self.print_gps_flight_log_zero_seconds()
        # Save GPS time corresponding to flight log zero time.
        self.gps_ymdhmsz_given_flight_log_zero_seconds_dir_body_ext = (
            self.save_gps_ymdhmsz_given_flight_log_zero_seconds()
        )

        # Find points where velocity abruptly changes.
        print("In TrajectoryAnalysis.__init__(), finding velocity xy change points...")
        self.gps_velocity_xy_change_minima = self.find_gps_velocity_xy_change_maxima(-1)
        self.gps_velocity_xy_change_maxima = self.find_gps_velocity_xy_change_maxima(1)
        # self.print_velocity_xy_change_points()
        # Save velocity change points.
        self.output_gps_velocity_xy_change_minima_dir_body_ext = self.save_gps_velocity_xy_change_points(
            self.gps_velocity_xy_change_minima, "minima"
        )
        self.output_gps_velocity_xy_change_maxima_dir_body_ext = self.save_gps_velocity_xy_change_points(
            self.gps_velocity_xy_change_maxima, "maxima"
        )

        # Find scan passes.
        # ?? SCAFFOLDING RCB -- COMMENT:  THIS APPROACH BECOMES BRITTLE WHEN INTER-PASS TURN ANGLE IS NOT NEAR 90 DEGREES.  MAYBE BASED ONSPEED IN DIRECTION PARALELL TO SCAN, AS INDICATED BY PLAN?  SIMILAR SPEED CHANGE?
        # ?? SCAFFOLDING RCB -- COMMENT:  VARIABLE NAMES ARE POOR.  RIGHT/LEFT-TURN SCAN PASS?
        # ?? SCAFFOLDING RCB -- COMMENT:  SCAN PASS CURRENTLY IS SIMPLY A LINE DRAWN FROM THE FIRST POINT OT HE LAST, ASSUMING SOME SORT OF STABILIZATION PERIOD.  A FIT LINE SEEMS MORE ACCURATE.
        # ?? SCAFFOLDING RCB -- COMMENT:  FURTHER, OUTLIERS SHOULD BE REJECTED, AND THE STABILIZATION PERIOD SOMEHOW EVALUATED.
        # ?? SCAFFOLDING RCB -- COMMENT:  LOTS OF REWORK NEEDED HERE.
        # ?? SCAFFOLDING RCB -- COMMENT:  MAX-TO-MIN AND MIN-TO-MAX ROUTINE ARE DUPLICATED.  SHOULD BE MERGED TO SUPPORT MORE CONSISTENT CODE MAINTENANCE.
        # ?? SCAFFOLDING RCB -- COMMENT:  MIN AND MAX NOMENCLATURE MUDDLED
        # ?? SCAFFOLDING RCB -- COMMENT:
        # ?? SCAFFOLDING RCB -- COMMENT:
        print("In TrajectoryAnalysis.__init__(), finding GPS scan passes...")
        # Find pairs of a qualifying local maximum followed by the nearest qualifying local minimum.
        self.maximum_to_minimum_pass_pair_list = self.find_velocity_xy_change_maximum_to_minimum_scan_passes()
        self.minimum_to_maximum_pass_pair_list = self.find_velocity_xy_change_minimum_to_maximum_scan_passes()
        # Refine min/max pairs to identify stable scan pass motions.
        self.gps_scan_passes = self.convert_gps_minima_maxima_passes_to_scan_passes()
        self.print_gps_scan_pass_summary()
        # Save GPS scan pass data.
        self.maximum_to_minimum_pass_pair_list_dir_body_ext = self.save_gps_velocity_xy_change_pairs(
            self.maximum_to_minimum_pass_pair_list, "maximum_to_minimum"
        )
        self.minimum_to_maximum_pass_pair_list_dir_body_ext = self.save_gps_velocity_xy_change_pairs(
            self.minimum_to_maximum_pass_pair_list, "minimum_to_maximum"
        )
        self.output_gps_scan_passes_dir_body_ext = self.save_gps_scan_passes()

        # Load the trajectory fragments associated with each heliostat 3-d reconstruction.
        print("In TrajectoryAnalysis.__init__(), loading trajectory fragments...")
        self.hel_frames_dict = self.load_trajectory_fragments()
        self.print_hel_frames_dict()
        self.hel_frames_dict_dir_body_ext = self.save_hel_frames_dict()

        # Identify time synchronization between GPS flight log and camera frames.
        # ?? SCAFFOLDING RCB -- NOTE THAT THIS IS NOT GENERAL.  MY CURRENT THINKING IS THAT IT SHOULD PROBABLY BE REPLACED BY AN EARLIER COMPUTATION THAT SUPPORTS KEY FRAME IDENTIFICATION, FOR EXAMPLE.  SEE COMMENTS IN ROUTINE.
        print("In TrajectoryAnalysis.__init__(), initializing GPS-frame synchronization constants...")
        self.synchronization_pair_list = self.initialize_synchronization_pair_list()
        (self.synchronization_slope, self.synchronization_intercept) = self.initialize_synchronization_constants()
        self.print_synchronization_pair_list()
        self.synchronization_constants_dir_body_ext = self.save_synchronization_constants()

        # Construct heliostat camera passes.
        print("In TrajectoryAnalysis.__init__(), constructing heliostat camera passes...")
        self.hel_camera_passes_dict = self.construct_heliostat_camera_passes()
        self.print_hel_camera_passes_dict(max_heliostats=3)
        self.hel_camera_passes_dict_dir_body_ext = self.save_hel_camera_passes_dict()

        # Construct data structures for GPS-camera analysis.
        print("In TrajectoryAnalysis.__init__(), constructing heliostat GPS-camera analysis...")
        self.hel_gps_camera_analysis_dict = self.construct_hel_gps_camera_analysis_dict()
        self.print_hel_gps_camera_analysis_dict(max_heliostats=3)
        self.hel_gps_camera_analysis_dict_dir_body_ext = self.save_hel_gps_camera_analysis_dict()

        # Figure control information.
        print("In TrajectoryAnalysis.__init__(), initializing figures...")
        fm.reset_figure_management()
        self.figure_control = rcfg.RenderControlFigure(tile_array=(2, 1), tile_square=True)
        self.axis_control_m = rca.meters()

        # Check pickle files.
        self.check_pickle_files()

        # Draw solar field plots.
        print("In TrajectoryAnalysis.__init__(), drawing solar field plots...")
        self.draw_and_save_solar_field_suite()

        # Draw GPS trajectory analysis plots.
        print("In TrajectoryAnalysis.__init__(), drawing GPS log analysis plots...")
        self.draw_and_save_gps_log_analysis_plots()

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  ENHANCE FLIGHT LOG
    #

    def add_velocity_columns_to_flight_log_df(self):
        """
        Adds velocity, speed, angle, and change information.
        """
        # Construct flight log (x,y,z) coordinates.
        print("In TrajectoryAnalysis.add_velocity_columns_to_flight_log_df(), adding position columns...")
        self.flight_log_xyz_list = []  # ?? SCAFFOLDING RCB -- ELIMINATE THIS, IN FAVOR OF THE DATAFRAME.
        empty_column = [np.nan] * len(self.flight_log_df.index)
        self.flight_log_df["x(m)"] = empty_column
        self.flight_log_df["y(m)"] = empty_column
        self.flight_log_df["z(m)"] = empty_column
        self.flight_log_df["time(sec)"] = empty_column
        print("In TrajectoryAnalysis.add_velocity_columns_to_flight_log_df(), added, now filling...")
        for (
            idx
        ) in (
            self.flight_log_df.index
        ):  # ?? SCAFFOLDING RCB -- THIS FOR LOOP IS VERY SLOW.  CAN WE VECTORIZE IT?  REPLACE THE .loc[] CALLS WITH SOMETHING FASTER?
            longitude = self.flight_log_df["longitude"][idx]
            latitude = self.flight_log_df["latitude"][idx]
            altitude = self.flight_log_df["altitude(m)"][idx]
            t_msec = self.flight_log_df["time(millisecond)"][idx]
            x, y = lln.nsttf_xy_given_lon_lat(longitude, latitude)
            z = altitude
            t_sec = t_msec / 1000.0
            self.flight_log_xyz_list.append(
                [x, y, z]
            )  # ?? SCAFFOLDING RCB -- ELIMINATE THIS, IN FAVOR OF THE DATAFRAME.
            self.flight_log_df.loc[idx, "x(m)"] = x
            self.flight_log_df.loc[idx, "y(m)"] = y
            self.flight_log_df.loc[idx, "z(m)"] = z
            self.flight_log_df.loc[idx, "time(sec)"] = t_sec

        # Construct flight log velocity before each time instant.
        print("In TrajectoryAnalysis.add_velocity_columns_to_flight_log_df(), adding velocity before columns...")
        empty_column = [np.nan] * len(self.flight_log_df.index)
        self.flight_log_df["velocity_before_x(m/sec)"] = empty_column
        self.flight_log_df["velocity_before_y(m/sec)"] = empty_column
        self.flight_log_df["velocity_before_z(m/sec)"] = empty_column
        self.flight_log_df["speed_before(m/sec)"] = empty_column
        print("In TrajectoryAnalysis.add_velocity_columns_to_flight_log_df(), added, now filling...")
        for (
            idx
        ) in (
            self.flight_log_df.index
        ):  # ?? SCAFFOLDING RCB -- THIS FOR LOOP IS VERY SLOW.  CAN WE VECTORIZE IT?  REPLACE THE .loc[] CALLS WITH SOMETHING FASTER?
            if idx > (self.velocity_calculation_offset_fwd_bwd - 1):
                x = self.flight_log_df.loc[idx, "x(m)"]
                y = self.flight_log_df.loc[idx, "y(m)"]
                z = self.flight_log_df.loc[idx, "z(m)"]
                t = self.flight_log_df.loc[idx, "time(sec)"]
                before_x = self.flight_log_df.loc[idx - self.velocity_calculation_offset_fwd_bwd, "x(m)"]
                before_y = self.flight_log_df.loc[idx - self.velocity_calculation_offset_fwd_bwd, "y(m)"]
                before_z = self.flight_log_df.loc[idx - self.velocity_calculation_offset_fwd_bwd, "z(m)"]
                before_t = self.flight_log_df.loc[idx - self.velocity_calculation_offset_fwd_bwd, "time(sec)"]
                delta_x = x - before_x
                delta_y = y - before_y
                delta_z = z - before_z
                delta_t = t - before_t
                d = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
                self.flight_log_df.loc[idx, "velocity_before_x(m/sec)"] = delta_x / delta_t
                self.flight_log_df.loc[idx, "velocity_before_y(m/sec)"] = delta_y / delta_t
                self.flight_log_df.loc[idx, "velocity_before_z(m/sec)"] = delta_z / delta_t
                self.flight_log_df.loc[idx, "speed_before(m/sec)"] = d / delta_t

        # Construct flight log velocity after each time instant.
        print("In TrajectoryAnalysis.add_velocity_columns_to_flight_log_df(), adding velocity after columns...")
        empty_column = [np.nan] * len(self.flight_log_df.index)
        self.flight_log_df["velocity_after_x(m/sec)"] = empty_column
        self.flight_log_df["velocity_after_y(m/sec)"] = empty_column
        self.flight_log_df["velocity_after_z(m/sec)"] = empty_column
        self.flight_log_df["speed_after(m/sec)"] = empty_column
        print("In TrajectoryAnalysis.add_velocity_columns_to_flight_log_df(), added, now filling...")
        for (
            idx
        ) in (
            self.flight_log_df.index
        ):  # ?? SCAFFOLDING RCB -- THIS FOR LOOP IS VERY SLOW.  CAN WE VECTORIZE IT?  REPLACE THE .loc[] CALLS WITH SOMETHING FASTER?
            if idx < (len(self.flight_log_df.index) - self.velocity_calculation_offset_fwd_bwd):
                x = self.flight_log_df.loc[idx, "x(m)"]
                y = self.flight_log_df.loc[idx, "y(m)"]
                z = self.flight_log_df.loc[idx, "z(m)"]
                t = self.flight_log_df.loc[idx, "time(sec)"]
                after_x = self.flight_log_df.loc[idx + self.velocity_calculation_offset_fwd_bwd, "x(m)"]
                after_y = self.flight_log_df.loc[idx + self.velocity_calculation_offset_fwd_bwd, "y(m)"]
                after_z = self.flight_log_df.loc[idx + self.velocity_calculation_offset_fwd_bwd, "z(m)"]
                after_t = self.flight_log_df.loc[idx + self.velocity_calculation_offset_fwd_bwd, "time(sec)"]
                delta_x = after_x - x
                delta_y = after_y - y
                delta_z = after_z - z
                delta_t = after_t - t
                d = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
                self.flight_log_df.loc[idx, "velocity_after_x(m/sec)"] = delta_x / delta_t
                self.flight_log_df.loc[idx, "velocity_after_y(m/sec)"] = delta_y / delta_t
                self.flight_log_df.loc[idx, "velocity_after_z(m/sec)"] = delta_z / delta_t
                self.flight_log_df.loc[idx, "speed_after(m/sec)"] = d / delta_t

        # Construct flight log before/after average velocity for time instant, and local change in speed and direction.
        print(
            "In TrajectoryAnalysis.add_velocity_columns_to_flight_log_df(), adding average velociy, velocity change, angle, and angle change columns..."
        )
        empty_column = [np.nan] * len(self.flight_log_df.index)
        self.flight_log_df["velocity_average_x(m/sec)"] = empty_column
        self.flight_log_df["velocity_average_y(m/sec)"] = empty_column
        self.flight_log_df["velocity_average_z(m/sec)"] = empty_column
        self.flight_log_df["speed_average(m/sec)"] = empty_column
        self.flight_log_df["delta_speed(m/sec)"] = empty_column
        self.flight_log_df["abs_delta_speed(m/sec)"] = empty_column
        self.flight_log_df["velocity_angle_xy(rad)"] = empty_column
        self.flight_log_df["velocity_angle_z(rad)"] = empty_column
        self.flight_log_df["delta_velocity_angle_xy(rad)"] = empty_column
        self.flight_log_df["delta_velocity_angle_z(rad)"] = empty_column
        print("In TrajectoryAnalysis.add_velocity_columns_to_flight_log_df(), added, now filling...")
        for (
            idx
        ) in (
            self.flight_log_df.index
        ):  # ?? SCAFFOLDING RCB -- THIS FOR LOOP IS VERY SLOW.  CAN WE VECTORIZE IT?  REPLACE THE .loc[] CALLS WITH SOMETHING FASTER?
            if idx > 0:
                # Fetch values.
                before_vx = self.flight_log_df.loc[idx, "velocity_before_x(m/sec)"]
                before_vy = self.flight_log_df.loc[idx, "velocity_before_y(m/sec)"]
                before_vz = self.flight_log_df.loc[idx, "velocity_before_z(m/sec)"]
                before_s = self.flight_log_df.loc[idx, "speed_before(m/sec)"]
                after_vx = self.flight_log_df.loc[idx, "velocity_after_x(m/sec)"]
                after_vy = self.flight_log_df.loc[idx, "velocity_after_y(m/sec)"]
                after_vz = self.flight_log_df.loc[idx, "velocity_after_z(m/sec)"]
                after_s = self.flight_log_df.loc[idx, "speed_after(m/sec)"]
                # Compute new values.
                average_vx = (before_vx + after_vx) / 2.0
                average_vy = (before_vy + after_vy) / 2.0
                average_vz = (before_vz + after_vz) / 2.0
                average_s = (before_s + after_s) / 2.0
                delta_speed = after_s - before_s
                abs_delta_speed = abs(delta_speed)
                # Directions.
                velocity_angle_xy = self.velocity_angle_xy(average_vx, average_vy)
                velocity_angle_z = self.velocity_angle_z(average_vx, average_vy, average_vz)
                before_velocity_angle_xy = self.velocity_angle_xy(before_vx, before_vy)
                before_velocity_angle_z = self.velocity_angle_z(before_vx, before_vy, before_vz)
                after_velocity_angle_xy = self.velocity_angle_xy(after_vx, after_vy)
                after_velocity_angle_z = self.velocity_angle_z(after_vx, after_vy, after_vz)
                delta_velocity_angle_xy = angle.angle2_minus_angle_1(before_velocity_angle_xy, after_velocity_angle_xy)
                delta_velocity_angle_z = angle.angle2_minus_angle_1(before_velocity_angle_z, after_velocity_angle_z)
                # Store results.
                self.flight_log_df.loc[idx, "velocity_average_x(m/sec)"] = average_vx
                self.flight_log_df.loc[idx, "velocity_average_y(m/sec)"] = average_vy
                self.flight_log_df.loc[idx, "velocity_average_z(m/sec)"] = average_vz
                self.flight_log_df.loc[idx, "speed_average(m/sec)"] = average_s
                self.flight_log_df.loc[idx, "delta_speed(m/sec)"] = delta_speed
                self.flight_log_df.loc[idx, "abs_delta_speed(m/sec)"] = abs_delta_speed
                self.flight_log_df.loc[idx, "velocity_angle_xy(rad)"] = velocity_angle_xy
                self.flight_log_df.loc[idx, "velocity_angle_z(rad)"] = velocity_angle_z
                self.flight_log_df.loc[idx, "delta_velocity_angle_xy(rad)"] = delta_velocity_angle_xy
                self.flight_log_df.loc[idx, "delta_velocity_angle_z(rad)"] = delta_velocity_angle_z

    def velocity_angle_xy(self, velocity_x, velocity_y):
        """
        Measured ccw from x axis.
        """
        if (velocity_x == 0) and (velocity_y == 0):
            return 0.0
        else:
            return math.atan2(velocity_y, velocity_x)

    def velocity_angle_z(self, velocity_x, velocity_y, velocity_z):
        """
        Measured from xy plane, analogous to elevation.
        For example, 30 degrees above the horizon is np.radians(+30).
        """
        if (velocity_x == 0) and (velocity_y == 0) and (velocity_z == 0):
            return 0.0
        else:
            velocity_xy = np.sqrt(velocity_x**2 + velocity_y**2)
            if velocity_xy == 0.0:
                if velocity_z > 0:
                    return math.pi / 2.0
                elif velocity_z < 0:
                    return -(math.pi / 2.0)
                else:
                    print("ERROR: In 190_TrajectoryAnalysis.velocity_direction_z(), unexpected situation encountered.")
                    assert False
            else:
                return math.atan2(velocity_z, velocity_xy)

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  FIND GPS TIME OFFSET
    #

    def gps_ymdhmsz_given_flight_log_seconds(
        self, time_sec
    ):  # Time is in seconds, corresponding to flight log 'time(sec)' column.
        """
        Unlike above routines, this calculation is not tied to visual content.
        It is simply the conversion from the flight log seconds column to the
        flight log GPS time columns.
        """
        return tdt.add_seconds_to_ymdhmsz(self.gps_ymdhmsz_given_flight_log_zero_seconds, time_sec)

    def compute_gps_ymdhmsz_given_flight_log_zero_seconds(self):
        """
        Unlike above routines, this calculation is not tied to visual content.
        It is simply the conversion from the flight log seconds column to the
        flight log GPS time columns.
        """
        # Fetch a time point and a GPS date time form the same row, to deduce the time offset.
        flight_log_first_row_time_sec = self.flight_log_df.loc[0, "time(sec)"]
        flight_log_first_row_gps_date_time_str = self.flight_log_df.loc[
            0, "datetime(local)"
        ]  # Example: '2020-12-03 15:44:13.802'
        # Parse the GPS date/time string.
        gps_date_str, gps_time_str = flight_log_first_row_gps_date_time_str.split(" ")
        gps_year_str, gps_month_str, gps_day_str = gps_date_str.split("-")
        gps_hour_str, gps_minute_str, gps_second_str = gps_time_str.split(":")
        gps_year = int(gps_year_str)
        gps_month = int(gps_month_str)
        gps_day = int(gps_day_str)
        gps_hour = int(gps_hour_str)
        gps_minute = int(gps_minute_str)
        gps_second = float(gps_second_str)
        # Fetch the time zone offset.
        zone = self.when_ymdhmsz[6]
        # Assemble the GPS time in ymdhmsz format.
        # ?? SCAFFOLDING RCB -- THIS SUGGESTS MIGRATING THE YMDHMSZ FORMAT TO A STANDARD PYTHON DATE TIME FORMAT, THROUGHOUT.
        gps_ymdhmsz = [gps_year, gps_month, gps_day, gps_hour, gps_minute, gps_second, zone]
        # Subtract the first-row time in seconds, to obtain the time corresponding to flight log seconds = 0.
        zero_time_ymdhmsz = tdt.subtract_seconds_from_ymdhmsz(gps_ymdhmsz, flight_log_first_row_time_sec)
        # Return.
        return zero_time_ymdhmsz

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  FLIGHT LOG TIME LOOKUP
    #

    def flight_log_xyz_given_time(self, input_time):
        for idx in self.flight_log_df.index:
            time = self.flight_log_df.loc[idx, "time(sec)"]
            if time == input_time:
                # Time matches exactly.
                x = self.flight_log_df.loc[idx, "x(m)"]
                y = self.flight_log_df.loc[idx, "y(m)"]
                z = self.flight_log_df.loc[idx, "z(m)"]
                return [x, y, z]
            elif time >= input_time:
                # Time is not an exact match.  Interpolate.
                if idx == 0:
                    print(
                        "ERROR: In TrajectoryAnalysis.flight_log_xyz_given_time(), input time is before flight log start time.  Input time =",
                        str(input_time),
                        "; Flight log start time =",
                        str(time),
                    )
                    assert False
                # The idx we have refers to a time just after input_time.
                time_2 = time
                x_2 = self.flight_log_df.loc[idx, "x(m)"]
                y_2 = self.flight_log_df.loc[idx, "y(m)"]
                z_2 = self.flight_log_df.loc[idx, "z(m)"]
                # Fetch prior time data.
                before_idx = idx - 1
                time_1 = self.flight_log_df.loc[before_idx, "time(sec)"]
                x_1 = self.flight_log_df.loc[before_idx, "x(m)"]
                y_1 = self.flight_log_df.loc[before_idx, "y(m)"]
                z_1 = self.flight_log_df.loc[before_idx, "z(m)"]
                # Compute deltas.
                dt_12 = time_2 - time_1
                dx_12 = x_2 - x_1
                dy_12 = y_2 - y_1
                dz_12 = z_2 - z_1
                # Interpolation fraction.
                dt_1i = input_time - time_1
                frac = dt_1i / dt_12
                # Interpolate.
                x = x_1 + (frac * dx_12)
                y = y_1 + (frac * dy_12)
                z = z_1 + (frac * dz_12)
                # Return.
                return [x, y, z]
        # We fell through loop and still haven't exceeded the input time.
        # The input time is therefore out of bounds.
        print(
            "ERROR: In TrajectoryAnalysis.flight_log_xyz_given_time(), input time is after flight log end time.  Input time =",
            str(input_time),
            "; Flight log end time =",
            str(time),
        )
        assert False

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  CONSTRUCT GPS SCAN PASSES
    #
    #

    # FIND VELOCITY XY CHANGE MINIMA AND MAXIMA

    def find_gps_velocity_xy_change_maxima(self, sign):  # sign is +1 or -1.
        """
        We look for large changes in xy velocity direction to find cues to GPS scan pass boundaries.

        If sign = +1, returns maxima.
        If sign = -1, returns minima.
        """
        gps_velocity_xy_change_maxima = []
        for idx in self.flight_log_df.index:
            if (idx > 4) and (idx < (len(self.flight_log_df.index) - 5)):
                # Fetch values.
                prev5_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx - 5, "delta_velocity_angle_xy(rad)"]
                prev4_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx - 4, "delta_velocity_angle_xy(rad)"]
                prev3_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx - 3, "delta_velocity_angle_xy(rad)"]
                prev2_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx - 2, "delta_velocity_angle_xy(rad)"]
                prev1_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx - 1, "delta_velocity_angle_xy(rad)"]
                this_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx, "delta_velocity_angle_xy(rad)"]
                next1_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx + 1, "delta_velocity_angle_xy(rad)"]
                next2_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx + 2, "delta_velocity_angle_xy(rad)"]
                next3_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx + 3, "delta_velocity_angle_xy(rad)"]
                next4_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx + 4, "delta_velocity_angle_xy(rad)"]
                next5_delta_velocity_angle_xy = sign * self.flight_log_df.loc[idx + 5, "delta_velocity_angle_xy(rad)"]
                if (
                    (prev5_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (prev4_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (prev3_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (prev2_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (prev1_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (this_delta_velocity_angle_xy >= self.delta_velocity_angle_xy_peak_threshold)
                    and (next1_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (next2_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (next3_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (next4_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                    and (next5_delta_velocity_angle_xy <= this_delta_velocity_angle_xy)
                ):
                    # Then this row qualifies as a peak.
                    this_time = self.flight_log_df.loc[idx, "time(sec)"]
                    this_x = self.flight_log_df.loc[idx, "x(m)"]
                    this_y = self.flight_log_df.loc[idx, "y(m)"]
                    this_z = self.flight_log_df.loc[idx, "z(m)"]
                    maximum_dict = {}
                    maximum_dict["time"] = this_time
                    maximum_dict["delta_velocity_angle_xy"] = sign * this_delta_velocity_angle_xy
                    maximum_dict["x"] = this_x
                    maximum_dict["y"] = this_y
                    maximum_dict["z"] = this_z
                    gps_velocity_xy_change_maxima.append(maximum_dict)
        # Return.
        return gps_velocity_xy_change_maxima

    # MINIMA/MAXIMA SCAN PASS PAIRS

    def find_velocity_xy_change_maximum_to_minimum_scan_passes(
        self,
    ):  # ?? SCAFFOLDING RCB -- THIS ROUTINE DUPLICATES ITS PAIR.  CAN THEY BE MERGED, WHILE RETAINING CORRECTNESS AND CLARITY?
        maximum_to_minimum_pass_pair_list = []
        for maximum_dict in self.gps_velocity_xy_change_maxima:  # Here "maximum" means "maximum_velocity_xy_change"
            # Here is a local maximum time to work with.
            time_maximum = maximum_dict[
                "time"
            ]  # Here "time_maximum" means "time of the maximum_velocity_xy_change," not the minimum time.
            # Find the next local minimum time.
            for minimum_dict in self.gps_velocity_xy_change_minima:
                time_minimum = minimum_dict["time"]
                if time_minimum > (time_maximum + self.turn_overshoot_skip_time):
                    break
            # See if there is a local maximum in between.
            for alternative_maximum_dict in self.gps_velocity_xy_change_maxima:
                alternative_time_maximum = alternative_maximum_dict["time"]
                if (alternative_time_maximum > time_maximum) and (alternative_time_maximum < time_minimum):
                    maximum_dict = alternative_maximum_dict
                if alternative_time_maximum > time_minimum:
                    break
            # Now the minimum and maximum times are adjacent.
            maximum_to_minimum_pair = [maximum_dict, minimum_dict]
            # Avoid duplicate entries.
            found = False
            time_maximum = maximum_dict["time"]  # Refresh in case it has changed.
            for search_maximum_to_minimum_pair in maximum_to_minimum_pass_pair_list:
                search_maximum_dict = search_maximum_to_minimum_pair[0]
                search_time_maximum = search_maximum_dict["time"]
                if search_time_maximum == time_maximum:
                    found = True
                    break
            if not found:
                # Then this is not a duplicate.
                # See if the interval between the maximum and the minimum matches the scan speed.
                interval_start_time = time_maximum + self.scan_establish_velocity_time
                interval_end_time = time_minimum - self.scan_discard_velocity_time
                interval_delta_time = interval_end_time - interval_start_time
                if interval_delta_time >= self.minimum_scan_pass_time:
                    # print('In TrajectoryAnalysis.__init__(), [interval_start_time, interval_end_time] = [',interval_start_time, ',', interval_end_time,']; interval_delta_time = ', interval_delta_time)  # ?? SCAFFOLDING RCB -- TEMPORARY
                    # Then this is long enough to be a valid test.
                    # Compute average speed over the interval.
                    speed_sum = 0
                    speed_count = 0
                    for idx in self.flight_log_df.index:
                        time = self.flight_log_df.loc[idx, "time(sec)"]
                        if (interval_start_time <= time) and (time <= interval_end_time):
                            speed_sum += self.flight_log_df.loc[idx, "speed_average(m/sec)"]
                            speed_count += 1
                    if speed_count <= 0:
                        print(
                            "In TrajectoryAnalysis.__init__(), encountered unexpected non-positive speed_count =",
                            speed_count,
                        )
                        assert False
                    interval_average_speed = speed_sum / speed_count
                    min_scan_speed = self.nominal_scan_speed - self.scan_speed_tolerance
                    max_scan_speed = self.nominal_scan_speed + self.scan_speed_tolerance
                    # print('In TrajectoryAnalysis.__init__(), [min_scan_speed, max_scan_speed] = [',min_scan_speed, ',',max_scan_speed,']; interval_average_speed = ', interval_average_speed)  # ?? SCAFFOLDING RCB -- TEMPORARY
                    if (min_scan_speed <= interval_average_speed) and (interval_average_speed <= max_scan_speed):
                        # Then the intervening motion speed matches the expected speed.
                        # ?? SCAFFOLDING RCB -- THIS CODE REPLICATION IS GETTING SERIOUSLY UGLY.  E.G., WRITE A ROUTINE TO FETCH AVERAGE VALUE OF ARBITRARY PARAMETER.
                        # See if the interval between the maximum and the minimum matches the scan z velocity.
                        # Compute average z velocity over the interval.
                        velocity_z_sum = 0
                        velocity_z_count = 0
                        for idx in self.flight_log_df.index:
                            time = self.flight_log_df.loc[idx, "time(sec)"]
                            if (interval_start_time <= time) and (time <= interval_end_time):
                                velocity_z_sum += self.flight_log_df.loc[idx, "velocity_average_z(m/sec)"]
                                velocity_z_count += 1
                        if velocity_z_count <= 0:
                            print(
                                "In TrajectoryAnalysis.__init__(), encountered unexpected non-positive velocity_z_count =",
                                velocity_z_count,
                            )
                            assert False
                        interval_average_velocity_z = velocity_z_sum / velocity_z_count
                        min_scan_velocity_z = self.nominal_scan_velocity_z - self.scan_velocity_z_tolerance
                        max_scan_velocity_z = self.nominal_scan_velocity_z + self.scan_velocity_z_tolerance
                        # print('In TrajectoryAnalysis.__init__(), [min_scan_velocity_z, max_scan_velocity_z] = [',min_scan_velocity_z, ',',max_scan_velocity_z,']; interval_average_velocity_z = ', interval_average_velocity_z)  # ?? SCAFFOLDING RCB -- TEMPORARY
                        if (min_scan_velocity_z <= interval_average_velocity_z) and (
                            interval_average_velocity_z <= max_scan_velocity_z
                        ):
                            # Then the intervening motion velocity_z matches the expected velocity_z.
                            # The [time_maximum, time_minimum] interval passes all tests.
                            maximum_to_minimum_pass_pair_list.append(maximum_to_minimum_pair)
        # Return.
        return maximum_to_minimum_pass_pair_list

    def find_velocity_xy_change_minimum_to_maximum_scan_passes(
        self,
    ):  # ?? SCAFFOLDING RCB -- THIS ROUTINE DUPLICATES ITS PAIR.  CAN THEY BE MERGED, WHILE RETAINING CORRECTNESS AND CLARITY?
        minimum_to_maximum_pass_pair_list = []
        for minimum_dict in self.gps_velocity_xy_change_minima:  # Here "minimum" means "minimum_velocity_xy_change"
            # Here is a local minimum time to work with.
            time_minimum = minimum_dict[
                "time"
            ]  # Here "time_minimum" means "time of the minimum_velocity_xy_change," not the maximum time.
            # Find the next local maximum time.
            for maximum_dict in self.gps_velocity_xy_change_maxima:
                time_maximum = maximum_dict["time"]
                if time_maximum > (time_minimum + self.turn_overshoot_skip_time):
                    break
            # See if there is a local minimum in between.
            for alternative_minimum_dict in self.gps_velocity_xy_change_minima:
                alternative_time_minimum = alternative_minimum_dict["time"]
                if (alternative_time_minimum > time_minimum) and (alternative_time_minimum < time_maximum):
                    minimum_dict = alternative_minimum_dict
                if alternative_time_minimum > time_maximum:
                    break
            # Now the maximum and minimum times are adjacent.
            minimum_to_maximum_pair = [minimum_dict, maximum_dict]
            # Avoid duplicate entries.
            found = False
            time_minimum = minimum_dict["time"]  # Refresh in case it has changed.
            for search_minimum_to_maximum_pair in minimum_to_maximum_pass_pair_list:
                search_minimum_dict = search_minimum_to_maximum_pair[0]
                search_time_minimum = search_minimum_dict["time"]
                if search_time_minimum == time_minimum:
                    found = True
                    break
            if not found:
                # Then this is not a duplicate.
                # See if the interval between the minimum and the maximum matches the scan speed.
                interval_start_time = time_minimum + self.scan_establish_velocity_time
                interval_end_time = time_maximum - self.scan_discard_velocity_time
                interval_delta_time = interval_end_time - interval_start_time
                if interval_delta_time >= self.minimum_scan_pass_time:
                    # print('In TrajectoryAnalysis.__init__(), [interval_start_time, interval_end_time] = [',interval_start_time, ',', interval_end_time,']; interval_delta_time = ', interval_delta_time)  # ?? SCAFFOLDING RCB -- TEMPORARY
                    # Then this is long enough to be a valid test.
                    # Compute average speed over the interval.
                    speed_sum = 0
                    speed_count = 0
                    for idx in self.flight_log_df.index:
                        time = self.flight_log_df.loc[idx, "time(sec)"]
                        if (interval_start_time <= time) and (time <= interval_end_time):
                            speed_sum += self.flight_log_df.loc[idx, "speed_average(m/sec)"]
                            speed_count += 1
                    if speed_count <= 0:
                        print(
                            "In TrajectoryAnalysis.__init__(), encountered unexpected non-positive speed_count =",
                            speed_count,
                        )
                        assert False
                    interval_average_speed = speed_sum / speed_count
                    min_scan_speed = self.nominal_scan_speed - self.scan_speed_tolerance
                    max_scan_speed = self.nominal_scan_speed + self.scan_speed_tolerance
                    # print('In TrajectoryAnalysis.__init__(), [min_scan_speed, max_scan_speed] = [',min_scan_speed, ',',max_scan_speed,']; interval_average_speed = ', interval_average_speed)  # ?? SCAFFOLDING RCB -- TEMPORARY
                    if (min_scan_speed <= interval_average_speed) and (interval_average_speed <= max_scan_speed):
                        # Then the intervening motion speed matches the expected speed.
                        # ?? SCAFFOLDING RCB -- THIS CODE REPLICATION IS GETTING SERIOUSLY UGLY.  E.G., WRITE A ROUTINE TO FETCH AVERAGE VALUE OF ARBITRARY PARAMETER.
                        # See if the interval between the maximum and the minimum matches the scan z velocity.
                        # Compute average z velocity over the interval.
                        velocity_z_sum = 0
                        velocity_z_count = 0
                        for idx in self.flight_log_df.index:
                            time = self.flight_log_df.loc[idx, "time(sec)"]
                            if (interval_start_time <= time) and (time <= interval_end_time):
                                velocity_z_sum += self.flight_log_df.loc[idx, "velocity_average_z(m/sec)"]
                                velocity_z_count += 1
                        if velocity_z_count <= 0:
                            print(
                                "In TrajectoryAnalysis.__init__(), encountered unexpected non-positive velocity_z_count =",
                                velocity_z_count,
                            )
                            assert False
                        interval_average_velocity_z = velocity_z_sum / velocity_z_count
                        min_scan_velocity_z = self.nominal_scan_velocity_z - self.scan_velocity_z_tolerance
                        max_scan_velocity_z = self.nominal_scan_velocity_z + self.scan_velocity_z_tolerance
                        # print('In TrajectoryAnalysis.__init__(), [min_scan_velocity_z, max_scan_velocity_z] = [',min_scan_velocity_z, ',',max_scan_velocity_z,']; interval_average_velocity_z = ', interval_average_velocity_z)  # ?? SCAFFOLDING RCB -- TEMPORARY
                        if (min_scan_velocity_z <= interval_average_velocity_z) and (
                            interval_average_velocity_z <= max_scan_velocity_z
                        ):
                            # Then the intervening motion velocity_z matches the expected velocity_z.
                            # The [time_minimum, time_maximum] interval passes all tests.
                            minimum_to_maximum_pass_pair_list.append(minimum_to_maximum_pair)
        # Return.
        return minimum_to_maximum_pass_pair_list

    def start_stop_pass_pair_start_time(self, start_stop_pass_pair):
        first_dict = start_stop_pass_pair[0]
        return first_dict["time"]

    def xyzt_list_given_gps_start_stop_pass_pair(self, start_stop_pass_pair):
        # Fetch pass start and end trajectory points.
        start_dict = start_stop_pass_pair[0]
        start_time = start_dict["time"]
        stop_dict = start_stop_pass_pair[1]
        stop_time = stop_dict["time"]
        # Fetch trajectory points along pass.
        xyzt_list = []
        for (
            idx
        ) in (
            self.flight_log_df.index
        ):  # ?? SCAFFOLDING RCB -- THIS FOR LOOP IS VERY SLOW.  CAN WE VECTORIZE IT?  REPLACE THE .loc[] CALLS WITH SOMETHING FASTER?
            time = self.flight_log_df.loc[idx, "time(sec)"]
            if (start_time <= time) and (time <= stop_time):
                x = self.flight_log_df.loc[idx, "x(m)"]
                y = self.flight_log_df.loc[idx, "y(m)"]
                z = self.flight_log_df.loc[idx, "z(m)"]
                xyzt_list.append([x, y, z, time])
        # Return.
        return xyzt_list

    # GPS SCAN PASS SETUP

    def convert_gps_minima_maxima_passes_to_scan_passes(self):
        # self.maximum_to_minimum_pass_pair_list = [ # ?? SCAFFOLDING RCB -- Values computed November 27 at 9:50 AM, hard-coded as a development shortcut.
        #                                             [{'time': 64.788, 'delta_velocity_angle_xy': 4.004437659559161, 'x': -94.70118828663304, 'y': 32.555555555435284, 'z': 16.7}, {'time': 83.801, 'delta_velocity_angle_xy': -0.6730385584829808, 'x': -126.02542748931391, 'y': 159.88888888847873, 'z': 17.0}],
        #                                             [{'time': 113.868, 'delta_velocity_angle_xy': 2.1180735327682347, 'x': -78.49271567537926, 'y': 31.888888888873474, 'z': 17.4}, {'time': 135.683, 'delta_velocity_angle_xy': -2.193623615020262, 'x': -109.36166070474988, 'y': 178.99999999947883, 'z': 16.9}],
        #                                             [{'time': 164.3, 'delta_velocity_angle_xy': 2.128043879286733, 'x': -62.28424306541952, 'y': 31.444444444762095, 'z': 17.9}, {'time': 186.065, 'delta_velocity_angle_xy': -2.0925600192178817, 'x': -88.69130518470344, 'y': 178.88888888785888, 'z': 17.3}],
        #                                             [{'time': 214.778, 'delta_velocity_angle_xy': 2.0884747492212674, 'x': -46.34894695815189, 'y': 30.999999999861227, 'z': 18.4}, {'time': 236.495, 'delta_velocity_angle_xy': -1.911432763924814, 'x': -67.83883199403887, 'y': 178.99999999947883, 'z': 18.0}],
        #                                             [{'time': 265.264, 'delta_velocity_angle_xy': 2.0310077469114756, 'x': -31.233180368018818, 'y': 30.555555554960357, 'z': 18.6}, {'time': 286.773, 'delta_velocity_angle_xy': -0.6582109130248106, 'x': -46.804241134050194, 'y': 179.33333333275974, 'z': 18.4}],
        #                                             [{'time': 319.696, 'delta_velocity_angle_xy': 1.9505682633483257, 'x': -18.12070814303891, 'y': 29.888888888398544, 'z': 19.1}, {'time': 344.168, 'delta_velocity_angle_xy': -1.936084871537165, 'x': -25.041179595471093, 'y': 200.3333333326148, 'z': 18.9}],
        #                                             [{'time': 375.481, 'delta_velocity_angle_xy': 1.877628579737848, 'x': -4.552941744748746, 'y': 29.444444443497677, 'z': 19.3}, {'time': 400.095, 'delta_velocity_angle_xy': -1.9062519966211062, 'x': -2.1854120370653103, 'y': 200.44444444344526, 'z': 19.5}],
        #                                             [{'time': 431.464, 'delta_velocity_angle_xy': 1.8389675548324715, 'x': 8.104236306920912, 'y': 28.999999999386297, 'z': 19.5}, {'time': 455.976, 'delta_velocity_angle_xy': -1.8213845026733095, 'x': 21.034590859988672, 'y': 200.44444444344526, 'z': 20.0}],
        #                                             [{'time': 487.498, 'delta_velocity_angle_xy': 1.6841468919777443, 'x': 19.75976717342594, 'y': 28.777777776935864, 'z': 19.9}, {'time': 512.466, 'delta_velocity_angle_xy': -1.7988202495261416, 'x': 44.80094676760301, 'y': 200.66666666589572, 'z': 20.4}],
        #                                             [{'time': 544.183, 'delta_velocity_angle_xy': 1.6237993453565485, 'x': 30.868945029370618, 'y': 28.666666666894884, 'z': 19.9}, {'time': 569.098, 'delta_velocity_angle_xy': -2.921698723806162, 'x': 68.8404791792035, 'y': 200.55555555506524, 'z': 21.0}],
        #                                             [{'time': 599.57, 'delta_velocity_angle_xy': 1.5217142628192941, 'x': 42.16024055593342, 'y': 28.55555555527492, 'z': 20.2}, {'time': 622.18, 'delta_velocity_angle_xy': -1.599670420101639, 'x': 85.41318712910547, 'y': 179.5555555552102, 'z': 21.3}],
        #                                             [{'time': 651.702, 'delta_velocity_angle_xy': 1.380303950150691, 'x': 52.35883006266954, 'y': 28.888888888555826, 'z': 20.4}, {'time': 674.666, 'delta_velocity_angle_xy': -1.5172965402297607, 'x': 108.26895468751125, 'y': 179.66666666604064, 'z': 21.8}],
        #                                             [{'time': 702.585, 'delta_velocity_angle_xy': 3.31857734736965, 'x': 56.91177180741829, 'y': 28.888888888555826, 'z': 20.7}, {'time': 727.296, 'delta_velocity_angle_xy': -1.4238712110423668, 'x': 131.58001641793325, 'y': 179.4444444435902, 'z': 22.3}]
        #                                          ]

        # self.minimum_to_maximum_pass_pair_list = [ # ?? SCAFFOLDING RCB -- Values computed November 27 at 9:50 AM, hard-coded as a development shortcut.
        #                                             [{'time': 88.455, 'delta_velocity_angle_xy': -2.7922927288855375, 'x': -119.28707370749986, 'y': 178.7777777770284, 'z': 17.0}, {'time': 110.465, 'delta_velocity_angle_xy': 1.71203134810327, 'x': -87.41648149555266, 'y': 31.77777777725351, 'z': 17.2}],
        #                                             [{'time': 139.587, 'delta_velocity_angle_xy': -1.6839803405464198, 'x': -98.4346005181293, 'y': 178.55555555536745, 'z': 17.2}, {'time': 161.197, 'delta_velocity_angle_xy': 1.741753856630214, 'x': -70.75271471098863, 'y': 31.22222222231166, 'z': 17.8}],
        #                                             [{'time': 189.869, 'delta_velocity_angle_xy': -1.7431128378521954, 'x': -77.76424499808286, 'y': 178.55555555536745, 'z': 17.7}, {'time': 211.581, 'delta_velocity_angle_xy': 1.8012770703210967, 'x': -54.63530093569093, 'y': 30.77777777741079, 'z': 18.1}],
        #                                             [{'time': 240.294, 'delta_velocity_angle_xy': -1.8064807069641795, 'x': -56.82071297275624, 'y': 178.66666666619793, 'z': 18.2}, {'time': 262.11, 'delta_velocity_angle_xy': 1.7580437090833776, 'x': -39.155299004321606, 'y': 30.444444444129886, 'z': 18.5}],
        #                                             [{'time': 292.178, 'delta_velocity_angle_xy': -2.750232538296551, 'x': -35.968239782091665, 'y': 200.3333333326148, 'z': 18.7}, {'time': 316.694, 'delta_velocity_angle_xy': 1.7997784729505129, 'x': -25.678591438105446, 'y': 29.777777777568073, 'z': 18.8}],
        #                                             [{'time': 348.07, 'delta_velocity_angle_xy': -1.950650923877637, 'x': -13.112472223685886, 'y': 199.99999999933388, 'z': 19.2}, {'time': 372.484, 'delta_velocity_angle_xy': 1.9310588790751277, 'x': -11.74658970116708, 'y': 29.22222222183673, 'z': 19.1}],
        #                                             [{'time': 403.999, 'delta_velocity_angle_xy': -1.9528555033835737, 'x': 9.925413004043996, 'y': 199.99999999933388, 'z': 19.7}, {'time': 428.462, 'delta_velocity_angle_xy': 1.9330070224257558, 'x': 1.365882525106855, 'y': 28.888888888555826, 'z': 19.4}],
        #                                             [{'time': 459.881, 'delta_velocity_angle_xy': -2.0449409914944034, 'x': 33.3275335717161, 'y': 200.11111111095386, 'z': 20.1}, {'time': 484.696, 'delta_velocity_angle_xy': 1.9488603768173913, 'x': 13.476707563628112, 'y': 28.44444444444445, 'z': 19.7}],
        #                                             [{'time': 516.265, 'delta_velocity_angle_xy': -2.078594616844208, 'x': 57.18494831269847, 'y': 200.22222222178434, 'z': 20.7}, {'time': 541.482, 'delta_velocity_angle_xy': 2.0689682169838313, 'x': 24.859061924852966, 'y': 28.222222221994013, 'z': 19.9}],
        #                                             [{'time': 574.703, 'delta_velocity_angle_xy': -0.6347544737482604, 'x': 74.12189160383669, 'y': 178.66666666619793, 'z': 21.2}, {'time': 596.869, 'delta_velocity_angle_xy': 2.166487711239905, 'x': 36.51459279135799, 'y': 28.00000000033307, 'z': 20.2}],
        #                                             [{'time': 625.987, 'delta_velocity_angle_xy': -2.216872168312239, 'x': 97.25083566622862, 'y': 179.1111111103093, 'z': 21.5}, {'time': 649.002, 'delta_velocity_angle_xy': 2.2505181713701052, 'x': 46.62212346602011, 'y': 28.333333332824484, 'z': 20.3}],
        #                                             [{'time': 678.367, 'delta_velocity_angle_xy': -2.3024949069575356, 'x': 120.01554438867834, 'y': 179.22222222192926, 'z': 21.9}, {'time': 701.685, 'delta_velocity_angle_xy': 3.127478264277907, 'x': 56.91177180741829, 'y': 28.888888888555826, 'z': 20.7}],
        #                                             [{'time': 731.095, 'delta_velocity_angle_xy': -2.2472725229256376, 'x': 143.0534296151142, 'y': 179.1111111103093, 'z': 22.5}, {'time': 743.657, 'delta_velocity_angle_xy': 2.341259471342911, 'x': 109.81695488064818, 'y': 107.77777777793199, 'z': 22.2}]
        #                                          ]

        # Combine max-to-min and min-to-max pass pairs, and sort in order of increasing time.
        start_stop_pass_pair_list = self.maximum_to_minimum_pass_pair_list + self.minimum_to_maximum_pass_pair_list
        start_stop_pass_pair_list.sort(key=self.start_stop_pass_pair_start_time)

        # start_stop_pass_pair_list = [ # ?? SCAFFOLDING RCB -- Values computed November 16 at 7:30 AM, hard-coded as a development shortcut.
        #                               [{'time':  64.788, 'delta_velocity_angle_xy':  4.004437659559161,  'x':  -94.70118828663304,  'y':  32.555555555435284, 'z': 16.7}, {'time':  83.801, 'delta_velocity_angle_xy': -0.6730385584829808, 'x': -126.02542748931391, 'y': 159.88888888847873,  'z': 17.0}],
        #                               [{'time':  88.455, 'delta_velocity_angle_xy': -2.7922927288855375, 'x': -119.28707370749986,  'y': 178.7777777770284,   'z': 17.0}, {'time': 110.465, 'delta_velocity_angle_xy':  1.71203134810327,   'x':  -87.41648149555266, 'y': 31.77777777725351,   'z': 17.2}],
        #                               [{'time': 113.868, 'delta_velocity_angle_xy':  2.1180735327682347, 'x':  -78.49271567537926,  'y':  31.888888888873474, 'z': 17.4}, {'time': 135.683, 'delta_velocity_angle_xy': -2.193623615020262,  'x': -109.36166070474988, 'y': 178.99999999947883,  'z': 16.9}],
        #                               [{'time': 139.587, 'delta_velocity_angle_xy': -1.6839803405464198, 'x':  -98.4346005181293,   'y': 178.55555555536745,  'z': 17.2}, {'time': 161.197, 'delta_velocity_angle_xy':  1.741753856630214,  'x':  -70.75271471098863,  'y': 31.22222222231166,  'z': 17.8}],
        #                               [{'time': 164.3,   'delta_velocity_angle_xy':  2.128043879286733,  'x':  -62.28424306541952,  'y':  31.444444444762095, 'z': 17.9}, {'time': 186.065, 'delta_velocity_angle_xy': -2.0925600192178817, 'x':  -88.69130518470344,  'y': 178.88888888785888, 'z': 17.3}],
        #                               [{'time': 189.869, 'delta_velocity_angle_xy': -1.7431128378521954, 'x':  -77.76424499808286,  'y': 178.55555555536745,  'z': 17.7}, {'time': 211.581, 'delta_velocity_angle_xy':  1.8012770703210967, 'x':  -54.63530093569093,  'y': 30.77777777741079,  'z': 18.1}],
        #                               [{'time': 214.778, 'delta_velocity_angle_xy':  2.0884747492212674, 'x':  -46.34894695815189,  'y':  30.999999999861227, 'z': 18.4}, {'time': 236.495, 'delta_velocity_angle_xy': -1.911432763924814,  'x':  -67.83883199403887,  'y': 178.99999999947883, 'z': 18.0}],
        #                               [{'time': 240.294, 'delta_velocity_angle_xy': -1.8064807069641795, 'x':  -56.82071297275624,  'y': 178.66666666619793,  'z': 18.2}, {'time': 262.11,  'delta_velocity_angle_xy':  1.7580437090833776, 'x':  -39.155299004321606, 'y': 30.444444444129886, 'z': 18.5}],
        #                               [{'time': 265.264, 'delta_velocity_angle_xy':  2.0310077469114756, 'x':  -31.233180368018818, 'y':  30.555555554960357, 'z': 18.6}, {'time': 286.773, 'delta_velocity_angle_xy': -0.6582109130248106, 'x':  -46.804241134050194, 'y': 179.33333333275974, 'z': 18.4}],
        #                               [{'time': 292.178, 'delta_velocity_angle_xy': -2.750232538296551,  'x':  -35.968239782091665, 'y': 200.3333333326148,   'z': 18.7}, {'time': 316.694, 'delta_velocity_angle_xy':  1.7997784729505129, 'x':  -25.678591438105446, 'y': 29.777777777568073, 'z': 18.8}],
        #                               [{'time': 319.696, 'delta_velocity_angle_xy':  1.9505682633483257, 'x':  -18.12070814303891,  'y':  29.888888888398544, 'z': 19.1}, {'time': 344.168, 'delta_velocity_angle_xy': -1.936084871537165,  'x':  -25.041179595471093, 'y': 200.3333333326148,  'z': 18.9}],
        #                               [{'time': 348.07,  'delta_velocity_angle_xy': -1.950650923877637,  'x':  -13.112472223685886, 'y': 199.99999999933388,  'z': 19.2}, {'time': 372.484, 'delta_velocity_angle_xy':  1.9310588790751277, 'x':  -11.74658970116708,  'y': 29.22222222183673,  'z': 19.1}],
        #                               [{'time': 375.481, 'delta_velocity_angle_xy':  1.877628579737848,  'x':   -4.552941744748746, 'y':  29.444444443497677, 'z': 19.3}, {'time': 400.095, 'delta_velocity_angle_xy': -1.9062519966211062, 'x':  -2.1854120370653103, 'y': 200.44444444344526, 'z': 19.5}],
        #                               [{'time': 403.999, 'delta_velocity_angle_xy': -1.9528555033835737, 'x':    9.925413004043996, 'y': 199.99999999933388,  'z': 19.7}, {'time': 428.462, 'delta_velocity_angle_xy':  1.9330070224257558, 'x':   1.365882525106855,  'y': 28.888888888555826, 'z': 19.4}],
        #                               [{'time': 431.464, 'delta_velocity_angle_xy':  1.8389675548324715, 'x':    8.104236306920912, 'y':  28.999999999386297, 'z': 19.5}, {'time': 455.976, 'delta_velocity_angle_xy': -1.8213845026733095, 'x':  21.034590859988672,  'y': 200.44444444344526, 'z': 20.0}],
        #                               [{'time': 459.881, 'delta_velocity_angle_xy': -2.0449409914944034, 'x':   33.3275335717161,   'y': 200.11111111095386,  'z': 20.1}, {'time': 484.696, 'delta_velocity_angle_xy':  1.9488603768173913, 'x':  13.476707563628112,  'y': 28.44444444444445,  'z': 19.7}],
        #                               [{'time': 487.498, 'delta_velocity_angle_xy':  1.6841468919777443, 'x':   19.75976717342594,  'y':  28.777777776935864, 'z': 19.9}, {'time': 512.466, 'delta_velocity_angle_xy': -1.7988202495261416, 'x':  44.80094676760301,   'y': 200.66666666589572, 'z': 20.4}],
        #                               [{'time': 516.265, 'delta_velocity_angle_xy': -2.078594616844208,  'x':   57.18494831269847,  'y': 200.22222222178434,  'z': 20.7}, {'time': 541.482, 'delta_velocity_angle_xy':  2.0689682169838313, 'x':  24.859061924852966,  'y': 28.222222221994013, 'z': 19.9}],
        #                               [{'time': 544.183, 'delta_velocity_angle_xy':  1.6237993453565485, 'x':   30.868945029370618, 'y':  28.666666666894884, 'z': 19.9}, {'time': 569.098, 'delta_velocity_angle_xy': -2.921698723806162,  'x':  68.8404791792035,    'y': 200.55555555506524, 'z': 21.0}],
        #                               [{'time': 574.703, 'delta_velocity_angle_xy': -0.6347544737482604, 'x':   74.12189160383669,  'y': 178.66666666619793,  'z': 21.2}, {'time': 596.869, 'delta_velocity_angle_xy':  2.166487711239905,  'x':  36.51459279135799,   'y': 28.00000000033307,  'z': 20.2}],
        #                               [{'time': 599.57,  'delta_velocity_angle_xy':  1.5217142628192941, 'x':   42.16024055593342,  'y':  28.55555555527492,  'z': 20.2}, {'time': 622.18,  'delta_velocity_angle_xy': -1.599670420101639,  'x':  85.41318712910547,   'y': 179.5555555552102,  'z': 21.3}],
        #                               [{'time': 625.987, 'delta_velocity_angle_xy': -2.216872168312239,  'x':   97.25083566622862,  'y': 179.1111111103093,   'z': 21.5}, {'time': 649.002, 'delta_velocity_angle_xy':  2.2505181713701052, 'x':  46.62212346602011,   'y': 28.333333332824484, 'z': 20.3}],
        #                               [{'time': 651.702, 'delta_velocity_angle_xy':  1.380303950150691,  'x':   52.35883006266954,  'y':  28.888888888555826, 'z': 20.4}, {'time': 674.666, 'delta_velocity_angle_xy': -1.5172965402297607, 'x': 108.26895468751125,   'y': 179.66666666604064, 'z': 21.8}],
        #                               [{'time': 678.367, 'delta_velocity_angle_xy': -2.3024949069575356, 'x':  120.01554438867834,  'y': 179.22222222192926,  'z': 21.9}, {'time': 701.685, 'delta_velocity_angle_xy':  3.127478264277907,  'x':  56.91177180741829,   'y': 28.888888888555826, 'z': 20.7}],
        #                               [{'time': 702.585, 'delta_velocity_angle_xy':  3.31857734736965,   'x':   56.91177180741829,  'y':  28.888888888555826, 'z': 20.7}, {'time': 727.296, 'delta_velocity_angle_xy': -1.4238712110423668, 'x': 131.58001641793325,   'y': 179.4444444435902,  'z': 22.3}],
        #                               [{'time': 731.095, 'delta_velocity_angle_xy': -2.2472725229256376, 'x':  143.0534296151142,   'y': 179.1111111103093,   'z': 22.5}, {'time': 743.657, 'delta_velocity_angle_xy':  2.341259471342911,  'x': 109.81695488064818,   'y': 107.77777777793199, 'z': 22.2}]
        #                             ]

        # Construct scan pass objects, discarding start and end points where velocity is not yet stabilized.
        gps_scan_passes = []
        for start_stop_pass_pair in start_stop_pass_pair_list:
            # Fetch trajectory points along pass.
            xyzt_list = self.xyzt_list_given_gps_start_stop_pass_pair(start_stop_pass_pair)
            # If there is a pause, break the list up.
            list_of_xyzt_lists_1 = self.split_xyzt_list_where_inter_point_speed_below_minimum(
                xyzt_list, self.minimum_gps_pass_inter_point_speed
            )
            # Discard lists that are too short.
            list_of_xyzt_lists_2 = self.discard_xyzt_lists_where_length_below_minimum(
                list_of_xyzt_lists_1, self.minimum_gps_pass_number_of_points
            )
            # Construct scan pass objects.
            for xyzt_list in list_of_xyzt_lists_2:
                # Construct the scan pass object.
                if len(xyzt_list) >= 2:
                    gps_scan_pass = self.scan_pass_given_xyzt_list(
                        xyzt_list, n_start_margin=self.gps_pass_start_margin, n_stop_margin=self.gps_pass_stop_margin
                    )
                    # Add to list.
                    gps_scan_passes.append(gps_scan_pass)

        # Return.
        return gps_scan_passes

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  SCAN PASS REPRESENTATION
    #
    # This represents both GPS and camera scan passes.
    #
    # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS.

    def scan_pass_given_xyzt_list(
        self, xyzt_list, n_start_margin, n_stop_margin
    ):  # ?? SCAFFOLDING RCB -- THIS START_MARGIN, STOP_MARGIN IS A STUB HACK.
        # Refine pass, collecting statistical variation data.
        (stable_begin_xyzt, stable_end_xyzt, line_3d, inlier_xyzt_list, distance_to_line_list, rms_distance_to_line) = (
            self.refine_xyzt_list(xyzt_list, n_start_margin, n_stop_margin)
        )
        # Construct scan pass object.
        scan_pass = (
            {}
        )  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS, SUITABLE FOR BOTH GPS SCAN PASSES AND TRAJECTORY FRAGMENTS.
        scan_pass["stable_begin_xyzt"] = stable_begin_xyzt
        scan_pass["stable_end_xyzt"] = stable_end_xyzt
        scan_pass["line_3d"] = line_3d
        scan_pass["inlier_xyzt_list"] = inlier_xyzt_list
        scan_pass["distance_to_line_list"] = distance_to_line_list
        scan_pass["rms_distance_to_line"] = rms_distance_to_line
        # Return.
        return scan_pass

    # ?? SCAFFOLDING RCB -- STUB HACK.  REPLACE STOP AND START MARGINS WITH A CALCULATION THAT REMOVES POINTS BASED ON WHETHER THEY ARE INLIERS OR OUTLIERS.
    def refine_xyzt_list(self, xyzt_list, n_start_margin, n_stop_margin=20):
        if len(xyzt_list) < 2:
            print(
                "ERROR: In TrajectoryAnalysis.refine_xyzt_list(), encountered insufficient xyzt points.  len(xyzt_list) =",
                len(xyzt_list),
            )
            assert False
        if len(xyzt_list) <= (
            n_start_margin + n_stop_margin
        ):  # ?? SCAFFOLDING RCB -- STUB HACK.  REPLACE THIS WITH A CALCULATION THAT REMOVED POINTS BASED ON WHETHER THEY ARE INLIERS OR OUTLIERS.
            print(
                "WARNING: In TrajectoryAnalysis.refine_xyzt_list(), encountered insufficient xyzt points; skipping margins.  len(xyzt_list) =",
                len(xyzt_list),
            )
            inlier_xyzt_list = xyzt_list
        else:
            inlier_xyzt_list = xyzt_list[n_start_margin : len(xyzt_list) - n_stop_margin]
        stable_begin_xyzt = inlier_xyzt_list[
            0
        ]  # ?? SCAFFOLDING RCB -- STUB HACK.  THIS SHOULD USE A FIT LINE, NOT FIRST AND LAST POINTS.
        stable_end_xyzt = inlier_xyzt_list[
            -1
        ]  # ?? SCAFFOLDING RCB -- STUB HACK.  THIS SHOULD USE A FIT LINE, NOT FIRST AND LAST POINTS.
        line_3d_xyz_1 = stable_begin_xyzt[0:3]
        line_3d_xyz_2 = stable_end_xyzt[0:3]
        line_3d = g3d.construct_line_3d_given_two_points(
            line_3d_xyz_1, line_3d_xyz_2
        )  # ?? SCAFFOLDING RCB -- LINE-3D SHOULD BE A CLASS.
        distance_to_line_list = [g3d.distance_to_line_3d(xyzt[0:3], line_3d) for xyzt in inlier_xyzt_list]
        rms_distance_to_line = mt.rms(distance_to_line_list)
        # Return.
        return (
            stable_begin_xyzt,
            stable_end_xyzt,
            line_3d,
            inlier_xyzt_list,
            distance_to_line_list,
            rms_distance_to_line,
        )

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  LOAD TRAJECTORY FRAGMENTS
    #
    #

    def load_trajectory_fragments(self):
        # Load trajectory fragments from previous 3-d reconstruction step.
        hel_frames_dict = {}
        for hel_name in self.solar_field.heliostat_name_list():
            if ft.directory_exists(self.input_reconstructed_heliostats_dir):
                reconstructed_hel_dir = os.path.join(self.input_reconstructed_heliostats_dir, hel_name)
                if ft.directory_exists(reconstructed_hel_dir):
                    frame_parameters_dir_body_ext = os.path.join(
                        reconstructed_hel_dir, hel_name + "_frame_dict_parameters.csv"
                    )
                    if not ft.file_exists(frame_parameters_dir_body_ext):
                        print(
                            "ERROR: In TrajectoryAnalysis.load_trajectory_fragments(), expected file does not exist: "
                            + frame_parameters_dir_body_ext
                        )
                        assert False
                    else:
                        # print('In TrajectoryAnalysis.load_trajectory_fragments(), reading heliostat frame parameters:', frame_parameters_dir_body_ext)
                        frame_parameters_df = pd.read_csv(frame_parameters_dir_body_ext)
                        # Convert to  dict, so we can extend it.
                        frames_dict = {}
                        for idx in frame_parameters_df.index:
                            frame_id = frame_parameters_df.loc[idx, "frame_id"]
                            rvec_x = frame_parameters_df.loc[idx, "rvec_x"]
                            rvec_y = frame_parameters_df.loc[idx, "rvec_y"]
                            rvec_z = frame_parameters_df.loc[idx, "rvec_z"]
                            tvec_x = frame_parameters_df.loc[idx, "tvec_x"]
                            tvec_y = frame_parameters_df.loc[idx, "tvec_y"]
                            tvec_z = frame_parameters_df.loc[idx, "tvec_z"]
                            n_missing = frame_parameters_df.loc[idx, "n_missing"]
                            frame_parameters_dict = {}
                            # ?? SCAFFOLDING RCB -- THIS IS ONE AMONG MANY PLACES WHERE THINGS WOULD RUN FASTER IF WE WERE STORING AND PASSING AROUND NUMPY ARRAYS.  LONG TERM: FIX THIS THROUGHOUT.
                            frame_parameters_dict["rvec_xyz"] = [rvec_x, rvec_y, rvec_z]
                            # ?? SCAFFOLDING RCB -- THIS IS ONE AMONG MANY PLACES WHERE THINGS WOULD RUN FASTER IF WE WERE STORING AND PASSING AROUND NUMPY ARRAYS.  LONG TERM: FIX THIS THROUGHOUT.
                            frame_parameters_dict["tvec_xyz"] = [tvec_x, tvec_y, tvec_z]
                            frame_parameters_dict["n_missing"] = n_missing
                            frames_dict[frame_id] = frame_parameters_dict
                        hel_frames_dict[hel_name] = frames_dict

        # Add camera transforms in heliostat coordinates.
        # ?? SCAFFOLDING RCB -- THIS SHOULD HAVE BEEN DONE IN PREVIOUS STEP.
        self.add_camera_transforms(hel_frames_dict)

        # Return.
        return hel_frames_dict

    def add_camera_transforms(self, hel_frames_dict):
        for hel_name in hel_frames_dict.keys():
            frames_dict = hel_frames_dict[hel_name]
            for frame_id in frames_dict.keys():
                frame_parameters_dict = frames_dict[frame_id]
                rvec_xyz = frame_parameters_dict["rvec_xyz"]
                tvec_xyz = frame_parameters_dict["tvec_xyz"]
                # For an excellent explanation of camera coordinates and transforms, see:
                #    https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
                # Convert the Rodrigues vector rvec to a rotation matrix.
                rvec_rotation_matrix, rvec_jacobian = cv.Rodrigues(np.array([rvec_xyz]))
                # Construct a homogeneous transform that will convert heliostat coordinates to camera coordinates.
                # (Where "heliostat coordinates" refers to the coordinate system at the heliostat's end effector.)
                transform = np.zeros([4, 4])
                transform[0, 0] = rvec_rotation_matrix[
                    0, 0
                ]  # ?? SCAFFOLDING RCB -- MAYBE THERE'S A CLEANER/FASTER WAY TO DO THIS.
                transform[0, 1] = rvec_rotation_matrix[0, 1]
                transform[0, 2] = rvec_rotation_matrix[0, 2]
                transform[1, 0] = rvec_rotation_matrix[1, 0]
                transform[1, 1] = rvec_rotation_matrix[1, 1]
                transform[1, 2] = rvec_rotation_matrix[1, 2]
                transform[2, 0] = rvec_rotation_matrix[2, 0]
                transform[2, 1] = rvec_rotation_matrix[2, 1]
                transform[2, 2] = rvec_rotation_matrix[2, 2]
                transform[0, 3] = tvec_xyz[0]
                transform[1, 3] = tvec_xyz[1]
                transform[2, 3] = tvec_xyz[2]
                transform[3, 3] = 1.0
                # Construct a transform from camera coordinates to heliostat coordinates.
                inv_transform = np.linalg.inv(transform)
                # Compute the camera position in heliostat coordinates.
                camera_xyz1 = inv_transform.dot(np.array([0, 0, 0, 1.0]))
                camera_xyz = list(camera_xyz1)[0:3]
                # Store results.
                frame_parameters_dict["transform_heliostat_to_camera"] = transform
                frame_parameters_dict["transform_camera_to_heliostat"] = inv_transform
                frame_parameters_dict["camera_xyz_in_heliostat_coords"] = camera_xyz

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  SYNCHRONIZE VIDEO FRAMES TO GPS LOG TIME
    #
    #

    def initialize_synchronization_pair_list(self):
        # ?? SCAFFOLDING RCB -- THIS IS FUNDAMENTALLY AN UNSOLVED ROUTINE.  HOW DO WE KNOW WHICH PASS TO PICK?  WHAT IF THERE IS NOT A CAMERA PASS WHICH HAS A CLEAN CORNER? HOW PRODUCE A ROBUST METHOD?
        # ?? SCAFFOLDING RCB -- THIS SUGGESTS A COMPLETELY DIFFERENT APPROACH IS NEEDED -- MAYBE DAN'S?  MAYBE SOLVING KEY FRAME IDENTIFICATION WILL ELIMINATE THIS PROBLEM?
        # Index of relevant pass within min-to-max or max-to-min xy direction change lists.  # ?? SCAFFOLDING RCB -- THIS LITERAL IS BECAUSE THIS IS A TEMPORARY PATCH, UNTIL TIME SYNCHRONIZXATION IS DONE AS PART OF RTK GPS/FRAME SYNCRHONIZATION, AND/OR KEY FRAME SELECTION(?).
        synch_gps_pass_idx_list = [0, 5, 6]
        # ?? SCAFFOLDING RCB -- THIS LITERAL IS BECAUSE THIS IS A TEMPORARY PATCH, UNTIL TIME SYNCHRONIZXATION IS DONE AS PART OF RTK GPS/FRAME SYNCRHONIZATION, AND/OR KEY FRAME SELECTION(?).
        synch_camera_hel_name_list = ["5W9", "5W1", "5E3"]
        synch_pair_list = []
        for synch_gps_pass_idx, synch_camera_hel_name in zip(synch_gps_pass_idx_list, synch_camera_hel_name_list):
            # GPS spec.
            # ?? SCAFFOLDING RCB -- THIS LITERAL IS BECAUSE THIS IS A TEMPORARY PATCH, UNTIL TIME SYNCHRONIZXATION IS DONE AS PART OF RTK GPS/FRAME SYNCRHONIZATION, AND/OR KEY FRAME SELECTION(?).
            synch_gps_pass_type = "max_to_min"
            # ?? SCAFFOLDING RCB -- THIS LITERAL IS BECAUSE THIS IS A TEMPORARY PATCH, UNTIL TIME SYNCHRONIZXATION IS DONE AS PART OF RTK GPS/FRAME SYNCRHONIZATION, AND/OR KEY FRAME SELECTION(?).
            synch_gps_start_or_stop = "stop"
            synch_gps_time = self.find_gps_synch_time(synch_gps_pass_type, synch_gps_pass_idx, synch_gps_start_or_stop)
            # ?? SCAFFOLDING RCB -- THIS INFORMATION IS DEFINED IN DIFFERENT PLACES IN THE CODE, CREATING A VULNERABILITY TO POSSIBLE FURTURE BUGS DUE TO MISMATCHES.  IF THIS APPROACH SURVIVES, RESOLVE THIS.
            synch_gps_spec = [synch_gps_pass_type, synch_gps_pass_idx, synch_gps_start_or_stop, synch_gps_time]
            # Camera spec.
            # Which pause to select out of heliostat camera scan trajectory fragment.  # ?? SCAFFOLDING RCB -- THIS LITERAL IS BECAUSE THIS IS A TEMPORARY PATCH, UNTIL TIME SYNCHRONIZXATION IS DONE AS PART OF RTK GPS/FRAME SYNCRHONIZATION, AND/OR KEY FRAME SELECTION(?).
            synch_camera_pause_idx = 0
            synch_camera_frame_id = self.find_camera_synch_frame(synch_camera_hel_name, synch_camera_pause_idx)
            # ?? SCAFFOLDING RCB -- THIS INFORMATION IS DEFINED IN DIFFERENT PLACES IN THE CODE, CREATING A VULNERABILITY TO POSSIBLE FURTURE BUGS DUE TO MISMATCHES.  IF THIS APPROACH SURVIVES, RESOLVE THIS.
            synch_camera_spec = [synch_camera_hel_name, synch_camera_pause_idx, synch_camera_frame_id]
            # Pair.
            synch_pair = [
                synch_gps_spec,
                synch_camera_spec,
            ]  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS, PERHAPS WITH TWO SUBCLASSES.
            synch_pair_list.append(synch_pair)
        # Return.
        return synch_pair_list

    def find_gps_synch_time(self, synch_gps_pass_type, synch_gps_pass_idx, synch_gps_start_or_stop):
        if synch_gps_pass_type == "min_to_max":
            pass_pair_list = self.maximum_to_minimum_pass_pair_list
        elif synch_gps_pass_type == "max_to_min":
            pass_pair_list = self.minimum_to_maximum_pass_pair_list
        else:
            print(
                "ERROR: In TrajectoryAnalysis.find_gps_synch_time(), encountered unexpected synch_gps_pass_type =",
                str(synch_gps_pass_type),
            )
            assert False
        pass_pair = pass_pair_list[synch_gps_pass_idx]
        if synch_gps_start_or_stop == "start":
            pass_endpoint = pass_pair[0]
        elif synch_gps_start_or_stop == "stop":
            pass_endpoint = pass_pair[1]
        else:
            print(
                "ERROR: In TrajectoryAnalysis.find_gps_synch_time(), encountered unexpected synch_gps_start_or_stop =",
                str(synch_gps_start_or_stop),
            )
            assert False
        synch_gps_time = pass_endpoint["time"]
        return synch_gps_time

    def find_camera_synch_frame(self, synch_camera_hel_name, synch_camera_pause_idx):
        frames_dict = self.hel_frames_dict[synch_camera_hel_name]
        first_time = True
        previous_frame_id = None
        previous_camera_xyz = None
        previous_minimum_motion = np.inf
        minimum_motion_frame_id = None
        for frame_id in dt.sorted_keys(frames_dict):
            frame_parameters_dict = frames_dict[frame_id]
            camera_xyz = frame_parameters_dict["camera_xyz_in_heliostat_coords"]
            if not first_time:
                dx = camera_xyz[0] - previous_camera_xyz[0]
                dy = camera_xyz[1] - previous_camera_xyz[1]
                dz = camera_xyz[2] - previous_camera_xyz[2]
                motion = np.sqrt((dx * dx) + (dy * dy) + (dz * dz))
                if motion < previous_minimum_motion:
                    previous_minimum_motion = motion
                    minimum_motion_frame_id = previous_frame_id
                # We are looking for the first stop point.  Exit if we have passed it.
                # ?? SCAFFOLDING RCB -- THESE MAGIC NUMBERS ARE CHARACTERISTIC OF THIS TEMPORARY STUB.  AS NOTED ABOVE, THIS COMPUTATION NEEDS TO BE REPLACED, PROBABLY WITH A COMPUTATION EARLIER.
                if (previous_minimum_motion < 0.02) and (motion > 0.15):
                    break
            first_time = False
            previous_frame_id = frame_id
            previous_camera_xyz = camera_xyz
        synch_camera_frame_id = minimum_motion_frame_id
        # Return.
        return synch_camera_frame_id

    def synchronization_time_list(self):
        gps_time_list = []
        for synchronization_pair in self.synchronization_pair_list:
            synchronization_gps_spec = synchronization_pair[0]
            synchronization_camera_spec = synchronization_pair[1]
            synchronization_gps_time = synchronization_gps_spec[3]
            gps_time_list.append(synchronization_gps_time)
        return gps_time_list

    def synchronization_heliostat_name_list(self):
        hel_name_list = []
        for synchronization_pair in self.synchronization_pair_list:
            synchronization_gps_spec = synchronization_pair[0]
            synchronization_camera_spec = synchronization_pair[1]
            synchronization_hel_name = synchronization_camera_spec[0]
            hel_name_list.append(synchronization_hel_name)
        return hel_name_list

    def get_synchronization_frame_id(self, hel_name):
        for synch_pair in self.synchronization_pair_list:
            synch_gps_spec = synch_pair[0]
            synch_camera_spec = synch_pair[1]
            synch_hel_name = synch_camera_spec[0]
            if synch_hel_name == hel_name:
                return synch_camera_spec[2]
        print("In TrajectoryAnalysis.get_synchronization_frame_id(), did not find input hel_name =", str(hel_name))
        assert False

    def initialize_synchronization_constants(self):
        """
        Time relationship deduced from coincident UAS pause moments at scan pass corners, seen in both the GPS log and video data.  # ?? SCAFFOLDING RCB -- UPDATE COMMENT IF RE-IMPLEMENTED.
        """
        # ?? SCAFFOLDING RCB -- THESE CONSTANTS SHOULD BE INPUT FROM A PREVIOUS COMPUTATION.  SEE COMMENTS IN initialize_synchronization_pair_list() ABOVE.
        slope_01, intercept_01 = self.compute_synchronization_slope_and_intercept(
            self.synchronization_pair_list[0], self.synchronization_pair_list[1]
        )
        slope_02, intercept_02 = self.compute_synchronization_slope_and_intercept(
            self.synchronization_pair_list[0], self.synchronization_pair_list[2]
        )
        slope_12, intercept_12 = self.compute_synchronization_slope_and_intercept(
            self.synchronization_pair_list[1], self.synchronization_pair_list[2]
        )
        # Use longest baseline.
        return slope_02, intercept_02

    def compute_synchronization_slope_and_intercept(self, synch_pair_1, synch_pair_2):
        """
        Time relationship deduced from coincident UAS pause moments at scan pass corners, seen in both the GPS log and video data.
        """
        frame_id_1, time_1 = self.extract_frame_id_and_time(synch_pair_1)
        frame_id_2, time_2 = self.extract_frame_id_and_time(synch_pair_2)
        if frame_id_2 == frame_id_1:
            print(
                "ERROR: In TrajectoryAnalysis.compute_synchronization_slope_and_intercept(), identical frame_id_1 == frame_id_2 ==",
                frame_id_1,
            )
            assert False
        slope = (time_2 - time_1) / (frame_id_2 - frame_id_1)
        intercept = time_1 - (slope * frame_id_1)
        return slope, intercept

    def extract_frame_id_and_time(self, synch_pair):
        synch_gps_spec = synch_pair[0]
        synch_time = synch_gps_spec[3]
        synch_camera_spec = synch_pair[1]
        synch_frame_id = synch_camera_spec[2]
        return synch_frame_id, synch_time

    def time_given_frame_id(self, frame_id):
        """
        Time relationship deduced from coincident UAS pause moments at scan pass corners, seen in both the GPS log and video data.
        """
        return (self.synchronization_slope * frame_id) + self.synchronization_intercept

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  CONSTRUCT HELIOSTAT CAMERA PASSES
    #
    #

    def construct_heliostat_camera_passes(self):
        hel_camera_passes_dict = {}
        for hel_name in self.hel_frames_dict.keys():
            frames_dict = self.hel_frames_dict[hel_name]
            # Fetch all (x,y,z,t) points, expressed in heliostat coordinates, discarding all points with n_missing too large.
            camera_xyzt_list = []
            for frame_id in frames_dict.keys():
                frame_parameters_dict = frames_dict[frame_id]
                n_missing = frame_parameters_dict["n_missing"]
                if n_missing <= maximum_n_missing:
                    camera_xyz = frame_parameters_dict["camera_xyz_in_heliostat_coords"]
                    frame_time = self.time_given_frame_id(frame_id)
                    camera_xyzt = camera_xyz + [frame_time]
                    camera_xyzt_list.append(camera_xyzt)
            # Split the (x,y,z,t) list into contiguous segments, corresponding to separate passes.
            # ?? SCAFFOLDING RCB -- REWORK THIS, WITHOUT THE MAGIC NUMBER DISCONNECT_THRESHOLD?
            list_of_camera_xyzt_lists_1 = self.split_xyzt_list_where_distance_exceeds_maximum(
                camera_xyzt_list, self.maximum_camera_pass_inter_point_distance
            )
            # Discard lists that are too short.
            list_of_camera_xyzt_lists_2 = self.discard_xyzt_lists_where_length_below_minimum(
                list_of_camera_xyzt_lists_1, self.minimum_camera_pass_number_of_points
            )
            # Split the resulting (x,y,z,t) lists by speed into segments with continuous motion, thus eliminating connected sequences that include stopping and proceeding in a differnt direction.
            list_of_camera_xyzt_lists_3 = []
            for camera_xyzt_list in list_of_camera_xyzt_lists_2:
                # If there is a pause, break the list up.
                list_of_camera_xyzt_lists_A = self.split_xyzt_list_where_inter_point_speed_below_minimum(
                    camera_xyzt_list, self.minimum_camera_pass_inter_point_speed
                )
                # Discard lists that are too short.
                list_of_camera_xyzt_lists_B = self.discard_xyzt_lists_where_length_below_minimum(
                    list_of_camera_xyzt_lists_A, self.minimum_camera_pass_number_of_points
                )
                # Add the result to our accumulating list.
                list_of_camera_xyzt_lists_3 += list_of_camera_xyzt_lists_B
            # Now construct the camera passes.
            camera_pass_list = []
            for camera_xyzt_list in list_of_camera_xyzt_lists_3:
                if len(camera_xyzt_list) >= self.minimum_camera_pass_number_of_points:
                    camera_pass = self.scan_pass_given_xyzt_list(
                        camera_xyzt_list,
                        n_start_margin=self.camera_pass_start_margin,
                        n_stop_margin=self.camera_pass_stop_margin,
                    )
                    camera_pass_list.append(camera_pass)
            # Add to result.
            hel_camera_passes_dict[hel_name] = camera_pass_list
        return hel_camera_passes_dict

    def split_xyz_list_where_distance_exceeds_maximum(
        self, xyz_list, maximum_inter_point_distance
    ):  # Split if inter-point distance is above this threshold.
        if len(xyz_list) == 0:
            return []
        elif len(xyz_list) == 1:
            return [xyz_list]
        else:
            first_xyz = xyz_list[0]
            remainder = self.split_xyz_list_where_distance_exceeds_maximum(xyz_list[1:], maximum_inter_point_distance)
            first_remainder_xyz_list = remainder[0]
            first_remainder_xyz = first_remainder_xyz_list[0]
            d = np.sqrt(
                (first_xyz[0] - first_remainder_xyz[0]) ** 2
                + (first_xyz[1] - first_remainder_xyz[1]) ** 2
                + (first_xyz[2] - first_remainder_xyz[2]) ** 2
            )
            if d > maximum_inter_point_distance:
                return [[first_xyz]] + remainder
            else:
                remainder[0] = [first_xyz] + remainder[0]
                return remainder

    def split_xyzt_list_where_distance_exceeds_maximum(
        self, xyzt_list, maximum_inter_point_distance
    ):  # Split if inter-point distance is above this threshold.
        if len(xyzt_list) == 0:
            return []
        elif len(xyzt_list) == 1:
            return [xyzt_list]
        else:
            first_xyzt = xyzt_list[0]
            remainder = self.split_xyzt_list_where_distance_exceeds_maximum(xyzt_list[1:], maximum_inter_point_distance)
            first_remainder_xyzt_list = remainder[0]
            first_remainder_xyzt = first_remainder_xyzt_list[0]
            d = np.sqrt(
                (first_xyzt[0] - first_remainder_xyzt[0]) ** 2
                + (first_xyzt[1] - first_remainder_xyzt[1]) ** 2
                + (first_xyzt[2] - first_remainder_xyzt[2]) ** 2
            )
            if d > maximum_inter_point_distance:
                return [[first_xyzt]] + remainder
            else:
                remainder[0] = [first_xyzt] + remainder[0]
                return remainder

    def split_xyzt_list_where_inter_point_speed_below_minimum(self, xyzt_list, minimum_inter_point_speed):
        if len(xyzt_list) == 0:
            return []
        elif len(xyzt_list) == 1:
            return [xyzt_list]
        else:
            first_xyzt = xyzt_list[0]
            remainder = self.split_xyzt_list_where_inter_point_speed_below_minimum(
                xyzt_list[1:], minimum_inter_point_speed
            )  # Split if inter-point distance is above this threshold.
            first_remainder_xyzt_list = remainder[0]
            first_remainder_xyzt = first_remainder_xyzt_list[0]
            distance = np.sqrt(
                (first_xyzt[0] - first_remainder_xyzt[0]) ** 2
                + (first_xyzt[1] - first_remainder_xyzt[1]) ** 2
                + (first_xyzt[2] - first_remainder_xyzt[2]) ** 2
            )
            time_gap = first_remainder_xyzt[3] - first_xyzt[3]
            if time_gap == 0:
                print(
                    "ERROR: In TrajectoryAnalysis.split_xyzt_list_where_inter_point_speed_below_minimum(), unexpected zero time gap encountered."
                )
                assert False
            if time_gap < 0:
                print(
                    "ERROR: In TrajectoryAnalysis.split_xyzt_list_where_inter_point_speed_below_minimum(), encountered unexpected negative time gap =",
                    time_gap,
                )
                assert False
            speed = distance / time_gap
            if speed < minimum_inter_point_speed:
                return [[first_xyzt]] + remainder
            else:
                remainder[0] = [first_xyzt] + remainder[0]
                return remainder

    def discard_xyzt_lists_where_length_below_minimum(self, list_of_xyzt_lists, minimum_list_length):
        list_of_sufficiently_long_xyzt_lists = []
        for xyzt_list in list_of_xyzt_lists:
            if len(xyzt_list) >= minimum_list_length:
                list_of_sufficiently_long_xyzt_lists.append(xyzt_list)
        return list_of_sufficiently_long_xyzt_lists

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  GPS-CAMERA ANALYSIS
    #
    # Identify correspondence between camera pass and corresponding GPS pass, and then identify
    # the heliostat (azimuth,elevation) configuration which will bring them into alignment.
    #

    def construct_hel_gps_camera_analysis_dict(self):
        hel_gps_camera_analysis_dict = {}
        for hel_name in dt.sorted_keys(self.hel_camera_passes_dict):
            list_of_gps_camera_analysis_dicts = []
            camera_pass_list = self.hel_camera_passes_dict[hel_name]
            camera_pass_idx = 0
            for camera_pass in camera_pass_list:
                matching_gps_pass = self.find_matching_gps_pass(camera_pass)
                if matching_gps_pass == None:
                    print(
                        "WARNING: In TrajectoryAanlsysi.construct_hel_gps_camera_analysis_dict(), no matching GPS scan pass found for heliostat "
                        + str(hel_name)
                        + " camera_pass_idx =",
                        camera_pass_idx,
                    )
                    pass
                else:
                    # Data structure for capturing GPS-camera pass analysis results.
                    gps_camera_analysis_dict = self.construct_gps_camera_analysis_dict(
                        hel_name, matching_gps_pass, camera_pass
                    )
                    list_of_gps_camera_analysis_dicts.append(gps_camera_analysis_dict)
                camera_pass_idx += 1
            if len(list_of_gps_camera_analysis_dicts) == 0:
                print(
                    "WARNING: In TrajectoryAnalysis.construct_hel_gps_camera_analysis_dict(), no matching GPS scans for heliostat "
                    + str(hel_name)
                )
                pass
            else:
                hel_gps_camera_analysis_dict[hel_name] = list_of_gps_camera_analysis_dicts
        # Return.
        return hel_gps_camera_analysis_dict

    def find_matching_gps_pass(self, camera_pass):
        # Fetch camera pass time parameters.
        camera_pass_begin_time = camera_pass["stable_begin_xyzt"][3]
        camera_pass_end_time = camera_pass["stable_end_xyzt"][3]
        # Lookup matching GPS scan pass.
        matching_gps_pass = None
        matching_gps_match_overlap_len = None
        for gps_pass in self.gps_scan_passes:
            gps_pass_begin_time = gps_pass["stable_begin_xyzt"][
                3
            ]  # ?? SCAFFOLDING RCB -- HAVE THIS FETCH ORIGINAL FULL-LENGTH START AND STOP TIME INTERVAL.
            gps_pass_end_time = gps_pass["stable_end_xyzt"][
                3
            ]  # ?? SCAFFOLDING RCB -- HAVE THIS FETCH ORIGINAL FULL-LENGTH START AND STOP TIME INTERVAL.
            # Check for overlap.
            if (gps_pass_begin_time <= camera_pass_end_time) and (camera_pass_begin_time <= gps_pass_end_time):
                # There is temporal overlap.
                overlap_min = max(camera_pass_begin_time, gps_pass_begin_time)
                overlap_max = min(camera_pass_end_time, gps_pass_end_time)
                overlap_len = overlap_max - overlap_min
                if overlap_len > 0:
                    # There is non-trivial temporal overlap.
                    if matching_gps_pass == None:
                        # First match encountered.  Save it.
                        matching_gps_pass = gps_pass
                        matching_gps_match_overlap_len = overlap_len
                    else:
                        # More than one GPS pass matched.  Take the one with the more substantial overlap.  # ?? SCAFFOLDING RCB -- STUDY THIS FURTHER, TO DETERMINE WHETHER THIS INDICATES A BUG.
                        print(
                            "WARNING: In TrajectoryAnalysis.find_matching_gps_pass(), multiple matching GPS scan passes found, with overlaps",
                            matching_gps_match_overlap_len,
                            " and ",
                            overlap_len,
                            "; keeping the one with greater overlap.",
                        )
                        if overlap_len > matching_gps_match_overlap_len:
                            matching_gps_pass = gps_pass
                            matching_gps_match_overlap_len = overlap_len
        if matching_gps_pass == None:
            # We fell through loop without finding a match.
            # print('WARNING: In TrajectoryAnalysis.find_matching_gps_pass(), no matching GPS scan pass found.') # Better warning printed by caller.
            pass
        # Return.
        return matching_gps_pass

    def construct_gps_camera_analysis_dict(
        self, hel_name, gps_pass, camera_pass
    ):  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS.
        # Fetch camera pass time interval.
        time_begin = camera_pass["stable_begin_xyzt"][3]  # Times correspond to the seconds column in the flight log.
        time_end = camera_pass["stable_end_xyzt"][3]  #
        time_mid = (time_begin + time_end) / 2.0  #
        # Compute heliostat (azimuth, elevation) from aim point and time.
        (
            azimuth_from_model_begin,
            elevation_from_model_begin,
            azimuth_from_model_mid,
            elevation_from_model_mid,
            azimuth_from_model_end,
            elevation_from_model_end,
        ) = self.compute_model_azimuth_elevation(hel_name, time_begin, time_mid, time_end)
        # Compute heliostat (azimuth, elevation) that will bring camera pass into parallel alignment with GPS pass.
        (azimuth_from_alignment, elevation_from_alignment, alignment_angle_error) = (
            self.compute_alignment_azimuth_elevation(
                gps_pass, camera_pass, azimuth_from_model_mid, elevation_from_model_mid
            )
        )
        # # Compute heliostat (azimuth, elevation) from log.
        # azimuth_from_log_begin,          \
        # elevation_from_log_begin,        \
        # azimuth_from_log_mid,            \
        # elevation_from_log_mid,          \
        # azimuth_from_log_end,            \
        # elevation_from_log_end,          \
        # azimuth_target_from_log_begin,   \
        # elevation_target_from_log_begin, \
        # azimuth_target_from_log_mid,     \
        # elevation_target_from_log_mid,   \
        # azimuth_target_from_log_end,     \
        # elevation_target_from_log_end = self.compute_log_azimuth_elevation(hel_name, time_begin, time_mid, time_end)
        # Compute camera_pass points, transformed by (azimuth,elevation).
        heliostat = self.solar_field.lookup_heliostat(hel_name)
        transform_translation = heliostat.centroid_nparray
        transform_rotation = Heliostat.heliostat_rotation_matrix(azimuth_from_alignment, elevation_from_alignment)
        camera_xyzt_list = camera_pass["inlier_xyzt_list"]
        transformed_camera_xyzt_list = []
        for camera_xyzt in camera_xyzt_list:
            camera_xyz_heliostat = camera_xyzt[0:3]
            camera_time = camera_xyzt[3]
            # ?? SCAFFOLDING RCB -- CONVERSION TO/FROM LIST AGAIN SUGGESTS CONVERTING TO NUMPY ARRAYS THROUGHOUT.
            camera_xyz_world = list(transform_translation + transform_rotation.dot(np.array(camera_xyz_heliostat)))
            camera_xyzt_world = camera_xyz_world + [camera_time]
            transformed_camera_xyzt_list.append(camera_xyzt_world)
        per_heliostat_transformed_camera_pass = self.scan_pass_given_xyzt_list(
            transformed_camera_xyzt_list, n_start_margin=0, n_stop_margin=0
        )  # We alrady shrunk, so don't shrink again.
        # Compute nearest-neighbor line segments.
        gps_line_3d = gps_pass["line_3d"]
        camera_gps_point_pair_list = []
        camera_gps_distance_list = []
        camera_gps_distance_squared_sum = 0
        camera_gps_distance_squared_count = 0
        for transformed_camera_xyzt in transformed_camera_xyzt_list:
            transformed_camera_inlier_xyz = transformed_camera_xyzt[0:3]
            nearest_gps_xyz = g3d.closest_point_on_line_3d(transformed_camera_inlier_xyz, gps_line_3d)
            camera_gps_point_pair = [transformed_camera_inlier_xyz, nearest_gps_xyz]
            camera_gps_distance = g3d.distance_between_xyz_points(transformed_camera_inlier_xyz, nearest_gps_xyz)
            camera_gps_point_pair_list.append(camera_gps_point_pair)
            camera_gps_distance_list.append(camera_gps_distance)
            camera_gps_distance_squared_sum += camera_gps_distance**2
            camera_gps_distance_squared_count += 1
        # RMS statistics.
        if camera_gps_distance_squared_count == 0:
            print(
                "ERROR: In TrajectoryAnalysis.construct_gps_camera_analysis_dict(), encountered empty camera pass inlier xyzt list."
            )
            assert False
        rms_distance = np.sqrt(camera_gps_distance_squared_sum / camera_gps_distance_squared_count)
        # Store results.
        gps_camera_analysis_dict = {}
        gps_camera_analysis_dict["hel_name"] = hel_name
        gps_camera_analysis_dict["gps_pass"] = gps_pass
        gps_camera_analysis_dict["camera_pass"] = camera_pass
        # Key times.
        gps_camera_analysis_dict["time_begin"] = time_begin
        gps_camera_analysis_dict["time_end"] = time_end
        gps_camera_analysis_dict["time_mid"] = time_mid
        # (azimuth, elevation) from aim point and time.
        gps_camera_analysis_dict["azimuth_from_model_begin"] = azimuth_from_model_begin
        gps_camera_analysis_dict["elevation_from_model_begin"] = elevation_from_model_begin
        gps_camera_analysis_dict["azimuth_from_model_mid"] = azimuth_from_model_mid
        gps_camera_analysis_dict["elevation_from_model_mid"] = elevation_from_model_mid
        gps_camera_analysis_dict["azimuth_from_model_end"] = azimuth_from_model_end
        gps_camera_analysis_dict["elevation_from_model_end"] = elevation_from_model_end
        # (azimuth, elevation) from parallel alignment.
        gps_camera_analysis_dict["azimuth_from_alignment"] = azimuth_from_alignment
        gps_camera_analysis_dict["elevation_from_alignment"] = elevation_from_alignment
        gps_camera_analysis_dict["alignment_angle_error"] = alignment_angle_error
        # # (azimuth, elevation) from log.
        # gps_camera_analysis_dict['azimuth_from_log_begin']          = azimuth_from_log_begin
        # gps_camera_analysis_dict['elevation_from_log_begin']        = elevation_from_log_begin
        # gps_camera_analysis_dict['azimuth_from_log_mid']            = azimuth_from_log_mid
        # gps_camera_analysis_dict['elevation_from_log_mid']          = elevation_from_log_mid
        # gps_camera_analysis_dict['azimuth_from_log_end']            = azimuth_from_log_end
        # gps_camera_analysis_dict['elevation_from_log_end']          = elevation_from_log_end
        # gps_camera_analysis_dict['azimuth_target_from_log_begin']   = azimuth_target_from_log_begin
        # gps_camera_analysis_dict['elevation_target_from_log_begin'] = elevation_target_from_log_begin
        # gps_camera_analysis_dict['azimuth_target_from_log_mid']     = azimuth_target_from_log_mid
        # gps_camera_analysis_dict['elevation_target_from_log_mid']   = elevation_target_from_log_mid
        # gps_camera_analysis_dict['azimuth_target_from_log_end']     = azimuth_target_from_log_end
        # gps_camera_analysis_dict['elevation_target_from_log_end']   = elevation_target_from_log_end
        # Corresponding point analysis.
        gps_camera_analysis_dict["per_heliostat_transformed_camera_pass"] = per_heliostat_transformed_camera_pass
        gps_camera_analysis_dict["camera_gps_point_pair_list"] = camera_gps_point_pair_list
        gps_camera_analysis_dict["camera_gps_distance_list"] = camera_gps_distance_list
        gps_camera_analysis_dict["rms_distance"] = rms_distance
        # Return.
        return gps_camera_analysis_dict

    def compute_model_azimuth_elevation(self, hel_name, time_begin, time_mid, time_end):
        (azimuth_begin, elevation_begin) = self.compute_model_azimuth_elevation_given_time(hel_name, time_begin)
        azimuth_mid, elevation_mid = self.compute_model_azimuth_elevation_given_time(hel_name, time_mid)
        azimuth_end, elevation_end = self.compute_model_azimuth_elevation_given_time(hel_name, time_end)
        # Return.
        return (azimuth_begin, elevation_begin, azimuth_mid, elevation_mid, azimuth_end, elevation_end)

    def compute_model_azimuth_elevation_given_time(self, hel_name, time):
        heliostat = self.solar_field.lookup_heliostat(hel_name)
        when_ymdhmsz = self.gps_ymdhmsz_given_flight_log_seconds(time)
        h_config = heliostat.compute_tracking_configuration(
            self.aimpoint_xyz, self.solar_field.origin_lon_lat, when_ymdhmsz
        )
        # Return.
        return h_config.az, h_config.el

    def compute_alignment_azimuth_elevation(
        self, gps_pass, camera_pass, azimuth_from_model, elevation_from_model
    ):  # ?? SCAFFOLDING RCB -- THIS SHOULD BE A CLASS.
        # ?? SCAFFOLDING RCB -- THIS COULD BE SIMPLER AND FASTER IF IT USED NUMPY ARRAYS?  UPDATE THROUGHOUT?
        """
        Compute heliostat (azimuth, elevation) that will bring camera pass into parallel alignment with GPS pass.
        """
        gps_line_3d = gps_pass["line_3d"]
        gps_uxyz = gps_line_3d["uxyz"]  # Unit vector pointing along the line.
        camera_line_3d = camera_pass["line_3d"]
        camera_uxyz = camera_line_3d["uxyz"]  # Unit vector pointing along the line.
        # ?? SCAFFOLDING RCB -- WE SHOULD BE ABLE TO FIGURE OUT THIS TRANSFORM AND AVOID THIS NUMERICAL SEARCH.  BUT i DON'T HAVE TIME TO WORK IT OUT RIGHT NOW.
        # Use the nominal tracking angles as the initial guess.
        az = azimuth_from_model
        el = elevation_from_model
        # Search for (azimuth,elevation) that align gps_uxyz and camera_xyz.
        d_az = 0.010  # radians. Start large to ensure that minimum is captured.
        d_el = 0.010  #
        for idx in range(0, 20):
            d_az /= 2.0
            d_el /= 2.0
            az, delta_uxyz = self.search_az(d_az, az, el, gps_uxyz, camera_uxyz)
            el, delta_uxyz = self.search_el(d_el, az, el, gps_uxyz, camera_uxyz)
            if delta_uxyz < 0.000001:  # radians
                break
        # Return.
        return az, el, delta_uxyz

    def search_az(self, d_az, start_az, el, gps_uxyz, camera_uxyz):
        n_steps = 20  # Not a particularly sensitive number, if we ramp d_az.
        half_n_steps = n_steps / 2
        best_az = np.nan
        best_delta_uxyz = np.inf
        for idx in range(0, n_steps + 1):
            az = start_az + ((idx - half_n_steps) * d_az)
            delta_uxyz = self.check_az_el(az, el, gps_uxyz, camera_uxyz)
            if delta_uxyz < best_delta_uxyz:
                best_az = az
                best_delta_uxyz = delta_uxyz
        return best_az, best_delta_uxyz

    def search_el(self, d_el, az, start_el, gps_uxyz, camera_uxyz):
        n_steps = 20  # Not a particularly sensitive number, if we ramp d_el.
        half_n_steps = n_steps / 2
        best_el = np.nan
        best_delta_uxyz = np.inf
        for idx in range(0, n_steps + 1):
            el = start_el + ((idx - half_n_steps) * d_el)
            delta_uxyz = self.check_az_el(az, el, gps_uxyz, camera_uxyz)
            if delta_uxyz < best_delta_uxyz:
                best_el = el
                best_delta_uxyz = delta_uxyz
        return best_el, best_delta_uxyz

    def check_az_el(self, az, el, gps_uxyz, camera_uxyz):
        # Code grabbed from Heliostat.set_configuration().
        rotation_about_x = (np.pi / 2) - el
        rotation_about_z = np.pi - az
        Rx_rotation = t3d.axisrotation(np.array([1, 0, 0]), rotation_about_x)
        Rz_rotation = t3d.axisrotation(np.array([0, 0, 1]), rotation_about_z)
        hel_rotation = Rz_rotation.dot(Rx_rotation)
        rotated_camera_uxyz = hel_rotation.dot(camera_uxyz)
        # Rx_rotation_is_rotation = t3d.is_rotation_matrix(Rx_rotation)
        # Rz_rotation_is_rotation = t3d.is_rotation_matrix(Rz_rotation)
        # hel_rotation_is_rotation = t3d.is_rotation_matrix(hel_rotation)
        # norm_rotated_camera_uxyz = g3d.vector_3d_norm(rotated_camera_uxyz)
        # Compute difference between gps_uxyz and camera_uxyz.
        delta_uxyz = np.sqrt(
            (gps_uxyz[0] - rotated_camera_uxyz[0]) ** 2
            + (gps_uxyz[1] - rotated_camera_uxyz[1]) ** 2
            + (gps_uxyz[2] - rotated_camera_uxyz[2]) ** 2
        )
        return delta_uxyz

    def compute_log_azimuth_elevation(self, hel_name, time_begin, time_mid, time_end):
        (azimuth_begin, elevation_begin, azimuth_target_begin, elevation_target_begin) = (
            self.compute_log_azimuth_elevation_given_time(hel_name, time_begin)
        )
        (azimuth_mid, elevation_mid, azimuth_target_mid, elevation_target_mid) = (
            self.compute_log_azimuth_elevation_given_time(hel_name, time_mid)
        )
        (azimuth_end, elevation_end, azimuth_target_end, elevation_target_end) = (
            self.compute_log_azimuth_elevation_given_time(hel_name, time_end)
        )
        # Return.
        return (
            azimuth_begin,
            elevation_begin,
            azimuth_mid,
            elevation_mid,
            azimuth_end,
            elevation_end,
            azimuth_target_begin,
            elevation_target_begin,
            azimuth_target_mid,
            elevation_target_mid,
            azimuth_target_end,
            elevation_target_end,
        )

    def compute_log_azimuth_elevation_given_time(self, hel_name, time):
        # Return.
        #        return azimuth, elevation, azimuth_target, elevation_target
        return -111, -222, -333, -444

    def set_per_heliostat_estimates_of_camera_xyz_given_overall_time(self):
        # Trajectory fragments.
        for hel_name in self.hel_frames_dict.keys():
            heliostat = self.solar_field.lookup_heliostat(hel_name)
            frames_dict = self.hel_frames_dict[hel_name]
            for frame_id in frames_dict.keys():
                frame_parameters_dict = frames_dict[frame_id]
                camera_xyz_heliostat = frame_parameters_dict["camera_xyz_in_heliostat_coords"]
                # ?? SCAFFOLDING RCB -- HERE WE COULD USE THE SPECIFIC TIME OF EACH FRAME, AND UPDATE HELIOSTAT TRACKING ANGLE TO CORRESPOND TO THAT TIME, BEFORE CALLING heliostat.transform_xyz().  HOWEVER, THIS IS NOT REQUIRED FOR SHORT FLIGHTS, SINCE THE DIFFERENCE WILL BE SMALL, AND IT IS REFINED LATER
                camera_xyz_world = list(
                    heliostat.transform_xyz(camera_xyz_heliostat)
                )  # transform_xyz() returns a numpy array.
                frame_parameters_dict["per_heliostat_estimate_of_camera_xyz_in_world_coords_overall_time"] = (
                    camera_xyz_world
                )
        # Camera passes.
        self.hel_transformed_camera_passes_dict = {}
        for hel_name in self.hel_camera_passes_dict.keys():
            heliostat = self.solar_field.lookup_heliostat(hel_name)
            camera_pass_list = self.hel_camera_passes_dict[hel_name]
            transformed_camera_pass_list = []
            for camera_pass in camera_pass_list:
                camera_xyzt_list = camera_pass["inlier_xyzt_list"]
                transformed_camera_xyzt_list = []
                for camera_xyzt in camera_xyzt_list:
                    camera_xyz_heliostat = camera_xyzt[0:3]
                    camera_time = camera_xyzt[3]
                    camera_xyz_world = list(
                        heliostat.transform_xyz(camera_xyz_heliostat)
                    )  # transform_xyz() returns a numpy array.
                    camera_xyzt_world = camera_xyz_world + [camera_time]
                    transformed_camera_xyzt_list.append(camera_xyzt_world)
                # ?? SCAFFOLDING RCB -- THIS START_MARGIN, STOP_MARGIN IS A STUB HACK.
                transformed_camera_pass = self.scan_pass_given_xyzt_list(
                    transformed_camera_xyzt_list, n_start_margin=0, n_stop_margin=0
                )
                transformed_camera_pass_list.append(transformed_camera_pass)
            # Add to result.
            self.hel_transformed_camera_passes_dict[hel_name] = transformed_camera_pass_list

    def set_per_heliosat_configurations_from_gps_camera_alignment(self):
        """
        Sets the configuration of each heliostat for which we found an (az,el) configuraiton using the alignment method.
        """
        for hel_name in self.hel_gps_camera_analysis_dict.keys():
            list_of_gps_camera_analysis_dicts = self.hel_gps_camera_analysis_dict[hel_name]
            gps_camera_analysis_dict = list_of_gps_camera_analysis_dicts[0]  # Arbitrarily choose first orientation.
            azimuth = gps_camera_analysis_dict["azimuth_from_alignment"]
            elevation = gps_camera_analysis_dict["elevation_from_alignment"]
            heliostat = self.solar_field.lookup_heliostat(hel_name)
            h_config = hc.HeliostatConfiguration(az=azimuth, el=elevation)
            heliostat.set_configuration(h_config)

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  RENDER
    #

    # PRINT KEY RESULTS

    def print_flight_log_df(self):
        print("In TrajectoryAnalysis.print_flight_log_df(), flight_log_df, after adding columns:")
        print(self.flight_log_df)

    def print_gps_flight_log_zero_seconds(self):
        print(
            "In TrajectoryAnalysis.print_gps_flight_log_zero_seconds(), zero_time_ymdhmsz =",
            self.gps_ymdhmsz_given_flight_log_zero_seconds,
        )

    def print_velocity_xy_change_points(self):
        print("In TrajectoryAnalysis.print_velocity_xy_change_points(), velocity_xy_change minima:")
        lt.print_list(self.gps_velocity_xy_change_minima, indent=4, max_items=50, max_item_length=200)
        print("In TrajectoryAnalysis.print_velocity_xy_change_points(), velocity_xy_change maxima:")
        lt.print_list(self.gps_velocity_xy_change_maxima, indent=4, max_items=50, max_item_length=200)

    def print_gps_scan_pass_summary(self):
        # Maximum to minimum.
        print("\nIn TrajectoryAnalysis.print_gps_scan_pass_summary(), self.maximum_to_minimum_pass_pair_list:")
        lt.print_list(self.maximum_to_minimum_pass_pair_list, indent=4, max_items=50, max_item_length=2000)
        # Minimum to maximum.
        print("\nIn TrajectoryAnalysis.print_gps_scan_pass_summary(), self.minimum_to_maximum_pass_pair_list:")
        lt.print_list(self.minimum_to_maximum_pass_pair_list, indent=4, max_items=50, max_item_length=2000)
        # GPS scan passes.
        print("\nIn TrajectoryAnalysis.print_gps_scan_pass_summary(), gps_scan_passes:")
        gps_scan_pass_idx = 0
        sum_square_rms = 0.0
        square_rms_count = 0
        for gps_scan_pass in self.gps_scan_passes:
            print(
                "In TrajectoryAnalysis.print_gps_scan_pass_summary(), gps_scan_passes "
                + str(gps_scan_pass_idx)
                + "  length =",
                gps_scan_pass["line_3d"]["length"],
                "  rms_distance_to_line =",
                gps_scan_pass["rms_distance_to_line"],
            )
            if gps_scan_pass_idx != 24:
                rms = gps_scan_pass["rms_distance_to_line"]
                sum_square_rms += rms * rms
                square_rms_count += 1
            gps_scan_pass_idx += 1
        overall_rms = np.sqrt(sum_square_rms / square_rms_count)
        print(
            "In TrajectoryAnalysis.print_gps_scan_pass_summary(), gps_scan_pass overall rms_distance_to_line =",
            overall_rms,
        )

    def print_hel_frames_dict(self):
        print("In TrajectoryAnalysis.print_hel_frames_dict(), self.hel_frames_dict:")
        self.print_hel_frames_dict_aux(self.hel_frames_dict)

    def print_hel_frames_dict_aux(self, hel_frames_dict):
        dt.print_dict_of_dict_of_dicts(
            hel_frames_dict,  # Dictionary to print.
            max_keys_1=2,  # Maximum number of level 1 keys to print.  Elipsis after that.
            max_keys_2=2,  # Maximum number of level 2 keys to print.  Elipsis after that.
            max_keys_3=10,  # Maximum number of level 3 keys to print.  Elipsis after that.
            max_value_3_length=200,  # Maximum level 2 value length to print.  Elipsis after that.
            indent_1=0,  # Number of blanks to print at the beginning of each top-level line.
            indent_2=4,  # Number of additional blanks to print for each second-level line.
            indent_3=4,
        )  # Number of additional blanks to print for each third-level line.

    def print_synchronization_pair_list(self):
        print(
            "\nIn TrajectoryAnalysis.print_synchronization_pair_list(), self.synchronization_pair_list:",
            self.synchronization_pair_list,
        )
        print(
            "In TrajectoryAnalysis.print_synchronization_pair_list(), self.synchronization_slope =",
            self.synchronization_slope,
        )
        print(
            "In TrajectoryAnalysis.print_synchronization_pair_list(), self.synchronization_intercept =",
            self.synchronization_intercept,
        )

    def print_hel_camera_passes_dict(self, max_heliostats=4):
        print("\nIn TrajectoryAnalysis.print_hel_camera_passes_dict(), heliostat camera passes:")
        self.print_hel_camera_passes_dict_aux(self.hel_camera_passes_dict, max_heliostats)

    def print_hel_camera_passes_dict_aux(self, hel_camera_passes_dict, max_heliostats):
        # Heliostat entries.
        hel_count = 1
        for hel_name in dt.sorted_keys(hel_camera_passes_dict):
            print(str(hel_name) + ":")
            camera_pass_list = hel_camera_passes_dict[hel_name]
            for camera_pass in camera_pass_list:
                print(
                    "    "
                    + str(camera_pass["stable_begin_xyzt"])
                    + "-->"
                    + str(camera_pass["stable_end_xyzt"])
                    + "; rms_to_line = "
                    + str(camera_pass["rms_distance_to_line"])
                )
            hel_count += 1
            if hel_count > max_heliostats:
                break
        # Postamble.
        if dt.number_of_keys(hel_camera_passes_dict) > max_heliostats:
            print("...")

    def print_hel_gps_camera_analysis_dict(self, max_heliostats=4):
        print("\nIn TrajectoryAnalysis.print_hel_gps_camera_analysis_dict(), heliostat GPS-camera analysis data:")
        self.print_hel_gps_camera_analysis_dict_aux(self.hel_gps_camera_analysis_dict, max_heliostats)

    def print_hel_gps_camera_analysis_dict_aux(self, hel_gps_camera_analysis_dict, max_heliostats):
        print("\nIn TrajectoryAnalysis.print_hel_gps_camera_analysis_dict(), heliostat GPS-camera analysis data:")
        # Heliostat entries.
        hel_count = 1
        for hel_name in dt.sorted_keys(hel_gps_camera_analysis_dict):
            print(str(hel_name) + ":")
            list_of_gps_camera_analysis_dicts = hel_gps_camera_analysis_dict[hel_name]
            for gps_camera_analysis_dict in list_of_gps_camera_analysis_dicts:
                transformed_camera_begin_xyzt = gps_camera_analysis_dict["per_heliostat_transformed_camera_pass"][
                    "stable_begin_xyzt"
                ]
                transformed_camera_end_xyzt = gps_camera_analysis_dict["per_heliostat_transformed_camera_pass"][
                    "stable_end_xyzt"
                ]
                gps_begin_xyzt = gps_camera_analysis_dict["gps_pass"]["stable_begin_xyzt"]
                gps_end_xyzt = gps_camera_analysis_dict["gps_pass"]["stable_end_xyzt"]
                azimuth = gps_camera_analysis_dict["azimuth_from_alignment"]
                elevation = gps_camera_analysis_dict["elevation_from_alignment"]
                alignment_error = gps_camera_analysis_dict["alignment_angle_error"]
                rms_distance = gps_camera_analysis_dict["rms_distance"]
                print(
                    "    Transformed camera pass:", transformed_camera_begin_xyzt, " --> ", transformed_camera_end_xyzt
                )
                print("      GPS pass:       ", gps_begin_xyzt, " --> ", gps_end_xyzt)
                print("      Azimuth:        ", azimuth, " rad")
                print("      Elevation:      ", elevation, " rad")
                print("      Alignment error:", alignment_error, " rad")
                print("      RMS distance:   ", rms_distance, " m")
            hel_count += 1
            if hel_count > max_heliostats:
                break
        # Postamble.
        if dt.number_of_keys(hel_gps_camera_analysis_dict) > max_heliostats:
            print("...")

    # DRAW SOLAR FIELD, WITH ANNOTATIONS

    def draw_and_save_solar_field_suite(self):
        # Style setup.
        self.solar_field_default_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.normal_outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.normal_outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style = self.solar_field_default_style
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)

        # Drawing styles.
        self.flight_log_style = rcps.data_curve(color="grey", linewidth=0.25, markersize=0.5)
        self.minimum_style = rcps.marker(color="m", markersize=0.8)
        self.maximum_style = rcps.marker(color="r", markersize=0.8)
        self.max_to_min_color = "g"
        self.min_to_max_color = "r"
        self.max_to_min_scan_pass_style = rcps.outline(color=self.max_to_min_color, linewidth=0.5)
        self.min_to_max_scan_pass_style = rcps.outline(color=self.min_to_max_color, linewidth=0.5)
        self.scan_pass_color = "k"
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.1)
        self.trajectory_fragment_style = rcps.data_curve(color="r", linewidth=0.25, markersize=0.4)
        self.camera_pass_color_wheel = ["red", "green", "blue", "magenta", "goldenrod"]
        self.trajectory_fragment_color_wheel = [
            "coral",
            "lawngreen",
            "cyan",
            "violet",
            "gold",
        ]  # Lighter version of each camera pass color.
        self.synchronization_point_style = rcps.marker(color="r", markersize=1)
        self.camera_pass_style = rcps.data_curve(color="r", linewidth=0.1, markersize=0.6)
        self.pass_connection_style = rcps.outline(color="r", linewidth=0.1)

        # What to include in each figure.
        draw_control_dict = {}  # ?? SCAFFOLDING RCB -- PROBABLY REPLACE WITH A REAL RENDER CONTROL CLASS OBJECT.
        draw_control_dict["draw_GPS_log"] = False
        draw_control_dict["draw_gps_velocity_xy_change_minima"] = False
        draw_control_dict["draw_gps_velocity_xy_change_maxima"] = False
        draw_control_dict["draw_gps_max_to_min_scan_passes"] = False
        draw_control_dict["draw_gps_min_to_max_scan_passes"] = False
        draw_control_dict["draw_gps_scan_passes"] = False
        draw_control_dict["draw_trajectory_fragments"] = False
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["trajectory_fragment_disconnect_threshold"] = 4.0  # m
        draw_control_dict["draw_camera_passes"] = False
        draw_control_dict["draw_gps_camera_analysis"] = False
        draw_control_dict["draw_gps_transformed_camera_inlier_xyzt_points"] = False
        draw_control_dict["draw_gps_camera_pass"] = False
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        # Standard views to render.
        # # 3-d only.
        # limits_3d_list = [ None ]
        # limits_xy_list = None
        # limits_xz_list = None
        # limits_yz_list = None
        # # xy only.
        # limits_3d_list = None
        # limits_xy_list = [ None ]
        # limits_xz_list = None
        # limits_yz_list = None
        # # xz only.
        # limits_3d_list = None
        # limits_xy_list = None
        # limits_xz_list = [ None ]
        # limits_yz_list = None
        # # yz only.
        # limits_3d_list = None
        # limits_xy_list = None
        # limits_xz_list = None
        # limits_yz_list = [ None ]
        # # xy and yz only.
        # limits_3d_list = None
        # limits_xy_list = [ None ]
        # limits_xz_list = None
        # limits_yz_list = [ None ]
        # All, full view.
        limits_3d_list = [None]
        limits_xy_list = [None]
        limits_xz_list = [None]
        limits_yz_list = [None]
        # # Selected sub views.
        # limits_3d_list = [ None, [[-50,0],[50,100],[0,50]], [[0,50],[50,100],[0,50]] ]
        # limits_xy_list = [ None, [[-50,0],[50,100]], [[-50,0],[0,50]] ]
        # limits_xz_list = [ None ]
        # limits_yz_list = [ None, [[100,150],[0,50]], [[160,205],[0,50]] ]

        # Draw the solar field all in stow.
        self.solar_field.set_full_field_stow()
        self.solar_field_style = rcsf.heliostat_normals_outlines(color="r")
        self.draw_and_save_solar_field_trajectories(
            "AA. " + self.solar_field.short_name + " Solar Field, In Stow",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field all face up.
        self.solar_field.set_full_field_face_up()
        self.solar_field_style = rcsf.heliostat_normals_outlines(color="b")
        self.draw_and_save_solar_field_trajectories(
            "AB. " + self.solar_field.short_name + " Solar Field, Face Up",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field nominal tracking, an hour before the flight.
        self.solar_field.set_full_field_tracking(
            aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz_minus_1_hour
        )
        self.solar_field_style = rcsf.heliostat_normals_outlines(color="k")
        aimpoint_str = "({x},{y},{z})".format(x=self.aimpoint_xyz[0], y=self.aimpoint_xyz[1], z=self.aimpoint_xyz[2])
        date_str = "{m}/{d}/{y}".format(
            m=self.when_ymdhmsz_minus_1_hour[1],
            d=self.when_ymdhmsz_minus_1_hour[2],
            y=self.when_ymdhmsz_minus_1_hour[0],
        )
        time_str = "{h:d}:{m:02d}:{s:02d}".format(
            h=self.when_ymdhmsz_minus_1_hour[3],
            m=self.when_ymdhmsz_minus_1_hour[4],
            s=self.when_ymdhmsz_minus_1_hour[5],
        )
        self.draw_and_save_solar_field_trajectories(
            "AC. "
            + self.solar_field.short_name
            + " Solar Field, Tracking to "
            + aimpoint_str
            + " on "
            + date_str
            + " at "
            + time_str,
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field nominal tracking, a half-hour before the flight.
        self.solar_field.set_full_field_tracking(
            aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz_minus_30_minutes
        )
        self.solar_field_style = rcsf.heliostat_normals_outlines(color="k")
        aimpoint_str = "({x},{y},{z})".format(x=self.aimpoint_xyz[0], y=self.aimpoint_xyz[1], z=self.aimpoint_xyz[2])
        date_str = "{m}/{d}/{y}".format(
            m=self.when_ymdhmsz_minus_30_minutes[1],
            d=self.when_ymdhmsz_minus_30_minutes[2],
            y=self.when_ymdhmsz_minus_30_minutes[0],
        )
        time_str = "{h:d}:{m:02d}:{s:02d}".format(
            h=self.when_ymdhmsz_minus_30_minutes[3],
            m=self.when_ymdhmsz_minus_30_minutes[4],
            s=self.when_ymdhmsz_minus_30_minutes[5],
        )
        self.draw_and_save_solar_field_trajectories(
            "AD. "
            + self.solar_field.short_name
            + " Solar Field, Tracking to "
            + aimpoint_str
            + " on "
            + date_str
            + " at "
            + time_str,
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field nominal tracking.
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field_style = rcsf.heliostat_normals_outlines(color="k")
        aimpoint_str = "({x},{y},{z})".format(x=self.aimpoint_xyz[0], y=self.aimpoint_xyz[1], z=self.aimpoint_xyz[2])
        date_str = "{m}/{d}/{y}".format(m=self.when_ymdhmsz[1], d=self.when_ymdhmsz[2], y=self.when_ymdhmsz[0])
        time_str = "{h:d}:{m:02d}:{s:02d}".format(
            h=self.when_ymdhmsz[3], m=self.when_ymdhmsz[4], s=self.when_ymdhmsz[5]
        )
        self.draw_and_save_solar_field_trajectories(
            "AE. "
            + self.solar_field.short_name
            + " Solar Field, Tracking to "
            + aimpoint_str
            + " on "
            + date_str
            + " at "
            + time_str,
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field nominal tracking, with exceptions of the day (showing normals).
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        self.solar_field_style = rcsf.heliostat_normals_outlines(color="grey")
        self.heliostat_up_style = rch.normal_outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.normal_outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        aimpoint_str = "({x},{y},{z})".format(x=self.aimpoint_xyz[0], y=self.aimpoint_xyz[1], z=self.aimpoint_xyz[2])
        date_str = "{m}/{d}/{y}".format(m=self.when_ymdhmsz[1], d=self.when_ymdhmsz[2], y=self.when_ymdhmsz[0])
        time_str = "{h:d}:{m:02d}:{s:02d}".format(
            h=self.when_ymdhmsz[3], m=self.when_ymdhmsz[4], s=self.when_ymdhmsz[5]
        )
        self.draw_and_save_solar_field_trajectories(
            "AF. Actual " + self.solar_field.short_name + " Field Configuration on " + date_str + " at " + time_str,
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field nominal tracking, with exceptions of the day.
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        aimpoint_str = "({x},{y},{z})".format(x=self.aimpoint_xyz[0], y=self.aimpoint_xyz[1], z=self.aimpoint_xyz[2])
        date_str = "{m}/{d}/{y}".format(m=self.when_ymdhmsz[1], d=self.when_ymdhmsz[2], y=self.when_ymdhmsz[0])
        time_str = "{h:d}:{m:02d}:{s:02d}".format(
            h=self.when_ymdhmsz[3], m=self.when_ymdhmsz[4], s=self.when_ymdhmsz[5]
        )
        self.draw_and_save_solar_field_trajectories(
            "AG. Actual " + self.solar_field.short_name + " Field Configuration on " + date_str + " at " + time_str,
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field and GPS trajectory.
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        draw_control_dict["draw_GPS_log"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("b")
        self.draw_and_save_solar_field_trajectories(
            "AH. GPS Trajectory Over Solar Field",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with GPS trajectory, with min/max scan pass construction steps identfiied.
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_velocity_xy_change_minima"] = True
        draw_control_dict["draw_gps_velocity_xy_change_maxima"] = True
        draw_control_dict["draw_gps_max_to_min_scan_passes"] = True
        draw_control_dict["draw_gps_min_to_max_scan_passes"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AI. GPS Scan Pass Min/Max Construction",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with GPS trajectory, with final scan passes identfiied.
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_velocity_xy_change_minima"] = False
        draw_control_dict["draw_gps_velocity_xy_change_maxima"] = False
        draw_control_dict["draw_gps_max_to_min_scan_passes"] = False
        draw_control_dict["draw_gps_min_to_max_scan_passes"] = False
        draw_control_dict["draw_gps_scan_passes"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AJ. Inferred GPS Scan Passes",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with GPS trajectory, with final scan passes highlighted (face up).
        self.solar_field.set_full_field_face_up()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_velocity_xy_change_minima"] = False
        draw_control_dict["draw_gps_velocity_xy_change_maxima"] = False
        draw_control_dict["draw_gps_max_to_min_scan_passes"] = False
        draw_control_dict["draw_gps_min_to_max_scan_passes"] = False
        draw_control_dict["draw_gps_scan_passes"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AK. Inferred GPS Scan Passes (All Heliostats Face Up)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with GPS trajectory, with final scan passes shown subtle (face up).
        self.solar_field.set_full_field_face_up()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_velocity_xy_change_minima"] = False
        draw_control_dict["draw_gps_velocity_xy_change_maxima"] = False
        draw_control_dict["draw_gps_max_to_min_scan_passes"] = False
        draw_control_dict["draw_gps_min_to_max_scan_passes"] = False
        draw_control_dict["draw_gps_scan_passes"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AL. Inferred GPS Scan Pass Reference Rendering (All Heliostats Face Up)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, used for time synchronization (tracking nominal time).
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = False
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = self.synchronization_heliostat_name_list()
        draw_control_dict["connect_trajectory_fragments"] = True
        draw_control_dict["draw_synchronization_points"] = True
        draw_control_dict["include_points_with_missing_corners"] = True
        draw_control_dict["include_non_refined_points"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        # Sub views for point synchronization plots.
        synch_limits_3d_list = None
        synch_limits_xy_list = [None, [[-100, -50], [0, 50]], [[-25, 25], [0, 50]]]
        synch_limits_xz_list = None
        synch_limits_yz_list = None
        self.draw_and_save_solar_field_trajectories(
            "AM. Time Synchronization Points",
            draw_control_dict,
            limits_3d_list=synch_limits_3d_list,
            limits_xy_list=synch_limits_xy_list,
            limits_xz_list=synch_limits_xz_list,
            limits_yz_list=synch_limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, including points viewing partial heliostats and connection lines (face up).
        self.solar_field.set_full_field_face_up()
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_GPS_log"] = False
        draw_control_dict["draw_gps_scan_passes"] = False
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = True
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = True
        draw_control_dict["include_non_refined_points"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AN. Trajectory Fragments for Each Heliostat Connected (Face Up)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, including points viewing partial heliostats (face up).
        self.solar_field.set_full_field_face_up()
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = True
        draw_control_dict["include_non_refined_points"] = True
        draw_control_dict["draw_camera_passes"] = False
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = False
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AO. Per Heliostat Scan Fragments Including Partial (Face Up)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, including points viewing partial heliostats (azimuth_face_south, tracking elevation).
        self.solar_field.set_full_field_tracking(
            aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz_minus_1_hour
        )
        for heliostat in self.solar_field.heliostats:
            az = heliostat.az
            el = heliostat.el
            new_h_config = hc.HeliostatConfiguration(az, el)
            new_h_config.az = np.radians(180.0)
            heliostat.set_configuration(new_h_config, clear_tracking=True)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = True
        draw_control_dict["include_non_refined_points"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AP. Per Heliostat Scan Fragments Including Partial (Az South, Elev Tracking Nominal Minus 1 hour)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, including points viewing partial heliostats (tracking minus 1 hour).
        self.solar_field.set_full_field_tracking(
            aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz_minus_1_hour
        )
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = True
        draw_control_dict["include_non_refined_points"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AQ. Per Heliostat Scan Fragments Including Partial (Tracking Nominal Minus 1 hour)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, including points viewing partial heliostats (tracking minus 30 minutes).
        self.solar_field.set_full_field_tracking(
            aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz_minus_30_minutes
        )
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = True
        draw_control_dict["include_non_refined_points"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AR. Per Heliostat Scan Fragments Including Partial (Tracking Nominal Minus 30 Minutes)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, including points viewing partial heliostats (tracking nominal time).
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = True
        draw_control_dict["include_non_refined_points"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AS. Per Heliostat Scan Fragments Including Partial (Tracking Nominal Time)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, without points viewing partial heliostats (tracking nominal time).
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AT. Per Heliostat Scan Fragments After Discarding Partial (Tracking Nominal Time)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after refining (tracking nominal time).
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.solar_field.set_heliostats_configuration(self.up_heliostats, self.up_configuration)
        self.solar_field.set_heliostats_configuration(self.down_heliostats, self.down_configuration)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AU. Per Heliostat Scan Fragments After Discarding Partial (Tracking Nominal Time)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after refining and including camera scan passes (tracking nominal time).
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AV. Per Heliostat Scan Pass Fit Lines (Tracking Nominal Time)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after refining and including camera scan passes and association with corresponding gps scan lines (tracking nominal time).
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = True
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = True
        draw_control_dict["include_non_refined_points"] = True
        draw_control_dict["draw_camera_passes"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.draw_and_save_solar_field_trajectories(
            "AW. Camera Scan Associations with GPS Scan Lines, Full Trajectory Fragments (Tracking Nominal Time)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after refining and including camera scan passes and association with corresponding gps scan lines (tracking nominal time).
        self.solar_field.set_full_field_tracking(aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz)
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = True  # False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = True  # False #True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.4)
        self.draw_and_save_solar_field_trajectories(
            "AX. Camera Scan Associations with GPS Scan Lines (Tracking Nominal Time)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after refining and including camera scan passes and association with corresponding gps scan lines (tracking minus 1 hour).
        self.solar_field.set_full_field_tracking(
            aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz_minus_1_hour
        )
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.1)
        self.draw_and_save_solar_field_trajectories(
            "AY. Camera Scan Associations with GPS Scan Lines (Tracking Nominal Minus 1 hour)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after refining and including camera scan passes and association with corresponding gps scan lines (tracking minus 1 hour).
        self.solar_field.set_full_field_tracking(
            aimpoint_xyz=self.aimpoint_xyz, when_ymdhmsz=self.when_ymdhmsz_minus_1_hour
        )
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = False
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.1)
        self.draw_and_save_solar_field_trajectories(
            "AZ. Camera Scan Lines (Tracking Nominal Minus 1 hour)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after refining and including camera scan passes and association with corresponding gps scan lines (face up).
        self.solar_field.set_full_field_face_up()
        self.set_per_heliostat_estimates_of_camera_xyz_given_overall_time()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = True
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = False
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.1)
        self.draw_and_save_solar_field_trajectories(
            "BA. Camera Scan Lines (Face Up)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after refining and including camera scan passes and association with corresponding gps scan lines (face up).
        self.solar_field.set_full_field_face_up()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = False
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = False
        draw_control_dict["draw_gps_camera_analysis"] = True
        draw_control_dict["draw_gps_transformed_camera_inlier_xyzt_points"] = True
        draw_control_dict["draw_gps_camera_pass"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.1)
        self.draw_and_save_solar_field_trajectories(
            "BB. GPS-Camera Alignment Results, Showing Connections (Face Up)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after aligning camera scan passes and association with corresponding gps scan lines (track individual heliostat analysis results).
        self.solar_field.set_full_field_face_up()
        self.set_per_heliosat_configurations_from_gps_camera_alignment()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = False
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = False
        draw_control_dict["draw_gps_camera_analysis"] = True
        draw_control_dict["draw_gps_transformed_camera_inlier_xyzt_points"] = True
        draw_control_dict["draw_gps_camera_pass"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = True
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.1)
        self.draw_and_save_solar_field_trajectories(
            "BC. GPS-Camera Alignment Results, Showing Connections (Per-Heliostat Tracking)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after aligning camera scan passes (track individual heliostat analysis results).
        self.solar_field.set_full_field_face_up()
        self.set_per_heliosat_configurations_from_gps_camera_alignment()
        draw_control_dict["draw_GPS_log"] = True
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = False
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = False
        draw_control_dict["draw_gps_camera_analysis"] = True
        draw_control_dict["draw_gps_transformed_camera_inlier_xyzt_points"] = True
        draw_control_dict["draw_gps_camera_pass"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = False
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.1)
        self.draw_and_save_solar_field_trajectories(
            "BD. GPS-Camera Alignment Results, with Trajectory Fragments (Per-Heliostat Tracking)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the trajectory fragments inferred from the reconstruction, after aligning camera scan passes (track individual heliostat analysis results).
        self.solar_field.set_full_field_face_up()
        self.set_per_heliosat_configurations_from_gps_camera_alignment()
        draw_control_dict["draw_GPS_log"] = False
        draw_control_dict["draw_gps_scan_passes"] = True
        draw_control_dict["draw_trajectory_fragments"] = False
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = False
        draw_control_dict["draw_gps_camera_analysis"] = True
        draw_control_dict["draw_gps_transformed_camera_inlier_xyzt_points"] = False
        draw_control_dict["draw_gps_camera_pass"] = True
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = False
        self.solar_field_style = rcsf.heliostat_outlines(color="grey")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.flight_log_style.set_color("grey")
        self.scan_pass_style = rcps.outline(color=self.scan_pass_color, linewidth=0.1)
        self.draw_and_save_solar_field_trajectories(
            "BE. GPS-Camera Alignment Results (Per-Heliostat Tracking)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

        # Draw the solar field with the heliostat configurations set by aligning camera scan passes, with no annotations (track individual heliostat analysis results).
        self.solar_field.set_full_field_face_up()
        self.set_per_heliosat_configurations_from_gps_camera_alignment()
        draw_control_dict["draw_GPS_log"] = False
        draw_control_dict["draw_gps_scan_passes"] = False
        draw_control_dict["draw_trajectory_fragments"] = False
        draw_control_dict["trajectory_fragment_selected_heliostats"] = None
        draw_control_dict["connect_trajectory_fragments"] = False
        draw_control_dict["draw_synchronization_points"] = False
        draw_control_dict["include_points_with_missing_corners"] = False
        draw_control_dict["include_non_refined_points"] = False
        draw_control_dict["draw_camera_passes"] = False
        draw_control_dict["draw_gps_camera_analysis"] = False
        draw_control_dict["draw_gps_transformed_camera_inlier_xyzt_points"] = False
        draw_control_dict["draw_gps_camera_pass"] = False
        draw_control_dict["draw_gps_transformed_camera_pass_connections"] = False
        self.solar_field_style = rcsf.heliostat_normals_outlines(color="k")
        self.heliostat_up_style = rch.outline(
            color="lightblue"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.heliostat_down_style = rch.outline(
            color="salmon"
        )  # Use normal_outline() to include surface normal needles, or outline() for just outlines.
        self.solar_field_style.heliostat_styles.add_special_names(up_heliostats, self.heliostat_up_style)
        self.solar_field_style.heliostat_styles.add_special_names(down_heliostats, self.heliostat_down_style)
        self.draw_and_save_solar_field_trajectories(
            "BF. Measured Solar Field Configuration After GPS-Camera Alignment (Per-Heliostat Tracking)",
            draw_control_dict,
            limits_3d_list=limits_3d_list,
            limits_xy_list=limits_xy_list,
            limits_xz_list=limits_xz_list,
            limits_yz_list=limits_yz_list,
        )

    # GPS TRAJECTORY ANALYSIS

    def draw_and_save_solar_field_trajectories(
        self,
        title,  # Used both above plot and for output filename (if not overridden).
        draw_control_dict,  # Dictionary of Boolean values indicating what to draw.
        output_figure_body=None,  # If None, use plot title, converting to file format.
        # In the following, a value of None means "Don't draw," while a
        # value of [ None ] means "draw without setting limits."
        limits_3d_list=None,  # Form:  [ None, [[x_min1,x_max1],[y_min1,y_max1],[z_min1,z_min2]], ...]
        limits_xy_list=None,  # Form:  [ None, [[x_min1,x_max1],[y_min1,y_max1]], ...]
        limits_xz_list=None,  # Form:  [ None, [[x_min1,x_max1],[y_min1,y_max1]], ...]
        limits_yz_list=None,
    ):  # Form:  [ None, [[x_min1,x_max1],[y_min1,y_max1]], ...]
        # Filename.
        if output_figure_body == None:
            output_figure_body = ft.convert_string_to_file_body(title)
        # 3-d oblique view.
        if limits_3d_list != None:
            view_3d = self.draw_solar_field_trajectories(title, draw_control_dict, vs.view_spec_3d())
            view_3d.show_and_save_multi_axis_limits(
                self.output_data_dir, output_figure_body, limits_list=limits_3d_list, grid=True
            )
        # xy view.
        if limits_xy_list != None:
            view_xy = self.draw_solar_field_trajectories(title, draw_control_dict, vs.view_spec_xy())
            view_xy.show_and_save_multi_axis_limits(
                self.output_data_dir, output_figure_body, limits_list=limits_xy_list, grid=False
            )
        # xz_view.
        if limits_xz_list != None:
            view_xz = self.draw_solar_field_trajectories(title, draw_control_dict, vs.view_spec_xz())
            view_xz.show_and_save_multi_axis_limits(
                self.output_data_dir, output_figure_body, limits_list=limits_xz_list, grid=True
            )
        # yz view.
        if limits_yz_list != None:
            view_yz = self.draw_solar_field_trajectories(title, draw_control_dict, vs.view_spec_yz())
            view_yz.show_and_save_multi_axis_limits(
                self.output_data_dir, output_figure_body, limits_list=limits_yz_list, grid=True
            )

    def draw_solar_field_trajectories(
        self, title, draw_control_dict, view_spec, grid=False
    ):  # ?? SCAFFOLDING RCB -- MOVE ANALYSIS OUT OF THIS RENDERING ROUTINE.
        # Solar field.
        # Required, because this creates the view.
        view = self.solar_field.draw_figure(
            self.figure_control, self.axis_control_m, view_spec, title, self.solar_field_style, grid=grid
        )
        # GPS flight log.
        if draw_control_dict["draw_GPS_log"]:
            view.draw_xyz_list(self.flight_log_xyz_list, style=self.flight_log_style, label="GPS Flight Log")
        if draw_control_dict["draw_gps_velocity_xy_change_minima"]:
            minima_xyz_list = []
            for minimum_dict in self.gps_velocity_xy_change_minima:
                x = minimum_dict["x"]
                y = minimum_dict["y"]
                z = minimum_dict["z"]
                minima_xyz_list.append([x, y, z])
            view.draw_xyz_list(minima_xyz_list, style=self.minimum_style, label="Delta Vx Minima")
        if draw_control_dict["draw_gps_velocity_xy_change_maxima"]:
            maxima_xyz_list = []
            for maximum_dict in self.gps_velocity_xy_change_maxima:
                x = maximum_dict["x"]
                y = maximum_dict["y"]
                z = maximum_dict["z"]
                maxima_xyz_list.append([x, y, z])
            view.draw_xyz_list(maxima_xyz_list, style=self.maximum_style, label="Delta Vx Maxima")
        if draw_control_dict["draw_gps_max_to_min_scan_passes"]:
            for max_min_pass_pair in self.maximum_to_minimum_pass_pair_list:
                maximum_dict = max_min_pass_pair[0]
                maximum_x = maximum_dict["x"]
                maximum_y = maximum_dict["y"]
                maximum_z = maximum_dict["z"]
                maximum_xyz = [maximum_x, maximum_y, maximum_z]
                minimum_dict = max_min_pass_pair[1]
                minimum_x = minimum_dict["x"]
                minimum_y = minimum_dict["y"]
                minimum_z = minimum_dict["z"]
                minimum_xyz = [minimum_x, minimum_y, minimum_z]
                view.draw_xyz_list([maximum_xyz, minimum_xyz], style=self.max_to_min_scan_pass_style)
        if draw_control_dict["draw_gps_min_to_max_scan_passes"]:
            for min_max_pass_pair in self.minimum_to_maximum_pass_pair_list:
                minimum_dict = min_max_pass_pair[0]
                minimum_x = minimum_dict["x"]
                minimum_y = minimum_dict["y"]
                minimum_z = minimum_dict["z"]
                minimum_xyz = [minimum_x, minimum_y, minimum_z]
                maximum_dict = min_max_pass_pair[1]
                maximum_x = maximum_dict["x"]
                maximum_y = maximum_dict["y"]
                maximum_z = maximum_dict["z"]
                maximum_xyz = [maximum_x, maximum_y, maximum_z]
                view.draw_xyz_list([minimum_xyz, maximum_xyz], style=self.min_to_max_scan_pass_style)
        if draw_control_dict["draw_synchronization_points"]:
            synch_time_list = self.synchronization_time_list()
            synch_xyz_list = []
            for synch_time in synch_time_list:
                synch_xyz = self.flight_log_xyz_given_time(synch_time)
                synch_xyz_list.append(synch_xyz)
            style = copy.deepcopy(self.synchronization_point_style)
            style.set_color(self.flight_log_style.color)
            view.draw_xyz_list(synch_xyz_list, style=style)
        if draw_control_dict["draw_gps_scan_passes"]:
            for gps_scan_pass in self.gps_scan_passes:
                begin_xyz = gps_scan_pass["stable_begin_xyzt"][0:3]
                end_xyz = gps_scan_pass["stable_end_xyzt"][0:3]
                view.draw_xyz_list([begin_xyz, end_xyz], style=self.scan_pass_style)
        # Trajectory fragments, synchronization points, and camera passes.
        if (
            draw_control_dict["draw_trajectory_fragments"]
            or draw_control_dict["draw_synchronization_points"]
            or draw_control_dict["draw_camera_passes"]
        ):
            idx = 0
            for hel_name in self.hel_frames_dict.keys():
                trajectory_fragment_color = color.color(idx, self.trajectory_fragment_color_wheel)
                camera_pass_color = color.color(idx, self.camera_pass_color_wheel)
                # Draw the fragments associated with this heliostat.
                if draw_control_dict["draw_trajectory_fragments"] and (
                    (draw_control_dict["trajectory_fragment_selected_heliostats"] == None)
                    or (hel_name in draw_control_dict["trajectory_fragment_selected_heliostats"])
                ):
                    # Draw the trajectory fragment for this heliostat.
                    frames_dict = self.hel_frames_dict[hel_name]
                    camera_xyz_world_list = []
                    for frame_id in frames_dict.keys():
                        frame_parameters_dict = frames_dict[frame_id]
                        if (draw_control_dict["include_points_with_missing_corners"] == True) or (
                            frame_parameters_dict["n_missing"] <= self.maximum_n_missing
                        ):
                            camera_xyz_world = frame_parameters_dict[
                                "per_heliostat_estimate_of_camera_xyz_in_world_coords_overall_time"
                            ]
                            camera_xyz_world_list.append(camera_xyz_world)
                    style = copy.deepcopy(self.trajectory_fragment_style)
                    style.set_color(trajectory_fragment_color)
                    if draw_control_dict["connect_trajectory_fragments"]:
                        view.draw_xyz_list(camera_xyz_world_list, style=style)
                    else:
                        list_of_camera_xyz_lists = self.split_xyz_list_where_distance_exceeds_maximum(
                            camera_xyz_world_list, draw_control_dict["trajectory_fragment_disconnect_threshold"]
                        )
                        for camera_xyz_list in list_of_camera_xyz_lists:
                            view.draw_xyz_list(camera_xyz_list, style=style)
                    # Synchronization point.
                    if draw_control_dict["draw_synchronization_points"]:
                        # Draw the synchronization point for this heliostat.
                        frames_dict = self.hel_frames_dict[hel_name]
                        synchronization_frame_id = self.get_synchronization_frame_id(hel_name)
                        synchronization_frame_parameters_dict = frames_dict[synchronization_frame_id]
                        synchronization_camera_xyz_world = synchronization_frame_parameters_dict[
                            "per_heliostat_estimate_of_camera_xyz_in_world_coords_overall_time"
                        ]
                        style = copy.deepcopy(self.synchronization_point_style)
                        style.set_color(
                            camera_pass_color
                        )  # Similar color, but slightly darker than trajectory fragment.
                        view.draw_xyz(synchronization_camera_xyz_world, style=style)
                # Draw the camera passes associated with each heliostat.
                if draw_control_dict["draw_camera_passes"]:
                    transformed_camera_pass_list = self.hel_transformed_camera_passes_dict[hel_name]
                    for transformed_camera_pass in transformed_camera_pass_list:
                        begin_xyz = transformed_camera_pass["stable_begin_xyzt"][0:3]
                        end_xyz = transformed_camera_pass["stable_end_xyzt"][0:3]
                        style = copy.deepcopy(self.camera_pass_style)
                        style.set_color(
                            camera_pass_color
                        )  # Similar color, but slightly darker than trajectory fragment.
                        view.draw_xyz_list([begin_xyz, end_xyz], style=style)
                        # Draw connecting lines between the camera passes and the corresponding GPS scan lines.
                        if draw_control_dict["draw_gps_transformed_camera_pass_connections"]:
                            style = copy.deepcopy(self.pass_connection_style)
                            style.set_color(trajectory_fragment_color)
                            gps_pass = self.find_matching_gps_pass(transformed_camera_pass)
                            if gps_pass != None:
                                # Compute nearest-neighbor line segments.
                                gps_line_3d = gps_pass["line_3d"]
                                camera_gps_point_pair_list = []
                                for transformed_camera_xyzt in transformed_camera_pass["inlier_xyzt_list"]:
                                    transformed_camera_inlier_xyz = transformed_camera_xyzt[0:3]
                                    nearest_gps_xyz = g3d.closest_point_on_line_3d(
                                        transformed_camera_inlier_xyz, gps_line_3d
                                    )
                                    view.draw_xyz_list([transformed_camera_inlier_xyz, nearest_gps_xyz], style=style)
                idx += 1
        # GPS-camera alignment analysis.
        if draw_control_dict["draw_gps_camera_analysis"]:
            idx = 0
            for hel_name in self.hel_gps_camera_analysis_dict:
                trajectory_fragment_color = color.color(idx, self.trajectory_fragment_color_wheel)
                camera_pass_color = color.color(idx, self.camera_pass_color_wheel)
                list_of_gps_camera_analysis_dicts = self.hel_gps_camera_analysis_dict[hel_name]
                for gps_camera_analysis_dict in list_of_gps_camera_analysis_dicts:
                    per_heliostat_transformed_camera_pass = gps_camera_analysis_dict[
                        "per_heliostat_transformed_camera_pass"
                    ]
                    if draw_control_dict["draw_gps_transformed_camera_inlier_xyzt_points"]:
                        xyzt_list = per_heliostat_transformed_camera_pass["inlier_xyzt_list"]
                        xyz_list = [xyzt[0:3] for xyzt in xyzt_list]
                        style = copy.deepcopy(self.trajectory_fragment_style)
                        style.set_color(trajectory_fragment_color)
                        view.draw_xyz_list(xyz_list, style=style)
                    if draw_control_dict["draw_gps_camera_pass"]:
                        begin_xyz = per_heliostat_transformed_camera_pass["stable_begin_xyzt"][0:3]
                        end_xyz = per_heliostat_transformed_camera_pass["stable_end_xyzt"][0:3]
                        style = copy.deepcopy(self.camera_pass_style)
                        style.set_color(camera_pass_color)
                        view.draw_xyz_list([begin_xyz, end_xyz], style=style)
                        # Draw connecting lines between the camera passes and the corresponding GPS scan lines.
                        if draw_control_dict["draw_gps_transformed_camera_pass_connections"]:
                            style = copy.deepcopy(self.pass_connection_style)
                            style.set_color(trajectory_fragment_color)
                            transformed_camera_gps_point_pair_list = gps_camera_analysis_dict[
                                "camera_gps_point_pair_list"
                            ]
                            for camera_gps_point_pair in transformed_camera_gps_point_pair_list:
                                camera_xyz = camera_gps_point_pair[0]
                                gps_xyz = camera_gps_point_pair[1]
                                view.draw_xyz_list([camera_xyz, gps_xyz], style=style)
                idx += 1
        # Return.
        return view

    def draw_and_save_gps_log_analysis_plots(self):
        data_linewidth = 0.5
        data_markersize = 1
        # Position.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Positions",
            x_column="time(sec)",
            y_column_label_styles=[
                ["x(m)", "x component", rcps.outline(color="b", linewidth=data_linewidth)],
                ["y(m)", "y component", rcps.outline(color="r", linewidth=data_linewidth)],
                ["z(m)", "z component", rcps.outline(color="g", linewidth=data_linewidth)],
            ],
            x_axis_label="time (sec)",
            y_axis_label="Position (m)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        figure_record.save(self.output_data_dir)
        # Velocity.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocities",
            x_column="time(sec)",
            y_column_label_styles=[
                ["velocity_average_x(m/sec)", "x component", rcps.outline(color="b", linewidth=data_linewidth)],
                ["velocity_average_y(m/sec)", "y component", rcps.outline(color="r", linewidth=data_linewidth)],
                ["velocity_average_z(m/sec)", "z component", rcps.outline(color="g", linewidth=data_linewidth)],
            ],
            x_axis_label="time (sec)",
            y_axis_label="Velocity (m/sec)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        figure_record.save(self.output_data_dir)
        # Speed.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Speed",
            x_column="time(sec)",
            y_column_label_styles=[
                ["speed_average(m/sec)", None, rcps.outline(color="b", linewidth=data_linewidth)],
                ["speed(mps)", None, rcps.outline(color="r", linewidth=data_linewidth)],
            ],
            x_axis_label="time (sec)",
            y_axis_label="Speed (m/sec)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        ax = plt.gca()
        ax.set_ylim([-0.4, 12.4])
        ax.yaxis.set_ticks(np.linspace(0, 12, 25))
        figure_record.save(self.output_data_dir)
        # Change in speed.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Speed Change",
            x_column="time(sec)",
            y_column_label_styles=[["delta_speed(m/sec)", None, rcps.outline(color="b", linewidth=data_linewidth)]],
            x_axis_label="time (sec)",
            y_axis_label="Change in Speed (m/sec)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        figure_record.save(self.output_data_dir)
        # Magnitude of change in speed.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Speed Change Magnitude",
            x_column="time(sec)",
            y_column_label_styles=[["abs_delta_speed(m/sec)", None, rcps.outline(color="b", linewidth=data_linewidth)]],
            # y_column_label_styles = [ ['speed_average(m/sec)', None, rcps.data_curve(color='b', linewidth=data_linewidth, markersize=data_markersize)] ],
            x_axis_label="time (sec)",
            y_axis_label="Magnitude of Change in Speed (m/sec)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        figure_record.save(self.output_data_dir)
        # XY direction.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity XY Angle",
            x_column="time(sec)",
            y_column_label_styles=[["velocity_angle_xy(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]],
            x_axis_label="time (sec)",
            y_axis_label="Velocity Angle in (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        figure_record.save(self.output_data_dir)
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity XY Angle Zoom 1",
            x_column="time(sec)",
            y_column_label_styles=[["velocity_angle_xy(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]],
            x_axis_label="time (sec)",
            y_axis_label="Velocity Angle in (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        ax = plt.gca()
        ax.set_ylim([1.5, 1.7])
        figure_record.save(self.output_data_dir)
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity XY Angle Zoom 2",
            x_column="time(sec)",
            y_column_label_styles=[["velocity_angle_xy(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]],
            x_axis_label="time (sec)",
            y_axis_label="Velocity Angle in (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        ax = plt.gca()
        ax.set_ylim([-1.5, -1.7])
        figure_record.save(self.output_data_dir)
        # Z direction.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity Z Angle",
            x_column="time(sec)",
            y_column_label_styles=[["velocity_angle_z(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]],
            x_axis_label="time (sec)",
            y_axis_label="Velocity Elevation Angle Above (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        figure_record.save(self.output_data_dir)
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity Z Angle Zoom",
            x_column="time(sec)",
            y_column_label_styles=[["velocity_angle_z(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]],
            x_axis_label="time (sec)",
            y_axis_label="Velocity Elevation Angle Above (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        ax = plt.gca()
        ax.set_ylim([-0.1, 0.1])
        figure_record.save(self.output_data_dir)
        # Change in xy direction.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity XY Angle Change",
            x_column="time(sec)",
            y_column_label_styles=[
                ["delta_velocity_angle_xy(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]
            ],
            x_axis_label="time (sec)",
            y_axis_label="Change in Velocity Angle in (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        # Indicate the maximum-to-minimum pairs.
        for max_min_pass_pair in self.maximum_to_minimum_pass_pair_list:
            maximum_dict = max_min_pass_pair[0]
            time_maximum = maximum_dict["time"]
            maximum_delta_velocity_angle_xy = maximum_dict["delta_velocity_angle_xy"]
            minimum_dict = max_min_pass_pair[1]
            time_minimum = minimum_dict["time"]
            minimum_delta_velocity_angle_xy = minimum_dict["delta_velocity_angle_xy"]
            gp.add_xy_list_to_plot(
                figure_record,
                [[time_maximum, maximum_delta_velocity_angle_xy], [time_minimum, minimum_delta_velocity_angle_xy]],
                rcps.outline(color=self.max_to_min_color),
            )
        # Indicate the minimum-to-maximum pairs.
        for min_max_pass_pair in self.minimum_to_maximum_pass_pair_list:
            minimum_dict = min_max_pass_pair[0]
            time_minimum = minimum_dict["time"]
            minimum_delta_velocity_angle_xy = minimum_dict["delta_velocity_angle_xy"]
            maximum_dict = min_max_pass_pair[1]
            time_maximum = maximum_dict["time"]
            maximum_delta_velocity_angle_xy = maximum_dict["delta_velocity_angle_xy"]
            gp.add_xy_list_to_plot(
                figure_record,
                [[time_minimum, minimum_delta_velocity_angle_xy], [time_maximum, maximum_delta_velocity_angle_xy]],
                rcps.outline(color=self.min_to_max_color),
            )
        # Indicate the local minima.
        minima_tdelta_list = []
        for minimum_dict in self.gps_velocity_xy_change_minima:
            time = minimum_dict["time"]
            delta_velocity_angle_xy = minimum_dict["delta_velocity_angle_xy"]
            minima_tdelta_list.append([time, delta_velocity_angle_xy])
        gp.add_xy_list_to_plot(
            figure_record, minima_tdelta_list, rcps.marker(color="m", markersize=data_markersize), label="Minima"
        )
        # Indicate the local maxima.
        maxima_tdelta_list = []
        for maximum_dict in self.gps_velocity_xy_change_maxima:
            time = maximum_dict["time"]
            delta_velocity_angle_xy = maximum_dict["delta_velocity_angle_xy"]
            maxima_tdelta_list.append([time, delta_velocity_angle_xy])
        gp.add_xy_list_to_plot(
            figure_record, maxima_tdelta_list, rcps.marker(color="r", markersize=data_markersize), label="Maxima"
        )
        figure_record.save(self.output_data_dir)
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity XY Angle Change Zoom",
            x_column="time(sec)",
            y_column_label_styles=[
                ["delta_velocity_angle_xy(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]
            ],
            x_axis_label="time (sec)",
            y_axis_label="Change in Velocity Angle in (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        ax = plt.gca()
        ax.set_ylim([-0.1, 0.1])
        figure_record.save(self.output_data_dir)
        # Change in z_direction.
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity Z Angle Change",
            x_column="time(sec)",
            y_column_label_styles=[
                ["delta_velocity_angle_z(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]
            ],
            x_axis_label="time (sec)",
            y_axis_label="Change in Velocity Elevation Angle Above (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        figure_record.save(self.output_data_dir)
        figure_record = pp.dataframe_plot(
            self.figure_control,
            self.flight_log_df,
            title="GPS Trajectory Velocity Z Angle Change Zoom",
            x_column="time(sec)",
            y_column_label_styles=[
                ["delta_velocity_angle_z(rad)", None, rcps.outline(color="b", linewidth=data_linewidth)]
            ],
            x_axis_label="time (sec)",
            y_axis_label="Change in Velocity Elevation Angle Above (x,y) Plane (rad)",
            x_axis_grid=True,
            y_axis_grid=True,
        )
        ax = plt.gca()
        ax.set_ylim([-0.1, 0.1])
        figure_record.save(self.output_data_dir)

    # def draw_solar_field_situation(self,
    #                                figure_control,
    #                                axis_control_m,
    #                                view_spec,
    #                                title,
    #                                solar_field,
    #                                solar_field_style):
    #     # View setup
    #     fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title=title)
    #     view = fig_record.view
    #     # Draw
    #     solar_field.draw(view, solar_field_style)
    #     # Return
    #     return view

    #     def draw_key_demonstration_figures(self,
    #                                        figure_control,
    #                                        axis_control_m,
    #                                        view_spec,
    #                                        solar_field,
    #                                        aimpoint_xyz,
    #                                        when_ymdhmsz,
    #                                        synch_az,
    #                                        synch_el,
    #                                        up_az,
    #                                        up_el):
    #         # Figure selection.
    #         draw_annotated_heliostat   = True
    #         draw_multi_heliostat       = True
    #         draw_annotated_solar_field = True
    #         draw_solar_field_subset    = True

    #         # Annotated heliostat.
    #         if draw_annotated_heliostat:
    #             # Heliostat selection
    #             heliostat_name = '5E10'
    #             # View setup
    #             fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title=('Heliostat ' + heliostat_name + ', with Highlighting'))
    #             view = fig_record.view
    #             # Tracking setup
    #             heliostat = solar_field.lookup_heliostat(heliostat_name)
    #             heliostat.set_tracking(aimpoint_xyz, solar_field.origin_lon_lat, when_ymdhmsz)
    #             # Style setup
    #             default_heliostat_style = rch.normal_facet_outlines()
    #             default_heliostat_style.facet_styles.add_special_name(16, rcf.corner_normals_outline_name(color='c'))
    #             default_heliostat_style.facet_styles.add_special_names([1,4, 7, 24, 25], rcf.normal_outline(color='r'))
    #             heliostat_styles = rce.RenderControlEnsemble(default_heliostat_style)
    #             # Comment
    #             fig_record.comment.append("Demonstration of example heliostat annotations.")
    #             fig_record.comment.append("Black:   Facet outlines.")
    #             fig_record.comment.append("Black:   Overall heliostat surface normal.")
    #             fig_record.comment.append("Red:     Highlighted facets and their surface normals.")
    #             fig_record.comment.append("Cyan:    Highlighted facet with facet name and facet surface normal drawn at corners.")
    #             # Draw
    #             heliostat.draw(view, heliostat_styles)
    #             view.show()

    #         # Multiple heliostats.
    #         if draw_multi_heliostat:
    #             # Heliostat selection
    #             heliostat_spec_list = [ ['11W1', hc.face_up(),    rch.name()                            ],
    #                                     ['11E1', hc.face_up(),    rch.centroid(color='r')               ],
    #                                     ['11E2', hc.face_up(),    rch.centroid_name(color='g')          ],
    #                                     ['12W1', hc.face_north(), rch.facet_outlines(color='b')         ],
    #                                     ['12E1', hc.face_south(), rch.normal_outline(color='c')         ],
    #                                     ['12E2', hc.face_east(),  rch.corner_normals_outline(color='m') ],
    #                                     ['13W1', hc.face_west(),  rch.normal_facet_outlines(color='g')  ],
    #                                     ['13E1', hc.face_up(),    rch.facet_outlines_normals(color='c') ],
    #                                     ['13E2', hc.NSTTF_stow(), rch.facet_outlines_corner_normals() ] ]
    #             # View setup
    #             fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title='Example Poses and Styles')
    #             view = fig_record.view
    #             # Setup and draw
    #             for heliostat_spec in heliostat_spec_list:
    #                 heliostat_name   = heliostat_spec[0]
    #                 heliostat_config = heliostat_spec[1]
    #                 heliostat_style  = heliostat_spec[2]
    #                 # Configuration setup
    #                 heliostat = solar_field.lookup_heliostat(heliostat_name)
    #                 heliostat.set_configuration(heliostat_config)
    #                 # Style setup
    #                 heliostat_styles = rce.RenderControlEnsemble(heliostat_style)
    #                 # Draw
    #                 heliostat.draw(view, heliostat_styles)
    #             # Comment
    #             fig_record.comment.append("Demonstration of various example heliostat drawing modes.")
    #             fig_record.comment.append("Black:   Name only.")
    #             fig_record.comment.append("Red:     Centroid only.")
    #             fig_record.comment.append("Green:   Centroid and name.")
    #             fig_record.comment.append("Blue:    Facet outlines.")
    #             fig_record.comment.append("Cyan:    Overall outline and overall surface normal.")
    #             fig_record.comment.append("Magneta: Overall outline and overall surface normal, drawn at corners.")
    #             fig_record.comment.append("Green:   Facet outlines and overall surface normal.")
    #             fig_record.comment.append("Cyan:    Facet outlines and facet surface normals.")
    #             fig_record.comment.append("Black:   Facet outlines and facet surface normals drawn at facet corners.")
    #             view.show()

    #         # Annotated solar field.
    #         if draw_annotated_solar_field:
    #             # Heliostat selection
    #             up_heliostats      = ['6E3', '8E3']
    #             stowed_heliostats  = ['9W1', '12E14']
    #             synched_heliostats = ['5E1', '5E2', '5E3', '5E4', '5E5', '5E6', '5E7',
    #                                 '6E1', '6E2',        '6E4', '6E5', '6E6', '6E7',
    #                                 '7E1', '7E2', '7E3', '7E4', '7E5', '7E6', '7E7',]
    #             # View setup
    #             fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title='Solar Field Situation')
    #             view = fig_record.view
    #             # Configuration setup
    #             solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
    #             solar_field.set_heliostats_configuration(stowed_heliostats, hc.NSTTF_stow())
    #             synch_configuration = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
    #             solar_field.set_heliostats_configuration(synched_heliostats, synch_configuration)
    #             up_configuration = hc.HeliostatConfiguration(az=up_az, el=up_el)
    #             solar_field.set_heliostats_configuration(up_heliostats, up_configuration)
    #             # Style setup
    #             solar_field_style = rcsf.heliostat_outlines(color='b')
    #             solar_field_style.heliostat_styles.add_special_names(up_heliostats, rch.normal_outline(color='c'))
    #             solar_field_style.heliostat_styles.add_special_names(stowed_heliostats, rch.normal_outline(color='r'))
    #             solar_field_style.heliostat_styles.add_special_names(synched_heliostats, rch.normal_outline(color='g'))
    #             # Comment
    #             fig_record.comment.append("A solar field situation with heliostats in varying status.")
    #             fig_record.comment.append("Blue heliostats are tracking.")
    #             fig_record.comment.append("Cyan heliostats are face up.")
    #             fig_record.comment.append("Red heliostats are in stow (out of service).")
    #             fig_record.comment.append("Green heliostats are in a fixed configuration (az={0:.1f} deg, el={1:.1f} deg).".format(np.rad2deg(synch_az), np.rad2deg(synch_el)))
    #             # Draw
    #             solar_field.draw(view, solar_field_style)
    #             view.show()

    #         # Solar field subset.
    #         if draw_solar_field_subset:
    #             # Heliostat selection
    #             up_heliostats       = ['6E3', '8E3']
    #             stowed_heliostats   = ['6E2', '8E5']
    #             synched_heliostats  = ['5E1', '5E2', '5E3', '5E4', '5E5', '5E6', '5E7',
    #                                 '6E1',               '6E4', '6E5', '6E6', '6E7',
    #                                 '7E1', '7E2', '7E3', '7E4', '7E5', '7E6', '7E7',]
    #             tracking_heliostats = ['8E1', '8E2',        '8E4',        '8E6', '8E7',
    #                                 '9E1', '9E2', '9E3', '9E4', '9E5', '9E6', '9E7',]
    #             # View setup
    #             fig_record = fm.setup_figure_for_3d_data(figure_control, axis_control_m, view_spec, title='Selected Heliostats')
    #             view = fig_record.view
    #             # Configuration setup
    #             solar_field.set_full_field_tracking(aimpoint_xyz=aimpoint_xyz, when_ymdhmsz=when_ymdhmsz)
    #             solar_field.set_heliostats_configuration(stowed_heliostats, hc.NSTTF_stow())
    #             synch_configuration = hc.HeliostatConfiguration(az=synch_az, el=synch_el)
    #             solar_field.set_heliostats_configuration(synched_heliostats, synch_configuration)
    #             up_configuration = hc.HeliostatConfiguration(az=up_az, el=up_el)
    #             solar_field.set_heliostats_configuration(up_heliostats, up_configuration)
    #             # Style setup
    #             solar_field_style = rcsf.heliostat_blanks()
    #             solar_field_style.heliostat_styles.add_special_names(up_heliostats, rch.normal_outline(color='c'))
    #             solar_field_style.heliostat_styles.add_special_names(stowed_heliostats, rch.normal_outline(color='r'))
    #             solar_field_style.heliostat_styles.add_special_names(synched_heliostats, rch.normal_outline(color='g'))
    #             solar_field_style.heliostat_styles.add_special_names(tracking_heliostats, rch.normal_outline(color='b'))
    #             # Comment
    #             fig_record.comment.append("A subset of heliostats selected, so that plot is effectively zoomed in.")
    #             fig_record.comment.append("Blue heliostats are tracking.")
    #             fig_record.comment.append("Cyan heliostats are face up.")
    #             fig_record.comment.append("Red heliostats are in stow (out of service).")
    #             fig_record.comment.append("Green heliostats are in a fixed configuration (az={0:.1f} deg, el={1:.1f} deg).".format(np.rad2deg(synch_az), np.rad2deg(synch_el)))
    #             # Draw
    #             solar_field.draw(view, solar_field_style)
    #             view.show()

    # ---------------------------------------------------------------------------------------------------------------------------------------
    #
    #  SAVE RESULTS
    #

    # SAVE FLIGHT LOG

    def save_enhanced_flight_log(self):
        if not (os.path.exists(self.output_data_dir)):
            os.makedirs(self.output_data_dir)
            (input_flight_log_dir, input_flight_log_body, input_flight_log_ext) = ft.path_components(
                self.input_flight_log_dir_body_ext
            )
            output_flight_log_plus_body_ext = input_flight_log_body + "_plus" + input_flight_log_ext
            output_flight_log_plus_dir_body_ext = os.path.join(self.output_data_dir, output_flight_log_plus_body_ext)
            print(
                "In TrajectoryAnalysis.save_enhanced_flight_log(), saving enhanced flight log file :",
                output_flight_log_plus_dir_body_ext,
            )
            # ?? SCAFFOLDING RCB -- FLIGHT_LOG SHOULD BE A CLASS, SUPPORTING FUNCTIONS SUCH A MID-TIME, ALTITUDE AND POSITION LIMITS, ETC.  SHOULD ALSO HANDLE LOGS FROM DIFFERENT UAS SOURCES.
            self.flight_log_df.to_csv(output_flight_log_plus_dir_body_ext)
        # Return.
        return output_flight_log_plus_dir_body_ext

    # SAVE GPS TIME CORRESPONDING TO FLIGHT LOG ZERO TIME

    def save_gps_ymdhmsz_given_flight_log_zero_seconds(self):
        output_body = self.input_video_body + "_gps_ymdhmsz_given_flight_log_zero_seconds"
        explain = "GPS time corresponding to flight log zero"
        # Convert list to dictionary, adding keys.
        output_dict = {
            "year": self.gps_ymdhmsz_given_flight_log_zero_seconds[0],
            "month": self.gps_ymdhmsz_given_flight_log_zero_seconds[1],
            "day": self.gps_ymdhmsz_given_flight_log_zero_seconds[2],
            "hour": self.gps_ymdhmsz_given_flight_log_zero_seconds[3],
            "minute": self.gps_ymdhmsz_given_flight_log_zero_seconds[4],
            "second": self.gps_ymdhmsz_given_flight_log_zero_seconds[5],
            "zone": self.gps_ymdhmsz_given_flight_log_zero_seconds[6],
        }
        # Save the dict to a csv file.
        return ft.write_dict_file(
            explain,  # Explanatory string to include in notification output.  None to skip.
            self.output_data_dir,  # Directory to write file.  See below if not exist.
            output_body,  # Body of output filename; extension is ".csv"
            output_dict,  # Dictionary to write.
            decimal_places=6,  # Number of decimal places to write for floating-point values.
            error_if_dir_not_exist=True,
        )  # If True, error if not exist.  If False, create dir if necessary.

    # SAVE GPS VELOCITY XY CHANGE POINTS

    def save_gps_velocity_xy_change_points(self, list_of_gps_velocity_xy_change_dicts, minima_or_maxima_str):
        output_body = self.input_video_body + "_gps_velocity_xy_change_" + minima_or_maxima_str
        explain = "GPS velocity xy change " + minima_or_maxima_str + " file"
        return dt.save_list_of_one_level_dicts(
            list_of_gps_velocity_xy_change_dicts,
            self.output_data_dir,
            output_body,
            explain,
            error_if_dir_not_exist=False,
        )

    # SAVE GPS SCAN ANALYSIS MINIMA/MAXIMA PAIRS

    def save_gps_velocity_xy_change_pairs(self, pass_pair_list, maximum_to_minimum_or_minimum_to_maximum_str):
        output_body = (
            self.input_video_body
            + "_gps_velocity_xy_change_"
            + maximum_to_minimum_or_minimum_to_maximum_str
            + "_pass_pair_list"
        )
        explain = "GPS velocity xy change " + maximum_to_minimum_or_minimum_to_maximum_str + " pass pair file"
        return dt.save_list_of_one_level_dict_pairs(
            pass_pair_list, self.output_data_dir, output_body, explain, error_if_dir_not_exist=False
        )

    # SAVE GPS SCAN PASSES

    def save_gps_scan_passes(self):
        heading_line = self.gps_scan_pass_heading_line()
        data_lines = []
        for gps_scan_pass in self.gps_scan_passes:
            data_lines.append(self.gps_scan_pass_data_line(gps_scan_pass))
        output_body = self.input_video_body + "_gps_scan_data"
        explain = "GPS scan pass file"
        output_dir_body_ext = ft.write_csv_file(
            explain,  # Explanatory string to include in notification output.  None to skip.
            self.output_data_dir,  # Directory to write file.  See below if not exist.
            output_body,  # Body of output filename; extension is ".csv"
            heading_line,  # First line to write to file.  None to skip.
            data_lines,  # Subsequent lines to write to file.
            error_if_dir_not_exist=False,
        )  # If True, error if not exist.  If False, create dir if necessary.
        # Return.
        return output_dir_body_ext

    def gps_scan_pass_heading_line(self):
        # Stable begin point.
        heading_line_str = ""
        heading_line_str += "stable_begin_x,stable_begin_y,stable_begin_z,stable_begin_t"
        # Stable end point.
        heading_line_str += ","
        heading_line_str += "stable_end_x,stable_end_y,stable_end_z,stable_end_t"
        # Embedding 3-d line.
        heading_line_str += ","
        heading_line_str += "line_3d_length,line_3d_ux,line_3d_uy,line_3d_uz,line_3d_theta,line_3d_eta"
        # RMS point-to-line distance.
        heading_line_str += ","
        heading_line_str += "rms_distance_to_line"
        # Number of points.
        heading_line_str += ","
        heading_line_str += "n_inliers"
        # List of individual point-to-line distances.
        heading_line_str += ","
        heading_line_str += "distance_to_line_list"
        # Heading interlude.
        # We cannot align the heading for xyzt points, because the number of points varies from row to row.
        heading_line_str += ","
        heading_line_str += "...varies..."
        # List of individual points.
        heading_line_str += ","
        heading_line_str += "inlier_xyzt_list"
        # Return.
        return heading_line_str

    def gps_scan_pass_data_line(self, scan_pass):
        # Stable begin point.
        data_line_str = ""
        data_line_str += (
            str(scan_pass["stable_begin_xyzt"][0])
            + ","
            + str(scan_pass["stable_begin_xyzt"][1])
            + ","
            + str(scan_pass["stable_begin_xyzt"][2])
            + ","
            + str(scan_pass["stable_begin_xyzt"][3])
        )
        # Stable end point.
        data_line_str += ","
        data_line_str += (
            str(scan_pass["stable_end_xyzt"][0])
            + ","
            + str(scan_pass["stable_end_xyzt"][1])
            + ","
            + str(scan_pass["stable_end_xyzt"][2])
            + ","
            + str(scan_pass["stable_end_xyzt"][3])
        )
        # Embedding 3-d line.
        data_line_str += ","
        data_line_str += (
            str(scan_pass["line_3d"]["length"])
            + ","
            + str(scan_pass["line_3d"]["uxyz"][0])
            + ","
            + str(scan_pass["line_3d"]["uxyz"][1])
            + ","
            + str(scan_pass["line_3d"]["uxyz"][2])
            + ","
            + str(scan_pass["line_3d"]["theta"])
            + ","
            + str(scan_pass["line_3d"]["eta"])
        )
        # RMS point-to-line distance.
        data_line_str += ","
        data_line_str += str(scan_pass["rms_distance_to_line"])
        # Number of points.
        n_points = len(scan_pass["inlier_xyzt_list"])
        data_line_str += ","
        data_line_str += str(n_points)
        # List of individual point-to-line distances.
        distance_to_line_list = scan_pass["distance_to_line_list"]
        if n_points != len(distance_to_line_list):
            print(
                "ERROR: In gps_scan_pass_data_line(), unexpected number of distance values encountered.  n_points =",
                n_points,
                "; len(distance_to_line_list) =",
                len(distance_to_line_list),
            )
        for d in distance_to_line_list:
            data_line_str += "," + str(d)
        # List of individual points.
        inlier_xyzt_list = scan_pass["inlier_xyzt_list"]
        if n_points != len(inlier_xyzt_list):
            print(
                "ERROR: In gps_scan_pass_data_line(), unexpected number of inliers encountered.  n_points =",
                n_points,
                "; len(inlier_xyzt_list) =",
                len(inlier_xyzt_list),
            )
        for xyzt in inlier_xyzt_list:
            data_line_str += "," + str(xyzt[0])
            data_line_str += "," + str(xyzt[1])
            data_line_str += "," + str(xyzt[2])
            data_line_str += "," + str(xyzt[3])
        # Return.
        return data_line_str

    # TRAJECTORY FRAGMENTS ASSOCIATED WITH EACH HELIOSTAT

    def save_hel_frames_dict(self):
        output_body = self.input_video_body + "_hel_frames_dict"
        explain = "trajectory fragments file"
        return ft.write_pickle_file(
            explain, self.output_data_dir, output_body, self.hel_frames_dict, error_if_dir_not_exist=False
        )

    def save_synchronization_constants(self):
        output_body = self.input_video_body + "_synchronization_constants"
        explain = "synchronization constants file"
        # Prepare data to write.
        heading_line = None
        data_lines = []
        data_lines.append("synchronization_slope," + str(self.synchronization_slope))
        data_lines.append("synchronization_intercept," + str(self.synchronization_intercept))
        data_lines.append("n_synchronization_pairs," + str(len(self.synchronization_pair_list)))
        idx = 1
        for synch_pair in self.synchronization_pair_list:
            # Example synchronization pair:  [['max_to_min', 0, 'stop', 110.465], ['5W9', 0, 1911]]
            # First item.
            gps_entry = synch_pair[0]
            direction = gps_entry[0]
            gps_halt_idx = gps_entry[1]
            start_or_stop = gps_entry[2]
            gps_halt_time = gps_entry[3]
            # Second item.
            camera_entry = synch_pair[1]
            hel_name = camera_entry[0]
            camera_halt_idx = camera_entry[1]
            camera_halt_frame = camera_entry[2]
            # Add to data lines.
            data_lines.append("direction_" + str(idx) + "," + str(direction))
            data_lines.append("gps_halt_idx_" + str(idx) + "," + str(gps_halt_idx))
            data_lines.append("start_or_stop_" + str(idx) + "," + str(start_or_stop))
            data_lines.append("gps_halt_time_" + str(idx) + "," + str(gps_halt_time))
            data_lines.append("hel_name_" + str(idx) + "," + str(hel_name))
            data_lines.append("camera_halt_idx_" + str(idx) + "," + str(camera_halt_idx))
            data_lines.append("camera_halt_frame_" + str(idx) + "," + str(camera_halt_frame))
            idx += 1
        # Write.
        output_dir_body_ext = ft.write_csv_file(
            explain,  # Explanatory string to include in notification output.  None to skip.
            self.output_data_dir,  # Directory to write file.  See below if not exist.
            output_body,  # Body of output filename; extension is ".csv"
            heading_line,  # First line to write to file.  None to skip.
            data_lines,  # Subsequent lines to write to file.
            error_if_dir_not_exist=False,
        )  # If True, error if not exist.  If False, create dir if necessary.
        # Return.
        return output_dir_body_ext

    def save_hel_camera_passes_dict(self):
        output_body = self.input_video_body + "_hel_camera_passes_dict"
        explain = "heliostat camera passes file"
        return ft.write_pickle_file(
            explain, self.output_data_dir, output_body, self.hel_camera_passes_dict, error_if_dir_not_exist=False
        )

    def save_hel_gps_camera_analysis_dict(self):
        output_body = self.input_video_body + "_hel_gps_camera_analysis_dict"
        explain = "heliost GPS-camera analysis file"
        return ft.write_pickle_file(
            explain, self.output_data_dir, output_body, self.hel_gps_camera_analysis_dict, error_if_dir_not_exist=False
        )

    def check_pickle_files(self):
        # Trajectory fragments.
        print("\nIn TrajectoryAnalysis.check_pickle_files(), loaded hel_frames_dict:")
        self.print_hel_frames_dict_aux(pickle.load(open(self.hel_frames_dict_dir_body_ext, "rb")))
        # Camera passes.
        print("\nIn TrajectoryAnalysis.check_pickle_files(), loaded hel_camera_passes_dict:")
        self.print_hel_camera_passes_dict_aux(
            pickle.load(open(self.hel_camera_passes_dict_dir_body_ext, "rb")), max_heliostats=2
        )
        # GPS-camera analysis.
        print("\nIn TrajectoryAnalysis.check_pickle_files(), loaded hel_gps_camera_analysis_dict:")
        self.print_hel_gps_camera_analysis_dict_aux(
            pickle.load(open(self.hel_gps_camera_analysis_dict_dir_body_ext, "rb")), max_heliostats=2
        )


# # # LOAD RESULT

# # def read_heliostat_corner_2d_trajectories(self):
# #     # Projected.
# #     print('In TrajectoryAnalysis.read_heliostat_corner_2d_trajectories(), reading heliostat projected distorted corner_2d trajectories file: ', self.heliostat_projected_corners_3d_dir_body_ext)
# #     self.heliostat_projected_corners_3d_nfxl = nfxl.NameFrameXyList()
# #     self.heliostat_projected_corners_3d_nfxl.load(self.heliostat_projected_corners_3d_dir_body_ext)
# #     # Confirm what was read.
# #     print('In TrajectoryAnalysis.read_heliostat_corners_3d(), heliostat projected corners_3d read:')
# #     self.heliostat_projected_corners_3d_nfxl.print(max_keys=12, max_value_length=200, indent=4)
# #     # Confirmed.
# #     print('In TrajectoryAnalysis.read_heliostat_corners_3d(), reading heliostat confirmed corners_3d file: ', self.heliostat_confirmed_corners_3d_dir_body_ext)
# #     self.heliostat_confirmed_corners_3d_nfxl = nfxl.NameFrameXyList()
# #     self.heliostat_confirmed_corners_3d_nfxl.load(self.heliostat_confirmed_corners_3d_dir_body_ext)
# #     # Confirm what was read.
# #     print('In TrajectoryAnalysis.read_heliostat_corners_3d(), heliostat confirmed corners_3d read:')
# #     self.heliostat_confirmed_corners_3d_nfxl.print(max_keys=12, max_value_length=200, indent=4)


# # def read_heliostat_corners_3d(self):
# #     # Projected.
# #     print('In TrajectoryAnalysis.read_heliostat_corners_3d(), reading heliostat projected corners_3d file: ', self.heliostat_projected_corners_3d_dir_body_ext)
# #     self.heliostat_projected_corners_3d_nfxl = nfxl.NameFrameXyList()
# #     self.heliostat_projected_corners_3d_nfxl.load(self.heliostat_projected_corners_3d_dir_body_ext)
# #     # Confirm what was read.
# #     print('In TrajectoryAnalysis.read_heliostat_corners_3d(), heliostat projected corners_3d read:')
# #     self.heliostat_projected_corners_3d_nfxl.print(max_keys=12, max_value_length=200, indent=4)
# #     # Confirmed.
# #     print('In TrajectoryAnalysis.read_heliostat_corners_3d(), reading heliostat confirmed corners_3d file: ', self.heliostat_confirmed_corners_3d_dir_body_ext)
# #     self.heliostat_confirmed_corners_3d_nfxl = nfxl.NameFrameXyList()
# #     self.heliostat_confirmed_corners_3d_nfxl.load(self.heliostat_confirmed_corners_3d_dir_body_ext)
# #     # Confirm what was read.
# #     print('In TrajectoryAnalysis.read_heliostat_corners_3d(), heliostat confirmed corners_3d read:')
# #     self.heliostat_confirmed_corners_3d_nfxl.print(max_keys=12, max_value_length=200, indent=4)

if __name__ == "__main__":
    # # Execution control.

    # log_dir_body_ext                           = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_190_TrajectoryAnalysis/mavic_zoom/log/TrajectoryAnalysis_log.txt'
    # # Input/output sources.
    # specifications                             = Dspec.nsttf_specifications()  # Solar field parameters.
    # theoretical_flat_heliostat_dir_body_ext    = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/010_HeliostatModel/data/NSTTF_Facet_Corners.csv'
    # input_video_dir_body_ext                   = experiment_dir() + '2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4'
    # input_frame_dir                            = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/'
    # input_frame_id_format                      = '06d' # Note different from format used in ffmpeg call, which is '.%06d'
    # input_video_projected_tracks_dir_body_ext  = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_170_HeliostatTracks/mavic_zoom/data/DJI_427t_428_429_heliostat_projected_tracks_nfxl.csv'
    # input_video_confirmed_tracks_dir_body_ext  = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_170_HeliostatTracks/mavic_zoom/data/DJI_427t_428_429_heliostat_confirmed_tracks_nfxl.csv'
    # output_data_dir                            = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_190_TrajectoryAnalysis/mavic_zoom/data/'
    # output_render_dir                          = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_190_TrajectoryAnalysis/mavic_zoom/render/'
    # output_construction_dir                    = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/Small_190c_TrajectoryAnalysis/mavic_zoom/'

    log_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/190_TrajectoryAnalysis/mavic_zoom/log/TrajectoryAnalysis_log.txt"
    )
    velocity_calculation_offset_fwd_bwd = (
        10  # Integer number of steps to skip to fetch a value for velocity computation.  Must be at least 1.
    )
    delta_velocity_angle_xy_peak_threshold = 0.5  # 1.0   # radian
    delta_velocity_angle_xy_non_peak_threshold = 0.1  # radian
    turn_overshoot_skip_time = 1.0  # sec.   Time window to allow turn correction overshoot without inferring an interval varying from turn min to turn max.
    scan_establish_velocity_time = 0.5  # sec.   Time for the UAS to reach a "reasonably stable " velocity at the beginning of a scan pass.  The UAS is "establishing" its constant scan velocity.
    scan_discard_velocity_time = 0.5  # sec.   Time prior to the end point when a UAS bgeins changing velocity prior to the pass endpoint.  The UAS is "discarding" its constant scan velocity.
    minimum_scan_pass_time = 10.0  # sec.   Duration of the shortest possible scan pass, after already trimming away the times to establish and discard the scan velocity.
    nominal_scan_speed = 7.0  # m/sec. Nominal speed of UAS flight during a linear scan pass.
    # m/sec. Tolerance to use wheen deciding that the average speed of a candidate pass is consistent with a possible scan.  Not the scan speed control tolerance; larger than that.
    scan_speed_tolerance = 0.5
    nominal_scan_velocity_z = 0.0  # m/sec. Nominal vertical speed of UAS flight during a linear scan pass.
    # m/sec. Tolerance to use wheen deciding that the average vertical speed of a candidate pass is consistent with a possible scan.  Not the scan speed control tolerance; larger than that.
    scan_velocity_z_tolerance = 0.25
    maximum_n_missing = 10  # Varies with the number of heliostat corners.
    minimum_gps_pass_inter_point_speed = (
        4.0  # m/sec. Minimum observed inter-point speed allowable along a contiguous GPS scan pass.
    )
    minimum_gps_pass_number_of_points = 20  # Minmum number of points required to constitute a GPS pass.
    gps_pass_start_margin = (
        5  # Number of points to shrink the start of a GPS pass after filtering to remove low-velocity points.
    )
    gps_pass_stop_margin = (
        5  # Number of points to shrink the end of a GPS pass after filtering to remove low-velocity points.
    )
    # m.     Maximum distance between estimated camera trajectory points (expressed in heliostat coordiantes), to consider part of a connected trajectory.
    maximum_camera_pass_inter_point_distance = 4.0
    minimum_camera_pass_inter_point_speed = (
        1.5  # m/sec. Minimum observed inter-point speed allowable along a contiguous camera pass.
    )
    minimum_camera_pass_number_of_points = 10  # Minmum number of points required to constitute a camera pass.
    camera_pass_start_margin = 3  # Number of points to shrink the start of a camera pass after removing points corresponding to excess missing corners.
    camera_pass_stop_margin = 3  # Number of points to shrink the end of a camera pass after removing points corresponding to excess missing corners.
    # Input/output sources.
    specifications = Dspec.nsttf_specifications()  # Solar field parameters.
    aimpoint_xyz = [
        60.0,
        8.8,
        60.0,
    ]  # ?? SCAFFOLDING RCB -- READ THIS FROM THE 030_Aimpoints DIRECTORY.  SUPPORT MULTIPLE HELIOSTATS.
    #                                            year, month, day, hour, minute, second, zone]
    when_ymdhmsz = [
        2020,
        12,
        3,
        15,
        45,
        0,
        -7,
    ]  # Nominal time, refined by computation.  Recommend use mid-point of flight.
    up_heliostats = ["6W5", "6E6", "6E8", "6E9", "13E10"]
    up_configuration = hc.HeliostatConfiguration(az=np.deg2rad(180), el=np.deg2rad(90))
    down_heliostats = ["5W10", "5W8", "5E1", "5E7", "5E10", "6E7", "9W10", "10W12", "11W5", "13W5", "13W14", "13E14"]
    down_configuration = hc.NSTTF_stow()
    input_video_dir_body_ext = (
        experiment_dir() + "2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4"
    )
    input_flight_log_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/060_FlightData/data/_F08_log_2020-12-03_15-44-13_v2.csv"
    )
    input_reconstructed_heliostats_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/180_Heliostats3d/mavic_zoom/data/"
    )
    output_data_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/190_TrajectoryAnalysis/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/190_TrajectoryAnalysis/mavic_zoom/render/"
    )
    output_construction_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/190c_TrajectoryAnalysis/mavic_zoom/"
    )

    #     # Render control.
    #     render_control_projected_distorted         = rchr.default(color='m')  # ?? SCAFFOLDING RCB -- TEMPORARY
    #     render_control_projected_undistorted       = rchr.default(color='r')  # ?? SCAFFOLDING RCB -- TEMPORARY
    #     render_control_confirmed_distorted         = rchr.default(color='c')  # ?? SCAFFOLDING RCB -- TEMPORARY
    #     render_control_confirmed_undistorted       = rchr.default(color='b')  # ?? SCAFFOLDING RCB -- TEMPORARY

    key_frames_object = TrajectoryAnalysis(  # Execution control.
        log_dir_body_ext,
        velocity_calculation_offset_fwd_bwd,
        delta_velocity_angle_xy_peak_threshold,
        delta_velocity_angle_xy_non_peak_threshold,
        turn_overshoot_skip_time,
        scan_establish_velocity_time,
        scan_discard_velocity_time,
        minimum_scan_pass_time,
        nominal_scan_speed,
        scan_speed_tolerance,
        nominal_scan_velocity_z,
        scan_velocity_z_tolerance,
        maximum_n_missing,
        minimum_gps_pass_inter_point_speed,
        minimum_gps_pass_number_of_points,
        gps_pass_start_margin,
        gps_pass_stop_margin,
        maximum_camera_pass_inter_point_distance,
        minimum_camera_pass_inter_point_speed,
        minimum_camera_pass_number_of_points,
        camera_pass_start_margin,
        camera_pass_stop_margin,
        # Input/output sources.
        specifications,
        aimpoint_xyz,
        when_ymdhmsz,
        up_heliostats,
        up_configuration,
        down_heliostats,
        down_configuration,
        input_video_dir_body_ext,
        input_flight_log_dir_body_ext,
        input_reconstructed_heliostats_dir,
        output_data_dir,
        output_render_dir,
        output_construction_dir,
        #                                            # Render control.
        #                                            render_control_projected_distorted,
        #                                            render_control_projected_undistorted,
        #                                            render_control_confirmed_distorted,
        #                                            render_control_confirmed_undistorted,
    )
