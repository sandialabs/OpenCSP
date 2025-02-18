"""
A data structure for estimating a heliostat's 3d shape given an ideal heliostat model and 
a series of frame observations with associated tracked corners.



"""

import copy
import csv
from cv2 import cv2 as cv
import logging
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os

from numpy.lib.function_base import kaiser

import opencsp.common.lib.geometry.geometry_2d as g2d
import opencsp.common.lib.geometry.transform_3d as t3d
import opencsp.common.lib.render.image_plot as ip
import opencsp.common.lib.render.PlotAnnotation as pa
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.list_tools as lt
import opencsp.common.lib.tool.log_tools as logt
import opencsp.common.lib.tool.time_date_tools as tdt
import DEPRECATED_specifications as Dspec  # ?? SCAFFOLDING RCB -- TEMPORARY
import FrameNameXyList as fnxl
import HeliostatInfer2dFrame as hi2df
import ufacet_heliostat_3d_analysis as uh3a
import ufacet_pipeline_frame as upf
import DEPRECATED_utils as utils  # ?? SCAFFOLDING RCB -- TEMPORARY


class HeliostatInfer3d:
    """
    Class for estimating a heliostat 3d-shape for a series of frame observations.

    Input is a nominal heliostat model, and a series of frames with associated found corners.

    This class contains several quality metrics computed along the way.

    """

    def __init__(
        self,
        # Data.
        hel_name,
        list_of_frame_id_observed_corner_xy_lists,  # Already undistorted if distorted_or_undistorted_str == 'undistorted'
        flat_corner_xyz_list,  # A "flat heliostat" model, oriented with z parallel to the nominal heliostat optical axis.  # ?? SCAFFOLDING RCB -- TWO PARAMETERS, SAME IDEA.  MERGE.
        # For this calculation, z values should be coplanar, for coplanar sections of the heliostat.
        # For most heliostats this means z will be a single common value.  But for heliostats such
        # as NSTTF which have a step in the middle, there will be two z values, with one z value
        # corresponding to rows 1, 2, 4, and 5, with a second z value corresponding to row 3.
        # Execution control.
        camera_matrix,  # Intrinsic camera model.
        distortion_coefficients,  # Lens distortion model.
        zero_distortion_coefficients,  # Distortion model for an ideal pinhole lens with no distortion.
        # Input/output sources.
        specifications,  # Solar field parameters.
        theoretical_flat_heliostat_dir_body_ext,  # File containing list of xyz facet corners for the ideal flat heliostat model.
        theoretical_flat_heliostat_dict,  # Ideal flat heliostat model.  Dictionary of form {facet_idx: corner_xyz_list}.
        theoretical_flat_heliostat_xyz_list,  # Ideal flat heliostat model.  Form [corner_1_xyz, corner_2_xyz, ...].  # ?? SCAFFOLDING RCB -- TWO PARAMETERS, SAME IDEA.  MERGE.
        input_video_body,  # Filename of video we are analyszing.  For output filename construction.
        input_frame_dir,  # Where to find original video frames.
        input_frame_id_format,  # Format that embeds frame numbers in frame filenames.
        projected_or_confirmed_str,  # Whether the input xy points are projected points or confirmed points.
        distorted_or_undistorted_str,  # Whether the input xy points have been corrected for lens distortion.
        output_data_dir,  # Directory to write files describing ideal reference model and measurement result.
        output_construct_corners_3d_dir,  # Directory to write files describing intermediate construction steps.
        # Render control.
        render_control,
    ):  # Flags to control rendering; e.g., whether to output intermediate construction steps.
        # Start progress log.
        self.search_log = []
        msg_line = tdt.current_time_string() + " " + str(hel_name) + " starting 3d inference..."
        self.search_log.append(msg_line)
        print("\n" + msg_line)

        # Data.
        self.hel_name = hel_name
        self.list_of_frame_id_observed_corner_xy_lists = list_of_frame_id_observed_corner_xy_lists
        self.flat_corner_xyz_list = flat_corner_xyz_list
        self.n_corners = len(self.flat_corner_xyz_list)
        # Execution control.
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.zero_distortion_coefficients = zero_distortion_coefficients
        # Input/output sources.
        self.specifications = specifications
        self.theoretical_flat_heliostat_dir_body_ext = theoretical_flat_heliostat_dir_body_ext
        self.theoretical_flat_heliostat_dict = theoretical_flat_heliostat_dict
        self.theoretical_flat_heliostat_xyz_list = theoretical_flat_heliostat_xyz_list
        self.theoretical_flat_heliostat_spec = (
            self.specifications.construct_flat_heliostat_spec()
        )  # ?? SCAFFOLDING RCB -- THIS IS CONSTRUCTED HERE INSTEAD OF WITH THE OTHER THEORETICAL FLAT ITEMS.  POSSIBLE INCONSISTENCY.  RESOLVE THIS.
        self.input_video_body = input_video_body
        self.input_frame_dir = input_frame_dir
        self.input_frame_id_format = input_frame_id_format
        self.projected_or_confirmed_str = projected_or_confirmed_str
        self.distorted_or_undistorted_str = distorted_or_undistorted_str
        self.output_data_dir = output_data_dir
        self.output_construct_corners_3d_dir = output_construct_corners_3d_dir
        # Render control.
        self.render_control = render_control

        # Execution control.   # ?? SCAFFOLDING RCB - MAKE THIS AN INPUT
        self.max_frames_to_process = 5000  # For this heliostat.  # ?? SCAFFOLDING RCB -- FIX VALUE
        self.max_missing_corners_to_allow = 35  # 0
        self.minimum_points_per_facet = 2
        self.n_steps_one_direction = 20
        self.n_planar_iterations = 50
        self.n_canting_iterations = 50
        self.variable_steps = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20]  # Search more coarsely where a larger step is needed.

        # Computation cmplexity measurement.
        self.calls_to_solve_pnp = 0
        self.calls_to_project_points = 0

        # Lookup the heliostat postiion.
        self.hel_xyz = self.specifications.heliostat_xyz(self.hel_name)

        # Determine the distortion model to apply, matching the input observed xy list.
        self.selected_distortion_model = self.select_distortion_model()

        # Construct an ideal version of this heliostat.
        msg_line = (
            tdt.current_time_string() + " " + str(hel_name) + " constructing ideal heliostat..."
        )  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        self.search_log.append(msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        print("\n" + msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        self.design_heliostat_spec = self.specifications.construct_design_heliostat_spec(self.hel_xyz)
        self.design_heliostat_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(
            self.design_heliostat_spec
        )
        (
            self.design_heliostat_spec,
            self.design_heliostat_xyz_list,
            self.design_heliostat_dir_body_ext,
            self.design_heliostat_spec_dir_body_ext,
        ) = self.save_and_analyze_design_heliostat(self.design_heliostat_spec)

        # Determine which frames to process.
        # (We assemble this list to make adjustment easy during debugging.)
        msg_line = (
            tdt.current_time_string() + " " + str(hel_name) + " analyzing frames..."
        )  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        self.search_log.append(msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        print("\n" + msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        self.dict_of_frame_dicts = {}
        for frame_id_xy_list in self.list_of_frame_id_observed_corner_xy_lists:
            frame_id = frame_id_xy_list[0]
            observed_corner_xy_list = frame_id_xy_list[1]
            n_missing = 0
            # Assemble lists of corresponding points.
            not_missing_mask = []
            correspond_observed_xy_list = []
            correspond_design_xyz_list = []
            for observed_xy, design_xyz in zip(observed_corner_xy_list, self.design_heliostat_xyz_list):
                if (observed_xy[0] != -1) or (observed_xy[1] != -1):  # ?? SCAFFOLDING RCB - MAKE A CALL TO A PREDICATE
                    # This is a valid point.
                    not_missing_mask.append(True)
                    correspond_design_xyz_list.append(design_xyz)
                    correspond_observed_xy_list.append(observed_xy)
                else:
                    not_missing_mask.append(False)
                    n_missing += 1
            # Compute camera pose, given only this frame.
            camera_rvec, camera_tvec = self.compute_camera_pose(correspond_design_xyz_list, correspond_observed_xy_list)
            # Collect results for this frame.
            frame_dict = {}  # ?? SCAFFOLDING RCB -- SHOULD THIS BE A CLASS?
            frame_dict["frame_id"] = frame_id
            frame_dict["all_design_xyz_list"] = self.design_heliostat_xyz_list
            frame_dict["all_observed_xy_list"] = observed_corner_xy_list
            frame_dict["not_missing_mask"] = not_missing_mask
            frame_dict["correspond_design_xyz_list"] = correspond_design_xyz_list
            frame_dict["correspond_observed_xy_list"] = correspond_observed_xy_list
            frame_dict["n_missing"] = n_missing
            frame_dict["use_for_metrology"] = n_missing <= self.max_missing_corners_to_allow
            frame_dict["single_frame_camera_rvec"] = camera_rvec
            frame_dict["single_frame_camera_tvec"] = camera_tvec
            # Add to results for all frames.
            self.dict_of_frame_dicts[frame_id] = frame_dict

        # Draw the camera positions.
        output_hel_dir = os.path.join(self.output_data_dir, self.hel_name)
        uh3a.plot_heliostat_with_camera_poses(
            hel_name,
            self.design_heliostat_dir_body_ext,
            specifications=self.specifications,
            dict_of_frame_dicts=self.dict_of_frame_dicts,
            saving_path=output_hel_dir,
            title_prefix=" ",
            explain=None,
            tracked_style=rcps.RenderControlPointSeq(
                linestyle="-", linewidth=0.3, color="m", marker="+", markersize=3  #'o',
            ),  # 0.4),
            processed_style=rcps.RenderControlPointSeq(
                linestyle="-", linewidth=0.5, marker="o", markersize=0.7, markeredgewidth=0.2, color="b"
            ),
        )

        # Report the reprojection error over all frames, assuming the ideal heliostat.
        design_error_n_pts = self.reprojection_error_all_frames_given_heliostat_spec(self.design_heliostat_spec)
        msg_line = (
            tdt.current_time_string()
            + " "
            + str(hel_name)
            + " overall error for "
            + str(hel_name)
            + " design_heliostat_spec = "
            + str(design_error_n_pts[0])
            + ",  n_points ="
            + str(design_error_n_pts[1])
        )  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        self.search_log.append(msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        print("\n" + msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.

        # # Draw frame images, for reference.
        # for frame_id in dt.sorted_keys(self.dict_of_frame_dicts):
        #     msg_line = tdt.current_time_string() + ' ' + str(hel_name) + ' drawing ' + str(frame_id) + ' annotated image...'  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        #     self.search_log.append(msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        #     print(msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        #     frame_dict = self.dict_of_frame_dicts[frame_id]
        #     observed_corner_xy_list = frame_dict['all_observed_xy_list']
        #     # Draw the frame with found corners, for later review.
        #     self.save_frame_track_image(frame_id, observed_corner_xy_list)

        # Find the 3-d heliostat which best fits the observed data.
        msg_line = (
            tdt.current_time_string()
            + " "
            + str(hel_name)
            + " adjusting design heliostat to minimize reprojection error..."
        )  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        self.search_log.append(msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        print("\n" + msg_line)  # ?? SCAFFOLDING RCB -- ENCAPSULATE THESE INTO A "LOG" MEMBER FUNCTION.
        self.search_heliostat_spec = copy.deepcopy(self.design_heliostat_spec)
        self.adjust_heliostat_spec_to_minimize_reprojection_error(
            self.search_heliostat_spec,  # Changed as a side effect.
            n_planar_iterations=self.n_planar_iterations,
            n_canting_iterations=self.n_canting_iterations,
            n_steps_one_direction=self.n_steps_one_direction,
            variable_steps=self.variable_steps,
        )

        # Save the projected points from the final step of the search.
        self.final_projected_points_dict = self.construct_final_projected_points_dict(self.search_heliostat_spec)

        # Save and render final result.
        self.save_and_analyze_final_heliostat_spec(
            self.search_heliostat_spec,
            self.search_log,
            self.search_overall_error_history,
            self.search_facet_error_history,
            self.final_projected_points_dict,
        )

    def select_distortion_model(self):
        if self.distorted_or_undistorted_str == "distorted":
            # The observed xy points are displaced by the effects of lens distortion.  So correct by the distortion model.
            return self.distortion_coefficients
        elif self.distorted_or_undistorted_str == "undistorted":
            # The observed xy points have alreay had distortion removed, so don't remove it again.
            return self.zero_distortion_coefficients
        else:
            print(
                'ERROR: In Heliostats3dInference.select_distortion_model(), unexpected self.distorted_or_undistorted_str="'
                + str(self.distorted_or_undistorted_str)
                + '" encountered.'
            )

    def adjust_heliostat_spec_to_minimize_reprojection_error(
        self, search_heliostat_spec, n_planar_iterations, n_canting_iterations, n_steps_one_direction, variable_steps
    ):
        # Save figures for initial state.
        self.save_and_analyze_heliostat_spec("SearchIter" + str(0), search_heliostat_spec)

        # Iterative search, allowing planar adjustment of facet positions (x,y,z,rot_z).
        self.search_overall_error_history = []
        self.search_facet_error_history = []
        n_facets = 25  # ?? SCAFFOLDING RCB -- MAKE GENERAL ACROSS DESIGNS
        facet_is_converged = [False] * n_facets
        for iteration in range(1, (n_planar_iterations + 1)):
            current_error_n_pts = self.reprojection_error_all_frames_given_heliostat_spec(search_heliostat_spec)
            current_overall_error = current_error_n_pts[0]
            current_n_points_sum = current_error_n_pts[1]
            self.search_overall_error_history.append([iteration, current_overall_error])
            msg_line = (
                tdt.current_time_string()
                + " "
                + str(self.hel_name)
                + " iteration {iteration:2d} overall error={overall_err:11.8f} n_points={n_points:d}".format(
                    iteration=iteration, overall_err=current_overall_error, n_points=current_n_points_sum
                )
            )
            self.search_log.append(msg_line)
            print("\n" + msg_line)
            previous_error = current_overall_error
            for facet_idx in range(0, n_facets):
                if not facet_is_converged[facet_idx]:
                    (best_var_name, best_min_error, best_min_error_del_var) = self.find_best_variable_xyz_rot_z(
                        search_heliostat_spec,
                        facet_to_adjust_idx=facet_idx,
                        n_steps_one_direction=n_steps_one_direction,
                        variable_steps=variable_steps,
                    )
                    if best_min_error_del_var == 0:
                        facet_is_converged[facet_idx] = True
                    search_heliostat_spec[facet_idx][best_var_name] += best_min_error_del_var
                    error_reduction = previous_error - best_min_error
                    previous_error = best_min_error
                    self.search_facet_error_history.append(
                        [iteration, facet_idx, best_var_name, best_min_error, best_min_error_del_var]
                    )
                    msg_line = (
                        tdt.current_time_string()
                        + " "
                        + str(self.hel_name)
                        + " iteration {iteration:2d} facet {facet_idx:2d} min error={min_err:11.8f} after adjust {var_name:8s} by {del_var:11.8f} reduced={err_reduction:11.7f}".format(
                            iteration=iteration,
                            facet_idx=facet_idx,
                            min_err=best_min_error,
                            var_name=best_var_name,
                            del_var=best_min_error_del_var,
                            err_reduction=error_reduction,
                        )
                    )
                    self.search_log.append(msg_line)
                    print(msg_line)
            if not (False in facet_is_converged):
                # Then all facets are converged.
                msg_line = (
                    tdt.current_time_string()
                    + " "
                    + str(self.hel_name)
                    + " iteration {iteration:2d}; all facets are converged (x,y,z,rot_z).".format(iteration=iteration)
                )
                self.search_log.append(msg_line)
                print("\n" + msg_line)
                break
        # Save figures for last planar iteration.
        self.save_and_analyze_heliostat_spec("SearchIter" + str(iteration), search_heliostat_spec)

        # Iterative search, allowing adjustment of canting angles (rot_x,rot_y).
        facet_is_converged = [False] * n_facets
        for iteration in range((n_planar_iterations + 1), (n_planar_iterations + n_canting_iterations + 1)):
            current_error_n_pts = self.reprojection_error_all_frames_given_heliostat_spec(search_heliostat_spec)
            current_overall_error = current_error_n_pts[0]
            current_n_points_sum = current_error_n_pts[1]
            self.search_overall_error_history.append([iteration, current_overall_error])
            msg_line = (
                tdt.current_time_string()
                + " "
                + str(self.hel_name)
                + " iteration {iteration:2d} overall error={overall_err:11.8f} n_points={n_points:d}".format(
                    iteration=iteration, overall_err=current_overall_error, n_points=current_n_points_sum
                )
            )
            self.search_log.append(msg_line)
            print("\n" + msg_line)
            previous_error = current_overall_error
            for facet_idx in range(0, n_facets):
                if not facet_is_converged[facet_idx]:
                    (best_var_name, best_min_error, best_min_error_del_var) = self.find_best_variable_rot_xy(
                        search_heliostat_spec,
                        facet_to_adjust_idx=facet_idx,
                        n_steps_one_direction=n_steps_one_direction,
                        variable_steps=variable_steps,
                    )
                    if best_min_error_del_var == 0:
                        facet_is_converged[facet_idx] = True
                    search_heliostat_spec[facet_idx][best_var_name] += best_min_error_del_var
                    error_reduction = previous_error - best_min_error
                    previous_error = best_min_error
                    self.search_facet_error_history.append(
                        [iteration, facet_idx, best_var_name, best_min_error, best_min_error_del_var]
                    )
                    msg_line = (
                        tdt.current_time_string()
                        + " "
                        + str(self.hel_name)
                        + " iteration {iteration:2d} facet {facet_idx:2d} min error={min_err:11.8f} after adjust {var_name:8s} by {del_var:8.5f} reduced={err_reduction:11.7f}".format(
                            iteration=iteration,
                            facet_idx=facet_idx,
                            min_err=best_min_error,
                            var_name=best_var_name,
                            del_var=best_min_error_del_var,
                            err_reduction=error_reduction,
                        )
                    )
                    self.search_log.append(msg_line)
                    print(msg_line)
            if not (False in facet_is_converged):
                # Then all facets are converged.
                msg_line = (
                    tdt.current_time_string()
                    + " "
                    + str(self.hel_name)
                    + " iteration {iteration:2d}; all facets are converged (rot_x,rot_y).".format(iteration=iteration)
                )
                self.search_log.append(msg_line)
                print("\n" + msg_line)
                break
        # Save figures for last canting iteration.
        self.save_and_analyze_heliostat_spec("SearchIter" + str(iteration), search_heliostat_spec)

    def find_best_variable_xyz_rot_z(
        self, search_heliostat_spec, facet_to_adjust_idx, n_steps_one_direction, variable_steps
    ):  # ?? SCAFFOLDING RCB -- EVALUATE WHETHER WE CAN REDUCE THE NUMBER OF DEEPCOPIES, OR REDUCE THE COMPLEXITY OF EACH DEEPCOPY.
        rot_z_min_err, min_err_del_rot_z = self.find_best_value(
            copy.deepcopy(search_heliostat_spec),
            facet_to_adjust_idx=facet_to_adjust_idx,
            var_name="rot_z",
            half_range=0.005,
            n_steps_one_direction=n_steps_one_direction,
            variable_steps=variable_steps,
        )
        c_x_min_err, min_err_del_c_x = self.find_best_value(
            copy.deepcopy(search_heliostat_spec),
            facet_to_adjust_idx=facet_to_adjust_idx,
            var_name="center_x",
            half_range=0.005,
            n_steps_one_direction=n_steps_one_direction,
            variable_steps=variable_steps,
        )
        c_y_min_err, min_err_del_c_y = self.find_best_value(
            copy.deepcopy(search_heliostat_spec),
            facet_to_adjust_idx=facet_to_adjust_idx,
            var_name="center_y",
            half_range=0.005,
            n_steps_one_direction=n_steps_one_direction,
            variable_steps=variable_steps,
        )
        # Find most productive error reduction variable.
        min_err = min(rot_z_min_err, c_x_min_err, c_y_min_err)
        if min_err == rot_z_min_err:
            return "rot_z", rot_z_min_err, min_err_del_rot_z
        elif min_err == c_x_min_err:
            return "center_x", c_x_min_err, min_err_del_c_x
        elif min_err == c_y_min_err:
            return "center_y", c_y_min_err, min_err_del_c_y
        else:
            print("ERROR: In Heliostats3dInference.find_best_variable(), unexpected situation encountered.")
            assert False

    def find_best_variable_rot_xy(
        self, search_heliostat_spec, facet_to_adjust_idx, n_steps_one_direction, variable_steps
    ):
        rot_x_min_err, min_err_del_rot_x = self.find_best_value(
            copy.deepcopy(search_heliostat_spec),
            facet_to_adjust_idx=facet_to_adjust_idx,
            var_name="rot_x",
            half_range=0.0005,
            n_steps_one_direction=n_steps_one_direction,
            variable_steps=variable_steps,
        )
        rot_y_min_err, min_err_del_rot_y = self.find_best_value(
            copy.deepcopy(search_heliostat_spec),
            facet_to_adjust_idx=facet_to_adjust_idx,
            var_name="rot_y",
            half_range=0.0005,
            n_steps_one_direction=n_steps_one_direction,
            variable_steps=variable_steps,
        )
        # Find most productive error reduction variable.
        min_err = min(rot_x_min_err, rot_y_min_err)
        if min_err == rot_x_min_err:
            return "rot_x", rot_x_min_err, min_err_del_rot_x
        elif min_err == rot_y_min_err:
            return "rot_y", rot_y_min_err, min_err_del_rot_y
        else:
            print("ERROR: In Heliostats3dInference.find_best_variable(), unexpected situation encountered.")
            assert False

    def find_best_value(
        self, search_heliostat_spec, facet_to_adjust_idx, var_name, half_range, n_steps_one_direction, variable_steps
    ):
        # Common values.
        original_var = search_heliostat_spec[facet_to_adjust_idx][var_name]
        var_step = half_range / n_steps_one_direction
        # Zero change case.
        # We test this first, because if nothing makes  a difference, we want to prescribe zero change.
        zero_change_del_var = 0.0
        zero_change_overall_error, zero_change_n_points_sum = self.find_best_value_aux(
            search_heliostat_spec, facet_to_adjust_idx, var_name, original_var, zero_change_del_var
        )
        # Initial error to beat.
        minimum_error = zero_change_overall_error
        minimum_error_del_var = zero_change_del_var
        for step in variable_steps:
            # Scan forward check.
            del_var = step * var_step
            overall_error, n_points_sum = self.find_best_value_aux(
                search_heliostat_spec, facet_to_adjust_idx, var_name, original_var, del_var
            )
            if overall_error < minimum_error:
                minimum_error = overall_error
                minimum_error_del_var = del_var
            # Scan backward check.
            del_var = -del_var
            overall_error, n_points_sum = self.find_best_value_aux(
                search_heliostat_spec, facet_to_adjust_idx, var_name, original_var, del_var
            )
            if overall_error < minimum_error:
                minimum_error = overall_error
                minimum_error_del_var = del_var
        # Return.
        return minimum_error, minimum_error_del_var

    def find_best_value_aux(self, search_heliostat_spec, facet_to_adjust_idx, var_name, original_var, del_var):
        var = original_var + del_var
        search_heliostat_spec[facet_to_adjust_idx][var_name] = var
        test_error_n_pts_by_facet = self.reprojection_error_single_facet_all_frames_given_heliostat_spec(
            search_heliostat_spec, facet_to_adjust_idx
        )
        overall_error = test_error_n_pts_by_facet[0]
        n_points_sum = test_error_n_pts_by_facet[1]
        return overall_error, n_points_sum

    def reprojection_error_single_facet_all_frames_given_heliostat_spec(self, heliostat_spec, facet_to_adjust_idx):
        heliostat_corner_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(heliostat_spec)
        return self.reprojection_error_single_facet_all_frames_given_corner_xyz_list(
            heliostat_corner_xyz_list, facet_to_adjust_idx
        )

    def reprojection_error_single_facet_all_frames_given_corner_xyz_list(
        self, heliostat_corner_xyz_list, facet_to_adjust_idx
    ):
        # Determine the error for each frame.
        reprojection_error_n_pts_per_frame_list = []
        for frame_id in dt.sorted_keys(self.dict_of_frame_dicts):
            frame_dict = self.dict_of_frame_dicts[frame_id]
            if frame_dict["use_for_metrology"]:
                reprojection_error_n_pts = self.reprojection_error_single_facet_one_frame_given_corner_xyz_list(
                    heliostat_corner_xyz_list, facet_to_adjust_idx, frame_id, frame_dict
                )
                facet_n_pts = reprojection_error_n_pts[1]
                if facet_n_pts >= self.minimum_points_per_facet:
                    reprojection_error_n_pts_per_frame_list.append(reprojection_error_n_pts)
        # Combine the errors.
        distance_sum = 0
        n_points_sum = 0
        for reprojection_error_n_pts in reprojection_error_n_pts_per_frame_list:
            error = reprojection_error_n_pts[0]
            n_points = reprojection_error_n_pts[1]
            this_distance_sum = error * n_points
            distance_sum += this_distance_sum
            n_points_sum += n_points
        # Compute the average error across all frames.
        overall_error = distance_sum / n_points_sum
        # Return.
        return [overall_error, n_points_sum]

    def reprojection_error_single_facet_one_frame_given_corner_xyz_list(
        self, current_all_heliostat_xyz_list, facet_to_adjust_idx, frame_id, frame_dict
    ):
        # Fetch data.
        all_observed_xy_list = frame_dict["all_observed_xy_list"]

        # Facet-specific points.
        hel_ul = current_all_heliostat_xyz_list[facet_to_adjust_idx * 4]
        hel_ur = current_all_heliostat_xyz_list[(facet_to_adjust_idx * 4) + 1]
        hel_lr = current_all_heliostat_xyz_list[(facet_to_adjust_idx * 4) + 2]
        hel_ll = current_all_heliostat_xyz_list[(facet_to_adjust_idx * 4) + 3]
        hel_facet_xyz_list = [hel_ul, hel_ur, hel_lr, hel_ll]

        obs_ul = all_observed_xy_list[facet_to_adjust_idx * 4]
        obs_ur = all_observed_xy_list[(facet_to_adjust_idx * 4) + 1]
        obs_lr = all_observed_xy_list[(facet_to_adjust_idx * 4) + 2]
        obs_ll = all_observed_xy_list[(facet_to_adjust_idx * 4) + 3]

        # Project the 3-d heliostat corner points into the camera plane using the camera pose associated with the frame.
        proj_facet_pts_reshaped = self.projected_points_given_facet_xyz_list(frame_dict, hel_facet_xyz_list)

        # Compare the projected point locations against the observed point locations, accumulating error measures.
        n_points = 0
        overall_error_sum = 0.0
        # Fetch facet corner points; some may be missing.
        proj_ul = proj_facet_pts_reshaped[0]
        proj_ur = proj_facet_pts_reshaped[1]
        proj_lr = proj_facet_pts_reshaped[2]
        proj_ll = proj_facet_pts_reshaped[3]
        # Update errors.
        n_points, overall_error_sum = self.update_facet_corner_error(obs_ul, proj_ul, n_points, overall_error_sum)
        n_points, overall_error_sum = self.update_facet_corner_error(obs_ur, proj_ur, n_points, overall_error_sum)
        n_points, overall_error_sum = self.update_facet_corner_error(obs_lr, proj_lr, n_points, overall_error_sum)
        n_points, overall_error_sum = self.update_facet_corner_error(obs_ll, proj_ll, n_points, overall_error_sum)
        # Normalize overall error.
        if n_points == 0:
            error = -999999.0  # ?? SCAFFOLDING RCB -- TIE THIS TO A GLOBAL CONSTANT.
        else:
            error = overall_error_sum / n_points  # ?? SCAFFOLDING RCB -- ELIMINATE THIS NORMALIZATION?
        # Return.
        return [error, n_points]

    def projected_points_given_facet_xyz_list(self, frame_dict, facet_xyz_list):
        # Fetch data.
        camera_rvec = frame_dict["single_frame_camera_rvec"]
        camera_tvec = frame_dict["single_frame_camera_tvec"]
        # Using the camera pose, transform the 3-d points into the camera image space.
        proj_pts, jacobian = cv.projectPoints(
            np.array(facet_xyz_list), camera_rvec, camera_tvec, self.camera_matrix, self.select_distortion_model()
        )  # ?? SCAFFOLDING RCB -- CORRECT?
        self.calls_to_project_points += 1
        proj_pts_reshaped = proj_pts.reshape(-1, 2)
        # Return.
        return proj_pts_reshaped

    def update_facet_corner_error(self, obs_xy, proj_xy, n_points, overall_error_sum):
        if not self.xy_is_missing(obs_xy):
            xy_error = self.xy_error(obs_xy, proj_xy)
            return n_points + 1, overall_error_sum + xy_error
        else:
            return n_points, overall_error_sum

    def xy_is_missing(self, xy):
        return (xy[0] == -1) or (xy[1] == -1)  # ?? SCAFFOLDING RCB -- DEFINE GLOBAL MISSING FLAG, USE THROUGHOUT

    def xy_error(self, xy1, xy2):
        return np.sqrt((xy2[0] - xy1[0]) ** 2 + (xy2[1] - xy1[1]) ** 2)

    def reprojection_error_all_frames_given_heliostat_spec(self, heliostat_spec):
        heliostat_corner_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(heliostat_spec)
        return self.reprojection_error_all_frames_given_corner_xyz_list(heliostat_corner_xyz_list)

    def reprojection_error_all_frames_given_corner_xyz_list(self, heliostat_corner_xyz_list):
        # Determine the error for each frame.
        reprojection_error_n_pts_by_facet_per_frame_list = []
        for frame_id in dt.sorted_keys(self.dict_of_frame_dicts):
            frame_dict = self.dict_of_frame_dicts[frame_id]
            if frame_dict["use_for_metrology"]:
                reprojection_error_n_pts_by_facet = self.reprojection_error_one_frame_given_corner_xyz_list(
                    heliostat_corner_xyz_list, frame_id, frame_dict
                )
                reprojection_error_n_pts_by_facet_per_frame_list.append(reprojection_error_n_pts_by_facet)
        # Combine the errors.
        distance_sum = 0
        n_points_sum = 0
        by_facet_sum = np.zeros(25)
        for reprojection_error_n_pts_by_facet in reprojection_error_n_pts_by_facet_per_frame_list:
            error = reprojection_error_n_pts_by_facet[0]
            n_points = reprojection_error_n_pts_by_facet[1]
            by_facet = reprojection_error_n_pts_by_facet[2]
            this_distance_sum = error * n_points
            distance_sum += this_distance_sum
            n_points_sum += n_points
            by_facet_sum += np.array(by_facet)
        # Compute the average error across all frames.
        overall_error = distance_sum / n_points_sum
        by_facet_error = by_facet_sum / n_points_sum
        # Return.
        return [overall_error, n_points_sum, by_facet_error]

    def reprojection_error_one_frame_given_corner_xyz_list(self, current_all_heliostat_xyz_list, frame_id, frame_dict):
        # Fetch data.
        all_observed_xy_list = frame_dict["all_observed_xy_list"]

        # Project the 3-d heliostat corner points into the camera plane using the camera pose associated with the frame.
        proj_pts_reshaped = self.projected_points_given_corner_xyz_list(frame_dict, current_all_heliostat_xyz_list)

        # Compare the projected point locations against the observed point locations, accumulating error measures.
        n_points = 0
        overall_error_sum = 0.0
        error_per_facet_list = []
        for idx in range(0, 25):  # ?? SCAFFOLDING RCB -- MAKE GENERAL ACROSS DESIGNS
            facet_error_sum = 0
            # Fetch facet corner points; some may be missing.
            obs_ul = all_observed_xy_list[idx * 4]
            obs_ur = all_observed_xy_list[(idx * 4) + 1]
            obs_lr = all_observed_xy_list[(idx * 4) + 2]
            obs_ll = all_observed_xy_list[(idx * 4) + 3]
            proj_ul = proj_pts_reshaped[idx * 4]
            proj_ur = proj_pts_reshaped[(idx * 4) + 1]
            proj_lr = proj_pts_reshaped[(idx * 4) + 2]
            proj_ll = proj_pts_reshaped[(idx * 4) + 3]
            # Update errors.
            n_points, overall_error_sum, facet_error_sum = self.update_corner_error(
                obs_ul, proj_ul, n_points, overall_error_sum, facet_error_sum
            )
            n_points, overall_error_sum, facet_error_sum = self.update_corner_error(
                obs_ur, proj_ur, n_points, overall_error_sum, facet_error_sum
            )
            n_points, overall_error_sum, facet_error_sum = self.update_corner_error(
                obs_lr, proj_lr, n_points, overall_error_sum, facet_error_sum
            )
            n_points, overall_error_sum, facet_error_sum = self.update_corner_error(
                obs_ll, proj_ll, n_points, overall_error_sum, facet_error_sum
            )
            # Store this facet's error.
            error_per_facet_list.append(facet_error_sum)
        # Normalize overall error.
        error = overall_error_sum / n_points  # ?? SCAFFOLDING RCB -- ELIMINATE THIS NORMALIZATION?
        # Return.
        return [error, n_points, error_per_facet_list]

    def projected_points_given_corner_xyz_list(self, frame_dict, correspond_heliostat_xyz_list):
        # Fetch data.
        camera_rvec = frame_dict["single_frame_camera_rvec"]
        camera_tvec = frame_dict["single_frame_camera_tvec"]
        # Using the camera pose, transform the 3-d points into the camera image space.
        proj_pts, jacobian = cv.projectPoints(
            np.array(correspond_heliostat_xyz_list),
            camera_rvec,
            camera_tvec,
            self.camera_matrix,
            self.select_distortion_model(),
        )  # ?? SCAFFOLDING RCB -- CORRECT?
        self.calls_to_project_points += 1
        proj_pts_reshaped = proj_pts.reshape(-1, 2)
        # Return.
        return proj_pts_reshaped

    def update_corner_error(self, obs_xy, proj_xy, n_points, overall_error_sum, facet_error_sum):
        if not self.xy_is_missing(obs_xy):
            xy_error = self.xy_error(obs_xy, proj_xy)
            return (n_points + 1, overall_error_sum + xy_error, facet_error_sum + xy_error)
        else:
            return n_points, overall_error_sum, facet_error_sum

    def compute_camera_pose(self, reference_xyz_list, observed_xy_list):
        """
        Input reference_xyz_list and observed_xy_list must have a 1-to-1 correspondence.
        """
        # Find camera position.
        retval, camera_rvec, camera_tvec = cv.solvePnP(
            np.array(reference_xyz_list), np.array(observed_xy_list), self.camera_matrix, self.select_distortion_model()
        )
        self.calls_to_solve_pnp += 1
        return camera_rvec, camera_tvec

    def construct_frame_camera_pose_dict(self, list_of_frame_id_observed_corner_xy_lists, heliostat_spec):
        # Construct (x,y,z) corner list.
        search_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(heliostat_spec)
        # Assemble camera pose dict.
        frame_camera_pose_dict = {}
        for frame_id_xy_list in list_of_frame_id_observed_corner_xy_lists:
            frame_id = frame_id_xy_list[0]
            observed_corner_xy_list = frame_id_xy_list[1]
            # Find camera position.
            retval, camera_rvec, camera_tvec = cv.solvePnP(
                np.array(search_xyz_list),
                np.array(observed_corner_xy_list),
                self.camera_matrix,
                self.select_distortion_model(),
            )
            self.calls_to_solve_pnp += 1
            # Store.
            frame_camera_pose_dict[frame_id] = [camera_rvec, camera_tvec]
        # Return.
        return frame_camera_pose_dict

    def construct_final_projected_points_dict(self, final_heliostat_spec):
        # Construct (x,y,z) list.
        final_heliostat_corner_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(
            final_heliostat_spec
        )
        # Collect projected points for each frame.
        final_projected_pts_dict = {}
        for frame_id in dt.sorted_keys(self.dict_of_frame_dicts):
            frame_dict = self.dict_of_frame_dicts[frame_id]
            proj_pts_reshaped = self.projected_points_given_corner_xyz_list(frame_dict, final_heliostat_corner_xyz_list)
            proj_pts_list = proj_pts_reshaped.tolist()
            final_projected_pts_dict[frame_id] = proj_pts_list
        # Return.
        return final_projected_pts_dict

    def save_and_analyze_heliostat_spec(self, hel_name_suffix, heliostat_spec):
        # Construct (x,y,z) list.
        heliostat_corner_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(heliostat_spec)
        # Save.
        print("In save_and_analyze_heliostat_spec(), heliostat_spec:")  # ?? SCAFFOLDING RCB -- WRITE TO DISK INSTEAD.
        print(heliostat_spec)  # ?? SCAFFOLDING RCB -- WRITE TO DISK INSTEAD.
        output_hel_name = self.hel_name + hel_name_suffix
        output_hel_dir = os.path.join(self.output_construct_corners_3d_dir, output_hel_name)
        output_hel_dir_body_ext = uh3a.save_heliostat_3d(output_hel_name, heliostat_corner_xyz_list, output_hel_dir)
        output_hel_spec_dir_body_ext = save_heliostat_spec(heliostat_spec, output_hel_name, output_hel_dir)
        # Analyze.
        camera_rvec = np.array([-2.68359887, -0.2037837, 0.215282]).reshape(
            3, 1
        )  # See 180_Heliostats3d.save_and_analyze_flat_heliostat() for background.
        camera_tvec = np.array([5.13947086, -2.22502302, 25.35294025]).reshape(3, 1)  #
        uh3a.analyze_and_render_heliostat_3d(
            self.theoretical_flat_heliostat_dir_body_ext,
            output_hel_dir_body_ext,
            None,
            self.specifications,
            output_hel_dir,
            camera_rvec=camera_rvec,
            camera_tvec=camera_tvec,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.zero_distortion_coefficients,
        )
        # Return.
        return output_hel_dir_body_ext

    def save_and_analyze_design_heliostat(self, design_spec):
        # Construct corner (x,y,z)) list.
        design_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(design_spec)
        # Save.
        design_hel_name = self.hel_name + "design"
        output_design_dir = os.path.join(self.output_data_dir, "DesignHeliostats", design_hel_name)
        design_hel_xyz_dir_body_ext = uh3a.save_heliostat_3d(design_hel_name, design_xyz_list, output_design_dir)
        design_hel_spec_dir_body_ext = save_heliostat_spec(design_spec, design_hel_name, output_design_dir)
        # Analyze.
        camera_rvec = np.array([-2.68359887, -0.2037837, 0.215282]).reshape(
            3, 1
        )  # See 180_Heliostats3d.save_and_analyze_flat_heliostat() for background.
        camera_tvec = np.array([5.13947086, -2.22502302, 25.35294025]).reshape(3, 1)  #
        uh3a.analyze_and_render_heliostat_3d(
            self.theoretical_flat_heliostat_dir_body_ext,
            design_hel_xyz_dir_body_ext,
            None,
            self.specifications,
            output_design_dir,
            camera_rvec=camera_rvec,
            camera_tvec=camera_tvec,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.zero_distortion_coefficients,
        )
        # Return.
        return (design_spec, design_xyz_list, design_hel_xyz_dir_body_ext, design_hel_spec_dir_body_ext)

    def save_frame_track_image(self, frame_id, observed_corner_xy_list):
        frame_track_step_suffix = "frameTrack"
        step_hel_name = self.step_hel_name(frame_id, frame_track_step_suffix)
        if self.distorted_or_undistorted_str == "undistorted":
            note = "Depicted points are undistorted.  Expect misalignments with image content."
        else:
            note = None
        output_frame_tracking_image_dir = os.path.join(self.output_construct_corners_3d_dir, step_hel_name)
        ft.create_directories_if_necessary(output_frame_tracking_image_dir)
        uh3a.draw_annotated_frame_image(
            observed_corner_xy_list,
            self.input_video_body,
            self.input_frame_dir,
            self.input_frame_id_format,
            frame_id,
            step_hel_name,
            note,
            output_frame_tracking_image_dir,
        )

    def save_and_analyze_final_heliostat_spec(
        self,
        heliostat_spec,
        search_log,
        search_overall_error_history,
        search_facet_error_history,
        final_projected_points_dict,
    ):
        # Construct (x,y,z) list.
        heliostat_corner_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(heliostat_spec)
        # Save.
        output_hel_name = self.hel_name
        output_hel_dir = os.path.join(self.output_data_dir, output_hel_name)
        output_hel_xyz_dir_body_ext = uh3a.save_heliostat_3d(output_hel_name, heliostat_corner_xyz_list, output_hel_dir)
        output_hel_spec_dir_body_ext = save_heliostat_spec(heliostat_spec, output_hel_name, output_hel_dir)
        output_frame_dict_dir_body_ext = save_frame_dict_parameters(
            output_hel_name, self.dict_of_frame_dicts, output_hel_dir
        )
        output_log_dir_body_ext = save_search_log(search_log, output_hel_name, output_hel_dir)
        output_overall_history_dir_body_ext = save_search_overall_error_history(
            search_overall_error_history, output_hel_name, output_hel_dir
        )
        output_facet_history_dir_body_ext = save_search_facet_error_history(
            search_facet_error_history, output_hel_name, output_hel_dir
        )
        output_complexity_dir_body_ext = save_computation_complexity(
            self.calls_to_solve_pnp, self.calls_to_project_points, output_hel_name, output_hel_dir
        )
        output_proj_pts_dir_body_ext = save_projected_points_dict(
            final_projected_points_dict, output_hel_name, output_hel_dir
        )
        # Analyze.
        camera_rvec = np.array([-2.68359887, -0.2037837, 0.215282]).reshape(
            3, 1
        )  # See 180_Heliostats3d.save_and_analyze_flat_heliostat() for background.
        camera_tvec = np.array([5.13947086, -2.22502302, 25.35294025]).reshape(3, 1)  #
        uh3a.analyze_and_render_heliostat_3d(
            self.theoretical_flat_heliostat_dir_body_ext,
            output_hel_xyz_dir_body_ext,
            None,
            self.specifications,
            output_hel_dir,
            camera_rvec=camera_rvec,
            camera_tvec=camera_tvec,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.zero_distortion_coefficients,
        )
        # Return.
        return (
            output_hel_xyz_dir_body_ext,
            output_hel_spec_dir_body_ext,
            output_log_dir_body_ext,
            output_overall_history_dir_body_ext,
            output_facet_history_dir_body_ext,
            output_complexity_dir_body_ext,
            output_proj_pts_dir_body_ext,
        )

    # ?? SCAFFOLDING RCB -- MOVE SOMEWHERE ELSE.
    def print_facets_canting_angles(
        self, heliostat_corner_xyz_list
    ):  # ?? SCAFFOLDING RCB -- MAKE GENERAL ACROSS HELIOSTAT DESIGNS
        for idx in range(0, 25):
            start_facet_idx = idx * 4
            facet_corner_xyz_list = heliostat_corner_xyz_list[start_facet_idx : (start_facet_idx + 4)]
            self.print_facets_canting_angles_aux(idx, facet_corner_xyz_list)

    # ?? SCAFFOLDING RCB -- MOVE SOMEWHERE ELSE.
    def print_facets_canting_angles_aux(
        self, idx, facet_corner_xyz_list
    ):  # ?? SCAFFOLDING RCB -- MAKE GENERAL ACROSS HELIOSTAT DESIGNS
        facet_ul_xyz = facet_corner_xyz_list[0]
        facet_ur_xyz = facet_corner_xyz_list[1]
        facet_lr_xyz = facet_corner_xyz_list[2]
        facet_ll_xyz = facet_corner_xyz_list[3]
        # Diagonal vectors.
        ll_to_ur = np.array(
            [
                (facet_ur_xyz[0] - facet_ll_xyz[0]),
                (facet_ur_xyz[1] - facet_ll_xyz[1]),
                (facet_ur_xyz[2] - facet_ll_xyz[2]),
            ]
        )
        lr_to_ul = np.array(
            [
                (facet_ul_xyz[0] - facet_lr_xyz[0]),
                (facet_ul_xyz[1] - facet_lr_xyz[1]),
                (facet_ul_xyz[2] - facet_lr_xyz[2]),
            ]
        )
        # Unit diagonals.
        u_ll_to_ur = ll_to_ur / np.linalg.norm(ll_to_ur)  # ?? SCAFFOLDING RCB -- MAKE THIS A SUBROUTINE
        u_lr_to_ul = lr_to_ul / np.linalg.norm(lr_to_ul)  # ?? SCAFFOLDING RCB -- MAKE THIS A SUBROUTINE
        cross_product = np.cross(u_ll_to_ur, u_lr_to_ul)
        norm_of_cross = np.linalg.norm(cross_product)
        diagonal_angle = np.arcsin(norm_of_cross)
        normalized_cross_product = cross_product / norm_of_cross
        n_x = normalized_cross_product[0]
        n_y = normalized_cross_product[1]
        n_z = normalized_cross_product[2]
        canting_angle_in_x_direction = math.atan2(n_x, n_z)
        canting_angle_in_y_direction = math.atan2(n_y, n_z)
        # print('In print_facets_canting_angles_aux(), facet ul, ur, lr, ll:', facet_ul_xyz, facet_ur_xyz, facet_lr_xyz, facet_ll_xyz)
        # print('In print_facets_canting_angles_aux(), ll_to_ur:', ll_to_ur[0], ll_to_ur[1], ll_to_ur[2])
        # print('In print_facets_canting_angles_aux(), lr_to_ul:', lr_to_ul[0], lr_to_ul[1], lr_to_ul[2])
        # print('In print_facets_canting_angles_aux(), cross_product:', cross_product)
        # print('In print_facets_canting_angles_aux(), norm_of_cross=', norm_of_cross)
        # print('In print_facets_canting_angles_aux(), diagonal_angle(deg)=', np.degrees(diagonal_angle))
        # print('In print_facets_canting_angles_aux(), canting_angle_in_x_direction =', canting_angle_in_x_direction)
        # print('In print_facets_canting_angles_aux(), canting_angle_in_y_direction =', canting_angle_in_y_direction)
        print(
            "In print_facets_canting_angles_aux(), idx = ",
            idx,
            "  cross_product:",
            cross_product,
            " canting_angle_in_x_direction =",
            canting_angle_in_x_direction,
            "  canting_angle_in_y_direction =",
            canting_angle_in_y_direction,
        )

    # ?? SCAFFOLDING RCB -- MOVE SOMEWHERE ELSE.
    def print_facet_canting_angles(
        self, facet_corner_xyz_list
    ):  # ?? SCAFFOLDING RCB -- MAKE GENERAL ACROSS HELIOSTAT DESIGNS
        facet_ul_xyz = facet_corner_xyz_list[0]
        facet_ur_xyz = facet_corner_xyz_list[1]
        facet_lr_xyz = facet_corner_xyz_list[2]
        facet_ll_xyz = facet_corner_xyz_list[3]
        # Diagonal vectors.
        ll_to_ur = np.array(
            [
                (facet_ur_xyz[0] - facet_ll_xyz[0]),
                (facet_ur_xyz[1] - facet_ll_xyz[1]),
                (facet_ur_xyz[2] - facet_ll_xyz[2]),
            ]
        )
        lr_to_ul = np.array(
            [
                (facet_ul_xyz[0] - facet_lr_xyz[0]),
                (facet_ul_xyz[1] - facet_lr_xyz[1]),
                (facet_ul_xyz[2] - facet_lr_xyz[2]),
            ]
        )
        # Unit diagonals.
        u_ll_to_ur = ll_to_ur / np.linalg.norm(ll_to_ur)  # ?? SCAFFOLDING RCB -- MAKE THIS A SUBROUTINE
        u_lr_to_ul = lr_to_ul / np.linalg.norm(lr_to_ul)  # ?? SCAFFOLDING RCB -- MAKE THIS A SUBROUTINE
        cross_product = np.cross(u_ll_to_ur, u_lr_to_ul)
        norm_of_cross = np.linalg.norm(cross_product)
        diagonal_angle = np.arcsin(norm_of_cross)
        normalized_cross_product = cross_product / norm_of_cross
        n_x = normalized_cross_product[0]
        n_y = normalized_cross_product[1]
        n_z = normalized_cross_product[2]
        canting_angle_in_x_direction = math.atan2(n_x, n_z)
        canting_angle_in_y_direction = math.atan2(n_y, n_z)
        print(
            "In print_facet_canting_angles(), facet ul, ur, lr, ll:",
            facet_ul_xyz,
            facet_ur_xyz,
            facet_lr_xyz,
            facet_ll_xyz,
        )
        print("In print_facet_canting_angles(), ll_to_ur:", ll_to_ur[0], ll_to_ur[1], ll_to_ur[2])
        print("In print_facet_canting_angles(), lr_to_ul:", lr_to_ul[0], lr_to_ul[1], lr_to_ul[2])
        print("In print_facet_canting_angles(), cross_product:", cross_product)
        print("In print_facet_canting_angles(), norm_of_cross=", norm_of_cross)
        print("In print_facet_canting_angles(), diagonal_angle(deg)=", np.degrees(diagonal_angle))
        print("In print_facet_canting_angles(), canting_angle_in_x_direction =", canting_angle_in_x_direction)
        print("In print_facet_canting_angles(), canting_angle_in_y_direction =", canting_angle_in_y_direction)

    # ACCESS

    # def sorted_hel_name_list(self):
    #     """
    #     Returns all hel_names, in ascending order.  You can apply subscripts.
    #     """
    #     hel_name_list = list( self.dictionary.keys())
    #     hel_name_list.sort()
    #     return hel_name_list

    # MODIFICATION

    # def add_list_of_frame_id_xy_lists(self, hel_name, input_list_of_frame_id_xy_lists):
    #     """
    #     Add a list of [frame_id, xy_list] pairs to the dictionary, under the given hel_name key.
    #     Assumes the hel_name is not already there.
    #     """
    #     if hel_name in self.dictionary:
    #         print('ERROR: In HeliostatInfer3d.add_list_of_frame_id_xy_lists(), attempt to add hel_name='+str(hel_name)+', which is already present.')
    #         assert False
    #     self.dictionary[hel_name] = input_list_of_frame_id_xy_lists

    # READ

    # def load(self, input_dir_body_ext):   # "nfxl" abbreviates "HeliostatInfer3d"
    #     """
    #     Reads the stored HeliostatInfer3d file, and adds it to the current dictionary.
    #     If data is already already present in this HeliostatInfer3d, extends the current
    #     content as follows:

    #       - If a heliostat is not already present, adds the heliostat, and the associated frame_id/xy_list pair.

    #       - If the loaded heliostat is present, but the frame_id of the loaded frame_id/xy_list pair is not
    #         already associated with the heliostat, then add the frame_id/xy_list pair to the heliostat entry.

    #       - If the loaded heliostat is present, and it already contains the frame_id with some associated
    #         (x,y) points, extend the existing xy_list for the frame_id by adding the newly loaded (x,y)
    #         points.  Do not check for duplicate points, simply append the loaded points to the points
    #         that are already there.  Also do not group the points -- thus if the points are supposed
    #         to represent a polygon, they are not combined geometrically, but rather by simply
    #         concatenating the point lists.
    #     """
    #     #print('In HeliostatInfer3d.load(), loading input file: ', input_dir_body_ext)
    #     # Check if the input file exists.
    #     if not ft.file_exists(input_dir_body_ext):
    #         raise OSError('In HeliostatInfer3d.load(), file does not exist: ' + str(input_dir_body_ext))
    #     # Open and read the file.
    #     with open(input_dir_body_ext, newline='') as input_stream:
    #         reader = csv.reader(input_stream, delimiter=',')
    #         for input_row in reader:
    #             self.add_row_to_dictionary(input_row)

    # WRITE

    def step_hel_name(
        self, frame_id, step_suffix
    ):  # String to add to heliostat name to denote this step result.  Cannot include "_" or "-" characters.
        return self.hel_name + "f" + upf.frame_id_str_given_frame_id(frame_id, self.input_frame_id_format) + step_suffix

    # RENDER

    # def print(self,
    #           max_keys=10,          # Maximum number of keys to print.  Elipsis after that.
    #           max_value_length=70,  # Maximum value length to print.  Elipsis after that.
    #           indent=None):         # Number of blankss to print at the beginning of each line.
    #     # Print.
    #     dt.print_dict(self.dictionary,
    #                   max_keys=max_keys,
    #                   max_value_length=max_value_length,
    #                   indent=indent)


# HELPER FUNCTIONS


def save_heliostat_spec(
    heliostat_spec, hel_name, output_dir
):  # ?? SCAFFOLDING RCB -- HELIOSTAT_SPEC SHOULD BE A CLASS, AND THIS SHOULD BE A CLASS MEMBER FUNCTION
    # Prepare output.
    heading_line = "facet_id,center_x,center_y,center_z,rot_x,rot_y,rot_z"
    data_lines = []
    for facet_id in dt.sorted_keys(heliostat_spec):
        facet_spec = heliostat_spec[facet_id]
        center_x = facet_spec["center_x"]
        center_y = facet_spec["center_y"]
        center_z = facet_spec["center_z"]
        rot_x = facet_spec["rot_x"]
        rot_y = facet_spec["rot_y"]
        rot_z = facet_spec["rot_z"]
        data_lines.append(
            "{0:d},{1:.6f},{2:.6f},{3:.6f},{4:.8f},{5:.8f},{6:.8f}".format(
                facet_id, center_x, center_y, center_z, rot_x, rot_y, rot_z
            )
        )
    # Write to disk.
    output_file_body = hel_name + " heliostat_spec"
    explain = output_file_body  # I don't feel like something fancier.
    print("In save_heliostat_spec(), saving heliostat_spec to file:", os.path.join(output_dir, output_file_body))
    output_dir_body_ext = ft.write_csv_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_file_body,  # Body of output filename; extension is ".csv"
        heading_line,  # First line to write to file. None to skip.
        data_lines,  # Subsequent lines to write to file.
        error_if_dir_not_exist=False,
    )  # If the directory does not exist, create it.
    # Return.
    return output_dir_body_ext


# ?? SCAFFOLDING RCB -- MOVE ALL OF THESE SAVE FUNCTIONS (OR MOST) INTO THE CLASS AS MENBER FUNCTIONS?


def save_frame_dict_parameters(
    hel_name, dict_of_frame_dicts, output_dir
):  # ?? SCAFFOLDING RCB -- FRAME_CAMERA_POSE_DICT SHOULD BE A CLASS, AND THIS SHOULD BE A CLASS MEMBER FUNCTION
    # Prepare output.
    heading_line = "frame_id,rvec_x,rvec_y,rvec_z,tvec_x,tvec_y,tvec_z,n_missing,use_for_metrology"
    data_lines = []
    for frame_id in dt.sorted_keys(dict_of_frame_dicts):
        # Fetch frame parameters.
        frame_dict = dict_of_frame_dicts[frame_id]
        camera_rvec = frame_dict["single_frame_camera_rvec"]
        camera_tvec = frame_dict["single_frame_camera_tvec"]
        n_missing = frame_dict["n_missing"]
        use_for_metrology = frame_dict["use_for_metrology"]
        # Fetch camera pose coordinates.
        r_x = camera_rvec[0][0]
        r_y = camera_rvec[1][0]
        r_z = camera_rvec[2][0]
        t_x = camera_tvec[0][0]
        t_y = camera_tvec[1][0]
        t_z = camera_tvec[2][0]
        data_lines.append(
            "{0:d},{1:.8f},{2:.8f},{3:.8f},{4:.8f},{5:.8f},{6:.8f},{7:d},{8:d}".format(
                frame_id, r_x, r_y, r_z, t_x, t_y, t_z, n_missing, use_for_metrology
            )
        )
    # Write to disk.
    output_file_body = hel_name + "_frame_dict_parameters"
    explain = output_file_body  # I don't feel like something fancier.
    print("In save_frame_dict_parameters(), saving camera poses to file:", os.path.join(output_dir, output_file_body))
    output_dir_body_ext = ft.write_csv_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_file_body,  # Body of output filename; extension is ".csv"
        heading_line,  # First line to write to file. None to skip.
        data_lines,  # Subsequent lines to write to file.
        error_if_dir_not_exist=False,
    )  # If the directory does not exist, create it.
    # Return.
    return output_dir_body_ext


def save_search_log(search_error_log, hel_name, output_dir):
    # Write to disk.
    output_file_body = hel_name + "_search_log"
    explain = output_file_body  # I don't feel like something fancier.
    output_dir_body_ext = ft.write_text_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_file_body,  # Body of output filename; extension is ".csv"
        search_error_log,  # List of strings to write, one per line.
        error_if_dir_not_exist=False,
    )  # If True, error if not exist.  If False, create dir if necessary.
    # Return.
    return output_dir_body_ext


def save_search_overall_error_history(search_overall_error_history, hel_name, output_dir):
    # Prepare output.
    heading_line = "iteration,overall_error"
    data_lines = []
    for fb in search_overall_error_history:
        iteration = fb[0]
        overall_error = fb[1]
        data_lines.append("{0:d},{1:.10f}".format(iteration, overall_error))
    # Write to disk.
    output_file_body = hel_name + "_search_overall_error_history"
    explain = output_file_body  # I don't feel like something fancier.
    output_dir_body_ext = ft.write_csv_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_file_body,  # Body of output filename; extension is ".csv"
        heading_line,  # First line to write to file. None to skip.
        data_lines,  # Subsequent lines to write to file.
        error_if_dir_not_exist=False,
    )  # If the directory does not exist, create it.
    # Return.
    return output_dir_body_ext


def save_search_facet_error_history(search_facet_error_history, hel_name, output_dir):
    # Prepare output.
    heading_line = "iteration,facet_idx,best_var_name,best_min_error,best_min_error_del_var"
    data_lines = []
    for fb in search_facet_error_history:
        iteration = fb[0]
        facet_idx = fb[1]
        best_var_name = fb[2]
        best_min_error = fb[3]
        best_min_error_del_var = fb[4]
        data_lines.append(
            "{0:d},{1:d},{2:s},{3:.10f},{4:.6f}".format(
                iteration, facet_idx, best_var_name, best_min_error, best_min_error_del_var
            )
        )
    # Write to disk.
    output_file_body = hel_name + "_search_facet_error_history"
    explain = output_file_body  # I don't feel like something fancier.
    output_dir_body_ext = ft.write_csv_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_file_body,  # Body of output filename; extension is ".csv"
        heading_line,  # First line to write to file. None to skip.
        data_lines,  # Subsequent lines to write to file.
        error_if_dir_not_exist=False,
    )  # If the directory does not exist, create it.
    # Return.
    return output_dir_body_ext


def save_computation_complexity(calls_to_solve_pnp, calls_to_project_points, hel_name, output_dir):
    # Assemble data.
    data_lines = []
    data_lines.append("calls_to_solve_pnp," + str(calls_to_solve_pnp))
    data_lines.append("calls_to_project_points," + str(calls_to_project_points))
    # Write to disk.
    output_file_body = hel_name + "_computation_complexity"
    explain = output_file_body  # I don't feel like something fancier.
    output_dir_body_ext = ft.write_csv_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_file_body,  # Body of output filename; extension is ".csv"
        None,  # First line to write to file. None to skip.
        data_lines,  # List of strings to write, one per line.
        error_if_dir_not_exist=False,
    )  # If True, error if not exist.  If False, create dir if necessary.
    # Return.
    return output_dir_body_ext


def save_projected_points_dict(projected_pts_dict, hel_name, output_dir):
    # Prepare output.
    data_lines = []
    for frame_id in dt.sorted_keys(projected_pts_dict):
        projected_pts_list = projected_pts_dict[frame_id]
        data_line = "{0:d},".format(frame_id)
        for xy in projected_pts_list:
            data_line += "{0:.8f},".format(xy[0])
            data_line += "{0:.8f},".format(xy[1])
        data_lines.append(data_line)
    # Write to disk.
    output_file_body = hel_name + "_final_reprojected_points"
    explain = output_file_body  # I don't feel like something fancier.
    print(
        "In save_projected_points_dict(), saving final reprojected points to file:",
        os.path.join(output_dir, output_file_body),
    )
    output_dir_body_ext = ft.write_csv_file(
        explain,  # Explanatory string to include in notification output.  None to skip.
        output_dir,  # Directory to write file.  See below if not exist.
        output_file_body,  # Body of output filename; extension is ".csv"
        None,  # First line to write to file. None to skip.
        data_lines,  # Subsequent lines to write to file.
        error_if_dir_not_exist=False,
    )  # If the directory does not exist, create it.
    # Return.
    return output_dir_body_ext
