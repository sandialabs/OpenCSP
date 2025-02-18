"""
Converting heliostat corner tracks into estimates of heliostat 3-d shape.  This includes removing camera distortion.



"""

import copy
import csv
from cv2 import cv2 as cv
import logging
from multiprocessing import Pool
import numpy as np
import os
import subprocess

import lib.DEPRECATED_specifications as Dspec  # ?? SCAFFOLDING RCB -- TEMPORARY

import opencsp.common.lib.tool.dict_tools as dt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.list_tools as lt
import lib.FrameNameXyList as fnxl
import lib.HeliostatInfer3d as hi3d
import opencsp.common.lib.tool.log_tools as logt
import lib.NameFrameXyList as nfxl
import opencsp.common.lib.render_control.RenderControlHeliostats3d as rchr
import opencsp.common.lib.render_control.RenderControlKeyFramesGivenManual as rckfgm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import lib.ufacet_heliostat_3d_analysis as uh3a
import lib.ufacet_pipeline_clear as upc
import lib.ufacet_pipeline_frame as upf
import lib.DEPRECATED_utils as utils  # ?? SCAFFOLDING RCB -- TEMPORARY
import opencsp.common.lib.render.video_manipulation as vm


class Heliostats3d:
    """
    Class for converting heliostat corner tracks into 3-d heliostat shapes.  This includes removing camera distortion.

    """

    def __init__(
        self,
        # Execution control.
        force_construction,  # Recompute even if results already exist.
        specific_frame_ids,  # List of frame_ids to process, e.g. [777, 1126, ...].  For all frames, set None.
        single_processor,  # Execute multi-processor steps with a single processor.
        log_dir_body_ext,  # File to add log messages to.  Only used when single_processor == False.
        # Input/output sources.
        specifications,  # Solar field parameters.
        theoretical_flat_heliostat_dir_body_ext,  # File containing list of xyz facet corners for the ideal flat heliostat model.
        input_video_dir_body_ext,  # Where to find the video file.
        input_frame_dir,  # Where to read full frames, for rendering.
        input_frame_id_format,  # Format that embeds frame numbers in frame filenames.
        input_heliostat_projected_tracks_dir_body_ext,  # Where read to the full video projected tracks file, encoding a FrameNameXyList object.
        input_heliostat_confirmed_tracks_dir_body_ext,  # Where read to the full video confirmed tracks file, encoding a FrameNameXyList object.
        output_data_dir,  # Where to save the resulting summary data.
        output_render_dir,  # Where to save the resulting plots showing final heliostat tracks.
        output_construction_dir,  # Where to save the detailed image processing step-by-step plots.
        # Render control.
        render_control_projected_distorted,  # Flags to control rendering on this run, for the projected, distorted data.
        render_control_projected_undistorted,  # Flags to control rendering on this run, for the projected, undistorted data.
        render_control_confirmed_distorted,  # Flags to control rendering on this run, for the confirmed, distorted data.
        render_control_confirmed_undistorted,
    ):  # Flags to control rendering on this run, for the confirmed, undistorted data.
        # Check input.
        if (input_video_dir_body_ext == None) or (len(input_video_dir_body_ext) == 0):
            raise ValueError("In Heliostats3d.__init__(), null input_video_dir_body_ext encountered.")
        if (output_data_dir == None) or (len(output_data_dir) == 0):
            raise ValueError("In Heliostats3d.__init__(), null output_data_dir encountered.")
        if (output_render_dir == None) or (len(output_render_dir) == 0):
            raise ValueError("In Heliostats3d.__init__(), null output_render_dir encountered.")

        # Parse input video path components.
        input_video_dir, input_video_body, input_video_ext = ft.path_components(input_video_dir_body_ext)

        # Store input.
        # Execution control.
        self.force_construction = force_construction
        self.specific_frame_ids = specific_frame_ids
        self.single_processor = single_processor
        self.log_dir_body_ext = log_dir_body_ext
        self.camera_matrix = utils.CameraMatrix  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
        self.distortion_coefficients = utils.DistCoefs  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
        self.zero_distortion_coefficients = utils.ZeroDistCoefs  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS
        # self.reconstruct_executable_path                   = '<home_dir>/Code/ufacet_code/Reconstruct/bin/reconstruct_main.out'  # ?? SCAFFOLDING RCB -- RE-EVALUATE THIS.  PASS IN?
        # Input/output sources.
        self.specifications = specifications
        self.theoretical_flat_heliostat_dir_body_ext = theoretical_flat_heliostat_dir_body_ext
        self.theoretical_flat_heliostat_dict = uh3a.read_txt_file_to_heliostat(
            self.theoretical_flat_heliostat_dir_body_ext, self.specifications
        )
        self.theoretical_flat_heliostat_xyz_list = uh3a.heliostat_xyz_list_given_dict(
            self.theoretical_flat_heliostat_dict
        )
        self.input_video_dir_body_ext = input_video_dir_body_ext
        self.input_video_dir = input_video_dir
        self.input_video_body = input_video_body
        self.input_video_ext = input_video_ext
        self.input_frame_dir = input_frame_dir
        self.input_frame_id_format = input_frame_id_format
        self.input_heliostat_projected_tracks_dir_body_ext = input_heliostat_projected_tracks_dir_body_ext
        self.input_heliostat_confirmed_tracks_dir_body_ext = input_heliostat_confirmed_tracks_dir_body_ext
        self.output_data_dir = output_data_dir
        self.output_render_dir = output_render_dir
        self.output_construction_dir = output_construction_dir
        # Render control.
        self.render_control_projected_distorted = render_control_projected_distorted
        self.render_control_projected_undistorted = render_control_projected_undistorted
        self.render_control_confirmed_distorted = render_control_confirmed_distorted
        self.render_control_confirmed_undistorted = render_control_confirmed_undistorted

        # Output heliostat 2-d corner trajectory directories.
        # # Projected.
        # self.output_heliostat_projected_distorted_corner_2d_trajectories_dir   = os.path.join(self.output_data_dir, 'projected_distorted_corner_2d_trajectories')
        # self.output_heliostat_projected_undistorted_corner_2d_trajectories_dir = os.path.join(self.output_data_dir, 'projected_undistorted_corner_2d_trajectories')
        # # Confirmed.
        # self.output_heliostat_confirmed_distorted_corner_2d_trajectories_dir   = os.path.join(self.output_data_dir, 'confirmed_distorted_corner_2d_trajectories')
        # self.output_heliostat_confirmed_undistorted_corner_2d_trajectories_dir = os.path.join(self.output_data_dir, 'confirmed_undistorted_corner_2d_trajectories')

        # Output ideal heliostat models and their analysis plots.
        self.output_ideal_model_dir = os.path.join(self.output_data_dir, "IdealSiteWide")
        # Flat.
        self.theoretical_flat_heliostat_dir_body_ext = self.save_and_analyze_flat_heliostat(
            self.specifications.heliostat_design_name
        )
        # Nearest.
        (self.nearest_smooth_dir_body_ext, self.nearest_design_dir_body_ext) = (
            self.construct_save_and_analyze_key_heliostats(self.specifications.heliostat_design_name, "Nearest")
        )
        # Farthest.
        (self.farthest_smooth_dir_body_ext, self.farthest_design_dir_body_ext) = (
            self.construct_save_and_analyze_key_heliostats(self.specifications.heliostat_design_name, "Farthest")
        )
        # Demonstration heliostat.
        self.demonstration_dir_body_ext = self.construct_save_and_analyze_demonstration_heliostat_corner_xyz_list()

        # Output heliostat 3-d corner directories.
        # Projected.
        self.output_heliostat_projected_distorted_corners_3d_dir = os.path.join(
            self.output_data_dir, "projected_distorted_corners_3d"
        )
        self.output_heliostat_projected_undistorted_corners_3d_dir = os.path.join(
            self.output_data_dir, "projected_undistorted_corners_3d"
        )
        # Confirmed.
        self.output_heliostat_confirmed_distorted_corners_3d_dir = os.path.join(
            self.output_data_dir, "confirmed_distorted_corners_3d"
        )
        self.output_heliostat_confirmed_undistorted_corners_3d_dir = os.path.join(
            self.output_data_dir, "confirmed_undistorted_corners_3d"
        )

        # Output heliostat 3-d cnstruction directories.
        # Projected.
        self.output_heliostat_projected_distorted_construct_corners_3d_dir = os.path.join(
            self.output_construction_dir, "projected_distorted_corners_3dc"
        )
        self.output_heliostat_projected_undistorted_construct_corners_3d_dir = os.path.join(
            self.output_construction_dir, "projected_undistorted_corners_3dc"
        )
        # Confirmed.
        self.output_heliostat_confirmed_distorted_construct_corners_3d_dir = os.path.join(
            self.output_construction_dir, "confirmed_distorted_corners_3dc"
        )
        self.output_heliostat_confirmed_undistorted_construct_corners_3d_dir = os.path.join(
            self.output_construction_dir, "confirmed_undistorted_corners_3dc"
        )

        # Load video tracks files.
        # # Projected.  # SCAFFOLDING RCB -- TEMPORARILY DISABLED. ****
        # print('In Heliostats3d.__init__(), reading heliostat projected tracks file: ', self.input_heliostat_projected_tracks_dir_body_ext)  # SCAFFOLDING RCB -- TEMPORARILY DISABLED. ****
        # self.heliostat_projected_tracks_nfxl = nfxl.NameFrameXyList()  # SCAFFOLDING RCB -- TEMPORARILY DISABLED. ****
        # self.heliostat_projected_tracks_nfxl.load(self.input_heliostat_projected_tracks_dir_body_ext)  # SCAFFOLDING RCB -- TEMPORARILY DISABLED. ****
        # # Confirm what was read.  # SCAFFOLDING RCB -- TEMPORARILY DISABLED. ****
        # print('In Heliostats3d.__init__(), heliostat projected tracks read:')  # SCAFFOLDING RCB -- TEMPORARILY DISABLED. ****
        # self.heliostat_projected_tracks_nfxl.print(max_keys=12, max_value_length=200, indent=4)  # SCAFFOLDING RCB -- TEMPORARILY DISABLED. ****
        # Confirmed.
        print(
            "In Heliostats3d.__init__(), reading heliostat confirmed tracks file: ",
            self.input_heliostat_confirmed_tracks_dir_body_ext,
        )
        self.heliostat_confirmed_tracks_nfxl = nfxl.NameFrameXyList()
        self.heliostat_confirmed_tracks_nfxl.load(self.input_heliostat_confirmed_tracks_dir_body_ext)
        # Confirm what was read.
        print("In Heliostats3d.__init__(), heliostat confirmed tracks read:")
        self.heliostat_confirmed_tracks_nfxl.print(max_keys=12, max_value_length=200, indent=4)

        # For each NameFrameXyList object, produce a 3-d heliostat shape estimate.
        self.construct_and_save_heliostat_corners_3d()

        # Load found tracks.
        # self.read_heliostat_corners_3d()   # ?? SCAFFOLDING RCB -- TEMPORARY

        # Render, if desired.
        # self.render()

    def save_and_analyze_flat_heliostat(self, heliostat_design_name):
        # Save.
        flat_name = heliostat_design_name + "Flat"
        flat_output_dir = os.path.join(self.output_ideal_model_dir, flat_name)
        flat_dir_body_ext = uh3a.save_heliostat_3d(flat_name, self.theoretical_flat_heliostat_xyz_list, flat_output_dir)
        # Analyze.
        # Viewpoint for projection analysis plots for ideal heliostats.
        #
        # For how to contruct rvec and tvec, see example 6 in: https://www.programcreek.com/python/example/89450/cv2.Rodrigues
        #    R_vec = np.array([float(lines[3]),float(lines[5]), float(lines[4])]).reshape(3, 1)
        #    T_vec = np.array([float(lines[0]),float(lines[1]), float(lines[2])]).reshape(3, 1)
        #
        # Values were obtained by running code for the December 2020 video, heliostat 11W7, first frame,
        # and capturing values from console output:
        #    In Heliostats3dInference.__init__(), camera_rvec = [[-2.68359887]
        #                                                        [-0.2037837 ]
        #                                                        [ 0.215282  ]]
        #    In Heliostats3dInference.__init__(), camera_tvec = [[ 5.13947086]
        #                                                        [-2.22502302]
        #                                                        [25.35294025]]
        #
        camera_rvec = np.array([-2.68359887, -0.2037837, 0.215282]).reshape(3, 1)
        camera_tvec = np.array([5.13947086, -2.22502302, 25.35294025]).reshape(3, 1)
        uh3a.analyze_and_render_heliostat_3d(
            flat_dir_body_ext,
            flat_dir_body_ext,
            None,
            self.specifications,
            flat_output_dir,
            camera_rvec=camera_rvec,
            camera_tvec=camera_tvec,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.zero_distortion_coefficients,
        )

        # Return.
        return flat_dir_body_ext

    def construct_save_and_analyze_key_heliostats(self, heliostat_design_name, key_name):
        # Smooth.
        # Facets are at z positions corresponding to an uninterrupted smooth paraboloid.
        # Construct.
        if key_name == "Nearest":
            focal_length = self.specifications.design_focal_length(self.specifications.nearest_heliostat_xyz())
        elif key_name == "Farthest":
            focal_length = self.specifications.design_focal_length(self.specifications.farthest_heliostat_xyz())
        else:
            print(
                'ERROR: In Heliostats3d.construct_save_and_analyze_key_heliostats(), unexpected key_name="'
                + str(key_name)
                + '" encountered (1).'
            )
            assert False
        smooth_xyz_list = self.specifications.smooth_heliostat_corner_xyz_list_given_focal_length(focal_length)
        # Save.
        smooth_name = heliostat_design_name + key_name + "Smooth"
        smooth_output_dir = os.path.join(self.output_ideal_model_dir, smooth_name)
        smooth_dir_body_ext = uh3a.save_heliostat_3d(smooth_name, smooth_xyz_list, smooth_output_dir)
        # Analyze.
        camera_rvec = np.array([-2.68359887, -0.2037837, 0.215282]).reshape(
            3, 1
        )  # See save_and_analyze_flat_heliostat() for background.
        camera_tvec = np.array([5.13947086, -2.22502302, 25.35294025]).reshape(3, 1)  #
        uh3a.analyze_and_render_heliostat_3d(
            self.theoretical_flat_heliostat_dir_body_ext,
            smooth_dir_body_ext,
            None,
            self.specifications,
            smooth_output_dir,
            camera_rvec=camera_rvec,
            camera_tvec=camera_tvec,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.zero_distortion_coefficients,
        )

        # Design.
        # Actual nominal design for the solar field.
        # Typically facets are at nominal flat z positions, but canted to achieve the desired focal length.
        # Construct.
        if key_name == "Nearest":
            hel_xyz = self.specifications.nearest_heliostat_xyz()
        elif key_name == "Farthest":
            hel_xyz = self.specifications.farthest_heliostat_xyz()
        else:
            print(
                'ERROR: In Heliostats3d.construct_save_and_analyze_key_heliostats(), unexpected key_name="'
                + str(key_name)
                + '" encountered (1).'
            )
            assert False
        design_xyz_list = self.specifications.design_heliostat_corner_xyz_list(hel_xyz)
        # Save.
        design_name = heliostat_design_name + key_name + "Design"
        design_output_dir = os.path.join(self.output_ideal_model_dir, design_name)
        design_dir_body_ext = uh3a.save_heliostat_3d(design_name, design_xyz_list, design_output_dir)
        # Analyze.
        camera_rvec = np.array([-2.68359887, -0.2037837, 0.215282]).reshape(
            3, 1
        )  # See save_and_analyze_flat_heliostat() for background.
        camera_tvec = np.array([5.13947086, -2.22502302, 25.35294025]).reshape(3, 1)  #
        uh3a.analyze_and_render_heliostat_3d(
            self.theoretical_flat_heliostat_dir_body_ext,
            design_dir_body_ext,
            None,
            self.specifications,
            design_output_dir,
            camera_rvec=camera_rvec,
            camera_tvec=camera_tvec,
            camera_matrix=self.camera_matrix,
            distortion_coefficients=self.zero_distortion_coefficients,
        )

        # Return.
        return smooth_dir_body_ext, design_dir_body_ext

    def construct_save_and_analyze_demonstration_heliostat_corner_xyz_list(self):
        # Construct.
        flat_heliostat_spec = self.specifications.construct_flat_heliostat_spec()
        demonstration_heliostat_spec = copy.deepcopy(flat_heliostat_spec)
        demonstration_heliostat_spec[0]["rot_x"] = 0.010  # np.radians(1.0)
        demonstration_heliostat_spec[4]["rot_y"] = 0.010  # np.radians(1.0)
        demonstration_heliostat_spec[10]["rot_z"] = np.radians(-15.0)
        demonstration_heliostat_spec[20]["rot_z"] = np.radians(-5.0)
        demonstration_heliostat_spec[24]["rot_z"] = np.radians(15.0)
        demonstration_heliostat_spec[19]["center_x"] += 0.5  # m
        demonstration_heliostat_spec[2]["center_y"] += 0.5  # m
        demonstration_heliostat_spec[4]["center_x"] += 0.25  # m
        demonstration_heliostat_spec[4]["center_y"] += 0.25  # m
        demonstration_heliostat_spec[22]["center_z"] += -0.0508  # m
        demonstration_heliostat_corner_xyz_list = self.specifications.heliostat_corner_xyz_list_given_heliostat_spec(
            demonstration_heliostat_spec
        )
        # Save.
        output_hel_name = "Demonstration"
        output_hel_dir = os.path.join(self.output_ideal_model_dir, output_hel_name)
        output_hel_dir_body_ext = uh3a.save_heliostat_3d(
            output_hel_name, demonstration_heliostat_corner_xyz_list, output_hel_dir
        )
        # Analyze.
        camera_rvec = np.array([-2.68359887, -0.2037837, 0.215282]).reshape(
            3, 1
        )  # See Heliostats3d.save_and_analyze_flat_heliostat() for background.
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

    # CONSTRUCT RESULT

    def construct_and_save_heliostat_corners_3d(self):
        # Check if corners_3d have already been found.
        if (
            self.force_construction
            or (not ft.directory_exists(self.output_data_dir))
            or ft.directory_is_empty(self.output_data_dir)
        ):
            #             # Projected.
            #             # ?? SCAFFOLDING RCB -- BELOW TEMPOARILY DISABLED.
            #             self.construct_and_save_heliostat_corners_3d_aux(self.heliostat_projected_tracks_nfxl, 'projected', 'distorted',
            # #self.output_heliostat_projected_distorted_corner_2d_trajectories_dir,
            #                                                              self.output_heliostat_projected_distorted_corners_3d_dir,
            #                                                              self.output_heliostat_projected_distorted_construct_corners_3d_dir,
            #                                                              self.render_control_projected_distorted)
            #             self.construct_and_save_heliostat_corners_3d_aux(self.heliostat_projected_tracks_nfxl, 'projected', 'undistorted',
            # #self.output_heliostat_projected_undistorted_corner_2d_trajectories_dir,
            #                                                              self.output_heliostat_projected_undistorted_corners_3d_dir,
            #                                                              self.output_heliostat_projected_undistorted_construct_corners_3d_dir,
            #                                                              self.render_control_projected_undistorted)
            #             # Confirmed.
            #             self.construct_and_save_heliostat_corners_3d_aux(self.heliostat_confirmed_tracks_nfxl, 'confirmed', 'distorted',
            # #self.output_heliostat_confirmed_distorted_corner_2d_trajectories_dir,
            #                                                              self.output_heliostat_confirmed_distorted_corners_3d_dir,
            #                                                              self.output_heliostat_confirmed_distorted_construct_corners_3d_dir,
            #                                                              self.render_control_confirmed_distorted)
            self.construct_and_save_heliostat_corners_3d_aux(
                self.heliostat_confirmed_tracks_nfxl,
                "confirmed",
                "undistorted",
                # self.output_heliostat_confirmed_undistorted_corner_2d_trajectories_dir,
                self.output_heliostat_confirmed_undistorted_corners_3d_dir,
                self.output_heliostat_confirmed_undistorted_construct_corners_3d_dir,
                self.render_control_confirmed_undistorted,
            )

    def construct_and_save_heliostat_corners_3d_aux(
        self,
        heliostat_tracks_nfxl,
        projected_or_confirmed_str,
        distorted_or_undistorted_str,
        # output_corner_2d_trajectories_dir,
        output_corners_3d_dir,
        output_construct_corners_3d_dir,
        render_control,
    ):
        print(
            "\nIn Heliostats3d.construct_and_save_heliostat_corners_3d_aux(), reconstructing "
            + projected_or_confirmed_str
            + " "
            + distorted_or_undistorted_str
            + " case..."
        )

        # Prepare output directories.
        # ft.create_directories_if_necessary(output_corner_2d_trajectories_dir)
        ft.create_directories_if_necessary(output_construct_corners_3d_dir)

        # Assemble problem specifications to pass to the execute_heliostat_3d_inference() function.
        list_of_infer_dicts = []
        for hel_name in heliostat_tracks_nfxl.sorted_hel_name_list():
            infer_dict = {}
            infer_dict["hel_name"] = hel_name
            infer_dict["heliostat_tracks_nfxl"] = heliostat_tracks_nfxl
            infer_dict["projected_or_confirmed_str"] = projected_or_confirmed_str
            infer_dict["distorted_or_undistorted_str"] = distorted_or_undistorted_str
            # infer_dict['output_corner_2d_trajectories_dir'] = output_corner_2d_trajectories_dir
            infer_dict["output_construct_corners_3d_dir"] = output_construct_corners_3d_dir
            infer_dict["render_control"] = render_control
            list_of_infer_dicts.append(infer_dict)

        # Call execute_heliostat_3d_inference() for each problem specification.
        if self.single_processor:
            print(
                "In Heliostats3d.construct_and_save_heliostat_corners_3d_aux(), starting heliostat 3-d inference (single processor)..."
            )
            list_of_result_hi3ds = []
            for infer_dict in list_of_infer_dicts:
                list_of_result_hi3ds.append(self.execute_heliostat_3d_inference(infer_dict))
        else:
            print(
                "In Heliostats3d.construct_and_save_heliostat_corners_3d_aux(), starting heliostat 3-d inference (multi-processor)..."
            )
            logger = logt.multiprocessing_logger(self.log_dir_body_ext, level=logging.INFO)
            logger.info("================================= Execution =================================")
            with Pool(25) as pool:
                list_of_result_hi3ds = pool.map(self.execute_heliostat_3d_inference, list_of_infer_dicts)

        print("In Heliostats3d.construct_and_save_heliostat_corners_3d_aux(), heliostat 3-d inference done.")

        # Save the result.
        for hi3d in list_of_result_hi3ds:
            if hi3d is not None:  # We might have skipped some heliostats, for example for debuggging.
                # Output file name.
                output_corners_3d_dir_body_ext = uh3a.corners_3d_dir_body_ext(
                    self.input_video_body,
                    hi3d.hel_name,
                    projected_or_confirmed_str,
                    distorted_or_undistorted_str,
                    output_corners_3d_dir,
                )

    def execute_heliostat_3d_inference(self, infer_dict):
        # Extract problem specfication components.
        hel_name = infer_dict["hel_name"]
        heliostat_tracks_nfxl = infer_dict["heliostat_tracks_nfxl"]
        projected_or_confirmed_str = infer_dict["projected_or_confirmed_str"]
        distorted_or_undistorted_str = infer_dict["distorted_or_undistorted_str"]
        # output_corner_2d_trajectories_dir = infer_dict['output_corner_2d_trajectories_dir']
        output_construct_corners_3d_dir = infer_dict["output_construct_corners_3d_dir"]
        render_control = infer_dict["render_control"]

        # print('In Heliostats3d.execute_heliostat_3d_inference(), hel_name =', hel_name)  # ?? SCAFFOLDING RCB -- TEMPORARY
        # if hel_name != '11W7': return  # ?? SCAFFOLDING RCB -- TEMPORARY ****
        # if (hel_name != '5W3') and \
        #    (hel_name != '11W7') and \
        #    (hel_name != '11W9') and \
        #    (hel_name != '9W1'):
        # if (hel_name != '11W7') and \
        #    (hel_name != '11W9'):
        #        if (hel_name != '5W3'):
        #        if (hel_name != '11W9'):
        #        if (hel_name != '11W9'):
        # if (hel_name != '5W3') and \
        #    (hel_name != '7W5') and \
        #    (hel_name != '9E9') and \
        #    (hel_name != '11W9') and \
        #    (hel_name != '14E3'):
        #            return  # ?? SCAFFOLDING RCB -- TEMPORARY ****

        print(
            "In Heliostats3d.execute_heliostat_3d_inference(), hel_name =", hel_name
        )  # ?? SCAFFOLDING RCB -- TEMPORARY

        # # First generate a per-heliostat corner trajectories file in the format required by the C++ inference executable.
        # corner_trajectory_dict = self.generate_corner_2d_trajectories_file(hel_name, heliostat_tracks_nfxl, projected_or_confirmed_str, distorted_or_undistorted_str, output_corner_2d_trajectories_dir)
        # print('In Heliostats3d.execute_heliostat_3d_inference(), corner_trajectory_dict:')  # ?? SCAFFOLDING RCB -- TEMPORARY
        # dt.print_dict(corner_trajectory_dict, indent=4)  # ?? SCAFFOLDING RCB -- TEMPORARY

        # Fetch the list of [frame xy_list] pairs for this heliostat.
        list_of_frame_id_xy_lists = heliostat_tracks_nfxl.list_of_frame_xy_lists(hel_name)
        # Remove lens distortion, if desired.
        if distorted_or_undistorted_str == "distorted":
            list_of_frame_id_xy_lists_to_process = list_of_frame_id_xy_lists
        elif distorted_or_undistorted_str == "undistorted":
            list_of_frame_id_xy_lists_to_process = [
                [frame_id_xy_list[0], self.undistort_xy_list(frame_id_xy_list[1])]
                for frame_id_xy_list in list_of_frame_id_xy_lists
            ]
        else:
            msg = (
                'In Heliostats3d.execute_heliostat_3d_inference(), encountered distorted_or_undistorted_str="'
                + str(distorted_or_undistorted_str)
                + '" which was neither "distorted" or "undistorted".'
            )
            print("ERROR: " + msg)
            raise ValueError(msg)

        # Construct the 3-d inference for this heliostat.
        result_hi3d = hi3d.HeliostatInfer3d(
            hel_name,
            list_of_frame_id_xy_lists_to_process,
            self.specifications.flat_corner_xyz_list,
            # Execution control.
            self.camera_matrix,
            self.distortion_coefficients,
            self.zero_distortion_coefficients,
            # Input/output sources.
            self.specifications,
            self.theoretical_flat_heliostat_dir_body_ext,
            self.theoretical_flat_heliostat_dict,
            self.theoretical_flat_heliostat_xyz_list,
            self.input_video_body,
            self.input_frame_dir,
            self.input_frame_id_format,
            projected_or_confirmed_str,
            distorted_or_undistorted_str,
            self.output_data_dir,
            output_construct_corners_3d_dir,
            # Render control.
            render_control,
        )  # Flags to control rendering on this run.

        # Return.
        return result_hi3d

    # # Now generate an estimate of the heliostat's 3-d corners.
    # self.generate_corners_3d_file(hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, list_of_frame_xy_lists, output_corners_3d_dir)

    # def generate_corner_2d_trajectories_file(self, hel_name, heliostat_tracks_nfxl, projected_or_confirmed_str, distorted_or_undistorted_str, output_corner_2d_trajectories_dir):
    #     # Output file name.
    #     output_corner_2d_trajectories_dir_body_ext = self.corner_2d_trajectories_dir_body_ext(hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, output_corner_2d_trajectories_dir)
    #     print('\nIn Heliostats3d.generate_corner_2d_trajectories_file(), generating 2d corner trajectory file: ', output_corner_2d_trajectories_dir_body_ext)

    #     # Determine whether to remove lens distortion.
    #     if distorted_or_undistorted_str == 'distorted':
    #         remove_lens_distortion = False
    #     elif distorted_or_undistorted_str == 'undistorted':
    #         remove_lens_distortion = True
    #     else:
    #         msg = 'In Heliostats3d.generate_corner_2d_trajectories_file(), encountered distorted_or_undistorted_str="'+str(distorted_or_undistorted_str)+'" which was neither "distorted" or "undistorted".'
    #         print('ERROR: ' + msg)
    #         raise ValueError(msg)

    #     # Generate dictionary of corner trajectories.
    #     # print('In Heliostats3d.generate_corner_2d_trajectories_file(), generating 2d corner trajectory dictionary...')
    #     list_of_frame_xy_lists = heliostat_tracks_nfxl.list_of_frame_xy_lists(hel_name)
    #     if len(list_of_frame_xy_lists) > 0:
    #         # Determine the number of corners.
    #         first_frame_xy_list = list_of_frame_xy_lists[0]
    #         first_frame_id = first_frame_xy_list[0]
    #         first_xy_list  = first_frame_xy_list[1]
    #         n_corners = len(first_xy_list)
    #         if n_corners == 0:
    #             msg = 'In Heliostats3d.generate_corner_2d_trajectories_file(), unexpected n_corners == 0 case encountered for heliostat: "' + hel_name + '".'
    #             print('ERROR: '+msg)
    #             raise ValueError(msg)
    #         # Create a dictionary to hold the corner trajectories.
    #         corner_trajectory_dict = {}
    #         for corner_idx in range(0,n_corners):
    #             corner_trajectory_dict[corner_idx] = []
    #         # Fill the dictionary.
    #         for frame_xy_list in list_of_frame_xy_lists:
    #             frame_id = frame_xy_list[0]
    #             xy_list  = frame_xy_list[1]
    #             if len(xy_list) == 0:
    #                 msg = 'In Heliostats3d.generate_corner_2d_trajectories_file(), unexpected zero-length xy_list encountered for heliostat: "' + hel_name + '".'
    #                 print('ERROR: '+msg)
    #                 raise ValueError(msg)
    #             if remove_lens_distortion:
    #                 output_xy_list = self.undistort_xy_list(xy_list)
    #             else:
    #                 output_xy_list = xy_list
    #             corner_idx = 0
    #             for xy in output_xy_list:
    #                 x_str = str(xy[0])
    #                 y_str = str(xy[1])
    #                 corner_trajectory_dict[corner_idx].append(x_str)
    #                 corner_trajectory_dict[corner_idx].append(y_str)
    #                 corner_idx += 1
    #         # # Print result.
    #         # dt.print_dict(corner_trajectory_dict, indent=4)

    #     # Convert corner trajectory dictionary to a list of rows.
    #     # print('In Heliostats3d.generate_corner_2d_trajectories_file(), generating 2d corner trajectory row list...')
    #     output_corner_trajectory_rows = dt.list_of_values_in_sorted_key_order(corner_trajectory_dict)
    #     # # Print result.
    #     # lt.print_list(output_corner_trajectory_rows)

    #     # Write the corner trajectory file.
    #     # print('In Heliostats3d.generate_corner_2d_trajectories_file(), writing file...')
    #     with open(output_corner_2d_trajectories_dir_body_ext, "w") as output_stream:
    #         wr = csv.writer(output_stream, delimiter=' ')
    #         wr.writerows(output_corner_trajectory_rows)

    #     # Return.
    #     return corner_trajectory_dict

    def undistort_xy_list(self, input_xy_list):
        """
        Corrects (x,y) image points to remove lens distortion.
        """
        camera_matrix_2 = self.camera_matrix.copy()
        xy_array = np.array(input_xy_list).reshape(-1, 2)
        undistorted_xy_array = cv.undistortPoints(
            xy_array, self.camera_matrix, self.distortion_coefficients, None, camera_matrix_2
        )
        undistorted_xy_list = undistorted_xy_array.reshape(-1, 2).tolist()
        # Points with coordinates [1,-1] are flags for "None" and are expected by OpenCV.  Reset these to their original values.
        return_xy_list = []
        for input_xy, undistorted_xy in zip(input_xy_list, undistorted_xy_list):
            if (input_xy[0] == -1.0) and (input_xy[1] == -1.0):
                return_xy_list.append([-1.0, -1, 0])
            else:
                return_xy_list.append(undistorted_xy)
        return return_xy_list


#     def generate_corners_3d_file(self, hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, list_of_frame_xy_lists, output_corners_3d_dir):
# # # Input file name.
# # input_corner_2d_trajectories_dir_body_ext = self.corner_2d_trajectories_dir_body_ext(hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, input_corner_2d_trajectories_dir)
# # print('In Heliostats3d.generate_corners_3d_file(), input 2d corner trajectory file:                  ', input_corner_2d_trajectories_dir_body_ext)  # Pad spaces to align output.
#         # Heliostat name.
#         print('\nIn Heliostats3d.generate_corners_3d_file(), hel_name = ', hel_name)
#         print('In Heliostats3d.generate_corners_3d_file(), len(list_of_frame_xy_lists) = ', len(list_of_frame_xy_lists))
#         print('In Heliostats3d.generate_corners_3d_file(), list_of_frame_xy_lists:')
#         lt.print_list(list_of_frame_xy_lists)
#         for frame_xy_list in list_of_frame_xy_lists[0:3]:
#             frame   = frame_xy_list[0]
#             xy_list = frame_xy_list[1]
#             print('In Heliostats3d.generate_corners_3d_file(), frame = ', frame, '  len(xy_list) = ', len(xy_list))
#         print('In Heliostats3d.generate_corners_3d_file(), ...')
#         print('In Heliostats3d.generate_corners_3d_file(), output_corners_3d_dir = ', output_corners_3d_dir)
#         # Output file name.
#         output_corners_3d_dir_body_ext = self.corners_3d_dir_body_ext(hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, output_corners_3d_dir)
#         print('In Heliostats3d.generate_corners_3d_file(), output 3d corner file = ', output_corners_3d_dir_body_ext)
#         print('In Heliostats3d.generate_corners_3d_file(), len(self.specifications.facets_centroids) = ', len(self.specifications.facets_centroids))
#         print('In Heliostats3d.generate_corners_3d_file(), self.specifications.facets_centroids = ', self.specifications.facets_centroids)
#         print('In Heliostats3d.generate_corners_3d_file(), len(self.specifications.facets_corners) = ', len(self.specifications.facets_corners))
#         print('In Heliostats3d.generate_corners_3d_file(), self.specifications.facets_corners = ', self.specifications.facets_corners)
#         print('In Heliostats3d.generate_corners_3d_file(), NOT Calling executable...')
#         print('In Heliostats3d.generate_corners_3d_file(), Returning No-Op.')


# Original OpenCV reconstruct version.
#     def generate_corners_3d_file(self, hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, input_corner_2d_trajectories_dir, output_corners_3d_dir):
#         # Input file name.
#         input_corner_2d_trajectories_dir_body_ext = self.corner_2d_trajectories_dir_body_ext(hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, input_corner_2d_trajectories_dir)
#         print('In Heliostats3d.generate_corners_3d_file(), input 2d corner trajectory file:                  ', input_corner_2d_trajectories_dir_body_ext)  # Pad spaces to align output.
#         # Output file name.
#         output_corners_3d_dir_body_ext = self.corners_3d_dir_body_ext(hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, output_corners_3d_dir)
#         print('In Heliostats3d.generate_corners_3d_file(), output 3d corner file:                            ', output_corners_3d_dir_body_ext)  # Pad spaces to align output.
#         # Generate 3-d reconstruction.
#         fx = self.camera_matrix[0,0]
#         fy = self.camera_matrix[1,1]
#         cx = self.camera_matrix[0,2]
#         cy = self.camera_matrix[1,2]
# # print('In Heliostats3d.generate_corners_3d_file(), self.reconstruct_executable_path = ', self.reconstruct_executable_path)  # ?? SCAFFOLDING RCB -- TEMPORARY
# # print('In Heliostats3d.generate_corners_3d_file(), input_corner_2d_trajectories_dir_body_ext = ', input_corner_2d_trajectories_dir_body_ext)
# # print('In Heliostats3d.generate_corners_3d_file(), str(fx)  = ', str(fx))
# # print('In Heliostats3d.generate_corners_3d_file(), str(fy)  = ', str(fy))
# # print('In Heliostats3d.generate_corners_3d_file(), str(cx)  = ', str(cx))
# # print('In Heliostats3d.generate_corners_3d_file(), str(cy)  = ', str(cy))
# # print('In Heliostats3d.generate_corners_3d_file(), hel_name = ', hel_name)
# # print('In Heliostats3d.generate_corners_3d_file(), output_corners_3d_dir = ', (output_corners_3d_dir+'/'))  # ?? SCAFFOLDING RCB -- PLATFORM-SPECIFIC
#         print('Calling executable...')
#         proc = subprocess.Popen([self.reconstruct_executable_path,
#                                  input_corner_2d_trajectories_dir_body_ext,
#                                  str(fx),
#                                  str(fy),
#                                  str(cx),
#                                  str(cy),
#                                  hel_name,
#                                  (output_corners_3d_dir+'/')])  # ?? SCAFFOLDING RCB -- PLATFORM-SPECIFIC
#         proc.wait()
#         print('Executable finished.')
#         # We plan to recompile the C++ executable to have more fine-grain control over its output filename, but not today.
#         # So rename the output file to match our naming standard.
#         executable_output_body_ext = hel_name + '_reconstructed.txt'
#         executable_output_dir_body_ext = os.path.join(output_corners_3d_dir, executable_output_body_ext)
#         ft.rename_file(executable_output_dir_body_ext, output_corners_3d_dir_body_ext)


# def corner_2d_trajectories_dir_body_ext(self, hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, corner_2d_trajectories_dir):
#     corner_2d_trajectories_body         = '_'.join([self.input_video_body, hel_name, projected_or_confirmed_str, distorted_or_undistorted_str, 'corner_2d_trajectories'])
#     corner_2d_trajectories_body_ext     = corner_2d_trajectories_body + '.txt'
#     corner_2d_trajectories_dir_body_ext = os.path.join(corner_2d_trajectories_dir, corner_2d_trajectories_body_ext)
#     return corner_2d_trajectories_dir_body_ext


# # WRITE RESULT

# def save_heliostat_corners_3d(self, heliostat_corners_3d_nfxl, projected_or_confirmed_str):
#     # Filenames.
#     # Write the NameFrameXyList object.
#     if projected_or_confirmed_str == 'projected':
#         heliostat_corners_3d_dir_body_ext = self.heliostat_projected_corners_3d_dir_body_ext
#     elif projected_or_confirmed_str == 'confirmed':
#         heliostat_corners_3d_dir_body_ext = self.heliostat_confirmed_corners_3d_dir_body_ext
#     else:
#         msg = 'In Heliostats3d.save_heliostat_corners_3d(), encountered projected_or_confirmed_str="'+str(projected_or_confirmed_str)+'" which was neither "projected" or "confirmed".'
#         print('ERROR: ' + msg)
#         raise ValueError(msg)
#     print('In Heliostats3d.save_heliostat_corners_3d(), writing heliostat track file: ', heliostat_corners_3d_dir_body_ext)
#     ft.create_directories_if_necessary(self.output_data_dir)
#     heliostat_corners_3d_nfxl.save(heliostat_corners_3d_dir_body_ext)


# def save_data(self, heliostat_corners_3d_nfxl, projected_or_confirmed_str):
#     # Filenames.
#     if projected_or_confirmed_str == 'projected':
#         dict_body     = self.dict_projected_body
#         vfph_body     = self.vfpph_body
#         vfph_body_ext = self.vfpph_body_ext
#     elif projected_or_confirmed_str == 'confirmed':
#         dict_body     = self.dict_confirmed_body
#         vfph_body     = self.vfpch_body
#         vfph_body_ext = self.vfpch_body_ext
#     else:
#         msg = 'In Heliostats3d.save_data(), encountered projected_or_confirmed_str="'+str(projected_or_confirmed_str)+'" which was neither "projected" or "confirmed".'
#         print('ERROR: ' + msg)
#         raise ValueError(msg)
#     # Statistics.
#     summary_dict = {}
#     summary_dict['n_heliostat_3d_corner_frames'] = heliostat_corners_3d_nfxl.number_of_frames()
#     print('In Heliostats3d.save_data(), writing key frame ' + projected_or_confirmed_str + ' summary statistics...')
#     ft.write_dict_file('heliostat ' + projected_or_confirmed_str + ' corners_3d summary statistics', self.output_data_dir, dict_body, summary_dict)
#     # Video frames per heliostat.
#     video_frames_per_heliostat_dict = heliostat_corners_3d_nfxl.frames_per_heliostat()
#     print('In Heliostats3d.save_data(), writing video frames per ' + projected_or_confirmed_str + ' heliostat:', os.path.join(self.output_data_dir, vfph_body_ext))
#     ft.write_dict_file(None, self.output_data_dir, vfph_body, video_frames_per_heliostat_dict)


# # LOAD RESULT

# def read_heliostat_corner_2d_trajectories(self):
#     # Projected.
#     print('In Heliostats3d.read_heliostat_corner_2d_trajectories(), reading heliostat projected distorted corner_2d trajectories file: ', self.heliostat_projected_corners_3d_dir_body_ext)
#     self.heliostat_projected_corners_3d_nfxl = nfxl.NameFrameXyList()
#     self.heliostat_projected_corners_3d_nfxl.load(self.heliostat_projected_corners_3d_dir_body_ext)
#     # Confirm what was read.
#     print('In Heliostats3d.read_heliostat_corners_3d(), heliostat projected corners_3d read:')
#     self.heliostat_projected_corners_3d_nfxl.print(max_keys=12, max_value_length=200, indent=4)
#     # Confirmed.
#     print('In Heliostats3d.read_heliostat_corners_3d(), reading heliostat confirmed corners_3d file: ', self.heliostat_confirmed_corners_3d_dir_body_ext)
#     self.heliostat_confirmed_corners_3d_nfxl = nfxl.NameFrameXyList()
#     self.heliostat_confirmed_corners_3d_nfxl.load(self.heliostat_confirmed_corners_3d_dir_body_ext)
#     # Confirm what was read.
#     print('In Heliostats3d.read_heliostat_corners_3d(), heliostat confirmed corners_3d read:')
#     self.heliostat_confirmed_corners_3d_nfxl.print(max_keys=12, max_value_length=200, indent=4)


# def read_heliostat_corners_3d(self):
#     # Projected.
#     print('In Heliostats3d.read_heliostat_corners_3d(), reading heliostat projected corners_3d file: ', self.heliostat_projected_corners_3d_dir_body_ext)
#     self.heliostat_projected_corners_3d_nfxl = nfxl.NameFrameXyList()
#     self.heliostat_projected_corners_3d_nfxl.load(self.heliostat_projected_corners_3d_dir_body_ext)
#     # Confirm what was read.
#     print('In Heliostats3d.read_heliostat_corners_3d(), heliostat projected corners_3d read:')
#     self.heliostat_projected_corners_3d_nfxl.print(max_keys=12, max_value_length=200, indent=4)
#     # Confirmed.
#     print('In Heliostats3d.read_heliostat_corners_3d(), reading heliostat confirmed corners_3d file: ', self.heliostat_confirmed_corners_3d_dir_body_ext)
#     self.heliostat_confirmed_corners_3d_nfxl = nfxl.NameFrameXyList()
#     self.heliostat_confirmed_corners_3d_nfxl.load(self.heliostat_confirmed_corners_3d_dir_body_ext)
#     # Confirm what was read.
#     print('In Heliostats3d.read_heliostat_corners_3d(), heliostat confirmed corners_3d read:')
#     self.heliostat_confirmed_corners_3d_nfxl.print(max_keys=12, max_value_length=200, indent=4)


# # RENDER RESULT

# def render(self):
#     print('In Heliostats3d.render(), self.output_construction_projected_dir=', self.output_construction_projected_dir)  # ?? SCAFFOLDING RCB -- TEMPORARY
#     print('In Heliostats3d.render(), self.output_construction_confirmed_dir=', self.output_construction_confirmed_dir)  # ?? SCAFFOLDING RCB -- TEMPORARY
#     # Projected.
#     if self.render_control_projected.draw_heliostat_corners_3d and self.generated_heliostat_projected_corners_3d:  # Don't render unless we generated.
#         self.render_aux(self.heliostat_projected_corners_3d_nfxl, self.output_construction_projected_dir, self.render_control_projected, 'projected')
#     # Confirmed.
#     if self.render_control_confirmed.draw_heliostat_corners_3d and self.generated_heliostat_confirmed_corners_3d:  # Don't render unless we generated.
#         self.render_aux(self.heliostat_confirmed_corners_3d_nfxl, self.output_construction_confirmed_dir, self.render_control_confirmed, 'confirmed')


# def render_aux(self, heliostat_corners_3d_nfxl, output_construction_dir, render_control, projected_or_confirmed_str):
#     print('In Heliostats3d.render_aux(), rendering heliostat ' + projected_or_confirmed_str + ' corners_3d...')
#     print('In Heliostats3d.render_aux(), output_construction_dir=', output_construction_dir)  # ?? SCAFFOLDING RCB -- TEMPORARY
#     print('WARNING: In Heliostats3d.render_aux(), not implemented yet.')


if __name__ == "__main__":
    # Execution control.
    force_construction = True  # False
    specific_frame_ids = None
    single_processor = False  # True

    # log_dir_body_ext                          = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_180_Heliostats3d/mavic_zoom/log/Heliostats3d_log.txt'
    # # Input/output sources.
    # specifications                            = Dspec.nsttf_specifications()  # Solar field parameters.
    # theoretical_flat_heliostat_dir_body_ext   = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/010_HeliostatModel/data/NSTTF_Facet_Corners.csv'
    # input_video_dir_body_ext                  = experiment_dir() + '2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4'
    # input_frame_dir                           = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/'
    # input_frame_id_format                     = '06d' # Note different from format used in ffmpeg call, which is '.%06d'
    # input_video_projected_tracks_dir_body_ext = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_170_HeliostatTracks/mavic_zoom/data/DJI_427t_428_429_heliostat_projected_tracks_nfxl.csv'
    # input_video_confirmed_tracks_dir_body_ext = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_170_HeliostatTracks/mavic_zoom/data/DJI_427t_428_429_heliostat_confirmed_tracks_nfxl.csv'
    # output_data_dir                           = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_180_Heliostats3d/mavic_zoom/data/'
    # output_render_dir                         = experiment_dir() + '2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/Small_180_Heliostats3d/mavic_zoom/render/'
    # output_construction_dir                   = experiment_dir() + '2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/Small_180c_Heliostats3d/mavic_zoom/'

    log_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/180_Heliostats3d/mavic_zoom/log/Heliostats3d_log.txt"
    )
    # Input/output sources.
    specifications = Dspec.nsttf_specifications()  # Solar field parameters.
    theoretical_flat_heliostat_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/010_HeliostatModel/data/NSTTF_Facet_Corners.csv"
    )
    input_video_dir_body_ext = (
        experiment_dir() + "2020-12-03_FastScan1/2_Data/20201203/1544_NS_U/mavic_zoom/DJI_427t_428_429.MP4"
    )
    input_frame_dir = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/080c_FramesNoDuplicates/mavic_zoom/frames/"
    )
    input_frame_id_format = "06d"  # Note different from format used in ffmpeg call, which is '.%06d'
    input_video_projected_tracks_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/170_HeliostatTracks/mavic_zoom/data/DJI_427t_428_429_heliostat_projected_tracks_nfxl.csv"
    )
    input_video_confirmed_tracks_dir_body_ext = (
        experiment_dir()
        + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/170_HeliostatTracks/mavic_zoom/data/DJI_427t_428_429_heliostat_confirmed_tracks_nfxl.csv"
    )
    output_data_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/180_Heliostats3d/mavic_zoom/data/"
    )
    output_render_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Answers/20201203/1544_NS_U/180_Heliostats3d/mavic_zoom/render/"
    )
    output_construction_dir = (
        experiment_dir() + "2020-12-03_FastScan1/3_Post/Construction/20201203/1544_NS_U/180c_Heliostats3d/mavic_zoom/"
    )

    # Render control.
    render_control_projected_distorted = rchr.default(color="m")  # ?? SCAFFOLDING RCB -- TEMPORARY
    render_control_projected_undistorted = rchr.default(color="r")  # ?? SCAFFOLDING RCB -- TEMPORARY
    render_control_confirmed_distorted = rchr.default(color="c")  # ?? SCAFFOLDING RCB -- TEMPORARY
    render_control_confirmed_undistorted = rchr.default(color="b")  # ?? SCAFFOLDING RCB -- TEMPORARY
    # render_control_projected_distorted        = rchr.fast()  # Don't draw frames.  # ?? SCAFFOLDING RCB -- TEMPORARY
    # render_control_projected_undistorted      = rchr.fast()  # Don't draw frames.  # ?? SCAFFOLDING RCB -- TEMPORARY
    # render_control_confirmed_distorted        = rchr.fast()  # Don't draw frames.  # ?? SCAFFOLDING RCB -- TEMPORARY
    # render_control_confirmed_undistorted      = rchr.fast()  # Don't draw frames.  # ?? SCAFFOLDING RCB -- TEMPORARY
    render_control_confirmed_undistorted = rckfgm.default()  # ?? SCAFFOLDING RCB -- TEMPORARY

    key_frames_object = Heliostats3d(  # Execution control.
        force_construction,
        specific_frame_ids,
        single_processor,
        log_dir_body_ext,
        # Input/output sources.
        specifications,
        theoretical_flat_heliostat_dir_body_ext,
        input_video_dir_body_ext,
        input_frame_dir,
        input_frame_id_format,
        input_video_projected_tracks_dir_body_ext,
        input_video_confirmed_tracks_dir_body_ext,
        output_data_dir,
        output_render_dir,
        output_construction_dir,
        # Render control.
        render_control_projected_distorted,
        render_control_projected_undistorted,
        render_control_confirmed_distorted,
        render_control_confirmed_undistorted,
    )
