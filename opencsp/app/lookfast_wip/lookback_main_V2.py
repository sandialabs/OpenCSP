import os
import time
from logging import DEBUG, ERROR
import tkinter as tk
from tkinter import filedialog
import re
import numpy as np
import cv2
from datetime import datetime, timedelta


import opencsp.app.sofast.lib.image_processing as imgp
import opencsp.common.lib.tool.file_tools as ft
import opencsp.app.lookback.lookback_tools as lbt
from opencsp.app.lookback.interactive_video_info_extract import interactive_video_select
import opencsp.app.lookback.coverage_map_mp as cvg_map
import opencsp.app.lookback.time_history_array_mp_npz as time_hist
import opencsp.app.lookback.time_history_transitions_npz_mp as transitions
import opencsp.app.lookback.celestial_vectors as astro_math
import opencsp.app.lookback.lookfast_camera_adjust_V2_module as rt_cam_adjust

from opencsp.common.lib.camera.Camera import Camera

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=ft.join(os.getcwd(), "error_logs"), log_file_name="error_log_lookback_main_debug.txt", log_type=DEBUG
)


def select_file(window_title="Select a File", file_types=None):
    """
    Opens a file dialog to select a file, with customizable window title and file types.

    Parameters:
        window_title (str): The title of the file dialog window (default is "Select a File").
        file_types (str): A string of file types in the format "*.ext1 *.ext2", e.g., "*.jpg *.png".
                         If None, defaults to allowing all files.

    Returns:
        str: The file path of the selected file, or None if no file was selected.
    """
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Parse file types into the required format for filedialog
    if file_types:
        file_types_list = [("Custom File Types", file_types.lower().split()), ("All Files", "*.*")]
    else:
        file_types_list = [("All Files", "*.*")]

    # Open the file dialog
    file_path = filedialog.askopenfilename(title=window_title, filetypes=file_types_list)
    if file_path:
        logger.info("File Selected: %s", file_path)
    else:
        logger.info("File Selected: None")

    root.destroy()
    # Return the selected file path
    return os.path.normpath(file_path) if file_path else None


def select_dir(window_title="Select a Directory"):
    """
    Opens a file dialog to select a Directory,
    Parameters:
        window_title (str): The title of the file dialog window (default is "Select a File").

    Returns:
        str: The file path of the selected directory, or None if no directory was selected.
    """
    # Create a hidden root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open the file dialog
    dir_path = filedialog.askdirectory(title=window_title, initialdir=os.path.normpath(os.getcwd()))
    if dir_path:
        logger.info("Directory Selected: %s", dir_path)
    else:
        logger.info("Directory Selected: None")

    root.destroy()
    # Return the selected file path
    return os.path.normpath(dir_path) if dir_path else None


def create_output_structure_and_ini_file(init_file=None):
    if init_file:
        init_exists = ft.file_exists(init_file)
    else:
        init_exists = False

    if init_exists is False:
        data_output_folders = [
            r'0_checkpoints',
            r'1_video_frames',
            r'2_video_frames_cropped',
            r'3_specific_cropped_frames',
            r'4_coverage_map',
            r'5_accelerated_test_video',
            r'6_time_history_output',
            r'7_pixel_timing_interrogation',
            r'8_pixel_vector_information',
            r'9_sofast_data_compare',
            r'reference_images',
        ]

        og_vid_path = select_file(window_title="Select Original Video to Copy and Process", file_types="*.mov *.mp4")
        og_vid_dir, og_vid_name, og_vid_ext = ft.path_components(og_vid_path)
        primary_folder = select_dir(window_title="Select Primary Output Directory")
        if not primary_folder:
            ft.create_directories_if_necessary(ft.join(og_vid_dir, "LookFast_Processing"))
            primary_folder = ft.join(og_vid_dir, "LookFast_Processing")

        ft.copy_file(input_dir_body_ext=og_vid_path, output_dir=primary_folder)
        for folder in data_output_folders:
            ft.create_directories_if_necessary(ft.join(primary_folder, folder))

        init_details = {
            "primary_folder": primary_folder,
            "og_vid_path": og_vid_path,
            "og_vid_dir": og_vid_dir,
            "og_vid_name": og_vid_name,
            "og_vid_ext": og_vid_ext,
        }

        ft.write_json(
            description=f"Initilization File for Lookfast Processing of Video {og_vid_path}",
            output_dir=primary_folder,
            output_file_body="lookfast_ini",
            output_object=init_details,
        )

        return init_details

    else:
        init_dir, init_name, init_ext = ft.path_components(init_file)
        init_details = ft.read_json(description=None, input_dir=init_dir, input_file_body_ext=init_name + init_ext)
        return init_details


def main():

    # primary_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025"
    # video_name = "DSC_0025.MOV"
    # light_image = cv2.imread(ft.join(primary_folder, "light_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    # dark_image = cv2.imread(ft.join(primary_folder, "dark_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    # do mask selection after coverage maps... use a coverage map and a dark image?
    '''
    mask_raw = imgp.calc_mask_raw(
        np.concatenate((dark_image[:, :, np.newaxis], light_image[:, :, np.newaxis]), axis=2),
        hist_thresh=0.5,
        filt_width=9,
        filt_thresh=4,
        thresh_active_pixels=0.01,
    )
    mask = imgp.keep_largest_mask_area(mask_raw)
    v_mask_centroid_image = imgp.centroid_mask(mask)
    v_edges_image = imgp.edges_from_mask(mask)
    '''
    init_file = select_file(window_title="Select Initilization Json File If Exists", file_types="*.json")
    init_details = create_output_structure_and_ini_file(init_file)

    primary_folder = init_details["primary_folder"]
    og_vid_dir = init_details["og_vid_dir"]
    og_vid_name = init_details["og_vid_name"]
    og_vid_ext = init_details["og_vid_ext"]

    checkpoint_folder = ft.join(primary_folder, "0_checkpoints")
    checkpoint_main_name = "lookback_main_checkpoint.json"

    # Load checkpoint if it exists
    checkpoint_main_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_main_name)
    if checkpoint_main_data is None:
        checkpoint_main_data = {"Completed_Steps": []}

    start_time = time.time()
    logger.info("Code Start Time: %s", str(time.ctime(start_time)))

    # fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    fractions = [0.5]
    analysis_fractions = [0.5]

    celestial_object = "sun"
    video_metadata = lbt.extract_detailed_video_metadata(ft.join(primary_folder, og_vid_name + og_vid_ext))
    timezone = "America/Denver"
    camera_time_shift = timedelta(hours=0, minutes=0, seconds=0)
    # Define observer location Example ≈NSTTF Tower 260 Level Balcony West Side
    observer_lat = (34, 57, 44.56)  # (degree, minute, second) negative degree for south
    observer_long = (-106, 30, 34.90)  # (degree, minute, second) negative degree for west
    observer_elevation = 1755.648 + 1.2192
    observer_loc = (lbt.lat_long_to_decimal(observer_lat), lbt.lat_long_to_decimal(observer_long), observer_elevation)

    # Define target location Example ≈Sun Data Marker 2 In front of 5E8
    target_lat = (34, 57, 45.92)  # (degree, minute, second) negative degree for south
    target_long = (-106, 30, 31.88)  # (degree, minute, second) negative degree for west
    target_elevation = 1706.88
    target_loc = (lbt.lat_long_to_decimal(target_lat), lbt.lat_long_to_decimal(target_long), target_elevation)

    # TODO Need to change this to load in a camera object
    ##### Setting camera object

    # Sofast Example Camera Intrinsic matrix
    K_intrin = np.array([[5492.064314084441, 0, 1920 / 2], [0, 5486.2706013814895, 1080 / 2], [0, 0, 1]])
    # Sofast Example Camera Distortion coefficients
    D_coeff = np.array([-0.144160742602367, 1.609744377391114, 2.503498158416561e-5, -0.001899042260179])

    cam = Camera(
        intrinsic_mat=K_intrin, distortion_coef=D_coeff, image_shape_xy=tuple[1920, 1080], name="Arbitrary_Example"
    )

    cam_to_optic_reference_distance = 99.94392  # meters
    # Need to select a reference pixel to align the horizonal and camera coordinate system. Should make this part of the Video Scrubber step.
    reference_pixel_key = "(500, 900)"
    # optical_axis_pixel_key = f"({int(K_intrin[1,2])}, {int(K_intrin[0,2])})"

    if "extracted_video_frames" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Video Frame Extraction : %s", str(time.time() - start_time))
        if ft.file_exists(input_dir_body_ext=ft.join(primary_folder, "interactive_scrubber_selections.json")):
            scrubber_details = lbt.read_json(ft.join(primary_folder, "interactive_scrubber_selections.json"))
    else:

        scrubber_details = interactive_video_select(
            video_path=ft.join(primary_folder, og_vid_name + og_vid_ext),
            dest_path=ft.join(primary_folder, "1_video_frames"),
            frame_subset_dir=ft.join(primary_folder, "3_specific_cropped_frames"),
        )
        lbt.write_json(scrubber_details, ft.join(primary_folder, "interactive_scrubber_selections.json"))

        checkpoint_main_data["Completed_Steps"].append("extracted_video_frames")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )
        logger.info("Time to Complete Video Frame Extraction: %s", str(time.time() - start_time))

    if "coverage_maps" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Coverage Maps : %s", str(time.time() - start_time))
        if ft.file_exists(ft.join(primary_folder, "threshold_bmap_mask_paths.json")):
            thresh_maps_paths = lbt.read_json(ft.join(primary_folder, "threshold_bmap_mask_paths.json"))
    else:
        cvg_map.construct_binary_maps_parallel(
            image_folder=ft.join(primary_folder, "3_specific_cropped_frames"),
            output_folder=ft.join(primary_folder, "4_coverage_map"),
            checkpoint_folder=checkpoint_folder,
            threshold_fractions=fractions,
            batch_size=1000,
            max_workers=4,
            checkpoint_file="coverage_map_mp_checkpoint.json",
        )
        binary_maps = ft.files_in_directory(
            input_dir=ft.join(primary_folder, "4_coverage_map"), sort=True, files_only=True, recursive=False
        )
        thresh_maps_compiled = [f for f in binary_maps if "compiled" in f]
        thresh_maps_paths = [ft.join(primary_folder, "4_coverage_map", f) for f in thresh_maps_compiled]
        lbt.write_json(thresh_maps_paths, ft.join(primary_folder, "threshold_bmap_mask_paths.json"))

        checkpoint_main_data["Completed_Steps"].append("coverage_maps")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )
        logger.info("Time to Complete Coverage Maps: %s", str(time.time() - start_time))

    if "accelerated_video" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Accelerated Video: %s", str(time.time() - start_time))
    else:
        accel_factor = 10
        lbt.accelerate_video_ffmpeg_no_audio(
            input_path=ft.join(primary_folder, og_vid_name + og_vid_ext),
            output_path=ft.join(
                primary_folder, "5_accelerated_test_video", f"{accel_factor}x_accelerated_test_video.MOV"
            ),
            playback_speed=accel_factor,
        )
        checkpoint_main_data["Completed_Steps"].append("accelerated_video")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )
        logger.info("Time to Complete Accelerated Video: %s", str(time.time() - start_time))

    if "time_history" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Time History: %s", str(time.time() - start_time))
    else:
        time_hist.create_binary_pixel_array_parallel_with_multiprocessing(
            image_folder_path=ft.join(primary_folder, "3_specific_cropped_frames"),
            percentages=fractions,
            output_folder=ft.join(primary_folder, "6_time_history_output"),
            checkpoint_folder=checkpoint_folder,
            batch_size=500,
            overlap=1,
            checkpoint_file="time_history_array_checkpoint_mp.json",
            num_workers=4,
        )
        checkpoint_main_data["Completed_Steps"].append("time_history")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )
        logger.info("Time to Complete Time History Arrays: %s", str(time.time() - start_time))

    if "pixel_transitions" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Transition History: %s", str(time.time() - start_time))
    else:
        for level in analysis_fractions:
            mask_key = f"{int(level*100):02d}"
            mask_file_path = [fp for fp in thresh_maps_paths if mask_key in os.path.basename(fp)]

            if len(mask_file_path) > 1:
                raise ValueError("Too many binary map mask files match the analysis fraction key")
            else:
                mask_raw = cv2.imread(mask_file_path[0], cv2.IMREAD_GRAYSCALE)
                bright_pixels = np.argwhere(mask_raw == np.max(mask_raw))
                pixel_locs = [(y, x) for y, x in bright_pixels]

                transitions.analyze_pixel_brightness_parallel_npz(
                    npz_folder=ft.join(primary_folder, "6_time_history_output", mask_key),
                    pixel_locations=pixel_locs,
                    output_folder=ft.join(primary_folder, "7_pixel_timing_interrogation", mask_key),
                    final_output_file=f"time_history_transition_parallel_{mask_key}_final.json.gz",
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name="time_history_transition_checkpoint_mp.json",
                )
                checkpoint_main_data["Completed_Steps"].append("pixel_transitions")
                lbt.save_checkpoint(
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name=checkpoint_main_name,
                    checkpoint_data=checkpoint_main_data,
                )
                logger.info("Time to Complete Time History Transitions: %s", str(time.time() - start_time))

    if "pixel_transition_plots" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Transition Plots: %s", str(time.time() - start_time))
    else:
        for level in analysis_fractions:
            mask_key = f"{int(level*100):02d}"

            transitions.create_timing_plots_json_parallel(
                compiled_json=ft.join(
                    primary_folder,
                    "7_pixel_timing_interrogation",
                    mask_key,
                    f"time_history_transition_parallel_{mask_key}_final.json.gz",
                ),
                output_folder=ft.join(primary_folder, "7_pixel_timing_interrogation", "pixel_timing_plots", mask_key),
                source_image_folder=ft.join(primary_folder, "3_specific_cropped_frames"),
                checkpoint_folder=checkpoint_folder,
                checkpoint_file="pixel_timing_plots_checkpoint_mp.json",
            )
            checkpoint_main_data["Completed_Steps"].append("pixel_transition_plots")
            lbt.save_checkpoint(
                checkpoint_folder=checkpoint_folder,
                checkpoint_file_name=checkpoint_main_name,
                checkpoint_data=checkpoint_main_data,
            )
            logger.info("Time to Complete Timing Plots: %s", str(time.time() - start_time))

    if "celestial_vectors_data" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Celestial Vector Processing: %s", str(time.time() - start_time))
    else:
        for level in analysis_fractions:
            mask_key = f"{int(level*100):02d}"
            mask_file_path = [fp for fp in thresh_maps_paths if mask_key in os.path.basename(fp)]

            if len(mask_file_path) > 1:
                raise ValueError("Too many binary map mask files match the analysis fraction key")
            else:

                astro_math.extract_pixel_timing_and_celestial_vectors_parallel(
                    celestial_object_name=celestial_object,
                    target_location=target_loc,
                    observer_location=observer_loc,
                    camera_time_shift=camera_time_shift,
                    data_time_zone=timezone,
                    data_location=ft.join(
                        primary_folder,
                        "7_pixel_timing_interrogation",
                        mask_key,
                        f"time_history_transition_parallel_{mask_key}_final.json.gz",
                    ),
                    video_metadata=video_metadata,
                    output_folder=ft.join(primary_folder, "8_pixel_vector_information", mask_key),
                    output_json_name=f"celestial_vector_data_mask_{mask_key}.json.gz",
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file="celestial_vector_data_checkpoint_mp.json",
                    batch_size=5000,
                )

                checkpoint_main_data["Completed_Steps"].append("celestial_vectors_data")
                lbt.save_checkpoint(
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name=checkpoint_main_name,
                    checkpoint_data=checkpoint_main_data,
                )
                logger.info("Time to Complete Celestial Vector Calculations: %s", str(time.time() - start_time))

    if "celestial_vectors_plots" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Celestial Vector Plotting: %s", str(time.time() - start_time))
    else:
        for level in analysis_fractions:
            mask_key = f"{int(level*100):02d}"
            mask_file_path = [fp for fp in thresh_maps_paths if mask_key in os.path.basename(fp)]

            if len(mask_file_path) > 1:
                raise ValueError("Too many binary map mask files match the analysis fraction key")
            else:

                astro_math.plotting_pixel_transition_vectors_decoupled_mp(
                    data_location=ft.join(
                        primary_folder,
                        "8_pixel_vector_information",
                        mask_key,
                        f"celestial_vector_data_mask_{mask_key}.json.gz",
                    ),
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_data_file="celestial_vector_data_checkpoint_mp.json",
                    checkpoint_plot_file="celestial_vector_plots_checkpoint_mp.json",
                    celestial_object=celestial_object,
                    output_folder_img=ft.join(
                        primary_folder, "8_pixel_vector_information", mask_key, "celestial_vector_plots"
                    ),
                    batch_size=1000,
                )

                checkpoint_main_data["Completed_Steps"].append("celestial_vectors_plots")
                lbt.save_checkpoint(
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name=checkpoint_main_name,
                    checkpoint_data=checkpoint_main_data,
                )
                logger.info("Time to Complete Celestial Vector Plots: %s", str(time.time() - start_time))

    if "lookfast_camera_adjust" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Lookfast Camera Adjustments: %s", str(time.time() - start_time))
    else:
        for level in analysis_fractions:
            mask_key = f"{int(level*100):02d}"
            mask_file_path = [fp for fp in thresh_maps_paths if mask_key in os.path.basename(fp)]

            if len(mask_file_path) > 1:
                raise ValueError("Too many binary map mask files match the analysis fraction key")
            else:

                rt_cam_adjust.main(
                    camera_obj=cam,
                    light_mask_path=mask_file_path[0],
                    vec_data_path=ft.join(
                        primary_folder,
                        "8_pixel_vector_information",
                        mask_key,
                        f"celestial_vector_data_mask_{mask_key}.json.gz",
                    ),
                    reference_distance_m=cam_to_optic_reference_distance,
                    reference_pixel_key=reference_pixel_key,
                    output_directory=ft.join(primary_folder, "9_sofast_data_compare"),
                    checkpoint_directory=checkpoint_folder,
                    checkpoint_file="rotate_translate_camera_adjust_checkpoint.json",
                )

                checkpoint_main_data["Completed_Steps"].append("lookfast_camera_adjust")
                lbt.save_checkpoint(
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name=checkpoint_main_name,
                    checkpoint_data=checkpoint_main_data,
                )
                logger.info(
                    "Time to Complete Rotate/Translate Adjustment Calculations: %s", str(time.time() - start_time)
                )

    print("here")
    # Need Lookfast camera adjust v2 section


if __name__ == "__main__":
    main()
