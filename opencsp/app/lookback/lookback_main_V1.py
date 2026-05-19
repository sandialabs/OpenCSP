import os
import time
import numpy as np
from logging import DEBUG, ERROR
import cv2

import opencsp.app.sofast.lib.image_processing as imgp
import opencsp.app.lookback.lookback_tools as lbt
import opencsp.app.lookback.coverage_map_mp as cvg_map
import opencsp.app.lookback.time_history_array_mp as time_hist
import opencsp.app.lookback.time_history_transitions_mp as transitions

from opencsp.common.lib.camera.Camera import Camera

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_lookback_main_debug.txt",
    log_type=DEBUG,
)


def main():
    primary_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025"
    video_name = "DSC_0025.MOV"
    checkpoint_folder = os.path.join(primary_folder, "0_checkpoints")
    checkpoint_main_name = "lookback_main_checkpoint.json"

    light_image = cv2.imread(os.path.join(primary_folder, "light_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    dark_image = cv2.imread(os.path.join(primary_folder, "dark_mask_test.png"), cv2.IMREAD_GRAYSCALE)

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

    # Load checkpoint if it exists
    checkpoint_main_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_main_name)
    if checkpoint_main_data is None:
        checkpoint_main_data = {"Completed_Steps": []}

    start_time = time.time()
    logger.info("Code Start Time: %s", str(time.ctime(start_time)))

    fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    if "coverage_maps" in checkpoint_main_data["Completed_Steps"]:
        logger.info("Skipped Already Completed Coverage Maps : %s", str(time.time() - start_time))
    else:
        cvg_map.construct_binary_maps_parallel(
            image_folder=os.path.join(primary_folder, "3_specific_cropped_frames"),
            output_folder=os.path.join(primary_folder, "4_coverage_map"),
            checkpoint_folder=checkpoint_folder,
            threshold_fractions=fractions,
            batch_size=1000,
            max_workers=4,
            checkpoint_file="coverage_map_mp_checkpoint.json",
        )
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
            input_path=os.path.join(primary_folder, video_name),
            output_path=os.path.join(
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
            image_folder_path=os.path.join(primary_folder, "3_specific_cropped_frames"),
            percentages=fractions,
            output_folder=os.path.join(primary_folder, "6_time_history_output"),
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
        '''
        height_range = (435, 670)  # TODO Add pixels_of_interest selection to interactive section
        width_range = (830, 1070)
        pixels_of_interest = [
            (height, width)
            for height in range(height_range[0], height_range[1])
            for width in range(width_range[0], width_range[1])
        ]
        '''
        transitions.analyze_pixel_brightness_parallel_hdf5(
            hdf5_folder=os.path.join(primary_folder, "6_time_history_output", "50"),
            pixel_locations=mask,
            output_folder=os.path.join(primary_folder, "7_pixel_timing_interrogation"),
            final_output_file="time_history_transition_parallel_final.hdf5",
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
        transitions.create_timing_plots_with_pillow_hdf5(
            compiled_hdf5=os.path.join(
                primary_folder, "7_pixel_timing_interrogation", "time_history_transition_parallel_final.hdf5"
            ),
            output_folder=os.path.join(primary_folder, "7_pixel_timing_interrogation", "pixel_timing_plots"),
            source_image_folder=os.path.join(primary_folder, "3_specific_cropped_frames"),
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


if __name__ == "__main__":
    main()
    # TODO Interactive components need to happen first. Ideally a single set of dialogues for all user input.
    # Video Scrubber, Pixel location selection for time history transitions, ...
    # TODO Align Data structure with SOFAST Back END
