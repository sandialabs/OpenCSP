import os
import sys
import time
import argparse
import configparser
import ast
from pathlib import Path
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
from opencsp.app.lookback.interactive_video_info_extract_ref_pixel import interactive_video_select
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

PATH_KEYS = {"primary_folder", "og_video_path", "file_camera", "video_path", "ephem_path"}


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


def normalize_path(value):
    """
    Normalize file paths, handling UNC and drive-letter paths.
    Convert backslashes to forward slashes for consistency.
    """
    if isinstance(value, str):
        # Detect UNC path (starts with \\ or //)
        if value.startswith(('\\\\', '//')):
            # Normalize but keep UNC prefix
            p = Path(value)
            return str(p.as_posix())
        # Detect Windows drive letter path (e.g., C:\)
        elif len(value) > 1 and value[1] == ':':
            p = Path(value)
            return str(p.as_posix())
    return value


def serialize_value(value):
    """
    Convert Python objects to strings suitable for writing to ini files.
    Paths are converted to forward slashes.
    """
    if isinstance(value, (list, tuple)):
        # Convert each element recursively and format as Python literal
        return str(tuple(value)) if isinstance(value, tuple) else str(list(value))
    elif isinstance(value, bool):
        return str(value)
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # Normalize path slashes
        return normalize_path(value)
    else:
        # Fallback to string conversion
        return str(value)


def write_ini_file_from_args(args_namespace, additional_values, output_path, section_name="DEFAULT"):
    """
    Write an INI file combining argparse inputs and additional values, compatible with configparser.

    Parameters:
        args_namespace (argparse.Namespace): Parsed CLI arguments.
        additional_values (dict): Additional key-value pairs to include.
        output_path (str): Full path to write the ini file.
        section_name (str): Section name in the INI file (default: "DEFAULT").

    Returns:
        None
    """
    # Convert Namespace to dict
    args_dict = vars(args_namespace).copy()
    # Merge additional values (overwrites if keys overlap)
    combined_dict = {**args_dict, **additional_values}

    # Serialize all values to strings
    str_dict = {k: serialize_value(v) for k, v in combined_dict.items()}

    config = configparser.ConfigParser()
    config[section_name] = str_dict

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as configfile:
        config.write(configfile)

    print(f"INI file written to: {output_path}")


def fraction_to_key(values):
    """
    Convert level specifications to a 2-digit percent key string ("00".."100").

    Accepts:
      - single value or list/tuple/set
      - values may be:
          * fraction floats/ints: 0.5, 1.0, 0
          * percent values as ints/floats: 50, 95, 100
          * percent keys as strings: "50", "05", "100"
          * fraction strings: "0.5", "1", "0.05"

    Returns:
      - str if input is scalar
      - list[str] if input is list/tuple/set

    Raises:
      - ValueError if a value cannot be parsed or is out of range.
    """

    def one_to_key(v):
        # parse strings
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                raise ValueError("Empty string is not a valid level.")
            # pure digits -> interpret as percent key/value
            if s.isdigit():
                p = int(s)
                if not (0 <= p <= 100):
                    raise ValueError(f"Percent key {p} out of range [0, 100].")
                return f"{p:02d}"
            # otherwise interpret as numeric (fraction-like)
            try:
                v = float(s)
            except ValueError as e:
                raise ValueError(f"Could not parse level value {v!r}.") from e

        # parse numbers
        if isinstance(v, (int, float)):
            x = float(v)
            # Heuristic matches normalize_to_fractions:
            #   x <= 1 -> fraction
            #   x >  1 -> percent
            if x <= 1.0:
                if x < 0.0:
                    raise ValueError(f"Fraction {x} out of range [0, 1].")
                p = int(round(x * 100))
            else:
                if not (0.0 <= x <= 100.0):
                    raise ValueError(f"Percent value {x} out of range [0, 100].")
                p = int(round(x))

            # validate after rounding
            if not (0 <= p <= 100):
                raise ValueError(f"Percent key {p} out of range [0, 100].")
            return f"{p:02d}"

        raise ValueError(f"Unsupported type for level value: {type(v).__name__}")

    is_iterable = isinstance(values, (list, tuple, set))
    if is_iterable:
        return [one_to_key(v) for v in values]
    return one_to_key(values)


def normalize_to_fractions(values):
    """
    Normalize level specifications to fractions in [0, 1].

    Accepts:
      - single value or list/tuple/set
      - values may be:
          * fraction floats/ints: 0.5, 1.0, 0
          * percent keys as ints: 50, 95, 100
          * percent keys as strings: "50", "05", "100"
          * fraction strings: "0.5", "1", "0.05"

    Returns:
      - list[float]: fractions in [0, 1] (order preserved for list-like input)

    Raises:
      - ValueError if a value cannot be parsed or is out of range.
    """

    def one_to_fraction(v):
        # parse strings
        if isinstance(v, str):
            s = v.strip()
            if s == "":
                raise ValueError("Empty string is not a valid level.")
            # pure digits -> interpret as percent key (e.g., "50" => 0.5)
            if s.isdigit():
                p = int(s)
                if not (0 <= p <= 100):
                    raise ValueError(f"Percent key {p} out of range [0, 100].")
                return p / 100.0
            # otherwise interpret as numeric (fraction-like)
            try:
                v = float(s)
            except ValueError as e:
                raise ValueError(f"Could not parse level value {v!r}.") from e

        # parse numbers
        if isinstance(v, (int, float)):
            x = float(v)
            # Heuristic:
            #   x <= 1 -> already a fraction
            #   x >  1 -> treat as percent key (e.g., 50 -> 0.5)
            if x <= 1.0:
                if x < 0.0:
                    raise ValueError(f"Fraction {x} out of range [0, 1].")
                return x
            else:
                if not (0.0 <= x <= 100.0):
                    raise ValueError(f"Percent value {x} out of range [0, 100].")
                return x / 100.0

        raise ValueError(f"Unsupported type for level value: {type(v).__name__}")

    # accept scalar or iterable
    if isinstance(values, (list, tuple, set)):
        out = [one_to_fraction(v) for v in values]
    else:
        out = [one_to_fraction(values)]

    # final clamp check (no silent clamping; just validate)
    for f in out:
        if not (0.0 <= f <= 1.0):
            raise ValueError(f"Normalized fraction {f} out of range [0, 1].")

    return out


def checkpoint_has_new_levels(
    checkpoint_main_data: dict,
    cam_intensity_fractions,
    analysis_fractions,
    *,
    key_func=None,
    require_subset: bool = True,
):
    """
    Compare requested fraction levels (from args/ini) against levels recorded in a checkpoint.

    Handles checkpoint lists that may be nested, e.g. [['50','60']] or [[0.5, 0.6]].

    Returns:
        (flags, details)

        flags (dict[str, bool]):
            Per-section boolean: True means this section has NEW requested levels not present
            in the checkpoint list and therefore should run.

        details (dict[str, dict]):
            Per-section info: requested, completed, missing (all as sorted lists of level keys).
    """

    def default_key_func(x):
        def _one(v):
            # Normalize: accept "50", 50, 0.5, "0.5"
            if isinstance(v, str):
                s = v.strip()
                if s.isdigit():
                    return f"{int(s):02d}"
                try:
                    v = float(s)
                except ValueError:
                    return s  # fallback: raw string compare
            if isinstance(v, (int, float)):
                xf = float(v)
                # treat <= 1 as fraction, > 1 as percent key
                if xf <= 1.0:
                    return f"{int(round(xf * 100)):02d}"
                return f"{int(round(xf)):02d}"
            return str(v)

        if isinstance(x, (list, tuple, set)):
            return {_one(v) for v in x}
        else:
            return _one(x)

    key_func = default_key_func if key_func is None else key_func

    def _flatten(seq):
        """Flatten arbitrarily nested lists/tuples/sets (but not strings/bytes)."""
        if seq is None:
            return
        if isinstance(seq, (list, tuple, set)):
            for item in seq:
                yield from _flatten(item)
        else:
            yield seq

    def to_key_set(seq):
        """
        Convert scalars, lists, or nested-lists into a flat set of normalized level keys.
        Examples:
          "50" -> {"50"}
          ["50","60"] -> {"50","60"}
          [["50","60"]] -> {"50","60"}
          [0.5, 0.6] -> {"50","60"}
        """
        if seq is None:
            return set()

        # Treat non-iterable scalar as one item
        if not isinstance(seq, (list, tuple, set)):
            seq = [seq]

        out = set()
        for v in _flatten(seq):
            kv = key_func(v)
            # key_func may return a set if v is iterable (defensive)
            if isinstance(kv, set):
                out |= kv
            else:
                out.add(kv)
        return out

    cam_req = to_key_set(cam_intensity_fractions)
    ana_req = to_key_set(analysis_fractions)

    if require_subset and not ana_req.issubset(cam_req):
        raise ValueError(
            f"analysis_fractions levels {sorted(ana_req)} must be a subset of "
            f"cam_intensity_fractions levels {sorted(cam_req)}."
        )

    section_map = {
        "coverage_maps_levels": cam_req,
        "time_history_levels": cam_req,
        "pixel_transition_levels": ana_req,
        "pixel_transition_plot_levels": ana_req,
        "celestial_vectors_data_levels": ana_req,
        "celestial_vectors_plots_levels": ana_req,
        "lookfast_camera_adjust_levels": ana_req,
    }

    flags = {}
    details = {}

    for ckpt_key, requested in section_map.items():
        completed = to_key_set(checkpoint_main_data.get(ckpt_key, []))
        missing = requested - completed

        flags[ckpt_key] = len(missing) > 0
        details[ckpt_key] = {"requested": sorted(requested), "completed": sorted(completed), "missing": sorted(missing)}

    return flags, details


def main(args):

    if args.primary_folder is None:
        args.primary_folder = select_dir(window_title="Select Primary Output Directory")
        primary_folder = args.primary_folder
    else:
        primary_folder = args.primary_folder
        ft.create_directories_if_necessary(primary_folder)

    if args.og_video_path is None:
        args.og_video_path = select_file(
            window_title="Select Original Video to Copy and Process", file_types="*.mov *.mp4"
        )
        ft.copy_file(input_dir_body_ext=args.og_video_path, output_dir=args.primary_folder)
        og_vid_dir, og_vid_name, og_vid_ext = ft.path_components(args.og_video_path)
    elif os.path.exists(args.og_video_path):
        og_vid_dir, og_vid_name, og_vid_ext = ft.path_components(args.og_video_path)
        if ft.file_exists(input_dir_body_ext=ft.join(args.primary_folder, og_vid_name + og_vid_ext)):
            pass
        else:
            ft.copy_file(input_dir_body_ext=args.og_video_path, output_dir=args.primary_folder)
    else:
        og_vid_dir, og_vid_name, og_vid_ext = ft.path_components(args.og_video_path)

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

    for folder in data_output_folders:
        ft.create_directories_if_necessary(ft.join(primary_folder, folder))

    checkpoint_folder = ft.join(primary_folder, "0_checkpoints")
    checkpoint_main_name = "lookback_main_checkpoint.json"

    # Load checkpoint if it exists
    checkpoint_main_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_main_name)
    if checkpoint_main_data is None:
        checkpoint_main_data = {
            "Completed_Steps": [],
            "coverage_maps_levels": [],
            "time_history_levels": [],
            "pixel_transition_levels": [],
            "pixel_transition_plot_levels": [],
            "celestial_vectors_data_levels": [],
            "celestial_vectors_plots_levels": [],
            "lookfast_camera_adjust_levels": [],
        }

    start_time = time.time()
    logger.info("Code Start Time: %s", str(time.ctime(start_time)))

    # fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    fractions = args.cam_intensity_fractions
    analysis_fractions = args.analysis_fractions

    cam_level_keys = [fraction_to_key(f) for f in args.cam_intensity_fractions]
    analysis_level_keys = [fraction_to_key(f) for f in args.analysis_fractions]

    has_new, chkpt_args_diff = checkpoint_has_new_levels(
        checkpoint_main_data=checkpoint_main_data,
        cam_intensity_fractions=cam_level_keys,
        analysis_fractions=analysis_level_keys,
        require_subset=True,
    )

    celestial_database_path = args.ephem_path
    celestial_object = args.celestial_object
    video_metadata = lbt.extract_detailed_video_metadata(ft.join(primary_folder, og_vid_name + og_vid_ext))
    timezone = args.timezone
    # camera_time_shift = timedelta(hours=0, minutes=0, seconds=0)
    camera_time_shift = timedelta(args.camera_time_shift[0], args.camera_time_shift[1], args.camera_time_shift[2])
    # Define observer location Example ≈NSTTF Tower 260 Level Balcony West Side
    observer_lat = (
        args.observer_lat[0],
        args.observer_lat[1],
        args.observer_lat[2],
    )  # (degree, minute, second) negative degree for south
    observer_long = (
        args.observer_long[0],
        args.observer_long[1],
        args.observer_long[2],
    )  # (degree, minute, second) negative degree for west
    observer_elevation = args.observer_elevation
    observer_loc = (lbt.lat_long_to_decimal(observer_lat), lbt.lat_long_to_decimal(observer_long), observer_elevation)

    # Define target location Example ≈Sun Data Marker 2 In front of 5E8
    target_lat = (
        args.target_lat[0],
        args.target_lat[1],
        args.target_lat[2],
    )  # (degree, minute, second) negative degree for south
    target_long = (
        args.target_long[0],
        args.target_long[1],
        args.target_long[2],
    )  # (degree, minute, second) negative degree for west
    target_elevation = args.target_elevation
    target_loc = (lbt.lat_long_to_decimal(target_lat), lbt.lat_long_to_decimal(target_long), target_elevation)

    ##### Setting camera object

    cam = Camera.load_from_hdf(args.file_camera)
    cam_to_optic_reference_distance = args.cam_to_optic_reference_distance  # meters

    '''
    # Sofast Example Camera Intrinsic matrix
    K_intrin = np.array([[5492.064314084441, 0, 1920 / 2], [0, 5486.2706013814895, 1080 / 2], [0, 0, 1]])
    # Sofast Example Camera Distortion coefficients
    D_coeff = np.array([-0.144160742602367, 1.609744377391114, 2.503498158416561e-5, -0.001899042260179])

    cam = Camera(
        intrinsic_mat=K_intrin, distortion_coef=D_coeff, image_shape_xy=tuple[1920, 1080], name="Arbitrary_Example"
    )
    reference_pixel_key = "(500, 900)"
    '''

    if "extracted_video_frames" in checkpoint_main_data["Completed_Steps"] or args.extract_video_frames is False:
        logger.info("Skipped Already Completed Video Frame Extraction : %s", str(time.time() - start_time))
        if ft.file_exists(input_dir_body_ext=ft.join(primary_folder, "interactive_scrubber_selections.json")):
            scrubber_details = lbt.read_json(ft.join(primary_folder, "interactive_scrubber_selections.json"))
            reference_pixel_key = scrubber_details['reference_pixel']
    else:

        scrubber_details = interactive_video_select(
            video_path=ft.join(primary_folder, og_vid_name + og_vid_ext),
            dest_path=ft.join(primary_folder, "1_video_frames"),
            frame_subset_dir=ft.join(primary_folder, "3_specific_cropped_frames"),
            start_frame=args.start_frame if args.start_frame else None,
            end_frame=args.end_frame if args.end_frame else None,
            reference_pixel=args.reference_pixel if args.reference_pixel else None,
        )
        reference_pixel_key = scrubber_details['reference_pixel']
        lbt.write_json(scrubber_details, ft.join(primary_folder, "interactive_scrubber_selections.json"))

        checkpoint_main_data["Completed_Steps"].append("extracted_video_frames")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )
        logger.info("Time to Complete Video Frame Extraction: %s", str(time.time() - start_time))

        write_ini_file_from_args(args, scrubber_details, ft.join(primary_folder, "full_processing_settings.ini"))

    if ("coverage_maps" in checkpoint_main_data["Completed_Steps"] or args.coverage_maps is False) and has_new[
        "coverage_maps_levels"
    ] is False:
        logger.info("Skipped Already Completed Coverage Maps : %s", str(time.time() - start_time))
        if ft.file_exists(ft.join(primary_folder, "threshold_bmap_mask_paths.json")):
            thresh_maps_paths = lbt.read_json(ft.join(primary_folder, "threshold_bmap_mask_paths.json"))
    else:
        missing_levels = fractions
        if has_new["coverage_maps_levels"] is True:
            missing_levels = chkpt_args_diff["coverage_maps_levels"]["missing"]

        cvg_map.construct_binary_maps_parallel(
            image_folder=ft.join(primary_folder, "3_specific_cropped_frames"),
            output_folder=ft.join(primary_folder, "4_coverage_map"),
            checkpoint_folder=checkpoint_folder,
            threshold_fractions=normalize_to_fractions(missing_levels),
            batch_size=1000,  # I think I fixed memory leaks, so this could be increased.
            max_workers=4,  # I think I fixed memory leaks, so this could be increased.
            checkpoint_file="coverage_map_mp_checkpoint.json",
        )
        binary_maps = ft.files_in_directory(
            input_dir=ft.join(primary_folder, "4_coverage_map"), sort=True, files_only=True, recursive=False
        )
        thresh_maps_compiled = [f for f in binary_maps if "compiled" in f]
        thresh_maps_paths = [ft.join(primary_folder, "4_coverage_map", f) for f in thresh_maps_compiled]
        lbt.write_json(thresh_maps_paths, ft.join(primary_folder, "threshold_bmap_mask_paths.json"))

        checkpoint_main_data["Completed_Steps"].append("coverage_maps")
        checkpoint_main_data["coverage_maps_levels"].extend(missing_levels)
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )
        logger.info("Time to Complete Coverage Maps: %s", str(time.time() - start_time))

    if "accelerated_video" in checkpoint_main_data["Completed_Steps"] or args.accelerated_video is False:
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

    if ("time_history" in checkpoint_main_data["Completed_Steps"] or args.time_history is False) and has_new[
        "time_history_levels"
    ] is False:
        logger.info("Skipped Already Completed Time History: %s", str(time.time() - start_time))
    else:
        missing_levels = fractions
        if has_new["time_history_levels"] is True:
            missing_levels = chkpt_args_diff["time_history_levels"]["missing"]
        # This function is still memory hungry, room for improvement here...
        time_hist.create_binary_pixel_array_parallel_with_multiprocessing(
            image_folder_path=ft.join(primary_folder, "3_specific_cropped_frames"),
            percentages=normalize_to_fractions(missing_levels),
            output_folder=ft.join(primary_folder, "6_time_history_output"),
            checkpoint_folder=checkpoint_folder,
            batch_size=500,
            overlap=1,
            checkpoint_file="time_history_array_checkpoint_mp.json",
            num_workers=4,
        )
        checkpoint_main_data["Completed_Steps"].append("time_history")
        checkpoint_main_data["time_history_levels"].extend(missing_levels)
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )
        logger.info("Time to Complete Time History Arrays: %s", str(time.time() - start_time))

    if ("pixel_transitions" in checkpoint_main_data["Completed_Steps"] or args.pixel_transitions is False) and has_new[
        "pixel_transition_levels"
    ] is False:
        logger.info("Skipped Already Completed Transition History: %s", str(time.time() - start_time))
    else:
        missing_levels = analysis_level_keys
        if has_new["pixel_transition_levels"] is True:
            missing_levels = chkpt_args_diff["pixel_transition_levels"]["missing"]

        for mask_key in missing_levels:
            # mask_key = fraction_to_key(mask)
            mask_file_path = [fp for fp in thresh_maps_paths if mask_key in os.path.basename(fp)]

            if mask_key in checkpoint_main_data["pixel_transition_levels"]:
                logger.info("Skipping completed analysis fraction pixel transitions: %s", str(mask_key))
                continue

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

                checkpoint_main_data["pixel_transition_levels"].extend([mask_key])
                lbt.save_checkpoint(
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name=checkpoint_main_name,
                    checkpoint_data=checkpoint_main_data,
                )
                logger.info("Time to Complete Time History Transitions: %s", str(time.time() - start_time))
        checkpoint_main_data["Completed_Steps"].append("pixel_transitions")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )

    if (
        "pixel_transition_plots" in checkpoint_main_data["Completed_Steps"] or args.pixel_transition_plots is False
    ) and has_new["pixel_transition_plot_levels"] is False:
        logger.info("Skipped Already Completed Transition Plots: %s", str(time.time() - start_time))
    else:
        missing_levels = analysis_level_keys
        if "pixel_transition_plot_levels" in has_new:
            missing_levels = chkpt_args_diff["pixel_transition_plot_levels"]["missing"]

        for mask_key in missing_levels:
            # mask_key = fraction_to_key(level)

            if mask_key in checkpoint_main_data["pixel_transition_plot_levels"]:
                logger.info("Skipping completed analysis fraction pixel transition plots: %s", str(mask_key))
                continue

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

            checkpoint_main_data["pixel_transition_plot_levels"].extend([mask_key])
            lbt.save_checkpoint(
                checkpoint_folder=checkpoint_folder,
                checkpoint_file_name=checkpoint_main_name,
                checkpoint_data=checkpoint_main_data,
            )
            logger.info("Time to Complete Timing Plots: %s", str(time.time() - start_time))
        checkpoint_main_data["Completed_Steps"].append("pixel_transition_plots")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )

    if (
        "celestial_vectors_data" in checkpoint_main_data["Completed_Steps"] or args.celestial_vectors_data is False
    ) and has_new["celestial_vectors_data_levels"] is False:
        logger.info("Skipped Already Completed Celestial Vector Processing: %s", str(time.time() - start_time))
    else:
        missing_levels = analysis_level_keys
        if "celestial_vectors_data_levels" in has_new:
            missing_levels = chkpt_args_diff["celestial_vectors_data_levels"]["missing"]

        for mask_key in missing_levels:
            # mask_key = fraction_to_key(level)
            mask_file_path = [fp for fp in thresh_maps_paths if mask_key in os.path.basename(fp)]

            if mask_key in checkpoint_main_data["celestial_vectors_data_levels"]:
                logger.info("Skipping completed analysis fraction celestial vectors data: %s", str(mask_key))
                continue

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
                    ephem_path=celestial_database_path,
                )

                checkpoint_main_data["celestial_vectors_data_levels"].extend([mask_key])
                lbt.save_checkpoint(
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name=checkpoint_main_name,
                    checkpoint_data=checkpoint_main_data,
                )
                logger.info("Time to Complete Celestial Vector Calculations: %s", str(time.time() - start_time))

        checkpoint_main_data["Completed_Steps"].append("celestial_vectors_data")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )

    if (
        "celestial_vectors_plots" in checkpoint_main_data["Completed_Steps"] or args.celestial_vectors_plots is False
    ) and has_new["celestial_vectors_plots_levels"] is False:
        logger.info("Skipped Already Completed Celestial Vector Plotting: %s", str(time.time() - start_time))
    else:
        missing_levels = analysis_level_keys
        if "celestial_vectors_plots_levels" in has_new:
            missing_levels = chkpt_args_diff["celestial_vectors_plots_levels"]["missing"]

        for mask_key in missing_levels:
            # mask_key = fraction_to_key(level)
            mask_file_path = [fp for fp in thresh_maps_paths if mask_key in os.path.basename(fp)]

            if mask_key in checkpoint_main_data["celestial_vectors_plots_levels"]:
                logger.info("Skipping completed analysis fraction celestial vectors plots: %s", str(mask_key))
                continue

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

                checkpoint_main_data["celestial_vectors_plots_levels"].extend([mask_key])
                lbt.save_checkpoint(
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name=checkpoint_main_name,
                    checkpoint_data=checkpoint_main_data,
                )
                logger.info("Time to Complete Celestial Vector Plots: %s", str(time.time() - start_time))

        checkpoint_main_data["Completed_Steps"].append("celestial_vectors_plots")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )

    if (
        "lookfast_camera_adjust" in checkpoint_main_data["Completed_Steps"] or args.lookfast_camera_adjust is False
    ) and has_new["lookfast_camera_adjust_levels"] is False:
        logger.info("Skipped Already Completed Lookfast Camera Adjustments: %s", str(time.time() - start_time))
    else:
        missing_levels = analysis_level_keys
        if "lookfast_camera_adjust_levels" in has_new:
            missing_levels = chkpt_args_diff["lookfast_camera_adjust_levels"]["missing"]

        for mask_key in missing_levels:
            # mask_key = fraction_to_key(level)
            mask_file_path = [fp for fp in thresh_maps_paths if mask_key in os.path.basename(fp)]

            if mask_key in checkpoint_main_data["lookfast_camera_adjust_levels"]:
                logger.info("Skipping completed analysis fraction lookfast camera adjust levels: %s", str(mask_key))
                continue

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
                    output_directory=ft.join(primary_folder, "9_sofast_data_compare", mask_key),
                    checkpoint_directory=checkpoint_folder,
                    checkpoint_file="rotate_translate_camera_adjust_checkpoint.json",
                )

                checkpoint_main_data["lookfast_camera_adjust_levels"].extend([mask_key])
                lbt.save_checkpoint(
                    checkpoint_folder=checkpoint_folder,
                    checkpoint_file_name=checkpoint_main_name,
                    checkpoint_data=checkpoint_main_data,
                )
                logger.info(
                    "Time to Complete Rotate/Translate Adjustment Calculations: %s", str(time.time() - start_time)
                )

        checkpoint_main_data["Completed_Steps"].append("lookfast_camera_adjust")
        lbt.save_checkpoint(
            checkpoint_folder=checkpoint_folder,
            checkpoint_file_name=checkpoint_main_name,
            checkpoint_data=checkpoint_main_data,
        )

    print("Finished LookFast Processing Script")
    print(time.strftime("%H:%M:%S", time.localtime()))


def parse_value(value):
    """
    Parse a string value from the ini file into a Python object if possible.
    If parsing fails, return the original string.
    """
    try:
        # Attempt to parse lists, tuples, numbers, bools, etc.
        parsed = ast.literal_eval(value)
        return parsed
    except (ValueError, SyntaxError):
        # Return as string if not a Python literal
        return value


def parse_config(config_file_path):
    config = configparser.ConfigParser()
    read_ok = config.read(config_file_path)
    if not read_ok:
        raise FileNotFoundError(f"Could not read config file: {config_file_path}")

    out = {}
    for section in config.sections():
        for key, value in config.items(section):
            v = parse_value(value)
            if key in PATH_KEYS and isinstance(v, str):
                v = normalize_path(v)
            out[key] = v

    # also include [DEFAULT] if you use it
    for key, value in config.defaults().items():
        if key not in out:
            v = parse_value(value)
            if key in PATH_KEYS and isinstance(v, str):
                v = normalize_path(v)
            out[key] = v

    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog=Path(__file__).stem,
        description='Analyze LookFast Source Video, image processing, celestial analysis, and analysis plots custom to Lookfast and through Sofast standard plot output.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "-s",
        "--settings",
        dest="settings_path",
        default=None,
        help="Settings file full path defining run parameters (input/output directories, etc).",
    )
    p.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Not Implemented Yet...Output detailed information reporting run progress and calculations.",
    )

    # Define EVERYTHING main() expects:
    # File Paths
    p.add_argument(
        "--primary-folder",
        dest="primary_folder",
        default=None,
        help="Directory location where original video and all output will be organized into including checkpoint files.",
    )
    p.add_argument(
        "--og-video-path",
        dest="og_video_path",
        default=None,
        help="Original video data path, all analysis will be done on an automatically generated copy of this video.",
    )
    p.add_argument(
        "-cam",
        "--file-camera",
        dest="file_camera",
        default=None,
        help="File path of the camera definition within OpenCSP camera calibration file conventions.",
    )
    p.add_argument(
        "-video-path",
        "--analysis-video-path",
        dest="video_path",
        default=None,
        help="File path of copied video, may be automatically added to ini file when run interactively. Not necessary to specify.",
    )
    p.add_argument(
        "-eph_path",
        "--ephemeris_database_path",
        dest="ephem_path",
        default=None,
        help="File path of de430t.bsp ephemeris file for celestial body references.",
    )

    p.add_argument(
        "-ifile",
        "--interactive-file_name",
        dest="ui_input_file_name",
        default="interactive_selections.json",
        help="Name of json file containing interactive file selection information.",
    )
    p.add_argument(
        "-sat-levels",
        "--cam-intensity-fractions",
        dest="cam_intensity_fractions",
        default=None,
        help="List of fractions ranging from 0.0 (unsaturated) to 1.0 (saturated) for camera pixel saturation thresholding for processing.",
    )
    p.add_argument(
        "-ana-levels",
        "--analysis-fractions",
        dest="analysis_fractions",
        default=None,
        help="Sublist of fractions as listed in '--cam-intensity-fractions' that will be caried forward for analysis after basic pixel thresholding. This list must contain at least one identical fraction as '--cam-intensity-fractions'.",
    )
    p.add_argument(
        "-astro-body",
        "--celestial-object",
        dest="celestial_object",
        default=None,
        help="Celestial body for analysis, currently supported are 'sun' and 'moon'.",
    )
    p.add_argument(
        "-tz",
        "--timezone",
        dest="timezone",
        default=None,
        help="timezone of location of data collection. 'America/Denver' for analysis at NSTTF.",
    )
    p.add_argument(
        "-c-shift",
        "--cam-time-offset",
        dest="camera_time_shift",
        default=None,
        help="Tuple of (hours, minutes, seconds) to shift video creation time to align with real world time. Negative for shifting backwards.",
    )
    p.add_argument(
        "-ref-dist",
        "--cam-optic_distance",
        dest="cam_to_optic_reference_distance",
        default=None,
        help="Reference distance between the camera and mirror system as close to mirror reference pixel as possible.",
    )
    p.add_argument(
        "-obs-lat",
        "--observer-latitude",
        dest="observer_lat",
        default=None,
        help="Tuple of (deg, min, sec) latitude for the observer (camera) location",
    )
    p.add_argument(
        "-obs-long",
        "--observer-longitude",
        dest="observer_long",
        default=None,
        help="Tuple of (deg, min, sec) longitude for the observer (camera) location",
    )
    p.add_argument(
        "-obs-ele",
        "--observer-elevation",
        dest="observer_elevation",
        default=None,
        help="Elevation above sea level for the observer (camera) location",
    )
    p.add_argument(
        "-tgt-lat",
        "--target-latitude",
        dest="target_lat",
        default=None,
        help="Tuple of (deg, min, sec) latitude for the target (mirror) location",
    )
    p.add_argument(
        "-tgt-long",
        "--target-longitude",
        dest="target_long",
        default=None,
        help="Tuple of (deg, min, sec) longitude for the target (mirror) location",
    )
    p.add_argument(
        "-tgt-ele",
        "--target-elevation",
        dest="target_elevation",
        default=None,
        help="Elevation above sea level for the target (mirror) location",
    )
    p.add_argument(
        "-startf",
        "--start-frame",
        dest="start_frame",
        default=None,
        help="Integer frame number to start the analysis for pixel behavior and celetial vectors.",
    )
    p.add_argument(
        "-endf",
        "--end-frame",
        dest="end_frame",
        default=None,
        help="Integer frame number to end the analysis for pixel behavior and celetial vectors.",
    )
    p.add_argument(
        "-px-ref",
        "--reference-pixel",
        dest="reference_pixel",
        default=None,
        help="Reference pixel closest to measured point with data to align coordinate systems. Input as (row, column) which is measured with the origin at the top left corner of the image and row increasing going down and column increasing going to the right.",
    )

    # code sections to run
    p.add_argument(
        "--run-extract-frames",
        dest="extract_video_frames",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to extract all frames from copy of original video.",
    )
    p.add_argument(
        "--run-coverage-maps",
        dest="coverage_maps",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to produce coverage maps of video frames at specified intensity fractions. Helpful for understanding the extent of data avaliability at different levels of saturation.",
    )
    p.add_argument(
        "--run-accelerated-video",
        dest="accelerated_video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to create an accelerated video of test for presentations and qualitative analysis of data quality.",
    )
    p.add_argument(
        "--run-time-history",
        dest="time_history",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to create time history structure for pixels within primary mask region for all data analysis.",
    )
    p.add_argument(
        "--run-pixel-transitions",
        dest="pixel_transitions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to create dictionary of transitions for pixels within the mask region.",
    )
    p.add_argument(
        "--run-pixel-transition-plots",
        dest="pixel_transition_plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to create timing plots for all pixels within mask region.",
    )
    p.add_argument(
        "--run-celestial-vectors",
        dest="celestial_vectors_data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to create celestial vector set of solutions for each pixel's timing data.",
    )
    p.add_argument(
        "--run-celestial-vector-plots",
        dest="celestial_vectors_plots",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to create plots for each pixel's celestial analysis.",
    )
    p.add_argument(
        "--run-lookfast-camera-adjustments",
        dest="lookfast_camera_adjust",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Processing section to apply camera model, align coordinate systems, align data with OpenCSP conventions, and plot data.",
    )

    return p


def validate_args(args):
    required = [
        "primary_folder",
        "og_video_path",
        "file_camera",
        "cam_intensity_fractions",
        "analysis_fractions",
        "celestial_object",
        "timezone",
        "observer_lat",
        "observer_long",
        "observer_elevation",
        "target_lat",
        "target_long",
        "target_elevation",
        "cam_to_optic_reference_distance",
    ]

    if any((f < 0 or f > 1) for f in args.cam_intensity_fractions):
        raise SystemExit("cam_intensity_fractions must be in [0,1].")

    if not set(args.analysis_fractions).issubset(set(args.cam_intensity_fractions)):
        raise SystemExit("analysis_fractions must be a subset of cam_intensity_fractions.")

    for name in ("observer_lat", "observer_long", "target_lat", "target_long"):
        t = getattr(args, name)
        if not (isinstance(t, (list, tuple)) and len(t) == 3):
            raise SystemExit(f"{name} must be a 3-tuple like (deg,min,sec).")

    missing = [k for k in required if getattr(args, k, None) in (None, "")]
    if missing:
        raise SystemExit(f"Missing required settings: {missing}")


if __name__ == "__main__":
    # Build the full parser (should include ALL arguments main() expects)
    parser = build_parser()

    # Pass 1: parse only to learn whether a settings file was provided
    pre_args, _unknown = parser.parse_known_args()

    if pre_args.settings_path is None:
        if not sys.stdin.isatty():
            raise SystemExit("Error: --settings_path (-s) is required when running non-interactively.")

        settings_file_path = select_file(
            window_title="Select a Setting.ini file for LookFast Processing", file_types="*.ini"
        )
        if not settings_file_path:
            raise SystemExit("A settings file must be specified.")

        pre_args.settings_path = settings_file_path

    # Load INI and set as defaults (so CLI can override)
    ini_settings = parse_config(pre_args.settings_path)
    parser.set_defaults(**ini_settings)

    # Pass 2: parse again with defaults populated from INI
    args = parser.parse_args()
    args.settings_path = pre_args.settings_path  # ensure it’s always present/consistent

    # Validate and run
    validate_args(args)

    main(args)
