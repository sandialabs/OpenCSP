import re
import os
import json
import gzip
import time
import subprocess
from multiprocessing import Pool, Manager
from opencsp.common.lib.tool.log_tools import multiprocessing_logger
import opencsp.app.lookback.lookback_tools as lbt
from logging import DEBUG, ERROR

# import warnings
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
import skyfield.api as skf
import numpy as np
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd, "error_logs"),
    log_file_name="error_log_celestial_vectors_debug.txt",
    log_type=ERROR,
)


def extract_video_metadata_exiftool(video_path):
    """
    Extracts the media creation date, frame rate, and duration from a video file's metadata using exiftool.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary containing:
            - 'creation_date': Media creation date and time (if available).
            - 'frame_rate': Frame rate of the video (frames per second, if available).
            - 'duration': Duration of the video (seconds, if available).
    """
    try:
        # Run exiftool to extract metadata
        result = subprocess.run(
            [
                "exiftool",
                "-CreateDate",
                "-MediaCreateDate",
                "-DateTimeOriginal",
                "-VideoFrameRate",
                "-Duration",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        # Initialize metadata dictionary
        metadata = {'creation_date': None, 'frame_rate': None, 'duration': None}

        # Parse the output to find relevant metadata
        for line in result.stdout.splitlines():
            if "Create Date" in line or "Media Create Date" in line or "Date/Time Original" in line:
                metadata['creation_date'] = line.split(": ", 1)[1].strip()
            elif "Video Frame Rate" in line:
                metadata['frame_rate'] = float(
                    line.split(": ", 1)[1].strip().split(" ")[0]
                )  # Extract frame rate as float
            elif "Duration" in line:
                duration_str = line.split(": ", 1)[1].strip()
                # Convert duration to seconds (e.g., "0:01:23.456" -> 83.456 seconds)
                parts = duration_str.split(":")
                if len(parts) == 3:  # Format is hours:minutes:seconds
                    hours, minutes, seconds = map(float, parts)
                    metadata['duration'] = hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # Format is minutes:seconds
                    minutes, seconds = map(float, parts)
                    metadata['duration'] = minutes * 60 + seconds

        return metadata
    except Exception:
        logger.error("extract_video_metadata_exiftool Error", exc_info=True)
        return None


def calculate_vectors_celestial_only(celestial_object_name, target_location, observation_time):
    """
    Calculates vectors from a celestial object to a target point and from the target point to an observer.

    Parameters:
        celestial_object_name (str): Name of the celestial object (e.g., "moon", "sun").
        target_location (tuple): Latitude, longitude, and elevation of the target point (degrees, meters).
        observation_time (tuple): Observation time as (year, month, day, hour, minute, second).

    Returns:
        dict: A dictionary containing the vectors:
            - "celestial_to_target": Vector from the celestial object to the target point (normalized).
            - "target_to_observer": Vector from the target point to the observer (normalized).
    """
    # Load ephemeris data (DE430 dataset)
    eph = skf.load("de430t.bsp")  # DE430 ephemeris file

    # Define celestial object
    celestial_object = eph[celestial_object_name]

    # Define target location
    target = skf.Topos(
        latitude_degrees=target_location[0], longitude_degrees=target_location[1], elevation_m=target_location[2]
    )

    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, skf.Time):
        pass
    elif isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)

    # Calculate position of the celestial object relative to the target
    earth = eph["earth"]
    target_position_vec = earth + target
    target_position_bary = target_position_vec.at(observation_time)
    target_obsv_celest = target_position_bary.observe(celestial_object)
    target_obsv_celest_apparent = target_obsv_celest.apparent()

    celestial_to_target = target_position_vec.at(observation_time).observe(celestial_object).position.km
    celestial_to_target_normalized = celestial_to_target / np.linalg.norm(celestial_to_target)

    earth_target_bary = earth.at(observation_time)
    earth_obsv_target = earth_target_bary.observe(target_position_vec)
    earth_obsv_target_apparent = earth_obsv_target.apparent()

    NSTTF_coord_toca = [
        target_obsv_celest_apparent.frame_xyz(target).km[1],
        target_obsv_celest_apparent.frame_xyz(target).km[0],
        target_obsv_celest_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_eota = [
        earth_obsv_target_apparent.frame_xyz(target).km[1],
        earth_obsv_target_apparent.frame_xyz(target).km[0],
        earth_obsv_target_apparent.frame_xyz(target).km[2],
    ]
    # Calculate angular size in radians
    distance_to_celestial_object = target_obsv_celest_apparent.distance().km  # Distance in kilometers

    if celestial_object_name.lower() == 'moon':
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon
    elif celestial_object_name.lower() == 'sun':
        radius_of_celestial_object = 696340  # Radius in kilometers of the Sun
    else:
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon

    angular_size_radians = 2 * np.arctan(radius_of_celestial_object / distance_to_celestial_object)

    # Apparent function returns a tuple with (Altitude, Azimuth, and Distance)
    # Altitude measures the angle above or below the horizon. The zenith is at +90°, an object on the horizon’s great circle is at 0°, and the nadir beneath your feet is at −90°.
    # Azimuth measures the angle around the sky from the north pole: 0° means exactly north, 90° is east, 180° is south, and 270° is west.
    # "earth_to_target_spherical": earth_obsv_target_apparent.altaz()
    # Cartesian for horizonal skyfield coordinates is Left Handed {x points north, y points east, z points to zenith}
    # NSTTF coordinates is Right Handed {x points east, y points north, z points to zenith}
    return {
        "celestial_to_target": celestial_to_target_normalized,
        "cel_to_target_cartesian": NSTTF_coord_toca / np.linalg.norm(NSTTF_coord_toca),
        "earth_to_target_cartesian": NSTTF_coord_eota / np.linalg.norm(NSTTF_coord_eota),
        "angular_size_radians": angular_size_radians,
    }


def calculate_vectors_celestial_observer(celestial_object_name, target_location, observer_location, observation_time):
    """
    Calculates vectors from a celestial object to a target point and from the target point to an observer.

    Parameters:
        celestial_object_name (str): Name of the celestial object (e.g., "moon", "sun").
        target_location (tuple): Latitude, longitude, and elevation of the target point (degrees, meters).
        observation_time (tuple): Observation time as (year, month, day, hour, minute, second).

    Returns:
        dict: A dictionary containing the vectors:
            - "celestial_to_target": Vector from the celestial object to the target point (normalized).
            - "target_to_observer": Vector from the target point to the observer (normalized).
    """
    # Load ephemeris data (DE430 dataset)
    eph = skf.load("de430t.bsp")  # DE430 ephemeris file

    # Define celestial object
    celestial_object = eph[celestial_object_name]

    # Define target location
    target = skf.Topos(
        latitude_degrees=target_location[0], longitude_degrees=target_location[1], elevation_m=target_location[2]
    )
    # Define observer location
    observer = skf.Topos(
        latitude_degrees=observer_location[0], longitude_degrees=observer_location[1], elevation_m=observer_location[2]
    )

    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, skf.Time):
        pass
    elif isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)

    # Calculate position of the celestial object relative to the target
    earth = eph["earth"]
    target_position_vec = earth + target
    target_position_bary = target_position_vec.at(observation_time)
    target_obsv_celest = target_position_bary.observe(celestial_object)
    target_obsv_celest_apparent = target_obsv_celest.apparent()

    celestial_to_target = target_position_vec.at(observation_time).observe(celestial_object).position.km
    celestial_to_target_normalized = celestial_to_target / np.linalg.norm(celestial_to_target)

    # Calculate position of the target relative to the observer
    observer_position = earth + observer
    target_to_observer = observer_position.at(observation_time).observe(target_position_vec)

    earth_target_bary = earth.at(observation_time)
    earth_obsv_target = earth_target_bary.observe(target_position_vec)
    earth_obsv_target_apparent = earth_obsv_target.apparent()

    NSTTF_coord_toca = [
        target_obsv_celest_apparent.frame_xyz(target).km[1],
        target_obsv_celest_apparent.frame_xyz(target).km[0],
        target_obsv_celest_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_eota = [
        earth_obsv_target_apparent.frame_xyz(target).km[1],
        earth_obsv_target_apparent.frame_xyz(target).km[0],
        earth_obsv_target_apparent.frame_xyz(target).km[2],
    ]
    NSTTF_coord_otta = [
        target_to_observer.frame_xyz(target).km[1],
        target_to_observer.frame_xyz(target).km[0] * -1,
        target_to_observer.frame_xyz(target).km[2] * -1,
    ]
    # Calculate angular size in radians
    distance_to_celestial_object = target_obsv_celest_apparent.distance().km  # Distance in kilometers

    if celestial_object_name.lower() == 'moon':
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon
    elif celestial_object_name.lower() == 'sun':
        radius_of_celestial_object = 1391400 / 2  # Radius in kilometers of the Sun
    else:
        radius_of_celestial_object = 1737.4  # Radius in kilometers of the Moon

    angular_size_radians = 2 * np.arctan(radius_of_celestial_object / distance_to_celestial_object)

    # Apparent function returns a tuple with (Altitude, Azimuth, and Distance)
    # Altitude measures the angle above or below the horizon. The zenith is at +90°, an object on the horizon’s great circle is at 0°, and the nadir beneath your feet is at −90°.
    # Azimuth measures the angle around the sky from the north pole: 0° means exactly north, 90° is east, 180° is south, and 270° is west.
    # "earth_to_target_spherical": earth_obsv_target_apparent.altaz()
    # Cartesian for horizonal skyfield coordinates is Left Handed {x points north, y points east, z points to zenith}
    # NSTTF coordinates is Right Handed {x points east, y points north, z points to zenith}
    return {
        "celestial_to_target": celestial_to_target_normalized,
        "cel_to_target_cartesian": NSTTF_coord_toca / np.linalg.norm(NSTTF_coord_toca),
        "earth_to_target_cartesian": NSTTF_coord_eota / np.linalg.norm(NSTTF_coord_eota),
        "target_to_observer": NSTTF_coord_otta / np.linalg.norm(NSTTF_coord_otta),
        "angular_size_radians": angular_size_radians,
    }


def extract_pixel_timing_and_celestial_vectors(
    celestial_object_name,
    target_location,
    observer_location,
    camera_time_shift,
    data_location,
    video_metadata,
    output_folder_vector,
    output_json_name,
    checkpoint_folder,
    checkpoint_file,
):
    video_start_time = datetime.strptime(video_metadata['creation_date'], "%Y:%m:%d %H:%M:%S")
    abq_tz = ZoneInfo("America/Denver")
    video_start_time = video_start_time.replace(tzinfo=abq_tz)
    # Read in pixel timing information
    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_pixels": [], "pixels_with_data": [], "pixels_without_data": []}
    data = lbt.read_compressed_json(data_location)
    # Create a tqdm progress bar outside the loop
    progress_bar = tqdm(total=len(data), desc="Calculating Pixel Vector Information")

    for pixel, transitions in data.items():
        if pixel in checkpoint_data["processed_pixels"]:
            # print(f"Skipping already processed pixel {pixel}.")
            progress_bar.update(1)
            continue

        frame_diff = 0
        if len(data[pixel]) == 0:
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_without_data"].append(pixel)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
            progress_bar.update(1)
            continue
        elif len(data[pixel]) < 2:
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_without_data"].append(pixel)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
            progress_bar.update(1)
            continue
        elif len(data[pixel]) == 2:
            frame_range = []
            for transition in transitions:
                if transition["transition"] == "bright":
                    frame_range.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
                elif transition["transition"] == "dark":
                    frame_range.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))
            frame_diff = frame_range[1][1] - frame_range[0][1]
        elif len(data[pixel]) > 2:
            frame_range_all = []
            for transition in transitions:
                if transition["transition"] == "bright":
                    frame_range_all.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
                elif transition["transition"] == "dark":
                    frame_range_all.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))
            frame_range, frame_diff = maximum_frame_range(frame_ranges=frame_range_all)

        elapsed_time_bright = frame_diff / video_metadata['frame_rate']
        elapsed_time_start = frame_range[0][1] / video_metadata['frame_rate']

        bright_start_time = video_start_time + timedelta(seconds=elapsed_time_start)
        bright_end_time = (
            video_start_time + timedelta(seconds=elapsed_time_start) + timedelta(seconds=elapsed_time_bright)
        )

        obsv_time_utc_start = define_observation_time(bright_start_time, camera_time_shift)
        obsv_time_utc_end = define_observation_time(bright_end_time, camera_time_shift)

        start_vector = calculate_vectors_celestial_observer(
            celestial_object_name=celestial_object_name,
            target_location=target_location,
            observer_location=observer_location,
            observation_time=obsv_time_utc_start,
        )
        end_vector = calculate_vectors_celestial_observer(
            celestial_object_name=celestial_object_name,
            target_location=target_location,
            observer_location=observer_location,
            observation_time=obsv_time_utc_end,
        )
        points = np.array([[0, 0, 0], start_vector['cel_to_target_cartesian'], end_vector['cel_to_target_cartesian']])
        radii = np.array([1, start_vector['angular_size_radians'] / 2, end_vector['angular_size_radians'] / 2])
        try:
            intersection_1, intersection_2 = trilaterate(points, radii, raise_on_no_solution=True)
        except ValueError:
            logger.debug("extract_pixel_timing_and_celestial_vectors Error pixel %s", pixel, exc_info=True)
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_without_data"].append(pixel)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
            progress_bar.update(1)
            continue

        observer_vec, slope_1, slope_2 = calculate_slope(
            start_vector['target_to_observer'], end_vector['target_to_observer'], intersection_1, intersection_2
        )

        data[pixel].append(
            {
                "start_time_Local": obsv_time_utc_start.astimezone(abq_tz),
                "start_time_UTC": obsv_time_utc_start.utc_datetime(),
                "start_vector": start_vector,
                "end_time_Local": obsv_time_utc_end.astimezone(abq_tz),
                "end_time_UTC": obsv_time_utc_end.utc_datetime(),
                "end_vector": end_vector,
                "intersection_1": intersection_1,
                "intersection_2": intersection_2,
                "slope_1": slope_1,
                "slope_2": slope_2,
                "observer_vector": observer_vec,
            }
        )
        checkpoint_data["processed_pixels"].append(pixel)
        checkpoint_data["pixels_with_data"].append(pixel)
        lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        lbt.write_compressed_json(
            data=data, file_path=os.path.join(output_folder_vector, output_json_name), print_path=False
        )
        # lbt.write_json(data=data, file_path=os.path.join(output_folder_vector, output_json_name[:-3]))
        progress_bar.update(1)
    progress_bar.close()


def process_pixel(args):
    (
        pixel,
        transitions,
        video_metadata,
        video_start_time,
        camera_time_shift,
        celestial_object_name,
        target_location,
        observer_location,
        abq_tz,
    ) = args

    checkpoint_data = {"processed_pixels": [], "pixels_with_data": [], "pixels_without_data": []}

    frame_diff = 0
    if len(transitions) == 0 or len(transitions) < 2:
        checkpoint_data["processed_pixels"].append(pixel)
        checkpoint_data["pixels_without_data"].append(pixel)
        # lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        return pixel, None, checkpoint_data

    if len(transitions) == 2:
        frame_range = []
        for transition in transitions:
            if transition["transition"] == "bright":
                frame_range.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
            elif transition["transition"] == "dark":
                frame_range.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))
        frame_diff = frame_range[1][1] - frame_range[0][1]
    elif len(transitions) > 2:
        frame_range_all = []
        for transition in transitions:
            if transition["transition"] == "bright":
                frame_range_all.append(tuple((1, frame_number_from_img_name(transition['to_frame']))))
            elif transition["transition"] == "dark":
                frame_range_all.append(tuple((0, frame_number_from_img_name(transition['to_frame']))))
        frame_range, frame_diff = maximum_frame_range(frame_ranges=frame_range_all)

    elapsed_time_bright = frame_diff / video_metadata['frame_rate']
    elapsed_time_start = frame_range[0][1] / video_metadata['frame_rate']

    bright_start_time = video_start_time + timedelta(seconds=elapsed_time_start)
    bright_end_time = video_start_time + timedelta(seconds=elapsed_time_start) + timedelta(seconds=elapsed_time_bright)

    obsv_time_utc_start = define_observation_time(bright_start_time, camera_time_shift)
    obsv_time_utc_end = define_observation_time(bright_end_time, camera_time_shift)

    start_vector = calculate_vectors_celestial_observer(
        celestial_object_name=celestial_object_name,
        target_location=target_location,
        observer_location=observer_location,
        observation_time=obsv_time_utc_start,
    )
    end_vector = calculate_vectors_celestial_observer(
        celestial_object_name=celestial_object_name,
        target_location=target_location,
        observer_location=observer_location,
        observation_time=obsv_time_utc_end,
    )
    points = np.array([[0, 0, 0], start_vector['cel_to_target_cartesian'], end_vector['cel_to_target_cartesian']])
    radii = np.array([1, start_vector['angular_size_radians'] / 2, end_vector['angular_size_radians'] / 2])

    pixel_data = {
        "start_time_Local": obsv_time_utc_start.astimezone(abq_tz),
        "start_time_UTC": obsv_time_utc_start.utc_datetime(),
        "start_vector": start_vector,
        "end_time_Local": obsv_time_utc_end.astimezone(abq_tz),
        "end_time_UTC": obsv_time_utc_end.utc_datetime(),
        "end_vector": end_vector,
        "intersection_1": [],
        "intersection_2": [],
        "slope_1": [],
        "slope_2": [],
        "observer_vector": (start_vector['target_to_observer'] + end_vector['target_to_observer']) / 2,
    }

    try:
        intersection_1, intersection_2 = trilaterate(points, radii, raise_on_no_solution=False)

        observer_vec, slope_1, slope_2 = safe_calculate_slope(
            start_vector['target_to_observer'], end_vector['target_to_observer'], intersection_1, intersection_2
        )

        pixel_data = {
            "start_time_Local": obsv_time_utc_start.astimezone(abq_tz),
            "start_time_UTC": obsv_time_utc_start.utc_datetime(),
            "start_vector": start_vector,
            "end_time_Local": obsv_time_utc_end.astimezone(abq_tz),
            "end_time_UTC": obsv_time_utc_end.utc_datetime(),
            "end_vector": end_vector,
            "intersection_1": intersection_1,
            "intersection_2": intersection_2,
            "slope_1": slope_1,
            "slope_2": slope_2,
            "observer_vector": observer_vec,
        }

        checkpoint_data["processed_pixels"].append(pixel)
        checkpoint_data["pixels_with_data"].append(pixel)
        # lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
    # except ValueError:
    except Exception:
        logger.debug("process_pixel Error pixel %s", pixel, exc_info=True)

        try:
            observer_vec, slope_1, slope_2 = safe_calculate_slope(
                start_vector['target_to_observer'], end_vector['target_to_observer'], intersection_1, intersection_2
            )

            pixel_data = {
                "start_time_Local": obsv_time_utc_start.astimezone(abq_tz),
                "start_time_UTC": obsv_time_utc_start.utc_datetime(),
                "start_vector": start_vector,
                "end_time_Local": obsv_time_utc_end.astimezone(abq_tz),
                "end_time_UTC": obsv_time_utc_end.utc_datetime(),
                "end_vector": end_vector,
                "intersection_1": intersection_1,
                "intersection_2": intersection_2,
                "slope_1": slope_1,
                "slope_2": slope_2,
                "observer_vector": observer_vec,
            }
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_with_data"].append(pixel)
        except Exception:
            logger.debug("process_pixel Error pixel %s", pixel, exc_info=True)
            checkpoint_data["processed_pixels"].append(pixel)
            checkpoint_data["pixels_without_data"].append(pixel)
            # lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
            return pixel, pixel_data, checkpoint_data

    return pixel, pixel_data, checkpoint_data


def extract_pixel_timing_and_celestial_vectors_parallel(
    celestial_object_name,
    target_location,
    observer_location,
    camera_time_shift,
    data_location,
    video_metadata,
    output_folder_vector,
    output_json_name,
    checkpoint_folder,
    checkpoint_file,
    batch_size=5000,  # Number of pixels to process in each batch
):
    if os.path.exists(os.path.join(output_folder_vector, output_json_name)):
        logger.info("Compiled output file already exists, skipping code block")
        return

    video_start_time = datetime.strptime(video_metadata['creation_date'], "%Y:%m:%d %H:%M:%S")
    abq_tz = ZoneInfo("America/Denver")
    video_start_time = video_start_time.replace(tzinfo=abq_tz)

    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_pixels": [], "pixels_with_data": [], "pixels_without_data": []}

    # Ensure the output folder exists
    os.makedirs(output_folder_vector, exist_ok=True)

    # Read compressed JSON file in chunks
    with gzip.open(data_location, 'rt', encoding='utf-8') as f:
        data = json.load(f)

    # Convert data to a list of items for batching
    data_items = list(data.items())
    total_pixels = len(data_items)

    # Process data in batches
    for batch_start in range(0, total_pixels, batch_size):
        batch_end = min(batch_start + batch_size, total_pixels)
        batch_data = data_items[batch_start:batch_end]
        # Filter batch_data to exclude already processed pixels
        filtered_batch_data = [
            (pixel, transitions)
            for pixel, transitions in batch_data
            if pixel not in checkpoint_data["processed_pixels"]
        ]
        # Use multiprocessing to parallelize the pixel processing
        with Manager() as manager:
            args_list = [
                (
                    pixel,
                    transitions,
                    video_metadata,
                    video_start_time,
                    camera_time_shift,
                    celestial_object_name,
                    target_location,
                    observer_location,
                    abq_tz,
                )
                for pixel, transitions in filtered_batch_data
            ]

            # with Pool(processes=1) as pool:  # Maybe look into hyperthreading
            with Pool(processes=os.cpu_count()) as pool:  # Use all available CPU cores
                results = list(
                    tqdm(
                        pool.imap(process_pixel, args_list),
                        total=len(filtered_batch_data),
                        desc=f"Processing Batch {batch_start}-{batch_end}",
                    )
                )

            # Aggregate checkpoint data
            for _, _, pixel_checkpoint_data in results:
                ppixel = pixel_checkpoint_data["processed_pixels"]
                dpixel = pixel_checkpoint_data["pixels_with_data"]
                wpixel = pixel_checkpoint_data["pixels_without_data"]
                checkpoint_data["processed_pixels"].append(ppixel[0])
                if dpixel:
                    checkpoint_data["pixels_with_data"].append(dpixel[0])
                if wpixel:
                    checkpoint_data["pixels_without_data"].append(wpixel[0])

            # Save checkpoint data serially after each batch
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)

            # Update the data with processed results
            for pixel, pixel_data, _ in results:
                if pixel_data is not None:
                    data[pixel] = pixel_data

            # Save intermediate results to avoid data loss
            if results:
                lbt.write_compressed_json(
                    data=data,
                    file_path=os.path.join(
                        output_folder_vector, f"{output_json_name}_batch_{batch_start}_{batch_end}.json.gz"
                    ),
                    print_path=False,
                )

    # Save final results
    if os.path.exists(os.path.join(output_folder_vector, output_json_name)):
        pass
    else:
        lbt.write_compressed_json(
            data=data, file_path=os.path.join(output_folder_vector, output_json_name), print_path=True
        )


def calculate_slope(observer_vec_start, observer_vec_end, inter_1, inter_2):
    observer_vec = (observer_vec_start + observer_vec_end) / 2
    slope_1 = (observer_vec + inter_1) / np.linalg.norm(observer_vec + inter_1)
    slope_2 = (observer_vec + inter_2) / np.linalg.norm(observer_vec + inter_2)
    return observer_vec, slope_1, slope_2


def safe_calculate_slope(start_vector, end_vector, intersection_1, intersection_2):
    """
    Wrapper for calculate_slope with error handling and logging.
    """
    try:
        # Call the calculate_slope function
        observer_vec, slope_1, slope_2 = calculate_slope(start_vector, end_vector, intersection_1, intersection_2)
        return observer_vec, slope_1, slope_2
    except KeyError as e:
        # Handle missing keys in dictionaries
        logger.debug("KeyError in calculate_slope: %s", e, exc_info=True)
        return None, None, None
    except TypeError as e:
        # Handle type-related issues (e.g., NoneType or invalid types)
        logger.debug("TypeError in calculate_slope: %s", e, exc_info=True)
        return None, None, None
    except Exception as e:
        # Catch any other unexpected errors
        logger.debug("Unexpected error in calculate_slope: %s", e, exc_info=True)
        return None, None, None


def plotting_pixel_transition_vectors(pixel, vector_dict, celestial_object, output_folder):
    start_vector = vector_dict['start_vector']
    end_vector = vector_dict['end_vector']
    local_time_start = vector_dict['start_time_Local']
    local_time_end = vector_dict['end_time_Local']
    inter_1 = vector_dict['intersection_1']
    inter_2 = vector_dict['intersection_2']
    observer_vec = vector_dict['observer_vector']
    # plotting vectors
    # Create a the top half of a unit sphere for visualization of the horizon
    u = np.linspace(np.pi, 2 * np.pi, 100)
    v = np.linspace(0, np.pi / 2, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Initialize 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the unit sphere
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)  # Initialize 3D plot

    # Initialize inset 3D plot
    inset_ax = fig.add_axes([0.02, 0.05, 0.3, 0.3], projection='3d')  # Adjust position and size of inset plot

    # Plot the unit sphere on the inset plot
    inset_ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)

    ins_x_lim = []
    ins_y_lim = []
    ins_z_lim = []
    # Plot the projections for each interpolated time
    for i, vector in enumerate([start_vector, end_vector]):

        # Convert angular size to radians and calculate angular radius
        angular_radius = vector['angular_size_radians'] / 2
        # Create a circle in 3D space for the projection
        num_points = 360  # Number of points to define the circle
        theta = np.linspace(0, 2 * np.pi, num_points)  # Angle around the circle

        # Find a basis for the plane perpendicular to the vector
        arbitrary_vector = (
            np.array([1, 0, 0])
            if not np.allclose(vector['cel_to_target_cartesian'], [1, 0, 0])
            else np.array([0, 1, 0])
        )
        basis1 = np.cross(vector['cel_to_target_cartesian'], arbitrary_vector)
        basis1 /= np.linalg.norm(basis1)  # Normalize

        basis2 = np.cross(vector['cel_to_target_cartesian'], basis1)
        basis2 /= np.linalg.norm(basis2)  # Normalize

        # Generate points on the circle
        circle_points = []
        for angle in theta:
            point = np.cos(angular_radius) * vector['cel_to_target_cartesian'] + np.sin(angular_radius) * (
                np.cos(angle) * basis1 + np.sin(angle) * basis2
            )
            circle_points.append(point)
        circle_points = np.array(circle_points)

        ins_x_lim.append([np.min(circle_points[:, 0]), np.max(circle_points[:, 0])])
        ins_y_lim.append([np.min(circle_points[:, 1]), np.max(circle_points[:, 1])])
        ins_z_lim.append([np.min(circle_points[:, 2]), np.max(circle_points[:, 2])])
        # Plot the circular projection
        ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red')

        # Plot the circular projection on the inset plot
        inset_ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red')

        # Plot the vector from the mixel
        if i == 0:
            ax.quiver(
                0,
                0,
                0,
                vector['cel_to_target_cartesian'][0],
                vector['cel_to_target_cartesian'][1],
                vector['cel_to_target_cartesian'][2],
                color='green',
                label=f"Start Time Local: {local_time_start}",
                arrow_length_ratio=0.1,
            )
            ax.quiver(
                0,
                0,
                0,
                observer_vec[0],
                observer_vec[1],
                observer_vec[2],
                color='black',
                label="Observer to Target",
                arrow_length_ratio=0.1,
            )
            ax.quiver(
                0,
                0,
                0,
                vector_dict['slope_1'][0],
                vector_dict['slope_1'][1],
                vector_dict['slope_1'][2],
                color='royalblue',
                label="Slope Option 1",
                arrow_length_ratio=0.1,
            )
            inset_ax.scatter(inter_1[0], inter_1[1], inter_1[2], c='royalblue', label="Intersection 1")
        else:
            ax.quiver(
                0,
                0,
                0,
                vector['cel_to_target_cartesian'][0],
                vector['cel_to_target_cartesian'][1],
                vector['cel_to_target_cartesian'][2],
                color='red',
                label=f"End Time Local: {local_time_end}",
                arrow_length_ratio=0.1,
            )
            ax.quiver(
                0,
                0,
                0,
                vector_dict['slope_2'][0],
                vector_dict['slope_2'][1],
                vector_dict['slope_2'][2],
                color='darkorange',
                label=f"Slope Option 2",
                arrow_length_ratio=0.1,
            )
            inset_ax.scatter(inter_2[0], inter_2[1], inter_2[2], c='darkorange', label="Intersection 2")

    # Set plot limits and labels
    ax.scatter([], [], [], c='royalblue', label="Intersection 1")
    ax.scatter([], [], [], c='darkorange', label="Intersection 2")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 0])
    ax.set_zlim([0, 1])
    ax.set_xlabel('X - East is Positive')
    ax.set_ylabel('Y - North is Positive')
    ax.set_zlabel('Z - Zenith is Positive')
    ax.set_title(f'Mixel {pixel} Bright to Dark on Unit Sphere with {celestial_object.capitalize()}')
    ax.set_aspect('equal')
    ax.legend(loc='lower center')

    # Set limits and labels for the inset plot
    inset_ax.set_xlim([np.min(ins_x_lim) - 0.01, np.max(ins_x_lim) + 0.01])
    inset_ax.set_ylim([np.min(ins_y_lim) - 0.01, np.max(ins_y_lim) + 0.01])
    inset_ax.set_zlim([np.min(ins_z_lim) - 0.01, np.max(ins_z_lim) + 0.01])
    # Reduce the number of tick marks on the inset plot
    inset_ax.set_xticks([np.round(np.min(ins_x_lim), 2), np.round(np.max(ins_x_lim), 2)])  # Fewer X-axis ticks
    inset_ax.set_yticks([np.round(np.min(ins_y_lim), 2), np.round(np.max(ins_y_lim), 2)])  # Fewer Y-axis ticks
    inset_ax.set_zticks([np.round(np.min(ins_z_lim), 2), np.round(np.max(ins_z_lim), 2)])  # Fewer Z-axis ticks

    inset_ax.set_title("Zoomed Projection")
    inset_ax.set_xlabel("X")
    inset_ax.set_ylabel("Y")
    inset_ax.set_zlabel("Z")
    # Save the plot to the output folder
    plot_file = os.path.join(output_folder, f"pixel_{pixel}_sky_plot_slope.png")

    if os.path.exists(plot_file):
        return f"File '{plot_file}' already exists. Skipping plot generation."
    else:
        plt.savefig(plot_file)
    plt.close()

    # plt.show()


def plotting_pixel_transition_vectors_decoupled(
    data_location, checkpoint_folder, checkpoint_data_file, checkpoint_plot_file, celestial_object, output_folder_img
):

    # Ensure the output folder exists
    os.makedirs(output_folder_img, exist_ok=True)

    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_data_file)
    checkpoint_plots = lbt.load_checkpoint(checkpoint_folder, checkpoint_plot_file)
    if checkpoint_plots is None:
        checkpoint_plots = {"plotted_pixels": []}
    data = lbt.read_compressed_json(data_location)
    # Create a tqdm progress bar outside the loop
    progress_bar = tqdm(total=len(data), desc="Creating 3D Pixel Vector Plots")

    for pixel, _ in data.items():
        # Save the plot to the output folder
        height, _ = eval(pixel)
        plot_file = os.path.normpath(os.path.join(output_folder_img, str(height), f"pixel_{pixel}_sky_plot_slope.png"))
        if pixel in checkpoint_data["pixels_without_data"]:
            logger.info("Skipping 3D plot for pixel %s without data.", pixel)
            # print(f"Skipping pixel {pixel} without data.")
            progress_bar.update(1)
            continue

        if isinstance(data[pixel], list):
            logger.info("Pixel %s incorrectly saved as having data", pixel)
            progress_bar.update(1)
            continue

        if os.path.exists(plot_file):
            logger.info("Pixel %s 3D plot already exists, skipping", pixel)
            progress_bar.update(1)
            continue

        start_vector = data[pixel]['start_vector']
        end_vector = data[pixel]['end_vector']
        local_time_start = data[pixel]['start_time_Local']
        local_time_end = data[pixel]['end_time_Local']
        inter_1 = data[pixel]['intersection_1']
        inter_2 = data[pixel]['intersection_2']
        slope_1 = data[pixel]['slope_1']
        slope_2 = data[pixel]['slope_2']
        observer_vec = data[pixel]['observer_vector']
        # plotting vectors
        # Create the Positive Zenith side of the south half of a unit sphere for visualization of the horizon
        u = np.linspace(np.pi, 2 * np.pi, 100)
        v = np.linspace(0, np.pi / 2, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Initialize 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the unit sphere
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)  # Initialize 3D plot

        # Initialize inset 3D plot
        inset_ax = fig.add_axes([0.02, 0.05, 0.3, 0.3], projection='3d')  # Adjust position and size of inset plot

        # Plot the unit sphere on the inset plot
        inset_ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)

        ins_x_lim = []
        ins_y_lim = []
        ins_z_lim = []

        # Calculate azimuthal angle (azim)
        azim = np.degrees(np.arctan2(slope_1[1], slope_1[0]))

        # Calculate elevation angle (elev)
        # Handle the case where x and y are both zero to avoid division by zero
        if slope_1[0] == 0 and slope_1[1] == 0:
            elev = 90 if slope_1[2] > 0 else -90  # Directly overhead or underneath
        else:
            elev = np.degrees(np.arctan2(slope_1[2], np.sqrt(slope_1[0] ** 2 + slope_1[1] ** 2)))

        inset_ax.view_init(elev=elev, azim=azim)

        # Plot the projections for each interpolated time
        for i, vector in enumerate([start_vector, end_vector]):

            # Convert angular size to radians and calculate angular radius
            angular_radius = vector['angular_size_radians'] / 2
            # Create a circle in 3D space for the projection
            num_points = 360  # Number of points to define the circle
            theta = np.linspace(0, 2 * np.pi, num_points)  # Angle around the circle

            # Find a basis for the plane perpendicular to the vector
            arbitrary_vector = (
                np.array([1, 0, 0])
                if not np.allclose(vector['cel_to_target_cartesian'], [1, 0, 0])
                else np.array([0, 1, 0])
            )
            basis1 = np.cross(vector['cel_to_target_cartesian'], arbitrary_vector)
            basis1 /= np.linalg.norm(basis1)  # Normalize

            basis2 = np.cross(vector['cel_to_target_cartesian'], basis1)
            basis2 /= np.linalg.norm(basis2)  # Normalize

            # Generate points on the circle
            circle_points = []
            for angle in theta:
                point = np.cos(angular_radius) * vector['cel_to_target_cartesian'] + np.sin(angular_radius) * (
                    np.cos(angle) * basis1 + np.sin(angle) * basis2
                )
                circle_points.append(point)
            circle_points = np.array(circle_points)

            ins_x_lim.append([np.min(circle_points[:, 0]), np.max(circle_points[:, 0])])
            ins_y_lim.append([np.min(circle_points[:, 1]), np.max(circle_points[:, 1])])
            ins_z_lim.append([np.min(circle_points[:, 2]), np.max(circle_points[:, 2])])
            # Plot the circular projection
            ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red')

            # Plot the circular projection on the inset plot
            inset_ax.plot(
                circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red'
            )

            # Plot the vector from the mixel
            if i == 0:
                ax.quiver(
                    0,
                    0,
                    0,
                    vector['cel_to_target_cartesian'][0],
                    vector['cel_to_target_cartesian'][1],
                    vector['cel_to_target_cartesian'][2],
                    color='green',
                    label=f"Start Time Local: {local_time_start}",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    observer_vec[0],
                    observer_vec[1],
                    observer_vec[2],
                    color='black',
                    label="Observer to Target",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    slope_1[0],
                    slope_1[1],
                    slope_1[2],
                    color='royalblue',
                    label="Slope Option 1",
                    arrow_length_ratio=0.1,
                )
                inset_ax.scatter(inter_1[0], inter_1[1], inter_1[2], c='royalblue', label="Intersection 1")
            else:
                ax.quiver(
                    0,
                    0,
                    0,
                    vector['cel_to_target_cartesian'][0],
                    vector['cel_to_target_cartesian'][1],
                    vector['cel_to_target_cartesian'][2],
                    color='red',
                    label=f"End Time Local: {local_time_end}",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    slope_2[0],
                    slope_2[1],
                    slope_2[2],
                    color='darkorange',
                    label="Slope Option 2",
                    arrow_length_ratio=0.1,
                )
                inset_ax.scatter(inter_2[0], inter_2[1], inter_2[2], c='darkorange', label="Intersection 2")

        # Set plot limits and labels
        ax.scatter([], [], [], c='royalblue', label="Intersection 1")
        ax.scatter([], [], [], c='darkorange', label="Intersection 2")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 0])
        ax.set_zlim([0, 1])
        ax.set_xlabel('X - East is Positive')
        ax.set_ylabel('Y - North is Positive')
        ax.set_zlabel('Z - Zenith is Positive')
        ax.set_title(f'Mixel {pixel} Bright to Dark on Unit Sphere with {celestial_object.capitalize()}')
        ax.set_aspect('equal')
        ax.legend(loc='lower center')

        # Set limits and labels for the inset plot
        inset_ax.set_xlim([np.min(ins_x_lim) - 0.01, np.max(ins_x_lim) + 0.01])
        inset_ax.set_ylim([np.min(ins_y_lim) - 0.01, np.max(ins_y_lim) + 0.01])
        inset_ax.set_zlim([np.min(ins_z_lim) - 0.01, np.max(ins_z_lim) + 0.01])
        # Reduce the number of tick marks on the inset plot
        inset_ax.set_xticks([np.round(np.min(ins_x_lim), 2), np.round(np.max(ins_x_lim), 2)])  # Fewer X-axis ticks
        inset_ax.set_yticks([np.round(np.min(ins_y_lim), 2), np.round(np.max(ins_y_lim), 2)])  # Fewer Y-axis ticks
        inset_ax.set_zticks([np.round(np.min(ins_z_lim), 2), np.round(np.max(ins_z_lim), 2)])  # Fewer Z-axis ticks

        inset_ax.set_title("Zoomed Projection")
        inset_ax.set_xlabel("X")
        inset_ax.set_ylabel("Y")
        inset_ax.set_zlabel("Z")
        # Save the plot to the output folder
        height, _ = eval(pixel)
        if os.path.isdir(os.path.join(output_folder_img, str(height))):
            plt.savefig(plot_file)
            plt.close()
        else:
            os.makedirs(os.path.join(output_folder_img, str(height)), exist_ok=True)
            plt.savefig(plot_file)
            plt.close()
        checkpoint_plots["plotted_pixels"].append(pixel)
        lbt.save_checkpoint(checkpoint_folder, checkpoint_plot_file, checkpoint_plots)
        progress_bar.update(1)
    progress_bar.close()


def plot_pixel_batch(batch_pixels, data, checkpoint_data, output_folder_img, celestial_object):
    """
    Process a batch of pixels and return the list of successfully plotted pixels.
    """
    plotted_pixels = []

    for pixel in batch_pixels:
        height, _ = eval(pixel)
        plot_file = os.path.normpath(os.path.join(output_folder_img, str(height), f"pixel_{pixel}_sky_plot_slope.png"))

        if pixel in checkpoint_data["pixels_without_data"]:
            logger.info("Skipping 3D plot for pixel %s without data.", pixel)
            continue

        if isinstance(data[pixel], list):
            logger.info("Pixel %s incorrectly saved as having data", pixel)
            continue

        if os.path.exists(plot_file):
            logger.info("Pixel %s 3D plot already exists, skipping", pixel)
            continue

        start_vector = data[pixel]['start_vector']
        end_vector = data[pixel]['end_vector']
        local_time_start = data[pixel]['start_time_Local']
        local_time_end = data[pixel]['end_time_Local']
        inter_1 = data[pixel]['intersection_1']
        inter_2 = data[pixel]['intersection_2']
        slope_1 = data[pixel]['slope_1']
        slope_2 = data[pixel]['slope_2']
        observer_vec = data[pixel]['observer_vector']

        # Create the Positive Zenith side of the south half of a unit sphere for visualization of the horizon
        u = np.linspace(np.pi, 2 * np.pi, 100)
        v = np.linspace(0, np.pi / 2, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        # Initialize 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the unit sphere
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)

        # Initialize inset 3D plot
        inset_ax = fig.add_axes([0.02, 0.05, 0.3, 0.3], projection='3d')

        # Plot the unit sphere on the inset plot
        inset_ax.plot_surface(x, y, z, color='lightblue', alpha=0.15)

        ins_x_lim = []
        ins_y_lim = []
        ins_z_lim = []

        # Calculate azimuthal angle (azim)
        azim = np.degrees(np.arctan2(slope_1[1], slope_1[0]))

        # Calculate elevation angle (elev)
        if slope_1[0] == 0 and slope_1[1] == 0:
            elev = 90 if slope_1[2] > 0 else -90
        else:
            elev = np.degrees(np.arctan2(slope_1[2], np.sqrt(slope_1[0] ** 2 + slope_1[1] ** 2)))

        inset_ax.view_init(elev=elev, azim=azim)

        # Plot the projections for each interpolated time
        for i, vector in enumerate([start_vector, end_vector]):
            angular_radius = vector['angular_size_radians'] / 2
            num_points = 360
            theta = np.linspace(0, 2 * np.pi, num_points)

            arbitrary_vector = (
                np.array([1, 0, 0])
                if not np.allclose(vector['cel_to_target_cartesian'], [1, 0, 0])
                else np.array([0, 1, 0])
            )
            basis1 = np.cross(vector['cel_to_target_cartesian'], arbitrary_vector)
            basis1 /= np.linalg.norm(basis1)

            basis2 = np.cross(vector['cel_to_target_cartesian'], basis1)
            basis2 /= np.linalg.norm(basis2)

            circle_points = []
            for angle in theta:
                point = np.cos(angular_radius) * vector['cel_to_target_cartesian'] + np.sin(angular_radius) * (
                    np.cos(angle) * basis1 + np.sin(angle) * basis2
                )
                circle_points.append(point)
            circle_points = np.array(circle_points)

            ins_x_lim.append([np.min(circle_points[:, 0]), np.max(circle_points[:, 0])])
            ins_y_lim.append([np.min(circle_points[:, 1]), np.max(circle_points[:, 1])])
            ins_z_lim.append([np.min(circle_points[:, 2]), np.max(circle_points[:, 2])])

            ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red')
            inset_ax.plot(
                circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color='green' if i == 0 else 'red'
            )

            if i == 0:
                ax.quiver(
                    0,
                    0,
                    0,
                    vector['cel_to_target_cartesian'][0],
                    vector['cel_to_target_cartesian'][1],
                    vector['cel_to_target_cartesian'][2],
                    color='green',
                    label=f"Start Time Local: {local_time_start}",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    observer_vec[0],
                    observer_vec[1],
                    observer_vec[2],
                    color='black',
                    label="Observer to Target",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    slope_1[0],
                    slope_1[1],
                    slope_1[2],
                    color='royalblue',
                    label="Slope Option 1",
                    arrow_length_ratio=0.1,
                )
                inset_ax.scatter(inter_1[0], inter_1[1], inter_1[2], c='royalblue', label="Intersection 1")
            else:
                ax.quiver(
                    0,
                    0,
                    0,
                    vector['cel_to_target_cartesian'][0],
                    vector['cel_to_target_cartesian'][1],
                    vector['cel_to_target_cartesian'][2],
                    color='red',
                    label=f"End Time Local: {local_time_end}",
                    arrow_length_ratio=0.1,
                )
                ax.quiver(
                    0,
                    0,
                    0,
                    slope_2[0],
                    slope_2[1],
                    slope_2[2],
                    color='darkorange',
                    label="Slope Option 2",
                    arrow_length_ratio=0.1,
                )
                inset_ax.scatter(inter_2[0], inter_2[1], inter_2[2], c='darkorange', label="Intersection 2")

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 0])
        ax.set_zlim([0, 1])
        ax.set_xlabel('X - East is Positive')
        ax.set_ylabel('Y - North is Positive')
        ax.set_zlabel('Z - Zenith is Positive')
        ax.set_title(f'Mixel {pixel} Bright to Dark on Unit Sphere with {celestial_object.capitalize()}')
        ax.set_aspect('equal')
        ax.legend(loc='lower center')

        inset_ax.set_xlim([np.min(ins_x_lim) - 0.01, np.max(ins_x_lim) + 0.01])
        inset_ax.set_ylim([np.min(ins_y_lim) - 0.01, np.max(ins_y_lim) + 0.01])
        inset_ax.set_zlim([np.min(ins_z_lim) - 0.01, np.max(ins_z_lim) + 0.01])
        inset_ax.set_xticks([np.round(np.min(ins_x_lim), 2), np.round(np.max(ins_x_lim), 2)])
        inset_ax.set_yticks([np.round(np.min(ins_y_lim), 2), np.round(np.max(ins_y_lim), 2)])
        inset_ax.set_zticks([np.round(np.min(ins_z_lim), 2), np.round(np.max(ins_z_lim), 2)])
        inset_ax.set_title("Zoomed Projection")
        inset_ax.set_xlabel("X")
        inset_ax.set_ylabel("Y")
        inset_ax.set_zlabel("Z")

        if os.path.isdir(os.path.join(output_folder_img, str(height))):
            plt.savefig(plot_file)
            plt.close()
        else:
            os.makedirs(os.path.join(output_folder_img, str(height)), exist_ok=True)
            plt.savefig(plot_file)
            plt.close()

        plotted_pixels.append(pixel)

    return plotted_pixels


def plotting_pixel_transition_vectors_decoupled_mp(
    data_location,
    checkpoint_folder,
    checkpoint_data_file,
    checkpoint_plot_file,
    celestial_object,
    output_folder_img,
    batch_size=5000,
):
    """
    Function to process pixels in batches, skipping already processed pixels.
    """
    os.makedirs(output_folder_img, exist_ok=True)

    # Load checkpoint data and plots
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_data_file)
    checkpoint_plots = lbt.load_checkpoint(checkpoint_folder, checkpoint_plot_file)
    if checkpoint_plots is None:
        checkpoint_plots = {"plotted_pixels": []}
    data = lbt.read_compressed_json(data_location)

    # Get the list of all pixels and filter out already processed ones
    all_pixels = list(data.keys())
    processed_pixels = checkpoint_plots["plotted_pixels"]
    unprocessed_pixels = [pixel for pixel in all_pixels if pixel not in processed_pixels]
    total_pixels = len(all_pixels)

    # Initialize progress bar
    progress_bar = tqdm(total=total_pixels, desc="Creating 3D Pixel Vector Plots")
    progress_bar.update(len(processed_pixels))  # Update progress bar for already processed pixels

    # Process pixels in batches
    for i in range(0, len(unprocessed_pixels), batch_size):
        batch_pixels = unprocessed_pixels[i : i + batch_size]

        with Pool() as pool:
            results = pool.starmap(
                plot_pixel_batch, [(batch_pixels, data, checkpoint_data, output_folder_img, celestial_object)]
            )

            for plotted_pixels in results:
                checkpoint_plots["plotted_pixels"].extend(plotted_pixels)
                lbt.save_checkpoint(checkpoint_folder, checkpoint_plot_file, checkpoint_plots)
                progress_bar.update(len(plotted_pixels))

    progress_bar.close()


def trilaterate(positions: np.ndarray, radii: np.ndarray, raise_on_no_solution: bool = False) -> np.ndarray:
    """
    Trilateration algorithm to find the intersection points of three spheres in 3D space.

    :param positions: A 3x3 array where each row represents the (x, y, z) coordinates of a sphere's center.
    :param radii: A 3-element array representing the radius of each sphere.
    :param raise_on_no_solution: If True, raises an error when no exact solution exists. If False, returns a close-enough solution.
    :return: A 2x3 array of intersection points (two solutions), or a 1x3 array if there is only one solution or no exact solution.

    Notes:
    - This implementation is based on the mathematical derivation of trilateration.
    - Modified from https://stackoverflow.com/a/18654302/313768
    """
    # Extract radii and positions
    radius1, radius2, radius3 = radii
    center1, center2, center3 = positions

    # Step 1: Compute inter-point vectors
    vector21 = center2 - center1  # Vector from center1 to center2
    vector31 = center3 - center1  # Vector from center1 to center3
    distance12 = np.linalg.norm(vector21)  # Distance between center1 and center2
    distance13 = np.linalg.norm(vector31)  # Distance between center1 and center3
    distance23 = np.linalg.norm(center3 - center2)  # Distance between center2 and center3

    # Step 2: Check for degenerate cases (e.g., overlapping spheres)
    if np.allclose(center1, center2) and np.isclose(radius1, radius2):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")
    if np.allclose(center1, center3) and np.isclose(radius1, radius3):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")
    if np.allclose(center2, center3) and np.isclose(radius2, radius3):
        raise ValueError("Two spheres have the same center and radius, resulting in an ambiguous or infinite solution.")

    # Check for non-overlapping spheres
    if distance12 > radius1 + radius2 or distance12 < abs(radius1 - radius2):
        raise ValueError("Spheres 1 and 2 do not overlap, no solution exists.")
    if distance13 > radius1 + radius3 or distance13 < abs(radius1 - radius3):
        raise ValueError("Spheres 1 and 3 do not overlap, no solution exists.")
    if distance23 > radius2 + radius3 or distance23 < abs(radius2 - radius3):
        # Compute the closest point on the unit sphere to the line segment connecting centers of spheres 2 and 3
        vector23 = center3 - center2
        unit_vector23 = vector23 / np.linalg.norm(vector23)  # Unit vector along vector23
        midpoint = center2 + unit_vector23 * (distance23 / 2)  # Midpoint of the line segment
        closest_point_on_unit_sphere = midpoint / np.linalg.norm(midpoint)  # Project midpoint onto the unit sphere
        logger.info(
            "Spheres 2 and 3 do not overlap. A single solution between the two spheres is returned for both solutions"
        )
        return np.stack(
            (closest_point_on_unit_sphere, closest_point_on_unit_sphere)
        )  # Return as a single point for both intersections
        # raise ValueError("Spheres 2 and 3 do not overlap, no solution exists.")

    # Step 3: Compute basis vectors for the coordinate system
    unit_vector_u = vector21 / distance12  # Unit vector along vector21
    projection_i = unit_vector_u.dot(vector31)  # Projection of vector31 onto unit_vector_u
    vector_v = vector31 - unit_vector_u * projection_i  # Orthogonal component of vector31
    vector_v /= np.linalg.norm(vector_v)  # Normalize vector_v
    projection_j = vector_v.dot(vector31)  # Projection of vector31 onto vector_v
    unit_vector_w = np.cross(unit_vector_u, vector_v)  # Unit vector orthogonal to both unit_vector_u and vector_v

    # Step 4: Solve for the x and y coordinates in the projected space
    x = 0.5 / distance12 * (radius1**2 - radius2**2 + distance12**2)
    y = 0.5 / projection_j * (radius1**2 - radius3**2 - 2 * projection_i * x + projection_i**2 + projection_j**2)
    radicand = radius1**2 - x**2 - y**2  # Compute the radicand for the z-coordinate

    # Step 5: Handle cases where the radicand is negative (no exact solution)
    if radicand < 0:
        logger.debug("Negative radicand %f: no exact solution exists", radicand)
        if raise_on_no_solution:
            raise ValueError(f"Negative radicand {radicand}: no exact solutions exist.")
        return (center1 + unit_vector_u * x + vector_v * y)[np.newaxis, :]  # Return a close-enough solution

    # Step 6: Compute the z-coordinate and intersection points
    z = np.sqrt(radicand)  # Compute the z-coordinate
    offset_z = unit_vector_w * z  # Offset in the z-direction
    solution_a = center1 + unit_vector_u * x + vector_v * y + offset_z  # First intersection point
    solution_b = center1 + unit_vector_u * x + vector_v * y - offset_z  # Second intersection point

    # Step 7: Return the solutions
    return np.stack((solution_a, solution_b))  # Return both intersection points as a 2x3 array


def lat_long_to_decimal(input):
    decimal = input[0] + input[1] / 60 + input[2] / 3600
    return decimal


def maximum_frame_range(frame_ranges):
    """
    Calculates the maximum range (duration) between sequential frames
    where the transition type alternates between bright (1) and dark (0).

    Parameters:
        frame_ranges (list of tuples): Each tuple contains (transition_type, frame_number).
                                       transition_type is 1 for bright and 0 for dark.
                                       frame_number is an integer representing the frame number.

    Returns:
        tuple: A tuple containing:
            - max_range_frames (list): The two tuples representing the start and end of the maximum range.
            - max_duration (int): The maximum duration between sequential frames.
    """
    # Ensure the input list is sorted by frame_number
    frame_ranges = sorted(frame_ranges, key=lambda x: x[1])

    # Initialize variables to track the maximum duration and corresponding frame range
    max_duration = 0
    max_range_frames = None

    # Iterate through the sorted list to calculate differences between sequential frames
    for i in range(len(frame_ranges) - 1):
        current_frame = frame_ranges[i]
        next_frame = frame_ranges[i + 1]

        # Check if the transition types alternate (bright -> dark or dark -> bright)
        if current_frame[0] != next_frame[0]:
            # Calculate the duration between the current and next frame
            duration = abs(next_frame[1] - current_frame[1])

            # Update the maximum duration and corresponding frame range if needed
            if duration > max_duration:
                max_duration = duration
                max_range_frames = [current_frame, next_frame]

    return max_range_frames, max_duration


def frame_number_from_img_name(image_name_str):
    # returns the integer number of a frame given the following format
    # "DSC_2832-09170.png" where DSC_2832 is the video source and "09170"
    # is the frame number
    _, tail = os.path.split(image_name_str)
    _, frame = re.findall(r'\d+', tail)
    return int(frame)


def define_observation_time(video_time, time_offset):
    video_time = video_time + time_offset
    observation_time = video_time.astimezone(timezone.utc)
    # Define observation time
    ts = skf.load.timescale()
    if isinstance(observation_time, skf.Time):
        pass
    elif isinstance(observation_time, datetime):
        # Extract components from the datetime object
        observation_time = ts.utc(
            observation_time.year,
            observation_time.month,
            observation_time.day,
            observation_time.hour,
            observation_time.minute,
            observation_time.second,
        )
    else:
        # If observation_time is already a tuple, unpack it directly
        observation_time = ts.utc(*observation_time)
    return observation_time


# Example usage
if __name__ == "__main__":
    start_time = time.time()
    # File Locations
    json_data_location = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/7_pixel_timing_interrogation/time_history_transition_parallel_facet.json.gz"
    output_folder_img = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/8_pixel_vector_information/pixel_vector_plots"
    output_folder_vector = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/8_pixel_vector_information/debug"
    output_vector_data_name = "pixel_vector_information_wslope_debug.json.gz"
    video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/DSC_0025.MOV"  # Video Path Used to Generate Frames
    checkpoint_dir = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints"
    checkpoint_file_data = "pixel_vector_checkpoint_info_parallel_debug.json"
    checkpoint_file_3d_plots = "pixel_vector_3D plotting_checkpoint_debug.json"

    logger.info(f"Code Start Time: {time.ctime(start_time)}.")

    celestial_object = "sun"
    metadata = extract_video_metadata_exiftool(video_file_path)

    # Define observer location Example ≈NSTTF Tower 260 Level Balcony West Side
    observer_lat = (34, 57, 44.56)  # (degree, minute, second) negative degree for south
    observer_long = (-106, 30, 34.90)  # (degree, minute, second) negative degree for west
    observer_elevation = 1755.648 + 1.2192
    observer_location = (lat_long_to_decimal(observer_lat), lat_long_to_decimal(observer_long), observer_elevation)

    # Define target location Example ≈Sun Data Marker 2 In front of 5E8
    target_lat = (34, 57, 45.92)  # (degree, minute, second) negative degree for south
    target_long = (-106, 30, 31.88)  # (degree, minute, second) negative degree for west
    target_elevation = 1706.88
    target_location = (
        lat_long_to_decimal(target_lat),
        lat_long_to_decimal(target_long),
        target_elevation,
    )  # Example target location (latitude, longitude, elevation in meters)

    '''
    # Define observer location Example ≈NSTTF Tower 260 Level Balcony East Side
    observer_lat = (34, 57, 44.56)  # (degree, minute, second) negative degree for south
    observer_long = (-106, 30, 34.65)  # (degree, minute, second) negative degree for west
    observer_elevation = 1755.648 + 1.2192
    observer_location = (lat_long_to_decimal(observer_lat), lat_long_to_decimal(observer_long), observer_elevation)

    # Define target location Example ≈NSTTF Heliostat 5E9
    target_lat = (34, 57, 46.08)  # (degree, minute, second) negative degree for south
    target_long = (-106, 30, 31.43)  # (degree, minute, second) negative degree for west
    target_elevation = 1706.88
    target_location = (
        lat_long_to_decimal(target_lat),
        lat_long_to_decimal(target_long),
        target_elevation,
    )  # Example target location (latitude, longitude, elevation in meters)


    extract_pixel_timing_and_celestial_vectors(
        celestial_object_name=celestial_object,
        target_location=target_location,
        observer_location=observer_location,
        camera_time_shift=camera_time_shift,
        data_location=json_data_location,
        video_metadata=metadata,
        output_folder_vector=output_folder_vector,
        output_json_name=output_vector_data_name,
        checkpoint_folder=checkpoint_dir,
        checkpoint_file=checkpoint_file_data,
    )
    '''

    camera_time_shift = timedelta(hours=0, minutes=0, seconds=0)

    extract_pixel_timing_and_celestial_vectors_parallel(
        celestial_object_name=celestial_object,
        target_location=target_location,
        observer_location=observer_location,
        camera_time_shift=camera_time_shift,
        data_location=json_data_location,
        video_metadata=metadata,
        output_folder_vector=output_folder_vector,
        output_json_name=output_vector_data_name,
        checkpoint_folder=checkpoint_dir,
        checkpoint_file=checkpoint_file_data,
        batch_size=5000,
    )
    math_time = time.time() - start_time
    logger.info(f"Completed Astronomy and Math in {math_time:.4f} seconds.")

    plotting_pixel_transition_vectors_decoupled_mp(
        data_location=os.path.join(output_folder_vector, output_vector_data_name),
        checkpoint_folder=checkpoint_dir,
        checkpoint_data_file=checkpoint_file_data,
        checkpoint_plot_file=checkpoint_file_3d_plots,
        celestial_object=celestial_object,
        output_folder_img=output_folder_img,
        batch_size=1000,
    )
    plot_time = time.time() - start_time
    logger.info(f"Completed Plotting in {plot_time:.4f} seconds measured from start.")
