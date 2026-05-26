import re
import os
import json
import gzip
import time
import subprocess
import time
import h5py
import shutil
from multiprocessing import Pool, Manager
from opencsp.common.lib.tool.log_tools import multiprocessing_logger
from logging import DEBUG, ERROR
import tkinter as tk
from tkinter import filedialog, messagebox

from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
import skyfield.api as skf
import numpy as np
from tqdm import tqdm
import cv2 as cv
from scipy.spatial.transform import Rotation


def logging_setup(log_folder: str, log_file_name: str = "error_log.txt", log_type=DEBUG):
    """
    Sets up a logger for error logging with robust handling of file names and extensions.

    Parameters:
        log_folder (str): The folder where the log file should be saved.
        log_file_name (str): The name of the log file. If it does not have a '.txt' extension, it will be added.
        log_type: The logging level (e.g., logging.DEBUG, logging.ERROR).

    Returns:
        Logger: Configured logger instance.
    """
    # Default folder path if none is provided
    if not log_folder:
        # log_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints/error_logs"
        log_folder = os.getcwd()

    # Ensure the folder exists
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # Ensure the file name has a '.txt' extension
    base_name, extension = os.path.splitext(log_file_name)
    if extension.lower() != ".txt":
        log_file_name = f"{base_name}.txt"

    # Construct the full path to the log file
    log_file_path = os.path.join(log_folder, log_file_name)

    # Create the logger using the multiprocessing_logger function
    try:
        logger = multiprocessing_logger(log_dir_body_ext=log_file_path, level=log_type)
    except Exception as e:
        raise RuntimeError(f"Failed to create logger: {e}")

    return logger


def frame_number_from_img_name(image_name_str):
    # returns the integer number of a frame given the following format
    # "DSC_2832-09170.png" where DSC_2832 is the video source and "09170"
    # is the frame number
    _, tail = os.path.split(image_name_str)
    _, frame = re.findall(r'\d+', tail)
    return int(frame)


def lat_long_to_decimal(input):
    decimal = input[0] + input[1] / 60 + input[2] / 3600
    return decimal


def define_observation_time_skyfield(source_time, time_offset):
    source_time = source_time + time_offset
    observation_time = source_time.astimezone(timezone.utc)
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


def save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data, print_path=True):
    """
    Saves the checkpoint data to a JSON file.

    Parameters:
        checkpoint_file_name (str): Path to the checkpoint file.
        checkpoint_data (dict): Dictionary containing checkpoint information.

    Returns:
        None
    """
    raw_path = os.path.join(checkpoint_folder, checkpoint_file_name)
    norm_path = os.path.normpath(raw_path)

    success = False
    for i in range(10):
        try:
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=4)
                success = True
                break
        except Exception:
            logger.error("Unable to save checkpoint", exc_info=True)
            time.sleep(0.01)

    if success:
        if print_path and i == 0:
            logger.info("Checkpoint saved to %s", raw_path)
        if i > 0:
            logger.info("Checkpoint saved to %s after %d attempts", raw_path, i)


def load_checkpoint(checkpoint_folder, checkpoint_file_name):
    """
    Loads the checkpoint data from a JSON file.

    Parameters:
        checkpoint_folder (str): Path to the folder containing the checkpoint file.
        checkpoint_file_name (str): Name of the checkpoint file.

    Returns:
        dict: Dictionary containing checkpoint information, or None if the file is empty or does not exist.
    """
    norm_path = os.path.normpath(os.path.join(checkpoint_folder, checkpoint_file_name))

    if os.path.exists(norm_path):
        # Check if the file is empty
        if os.path.getsize(norm_path) == 0:
            print(f"Checkpoint file {norm_path} is empty.")
            return None

        # Attempt to load the JSON data
        with open(norm_path, 'r', encoding='utf-8') as f:
            try:
                checkpoint_data = json.load(f)
                # Check if the loaded JSON is empty (e.g., {} or [])
                if not checkpoint_data:
                    print(f"Checkpoint file {norm_path} contains empty JSON data.")
                    return None
                print(f"Checkpoint loaded from {norm_path}")
                return checkpoint_data
            except json.JSONDecodeError:
                logger.error("Invalid Json File", exc_info=True)
                print(f"Checkpoint file {norm_path} contains invalid JSON.")
                return None
    else:
        print(f"Checkpoint file {norm_path} does not exist.")
        return None


def custom_serializer(obj):
    """
    Custom serializer to handle numpy arrays, datetime objects, floats, integers, and strings.
    Converts unsupported types into JSON-compatible formats.
    """
    try:
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list

        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string

        # Handle floats
        elif isinstance(obj, float):
            return float(obj)  # Ensure it's a float (JSON-compatible)

        # Handle integers
        elif isinstance(obj, int):
            return int(obj)  # Ensure it's an integer (JSON-compatible)

        # Handle strings
        elif isinstance(obj, str):
            return str(obj)  # Ensure it's a string (JSON-compatible)

        # Handle sets (convert to list for JSON compatibility)
        elif isinstance(obj, set):
            return list(obj)  # Convert set to list

        # Handle Rotation objects
        elif isinstance(obj, Rotation):
            return {"__rotation__": True, "quat": obj.as_quat().tolist()}  # Serialize as quaternion

        # Unsupported type
        else:
            # logger.error("Unsupported type encountered: %r", type(obj), exc_info=True)
            raise TypeError(f"Type {type(obj)} not serializable")

    except Exception:
        # logger.error("Error during serialization", exc_info=True)
        raise


def custom_deserializer(obj):
    """
    Custom deserializer to handle strings, datetime strings, numpy arrays, floats, integers, and sets.
    Converts JSON-compatible formats back into their original types.
    """
    rotation = False
    for key, value in obj.items():
        if rotation:
            if key == 'quat' and isinstance(value, list):
                rot_temp = Rotation.from_quat(value)  # Deserialize from quaternion
                return rot_temp

        # Handle Rotation objects
        if isinstance(obj, dict) and "__rotation__" in key and value is True:
            rotation = True

        # Handle numpy arrays (lists of numbers)
        elif isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            try:
                obj[key] = np.array(value)  # Convert list back to numpy array
            except ValueError:
                # If conversion fails, leave as list
                pass

        # Handle sets (convert lists back to sets)
        elif isinstance(value, list) and all(isinstance(i, (int, float, str)) for i in value):
            obj[key] = set(value)  # Convert list back to set

        # Handle datetime strings
        elif isinstance(value, str):
            try:
                obj[key] = datetime.fromisoformat(value)  # Convert ISO 8601 string back to datetime
            except ValueError:
                # If conversion fails, leave as string
                obj[key] = str(value)

        # Handle floats explicitly
        elif isinstance(value, float):
            obj[key] = float(value)  # Ensure it's a float (redundant but explicit)

        # Handle integers explicitly
        elif isinstance(value, int) and value is not bool:
            obj[key] = int(value)  # Ensure it's an integer (redundant but explicit)

    return obj


def read_json(file_path):
    """
    Reads a regular JSON file and returns the data.

    Parameters:
        file_path (str): Path to the .json file.

    Returns:
        dict or list: Data from the file.
    """
    norm_path = os.path.normpath(file_path)
    try:
        with open(norm_path, 'r', encoding='utf-8') as f:
            data = json.load(f, object_hook=custom_deserializer)
        return data
    except Exception:
        logger.error("read_json Error", exc_info=True)
        return None


def write_json(data, file_path):
    """
    Writes data to a regular JSON file.

    Parameters:
        data (dict or list): Data to write to the file.
        file_path (str): Path to the .json file.

    Returns:
        None
    """
    norm_path = os.path.normpath(file_path)
    try:
        with open(norm_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=custom_serializer)
        print(f"Data written to {norm_path}")
    except Exception:
        logger.error("write_json Error", exc_info=True)


def read_compressed_json(file_path):
    """
    Reads a compressed JSON file (.json.gz) and returns the data.

    Parameters:
        file_path (str): Path to the .json.gz file.

    Returns:
        list: List of dictionaries containing the data from the file.
    """
    norm_path = os.path.normpath(file_path)
    try:
        with gzip.open(norm_path, 'rt', encoding='utf-8') as f:
            data = json.load(f, object_hook=custom_deserializer)
        return data
    except Exception:
        logger.error("read_compressed_json Error", exc_info=True)
        return None


def write_compressed_json(data, file_path, print_path=True):
    """
    Writes data to a compressed JSON file (.json.gz).

    Parameters:
        data (dict or list): Data to write to the file.
        file_path (str): Path to the .json.gz file.

    Returns:
        None
    """
    norm_path = os.path.normpath(file_path)
    success = False
    for i in range(10):
        try:
            with gzip.open(norm_path, 'wt', encoding='utf-8') as f:
                try:
                    json.dump(data, f, indent=4, default=custom_serializer)
                    success = True
                    break
                except Exception:
                    logger.error("write_compressed_json Error", exc_info=True)
                    time.sleep(0.01)
        except Exception:
            logger.error("write_compressed_json Error", exc_info=True)

    if success:
        if print_path and i == 0:
            logger.info("File Saved to %s", norm_path)
        if i > 0:
            logger.info("File saved to %s after %d attempts", norm_path, i)


def save_batch_to_npz(batch_data, output_path):
    """
    Saves batch data to a .npz file.

    Parameters:
        batch_data (list of dict): List of dictionaries containing image names and binary arrays.
        output_path (str): Path to save the .npz file.

    Returns:
        None
    """
    try:
        # Prepare data for saving
        data_dict = {item["image_name"]: np.array(item["binary_array"], dtype=np.uint8) for item in batch_data}

        # Save data to .npz format
        np.savez_compressed(output_path, allow_pickle=True, **data_dict)
        logger.info("Batch data saved successfully to: %s", output_path)
        print()
    except Exception as e:
        logger.error("An error occurred while saving batch data: ", exc_info=True)


def read_batch_from_npz(input_path):
    """
    Reads batch data from a .npz file.

    Parameters:
        input_path (str): Path to the .npz file.

    Returns:
        list of dict: List of dictionaries containing image names and binary arrays.
    """
    try:
        # Load data from .npz file
        data_dict = np.load(input_path)

        # Convert data back to list of dictionaries
        batch_data = [{"image_name": key, "binary_array": data_dict[key].tolist()} for key in data_dict.files]
        logger.info("Batch data loaded successfully from: %s", input_path)
        return batch_data
    except Exception as e:
        logger.error("An error occurred while reading batch data: %s", e, exc_info=True)
        return None


def extract_detailed_video_metadata(video_path):
    """
    Extracts detailed metadata from a video file using ExifTool, including general video metadata,
    camera metadata, and date/time metadata.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary containing metadata fields. Missing fields will have a value of None.
    """
    try:
        # Run ExifTool to extract metadata
        result = subprocess.run(
            [
                "exiftool",
                "-FileName",
                "-FileSize",
                "-FileFormat",
                "-Duration",
                "-VideoFrameRate",
                "-ImageWidth",
                "-ImageHeight",
                "-AspectRatio",
                "-VideoBitrate",
                "-Compression",
                "-Make",
                "-Model",
                "-SerialNumber",
                "-LensMake",
                "-LensModel",
                "-FocalLength",
                "-Aperture",
                "-ISO",
                "-ShutterSpeed",
                "-WhiteBalance",
                "-ExposureMode",
                "-MeteringMode",
                "-FocusMode",
                "-ImageStabilization",
                "-CreateDate",
                "-ModifyDate",
                "-MediaCreateDate",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Initialize metadata dictionary with all fields set to None
        metadata = {
            # General Video Metadata
            'file_name': None,
            'file_size': None,
            'file_format': None,
            'duration': None,
            'frame_rate': None,
            'resolution': None,
            'aspect_ratio': None,
            'bitrate': None,
            'compression': None,
            # Camera Metadata
            'camera_make': None,
            'camera_model': None,
            'camera_serial_number': None,
            'lens_make': None,
            'lens_model': None,
            'focal_length': None,
            'aperture': None,
            'iso': None,
            'shutter_speed': None,
            'white_balance': None,
            'exposure_mode': None,
            'metering_mode': None,
            'focus_mode': None,
            'image_stabilization': None,
            # Date and Time Metadata
            'create_date': None,
            'modify_date': None,
            'media_create_date': None,
        }

        # Parse the output to populate the metadata dictionary
        for line in result.stdout.splitlines():
            if "File Name" in line:
                metadata['file_name'] = line.split(": ", 1)[1].strip()
            elif "File Size" in line:
                metadata['file_size'] = line.split(": ", 1)[1].strip()
            elif "File Format" in line:
                metadata['file_format'] = line.split(": ", 1)[1].strip()
            elif "Duration" in line:
                duration_str = line.split(": ", 1)[1].strip()
                parts = duration_str.split(":")
                if len(parts) == 3:  # Format is hours:minutes:seconds
                    hours, minutes, seconds = map(float, parts)
                    metadata['duration'] = hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:  # Format is minutes:seconds
                    minutes, seconds = map(float, parts)
                    metadata['duration'] = minutes * 60 + seconds
            elif "Video Frame Rate" in line:
                metadata['frame_rate'] = float(line.split(": ", 1)[1].strip().split(" ")[0])
            elif "Image Width" in line:
                width = int(line.split(": ", 1)[1].strip())
                metadata['resolution'] = f"{width}x"  # Start resolution string
            elif "Image Height" in line:
                height = int(line.split(": ", 1)[1].strip())
                if metadata['resolution']:
                    metadata['resolution'] += f"{height}"  # Complete resolution string
            elif "Aspect Ratio" in line:
                metadata['aspect_ratio'] = line.split(": ", 1)[1].strip()
            elif "Video Bitrate" in line:
                metadata['bitrate'] = line.split(": ", 1)[1].strip()
            elif "Compression" in line:
                metadata['compression'] = line.split(": ", 1)[1].strip()
            elif "Make" in line:
                metadata['camera_make'] = line.split(": ", 1)[1].strip()
            elif "Model" in line:
                metadata['camera_model'] = line.split(": ", 1)[1].strip()
            elif "Serial Number" in line:
                metadata['camera_serial_number'] = line.split(": ", 1)[1].strip()
            elif "Lens Make" in line:
                metadata['lens_make'] = line.split(": ", 1)[1].strip()
            elif "Lens Model" in line:
                metadata['lens_model'] = line.split(": ", 1)[1].strip()
            elif "Focal Length" in line:
                metadata['focal_length'] = line.split(": ", 1)[1].strip()
            elif "Aperture" in line:
                metadata['aperture'] = line.split(": ", 1)[1].strip()
            elif "ISO" in line:
                metadata['iso'] = int(line.split(": ", 1)[1].strip())
            elif "Shutter Speed" in line:
                metadata['shutter_speed'] = line.split(": ", 1)[1].strip()
            elif "White Balance" in line:
                metadata['white_balance'] = line.split(": ", 1)[1].strip()
            elif "Exposure Mode" in line:
                metadata['exposure_mode'] = line.split(": ", 1)[1].strip()
            elif "Metering Mode" in line:
                metadata['metering_mode'] = line.split(": ", 1)[1].strip()
            elif "Focus Mode" in line:
                metadata['focus_mode'] = line.split(": ", 1)[1].strip()
            elif "Image Stabilization" in line:
                metadata['image_stabilization'] = line.split(": ", 1)[1].strip()
            elif "Create Date" in line:
                metadata['create_date'] = line.split(": ", 1)[1].strip()
            elif "Modify Date" in line:
                metadata['modify_date'] = line.split(": ", 1)[1].strip()
            elif "Media Create Date" in line:
                metadata['media_create_date'] = line.split(": ", 1)[1].strip()

        return metadata
    except Exception as e:
        print(f"Error extracting metadata with ExifTool: {e}")
        return None


def accelerate_video_ffmpeg_no_audio(input_path, output_path, playback_speed):
    """
    Accelerates the playback speed of a video using ffmpeg, removes audio, and saves the output.

    Parameters:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the output video file.
        playback_speed (float): Factor by which to accelerate the video (e.g., 3.0 for triple speed).
    """
    try:
        if playback_speed <= 0:
            raise ValueError("Playback speed must be a positive number.")

        # Calculate the video filter for speed adjustment
        video_filter = f"setpts={1/playback_speed}*PTS"

        # Build the ffmpeg command to remove audio and adjust video speed
        command = [
            "ffmpeg",
            "-i",
            input_path,  # Input file
            "-filter:v",
            video_filter,  # Video speed adjustment
            "-an",  # Remove audio
            "-preset",
            "fast",  # Encoding preset for faster processing
            output_path,  # Output file
        ]

        # Run the ffmpeg command
        subprocess.run(command, check=True)
        logger.info("Video saved successfully at %s", output_path)
    except subprocess.CalledProcessError as e:
        logger.error("An error occurred while processing the video: %s", e, exc_info=True)
    except Exception as e:
        logger.error("An error occurred: %s", e, exc_info=True)


def write_to_hdf5(data, filename):
    """
    Write a dictionary, list of dictionaries, or set of dictionaries to an HDF5 file.

    Parameters:
        data (dict, list, set): The data to write to the HDF5 file.
        filename (str): The name of the HDF5 file to write to.
    """

    def write_data(group, key, value):
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            for subkey, subvalue in value.items():
                write_data(subgroup, subkey, subvalue)
        elif isinstance(value, list):
            subgroup = group.create_group(key)
            for i, item in enumerate(value):
                write_data(subgroup, str(i), item)
        elif isinstance(value, set):
            subgroup = group.create_group(key)
            for i, item in enumerate(sorted(value)):  # Convert set to sorted list for consistent storage
                write_data(subgroup, str(i), item)
        elif isinstance(value, np.ndarray):
            group.create_dataset(key, data=value)
        elif isinstance(value, datetime):
            group.create_dataset(key, data=value.isoformat())
        elif isinstance(value, (float, int, str)):
            group.create_dataset(key, data=value)
        else:
            raise TypeError(f"Unsupported data type: {type(value)}")

    with h5py.File(filename, 'w') as hdf5_file:
        write_data(hdf5_file, 'root', data)


def read_from_hdf5(filename):
    """
    Read data from an HDF5 file and reconstruct it as a dictionary, list of dictionaries, or set of dictionaries.

    Parameters:
        filename (str): The name of the HDF5 file to read from.

    Returns:
        dict, list, set: The reconstructed data.
    """

    def read_data(group):
        if isinstance(group, h5py.Dataset):
            data = group[()]
            if isinstance(data, bytes):  # Handle string decoding
                data = data.decode('utf-8')
            try:
                # Attempt to parse datetime strings
                return datetime.fromisoformat(data)
            except (ValueError, TypeError):
                return data
        elif isinstance(group, h5py.Group):
            if all(key.isdigit() for key in group.keys()):  # Check if keys are numeric (list-like structure)
                items = [read_data(group[key]) for key in sorted(group.keys(), key=int)]
                return set(items) if isinstance(items[0], set) else items
            else:
                return {key: read_data(group[key]) for key in group.keys()}
        else:
            raise TypeError(f"Unsupported HDF5 group type: {type(group)}")

    with h5py.File(filename, 'r') as hdf5_file:
        return read_data(hdf5_file['root'])


def save_hdf5_datasets_compressed(
    data: list | dict, datasets: list, file: str, compression: str = "gzip", compression_level: int = 4
):
    """
    Saves data to HDF5 file in a compressed format.

    Parameters:
        data (list or dict): List of data arrays to save or dictionary that can be a single level nested dictionary.
        datasets (list): List of dataset names corresponding to the data.
        file (str): Path to the HDF5 file.
        compression (str): Compression algorithm to use (default: "gzip").
        compression_level (int): Compression level (default: 4).
    """

    def sanitize_name(name):
        return str(name).replace("(", "").replace(")", "").replace(",", "_").replace(" ", "_")

    def save_dict_to_group(group, data):
        for key, value in data.items():
            sanitized_key = sanitize_name(key)
            if isinstance(value, dict):
                subgroup = group.create_group(sanitized_key)
                save_dict_to_group(subgroup, value)
            elif isinstance(value, (int, float, str)):
                group.create_dataset(sanitized_key, data=value)
            else:
                group.create_dataset(
                    sanitized_key, data=value, compression=compression, compression_opts=compression_level, chunks=True
                )

    try:
        if isinstance(data, dict):
            with h5py.File(file, "w") as hf:
                if isinstance(data, dict):
                    for pixel_coord, pixel_data in data.items():
                        if not pixel_data:
                            continue
                        group_name = sanitize_name(pixel_coord)
                        group = hf.create_group(group_name)
                        if isinstance(pixel_data, list) and all(isinstance(item, dict) for item in pixel_data):
                            for idx, dict_data in enumerate(pixel_data):
                                subgroup_name = f"entry_{idx}"
                                subgroup = group.create_group(subgroup_name)
                                save_dict_to_group(subgroup, dict_data)
                        elif isinstance(pixel_data, dict):
                            save_dict_to_group(group, pixel_data)
                elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    for idx, dict_data in enumerate(data):
                        group_name = f"group_{idx}"
                        group = hf.create_group(group_name)
                        save_dict_to_group(group, dict_data)
                elif isinstance(data, list):
                    for d, dataset in zip(data, datasets):
                        if dataset in hf:
                            del hf[dataset]
                        hf.create_dataset(
                            dataset, data=d, compression=compression, compression_opts=compression_level, chunks=True
                        )

        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Handle list of dictionaries
            with h5py.File(file, "w") as hf:
                for idx, dict_data in enumerate(data):
                    group_name = f"group_{idx}"  # Create a unique group name for each dictionary
                    group = hf.create_group(group_name)
                    for dataset_name, dataset_value in dict_data.items():
                        # Check if dataset_value is scalar or array-like
                        if isinstance(dataset_value, (int, float, str)):
                            # Handle scalar data
                            group.create_dataset(dataset_name, data=dataset_value)
                        else:
                            # Handle array-like data
                            group.create_dataset(
                                dataset_name,
                                data=dataset_value,
                                compression=compression,
                                compression_opts=compression_level,
                            )

        elif isinstance(data, list):
            with h5py.File(file, "a") as hf:
                # Loop through datasets
                for d, dataset in zip(data, datasets):
                    if dataset in hf:
                        # Delete dataset if it already exists
                        del hf[dataset]
                    # Write dataset with compression
                    hf.create_dataset(
                        dataset, data=d, compression=compression, compression_opts=compression_level, chunks=True
                    )

        hf.close()

    except Exception as e:
        logger.error("Error writing to hdf5 file %s", e, exc_info=True)


def read_hdf5_datasets(file: str, datasets: list = None):
    """
    Reads data from a compressed HDF5 file, including nested groups.

    Parameters:
        file (str): Path to the HDF5 file.
        datasets (list): List of dataset names to read. If empty or None, reads all datasets.

    Returns:
        dict: A dictionary where keys are dataset names or group names, and values are the corresponding data arrays.
    """

    def read_group(group):
        """Recursively reads datasets and groups."""
        group_data = {}
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # Recursively read nested groups
                group_data[key] = read_group(item)
            else:
                # Check if dataset_value is scalar or array-like
                if isinstance(item[()], (int, float, str)):
                    group_data[key] = item[()]
                elif isinstance(item[()], bytes):
                    group_data[key] = item[()].decode('utf-8')
                else:
                    # Read dataset
                    group_data[key] = item[:]
        return group_data

    data_dict = {}
    with h5py.File(file, "r") as f:
        if datasets:
            # Read specific datasets
            for dataset in datasets:
                if dataset in f:
                    data_dict[dataset] = f[dataset][:]
                else:
                    raise KeyError(f"Dataset '{dataset}' not found in file '{file}'.")
        else:
            # Read all datasets and groups
            data_dict = read_group(f)

    return data_dict


logger = logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"), log_file_name="error_log_lookback_tools.txt", log_type=ERROR
)


'''
def main():
    pass


if __name__ == "__main__":
    checkpoint_folder = os.getcwd()
    logger = logging_setup(
        log_folder=os.path.join(checkpoint_folder, "error_logs"),
        log_file_name="error_log_lookback_tools.txt",
        log_type=ERROR,
    )
    main()

'''
