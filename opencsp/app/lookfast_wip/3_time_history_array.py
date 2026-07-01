import os
import json
import gzip
import time
import cv2
from multiprocessing import Pool
import multiprocessing
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# from opencsp.common.lib.tool.log_tools import multiprocessing_logger
import opencsp.app.lookback.lookback_tools as lbt
from logging import DEBUG, ERROR

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd, "error_logs"),
    log_file_name="error_log_time_history_array_debug_mp.txt",
    log_type=DEBUG,
)


def process_image(image_path, fraction):
    """
    Processes a single image to create a binary array based on a threshold.

    Parameters:
        image_path (str): Path to the image file.
        fraction (float): Percentage (0-1) of the maximum pixel value to use as the threshold.

    Returns:
        dict: Dictionary containing the image name and binary array.
    """
    # Load the image using PIL and convert it to grayscale
    image = Image.open(image_path).convert('L')  # 'L' mode converts to grayscale

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Calculate the threshold as a percentage of the maximum pixel value
    max_pixel_value = image_array.max()
    threshold = fraction * max_pixel_value

    # Apply the threshold to create a binary array
    binary_array = (image_array > threshold).astype(np.uint8)  # 1 for pixels > threshold, 0 otherwise

    # Return the image name and binary array
    return {"image_name": os.path.basename(image_path), "binary_array": binary_array.tolist()}


def create_binary_pixel_array_parallel(
    folder_path,
    percentages,
    output_folder,
    checkpoint_folder,
    logger,
    batch_size=1000,
    checkpoint_file="time_history_array_checkpoint.json",
):
    """
    Reads all images from a folder, processes them in parallel in batches, calculates a threshold based on a percentage
    of the maximum pixel value for each image, and saves the results to disk in batches as compressed JSON files. Supports restarting from a checkpoint.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        percentages (list of float): List of percentages (0-1) of the maximum pixel value to use as thresholds.
        output_folder (str): Folder to save the structured binary arrays to disk.
        checkpoint_folder (str): Path to the folder with checkpoints for this dataset.
        batch_size (int): Number of images to process in each batch.
        checkpoint_file (str): Checkpoint file name.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": {str(int(p * 100)): [] for p in percentages}}

    # Get all image file paths from the folder
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    if not image_paths:
        raise ValueError("No valid image files found in the specified folder.")

    for percentage in percentages:
        percentage_key = str(int(percentage * 100))
        os.makedirs(os.path.join(output_folder, percentage_key), exist_ok=True)
        print(f"Processing images for threshold percentage: {percentage * 100}%")

        # Process images in batches
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            batch_index = int(batch_start // batch_size + 1)
            if batch_index in checkpoint_data["processed_batches"][percentage_key]:
                print(f"Skipping already processed batch {batch_index} for threshold {percentage * 100}%.")
                continue

            batch_data = []
            with ProcessPoolExecutor(max_workers=3) as executor:
                # Use tqdm to wrap the executor.map for progress tracking
                for result in tqdm(
                    executor.map(process_image_bit_depth, batch_paths, [percentage] * len(batch_paths)),
                    total=len(batch_paths),
                    desc=f"Processing Batch {batch_index} (Threshold {percentage * 100}%)",
                ):
                    batch_data.append(result)

            # Save the batch to disk as a compressed JSON file
            batch_output_path = os.path.join(
                output_folder, percentage_key, f"threshold_{int(percentage * 100):03d}_batch_{batch_index:04d}.json.gz"
            )
            lbt.write_compressed_json(batch_data, batch_output_path, print_path=False)
            logger.info("Batch %s written to %s", str(batch_index), batch_output_path)
            # with gzip.open(batch_output_path, 'wt', encoding='utf-8') as f:
            #     json.dump(batch_data, f)
            # print(f"Batch saved to {batch_output_path}")

            # Update checkpoint
            checkpoint_data["processed_batches"][percentage_key].append(batch_index)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        print(f"Finished processing for threshold {percentage * 100}%.")


def process_image_bit_depth(image_path, fraction):
    """
    Processes a single image to create a binary array based on a threshold.

    Parameters:
        image_path (str): Path to the image file.
        fraction (float): Percentage (0-1) of the maximum possible pixel value to use as the threshold.

    Returns:
        dict: Dictionary containing the image name and binary array.
    """
    # Load the image using PIL and convert it to grayscale
    image = Image.open(image_path).convert('L')  # 'L' mode converts to grayscale

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Determine the maximum possible pixel value based on the bit depth
    # 'L' mode corresponds to 8-bit grayscale (0-255)
    # 'I' mode corresponds to 16-bit grayscale (0-65535)
    if image.mode == 'L':
        max_possible_pixel_value = 255
    elif image.mode == 'I':
        max_possible_pixel_value = 65535
    else:
        raise ValueError(f"Unsupported image mode: {image.mode}")

    # Calculate the threshold as a percentage of the maximum possible pixel value
    threshold = fraction * max_possible_pixel_value

    # Apply the threshold to create a binary array
    binary_array = (image_array > threshold).astype(np.uint8)  # 1 for pixels > threshold, 0 otherwise

    # Return the image name and binary array
    return {"image_name": os.path.basename(image_path), "binary_array": binary_array.tolist()}


def process_image_bit_depth_cv(image_path, fraction):
    """
    Processes a single image to create a binary array based on a threshold.
    """
    # Load the image using OpenCV and convert it to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Determine the maximum possible pixel value based on the bit depth
    max_possible_pixel_value = 255  # OpenCV loads images as 8-bit by default

    # Calculate the threshold as a percentage of the maximum possible pixel value
    threshold = fraction * max_possible_pixel_value

    # Apply the threshold to create a binary array
    binary_array = (image_array > threshold).astype(np.uint8)  # 1 for pixels > threshold, 0 otherwise

    # Return the image name and binary array
    return {"image_name": os.path.basename(image_path), "binary_array": binary_array.tolist()}


def process_batch(batch_index, batch_paths, percentage, output_folder, percentage_key):
    """
    Processes a single batch of images sequentially and saves the results to disk.

    Parameters:
        batch_index (int): Index of the batch being processed.
        batch_paths (list of str): List of image file paths in the batch.
        percentage (float): Threshold percentage for processing.
        output_folder (str): Folder to save the structured binary arrays to disk.
        percentage_key (str): Subfolder name for the current percentage.

    Returns:
        int: The batch index (for checkpoint updates).
    """
    batch_data = []
    for image_path in batch_paths:
        result = process_image_bit_depth_cv(image_path, percentage)
        batch_data.append(result)

    # Save the batch to disk as a compressed JSON file
    batch_output_path = os.path.join(
        output_folder, percentage_key, f"threshold_{int(percentage * 100):03d}_batch_{batch_index:04d}.json.gz"
    )

    # Save the batch to disk as a .npz file
    batch_output_path = os.path.join(
        output_folder, percentage_key, f"threshold_{int(percentage * 100):03d}_batch_{batch_index:04d}.npz"
    )
    lbt.save_batch_to_npz(batch_data, batch_output_path)
    # lbt.write_compressed_json(batch_data, batch_output_path, print_path=False)
    # with gzip.open(batch_output_path, 'wt', encoding='utf-8') as f:
    #     json.dump(batch_data, f)
    # print(f"Batch {batch_index} saved to {batch_output_path}")
    logger.info("Batch %s written to %s", str(batch_index), batch_output_path)

    return batch_index


def create_binary_pixel_array_parallel_with_overlap(
    folder_path,
    percentages,
    output_folder,
    checkpoint_folder,
    logger,
    batch_size=1000,
    overlap=1,  # Number of overlapping images between batches
    checkpoint_file="time_history_array_checkpoint.json",
):
    """
    Reads all images from a folder, processes them in parallel in overlapping batches, calculates a threshold based on a percentage
    of the maximum pixel value for each image, and saves the results to disk in batches as compressed JSON files. Supports restarting from a checkpoint.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        percentages (list of float): List of percentages (0-1) of the maximum pixel value to use as thresholds.
        output_folder (str): Folder to save the structured binary arrays to disk.
        checkpoint_folder (str): Path to the folder with checkpoints for this dataset.
        batch_size (int): Number of images to process in each batch.
        overlap (int): Number of overlapping images between batches.
        checkpoint_file (str): Checkpoint file name.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": {str(int(p * 100)): [] for p in percentages}}

    # Get all image file paths from the folder
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    image_paths.sort()

    if not image_paths:
        raise ValueError("No valid image files found in the specified folder.")

    for percentage in percentages:
        percentage_key = str(int(percentage * 100))
        os.makedirs(os.path.join(output_folder, percentage_key), exist_ok=True)
        print(f"Processing images for threshold percentage: {percentage * 100}%")

        # Create overlapping batches
        batches = []
        for batch_start in range(0, len(image_paths), batch_size - overlap):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batches.append((batch_start // (batch_size - overlap) + 1, image_paths[batch_start:batch_end]))

        # Process batches in parallel
        processed_batches = set(checkpoint_data["processed_batches"][percentage_key])
        futures = []
        with ProcessPoolExecutor(max_workers=3) as executor:
            for batch_index, batch_paths in batches:
                if batch_index in processed_batches:
                    print(f"Skipping already processed batch {batch_index} for threshold {percentage * 100}%.")
                    continue
                futures.append(
                    executor.submit(process_batch, batch_index, batch_paths, percentage, output_folder, percentage_key)
                )

            # Wait for all batches to complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
                batch_index = future.result()
                processed_batches.add(batch_index)

                # Update checkpoint after all batches are processed
                checkpoint_data["processed_batches"][percentage_key] = list(processed_batches)
                lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        print(f"Finished processing for threshold {percentage * 100}%.")


def process_batch_multiprocessing(args):
    """
    Wrapper function for multiprocessing to process a batch of images.
    """
    batch_index, batch_paths, percentage, output_folder, percentage_key, logger = args
    return process_batch(batch_index, batch_paths, percentage, output_folder, percentage_key, logger)


def create_binary_pixel_array_parallel_with_multiprocessing(
    folder_path,
    percentages,
    output_folder,
    checkpoint_folder,
    batch_size=1000,
    overlap=1,
    checkpoint_file="time_history_array_checkpoint.json",
    num_workers=4,  # Number of parallel processes
):
    """
    Processes images in parallel using multiprocessing.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": {str(int(p * 100)): [] for p in percentages}}

    # Get all image file paths from the folder
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    image_paths.sort()

    if not image_paths:
        raise ValueError("No valid image files found in the specified folder.")

    for percentage in percentages:
        percentage_key = str(int(percentage * 100))
        os.makedirs(os.path.join(output_folder, percentage_key), exist_ok=True)
        print(f"Processing images for threshold percentage: {percentage * 100}%")

        # Create overlapping batches
        batches = []
        for batch_start in range(0, len(image_paths), batch_size - overlap):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batches.append((batch_start // (batch_size - overlap) + 1, image_paths[batch_start:batch_end]))

        # Process batches in parallel using multiprocessing
        processed_batches = set(checkpoint_data["processed_batches"][percentage_key])
        args = [
            (batch_index, batch_paths, percentage, output_folder, percentage_key, logger)
            for batch_index, batch_paths in batches
            if batch_index not in processed_batches
        ]

        with Pool(processes=num_workers) as pool:
            for batch_index in tqdm(
                pool.imap_unordered(process_batch_multiprocessing, args), total=len(args), desc="Processing Batches"
            ):
                processed_batches.add(batch_index)

                # Update checkpoint after all batches are processed
                checkpoint_data["processed_batches"][percentage_key] = list(processed_batches)
                lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        print(f"Finished processing for threshold {percentage * 100}%.")


if __name__ == "__main__":
    # Example usage:

    # Assuming you have a folder containing images, a percentage value, and an output file path
    folder_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/3_specific_cropped_frames"
    output_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/6_time_history_output"
    checkpoint_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints"
    checkpoint_file_name = "time_history_array_checkpoint_mp.json"

    num_workers = multiprocessing.cpu_count() - 20  # Use all but xx cores

    percentage = [0.5]

    create_binary_pixel_array_parallel_with_multiprocessing(
        folder_path=folder_path,
        percentages=percentage,
        output_folder=output_folder,
        checkpoint_folder=checkpoint_folder,
        batch_size=500,
        overlap=1,
        checkpoint_file=checkpoint_file_name,
        num_workers=num_workers,
    )

    '''
    percentage = [
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
    ]  # Example: threshold 0.7 is 70% of the maximum pixel value
    
    create_binary_pixel_array_parallel_with_overlap(
        folder_path,
        percentage,
        output_folder,
        checkpoint_folder=checkpoint_folder,
        batch_size=500,
        overlap=1,
        checkpoint_file=checkpoint_file_name,
    )
    '''
