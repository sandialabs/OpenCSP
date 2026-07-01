import os
from logging import DEBUG, ERROR
from multiprocessing import Pool
import cv2
import numpy as np
from tqdm import tqdm
import h5py

# from opencsp.common.lib.tool.log_tools import multiprocessing_logger
import opencsp.app.lookback.lookback_tools as lbt
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.hdf5_tools as h5

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_time_history_array_debug_mp.txt",
    log_type=DEBUG,
)


def process_image_bit_depth_cv(image_path, fraction):
    """
    Processes a single image to create a binary array based on a threshold.
    """
    # Load the image using OpenCV and convert it to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Determine the maximum possible pixel value based on the bit depth
    max_intensity = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 255
    # max_possible_pixel_value = 255  # OpenCV loads images as 8-bit by default

    # Calculate the threshold as a percentage of the maximum possible pixel value
    threshold = fraction * max_intensity

    # Apply the threshold to create a binary array
    binary_array = (image_array > threshold).astype(bool)  # 1 for pixels > threshold, 0 otherwise

    # Return the image name and binary array
    return os.path.basename(image_path), binary_array.tolist()


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
    img_names = []
    for image_path in batch_paths:
        result = process_image_bit_depth_cv(image_path, percentage)
        batch_data.append(result[1])
        img_names.append(result[0])
        del result

    # Save the batch to disk as a compressed numpy file
    batch_output_path = ft.join(
        output_folder, percentage_key, f"threshold_{int(percentage * 100):03d}_batch_{batch_index:04d}.npz"
    )

    np.savez_compressed(batch_output_path, np.array(batch_data))
    _, batch_name, _ = ft.path_components(batch_output_path)
    lbt.write_json(img_names, ft.join(output_folder, percentage_key, batch_name + ".json"))
    logger.info("Batch %s written to %s", str(batch_index), batch_output_path)
    return batch_index


def process_batch_multiprocessing(args):
    """
    Wrapper function for multiprocessing to process a batch of images.
    """
    batch_index, batch_paths, percentage, output_folder, percentage_key = args
    return process_batch(batch_index, batch_paths, percentage, output_folder, percentage_key)


def create_binary_pixel_array_parallel_with_multiprocessing(
    image_folder_path,
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
        os.path.join(image_folder_path, file)
        for file in os.listdir(image_folder_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]
    image_paths.sort()

    if not image_paths:
        raise ValueError("No valid image files found in the specified folder.")

    for percentage in percentages:
        percentage_key = f"{int(percentage * 100):02d}"
        os.makedirs(os.path.join(output_folder, percentage_key), exist_ok=True)
        print(f"Processing images for threshold percentage: {int(percentage * 100):02d}%")

        # Create overlapping batches
        batches = []
        for batch_start in range(0, len(image_paths), batch_size - overlap):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batches.append((batch_start // (batch_size - overlap) + 1, image_paths[batch_start:batch_end]))

        # Process batches in parallel using multiprocessing
        if percentage_key in checkpoint_data["processed_batches"]:
            processed_batches = set(checkpoint_data["processed_batches"][percentage_key])
        else:
            checkpoint_data["processed_batches"][percentage_key] = []
            processed_batches = set(checkpoint_data["processed_batches"][percentage_key])

        args = [
            (batch_index, batch_paths, percentage, output_folder, percentage_key)
            for batch_index, batch_paths in batches
            if batch_index not in processed_batches
        ]

        with Pool(processes=num_workers) as pool:
            for batch_index in tqdm(
                pool.imap_unordered(process_batch_multiprocessing, args), total=len(args), desc="Time History Arrays"
            ):
                processed_batches.add(batch_index)

                # Update checkpoint after all batches are processed
                checkpoint_data["processed_batches"][percentage_key] = list(processed_batches)
                lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        print(f"Finished processing for threshold {int(percentage * 100):02d}%.")
