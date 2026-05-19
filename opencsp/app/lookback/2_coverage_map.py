import os
import time
import numpy as np
import imageio.v2 as imageio
import rawpy  # Import rawpy for processing RAW image files
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json

# from opencsp.common.lib.tool.log_tools import multiprocessing_logger
import opencsp.app.lookback.lookback_tools as lbt
from logging import DEBUG, ERROR

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd, "error_logs"), log_file_name="error_log_coverage_map_debug.txt", log_type=ERROR
)


def process_image(image_file, threshold_fractions, max_intensity, is_raw):
    """
    Processes a single image to create binary maps for the given thresholds.

    Parameters:
        image_file (str): Path to the image file.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        max_intensity (float): Maximum possible intensity value for the image format.
        is_raw (bool): Whether the image is a RAW file.

    Returns:
        dict: A dictionary where keys are threshold fractions and values are binary maps for the image.
    """
    try:
        # Read the image
        if is_raw:
            with rawpy.imread(image_file) as raw:
                image = raw.postprocess()
        else:
            image = imageio.imread(image_file)

        # Handle multi-channel images (e.g., RGB)
        if len(image.shape) == 3:  # Multi-channel image
            # Calculate pixel intensity as the average across channels
            pixel_intensity = np.mean(image, axis=-1)
        else:  # Grayscale image
            pixel_intensity = image

        # Create binary maps for each threshold
        binary_maps = {}
        for threshold_fraction in threshold_fractions:
            intensity_threshold = threshold_fraction * max_intensity
            binary_maps[threshold_fraction] = (pixel_intensity > intensity_threshold).astype(np.uint8)

        return binary_maps

    except Exception as e:
        logger.error("Error processing %s : %s", image_file, e, exc_info=True)
        return None


def process_images_in_batches(
    image_files, threshold_fractions, max_intensity, is_raw, batch_size, output_folder, prefix
):
    """
    Processes images in batches to create binary maps for the given thresholds.

    Parameters:
        image_files (list of str): List of image file paths to process.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        max_intensity (float): Maximum possible intensity value for the image format.
        is_raw (bool): Whether the images are RAW files.
        batch_size (int): Number of images to process in each batch.
        output_folder (str): Path to location for binary map output.
        prefix (str): Prefix for output filenames.

    Returns:
        None
    """
    # Initialize binary maps for each threshold
    first_image = imageio.imread(image_files[0]) if not is_raw else rawpy.imread(image_files[0]).postprocess()
    binary_maps = {
        threshold: np.zeros((first_image.shape[0], first_image.shape[1]), dtype=np.uint8)
        for threshold in threshold_fractions
    }

    # Process images in batches
    for batch_start in range(0, len(image_files), batch_size):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_image, image_file, threshold_fractions, max_intensity, is_raw): image_file
                for image_file in batch_files
            }

            # Initialize tqdm progress bar for the batch
            with tqdm(
                total=len(batch_files), desc=f"Processing Batch {batch_start // batch_size + 1}", unit="image"
            ) as pbar:
                for future in futures:
                    result = future.result()
                    if result:
                        for threshold_fraction in threshold_fractions:
                            binary_maps[threshold_fraction] = np.maximum(
                                binary_maps[threshold_fraction], result[threshold_fraction]
                            )
                    pbar.update(1)

    # Save binary maps for the batch
    for threshold_fraction, binary_map in binary_maps.items():
        output_path = os.path.join(output_folder, f"{prefix}_binary_map_{int(threshold_fraction * 100)}.png")
        imageio.imwrite(output_path, binary_map * 255)  # Scale binary map to 0-255 for saving as an image
        print(f"Binary map for threshold {threshold_fraction} saved to {output_path}")


def compile_binary_maps(output_folder, threshold_fractions, prefix):
    """
    Compiles all binary maps from batches into a single final output for each threshold.

    Parameters:
        output_folder (str): Path to the folder containing the binary map outputs.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        prefix (str): Prefix for identifying binary maps (e.g., "traditional" or "raw").

    Returns:
        None
    """
    # Initialize a dictionary to store the compiled binary maps
    compiled_binary_maps = {}

    # Iterate over each threshold fraction
    for threshold_fraction in threshold_fractions:
        compiled_map = None
        # Find all batch files for the current threshold
        batch_files = [
            os.path.join(output_folder, f)
            for f in os.listdir(output_folder)
            if f.startswith(f"{prefix}_binary_map_{int(threshold_fraction * 100)}")
        ]

        # Combine all batch files
        for batch_file in batch_files:
            binary_map = imageio.imread(batch_file) // 255  # Convert back to binary (0 or 1)
            if compiled_map is None:
                compiled_map = binary_map
            else:
                compiled_map = np.maximum(compiled_map, binary_map)  # Combine using logical OR

        # Store the compiled map
        compiled_binary_maps[threshold_fraction] = compiled_map

        # Save the compiled map to disk
        output_path = os.path.join(output_folder, f"{prefix}_compiled_binary_map_{int(threshold_fraction * 100)}.png")
        imageio.imwrite(output_path, compiled_map * 255)  # Scale binary map to 0-255 for saving as an image
        print(f"Compiled binary map for threshold {threshold_fraction} saved to {output_path}")


def construct_binary_maps_parallel(
    image_folder,
    output_folder,
    checkpoint_folder,
    threshold_fractions,
    batch_size=10,
    max_workers=4,
    checkpoint_file="coverage_map_checkpoint.json",
):
    """
    Constructs a series of binary maps of pixels that exceeded intensity thresholds across all images,
    including traditional formats and RAW image files, using parallel processing and batching.
    Supports restarting from a checkpoint.

    Parameters:
        image_folder (str): Path to the folder containing the sequence of images.
        output_folder (str): Path to location for binary map output.
        checkpoint_folder (str): Path to the checkpoint folder.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        batch_size (int): Number of images to process in each batch.
        max_workers (int): Maximum number of parallel threads to use.
        checkpoint_file (str): Checkpoint file name.

    Returns:
        None
    """
    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_images": [], "current_batch": 0, "completed_thresholds": [], "prefix": None}

    # List all image files in the folder
    image_files_all = os.listdir(image_folder)
    image_files = []
    image_files_raw = []
    for f in image_files_all:
        if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP")):
            image_files.append(os.path.join(image_folder, f))
        elif f.endswith((".nef", ".cr2", ".arw", ".dng", ".NEF", ".CR2", ".ARW", ".DNG")):
            image_files_raw.append(os.path.join(image_folder, f))

    if not image_files and not image_files_raw:
        raise ValueError("No image files (traditional or RAW) found in the specified folder.")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process traditional images
    if image_files:
        # Determine the maximum possible intensity value for traditional images
        first_image = imageio.imread(image_files[0])
        max_intensity = np.iinfo(first_image.dtype).max if np.issubdtype(first_image.dtype, np.integer) else 1.0

        # Filter out already processed images
        unprocessed_images = [img for img in image_files if img not in checkpoint_data["processed_images"]]

        # Process images in batches
        for batch_start in range(checkpoint_data["current_batch"], len(unprocessed_images), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_images))
            batch_files = unprocessed_images[batch_start:batch_end]
            # Extract file names
            file_names = [os.path.basename(file) for file in batch_files]

            process_images_in_batches(
                batch_files,
                threshold_fractions,
                max_intensity,
                is_raw=False,
                batch_size=batch_size,
                output_folder=output_folder,
                prefix="traditional",
            )

            # Update checkpoint
            checkpoint_data["processed_images"].extend(file_names)
            checkpoint_data["current_batch"] = batch_start + batch_size
            checkpoint_data["prefix"] = "traditional"
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        # Compile binary maps for traditional images
        if "traditional" not in checkpoint_data["completed_thresholds"]:
            compile_binary_maps(output_folder, threshold_fractions, prefix="traditional")
            checkpoint_data["completed_thresholds"].append("traditional")
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

    # Process RAW images
    if image_files_raw:
        # Determine the maximum possible intensity value for RAW images
        with rawpy.imread(image_files_raw[0]) as raw:
            first_image_raw = raw.postprocess()
        max_intensity_raw = (
            np.iinfo(first_image_raw.dtype).max if np.issubdtype(first_image_raw.dtype, np.integer) else 1.0
        )

        # Filter out already processed images
        unprocessed_images_raw = [img for img in image_files_raw if img not in checkpoint_data["processed_images"]]

        # Process images in batches
        for batch_start in range(checkpoint_data["current_batch"], len(unprocessed_images_raw), batch_size):
            batch_end = min(batch_start + batch_size, len(unprocessed_images_raw))
            batch_files = unprocessed_images_raw[batch_start:batch_end]
            # Extract file names
            file_names = [os.path.basename(file) for file in batch_files]

            process_images_in_batches(
                batch_files,
                threshold_fractions,
                max_intensity_raw,
                is_raw=True,
                batch_size=batch_size,
                output_folder=output_folder,
                prefix="raw",
            )

            # Update checkpoint
            checkpoint_data["processed_images"].extend(file_names)
            checkpoint_data["current_batch"] = batch_start + batch_size
            checkpoint_data["prefix"] = "raw"
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        # Compile binary maps for RAW images
        if "raw" not in checkpoint_data["completed_thresholds"]:
            compile_binary_maps(output_folder, threshold_fractions, prefix="raw")
            checkpoint_data["completed_thresholds"].append("raw")
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)


# Example usage
if __name__ == "__main__":
    checkpoint_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2842/0_checkpoints"
    checkpoint_file_name = "coverage_map_checkpoint.json"
    image_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2842/3_specific_cropped_frames"  # Replace with the path to your image folder
    output_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2842/4_coverage_map"
    # image_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/Nikon_CSOL_1_DSC2687_2826/1_video_frames"  # Replace with the path to your image folder
    # output_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/Nikon_CSOL_1_DSC2687_2826/4_coverage_map"

    threshold_fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    batch_size = 1000  # Process 1000 images per batch

    construct_binary_maps_parallel(
        image_folder,
        output_folder,
        checkpoint_folder,
        threshold_fractions,
        batch_size=batch_size,
        max_workers=4,
        checkpoint_file=checkpoint_file_name,
    )
