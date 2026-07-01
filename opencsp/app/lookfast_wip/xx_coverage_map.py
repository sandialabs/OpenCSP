import os
import numpy as np
import imageio.v2 as imageio
import rawpy  # Import rawpy for processing RAW image files
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


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
        print(f"Error processing {image_file}: {e}")
        return None


def construct_binary_maps_parallel(image_folder, output_folder, threshold_fractions, max_workers=4):
    """
    Constructs a series of binary maps of pixels that exceeded intensity thresholds across all images,
    including traditional formats and RAW image files, using parallel processing.

    Parameters:
        image_folder (str): Path to the folder containing the sequence of images.
        output_folder (str): Path to location for binary map output
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        max_workers (int): Maximum number of parallel threads to use.

    Returns:
        None
    """
    # List all traditional image files in the folder
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

    # Initialize binary maps for each threshold using the first available image
    if image_files:
        first_image = imageio.imread(image_files[0])
        binary_maps = {
            threshold: np.zeros((first_image.shape[0], first_image.shape[1]), dtype=np.uint8)
            for threshold in threshold_fractions
        }
    elif image_files_raw:
        with rawpy.imread(image_files_raw[0]) as raw:
            first_image_raw = raw.postprocess()
            binary_maps_raw = {
                threshold: np.zeros((first_image_raw.shape[0], first_image_raw.shape[1]), dtype=np.uint8)
                for threshold in threshold_fractions
            }

    # Determine the maximum possible intensity value for the image format
    max_intensity = np.iinfo(first_image.dtype).max if np.issubdtype(first_image.dtype, np.integer) else 1.0

    # Combine all image files into a single list with a flag indicating whether they are RAW
    all_image_files = [(image_file, False) for image_file in image_files] + [
        (image_file, True) for image_file in image_files_raw
    ]

    # Process traditional images
    if image_files:
        first_image = imageio.imread(image_files[0])
        max_intensity = np.iinfo(first_image.dtype).max if np.issubdtype(first_image.dtype, np.integer) else 1.0

        binary_maps_traditional = {
            threshold: np.zeros((first_image.shape[0], first_image.shape[1]), dtype=np.uint8)
            for threshold in threshold_fractions
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, image_file, threshold_fractions, max_intensity, False): image_file
                for image_file in image_files
            }

            # Initialize tqdm progress bar
            with tqdm(total=len(image_files), desc="Processing Traditional Images", unit="image") as pbar:
                for future in futures:
                    result = future.result()
                    if result:
                        for threshold_fraction in threshold_fractions:
                            binary_maps_traditional[threshold_fraction] = np.maximum(
                                binary_maps_traditional[threshold_fraction], result[threshold_fraction]
                            )
                    pbar.update(1)

        # Save binary maps for traditional images
        for threshold_fraction, binary_map in binary_maps_traditional.items():
            output_path = os.path.join(output_folder, f"binary_map_traditional_{int(threshold_fraction * 100)}.png")
            imageio.imwrite(output_path, binary_map * 255)  # Scale binary map to 0-255 for saving as an image
            print(f"Binary map for threshold {threshold_fraction} (traditional) saved to {output_path}")

    # Process RAW images
    if image_files_raw:
        with rawpy.imread(image_files_raw[0]) as raw:
            first_image_raw = raw.postprocess()
        max_intensity_raw = (
            np.iinfo(first_image_raw.dtype).max if np.issubdtype(first_image_raw.dtype, np.integer) else 1.0
        )

        binary_maps_raw = {
            threshold: np.zeros((first_image_raw.shape[0], first_image_raw.shape[1]), dtype=np.uint8)
            for threshold in threshold_fractions
        }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, image_file, threshold_fractions, max_intensity_raw, True): image_file
                for image_file in image_files_raw
            }

            # Initialize tqdm progress bar
            with tqdm(total=len(image_files_raw), desc="Processing RAW Images", unit="image") as pbar:
                for future in futures:
                    result = future.result()
                    if result:
                        for threshold_fraction in threshold_fractions:
                            binary_maps_raw[threshold_fraction] = np.maximum(
                                binary_maps_raw[threshold_fraction], result[threshold_fraction]
                            )
                    pbar.update(1)

        # Save binary maps for RAW images
        for threshold_fraction, binary_map in binary_maps_raw.items():
            output_path = os.path.join(output_folder, f"binary_map_raw_{int(threshold_fraction * 100)}.png")
            imageio.imwrite(output_path, binary_map * 255)  # Scale binary map to 0-255 for saving as an image
            print(f"Binary map for threshold {threshold_fraction} (RAW) saved to {output_path}")
    # Use ThreadPoolExecutor for parallel processing


# Example usage
if __name__ == "__main__":
    image_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0051/1_video_frames"  # Replace with the path to your image folder
    output_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0051/4_coverage_map"

    # image_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/Nikon_CSOL_1_DSC2687_2826/1_video_frames"  # Replace with the path to your image folder
    # output_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/Nikon_CSOL_1_DSC2687_2826/4_coverage_map"

    threshold_fractions = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    construct_binary_maps_parallel(image_folder, output_folder, threshold_fractions, max_workers=4)
