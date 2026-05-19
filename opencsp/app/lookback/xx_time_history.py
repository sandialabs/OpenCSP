import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import h5py  # Import h5py for HDF5 storage


def process_image(image_file, exiftool_path="exiftool"):
    """
    Processes a single image to extract timing information using ExifTool.

    Parameters:
        image_file (str): Path to the image file.
        exiftool_path (str): Path to the ExifTool executable.

    Returns:
        tuple: A tuple containing the image filename and its extracted timestamp, or None if an error occurs.
    """
    try:
        # Run ExifTool to extract metadata
        result = subprocess.run(
            [exiftool_path, "-DateTimeOriginal", image_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Check for errors
        if result.returncode != 0:
            print(f"Error processing {image_file}: {result.stderr}")
            return None

        # Parse the output to extract the timestamp
        for line in result.stdout.splitlines():
            if "Date/Time Original" in line:
                timestamp = line.split(": ", 1)[1].strip()
                return os.path.basename(image_file), timestamp

    except Exception as e:
        print(f"Error processing {image_file}: {e}")
        return None


def extract_frame_timing_to_hdf5(image_folder, hdf5_path, exiftool_path="exiftool", max_workers=4):
    """
    Extracts precise timing information from a series of video frames written to disk using ExifTool in parallel
    and stores the results in an HDF5 file.

    Parameters:
        image_folder (str): Path to the folder containing the extracted video frames.
        hdf5_path (str): Path to the HDF5 file where the timing information will be stored.
        exiftool_path (str): Path to the ExifTool executable (default assumes it's in the system PATH).
        max_workers (int): Maximum number of parallel threads to use.

    Returns:
        None
    """
    # List all image files in the folder
    image_files = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    )

    if not image_files:
        raise ValueError("No image files found in the specified folder.")

    # Use ThreadPoolExecutor for parallel processing
    frame_timing = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, image_file, exiftool_path): image_file for image_file in image_files}

        # Initialize tqdm progress bar
        with tqdm(total=len(image_files), desc="Extracting Timing Information", unit="image") as pbar:
            for future in futures:
                result = future.result()
                if result:
                    frame_timing[result[0]] = result[1]
                pbar.update(1)

    # Save the timing information to an HDF5 file
    with h5py.File(hdf5_path, "w") as hdf5_file:
        # Create a dataset for filenames
        hdf5_file.create_dataset("filenames", data=list(frame_timing.keys()), dtype=h5py.string_dtype())
        # Create a dataset for timestamps
        hdf5_file.create_dataset("timestamps", data=list(frame_timing.values()), dtype=h5py.string_dtype())

    print(f"Frame timing information saved to {hdf5_path}")

    '''
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


def construct_binary_maps_and_record_exceedance(image_folder, threshold_fractions, output_folder, max_workers=4):
    """
    Constructs a series of binary maps of pixels that exceeded intensity thresholds across all images,
    including traditional formats and RAW image files, using parallel processing. Records when each pixel
    exceeds the given threshold value in a multidimensional NumPy array and writes it to disk.

    Parameters:
        image_folder (str): Path to the folder containing the sequence of images.
        threshold_fractions (list of float): List of fractions of the maximum intensity value to use as thresholds.
        output_folder (str): Path to the folder where the NumPy arrays and binary maps will be saved.
        max_workers (int): Maximum number of parallel threads to use.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # List all traditional image files in the folder
    image_files = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"))
        ]
    )

    # List all RAW image files in the folder
    image_files_raw = sorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.endswith((".nef", ".cr2", ".arw", ".dng", ".NEF", ".CR2", ".ARW", ".DNG"))
        ]
    )

    if not image_files and not image_files_raw:
        raise ValueError("No image files (traditional or RAW) found in the specified folder.")

    # Process traditional images
    if image_files:
        first_image = imageio.imread(image_files[0])
        max_intensity = np.iinfo(first_image.dtype).max if np.issubdtype(first_image.dtype, np.integer) else 1.0

        binary_maps_traditional = {
            threshold: np.zeros((first_image.shape[0], first_image.shape[1]), dtype=np.uint8)
            for threshold in threshold_fractions
        }

        exceedance_array_traditional = np.zeros(
            (len(image_files), first_image.shape[0], first_image.shape[1]), dtype=np.uint8
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, image_file, threshold_fractions, max_intensity, False): image_file
                for image_file in image_files
            }

            # Initialize tqdm progress bar
            with tqdm(total=len(image_files), desc="Processing Traditional Images", unit="image") as pbar:
                for i, future in enumerate(futures):
                    result = future.result()
                    if result:
                        for threshold_fraction in threshold_fractions:
                            binary_maps_traditional[threshold_fraction] = np.maximum(
                                binary_maps_traditional[threshold_fraction], result[threshold_fraction]
                            )
                        exceedance_array_traditional[i] = np.any(
                            [result[threshold_fraction] for threshold_fraction in threshold_fractions], axis=0
                        )
                    pbar.update(1)

        # Save binary maps for traditional images
        for threshold_fraction, binary_map in binary_maps_traditional.items():
            output_path = os.path.join(output_folder, f"binary_map_traditional_{int(threshold_fraction * 100)}.png")
            imageio.imwrite(output_path, binary_map * 255)  # Scale binary map to 0-255 for saving as an image
            print(f"Binary map for threshold {threshold_fraction} (traditional) saved to {output_path}")

        # Save exceedance array for traditional images
        np.save(os.path.join(output_folder, "exceedance_array_traditional.npy"), exceedance_array_traditional)
        print(f"Exceedance array for traditional images saved to {output_folder}/exceedance_array_traditional.npy")

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

        exceedance_array_raw = np.zeros(
            (len(image_files_raw), first_image_raw.shape[0], first_image_raw.shape[1]), dtype=np.uint8
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_image, image_file, threshold_fractions, max_intensity_raw, True): image_file
                for image_file in image_files_raw
            }

            # Initialize tqdm progress bar
            with tqdm(total=len(image_files_raw), desc="Processing RAW Images", unit="image") as pbar:
                for i, future in enumerate(futures):
                    result = future.result()
                    if result:
                        for threshold_fraction in threshold_fractions:
                            binary_maps_raw[threshold_fraction] = np.maximum(
                                binary_maps_raw[threshold_fraction], result[threshold_fraction]
                            )
                        exceedance_array_raw[i] = np.any(
                            [result[threshold_fraction] for threshold_fraction in threshold_fractions], axis=0
                        )
                    pbar.update(1)

        # Save binary maps for RAW images
        for threshold_fraction, binary_map in binary_maps_raw.items():
            output_path = os.path.join(output_folder, f"binary_map_raw_{int(threshold_fraction * 100)}.png")
            imageio.imwrite(output_path, binary_map * 255)  # Scale binary map to 0-255 for saving as an image
            print(f"Binary map for threshold {threshold_fraction} (RAW) saved to {output_path}")

        # Save exceedance array for RAW images
        np.save(os.path.join(output_folder, "exceedance_array_raw.npy"), exceedance_array_raw)
        print(f"Exceedance array for RAW images saved to {output_folder}/exceedance_array_raw.npy")


# Example usage
if __name__ == "__main__":
    image_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/Nikon_CSOL_1_DSC2687_2826/1_video_frames"  # Replace with the path to your folder containing images
    output_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/Nikon_CSOL_1_DSC2687_2826/6_time_history_output"  # Replace with the path to your output folder
    threshold_fractions = [
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
    ]  # Example: 50%, 70%, and 90% of the maximum intensity value

    construct_binary_maps_and_record_exceedance(image_folder, threshold_fractions, output_folder, max_workers=4)

    '''


# Example usage
if __name__ == "__main__":
    image_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/1_video_frames"  # Replace with the path to your image folder
    hdf5_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/6_time_history_output/frame_timing.hdf5"  # Replace with the path to your folder containing extracted frames

    extract_frame_timing_to_hdf5(image_folder, hdf5_path, max_workers=4)
