import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

'''
# npz style file collection
def save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data):
    """
    Saves the checkpoint data to a JSON file.

    Parameters:
        checkpoint_file (str): Path to the checkpoint file.
        checkpoint_data (dict): Dictionary containing checkpoint information.

    Returns:
        None
    """
    with open(os.path.join(checkpoint_folder, checkpoint_file_name), 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    print(f"Checkpoint saved to {os.path.join(checkpoint_folder,checkpoint_file_name)}")


def load_checkpoint(checkpoint_folder, checkpoint_file_name):
    """
    Loads the checkpoint data from a JSON file.

    Parameters:
        checkpoint_file (str): Path to the checkpoint file.

    Returns:
        dict: Dictionary containing checkpoint information.
    """
    if os.path.exists(os.path.join(checkpoint_folder, checkpoint_file_name)):
        with open(os.path.join(checkpoint_folder, checkpoint_file_name), 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint loaded from {os.path.join(checkpoint_folder,checkpoint_file_name)}")
        return checkpoint_data
    else:
        return None


def process_image(image_path, fraction):
    """
    Processes a single image to create a binary array based on a threshold.

    Parameters:
        image_path (str): Path to the image file.
        fraction (float): Percentage (0-1) of the maximum pixel value to use as the threshold.

    Returns:
        np.ndarray: Binary array for the image.
    """
    # Load the image using PIL and convert it to grayscale
    image = Image.open(image_path).convert('L')  # 'L' mode converts to grayscale

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Calculate the threshold as a percentage of the maximum pixel value
    max_pixel_value = image_array.max()
    threshold = (fraction) * max_pixel_value

    # Apply the threshold to create a binary array
    binary_array = (image_array > threshold).astype(np.uint8)  # 1 for pixels > threshold, 0 otherwise

    return binary_array


def create_binary_pixel_array_parallel(
    folder_path,
    percentages,
    output_folder,
    checkpoint_folder,
    batch_size=1000,
    checkpoint_file="time_history_array_checkpoint.json",
):
    """
    Reads all images from a folder, processes them in parallel in batches, calculates a threshold based on a percentage
    of the maximum pixel value for each image, and saves the results to disk in batches. Supports restarting from a checkpoint.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        percentages (list of float): List of percentages (0-1) of the maximum pixel value to use as thresholds.
        output_folder (str): Folder to save the structured binary arrays to disk.
        checkpoint folder (str): Path to the folder with checkpoints for this dataset.
        batch_size (int): Number of images to process in each batch.
        checkpoint_file (str): Checkpoint file name.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_data = load_checkpoint(checkpoint_folder, checkpoint_file)
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

            binary_images = []
            with ProcessPoolExecutor() as executor:
                # Use tqdm to wrap the executor.map for progress tracking
                for binary_array in tqdm(
                    executor.map(process_image, batch_paths, [percentage] * len(batch_paths)),
                    total=len(batch_paths),
                    desc=f"Processing Batch {batch_index} (Threshold {percentage * 100}%)",
                ):
                    binary_images.append(binary_array)

            # Stack the batch of binary arrays along a new axis
            batch_array = np.stack(binary_images, axis=0)

            # Save the batch to disk with a unique filename
            batch_output_path = os.path.join(
                output_folder, percentage_key, f"threshold_{int(percentage * 100):03d}_batch_{batch_index:04d}.npz"
            )
            np.savez_compressed(batch_output_path, batch_array)
            print(f"Batch saved to {batch_output_path}")

            # Update checkpoint
            checkpoint_data["processed_batches"][percentage_key].append(batch_index)
            save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data)

        print(f"Finished processing for threshold {percentage * 100}%.")

        
# Time History Transition with npz files
def save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data):
    """
    Saves the checkpoint data to a JSON file.

    Parameters:
        checkpoint_file (str): Path to the checkpoint file.
        checkpoint_data (dict): Dictionary containing checkpoint information.

    Returns:
        None
    """
    with open(os.path.join(checkpoint_folder, checkpoint_file_name), 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    print(f"Checkpoint saved to {os.path.join(checkpoint_folder,checkpoint_file_name)}")


def load_checkpoint(checkpoint_folder, checkpoint_file_name):
    """
    Loads the checkpoint data from a JSON file.

    Parameters:
        checkpoint_file (str): Path to the checkpoint file.

    Returns:
        dict: Dictionary containing checkpoint information.
    """
    if os.path.exists(os.path.join(checkpoint_folder, checkpoint_file_name)):
        with open(os.path.join(checkpoint_folder, checkpoint_file_name), 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint loaded from {os.path.join(checkpoint_folder,checkpoint_file_name)}")
        return checkpoint_data
    else:
        return None


def read_compressed_json(file_path):
    """
    Reads a compressed JSON file (.json.gz) and returns the data.

    Parameters:
        file_path (str): Path to the .json.gz file.

    Returns:
        list: List of dictionaries containing the data from the file.
    """
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def write_pixel_results_to_disk(results, output_folder, output_file_name):
    """
    Writes the pixel brightness analysis results to a JSON file.

    Parameters:
        results (dict): Dictionary containing pixel brightness analysis results.
        output_file (str): Path to the output JSON file.

    Returns:
        None
    """
    # Convert datetime objects to ISO 8601 strings for JSON serialization
    serialized_results = {
        str(pixel): [
            {"transition": transition[0], "timestamp": transition[1].isoformat()} for transition in transitions
        ]
        for pixel, transitions in results.items()
    }

    # Write serialized results to a JSON file
    with open(os.path.join(output_folder, output_file_name), 'w') as f:
        json.dump(serialized_results, f, indent=4)
    print(f"Results written to {os.path.join(output_folder,output_file_name)}")


def process_npz_file(npz_file, pixel_locations, creation_datetime, frame_rate, global_frame_start):
    """
    Processes a single .npz file to analyze pixel brightness transitions.

    Parameters:
        npz_file (str): Path to the .npz file.
        pixel_locations (list of tuple): List of pixel locations to interrogate.
        creation_datetime (datetime): Media creation date as a datetime object.
        frame_rate (float): Frame rate of the video (frames per second).
        global_frame_start (int): Starting frame number for the current batch.

    Returns:
        dict: Intermediate results for the pixel transitions.
    """
    data = np.load(npz_file)
    binary_array = data['arr_0']  # Extract the binary array from the .npz file

    # Determine the number of frames in the current batch
    num_frames = binary_array.shape[0]

    # Initialize a dictionary to store results for each pixel
    pixel_results = {str(pixel): [] for pixel in pixel_locations}  # Convert tuple keys to strings

    # Iterate over each pixel location
    for pixel in pixel_locations:
        x, y = pixel  # Pixel coordinates

        # Extract the binary values for the pixel across all frames in the current batch
        pixel_values = binary_array[:, x, y]

        # Track transitions (bright -> dark or dark -> bright)
        for frame_idx in range(num_frames):
            global_frame = global_frame_start + frame_idx
            relative_time = timedelta(seconds=global_frame / frame_rate)  # Time offset from creation date
            absolute_time = creation_datetime + relative_time  # Absolute timestamp

            # Bright transition (0 -> 1)
            if frame_idx > 0 and pixel_values[frame_idx] == 1 and pixel_values[frame_idx - 1] == 0:
                pixel_results[str(pixel)].append({"transition": "bright", "timestamp": absolute_time.isoformat()})

            # Dark transition (1 -> 0)
            if frame_idx > 0 and pixel_values[frame_idx] == 0 and pixel_values[frame_idx - 1] == 1:
                pixel_results[str(pixel)].append({"transition": "dark", "timestamp": absolute_time.isoformat()})

    return pixel_results


def write_intermediate_results(results, output_folder, batch_index):
    """
    Writes intermediate results to disk as a JSON file.

    Parameters:
        results (dict): Intermediate results for pixel transitions.
        output_folder (str): Path to the output folder.
        batch_index (int): Index of the current batch.

    Returns:
        None
    """
    output_file = os.path.join(output_folder, f"batch_{batch_index+1:04d}_results.json.gz")
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    # with open(output_file, 'w') as f:
    #     json.dump(results, f, indent=4)
    print(f"Intermediate results written to {output_file}")


def merge_results(output_folder, final_output_file):
    """
    Merges all intermediate results into a single JSON file.

    Parameters:
        output_folder (str): Path to the folder containing intermediate results.
        final_output_file (str): Path to the final output JSON file.

    Returns:
        None
    """
    merged_results = {}

    # Get all intermediate result files
    result_files = sorted(
        [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith("_results.json")]
    )

    # Merge results from all files
    for result_file in result_files:
        with gzip.open(result_file, 'rt', encoding='utf-8') as f:
            batch_results = json.load(f)
            for pixel, transitions in batch_results.items():
                if pixel not in merged_results:
                    merged_results[pixel] = []
                merged_results[pixel].extend(transitions)

    # Write merged results to the final output file
    output_file = os.path.join(output_folder, final_output_file)
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=4)
    print(f"Merged results written to {output_file}")


def analyze_pixel_brightness_parallel(
    npz_folder,
    pixel_locations,
    media_creation_date,
    frame_rate,
    output_folder,
    final_output_file,
    checkpoint_folder,
    checkpoint_file_name,
):
    """
    Analyzes pixel brightness transitions in parallel and writes results to disk, with checkpoint functionality.

    Parameters:
        npz_folder (str): Path to the folder containing .npz files.
        pixel_locations (list of tuple): List of pixel locations to interrogate.
        media_creation_date (str): Media creation date and time as a string (e.g., "2023-07-16T12:34:56").
        frame_rate (float): Frame rate of the video (frames per second).
        output_folder (str): Path to the folder for intermediate results.
        final_output_file (str): Path to the final output JSON file.
        checkpoint_folder (str): Path to the checkpoint file.
        checkpoint_file_name (str): Checkpoint file name.

    Returns:
        None
    """
    # Parse the media creation date into a datetime object
    creation_datetime = datetime.strptime(media_creation_date, "%Y:%m:%d %H:%M:%S")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all .npz files in the folder, sorted by batch order
    npz_files = sorted([os.path.join(npz_folder, f) for f in os.listdir(npz_folder) if f.endswith(".npz")])

    if not npz_files:
        raise ValueError("No .npz files found in the specified folder.")

    # Load checkpoint if it exists
    checkpoint_data = load_checkpoint(checkpoint_folder, checkpoint_file_name)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": []}

    # Track the global frame number across batches
    global_frame_start = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for batch_index, npz_file in enumerate(npz_files):
            if batch_index in checkpoint_data["processed_batches"]:
                print(f"Skipping already processed batch {batch_index+1}")
                continue

            futures.append(
                executor.submit(
                    process_npz_file, npz_file, pixel_locations, creation_datetime, frame_rate, global_frame_start
                )
            )
            global_frame_start += np.load(npz_file)['arr_0'].shape[0]  # Update global frame counter

        # Use tqdm to track progress of futures
        for batch_index, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Processing Batches")
        ):
            results = future.result()
            write_intermediate_results(results, output_folder, batch_index)

            # Update checkpoint
            checkpoint_data["processed_batches"].append(batch_index)
            save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data)

    # Merge intermediate results into a final output file
    merge_results(output_folder, final_output_file)


def analyze_pixel_brightness(
    npz_folder,
    pixel_locations,
    media_creation_date,
    frame_rate,
    output_folder,
    final_output_file,
    checkpoint_folder,
    checkpoint_file_name,
):
    """
    Analyzes pixel brightness transitions in serial and writes results to disk, with checkpoint functionality.

    Parameters:
        npz_folder (str): Path to the folder containing .npz files.
        pixel_locations (list of tuple): List of pixel locations to interrogate.
        media_creation_date (str): Media creation date and time as a string (e.g., "2023-07-16T12:34:56").
        frame_rate (float): Frame rate of the video (frames per second).
        output_folder (str): Path to the folder for intermediate results.
        final_output_file (str): Path to the final output JSON file.
        checkpoint_folder (str): Path to the checkpoint file.
        checkpoint_file_name (str): Checkpoint file name.

    Returns:
        None
    """
    # Parse the media creation date into a datetime object
    creation_datetime = datetime.strptime(media_creation_date, "%Y:%m:%d %H:%M:%S")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all .npz files in the folder, sorted by batch order
    npz_files = sorted([os.path.join(npz_folder, f) for f in os.listdir(npz_folder) if f.endswith(".npz")])

    if not npz_files:
        raise ValueError("No .npz files found in the specified folder.")

    # Load checkpoint if it exists
    checkpoint_data = load_checkpoint(checkpoint_folder, checkpoint_file_name)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": []}

    # Track the global frame number across batches
    global_frame_start = 0

    # Process files serially
    for batch_index, npz_file in enumerate(tqdm(npz_files, desc="Processing Batches")):
        if batch_index in checkpoint_data["processed_batches"]:
            print(f"Skipping already processed batch {batch_index}")
            continue

        # Process the current .npz file
        results = process_npz_file(npz_file, pixel_locations, creation_datetime, frame_rate, global_frame_start)

        # Write intermediate results to disk
        write_intermediate_results(results, output_folder, batch_index)

        # Update checkpoint
        checkpoint_data["processed_batches"].append(batch_index)
        save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data)

        # Update the global frame counter
        global_frame_start += np.load(npz_file)['arr_0'].shape[0]  # Update global frame counter

    # Merge intermediate results into a final output file
    merge_results(output_folder, final_output_file)


'''


def process_image(image_path, fraction):
    """
    Processes a single image to create a binary array based on a threshold.

    Parameters:
        image_path (str): Path to the image file.
        fraction (float): Percentage (0-1) of the maximum pixel value to use as the threshold.

    Returns:
        np.ndarray: Binary array for the image.
    """
    # Load the image using PIL and convert it to grayscale
    image = Image.open(image_path).convert('L')  # 'L' mode converts to grayscale

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Calculate the threshold as a percentage of the maximum pixel value
    max_pixel_value = image_array.max()
    threshold = (fraction) * max_pixel_value

    # Apply the threshold to create a binary array
    binary_array = (image_array > threshold).astype(np.uint8)  # 1 for pixels > threshold, 0 otherwise

    return binary_array


def create_binary_pixel_array_parallel(folder_path, percentages, output_folder, batch_size=1000):
    """
    Reads all images from a folder, processes them in parallel in batches, calculates a threshold based on a percentage
    of the maximum pixel value for each image, and saves the results to disk in batches.

    Parameters:
        folder_path (str): Path to the folder containing the images.
        percentages (list of float): List of percentages (0-1) of the maximum pixel value to use as thresholds.
        output_folder (str): Folder to save the structured binary arrays to disk.
        batch_size (int): Number of images to process in each batch.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image file paths from the folder
    image_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    if not image_paths:
        raise ValueError("No valid image files found in the specified folder.")

    for percentage in percentages:
        os.makedirs(os.path.join(output_folder, str(int(percentage * 100))), exist_ok=True)
        print(f"Processing images for threshold percentage: {percentage * 100}%")

        # Process images in batches
        for batch_start in range(0, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            batch_paths = image_paths[batch_start:batch_end]

            binary_images = []
            with ProcessPoolExecutor() as executor:
                # Use tqdm to wrap the executor.map for progress tracking
                for binary_array in tqdm(
                    executor.map(process_image, batch_paths, [percentage] * len(batch_paths)),
                    total=len(batch_paths),
                    desc=f"Processing Batch {batch_start // batch_size + 1} (Threshold {percentage * 100}%)",
                ):
                    binary_images.append(binary_array)

            # Stack the batch of binary arrays along a new axis
            batch_array = np.stack(binary_images, axis=0)

            # Save the batch to disk with a unique filename
            batch_output_path = os.path.join(
                output_folder,
                str(int(percentage * 100)),
                f"threshold_{int(percentage * 100):03d}_batch_{int(batch_start // batch_size + 1):04d}.npz",
            )
            np.savez_compressed(batch_output_path, batch_array)
            print(f"Batch saved to {batch_output_path}")

        print(f"Finished processing for threshold {percentage * 100}%.")


if __name__ == "__main__":
    # Example usage:
    # Assuming you have a folder containing images, a percentage value, and an output file path
    folder_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/1_video_frames"
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
    ]  # Example: threshold is 70% of the maximum pixel value
    output_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/6_time_history_output"  # File will be saved as a NumPy binary file

    create_binary_pixel_array_parallel(folder_path, percentage, output_folder)
    # percentage = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
