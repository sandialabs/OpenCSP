import os
import re
from logging import DEBUG, ERROR
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# from opencsp.common.lib.tool.log_tools import multiprocessing_logger
import opencsp.app.lookback.lookback_tools as lbt

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_time_history_transition_mp_debug.txt",
    log_type=DEBUG,
)


def merge_results(output_folder, final_output_file):
    """
    Merges all intermediate results into a single .npz file.

    Parameters:
        output_folder (str): Path to the folder containing intermediate results.
        final_output_file (str): Path to the final output .npz file.

    Returns:
        None
    """
    merged_results = {}

    # Get all intermediate result files
    result_files = sorted(
        [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith("_results.json.gz")]
    )

    # Merge results from all files
    for result_file in result_files:
        batch_results = lbt.read_compressed_json(file_path=result_file)
        for pixel, transitions in batch_results.items():
            if pixel not in merged_results:
                merged_results[pixel] = []
            if isinstance(transitions, list):
                merged_results[pixel].extend(transitions)

    # Write merged results to the final output file
    # np.savez_compressed(os.path.join(output_folder, final_output_file), allow_pickle=True, **merged_results)
    # lbt.save_batch_to_npz(batch_data=merged_results, output_path=os.path.join(output_folder, final_output_file))
    lbt.write_compressed_json(
        data=merged_results, file_path=os.path.join(output_folder, final_output_file), print_path=True
    )


def write_intermediate_results(results, output_folder, batch_index):
    """
    Writes intermediate results to disk as a .npz file.

    Parameters:
        results (dict): Intermediate results for pixel transitions.
        output_folder (str): Path to the output folder.
        batch_index (int): Index of the current batch.

    Returns:
        None
    """
    output_file = os.path.join(output_folder, f"batch_{batch_index+1:04d}_results.json.gz")
    # np.savez_compressed(output_file, allow_pickle=True, **results)
    lbt.write_compressed_json(data=results, file_path=output_file, print_path=False)


def process_npz_file_within_batch(npz_file, json_file, pixel_locations):
    """
    Processes a single .npz file to analyze pixel brightness transitions within the batch.

    Parameters:
        npz_file (str): Path to the .npz file.
        pixel_locations (list of tuple): List of pixel locations to interrogate.

    Returns:
        dict: Results for the pixel transitions within the batch.
    """
    # Read the .npz file
    data_all = lbt.read_batch_from_npz(npz_file)
    data = data_all[0]['binary_array']

    img_names = lbt.read_json(json_file)

    # Initialize a dictionary to store results for each pixel
    pixel_results = {str(pixel): [] for pixel in pixel_locations}  # Convert tuple keys to strings

    image_names = []
    binary_arrays = []
    for frame, img_name in zip(data, img_names):
        image_names.append(img_name)
        binary_arrays.append(frame)

    # Iterate over the images in the batch
    for i in range(len(binary_arrays) - 1):  # Compare each image with the next one
        current_image_name = image_names[i]
        next_image_name = image_names[i + 1]

        current_binary_array = np.array(binary_arrays[i])
        next_binary_array = np.array(binary_arrays[i + 1])

        # Compare pixel values between the current image and the next image
        for pixel in pixel_locations:
            row, col = pixel  # Pixel coordinates

            # Current and next pixel values
            current_pixel_value = current_binary_array[row, col]
            next_pixel_value = next_binary_array[row, col]

            # Bright transition (0 -> 1)
            if current_pixel_value == 0 and next_pixel_value == 1:
                pixel_results[str(pixel)].append(
                    {"transition": "bright", "from_frame": current_image_name, "to_frame": next_image_name}
                )

            # Dark transition (1 -> 0)
            if current_pixel_value == 1 and next_pixel_value == 0:
                pixel_results[str(pixel)].append(
                    {"transition": "dark", "from_frame": current_image_name, "to_frame": next_image_name}
                )

    return pixel_results


def conditional_process(args):
    batch_index, npz_file, json_file, pixel_locations, checkpoint_data = args
    if batch_index in checkpoint_data["processed_batches"]:
        print(f"Skipping already processed batch {npz_file}")
        return None  # Return None for already processed files
    return process_npz_file_within_batch(npz_file, json_file, pixel_locations)


def analyze_pixel_brightness_parallel_npz(
    npz_folder, pixel_locations, output_folder, final_output_file, checkpoint_folder, checkpoint_file_name
):
    """
    Analyzes pixel brightness transitions in parallel and writes results to disk, with checkpoint functionality.

    Parameters:
        npz_folder (str): Path to the folder containing .npz files.
        pixel_locations (list of tuple): List of pixel locations to interrogate.
        output_folder (str): Path to the folder for intermediate results.
        final_output_file (str): Path to the final output .npz file.
        checkpoint_folder (str): Path to the checkpoint file.
        checkpoint_file_name (str): Checkpoint file name.

    Returns:
        None
    """
    # Check if the final output file already exists
    if os.path.exists(os.path.join(output_folder, final_output_file)):
        print(f"Final output file '{final_output_file}' already exists. Skipping analysis to avoid rework.")
        return
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all .npz files in the folder, sorted by batch order
    npz_files = sorted([os.path.join(npz_folder, f) for f in os.listdir(npz_folder) if f.endswith(".npz")])
    json_files = sorted([os.path.join(npz_folder, f) for f in os.listdir(npz_folder) if f.endswith(".json")])

    if not npz_files:
        raise ValueError("No .npz files found in the specified folder.")

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file_name)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": []}

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=3) as executor:

        tasks = [
            (batch_index, npz_file, json_file, pixel_locations, checkpoint_data)
            for batch_index, (npz_file, json_file) in enumerate(zip(npz_files, json_files))
        ]

        results = list(tqdm(executor.map(conditional_process, tasks), total=len(tasks), desc="Processing Batches"))

        '''
        futures = []
        for batch_index, (npz_file, json_file) in enumerate(zip(npz_files, json_files)):
            if batch_index in checkpoint_data["processed_batches"]:
                print(f"Skipping already processed batch {npz_file}")
                continue
        futures.append(executor.submit(process_npz_file_within_batch, npz_file, json_file, pixel_locations))
        # Use tqdm to track progress of futures
                for batch_comp_index, future in enumerate(
                    tqdm(as_completed(futures), total=len(futures), desc="Processing Batches")
                ):
                    results = future.result()
                    write_intermediate_results(results, output_folder, batch_index)

        '''
        for batch_comp_index, result in enumerate(results):
            if result is None:
                continue
            write_intermediate_results(result, output_folder, batch_comp_index)

            # Update checkpoint
            checkpoint_data["processed_batches"].append(batch_comp_index)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data)

    # Merge intermediate results into a final output file
    merge_results(output_folder, final_output_file)


def create_timing_plots_json(
    compiled_json, output_folder, source_image_folder, checkpoint_folder, checkpoint_file="timing_plots_checkpoint.json"
):
    """
    Create timing plots for each pixel using Pillow and update a checkpoint file.

    Parameters:
        compiled_json (str): Path to the compiled JSON file.
        output_folder (str): Folder to save the timing plots.
        source_image_folder (str): Folder containing source images.
        checkpoint_folder (str): Folder to save the checkpoint file.
        checkpoint_file (str): Name of the checkpoint file.

    Returns:
        None
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_pixels": []}

    # Get all .png files in the folder, sorted by batch order
    image_files = sorted(
        [os.path.join(source_image_folder, f) for f in os.listdir(source_image_folder) if f.lower().endswith(".png")]
    )
    if not image_files:
        raise ValueError("No source image files (.png) found in the specified folder.")

    frames = []
    for item in image_files:
        frames.append(lbt.frame_number_from_img_name(item))
    frames = sorted(frames)

    if not compiled_json:
        raise ValueError("No .json.gz files found in the specified folder.")

    data = lbt.read_compressed_json(compiled_json)
    # Create a tqdm progress bar outside the loop
    progress_bar = tqdm(total=len(data), desc="Creating Timing Plots and Writing Data")

    # Iterate over each pixel in the compiled JSON data
    for pixel, transitions in data.items():
        if pixel in checkpoint_data["processed_pixels"]:
            print(f"Skipping already processed pixel {pixel}.")
            progress_bar.update(1)  # Update the progress bar even if skipping
            continue

        # Initialize a binary array for the pixel
        # binary_state = {frame: 0 for frame in frames}  # Default to dark (0) for all frames
        binary_state = np.array([frames, np.zeros(len(frames))], dtype=np.int64).T

        transitions_arr = np.zeros((len(transitions), 2), dtype=np.int64)
        for ti, transition in enumerate(transitions):
            frame_index = lbt.frame_number_from_img_name(transition["to_frame"])  # Find the index of the frame
            transitions_arr[ti, 0] = frame_index
            if transition["transition"] == "bright":
                transitions_arr[ti, 1] = 1  # Set to bright (1)

            elif transition["transition"] == "dark":
                transitions_arr[ti, 1] = 0  # Set to dark (0)

        # Iterate through the transitions
        for i in range(len(transitions_arr)):
            frame_index = transitions_arr[i, 0]
            transition_state = transitions_arr[i, 1]

            # Find the index in the data array where the frame_index matches
            data_index = np.where(binary_state[:, 0] == frame_index)[0]

            if data_index.size > 0:
                # Fill the values in between the current transition and the next one
                if i < len(transitions_arr) - 1:
                    next_frame_index = transitions_arr[i + 1, 0]
                    # Fill the range between the current frame and the next frame
                    data_range = np.where(
                        (binary_state[:, 0] >= frame_index) & (binary_state[:, 0] < next_frame_index)
                    )[0]
                    binary_state[data_range, 1] = transition_state

        plt.figure(figsize=(6, 4))
        plt.plot(binary_state[:, 0], binary_state[:, 1])
        plt.title(f"Pixel {pixel} Timing Plot")
        plt.xlabel("Frame Number")
        plt.ylabel("Binary State")

        # Save the plot to the output folder
        height, _ = eval(pixel)
        if os.path.isdir(os.path.join(output_folder, str(height))):
            plot_file = os.path.normpath(os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot.jpg"))
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            lbt.write_compressed_json(
                binary_state, os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_data.json.gz")
            )
        else:
            os.makedirs(os.path.join(output_folder, str(height)), exist_ok=True)
            plot_file = os.path.normpath(os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot.jpg"))
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            lbt.write_compressed_json(
                binary_state, os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_data.json.gz")
            )
        # Update checkpoint
        checkpoint_data["processed_pixels"].append(pixel)
        lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        # Update the progress bar
        progress_bar.update(1)


def pixel_timing_plot(pixel, transitions, frames, output_folder, checkpoint_data):
    """
    Process a single pixel to create its timing plot and save the data.

    Parameters:
        pixel (str): The pixel identifier.
        transitions (list): The transitions for the pixel.
        frames (list): The list of frame numbers.
        output_folder (str): Folder to save the timing plots.
        checkpoint_data (dict): Checkpoint data to update.

    Returns:
        str: The pixel identifier if processed successfully, else None.
    """
    # Initialize a binary array for the pixel
    binary_state = np.array([frames, np.zeros(len(frames))], dtype=np.int64).T

    transitions_arr = np.zeros((len(transitions), 2), dtype=np.int64)
    for ti, transition in enumerate(transitions):
        frame_index = lbt.frame_number_from_img_name(transition["to_frame"])  # Find the index of the frame
        transitions_arr[ti, 0] = frame_index
        transitions_arr[ti, 1] = 1 if transition["transition"] == "bright" else 0

    # Iterate through the transitions
    for i in range(len(transitions_arr)):
        frame_index = transitions_arr[i, 0]
        transition_state = transitions_arr[i, 1]

        # Find the index in the data array where the frame_index matches
        data_index = np.where(binary_state[:, 0] == frame_index)[0]

        if data_index.size > 0:
            # Fill the values in between the current transition and the next one
            if i < len(transitions_arr) - 1:
                next_frame_index = transitions_arr[i + 1, 0]
                # Fill the range between the current frame and the next frame
                data_range = np.where((binary_state[:, 0] >= frame_index) & (binary_state[:, 0] < next_frame_index))[0]
                binary_state[data_range, 1] = transition_state

    plt.figure(figsize=(6, 4))
    plt.plot(binary_state[:, 0], binary_state[:, 1])
    plt.title(f"Pixel {pixel} Timing Plot")
    plt.xlabel("Frame Number")
    plt.ylabel("Binary State")

    height, _ = eval(pixel)
    os.makedirs(os.path.join(output_folder, str(height)), exist_ok=True)
    plot_file = os.path.normpath(os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot.jpg"))
    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
    plt.close()
    lbt.write_compressed_json(
        binary_state, os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_data.json.gz")
    )

    return pixel  # Return the processed pixel


def create_timing_plots_json_parallel(
    compiled_json, output_folder, source_image_folder, checkpoint_folder, checkpoint_file="timing_plots_checkpoint.json"
):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file)
    if checkpoint_data is None:
        checkpoint_data = {"processed_pixels": []}

    # Get all .png files in the folder, sorted by batch order
    image_files = sorted(
        [os.path.join(source_image_folder, f) for f in os.listdir(source_image_folder) if f.lower().endswith(".png")]
    )
    if not image_files:
        raise ValueError("No source image files (.png) found in the specified folder.")

    frames = sorted(lbt.frame_number_from_img_name(item) for item in image_files)

    if not compiled_json:
        raise ValueError("No .json.gz files found in the specified folder.")

    data = lbt.read_compressed_json(compiled_json)
    progress_bar = tqdm(total=len(data), desc="Creating Timing Plots and Writing Data")

    with ProcessPoolExecutor() as executor:
        futures = []
        for pixel, transitions in data.items():
            if pixel in checkpoint_data["processed_pixels"]:
                print(f"Skipping already processed pixel {pixel}.")
                progress_bar.update(1)
                continue

            futures.append(
                executor.submit(pixel_timing_plot, pixel, transitions, frames, output_folder, checkpoint_data)
            )

        for future in futures:
            pixel = future.result()
            if pixel:
                checkpoint_data["processed_pixels"].append(pixel)
                lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
                progress_bar.update(1)

    progress_bar.close()
