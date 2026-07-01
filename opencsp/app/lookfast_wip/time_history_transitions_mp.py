import os
import re
from logging import DEBUG, ERROR
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# from opencsp.common.lib.tool.log_tools import multiprocessing_logger
import opencsp.app.lookback.lookback_tools as lbt


# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_time_history_transition_mp_debug.txt",
    log_type=DEBUG,
)


def merge_hdf5_results(output_folder, final_output_file):
    """
    Merges all intermediate results into a single hdf5 file.

    Parameters:
        output_folder (str): Path to the folder containing intermediate results.
        final_output_file (str): Path to the final output hdf5 file.

    Returns:
        None
    """
    merged_results = {}

    # Get all intermediate result files
    result_files = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".hdf5")])

    # Merge results from all files
    for result_file in result_files:
        batch_results = lbt.read_hdf5_datasets(os.path.normpath(result_file))
        for pixel, transitions in batch_results.items():
            if pixel not in merged_results:
                merged_results[pixel] = []

            merged_results[pixel].append(transitions)

    # Write merged results to the final output file
    lbt.save_hdf5_datasets_compressed(
        merged_results, [], os.path.join(output_folder, final_output_file), compression="gzip", compression_level=4
    )


def process_hdf5_file_within_batch(hdf5_file, pixel_locations):
    """
    Processes a single .hdf5 file to analyze pixel brightness transitions within the batch.

    Parameters:
        hdf5_file (str): Path to the .hdf5 file.
        pixel_locations (list of tuple): List of pixel locations to interrogate.

    Returns:
        dict: Results for the pixel transitions within the batch.
    """
    head, tail = os.path.split(hdf5_file)
    base, _ = os.path.splitext(tail)
    datasets = lbt.read_json(os.path.join(head, base + ".json"))
    # Read the .hdf5 file
    data = lbt.read_hdf5_datasets(hdf5_file, datasets)

    # Initialize a dictionary to store results for each pixel
    pixel_results = {str(pixel): [] for pixel in pixel_locations}  # Convert tuple keys to strings

    image_names = []
    binary_arrays = []
    for frame, array in data.items():
        image_names.append(frame)
        binary_arrays.append(array)

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


def process_hdf5_file_within_mask_batch(hdf5_file, pixel_mask):
    """
    Processes a single .hdf5 file to analyze pixel brightness transitions within the batch.

    Parameters:
        hdf5_file (str): Path to the .hdf5 file.
        pixel_mask (np.ndarray): Binary mask indicating pixel locations to interrogate (same dimensions as image arrays).

    Returns:
        dict: Results for the pixel transitions within the batch.
    """
    head, tail = os.path.split(hdf5_file)
    base, _ = os.path.splitext(tail)
    datasets = lbt.read_json(os.path.join(head, base + ".json"))
    # Read the .hdf5 file
    data = lbt.read_hdf5_datasets(hdf5_file, datasets)

    # Initialize a dictionary to store results for each pixel
    pixel_results = {}  # Dictionary to store results for pixels with transitions

    image_names = []
    binary_arrays = []
    for frame, array in data.items():
        image_names.append(frame)
        binary_arrays.append(array)

    # Iterate over the images in the batch
    for i in range(len(binary_arrays) - 1):  # Compare each image with the next one
        current_image_name = image_names[i]
        next_image_name = image_names[i + 1]

        current_binary_array = np.array(binary_arrays[i])
        next_binary_array = np.array(binary_arrays[i + 1])

        # Compare pixel values between the current image and the next image
        # Use the pixel mask to filter relevant pixels
        for row in range(pixel_mask.shape[0]):
            for col in range(pixel_mask.shape[1]):
                if pixel_mask[row, col]:  # Only process pixels where the mask is True
                    # Current and next pixel values
                    current_pixel_value = current_binary_array[row, col]
                    next_pixel_value = next_binary_array[row, col]

                    # Bright transition (0 -> 1)
                    if current_pixel_value == 0 and next_pixel_value == 1:
                        pixel_key = f"{row},{col}"
                        if pixel_key not in pixel_results:
                            pixel_results[pixel_key] = []
                        pixel_results[pixel_key].append(
                            {"transition": "bright", "from_frame": current_image_name, "to_frame": next_image_name}
                        )

                    # Dark transition (1 -> 0)
                    if current_pixel_value == 1 and next_pixel_value == 0:
                        pixel_key = f"{row},{col}"
                        if pixel_key not in pixel_results:
                            pixel_results[pixel_key] = []
                        pixel_results[pixel_key].append(
                            {"transition": "dark", "from_frame": current_image_name, "to_frame": next_image_name}
                        )

    return pixel_results


def analyze_pixel_brightness_parallel_hdf5(
    hdf5_folder, pixel_locations, output_folder, final_output_file, checkpoint_folder, checkpoint_file_name
):
    """
    Analyzes pixel brightness transitions in parallel and writes results to disk, with checkpoint functionality.

    Parameters:
        hdf5_folder (str): Path to the folder containing .hdf5 files.
        pixel_locations (list of tuple or numpy binary array mask): List of pixel locations to interrogate.
        output_folder (str): Path to the folder for intermediate results.
        final_output_file (str): Path to the final output .hdf5 file.
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
    hdf5_files = sorted([os.path.join(hdf5_folder, f) for f in os.listdir(hdf5_folder) if f.endswith(".hdf5")])

    if not hdf5_files:
        raise ValueError("No .hdf5 files found in the specified folder.")

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file_name)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": []}

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = []
        for batch_index, hdf5_file in enumerate(hdf5_files):
            if batch_index in checkpoint_data["processed_batches"]:
                print(f"Skipping already processed batch {batch_index+1}")
                continue

            futures.append(executor.submit(process_hdf5_file_within_mask_batch, hdf5_file, pixel_locations))

        # Use tqdm to track progress of futures
        for batch_index, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Time History Transitions Batches")
        ):
            results = future.result()
            lbt.save_hdf5_datasets_compressed(
                results,
                [],
                os.path.join(output_folder, f"batch_{batch_index+1:04d}_results.hdf5"),
                compression="gzip",
                compression_level=4,
            )
            # write_intermediate_results_npz(results, output_folder, batch_index)

            # Update checkpoint
            checkpoint_data["processed_batches"].append(batch_index)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data)

    # Merge intermediate results into a final output file
    merge_hdf5_results(output_folder, final_output_file)


def create_timing_plots_with_pillow_hdf5(
    compiled_hdf5, output_folder, source_image_folder, checkpoint_folder, checkpoint_file="timing_plots_checkpoint.json"
):
    """
    Create timing plots for each pixel using Pillow and update a checkpoint file.

    Parameters:
        compiled_hdf5 (str): Path to the compiled .hdf5 file.
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

    if not compiled_hdf5:
        raise ValueError("No .hdf5 file found in the specified folder.")

    # Load the compiled .npz file
    data = lbt.read_hdf5_datasets(compiled_hdf5)
    unpacked_data = unpack_dict_data(data)

    # Create a tqdm progress bar outside the loop
    progress_bar = tqdm(total=len(unpacked_data), desc="Creating Timing Plots and Writing Data")

    # Iterate over each pixel in the compiled .npz data
    for pixel, transitions in unpacked_data.items():
        if pixel in checkpoint_data["processed_pixels"]:
            print(f"Skipping already processed pixel {pixel}.")
            progress_bar.update(1)  # Update the progress bar even if skipping
            continue

        # Initialize a binary array for the pixel
        binary_state = {frame: 0 for frame in frames}  # Default to dark (0) for all frames

        # Apply transitions to the binary state array
        current_state = 0  # Start with dark (0)
        for frame in frames:
            # Check if there is a transition for the current frame
            for start_key, transition in transitions.items():
                frame_index = lbt.frame_number_from_img_name(transition["to_frame"])  # Find the index of the frame
                if frame_index == frame:
                    if transition["transition"] == "bright":
                        current_state = 1  # Set to bright (1)
                    elif transition["transition"] == "dark":
                        current_state = 0  # Set to dark (0)

            # Propagate the current state to the binary_state dictionary
            binary_state[frame] = current_state

        # Create a plot for the pixel using Pillow
        width, height = 1000, 400
        margin = 50
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)

        # Draw axes
        draw.line([(margin, height - margin), (width - margin, height - margin)], fill="black", width=2)  # X-axis
        draw.line([(margin, margin), (margin, height - margin)], fill="black", width=2)  # Y-axis

        # Draw grid lines
        num_grid_lines = 10
        x_grid_spacing = (width - 2 * margin) / num_grid_lines
        y_grid_spacing = (height - 2 * margin) / 2  # Binary states are 0 or 1
        for i in range(num_grid_lines + 1):
            x = margin + i * x_grid_spacing
            draw.line([(x, margin), (x, height - margin)], fill="lightgray", width=1)  # Vertical grid lines
        for i in range(3):  # Binary states are 0, 1 (and optionally 2 for future use)
            y = height - margin - i * y_grid_spacing
            draw.line([(margin, y), (width - margin, y)], fill="lightgray", width=1)  # Horizontal grid lines

        # Draw binary state plot
        x_scale = (width - 2 * margin) / len(frames)
        y_scale = (height - 2 * margin) / 2  # Binary states are 0 or 1
        prev_x, prev_y = margin, height - margin - binary_state[frames[0]] * y_scale
        for i, frame in enumerate(frames):
            x = margin + i * x_scale
            y = height - margin - binary_state[frame] * y_scale
            draw.line([(prev_x, prev_y), (x, prev_y)], fill="blue", width=2)  # Horizontal line
            draw.line([(x, prev_y), (x, y)], fill="blue", width=2)  # Vertical line
            prev_x, prev_y = x, y

        # Add periodic frame number labels on the x-axis
        font = ImageFont.load_default()
        label_interval = max(1, len(frames) // num_grid_lines)  # Determine label interval
        for i, frame in enumerate(frames):
            if i % label_interval == 0:  # Add label at regular intervals
                x = margin + i * x_scale
                draw.text((x - 10, height - margin + 5), str(frame), fill="black", font=font)

        # Add labels and title
        draw.text((width // 2 - margin, margin // 2), f"Timing Plot for Pixel {pixel}", fill="black", font=font)
        draw.text((width // 2 - margin, height - margin + 20), "Frame Number", fill="black", font=font)

        # Save the plot to the output folder
        row, col = re.split(r"_", pixel, maxsplit=1)
        # height, _ = eval(pixel)
        if os.path.isdir(os.path.join(output_folder, str(row))):
            plot_file = os.path.normpath(os.path.join(output_folder, str(row), f"pixel_{pixel}_timing_plot_PIL.jpg"))
            img.save(plot_file)
            np.savez_compressed(
                os.path.normpath(os.path.join(output_folder, str(row), f"pixel_{pixel}_timing_plot_data.npz")),
                binary_state,
            )
        else:
            os.makedirs(os.path.join(output_folder, str(row)), exist_ok=True)
            plot_file = os.path.normpath(os.path.join(output_folder, str(row), f"pixel_{pixel}_timing_plot_PIL.jpg"))
            img.save(plot_file)
            np.savez_compressed(
                os.path.normpath(os.path.join(output_folder, str(row), f"pixel_{pixel}_timing_plot_data.npz")),
                binary_state,
            )
        # Update checkpoint
        checkpoint_data["processed_pixels"].append(pixel)
        lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        # Update the progress bar
        progress_bar.update(1)
    progress_bar.close()


def unpack_dict_data(data_packet):
    output_dict = {}
    pixels = data_packet.keys()
    try:
        for pixel in pixels:
            foo = data_packet[pixel]
            temp_dict = {}
            for _, batch_transitions in foo.items():
                if isinstance(batch_transitions, dict):
                    for _, transition in batch_transitions.items():
                        frame_begin = lbt.frame_number_from_img_name(transition['from_frame'])
                        temp_dict[frame_begin] = transition
                else:
                    return None
                output_dict[pixel] = temp_dict

        return output_dict
    except Exception:
        logger.error("Cannot Unpack data in this format", exc_info=True)
