from datetime import datetime, timedelta
import subprocess
import re
import json
import gzip
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# from opencsp.common.lib.tool.log_tools import multiprocessing_logger
import opencsp.app.lookback.lookback_tools as lbt
from logging import DEBUG, ERROR

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd, "error_logs"),
    log_file_name="error_log_time_history_transition_debug.txt",
    log_type=ERROR,
)


def process_json_file_within_batch(json_file, pixel_locations):
    """
    Processes a single .json.gz file to analyze pixel brightness transitions within the batch.

    Parameters:
        json_file (str): Path to the .json.gz file.
        pixel_locations (list of tuple): List of pixel locations to interrogate.

    Returns:
        dict: Results for the pixel transitions within the batch.
    """
    # Read the compressed JSON file
    data = lbt.read_compressed_json(json_file)

    # Initialize a dictionary to store results for each pixel
    pixel_results = {str(pixel): [] for pixel in pixel_locations}  # Convert tuple keys to strings

    # Iterate over the images in the batch
    for i in range(len(data) - 1):  # Compare each image with the next one
        current_entry = data[i]
        next_entry = data[i + 1]

        current_image_name = current_entry["image_name"]
        next_image_name = next_entry["image_name"]

        current_binary_array = np.array(current_entry["binary_array"])  # Convert binary array to NumPy array
        next_binary_array = np.array(next_entry["binary_array"])  # Convert binary array to NumPy array

        # Compare pixel values between the current image and the next image
        for pixel in pixel_locations:
            x, y = pixel  # Pixel coordinates

            # Current and next pixel values
            current_pixel_value = current_binary_array[x, y]
            next_pixel_value = next_binary_array[x, y]

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


def write_intermediate_results(results, output_folder, batch_index):
    """
    Writes intermediate results to disk as a compressed JSON file.

    Parameters:
        results (dict): Intermediate results for pixel transitions.
        output_folder (str): Path to the output folder.
        batch_index (int): Index of the current batch.

    Returns:
        None
    """
    output_file = os.path.join(output_folder, f"batch_{batch_index+1:04d}_results.json.gz")
    lbt.write_compressed_json(results, output_file)


def merge_results(output_folder, final_output_file):
    """
    Merges all intermediate results into a single compressed JSON file.

    Parameters:
        output_folder (str): Path to the folder containing intermediate results.
        final_output_file (str): Path to the final output JSON file.

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
        batch_results = lbt.read_compressed_json(result_file)
        for pixel, transitions in batch_results.items():
            if pixel not in merged_results:
                merged_results[pixel] = []
            merged_results[pixel].extend(transitions)

    # Write merged results to the final output file
    lbt.write_compressed_json(merged_results, os.path.join(output_folder, final_output_file))


def analyze_pixel_brightness_parallel(
    json_folder, pixel_locations, output_folder, final_output_file, checkpoint_folder, checkpoint_file_name
):
    """
    Analyzes pixel brightness transitions in parallel and writes results to disk, with checkpoint functionality.

    Parameters:
        json_folder (str): Path to the folder containing .json.gz files.
        pixel_locations (list of tuple): List of pixel locations to interrogate.
        output_folder (str): Path to the folder for intermediate results.
        final_output_file (str): Path to the final output JSON file.
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

    # Get all .json.gz files in the folder, sorted by batch order
    json_files = sorted([os.path.join(json_folder, f) for f in os.listdir(json_folder) if f.endswith(".json.gz")])

    if not json_files:
        raise ValueError("No .json.gz files found in the specified folder.")

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file_name)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": []}

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for batch_index, json_file in enumerate(json_files):
            if batch_index in checkpoint_data["processed_batches"]:
                print(f"Skipping already processed batch {batch_index+1}")
                continue

            futures.append(executor.submit(process_json_file_within_batch, json_file, pixel_locations))

        # Use tqdm to track progress of futures
        for batch_index, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Processing Batches")
        ):
            results = future.result()
            write_intermediate_results(results, output_folder, batch_index)

            # Update checkpoint
            checkpoint_data["processed_batches"].append(batch_index)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data)

    # Merge intermediate results into a final output file
    merge_results(output_folder, final_output_file)


def process_npz_file_within_batch(npz_file, pixel_locations):
    """
    Processes a single .npz file to analyze pixel brightness transitions within the batch.

    Parameters:
        npz_file (str): Path to the .npz file.
        pixel_locations (list of tuple): List of pixel locations to interrogate.

    Returns:
        dict: Results for the pixel transitions within the batch.
    """
    # Read the .npz file
    data = lbt.read_batch_from_npz(npz_file)

    # Initialize a dictionary to store results for each pixel
    pixel_results = {str(pixel): [] for pixel in pixel_locations}  # Convert tuple keys to strings

    image_names = []
    binary_arrays = []
    for frame in data:
        image_names.append(frame['image_name'])
        binary_arrays.append(frame['binary_array'])

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


def write_intermediate_results_npz(results, output_folder, batch_index):
    """
    Writes intermediate results to disk as a .npz file.

    Parameters:
        results (dict): Intermediate results for pixel transitions.
        output_folder (str): Path to the output folder.
        batch_index (int): Index of the current batch.

    Returns:
        None
    """
    output_file = os.path.join(output_folder, f"batch_{batch_index+1:04d}_results.npz")
    np.savez_compressed(output_file, **results)


def merge_results_npz(output_folder, final_output_file):
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
        [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith("_results.npz")]
    )

    # Merge results from all files
    for result_file in result_files:
        batch_results = np.load(result_file, allow_pickle=True)
        for pixel, transitions in batch_results.items():
            if pixel not in merged_results:
                merged_results[pixel] = []
            merged_results[pixel].extend(transitions.tolist())

    # Write merged results to the final output file
    np.savez_compressed(os.path.join(output_folder, final_output_file), **merged_results)


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

    if not npz_files:
        raise ValueError("No .npz files found in the specified folder.")

    # Load checkpoint if it exists
    checkpoint_data = lbt.load_checkpoint(checkpoint_folder, checkpoint_file_name)
    if checkpoint_data is None:
        checkpoint_data = {"processed_batches": []}

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = []
        for batch_index, npz_file in enumerate(npz_files):
            if batch_index in checkpoint_data["processed_batches"]:
                print(f"Skipping already processed batch {batch_index+1}")
                continue

            futures.append(executor.submit(process_npz_file_within_batch, npz_file, pixel_locations))

        # Use tqdm to track progress of futures
        for batch_index, future in enumerate(
            tqdm(as_completed(futures), total=len(futures), desc="Processing Batches")
        ):
            results = future.result()
            write_intermediate_results_npz(results, output_folder, batch_index)

            # Update checkpoint
            checkpoint_data["processed_batches"].append(batch_index)
            lbt.save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data)

    # Merge intermediate results into a final output file
    merge_results_npz(output_folder, final_output_file)


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
    except Exception as e:
        print(f"Error extracting metadata with exiftool: {e}")
        return None


def create_vertical_slice(image_shape, column_index, row_range=False):
    """
    Creates a list of tuples representing a vertical slice of an image.

    Parameters:
        image_shape (tuple): Shape of the image as (height, width).
        column_index (int): Horizontal pixel location (column index) for the vertical slice.
        row_range (bool or tuple): Defaults to False, but if sent as a tuple the list of pixels
            will be between the two values assuming that the image location starts in the top left.
            (top, bottom)

    Returns:
        list of tuple: List of pixel locations (row, column) for the vertical slice.
    """
    height, width = image_shape

    # Ensure the column index is within bounds
    if column_index < 0 or column_index >= width:
        raise ValueError(f"Column index {column_index} is out of bounds for image width {width}.")

    # Generate the list of tuples for the vertical slice
    if row_range:
        vertical_slice = [(row, column_index) for row in range(*row_range)]
    else:
        vertical_slice = [(row, column_index) for row in range(height)]

    return vertical_slice


def load_image(image_path):
    """
    Loads an image and converts it to grayscale.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Grayscale image as a NumPy array.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)


def on_select(eclick, erelease):
    """
    Callback function for rectangle selection.

    Parameters:
        eclick: Mouse click event (start of rectangle).
        erelease: Mouse release event (end of rectangle).
    """
    global selected_pixels
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure coordinates are in proper order (top-left to bottom-right)
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    # Generate list of pixel locations within the selected region
    selected_pixels = [(y, x) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]
    # print(f"Selected pixels: {selected_pixels}")


def interactive_image_plot(predefined_pixels=None):
    """
    Displays an interactive image plot where users can either load predefined pixel locations
    or select a region interactively by drawing a rectangle. Prompts for image selection via file dialog.

    Parameters:
        predefined_pixels (list of tuple, optional): Predefined list of pixel locations (row, column).

    Returns:
        list of tuple: Selected pixel locations (row, column).
    """
    global selected_pixels
    selected_pixels = []  # Initialize global variable to store selected pixels

    # Prompt user to select an image file using a file dialog
    Tk().withdraw()  # Hide the root Tkinter window
    image_path = askopenfilename(
        title="Select an Image File", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )

    if not image_path:
        print("No file selected. Exiting.")
        return []

    print(f"Selected image: {image_path}")

    # Load the image
    image = load_image(image_path)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray', interpolation='nearest')
    ax.set_title("Interactive Image Plot: Draw a rectangle or load predefined pixels")

    # Display predefined pixels if provided
    if predefined_pixels:
        selected_pixels = predefined_pixels
        for row, col in predefined_pixels:
            ax.plot(col, row, 'ro', markersize=1, alpha=0.2)  # Mark predefined pixels in red

    # Add rectangle selector for interactive region selection
    rectangle_selector = RectangleSelector(
        ax, on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True
    )

    plt.show()

    return selected_pixels


def frame_number_from_img_name(image_name_str):
    # returns the integer number of a frame given the following format
    # "DSC_2832-09170.png" where DSC_2832 is the video source and "09170"
    # is the frame number
    _, tail = os.path.split(image_name_str)
    _, frame = re.findall(r'\d+', tail)
    return int(frame)


def create_timing_plots(compiled_json, output_folder, source_image_folder):

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all .json.gz files in the folder, sorted by batch order
    image_files = sorted(
        [os.path.join(source_image_folder, f) for f in os.listdir(source_image_folder) if f.lower().endswith(".png")]
    )
    if not image_files:
        raise ValueError("No source image files (.png) found in the specified folder.")

    frames = []
    for item in image_files:
        frames.append(frame_number_from_img_name(item))
    frames = sorted(frames)

    if not compiled_json:
        raise ValueError("No .json.gz files found in the specified folder.")

    data = lbt.read_compressed_json(compiled_json)

    # Iterate over each pixel in the compiled JSON data
    for pixel, transitions in data.items():
        # Initialize a binary array for the pixel
        binary_state = {frame: 0 for frame in frames}  # Default to dark (0) for all frames

        # Apply transitions to the binary state array
        current_state = 0  # Start with dark (0)
        for frame in frames:
            # Check if there is a transition for the current frame
            for transition in transitions:
                frame_index = frame_number_from_img_name(transition["to_frame"])  # Find the index of the frame
                if frame_index == frame:
                    if transition["transition"] == "bright":
                        current_state = 1  # Set to bright (1)
                    elif transition["transition"] == "dark":
                        current_state = 0  # Set to dark (0)

            # Propagate the current state to the binary_state dictionary
            binary_state[frame] = current_state

        # Create a plot for the pixel
        plt.figure(figsize=(10, 4))
        plt.plot(
            list(binary_state.keys()),  # X-axis: Frame numbers
            list(binary_state.values()),  # Y-axis: Binary states
            drawstyle="steps-post",
            label=f"Pixel {pixel}",
        )
        plt.xlabel("Frame Number")
        plt.ylabel("Binary State (0=Dark, 1=Bright)")
        plt.title(f"Timing Plot for Pixel {pixel}")
        plt.grid(True)
        plt.legend()

        # Save the plot to the output folder
        plot_file = os.path.join(output_folder, f"pixel_{pixel}_timing_plot.png")
        plt.savefig(plot_file)
        plt.close()
        lbt.write_compressed_json(binary_state, os.path.join(output_folder, f"pixel_{pixel}_timing_plot_data.json.gz"))
        print(f"Saved timing plot for pixel {pixel} to {plot_file}")


def create_timing_plots_with_pillow(
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
        frames.append(frame_number_from_img_name(item))
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
        binary_state = {frame: 0 for frame in frames}  # Default to dark (0) for all frames

        # Apply transitions to the binary state array
        current_state = 0  # Start with dark (0)
        for frame in frames:
            # Check if there is a transition for the current frame
            for transition in transitions:
                frame_index = frame_number_from_img_name(transition["to_frame"])  # Find the index of the frame
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
        height, _ = eval(pixel)
        if os.path.isdir(os.path.join(output_folder, str(height))):
            plot_file = os.path.normpath(os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_PIL.jpg"))
            img.save(plot_file)
            lbt.write_compressed_json(
                binary_state, os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_data.json.gz")
            )
        else:
            os.makedirs(os.path.join(output_folder, str(height)), exist_ok=True)
            plot_file = os.path.normpath(os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_PIL.jpg"))
            img.save(plot_file)
            lbt.write_compressed_json(
                binary_state, os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_data.json.gz")
            )
        # Update checkpoint
        checkpoint_data["processed_pixels"].append(pixel)
        lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        # Update the progress bar
        progress_bar.update(1)
    progress_bar.close()


def create_timing_plots_with_pillow_npz(
    compiled_npz, output_folder, source_image_folder, checkpoint_folder, checkpoint_file="timing_plots_checkpoint.json"
):
    """
    Create timing plots for each pixel using Pillow and update a checkpoint file.

    Parameters:
        compiled_npz (str): Path to the compiled .npz file.
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
        frames.append(frame_number_from_img_name(item))
    frames = sorted(frames)

    if not compiled_npz:
        raise ValueError("No .npz file found in the specified folder.")

    # Load the compiled .npz file
    data = np.load(compiled_npz)

    # Create a tqdm progress bar outside the loop
    progress_bar = tqdm(total=len(data.files), desc="Creating Timing Plots and Writing Data")

    # Iterate over each pixel in the compiled .npz data
    for pixel in data.files:
        if pixel in checkpoint_data["processed_pixels"]:
            print(f"Skipping already processed pixel {pixel}.")
            progress_bar.update(1)  # Update the progress bar even if skipping
            continue

        # Get transitions for the current pixel
        transitions = data[pixel].tolist()

        # Initialize a binary array for the pixel
        binary_state = {frame: 0 for frame in frames}  # Default to dark (0) for all frames

        # Apply transitions to the binary state array
        current_state = 0  # Start with dark (0)
        for frame in frames:
            # Check if there is a transition for the current frame
            for transition in transitions:
                frame_index = frame_number_from_img_name(transition["to_frame"])  # Find the index of the frame
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
        height, _ = eval(pixel)
        if os.path.isdir(os.path.join(output_folder, str(height))):
            plot_file = os.path.normpath(os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_PIL.jpg"))
            img.save(plot_file)
            np.savez_compressed(
                os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_data.npz"), **binary_state
            )
        else:
            os.makedirs(os.path.join(output_folder, str(height)), exist_ok=True)
            plot_file = os.path.normpath(os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_PIL.jpg"))
            img.save(plot_file)
            np.savez_compressed(
                os.path.join(output_folder, str(height), f"pixel_{pixel}_timing_plot_data.npz"), **binary_state
            )
        # Update checkpoint
        checkpoint_data["processed_pixels"].append(pixel)
        lbt.save_checkpoint(checkpoint_folder, checkpoint_file, checkpoint_data, print_path=False)
        # Update the progress bar
        progress_bar.update(1)
    progress_bar.close()


# Example usage:
if __name__ == "__main__":
    # 0051
    # video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0051/DSC_0051.MOV"  # Video Path Used to Generate Frames
    # npz_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0051/6_time_history_output/50"  # Replace with the path to your `.npz` file folder
    # output_data_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0051/7_pixel_timing_interrogation"  # Replace with the path to save the output
    # output_data_name = "time_history_transition_parallel_facet.json"
    # checkpoint_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_0051/0_checkpoints"
    # 2844
    # video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/DSC_2844.MOV"  # Video Path Used to Generate Frames
    # npz_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/6_time_history_output/50"  # Replace with the path to your `.npz` file folder
    # output_data_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/7_pixel_timing_interrogation"  # Replace with the path to save the output
    # output_data_name = "time_history_transition_parallel_facet.json"
    # checkpoint_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/0_checkpoints"
    # 2832
    # video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/DSC_2832.MOV"  # Video Path Used to Generate Frames
    # image_folder_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/3_specific_cropped_frames"  # Video Path Used to Generate Frames
    # json_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/6_time_history_output/50"  # Replace with the path to your `.npz` file folder
    # output_data_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/7_pixel_timing_interrogation"  # Replace with the path to save the output
    # output_data_name = "time_history_transition_parallel_facet.json.gz"
    # checkpoint_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/0_checkpoints"
    # 0025 Sun Data
    video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/DSC_0025.MOV"  # Video Path Used to Generate Frames
    image_folder_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/3_specific_cropped_frames"  # Video Path Used to Generate Frames
    # json_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/6_time_history_output/50"  # Replace with the path to your `.npz` file folder
    npz_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/6_time_history_output/50"
    output_data_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/7_pixel_timing_interrogation/npz"  # Replace with the path to save the output
    # output_data_name = "time_history_transition_parallel_facet.json.gz"
    output_data_name = "time_history_transition_parallel_facet.npz"
    checkpoint_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints"

    # Assuming the original image has a shape of (1080, 1920) (height=1080, width=1920)
    # height_range = (430, 680)
    # width_range = (1130, 1360)
    # height_range = (305, 525)
    # width_range = (1040, 1270)
    height_range = (435, 670)
    width_range = (830, 1070)
    pixel_locations = [
        (height, width)
        for height in range(height_range[0], height_range[1])
        for width in range(width_range[0], width_range[1])
    ]
    # _ = interactive_image_plot(predefined_pixels=pixel_locations)
    metadata = extract_video_metadata_exiftool(video_file_path)

    checkpoint_file_name = (
        f"time_history_transition_H{height_range[0]}_{height_range[1]}_W{width_range[0]}_{width_range[1]}_npz.json"
    )

    analyze_pixel_brightness_parallel_npz(
        npz_folder=npz_folder,
        pixel_locations=pixel_locations,
        output_folder=output_data_path,
        final_output_file=output_data_name,
        checkpoint_folder=checkpoint_folder,
        checkpoint_file_name=checkpoint_file_name,
    )

    create_timing_plots_with_pillow_npz(
        compiled_npz=os.path.join(output_data_path, output_data_name),
        output_folder=os.path.join(output_data_path, "pixel_timing_plots_PIL"),
        source_image_folder=image_folder_path,
        checkpoint_folder=checkpoint_folder,
        checkpoint_file="timing_plots_checkpoint.json",
    )

    '''
    npz_folder, pixel_locations, output_folder, final_output_file, checkpoint_folder, checkpoint_file_name
    analyze_pixel_brightness_parallel(
        json_folder, pixel_locations, output_data_path, output_data_name, checkpoint_folder, checkpoint_file_name
    )
    create_timing_plots(
        os.path.join(output_data_path, output_data_name),
        os.path.join(output_data_path, "pixel_timing_plots"),
        image_folder_path,
    )
    create_timing_plots_with_pillow(
        compiled_json=os.path.join(output_data_path, output_data_name),
        output_folder=os.path.join(output_data_path, "pixel_timing_plots_PIL"),
        source_image_folder=image_folder_path,
        checkpoint_folder=checkpoint_folder,
    )

    '''
