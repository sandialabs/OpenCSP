import numpy as np
import os
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def visualize_pixel_changes(npz_file, pixel_locations, plot_range=False):
    """
    Visualizes binary changes of pixel values across frames from a structured binary array saved in an .npz file.

    Parameters:
        npz_file (str): Path to the .npz file containing the structured binary array.
        pixel_locations (list of tuple): List of pixel locations to visualize. Each location is a tuple (x, y).
        plot_range (bool): If True, plots a range of pixels as a heatmap. If False, plots individual pixel changes.

    Returns:
        None
    """
    # Load the structured binary array from the .npz file
    data = np.load(npz_file)
    structured_array = data['arr_0']  # Extract the array from the .npz file

    # Get the number of frames
    num_frames = structured_array.shape[0]

    if plot_range:
        # Plot a heatmap for a range of pixels
        x_range = [loc[0] for loc in pixel_locations]
        y_range = [loc[1] for loc in pixel_locations]

        # Extract the binary values for the specified range of pixels across all frames
        pixel_values = structured_array[:, x_range, y_range]

        plt.figure(figsize=(10, 6))
        plt.imshow(pixel_values.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Binary Value (0 or 1)')
        plt.xlabel('Frame Number')
        plt.ylabel('Pixel Index in Range')
        plt.title('Binary Changes Across Frames for Pixel Range')
        plt.show()
    else:
        # Plot individual pixel changes
        plt.figure(figsize=(10, 6))
        for i, (x, y) in enumerate(pixel_locations):
            # Extract the binary values for the specified pixel across all frames
            pixel_values = structured_array[:, x, y]

            # plt.plot(range(num_frames), pixel_values, label=f'Pixel ({x}, {y})')
            plt.plot(range(num_frames), pixel_values)

        plt.xlabel('Frame Number')
        plt.ylabel('Binary Value (0 or 1)')
        plt.title('Binary Changes Across Frames for Individual Pixels')
        plt.legend()
        plt.grid(True)
        plt.show()


def process_pixel_values(
    npz_file_folder, npz_file, pixel_locations, output_array_path, height_range, width_range, plot_range
):
    """
    Extracts binary values for specified pixel locations from a single .npz file.

    Parameters:
        npz_file (str): Path to the .npz file containing the structured binary array.
        pixel_locations (list of tuple): List of pixel locations to extract. Each location is a tuple (x, y).
        plot_range (bool): If True, extracts a range of pixels. If False, extracts individual pixel values.

    Returns:
        np.ndarray: Extracted binary values for the specified pixel locations.
    """
    # Load the structured binary array from the .npz file
    data = np.load(os.path.join(npz_file_folder, npz_file))
    structured_array = data['arr_0']  # Extract the array from the .npz file

    batch_pattern = r"batch_[0-9][0-9][0-9][0-9]"
    batch_number = re.search(batch_pattern, npz_file).group(0)

    if plot_range:
        # Extract binary values for a range of pixels
        x_range = [loc[0] for loc in pixel_locations]
        y_range = [loc[1] for loc in pixel_locations]
        pixel_values = structured_array[:, x_range, y_range]
    else:
        # Extract binary values for individual pixels
        pixel_values = np.array([structured_array[frame, height, width] for frame, height, width in pixel_locations])

    np.savez_compressed(os.path.join(output_array_path, batch_number + f"h{height_range}_w{width_range}"), pixel_values)
    return pixel_values


# Define a helper function for parallel processing
def process_file(args):
    """
    Wrapper function for parallel processing of files.

    Parameters:
        args (tuple): Tuple containing (npz_file, pixel_locations, plot_range).

    Returns:
        np.ndarray: Extracted binary values for the specified pixel locations.
    """
    npz_file_folder, npz_file, pixel_locations, output_array_path, height_range, width_range, plot_range = args
    return process_pixel_values(
        npz_file_folder, npz_file, pixel_locations, output_array_path, height_range, width_range, plot_range
    )


def pad_arrays_to_match_shape(arrays, padding_value=0):
    """
    Pads arrays along the concatenation axis to ensure they all have the same shape.

    Parameters:
        arrays (list of np.ndarray): List of arrays to pad.
        padding_value (int): Value to use for padding.

    Returns:
        list of np.ndarray: List of padded arrays.
    """
    # Find the maximum size along the concatenation axis (dimension 1)
    max_size = max(array.shape[1] for array in arrays)

    # Pad each array to match the maximum size
    padded_arrays = []
    for array in arrays:
        pad_width = max_size - array.shape[1]
        if pad_width > 0:
            # Pad the array with the specified padding value
            padding = ((0, 0), (0, pad_width))
            padded_array = np.pad(array, padding, mode='constant', constant_values=padding_value)
        else:
            padded_array = array
        padded_arrays.append(padded_array)

    return padded_arrays


def visualize_pixel_changes_batch_parallel(
    npz_file_folder, pixel_locations, output_array_path, height_range, width_range, plot_range=False
):
    """
    Visualizes binary changes of pixel values across frames from multiple .npz files in parallel.

    Parameters:
        npz_files (list of str): List of paths to .npz files containing structured binary arrays.
        pixel_locations (list of tuple): List of pixel locations to visualize. Each location is a tuple (x, y).
        plot_range (bool): If True, plots a range of pixels as a heatmap. If False, plots individual pixel changes.

    Returns:
        None
    """
    npz_files = os.listdir(npz_file_folder)

    # Prepare arguments for parallel processing
    args = [
        (npz_file_folder, npz_file, pixel_locations, output_array_path, height_range, width_range, plot_range)
        for npz_file in npz_files
    ]

    # Process files in parallel using ProcessPoolExecutor
    combined_pixel_values = None
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Use tqdm to wrap the executor.map for progress tracking
        results = list(tqdm(executor.map(process_file, args), total=len(npz_files), desc="Processing Files"))

    # Pad arrays to ensure they all have the same shape
    padded_results = pad_arrays_to_match_shape(results)

    # Combine results from all files
    combined_pixel_values = np.concatenate(padded_results, axis=0)

    # Calculate the total number of frames
    total_frames = combined_pixel_values.shape[0]

    # Visualization
    if plot_range:
        # Plot a heatmap for a range of pixels
        plt.figure(figsize=(10, 6))
        plt.imshow(combined_pixel_values.T, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Binary Value (0 or 1)')
        plt.xlabel('Frame Number')
        plt.ylabel('Pixel Index in Range')
        plt.title('Binary Changes Across Frames for Pixel Range')
        plt.show()
    else:
        # Plot individual pixel changes
        plt.figure(figsize=(10, 6))
        for i, (x, y) in enumerate(pixel_locations):
            # plt.plot(range(num_frames), combined_pixel_values[i], label=f'Pixel ({x}, {y})')
            plt.plot(range(1, total_frames + 1), combined_pixel_values[i])

        plt.xlabel('Frame Number')
        plt.ylabel('Binary Value (0 or 1)')
        plt.title('Binary Changes Across Frames for Individual Pixels')
        plt.legend()
        plt.grid(True)
        plt.show()


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
            ax.plot(col, row, 'ro', markersize=2)  # Mark predefined pixels in red

    # Add rectangle selector for interactive region selection
    rectangle_selector = RectangleSelector(
        ax, on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True
    )

    plt.show()

    return selected_pixels


# Example usage
if __name__ == "__main__":
    npz_file_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/6_time_history_output/70"  # Replace with the path to your `.npz` file
    output_data_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/7_pixel_timing_interrogation"  # Replace with the path to save the plot (optional)

    # Assuming the original image has a shape of (100, 200) (height=100, width=200)
    height_range = (400, 670)
    width_range = (815, 1090)
    pixel_locations = [(height, width) for height in range(height_range) for width in range(width_range)]
    # pixel_locations = create_vertical_slice((4000, 6000), column_index=3000, row_range=(1500, 2000))  # Replace with the pixel locations you want to plot
    # pixel_img = interactive_image_plot()
    pixel_img = interactive_image_plot(predefined_pixels=pixel_locations)
    visualize_pixel_changes_batch_parallel(
        npz_file_path, pixel_img, output_data_path, height_range, width_range, plot_range=False
    )

    '''
    # Option 1: Load predefined pixel locations 
    selected_pixels = interactive_image_plot(predefined_pixels=predefined_pixels)
    print("Final selected pixels:", selected_pixels)
    # Option 2: Interactively select a region on the image
    selected_pixels = interactive_image_plot()
    print("Final selected pixels:", selected_pixels)
    '''
