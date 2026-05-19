import json
import time
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import ast  # For safe parsing of string tuples
import gzip
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import subprocess
import opencsp.app.lookback.lookback_tools as lbt
from logging import DEBUG, ERROR

logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd, "error_logs"), log_file_name="error_log_slope_data_types.txt", log_type=ERROR
)


# Extract slope data and pixel coordinates
def extract_data(data):
    slope_1 = []
    slope_2 = []
    pixel_coords = []
    pixel_no_slope = []
    pixel_no_overlap = []
    slope_no_overlap = []
    for pixel, details in data.items():
        if isinstance(details, dict):
            if "slope_1" in details or "slope_2" in details:
                if details["slope_1"].size < 3 or details["slope_2"].size < 3:
                    pixel_no_slope.append(ast.literal_eval(pixel))
                    continue
                elif np.array_equal(details["slope_1"], details["slope_2"]):
                    pixel_no_overlap.append(ast.literal_eval(pixel))
                    slope_no_overlap.append(details["slope_1"])
                else:
                    slope_1.append(details["slope_1"])
                    slope_2.append(details["slope_2"])
                    pixel_coords.append(ast.literal_eval(pixel))  # Convert string "(x, y)" to tuple
            else:
                continue
    return (
        np.array(pixel_coords),
        np.array(pixel_no_slope),
        np.array(pixel_no_overlap),
        np.array(slope_1),
        np.array(slope_2),
        np.array(slope_no_overlap),
    )


# Compute slope magnitudes
def compute_magnitude(slope_data):
    return np.linalg.norm(slope_data, axis=1)


# Generate contour plots
def plot_contours(pixel_coords, values, title, cmap="viridis"):
    x = pixel_coords[:, 0]
    y = pixel_coords[:, 1]
    z = values

    # Create grid for contour plot
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = plt.tricontourf(x, y, z, levels=100, cmap=cmap).get_array()

    plt.figure(figsize=(8, 6))
    plt.tricontourf(x, y, z, levels=100, cmap=cmap)
    plt.colorbar(label="Value")
    plt.title(title)
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    # plt.show()


# Overlay slope magnitude on source image
def overlay_on_image(pixel_coords, values, image_path, cmap="viridis"):
    # Load the source image
    img = plt.imread(image_path)

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap="gray")
    plt.scatter(pixel_coords[:, 1], pixel_coords[:, 0], c=values, cmap=cmap, s=10)
    plt.colorbar(label="Slope Magnitude")
    plt.title("Slope Magnitude Overlay")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    # plt.show()


def disp_pixels_with_data(data_path, chkpt_data_path, image_path):
    data = lbt.read_compressed_json(data_path)
    chkpt_data = lbt.read_json(chkpt_data_path)
    # chkpt_plot_data = lbt.read_json(chkpt_plot_path)

    pixels, pixel_no_slope, pixel_no_overlap, _, _, _ = extract_data(data)
    all_pixels = []
    for pix in chkpt_data["processed_pixels"]:
        x, y = ast.literal_eval(pix)
        all_pixels.append([x, y])

    all_pixels = np.array(all_pixels)
    # Load the source image
    img = plt.imread(image_path)

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap="gray")
    plt.scatter(all_pixels[:, 1], all_pixels[:, 0], c="r", s=1, marker="s", alpha=0.1, label="All Processed Pixels")
    plt.scatter(pixels[:, 1], pixels[:, 0], c="b", s=1, marker="s", alpha=0.5, label="Pixels with Calculated Slopes")
    plt.scatter(
        pixel_no_slope[:, 1], pixel_no_slope[:, 0], c="m", s=1, marker="s", alpha=0.5, label="Pixels without Slopes"
    )
    plt.scatter(
        pixel_no_overlap[:, 1],
        pixel_no_overlap[:, 0],
        c="g",
        s=1,
        marker="s",
        alpha=0.5,
        label="Pixels without Overlap",
    )

    plt.title("Data Coverage Map from Full Sampeled Set")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.legend()
    plt.show()
    print("this line")


# Main execution
if __name__ == "__main__":
    # File paths
    compressed_json_filepath = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/8_pixel_vector_information/debug/pixel_vector_information_wslope_debug.json.gz"
    image_filepath = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/4_coverage_map/traditional_compiled_binary_map_50.png"  # Replace with your source image path

    checkpoint_dir = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints"
    checkpoint_file_data = "pixel_vector_checkpoint_info_parallel_debug.json"
    # checkpoint_file_3d_plots = "pixel_vector_3D plotting_checkpoint.json"

    disp_pixels_with_data(
        data_path=compressed_json_filepath,
        chkpt_data_path=os.path.join(checkpoint_dir, checkpoint_file_data),
        image_path=image_filepath,
    )

    '''
    # Load and process data
    data = lbt.read_json(json_filepath)
    pixel_coords, slope_1, slope_2 = extract_data(data)
    slope_magnitude = compute_magnitude(slope_1)

    # Contour plots
    plot_contours(pixel_coords, slope_magnitude, "Surface Normal Magnitude Global")
    plot_contours(pixel_coords, slope_1[:, 0], "Surface Normal X Component")
    plot_contours(pixel_coords, slope_1[:, 1], "Surface Normal Y Component")
    plot_contours(pixel_coords, slope_1[:, 2], "Surface Normal Z Component")

    # Overlay slope magnitude on source image
    overlay_on_image(pixel_coords, slope_magnitude, image_filepath)
    plt.show()
    print("done")
    # TODO I need to think of a way to visualize the slope map generated,
    # because all of the slopes are normalized...
    '''
