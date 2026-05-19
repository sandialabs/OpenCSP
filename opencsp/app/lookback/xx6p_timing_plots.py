import re
import json
import gzip
import os
import matplotlib.pyplot as plt


def save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data):
    """
    Saves the checkpoint data to a JSON file.

    Parameters:
        checkpoint_file_name (str): Path to the checkpoint file.
        checkpoint_data (dict): Dictionary containing checkpoint information.

    Returns:
        None
    """
    with open(os.path.join(checkpoint_folder, checkpoint_file_name), 'w') as f:
        json.dump(checkpoint_data, f, indent=4)
    print(f"Checkpoint saved to {os.path.join(checkpoint_folder, checkpoint_file_name)}")


def load_checkpoint(checkpoint_folder, checkpoint_file_name):
    """
    Loads the checkpoint data from a JSON file.

    Parameters:
        checkpoint_file_name (str): Path to the checkpoint file.

    Returns:
        dict: Dictionary containing checkpoint information.
    """
    if os.path.exists(os.path.join(checkpoint_folder, checkpoint_file_name)):
        with open(os.path.join(checkpoint_folder, checkpoint_file_name), 'r') as f:
            checkpoint_data = json.load(f)
        print(f"Checkpoint loaded from {os.path.join(checkpoint_folder, checkpoint_file_name)}")
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


def write_compressed_json(data, file_path):
    """
    Writes data to a compressed JSON file (.json.gz).

    Parameters:
        data (dict or list): Data to write to the file.
        file_path (str): Path to the .json.gz file.

    Returns:
        None
    """
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"Data written to {file_path}")


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

    data = read_compressed_json(compiled_json)

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
        print(f"Saved timing plot for pixel {pixel} to {plot_file}")


# Example usage:
if __name__ == "__main__":
    # 2832
    image_folder_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/3_specific_cropped_frames"  # Video Path Used to Generate Frames
    json_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/7_pixel_timing_interrogation"
    json_file = "time_history_transition_parallel_facet.json.gz"
    output_data_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/7_pixel_timing_interrogation"  # Plots Output Here
    checkpoint_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2832/0_checkpoints"

    create_timing_plots(
        os.path.join(json_folder, json_file), os.path.join(json_folder, "pixel_timing_plots"), image_folder_path
    )
