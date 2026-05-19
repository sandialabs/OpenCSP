import os
import subprocess
import gzip
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def extract_video_metadata_exiftool(video_path):
    """
    Extracts the media creation date, frame rate, duration, and image size (width and height)
    from a video file's metadata using exiftool.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        dict: A dictionary containing:
            - 'creation_date': Media creation date and time (if available).
            - 'frame_rate': Frame rate of the video (frames per second, if available).
            - 'duration': Duration of the video (seconds, if available).
            - 'frame_width': Width of the video frame (pixels, if available).
            - 'frame_height': Height of the video frame (pixels, if available).
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
                "-ImageWidth",
                "-ImageHeight",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Initialize metadata dictionary
        metadata = {
            'creation_date': None,
            'frame_rate': None,
            'duration': None,
            'frame_width': None,
            'frame_height': None,
        }

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
            elif "Image Width" in line:
                metadata['frame_width'] = int(line.split(": ", 1)[1].strip())
            elif "Image Height" in line:
                metadata['frame_height'] = int(line.split(": ", 1)[1].strip())

        return metadata
    except Exception as e:
        print(f"Error extracting metadata with exiftool: {e}")
        return None


def load_compressed_results_batch(result_file):
    """
    Loads a single compressed JSON file.

    Parameters:
        result_file (str): Path to the compressed JSON file.

    Returns:
        dict: A dictionary containing pixel transition data for the batch.
    """
    with gzip.open(result_file, 'rt', encoding='utf-8') as f:
        batch_results = json.load(f)
        return {
            tuple(map(int, pixel_str.strip("()").split(", "))): transitions
            for pixel_str, transitions in batch_results.items()
        }


def apply_global_time_offset_batch(batch_results, time_offset_seconds):
    """
    Applies a global time offset to all timestamps in the batch results.

    Parameters:
        batch_results (dict): Dictionary containing pixel transition data for the batch.
        time_offset_seconds (float): Global time offset in seconds (positive or negative).

    Returns:
        dict: Updated batch results with adjusted timestamps.
    """
    time_offset = timedelta(seconds=time_offset_seconds)

    for pixel, transitions in batch_results.items():
        for transition in transitions:
            original_timestamp = datetime.fromisoformat(transition["timestamp"])
            adjusted_timestamp = original_timestamp + time_offset
            transition["timestamp"] = adjusted_timestamp.isoformat()

    return batch_results


def create_transition_map_batches(output_folder, frame_rate, image_shape, time_offset_seconds=0):
    """
    Creates an animated map showing pixel transitions over time, processing results in batches.

    Parameters:
        output_folder (str): Path to the folder containing compressed JSON files.
        frame_rate (float): Frame rate of the video (frames per second).
        image_shape (tuple): Shape of the image as (height, width).
        time_offset_seconds (float): Global time offset in seconds (default is 0).

    Returns:
        FuncAnimation: Animated map of pixel transitions.
    """
    # Get all compressed result files
    result_files = sorted(
        [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith("_results.json")]
    )

    # Initialize the transition map
    height, width = image_shape
    transition_map = np.zeros((height, width), dtype=np.uint8)  # 0 for dark, 1 for bright

    # Create a dictionary to store transitions by adjusted global time
    transitions_by_time = {}

    # Process each batch
    for result_file in result_files:
        print(f"Processing batch: {result_file}")
        batch_results = load_compressed_results_batch(result_file)
        batch_results = apply_global_time_offset_batch(batch_results, time_offset_seconds)

        # Organize transitions by adjusted global time
        for pixel, transitions in batch_results.items():
            for transition in transitions:
                timestamp = datetime.fromisoformat(transition["timestamp"])
                if timestamp not in transitions_by_time:
                    transitions_by_time[timestamp] = []
                transitions_by_time[timestamp].append((pixel, transition["transition"]))

    # Sort timestamps to ensure chronological order
    sorted_timestamps = sorted(transitions_by_time.keys())

    # Create the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(transition_map, cmap="gray", vmin=0, vmax=2)  # Use grayscale colormap
    time_text = ax.text(0.5, 1.05, "", transform=ax.transAxes, ha="center", fontsize=12)
    ax.set_title("Pixel Transition Map")
    ax.axis("off")

    def update(frame):
        """
        Updates the transition map for the current timestamp.

        Parameters:
            frame (int): Current frame index.

        Returns:
            list: Updated image and time text.
        """
        # Reset the transition map
        transition_map.fill(0)

        # Get the current timestamp
        current_timestamp = sorted_timestamps[frame]

        # Apply transitions for the current timestamp
        if current_timestamp in transitions_by_time:
            for pixel, transition in transitions_by_time[current_timestamp]:
                x, y = pixel
                if transition == "bright":
                    transition_map[x, y] = 2  # Green for bright transition
                elif transition == "dark":
                    transition_map[x, y] = 1  # Red for dark transition

        # Update the image and time text
        im.set_array(transition_map)
        time_text.set_text(f"Time: {current_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        return [im, time_text]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(sorted_timestamps), interval=1000 / frame_rate, blit=True)

    return ani


# Example usage:
if __name__ == "__main__":
    video_file_path = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/DSC_2844.MOV"  # Video Path Used to Generate Frames
    npz_folder = r"//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/6_time_history_output/50"  # Replace with the path to your `.npz` file folder
    output_data_path = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/7_pixel_timing_interrogation"  # Replace with the path to save the output
    output_data_name = "time_history_transition_parallel_facet.json"
    checkpoint_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06-14_NsttfHeliostatMoon/3_Post/DSC_2844/0_checkpoints"
    metadata = extract_video_metadata_exiftool(video_file_path)

    time_offset_seconds = 21060  # Example global time offset (seconds, negative for backward)

    # Create the animated map
    ani = create_transition_map_batches(
        output_data_path,
        metadata['frame_rate'],
        (metadata['frame_height'], metadata['frame_width']),
        time_offset_seconds,
    )

    # Save the animation as a video file
    ani.save(
        os.path.join(output_data_path, "pixel_transitions.mp4"),
        fps=metadata['frame_rate'],
        extra_args=["-vcodec", "libx264"],
    )
    print("Animation saved as pixel_transitions.mp4")
