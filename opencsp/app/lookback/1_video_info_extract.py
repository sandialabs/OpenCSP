import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import math
import time
import shutil
import subprocess
import imageio
import json
import gzip
import cv2
import numpy as np
from PIL import Image, ImageTk
from zoneinfo import ZoneInfo
from datetime import datetime, timezone, timedelta

import opencsp.common.lib.render.VideoHandler as vh
import opencsp.common.lib.render_control.RenderControlVideoFrames as rcvf
from opencsp.common.lib.cv.spot_analysis.image_processor.PopulationStatisticsImageProcessor import (
    PopulationStatisticsImageProcessor,
)
from opencsp.common.lib.cv.spot_analysis.image_processor.CroppingImageProcessor import CroppingImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.EchoImageProcessor import EchoImageProcessor
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.file_tools as ft
from opencsp.common.lib.cv import SpotAnalysis as sa
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperableAttributeParser import SpotAnalysisOperableAttributeParser
import opencsp.common.lib.tool.log_tools as lt

# from contrib.common.lib.cv.spot_analysis.image_processor.StabilizationImageProcessor import StabilizationImageProcessor


def frame_number_from_img_name(image_name_str):
    # returns the integer number of a frame given the following format
    # "DSC_2832-09170.png" where DSC_2832 is the video source and "09170"
    # is the frame number
    _, tail = os.path.split(image_name_str)
    _, frame = re.findall(r'\d+', tail)
    return int(frame)


def custom_serializer(obj):
    """
    Custom serializer to handle numpy arrays and datetime objects.
    Converts unsupported types into JSON-compatible formats.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy array to list
    elif isinstance(obj, datetime):
        return obj.isoformat()  # Convert datetime to ISO 8601 string
    elif isinstance(obj, set):
        return list(obj)  # Convert set to list
    raise TypeError(f"Type {type(obj)} not serializable")


def custom_deserializer(obj):
    """
    Custom deserializer to handle numpy arrays and datetime objects.
    Converts JSON-compatible formats back into their original types.
    """
    for key, value in obj.items():
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            try:
                obj[key] = np.array(value)  # Convert list back to numpy array
            except ValueError:
                pass  # If conversion fails, leave as list
        elif isinstance(value, str):
            try:
                obj[key] = datetime.fromisoformat(value)  # Convert ISO 8601 string back to datetime
            except ValueError:
                pass  # If conversion fails, leave as string
    return obj


class LookBack_ExEx:
    def __init__(self, root):
        self.root = root
        self.root.title("LookBack Pre-Processing User Interface")

        # Initialize variables
        self.selected_files = []
        run_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        self.key = str(run_time + "_ExEx_Test")
        self.frame_control = None
        self.video_in_dirs = None
        self.video_files = None
        self.current_video_dir = None
        self.current_video_name = None
        self.output_dir_top = None
        self.output_dir_struct = [
            r'0_checkpoints',
            r'1_video_frames',
            r'2_video_frames_cropped',
            r'3_specific_cropped_frames',
            r'4_coverage_map',
            r'5_accelerated_test_video',
            r'6_time_history_output',
            r'7_pixel_timing_interrogation',
            r'8_pixel_vector_information',
            r'9_sofast_data_compare',
            r'reference_images',
        ]
        self.output_video_folders = None
        self.output_text = None
        self.start_frame = None
        self.end_frame = None
        self.start_time = None
        self.end_time = None
        self.crop_coordinates = []
        self.results = ""

        # Create the main layout
        self.create_widgets()

    def create_widgets(self):
        # Left frame for buttons and file selection
        self.left_frame = tk.Frame(self.root, padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        self.bottom_frame = tk.Frame(self.root, padx=10, pady=10)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Buttons for running specific sections of code
        self.select_file_button = tk.Button(self.left_frame, text="Select Files", command=self.select_files)
        self.select_file_button.pack(pady=5)

        self.load_checkpoint_button = tk.Button(self.left_frame, text="Load Checkpoint", command=self.load_checkpoint)
        self.load_checkpoint_button.pack(pady=5)
        self.save_checkpoint_button = tk.Button(self.left_frame, text="Save Checkpoint", command=self.save_checkpoint)
        self.save_checkpoint_button.pack(pady=5)

        self.show_values_button = tk.Button(
            self.left_frame, text="Show Current Values", command=self.show_current_values
        )
        self.show_values_button.pack(pady=5)

        self.select_output_dir_button = tk.Button(self.left_frame, text="Select Output Dir", command=self.select_dir)
        self.select_output_dir_button.pack(pady=5)

        self.output_structure_button = tk.Button(
            self.left_frame, text="Create Output Structure", command=self.create_output_dir_structure
        )
        self.output_structure_button.pack(pady=5)

        self.scrub_select_frames_button = tk.Button(
            self.left_frame, text="Open Video Scrubber", command=self.open_video_scrubber
        )
        self.scrub_select_frames_button.pack(pady=5)

        self.extract_frames_button = tk.Button(
            self.left_frame, text="Extract All Frames", command=self.extract_full_frames
        )
        self.extract_frames_button.pack(pady=5)

        self.extract_cropped_button = tk.Button(
            self.left_frame, text="Extract Cropped Frames", command=self.extract_cropped_frames
        )
        self.extract_cropped_button.pack(pady=5)

        self.extract_frames_of_interest_button = tk.Button(
            self.left_frame, text="Extract Frames of Interest", command=self.extract_frames_of_interest
        )
        self.extract_frames_of_interest_button.pack(pady=5)

        # Output box for results
        self.output_label = tk.Label(self.bottom_frame, text="Output Results:", anchor="w")
        self.output_label.pack(fill=tk.X)

        self.output_text = tk.Text(self.bottom_frame, height=20, width=40, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def save_checkpoint(self):
        """
        Saves all class attributes to a JSON file in the specified checkpoint folder,
        excluding tkinter objects and ensuring proper serialization of supported data types.

        Parameters:
            checkpoint_folder (str): Path to the folder where the checkpoint file will be saved.
            checkpoint_file_name (str): Name of the checkpoint file.

        Returns:
            None
        """

        # Collect all class attributes, excluding tkinter objects
        checkpoint_data = {}
        for key, value in self.__dict__.items():
            # Exclude tkinter objects
            if isinstance(value, (tk.Tk, tk.Widget)):
                print(f"Skipping {key}: tkinter object.")
                continue

            # Handle supported data types
            checkpoint_data[key] = value

        # Serialize the data using the custom serializer
        serialized_data = json.dumps(checkpoint_data, default=custom_serializer, indent=4)

        # Save the serialized data to the specified file
        file_path = os.path.join(
            self.output_dir_top,
            self.video_files[0][0:-4],
            self.output_dir_struct[0],
            "video_info_extract_checkpoint_data.json",
        )
        with open(file_path, 'w') as f:
            f.write(serialized_data)

        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self):
        """
        Loads the checkpoint data from a JSON file and updates matching class variables.

        Returns:
            dict: Dictionary containing checkpoint information.
        """
        file_path = filedialog.askopenfilename(
            title="Select Checkpoint Data to Load", filetypes=(("JSON Files", "*.json"), ("All files", "*.*"))
        )
        if not file_path:
            print("No file selected.")
            return None

        checkpoint_folder, checkpoint_file = os.path.split(file_path)
        full_path = os.path.join(checkpoint_folder, checkpoint_file)

        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                checkpoint_data = json.load(f)
            print(f"Checkpoint loaded from {full_path}")

            # Update matching class variables
            for key, value in checkpoint_data.items():
                if hasattr(self, key):  # Check if the class has an attribute with the same name
                    setattr(self, key, value)  # Update the attribute with the value from the JSON
                    print(f"Updated {key} to {value}")
                else:
                    print(f"Skipped {key}: No matching class variable found.")

            return checkpoint_data
        else:
            print(f"File does not exist: {full_path}")
            return None

    def show_current_values(self):
        checkpoint_data = {key: value for key, value in self.__dict__.items()}
        for key, value in checkpoint_data.items():
            self.print_to_output_only(f"Key {key}: Value {value}")
            self.print_to_output_only("________")

    def append_to_output(self, text):
        """
        Appends text or a list of text items to the output box.

        Parameters:
            text (str or list): The text or list of text items to append.
        """
        if isinstance(text, list):
            for item in text:
                self.output_text.insert(tk.END, str(item) + "\n")
        else:
            self.output_text.insert(tk.END, str(text) + "\n")
        self.output_text.see(tk.END)  # Scroll to the end

    def print_to_output_only(self, text):
        """
        Appends text or a list of text items to the output box.

        Parameters:
            text (str or list): The text or list of text items to append.
        """
        if isinstance(text, list):
            for item in text:
                self.output_text.insert(tk.END, str(item) + "\n")
        else:
            self.output_text.insert(tk.END, str(text) + "\n")
        self.output_text.see(tk.END)  # Scroll to the end

    def select_files(self, dialog="Pick File(s) for..."):
        # Open a file selection dialog and allow the user to select multiple files
        file_paths = tk.filedialog.askopenfilenames(
            title=dialog, filetypes=(("MOV files", "*.mov"), ("All files", "*.*"))
        )
        file_dirs = []
        file_names = []
        for file in enumerate(file_paths):
            tempdir, name = os.path.split(file[1])
            file_dirs.append(tempdir)
            file_names.append(name)
        if file_paths:
            print(f"Selected directory: {file_paths}")
        else:
            print("No files selected.")
        self.video_in_dirs = file_dirs
        self.video_files = file_names
        self.append_to_output(file_dirs)
        self.append_to_output(file_names)

    def select_dir(self, dialog="Pick Directory Location for Output"):
        dir_path = tk.filedialog.askdirectory(title=dialog)
        if dir_path:
            print(f"Selected directory: {dir_path}")
        else:
            print("No directory selected.")
        self.output_dir_top = dir_path
        self.append_to_output(dir_path)

    def create_output_dir_structure(self):
        output_targets = []
        for item in enumerate(self.video_files):
            output_struct = os.path.join(self.output_dir_top, item[1][0:-4])
            output_targets.append(output_struct)
            if os.path.exists(output_struct):
                continue
            else:
                os.mkdir(output_struct)
                for tempdir in enumerate(self.output_dir_struct):
                    os.mkdir(os.path.join(output_struct, tempdir[1]))
            video_copy_path = os.path.join(output_struct, item[1])
            if os.path.exists(video_copy_path):
                continue
            else:
                shutil.copyfile(os.path.join(self.video_in_dirs[item[0]], item[1]), video_copy_path)
        self.output_video_folders = output_targets

    def open_video_scrubber(self):
        # Open a new window for the VideoScrubber
        scrubber_window = tk.Toplevel(self.root)
        scrubber_app = VideoScrubber(scrubber_window, self)

    def set_frame_range(self, start_frame, end_frame, fps):
        # Set the start and end frames from the VideoScrubber
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_time = math.floor(start_frame / fps)
        self.end_time = math.ceil(end_frame / fps)
        self.append_to_output(f"Selected frame range: Start = {start_frame}, End = {end_frame}")
        self.append_to_output(f"Selected frame range: Start_Time = {self.start_time}, End = {self.end_time}")

    def set_crop_coords(self, coords_list):
        # Set the crop coordinates from the VideoScrubber
        self.crop_coordinates = coords_list
        self.append_to_output(
            f"Rectangle Coordinates:: Top-left: ({coords_list[0]}, {coords_list[1]}), Bottom-right: ({coords_list[2]}, {coords_list[3]})"
        )

    def set_current_scrub_video(self, video_path):
        tempdir, name = os.path.split(video_path)
        self.current_video_dir = tempdir
        self.current_video_name = name
        self.append_to_output(f"Current Video Dir: {self.current_video_dir}")
        self.append_to_output(f"Current Video Name: {self.current_video_name}")

    def extract_full_frames(self):
        self.frame_control = rcvf.RenderControlVideoFrames(outframe_name="-%05d", outframe_format="png")
        video_handler = vh.VideoHandler.VideoExtractor(
            os.path.join(self.current_video_dir, self.current_video_name),
            os.path.join(self.current_video_dir, self.output_dir_struct[1]),
            None,
            self.frame_control,
        )

        video_handler.extract_frames(start_time=self.start_time, end_time=self.end_time)
        print(f"All Frames of {self.current_video_name} Extracted to {self.output_dir_struct[1]}")

    def extract_cropped_frames(self):
        frame_names_exts = it.image_files_in_directory(os.path.join(self.current_video_dir, self.output_dir_struct[1]))
        frame_paths_names_exts = [
            ft.join(os.path.join(self.current_video_dir, self.output_dir_struct[1]), f) for f in frame_names_exts
        ]

        image_processors = {
            'Crop': CroppingImageProcessor(
                x1=self.crop_coordinates[0],
                x2=self.crop_coordinates[2],
                y1=self.crop_coordinates[1],
                y2=self.crop_coordinates[3],
            ),
            'Echo': EchoImageProcessor(),
        }

        image_processors_list = list(image_processors.values())

        spot_analysis = sa.SpotAnalysis(
            f"{self.current_video_name[:-4]}_crop",
            image_processors_list,
            save_dir=os.path.join(self.current_video_dir, self.output_dir_struct[2]),
            save_overwrite=False,
        )
        spot_analysis.set_primary_images(frame_paths_names_exts)

        for i, result in enumerate(spot_analysis):
            # save out the images and associated attributes
            save_path = spot_analysis.save_image(result)
            if save_path is None:
                lt.warn(
                    f"Failed to save image. Maybe SpotAnalaysis.save_overwrite is False? ({spot_analysis.save_overwrite=})"
                )
            else:
                lt.info(f"Saved image to {save_path}")

            # check that the attributes were saved
            parser = SpotAnalysisOperableAttributeParser(result, sa)
            if i == 0:
                print(f"{parser.image_processors=}")

    def extract_frames_of_interest(self):
        frame_names_exts = it.image_files_in_directory(os.path.join(self.current_video_dir, self.output_dir_struct[1]))
        for i, name in enumerate(frame_names_exts):
            if i < self.start_frame:
                continue
            elif i >= self.start_frame and i <= self.end_frame:
                shutil.copyfile(
                    os.path.join(self.current_video_dir, self.output_dir_struct[1], name),
                    os.path.join(self.current_video_dir, self.output_dir_struct[3], name),
                )
            elif i > self.end_frame:
                continue


class VideoScrubber:
    def __init__(self, root, parent_app):
        self.root = root
        self.root.title("Video Scrubber")
        self.parent_app = parent_app

        # Initialize variables
        self.video_path = None
        self.video_name = None
        self.start_frame = 0
        self.end_frame = 0
        self.total_frames = 0
        self.current_frame = 0
        self.cap = None
        self.frame_image = None
        self.frame_step = 1  # Default step size for advancing frames
        self.corners = []  # List to store the 4 corners
        self.rect_id = None  # ID for the rectangle drawn on the canvas
        self.corner_text_ids = []  # IDs for corner text annotations
        self.crop_coordinates = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Load video button
        self.load_button = tk.Button(self.root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        # Video display
        self.video_label = tk.Canvas(self.root, bg="black")
        self.video_label.pack(fill="both", expand=True)
        self.video_label.bind("<Button-1>", self.on_canvas_click)  # Bind left mouse click
        self.video_label.bind("<Button-3>", self.reset_selection)  # Bind right mouse click
        self.video_label.bind("<Motion>", self.on_mouse_move)  # Bind mouse motion

        # Mouse position readout
        self.mouse_position_label = tk.Label(self.root, text="Mouse Position: (X: 0, Y: 0)")
        self.mouse_position_label.pack()

        # Frame controls
        self.frame_controls = tk.Frame(self.root)
        self.frame_controls.pack()

        self.start_label = tk.Label(self.frame_controls, text="Start Frame:")
        self.start_label.grid(row=0, column=0)
        self.start_entry = tk.Entry(self.frame_controls, width=10)
        self.start_entry.grid(row=0, column=1)

        self.end_label = tk.Label(self.frame_controls, text="End Frame:")
        self.end_label.grid(row=1, column=0)
        self.end_entry = tk.Entry(self.frame_controls, width=10)
        self.end_entry.grid(row=1, column=1)

        # Navigation buttons
        self.prev_button = tk.Button(self.frame_controls, text="Previous Frame", command=self.prev_frame)
        self.prev_button.grid(row=2, column=0)
        self.next_button = tk.Button(self.frame_controls, text="Next Frame", command=self.next_frame)
        self.next_button.grid(row=2, column=1)

        # Current frame indicator
        self.frame_indicator = tk.Label(self.root, text="Frame: 0 / 0")
        self.frame_indicator.pack()

        # Frame navigation slider and input
        self.frame_navigation = tk.Frame(self.root)
        self.frame_navigation.pack()

        self.frame_slider = tk.Scale(
            self.frame_navigation, from_=0, to=0, orient=tk.HORIZONTAL, command=self.jump_to_frame
        )
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.frame_input = tk.Entry(self.frame_navigation, width=10)
        self.frame_input.pack(side=tk.LEFT)
        self.jump_button = tk.Button(self.frame_navigation, text="Jump", command=self.jump_to_frame_input)
        self.jump_button.pack(side=tk.LEFT)

        # Frame step size controls
        self.step_controls = tk.Frame(self.root)
        self.step_controls.pack()

        self.step_label = tk.Label(self.step_controls, text="Step Size:")
        self.step_label.pack(side=tk.LEFT)
        self.step_entry = tk.Entry(self.step_controls, width=10)
        self.step_entry.insert(0, "1")  # Default step size
        self.step_entry.pack(side=tk.LEFT)
        self.set_step_button = tk.Button(self.step_controls, text="Set Step", command=self.set_step_size)
        self.set_step_button.pack(side=tk.LEFT)

        # Start and End Frames button
        self.start_to_end = tk.Button(self.root, text="Return Start and End Frames", command=self.confirm_frame_range)
        self.start_to_end.pack()

        # Confirm button
        self.confirm_button = tk.Button(self.root, text="Confirm Rectangle", command=self.confirm_rectangle)
        self.confirm_button.pack(side=tk.RIGHT)

    def load_video(self):
        # Open file dialog to select video
        self.video_path = filedialog.askopenfilename(
            title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if not self.video_path:
            return

        # Open video using OpenCV
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open video file.")
            return

        # Read the first frame
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read the first frame.")
            return

        # Get total frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        # Update slider and frame indicator
        self.frame_slider.config(to=self.total_frames - 1)
        self.update_frame_indicator()

        # Dynamically resize the canvas to match the frame dimensions
        frame_height, frame_width, _ = frame.shape
        self.video_label.config(width=frame_width, height=frame_height)

        # Display the first frame
        self.display_frame(self.current_frame)
        self.parent_app.set_current_scrub_video(self.video_path)

    def update_frame_indicator(self):
        self.frame_indicator.config(text=f"Frame: {self.current_frame} / {self.total_frames - 1}")
        self.frame_slider.set(self.current_frame)

    def on_canvas_click(self, event):
        # Record the clicked position
        if len(self.corners) < 4:
            self.corners.append((event.x, event.y))
            self.display_corner(event.x, event.y)
            self.draw_rectangle()
        else:
            messagebox.showinfo("Info", "You have already selected 4 corners.")

    def reset_selection(self, event):
        # Reset the selection of corners and clear the canvas annotations
        self.corners = []
        self.corner_text_ids = []
        self.crop_coordinates = None
        if self.rect_id:
            self.video_label.delete(self.rect_id)
        self.rect_id = None

        # Clear corner text annotations
        for text_id in self.corner_text_ids:
            self.video_label.delete(text_id)

        messagebox.showinfo("Info", "Selection reset. You can start over.")

    def on_mouse_move(self, event):
        # Update the mouse position readout
        self.mouse_position_label.config(text=f"Mouse Position: (X: {event.x}, Y: {event.y})")

    def display_corner(self, x, y):
        # Display the corner coordinates on the canvas
        corner_text = f"({x}, {y})"
        text_id = self.video_label.create_text(x + 10, y - 10, text=corner_text, fill="green", font=("Arial", 10))
        self.corner_text_ids.append(text_id)

    def draw_rectangle(self):
        # Draw the rectangle based on the selected corners
        if len(self.corners) == 2:
            x1, y1 = self.corners[0]
            x2, y2 = self.corners[1]
            if self.rect_id:
                self.video_label.delete(self.rect_id)
            self.rect_id = self.video_label.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
        elif len(self.corners) == 4:
            x1, y1 = self.corners[0]
            x2, y2 = self.corners[2]
            if self.rect_id:
                self.video_label.delete(self.rect_id)
            self.rect_id = self.video_label.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def display_frame(self, frame_number):
        if self.cap is None:
            return

        # Set the video to the specified frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to read frame.")
            return

        # Convert the frame to a format suitable for Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        self.frame_image = ImageTk.PhotoImage(image)

        # Update the video display
        self.video_label.delete("all")
        self.video_label.create_image(0, 0, anchor=tk.NW, image=self.frame_image)

        # Draw the rectangle if coordinates are available
        if self.crop_coordinates:
            x_min, y_min, x_max, y_max = self.crop_coordinates
            self.video_label.create_rectangle(x_min, y_min, x_max, y_max, outline="red", width=2)

        # Update the frame indicator
        self.update_frame_indicator()

    def prev_frame(self):
        # Move backward by the step size
        self.current_frame = max(0, self.current_frame - self.frame_step)
        self.update_frame_indicator()

    def next_frame(self):
        # Move forward by the step size
        self.current_frame = min(self.total_frames - 1, self.current_frame + self.frame_step)
        self.update_frame_indicator()

    def set_step_size(self):
        # Set the step size for advancing frames
        try:
            step_size = int(self.step_entry.get())
            if step_size > 0:
                self.frame_step = step_size
            else:
                tk.messagebox.showerror("Error", "Step size must be positive.")
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid step size.")

    def jump_to_frame(self, frame_number):
        # Jump to the frame specified by the slider
        self.current_frame = int(frame_number)
        self.display_frame(self.current_frame)

    def jump_to_frame_input(self):
        # Jump to the frame specified by the input field
        try:
            frame_number = int(self.frame_input.get())
            if 0 <= frame_number < self.total_frames:
                self.current_frame = frame_number
                self.display_frame(self.current_frame)
            else:
                tk.messagebox.showerror("Error", "Frame number out of range.")
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid frame number.")

    def process_frames(self):
        # Get start and end frames from the user
        try:
            self.start_frame = int(self.start_entry.get())
            self.end_frame = int(self.end_entry.get())
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid frame numbers.")
            return

        if self.start_frame < 0 or self.end_frame >= self.total_frames or self.start_frame >= self.end_frame:
            tk.messagebox.showerror("Error", "Frame range is invalid.")
            return

        # Select output directory
        output_path = filedialog.askdirectory(title="Select Output Directory")
        if not output_path:
            return

        # Run ffmpeg command to extract frames
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            self.video_path,
            "-vf",
            f"select='between(n,{self.start_frame},{self.end_frame})'",
            "-vsync",
            "vfr",
            f"{output_path}/{self.video_name}_frame_%04d.png",
        ]

        try:
            subprocess.run(ffmpeg_command, check=True)
            tk.messagebox.showinfo("Success", f"Frames {self.start_frame} to {self.end_frame} processed successfully.")
        except subprocess.CalledProcessError:
            tk.messagebox.showerror("Error", "Failed to process frames with ffmpeg.")

    def return_frames(self):
        # Get start and end frames from the user
        try:
            self.start_frame = int(self.start_entry.get())
            self.end_frame = int(self.end_entry.get())
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid frame numbers.")
            return

        if self.start_frame < 0 or self.end_frame >= self.total_frames or self.start_frame >= self.end_frame:
            tk.messagebox.showerror("Error", "Frame range is invalid.")
            return

        return self.start_frame, self.end_frame

    def confirm_frame_range(self):
        # Get start and end frames from the user
        try:
            self.start_frame = int(self.start_entry.get())
            self.end_frame = int(self.end_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid frame numbers.")
            return

        if self.start_frame < 0 or self.end_frame >= self.total_frames or self.start_frame >= self.end_frame:
            messagebox.showerror("Error", "Frame range is invalid.")
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Pass the frame range back to the main app
        self.parent_app.set_frame_range(self.start_frame, self.end_frame, fps)
        self.parent_app.set_current_scrub_video(self.video_path)

    def confirm_rectangle(self):
        # Ensure 4 corners are selected
        if len(self.corners) != 4:
            messagebox.showerror("Error", "Please select 4 corners of the rectangle.")
            return

        # Calculate the bounding box
        x_coords = [corner[0] for corner in self.corners]
        y_coords = [corner[1] for corner in self.corners]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Display the bounding box coordinates
        messagebox.showinfo("Rectangle Coordinates", f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max})")

        # Save the coordinates for cropping
        self.parent_app.set_crop_coords([x_min, y_min, x_max, y_max])
        self.parent_app.set_current_scrub_video(self.video_path)
        self.crop_coordinates = (x_min, y_min, x_max, y_max)
        print("Crop coordinates saved: ", f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max})")


if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    app = LookBack_ExEx(root)
    root.title("LookBack ExEx Pre Processing")
    root.mainloop()
