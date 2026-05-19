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

import opencsp.app.lookback.lookback_tools as lbt

import opencsp.common.lib.render.VideoHandler as vh
import opencsp.common.lib.render_control.RenderControlVideoFrames as rcvf
from opencsp.common.lib.process.parallel_video_tools import parallel_video_to_frames
import opencsp.common.lib.process.ServerSynchronizer as ss
from opencsp.common.lib.cv.spot_analysis.image_processor.CroppingImageProcessor import CroppingImageProcessor
from opencsp.common.lib.cv.spot_analysis.image_processor.EchoImageProcessor import EchoImageProcessor
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.file_tools as ft
from opencsp.common.lib.cv import SpotAnalysis as sa
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperableAttributeParser import SpotAnalysisOperableAttributeParser
import opencsp.common.lib.tool.log_tools as lt

# from contrib.common.lib.cv.spot_analysis.image_processor.StabilizationImageProcessor import StabilizationImageProcessor


class VideoScrubber:
    def __init__(self, root, video_path=None):
        self.root = root
        self.root.title("Video Scrubber")

        self.root.protocol("WM_Delete_Window", self.on_exit)

        # Initialize variables
        self.video_path = video_path
        self.video_name = None
        self.start_frame = 0
        self.end_frame = 0
        self.total_frames = 0
        self.fps = None
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
        if self.video_path is None:
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
        # self.video_label.config(image=self.frame_image)

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
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

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
        self.crop_coordinates = (x_min, y_min, x_max, y_max)
        print("Crop coordinates saved: ", f"Top-left: ({x_min}, {y_min}), Bottom-right: ({x_max}, {y_max})")

    def on_exit(self):

        self.confirm_frame_range()
        self.confirm_rectangle()

        self.root.destroy()


def extract_all_frames(source_vid_path, frame_dest_path):
    # sync_obj = ss.ServerSynchronizer
    frame_control = rcvf.RenderControlVideoFrames(outframe_name="-%05d", outframe_format="png")
    vid_handler = vh.VideoHandler.VideoExtractor(source_vid_path, frame_dest_path, None, frame_control)

    # parallel_video_to_frames(num_servers=4, server_index=0, video_handler=vid_handler, server_synchronizer=sync_obj)
    vid_handler.extract_frames(start_time=None, end_time=None)
    print(f"All Frames of {source_vid_path} Extracted to {frame_dest_path}")


def shuttil_frames(image_folder, destination_folder, start_frame, end_frame):
    frame_names_exts = it.image_files_in_directory(image_folder)
    for name in frame_names_exts:
        frame_num = lbt.frame_number_from_img_name(name)

        if frame_num < start_frame:
            continue
        elif frame_num >= start_frame and frame_num <= end_frame:
            shutil.copyfile(os.path.join(image_folder, name), os.path.join(destination_folder, name))
        elif frame_num > end_frame:
            continue


def interactive_video_select(video_path, dest_path, frame_subset_dir):
    is_video = os.path.exists(video_path)
    if not is_video:
        raise ValueError("Invalid Path to Video File")

    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(frame_subset_dir, exist_ok=True)

    root = tk.Tk()
    vid_scrub_app = VideoScrubber(root, video_path=video_path)
    root.title("LookFast Video Scrubber")
    root.mainloop()

    extract_all_frames(source_vid_path=video_path, frame_dest_path=dest_path)
    shuttil_frames(
        image_folder=dest_path,
        destination_folder=frame_subset_dir,
        start_frame=int(vid_scrub_app.start_frame),
        end_frame=int(vid_scrub_app.end_frame),
    )
    return {
        "start_frame": vid_scrub_app.start_frame,
        "end_frame": vid_scrub_app.end_frame,
        "video_path": vid_scrub_app.video_path,
        "roi_extent": vid_scrub_app.crop_coordinates,
        "roi_coords": vid_scrub_app.corners,
    }


if __name__ == "__main__":
    # Create the main window
    pass
    '''
    root = tk.Tk()
    app = LookBack_ExEx(root)
    root.title("LookBack ExEx Pre Processing")
    root.mainloop()
    '''
