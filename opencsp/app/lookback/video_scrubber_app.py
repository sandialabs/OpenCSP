import tkinter as tk
from tkinter import filedialog
import subprocess
import cv2
from PIL import Image, ImageTk
import os


class VideoProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processor")

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

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Load video button
        self.load_button = tk.Button(self.root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        # Video display
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

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

        # Process button
        self.process_button = tk.Button(self.root, text="Return Start and End Frames", command=self.return_frames)
        self.process_button.pack()

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
            tk.messagebox.showerror("Error", "Failed to open video file.")
            return

        # Get total frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        # Update slider and frame indicator
        self.frame_slider.config(to=self.total_frames - 1)
        self.update_frame_indicator()

        # Display the first frame
        self.display_frame(self.current_frame)

    def update_frame_indicator(self):
        self.frame_indicator.config(text=f"Frame: {self.current_frame} / {self.total_frames - 1}")
        self.frame_slider.set(self.current_frame)

    def display_frame(self, frame_number):
        if self.cap is None:
            return

        # Set the video to the specified frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret:
            tk.messagebox.showerror("Error", "Failed to read frame.")
            return

        # Convert the frame to a format suitable for Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        self.frame_image = ImageTk.PhotoImage(image)

        # Update the video label
        self.video_label.config(image=self.frame_image)

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

        # Pass the frame range back to the main app
        self.parent_app.set_frame_range(self.start_frame, self.end_frame)
        self.root.destroy()
