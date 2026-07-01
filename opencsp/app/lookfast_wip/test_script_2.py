import tkinter as tk
from tkinter import filedialog, messagebox
import os
import time
import shutil
import cv2
from PIL import Image, ImageTk


class LookBack_ExEx:
    def __init__(self, root):
        self.root = root
        self.root.title("LookBack Pre-Processing User Interface")

        # Initialize variables
        self.selected_files = []
        run_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
        self.key = str(run_time + "_ExEx_Test")
        self.video_in_dirs = None
        self.video_files = None
        self.output_dir_top = None
        self.output_video_folders = None
        self.results = ""
        self.start_frame = None
        self.end_frame = None

        # Create the main layout
        self.create_widgets()

    def create_widgets(self):
        # Left frame for buttons and file selection
        self.left_frame = tk.Frame(self.root, padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=True)

        # Bottom frame for output box
        self.bottom_frame = tk.Frame(self.left_frame, padx=10, pady=10)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Buttons for running specific sections of code
        self.select_file_button = tk.Button(self.left_frame, text="Select Files", command=self.select_files)
        self.select_file_button.pack(pady=5)

        self.select_output_dir_button = tk.Button(self.left_frame, text="Select Output Dir", command=self.select_dir)
        self.select_output_dir_button.pack(pady=5)

        self.output_structure_button = tk.Button(
            self.left_frame, text="Create Output Structure", command=self.create_output_dir_structure
        )
        self.output_structure_button.pack(pady=5)

        self.video_scrubber_button = tk.Button(
            self.left_frame, text="Open VideoScrubber", command=self.open_video_scrubber
        )
        self.video_scrubber_button.pack(pady=5)

        # Listbox to display selected files
        self.file_listbox = tk.Listbox(self.left_frame, height=10, width=40)
        self.file_listbox.pack(pady=5)

        # Output box for results
        self.output_label = tk.Label(self.bottom_frame, text="Output Results:", anchor="w")
        self.output_label.pack(fill=tk.X)

        self.output_text = tk.Text(self.bottom_frame, height=10, wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def update_file_listbox(self):
        # Clear the listbox and add selected files
        self.file_listbox.delete(0, tk.END)
        for file in self.selected_files:
            self.file_listbox.insert(tk.END, file)

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

    def select_files(self, dialog="Pick File(s) for..."):
        # Open a file selection dialog and allow the user to select multiple files
        file_paths = filedialog.askopenfilenames(title=dialog, filetypes=(("MOV files", "*.mov"), ("All files", "*.*")))
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
        dir_path = filedialog.askdirectory(title=dialog)
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
                os.mkdir(os.path.join(output_struct, r'1_video_frames'))
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

    def set_frame_range(self, start_frame, end_frame):
        # Set the start and end frames from the VideoScrubber
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.append_to_output(f"Selected frame range: Start = {start_frame}, End = {end_frame}")


class VideoScrubber:
    def __init__(self, root, parent_app):
        self.root = root
        self.root.title("Video Scrubber")
        self.parent_app = parent_app  # Reference to the main app

        # Initialize variables
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.frame_image = None
        self.start_frame = 0
        self.end_frame = 0

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Load video button
        self.load_button = tk.Button(self.root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        # Video display
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Scrubbing slider
        self.scrub_slider = tk.Scale(self.root, from_=0, to=0, orient=tk.HORIZONTAL, command=self.scrub_video)
        self.scrub_slider.pack(fill=tk.X, expand=True)

        # Frame range selection
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

        # Confirm button
        self.confirm_button = tk.Button(self.root, text="Confirm Frame Range", command=self.confirm_frame_range)
        self.confirm_button.pack()

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

        # Get total frames
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

        # Update slider
        self.scrub_slider.config(to=self.total_frames - 1)

        # Display the first frame
        self.display_frame(self.current_frame)

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

        # Update the video label
        self.video_label.config(image=self.frame_image)

    def scrub_video(self, frame_number):
        # Scrub to the selected frame using the slider
        self.current_frame = int(frame_number)
        self.display_frame(self.current_frame)

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


if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    app = LookBack_ExEx(root)
    root.mainloop()
