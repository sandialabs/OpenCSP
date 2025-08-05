import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
import tkinter as tk
from tkinter import ttk

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.cv.image_reshapers as ir


class ImageComparisonTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparison Tool")

        # Initialize variables
        self.img1: np.ndarray = None
        self.img2: np.ndarray = None
        self.cached_gaussian_img1: np.ndarray = None
        self.cached_gaussian_img2: np.ndarray = None
        self.small_img1: Image.Image = None
        self.small_img2: Image.Image = None
        self.display_img1: Image.Image = None
        self.display_img2: Image.Image = None
        self.highlight_threshold = tk.DoubleVar(value=0)
        self.apply_gaussian = tk.BooleanVar(value=False)
        self.mode = tk.StringVar(value="raw_images")
        self.resize_timeout = None  # Timeout for resizing canvas

        # Create UI elements
        self.create_ui()

    def create_ui(self):
        def create_tooltip(widget, text):
            """Creates a floating tooltip for a widget."""
            tooltip = tk.Label(
                self.root, text=text, bg="yellow", fg="black", relief="solid", borderwidth=1, wraplength=200
            )
            tooltip.place_forget()

            def on_enter(event):
                x, y, _, _ = widget.bbox("insert")
                tooltip.place(x=widget.winfo_rootx() + x, y=widget.winfo_rooty() + y + 20)

            def on_leave(event):
                tooltip.place_forget()

            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        # Create a container frame to hold all elements
        self.container = ttk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Highlight Difference Numeric Input
        self.highlight_container = ttk.Frame(self.container)
        self.highlight_container.grid(row=1, column=0, columnspan=3, sticky='W')
        self.highlight_label = ttk.Label(self.highlight_container, text="Highlight Difference Threshold:")
        self.highlight_label.grid(row=0, column=0, padx=5, pady=5, sticky='W')
        self.highlight_input = ttk.Entry(self.highlight_container, textvariable=self.highlight_threshold)
        self.highlight_input.grid(row=0, column=1, padx=5, pady=5, sticky='W')
        self.highlight_input_update = ttk.Button(
            self.highlight_container, text="Update", command=self.validate_highlight_difference
        )
        self.highlight_input_update.grid(row=0, column=2, padx=5, pady=5, sticky='W')
        create_tooltip(self.highlight_input, "Set the threshold for highlighting differences")

        # Gaussian Checkbox
        self.gaussian_checkbox = ttk.Checkbutton(
            self.container, text="Apply Gaussian Blur", variable=self.apply_gaussian, command=self.update_images
        )
        self.gaussian_checkbox.grid(row=2, column=0, padx=5, pady=5, sticky='W')
        create_tooltip(self.gaussian_checkbox, "Apply a 2px Gaussian blur to both images")

        # Mode Radio Buttons (placed next to each other on the same row)
        self.raw_images_radio = ttk.Radiobutton(
            self.container, text="Compare Images", variable=self.mode, value="raw_images", command=self.update_images
        )
        self.raw_images_radio.grid(row=3, column=0, padx=5, pady=5, sticky='W')
        create_tooltip(self.raw_images_radio, "Just show the two images as they are")

        self.highlight_diffs_radio = ttk.Radiobutton(
            self.container,
            text="Highlight Diffs",
            variable=self.mode,
            value="highlight_diffs",
            command=self.update_images,
        )
        self.highlight_diffs_radio.grid(row=3, column=1, padx=5, pady=5, sticky='W')
        create_tooltip(self.highlight_diffs_radio, "Highlight differences in red in both images")

        self.show_differences_radio = ttk.Radiobutton(
            self.container,
            text="Show Differences",
            variable=self.mode,
            value="show_differences",
            command=self.update_images,
        )
        self.show_differences_radio.grid(row=3, column=2, padx=5, pady=5, sticky='W')
        create_tooltip(self.show_differences_radio, "Replace the first image with the difference image")

        # Display Area for Images
        self.image_canvas = tk.Canvas(self.container, width=800, height=400)
        self.image_canvas.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky='W')
        self.image_canvas.bind("<Motion>", self.on_mouse_motion)

        # Bind the resize function to the Configure event
        self.root.bind("<Configure>", self.resize_canvas)

    def validate_highlight_difference(self):
        """Validates the highlight difference threshold input."""
        try:
            value = int(self.highlight_input.get())
            if 0 <= value:
                self.highlight_threshold.set(value)
                self.update_images()
            else:
                lt.error(
                    "Error: Highlight difference threshold must be greater than or equal to 0 (and probably less than 255)."
                )
        except ValueError:
            lt.error("Error: Highlight difference threshold must be an integer.")

    def get_img_fit_size(self):
        """Returns the width and height of the images to fit them within the available canvas space."""
        (img_height, img_width), nchannels = it.dims_and_nchannels(self.img1)
        can_width = self.image_canvas.winfo_width()
        can_height = self.image_canvas.winfo_height()

        fit_height = min(can_height, img_height)
        fit_width = (fit_height / img_height) * img_width
        fit_width = int(fit_width)
        fit_height = int(fit_height)

        # sanity check
        assert fit_width <= can_width
        assert fit_height <= can_height

        return fit_width, fit_height

    def resize_canvas(self, event):
        """Resizes the canvas to fit the window after a timeout."""
        if self.resize_timeout:
            self.root.after_cancel(self.resize_timeout)

        self.resize_timeout = self.root.after(200, self._resize_canvas)

    def _resize_canvas(self):
        """Resizes the canvas based on the window size, subtracting the size of all other elements."""
        win_width = self.root.winfo_width()
        win_height = self.root.winfo_height()

        # Calculate the total height of all other elements in the container
        other_elements_height = sum(
            widget.winfo_height() for widget in self.container.winfo_children() if widget != self.image_canvas
        )

        # Calculate new canvas dimensions
        new_width = max(win_width - 40, 800)  # Ensure a minimum width of 800
        new_height = max(win_height - other_elements_height - 40, 400)  # Ensure a minimum height of 400

        # Update the canvas size
        self.image_canvas.config(width=new_width, height=new_height)

        # Update the displayed images to fit the new canvas size
        self.update_images()

    def on_mouse_motion(self, event):
        x = event.x
        fit_width, fit_height = self.get_img_fit_size()
        self.display_images(np.clip(x / fit_width, 0.0, 1.0))

    def update_images(self):
        if self.img1 is None or self.img2 is None:
            # Alert the user that there are no images to show
            lt.error("Error: No images loaded for comparison!")
            return

        # Use copies so that we can apply edits freely without changing the underlying data
        display_img1 = self.img1.copy()
        display_img2 = self.img2.copy()

        # Apply Gaussian blur and cache the results
        # Use cached results if available
        if self.apply_gaussian.get():
            if self.cached_gaussian_img1 is None or self.cached_gaussian_img2 is None:
                self.cached_gaussian_img1 = cv2.GaussianBlur(display_img1, ksize=(5, 5), sigmaX=1)
                self.cached_gaussian_img2 = cv2.GaussianBlur(display_img2, ksize=(5, 5), sigmaX=1)
            display_img1 = self.cached_gaussian_img1.copy()
            display_img2 = self.cached_gaussian_img2.copy()

        # Process images based on selected mode
        if self.mode.get() == "highlight_diffs":
            diff = np.abs(display_img2 - display_img1)
            diffs = [diff[:, :, i] > self.highlight_threshold.get() for i in range(3)]
            mask = diffs[0] & diffs[1] & diffs[2]
            display_img1[mask, :] = [0, 0, 255]  # Highlight differences in red
        elif self.mode.get() == "show_differences":
            display_img1 = np.abs(display_img2 - display_img1)
        else:
            pass  # raw images

        # Convert to Pillow images
        self.display_img1 = Image.fromarray(cv2.cvtColor(display_img1, cv2.COLOR_BGR2RGB))
        self.display_img2 = Image.fromarray(cv2.cvtColor(display_img2, cv2.COLOR_BGR2RGB))

        self.display_images(update=True)

    def display_images(self, percent_img1=1.0, update=False):
        # Get the split width
        fit_width, fit_height = self.get_img_fit_size()
        split_width = np.clip(int(np.round(percent_img1 * fit_width)), 0, fit_width)

        # Resize for the images, as necessary
        if (
            update
            or self.small_img1 is None
            or self.small_img1.width != fit_width
            or self.small_img1.height != fit_height
        ):
            self.small_img1 = self.display_img1.resize(
                (int(fit_width), int(fit_height)), resample=Image.Resampling.LANCZOS
            )
            self.small_img2 = self.display_img2.resize(
                (int(fit_width), int(fit_height)), resample=Image.Resampling.LANCZOS
            )

        # Crop the images to fit next to each other
        img1_cropped = self.small_img1.crop((0, 0, split_width, fit_height))
        img2_cropped = self.small_img2.crop((split_width, 0, fit_width, fit_height))

        # Create ImageTk objects
        img1_tk_cropped = ImageTk.PhotoImage(img1_cropped)
        img2_tk_cropped = ImageTk.PhotoImage(img2_cropped)

        # Display images side by side
        self.image_canvas.create_image(0, 0, image=img1_tk_cropped, anchor=tk.NW)
        self.image_canvas.create_image(split_width, 0, image=img2_tk_cropped, anchor=tk.NW)

        # Keep references to avoid garbage collection
        self.image_canvas.img1_tk = img1_tk_cropped
        self.image_canvas.img2_tk = img2_tk_cropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Comparison Tool")
    parser.add_argument(
        "--base-path", required=False, dest="basepath", default="", help="The directory to open images relative to."
    )
    parser.add_argument("paths", nargs="+", type=str, help="Paths to images")
    args = parser.parse_args()
    basepath: str = args.basepath
    paths: list[str] = list(args.paths)

    # To facilitate the usage of multiline strings (such as from error logs) use double quotes
    splitchars = ["\n", "\r", "\t", ":", "=", "[", "]", "(", ")", "<", ">", ",", "vs", "vs.", "'", "\"", ";", "|"]
    for splitchar in splitchars:
        new_paths: list[str] = []
        for path in paths:
            new_paths += path.split(splitchar)
        paths = new_paths
    paths = [os.path.expanduser(p.strip()) for p in paths]
    paths = list(filter(lambda p: p != "", paths))

    # Change directories to the base path directory
    if basepath != "":
        basepath = os.path.expanduser(basepath)
        os.chdir(basepath)

    # Filter valid image paths
    valid_file_paths = [path for path in paths if ft.file_exists(path)]
    valid_image_paths = [
        path for path in paths if ft.path_components(path)[2].strip(".") in it.pil_image_formats_readable
    ]

    if len(valid_image_paths) != 2:
        lt.error(
            f"Error: exactly two valid image paths are required! Found {len(valid_image_paths)} ({valid_image_paths=})"
        )
        sys.exit(1)

    # Select the first two valid image paths
    image1_path, image2_path = valid_image_paths[:2]

    imgs = []
    try:
        # Open the images and add additional color channels as necessary
        for no, path in [(1, image1_path), (2, image2_path)]:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                if " " in path:
                    for part in path.split(" "):
                        img = cv2.imread(part, cv2.IMREAD_COLOR)
                        if img is not None:
                            path = part
                            break
            if img is None:
                raise ValueError(f"In image_diff: error while trying to open image {no} ({path=})!")
            img = ir.nchannels_reshaper(img, 3)
            imgs.append(img)
    except Exception as ex:
        raise RuntimeError(f"In image_diff: failed to read or reshape image {no} ({path=})!") from ex
    img1 = imgs[0]
    img2 = imgs[1]

    try:
        # Resize second image if necessary
        if img1.shape != img2.shape:
            lt.info(f"Resizing second image from {img2.shape} to {img1.shape} to match the shape of the first image.")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    except Exception as ex:
        raise RuntimeError(
            f"In image_diff: failed to resize img2 to match the shape of img1.\n"
            + f"\t{image1_path=}\n\t{image2_path=}"
        ) from ex

    # Start the GUI
    root = tk.Tk()
    app = ImageComparisonTool(root)
    app.img1 = img1
    app.img2 = img2
    root.mainloop()
