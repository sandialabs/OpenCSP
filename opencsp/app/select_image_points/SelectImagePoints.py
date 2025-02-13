"""
When run, instantiates SelectImagePoints class and prompts user
to select image and find key points.

'escape' key closes window.
's' key saves data.

"""

import tkinter as tk
from tkinter.filedialog import askopenfilename
import os

import PIL
from PIL import Image, ImageTk
import imageio.v3 as imageio
import numpy as np
import rawpy

import opencsp.common.lib.tool.tk_tools as tkt


class SelectImagePoints:
    """
    Class to handle displaying images and recording user inputs

    """

    def __init__(self, root: tk.Tk, file_name: str) -> "SelectImagePoints":
        """
        Select file to show

        Parameters
        ----------
            root : tk.Tk
                Root tkinter window in which to create figure
            file_name : str
                Image file to load

        """
        # Define defaults
        frac_window = 0.9
        frac_roi = 0.03

        # Define system parameters
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        self.win_size_max = (int(screen_width * frac_window), int(screen_height * frac_window))
        self.roi_width = int(min(screen_height, screen_width) * frac_roi)  # pixels
        self.save_name = os.path.basename(file_name).split(".")[-2]
        self.image_file_name = file_name
        self.rough = True

        # Create window
        self.root = root
        self.root.overrideredirect(1)
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("s", lambda e: self.save())
        self.root.geometry(
            f"+{int(screen_width * (1 - frac_window) / 2):d}+{int(screen_height * (1 - frac_window) / 2):d}"
        )

        # Create canvas
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack()
        self.canvas.configure(background="black", highlightthickness=0)
        self.canvas.bind("<Button 1>", self.click)

        # Initialize points
        self.pts = []
        self.scale = None
        self.image_tk = None

        # Load image from file
        self.load_image_from_file(file_name)
        self.update_image()

    def run(self) -> None:
        """
        Runs window
        """
        self.root.mainloop()

    def update_image(self) -> None:
        """
        Updates the current displayed image to current loaded image
        """
        self.image_display = self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

    def click(self, event) -> None:
        """
        Called when image is clicked
        """
        if self.rough:
            self.click_rough(event)
        else:
            self.click_fine(event)

        self.rough = not self.rough

    def click_fine(self, event):
        """
        Called when fine res image is clicked
        """
        self.pts[-1] += np.array([event.x / self.scale, event.y / self.scale]).astype(int)  # image pixels
        self.revert_main_image()

    def click_rough(self, event):
        """
        Called when full res image is clicked
        """
        # Save click point
        x_corn = int(event.x / self.scale) - self.roi_width  # image pixels
        y_corn = int(event.y / self.scale) - self.roi_width  # image pixels
        self.pts.append(np.array([x_corn, y_corn]))  # image pixels

        # Show image clip
        x_1 = x_corn
        x_2 = x_corn + self.roi_width * 2
        y_1 = y_corn
        y_2 = y_corn + self.roi_width * 2

        clip = self.image_array_main[y_1:y_2, x_1:x_2]
        self.load_image_from_array(clip)
        self.update_image()

    def revert_main_image(self):
        """
        Reverts the displayed image to the current main image
        """
        self.load_image_from_array(self.image_array_main)
        self.update_image()

    def load_image_from_array(self, image: np.ndarray) -> None:
        """
        Loads image into class from ndarray
        """
        # Create PIL image from array
        image_pil = Image.fromarray(image.copy(), "RGB")

        # Resize image
        size_x = float(self.win_size_max[0]) / image_pil.size[0]  # window pixels / image pixels
        size_y = float(self.win_size_max[1]) / image_pil.size[1]  # window pixels / image pixels
        self.scale = min(size_x, size_y)  # window pixels / image pixels
        shape = (
            int(float(image_pil.size[0]) * self.scale),
            int(float(image_pil.size[1]) * self.scale),
        )  # window pixels
        image_pil = image_pil.resize(shape, PIL.Image.NEAREST)

        # Resize canvas
        self.canvas.configure(height=shape[1], width=shape[0])

        # Convert to photoImage
        self.image_tk = ImageTk.PhotoImage(image_pil)

    def load_image_from_file(self, file: str) -> None:
        """
        Loads image into class given filename
        """
        # Load image
        if file.split(".")[-1] in ["NEF", "RAW", "nef", "raw"]:
            im_array = self._load_raw_image(file)
        else:
            im_array = imageio.imread(file)

        im_array = im_array.astype(float) / float(np.percentile(im_array, 98)) * 255
        im_array[im_array > 255] = 255

        if np.ndim(im_array) == 2:
            im_array = np.concatenate([im_array[..., None]] * 3, axis=2)

        self.image_array_main = im_array.astype("uint8")
        self.load_image_from_array(self.image_array_main)

    def save(self) -> None:
        """
        Saves corner data then closes
        """
        with open(f"points_{self.save_name}.txt", "w", encoding="UTF-8") as file:
            file.write(f"{self.image_file_name:s}\n")
            for point in self.pts:
                file.write(f"{point[0]:.1f}, {point[1]:.1f}\n")
        self.close()

    def close(self) -> None:
        """
        Closes window
        """
        self.root.destroy()

    @staticmethod
    def _load_raw_image(file) -> np.ndarray:
        with rawpy.imread(file) as raw:
            return raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)


if __name__ == "__main__":
    # Select file name
    file_selected = askopenfilename(
        title="Select file to open", filetypes=[("RAW", "*.NEF"), ("RAW", "*.RAW"), ("All Files", "*.*")]
    )
    if file_selected != "":
        # Create window
        win = SelectImagePoints(tkt.window(), file_selected)
        win.run()
