"""
When run, instantiates SelectImagePoints class and prompts user
to select image and find key points.

'escape' key closes window.
's' key saves data.

Notes
-----
1. When the window launches, it first raises a dialog to select an image file. The dialog box only 
   shows .RAW or .NEF files. Switch to "All files" to select images of other types (.jpg, .png, etc). 

2. After selecting an image file, the program displays the image. But for some reason it shows up 
   behind all other windows. If you minimize all other windows, then you can see the image.  
   Note that the display does not include the window's border, or minimize/maximize/close buttons, etc.

3. There are no prompts, but the program silently waits for you to click on the image somewhere.  
   When you do, the image will be replaced by a small tile from the image in the vicinity of where 
   you clicked, shown highly magnified. This enables you to select individual pixels while seeing 
   surrounding context.

4. After you make the fine-grain selection on the enlarged window, the program redisplays the full 
   image and waits.

5. You can then repeat the process to select additional points. The program is logging your selections 
   in the background.

6. When you are done selecting points, press "s" to save and exit. There is no confirmation raised 
   or written to the console.

7. The program writes the selected points in a file "points_<image_name>.txt", which is written to 
   the directory from which you launched the program. The file contains the path to the selected image.

8. Thus, to control the location where the list of points is written, cd to the directory you wish to 
   save to, and then launch the program. You can use the file selection dialog to navigate to the 
   image you want to select.

   For example:
   (env_310_OpenCSP) PS C:\> cd C:\ctemp\select_image_points_test\
   (env_310_OpenCSP) PS C:\ctemp\select_image_points_test> python C:\<path_to_code>\Code\OpenCSP\opencsp\app\select_image_points\SelectImagePoints.py

   The above will work if you have your default python run environment set to the virtual environment 
   env_310_OpenCSP. If not, then you should explicitly specify this to ensure that all of the OpenCSP 
   packages are available. You can do this by:
   PS C:\ctemp\select_image_points_test> C:\<path_to_code>\Code\env_310_OpenCSP\Scripts\python.exe C:\<path_to_code>\Code\OpenCSP\opencsp\app\select_image_points\SelectImagePoints.py

   If you are unsure whether the virtual environment will be run by default, you can use the get-command 
   function:
   (env_310_OpenCSP) PS C:\ctemp\select_image_points_test> get-command python

   This will show which python executable will be run. If the virtual environment is set as default, 
   you should see something like this:
   CommandType     Name                                               Version    Source
   -----------     ----                                               -------    ------
   Application     python.exe                                         3.10.91... C:\<path_to_code>\Code\env_310_OpenCSP\Scripts\python.exe
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
