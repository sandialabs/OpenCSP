"""
image grid generation for example, see 9-up canting detail images



"""

import csv as csv
import os
import sys as sys
from PIL import Image, ImageTk
import tkinter as tk

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp


class ImageGrid:
    def tile_images(self, image_folder):
        """
        Parameters
        ----------
        image_folder : location of directory that holds all the images to put into the grid

        """
        root = tk.Tk()
        root.title("Image Grid")

        # Define the specific order of images (can add more or less images, will have to change the code to correspond)
        # image is ordered as such:
        # 9,8,7
        # 6,5,4
        # 3,2,1
        image_order = [
            'tca059_on-Axis_Canted_NSTTF_Heliostat_5E9_exaggerated_z__3d_[5.2]z[5.91].png',  # 1
            'tca059_on-Axis_Canted_NSTTF_Heliostat_5W1_exaggerated_z__3d_[3.85]z[4.56].png',  # 2
            'tca059_on-Axis_Canted_NSTTF_Heliostat_5W9_exaggerated_z__3d_[2.53]z[3.24].png',  # 3
            'tca059_on-Axis_Canted_NSTTF_Heliostat_9E11_exaggerated_z__3d_[6.41]z[7.12].png',  # 4
            'tca059_on-Axis_Canted_NSTTF_Heliostat_9W1_exaggerated_z__3d_[4.22]z[4.93].png',  # 5
            'tca059_on-Axis_Canted_NSTTF_Heliostat_9W11_exaggerated_z__3d_[1.91]z[2.62].png',  # 6
            'tca059_on-Axis_Canted_NSTTF_Heliostat_14E6_exaggerated_z__3d_[5.73]z[6.44].png',  # 7
            'tca059_on-Axis_Canted_NSTTF_Heliostat_14W1_exaggerated_z__3d_[4.42]z[5.13].png',  # 8
            'tca059_on-Axis_Canted_NSTTF_Heliostat_14W6_exaggerated_z__3d_[3.22]z[3.93].png',  # 9
        ]

        # Ensure there are exactly 9 images in the specified order
        images = [f for f in image_order if os.path.isfile(os.path.join(image_folder, f))]
        if len(images) != 9:
            raise ValueError("There must be exactly 9 images in the specified order in the directory.")

        # Set the size for each image
        # img_size = (610, 330)  # Adjust the size as needed (for width to height ratio of 3:2)
        img_size = (400, 350)  # for square images

        # Create a grid of images
        for i, image_file in enumerate(images):
            image_path = os.path.join(image_folder, image_file)
            img = Image.open(image_path)

            # Resize the image
            photo = ImageTk.PhotoImage(img.resize(img_size, Image.LANCZOS))

            # Create a label to display the image
            label = tk.Label(root, image=photo)
            label.image = photo  # Keep a reference to avoid garbage collection

            # Place the label in the grid
            label.grid(row=i // 3, column=i % 3)  # 3 columns for a 3x3 grid

        # Start the Tkinter event loop
        root.mainloop()

    # Example usage


if __name__ == "__main__":
    image_folder = os.path.join(
        orp.opencsp_code_dir(),
        'common',
        'lib',
        'test',
        'data',
        'input',
        'sandia_nsttf_test_definition',
        'NSTTF_Canting_Prescriptions',
        'images_to_grid',
    )  # Change this to your image folder path, just an example
    grid = ImageGrid()
    grid.tile_images(image_folder=image_folder)
