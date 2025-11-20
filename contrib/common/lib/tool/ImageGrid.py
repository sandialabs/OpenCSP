"""
image grid generation for example, see 9-up canting detail images



"""

import csv as csv
import os
import sys as sys
from PIL import Image, ImageTk
import tkinter as tk

import numpy as np
import cv2 as cv

import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.Color as clr
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class ImageGrid:
    def __init__(
        self,
        ncols_nrows: tuple[int, int] = None,
        *images: it.ImageLike,
        total_size: tuple[int, int] = None,
        subimage_size: tuple[int, int] = None,
        number_images: bool | int = False,
        numbering_color: tuple[int, int, int] | clr.Color = clr.black(),
        numbering_outline_color: tuple[int, int, int] | clr.Color = None,
    ):
        # validate the input
        if total_size is not None and subimage_size is not None:
            if (total_size[0] % subimage_size[0] != 0) or (total_size[1] % subimage_size[1] != 0):
                lt.error_and_raise(
                    ValueError,
                    "Error in ImageGrid: "
                    + "total_size should be divisible by subimage_size, "
                    + f"but {total_size=} and {subimage_size=}",
                )

        self._ncols_nrows = ncols_nrows
        self._total_size = total_size
        self._subimage_size = subimage_size
        self.number_images = number_images
        self.numbering_color = clr.Color.from_generic(numbering_color)
        self.numbering_outline_color = (
            None if numbering_outline_color is None else clr.Color.from_generic(numbering_outline_color)
<<<<<<< Updated upstream
        )
=======
        ) #clr.Color.from_generic(numbering_outline_color)
>>>>>>> Stashed changes

        self.images: list[Image.Image] = []
        self.add_images(*images)

    @classmethod
    def from_aspect_ratio(
        cls,
        output_aspect_ratio: float | tuple[float, float],
        *images_or_images_aspect_ratio: it.ImageLike | float | tuple[float, float],
    ):
        # normalize input
        if isinstance(images_or_images_aspect_ratio[0], tuple) or isinstance(images_or_images_aspect_ratio[0], list):
            images_aspect_ratio = images_or_images_aspect_ratio[0][0] / images_or_images_aspect_ratio[0][1]
            min_width = 640  # assume images are at least 640 x 480
            min_height = min_width / images_aspect_ratio
            images = []
        elif isinstance(images_or_images_aspect_ratio[0], float) or isinstance(images_or_images_aspect_ratio[0], int):
            images_aspect_ratio = images_or_images_aspect_ratio[0]
            min_width = 640  # assume images are at least 640 x 480
            min_height = min_width / images_aspect_ratio
            images = []
        else:
            images = images_or_images_aspect_ratio
            images = [it.to_image(img, "pillow") for img in images]
            min_width = np.min([img.width for img in images])
            min_height = np.min([img.height for img in images])
            images_aspect_ratio = min_width / min_height

        if isinstance(output_aspect_ratio, list) or isinstance(output_aspect_ratio, tuple):
            output_aspect_ratio = output_aspect_ratio[0] / output_aspect_ratio[1]

        # determine the desired output sub-image resolution
        min_width = min([img.width for img in images])
        min_height = min([img.height for img in images])
        if min_width / output_aspect_ratio < min_height:
            desired_sub_width, desired_sub_height = min_width, min_width / output_aspect_ratio
        else:
            desired_sub_width, desired_sub_height = min_height * output_aspect_ratio, min_height

        # determine the output image resolution
        for ncols in range(1, len(images) + 1):
            out_width = int(np.round(ncols * desired_sub_width))
            out_height = int(np.round(out_width / output_aspect_ratio))
            nrows = int(np.round(out_height / desired_sub_height))
            needed_nrows = int(np.ceil(len(images) / ncols))
            if nrows >= needed_nrows:
                continue
        assert ncols * nrows >= len(images)

        # build the instance
        return cls((nrows, ncols), *images, total_size=(out_width, out_height))

    @property
    def ncols(self) -> int:
        return self._ncols_nrows[0]

    @property
    def nrows(self) -> int:
        return self._ncols_nrows[1]

    @property
    def subimage_size(self) -> tuple[int, int]:
        if self._total_size is None:
            if self._subimage_size is None:
                min_width = np.min([img.width for img in self.images])
                min_height = np.min([img.height for img in self.images])
                return min_width, min_height
            else:
                return self._subimage_size
        else:
            subimage_width = int(np.round(self._total_size[0] / self.ncols))
            subimage_height = int(np.round(self._total_size[1] / self.nrows))

            if self._subimage_size is None:
                return subimage_width, subimage_height
            else:
                subimage_width = min(subimage_width, self._subimage_size[0])
                subimage_height = min(subimage_height, self._subimage_size[1])
                return subimage_width, subimage_height

    def add_images(self, *images: it.ImageLike):
        self.images += [it.to_image(img, "pillow") for img in list(images)]

    def tile_images(self) -> Image:
        # Resize the images
        sub_width, sub_height = self.subimage_size
        images = [img.resize([sub_width, sub_height]) for img in self.images]

        # determine the output image resolution
        nrows, ncols = self.nrows, self.ncols
        out_width, out_height = sub_width * ncols, sub_height * nrows

        # Generate the output image
        ret = Image.new(mode=images[0].mode, size=(out_width, out_height))
        for row in range(nrows):
            for col in range(ncols):
                image_idx = row * ncols + col
                if image_idx >= len(images):
                    break
                subimg = images[image_idx]

                # apply numbering
                if self.number_images is not False:
                    starting_value = 1 if self.number_images is True else self.number_images
                    image_number = image_idx + starting_value
                    x, y, s = int(10 / 600 * sub_height), int(30 / 600 * sub_height), 1 / 600 * sub_height
                    subimg = np.array(subimg)
                    if self.numbering_outline_color is not None:
                        subimg = cv.putText(
                            subimg,
                            str(image_number),
                            (x, y),
                            cv.FONT_HERSHEY_DUPLEX,
                            s,
                            self.numbering_outline_color.rgb_255(),
                            thickness=3,
                        )
                    subimg = cv.putText(
                        subimg,
                        str(image_number),
                        (x, y),
                        cv.FONT_HERSHEY_DUPLEX,
                        s,
                        self.numbering_color.rgb_255(),
                        thickness=2,
                    )
                    subimg = Image.fromarray(subimg)

                # add to the return value
                ret.paste(subimg, [col * sub_width, row * sub_height])

        return ret

    def tile_images_tk(self):
        root = tk.Tk()
        root.title("Image Grid")

        # Use a standard ordering, versus whatever order Tk decides is best.
        # Tkinter's order:
        #     9,8,7
        #     6,5,4
        #     3,2,1
        # Our order:
        #     1,2,3
        #     4,5,6
        #     7,8,9
        nrows, ncols = self.nrows, self.ncols
        curr_row, curr_col = nrows - 1, ncols - 1

        # Get the necessary subimage size
        subimage_size = self.subimage_size

        # Create a grid of images
        for i, image_file in enumerate(images):
            image_path = os.path.join(image_folder, image_file)
            img = Image.open(image_path)

            # Resize the image
            if subimage_size is None:
                photo = ImageTk.PhotoImage(img)
            else:
                photo = ImageTk.PhotoImage(img.resize(subimage_size, Image.LANCZOS))

            # Create a label to display the image
            label = tk.Label(root, image=photo)
            label.image = photo  # Keep a reference to avoid garbage collection

            # Place the label in the grid
            label.grid(row=curr_row, column=curr_col)  # 3 columns for a 3x3 grid

            # Update the row/col index
            curr_col -= 1
            if curr_col < 0:
                curr_col = ncols - 1
                curr_row -= 1

        # Start the Tkinter event loop
        root.mainloop()

    # Example usage


if __name__ == "__main__":
    image_folder = ft.join(
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
    image_order = [
        'tca059_on-Axis_Canted_NSTTF_Heliostat_14W6_exaggerated_z__3d_[3.22]z[3.93].png',  # 1
        'tca059_on-Axis_Canted_NSTTF_Heliostat_14W1_exaggerated_z__3d_[4.42]z[5.13].png',  # 2
        'tca059_on-Axis_Canted_NSTTF_Heliostat_14E6_exaggerated_z__3d_[5.73]z[6.44].png',  # 3
        'tca059_on-Axis_Canted_NSTTF_Heliostat_9W11_exaggerated_z__3d_[1.91]z[2.62].png',  # 4
        'tca059_on-Axis_Canted_NSTTF_Heliostat_9W1_exaggerated_z__3d_[4.22]z[4.93].png',  # 5
        'tca059_on-Axis_Canted_NSTTF_Heliostat_9E11_exaggerated_z__3d_[6.41]z[7.12].png',  # 6
        'tca059_on-Axis_Canted_NSTTF_Heliostat_5W9_exaggerated_z__3d_[2.53]z[3.24].png',  # 7
        'tca059_on-Axis_Canted_NSTTF_Heliostat_5W1_exaggerated_z__3d_[3.85]z[4.56].png',  # 8
        'tca059_on-Axis_Canted_NSTTF_Heliostat_5E9_exaggerated_z__3d_[5.2]z[5.91].png',  # 9
    ]
    images = [ft.join(image_folder, image_name_ext) for image_name_ext in image_order]

    grid = ImageGrid((3, 3), images, subimage_size=(400, 350))
    grid.tile_images_tk()
