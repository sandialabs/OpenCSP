import os
import re
import sys

import numpy as np
import numpy.typing as npt
from PIL import Image

from opencsp import opencsp_settings
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class NSTTFPixelDataConverter:
    """
    Some of the files that the NSTTF BCS camera capture software generates are the
    raw pixel data, in the form of text-readable values, one value per pixel, one
    row per line. For example, a 5x5 image with a bright white dot in the center
    would look like::

        0,0,0,0,0
        0,0,0,0,0
        0,0,255,0,0
        0,0,0,0,0
        0,0,0,0,0
        70,0,0,0,0

    Notice the last line, which starts with "70" followed by many 0s. This indicates
    the end of an image (an image break). For files with multiple images in them,
    there will be more data after this line, and another image break line after each
    image.

    This class interprets the image data from a yyyymmDD_HHMMss_PixelData.csv file.
    """

    def __init__(self, csv_path_name_ext: str):
        self.csv_path_name_ext = csv_path_name_ext
        self._nrows: int = None
        self._ncols: int = None
        self._nimages: int = None

        self._parse_file_stats()

    def raw_image_names(self) -> list[str]:
        """The names of the images in the associated "Raw Images" directory."""
        csv_path, csv_name, csv_ext = ft.path_components(self.csv_path_name_ext)
        raw_dir = ft.join(csv_path, "Raw Images")
        image_files = it.image_files_in_directory(raw_dir)
        return image_files

    def _parse_file_stats(self):
        if self._nrows is not None:
            return

        # parse the file statistics
        example_line = ""
        nrows = 0
        nempty = 0
        with open(self.csv_path_name_ext, "r") as fin:
            for row in fin:
                row = row.strip()
                if row != "":
                    example_line = row
                    nrows += 1
                else:
                    nempty += 1
        if example_line == "" or nrows == 0:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): "
                + f"expected at least one row for the csv file {self.csv_path_name_ext}, "
                + f"but found {nrows} rows",
            )
        ncols = len(example_line.split(","))

        # retrieve an example image, to compare height and width to rows and columns
        image_files = self.raw_image_names()
        if len(image_files) == 0:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): " + f"expected at least one image in {raw_dir}, but found 0",
            )
        expected_nimages = len(image_files)
        raw_dir = ft.join(ft.path_components(self.csv_path_name_ext)[0], "Raw Images")
        image_file = ft.join(raw_dir, image_files[0])
        img = Image.open(image_file)

        # validate the parsed statistics
        expected_height, expected_width = img.height, img.width
        if ncols != expected_width:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): "
                + f"expected images in {self.csv_path_name_ext} to be {expected_width} pixels wide, "
                + f"but images are instead {ncols} columns wide",
            )
        nimages = int(np.floor(nrows / expected_height))
        if nimages != expected_nimages:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): "
                + f"expected to find {expected_nimages} images in {self.csv_path_name_ext}, "
                + f"but instead found {nimages}",
            )
        expected_nrows = nimages * (expected_height + 1)
        if nrows != expected_nrows:
            lt.error_and_raise(
                RuntimeError,
                "Error in PixelData._get_file_stats(): "
                + f"expected to find {expected_nrows} rows ({expected_height}+1 per image) in {self.csv_path_name_ext}, "
                + f"but instead found {nrows}",
            )

        # statistics look good, cache them
        self._nrows = nrows
        self._ncols = ncols
        self._nimages = nimages

    @property
    def nrows(self) -> int:
        """The total number of non-empty rows in this instance's "PixelData" csv file."""
        return self._nrows

    @property
    def ncols(self) -> int:
        """The number of columns per row in this instance's "PixelData" csv file."""
        return self._ncols

    @property
    def nimages(self) -> int:
        """The total number of images in this instance's "PixelData" csv file."""
        return self._nimages

    @property
    def width(self) -> int:
        """The width of each of this instance's images."""
        return self.ncols

    @property
    def height(self) -> int:
        """The height of each of this instance's images."""
        return int(np.floor(self.nrows / self.nimages)) - 1

    def parse_images(self) -> list[np.ndarray]:
        """Retrieves the images from this instance's "PixelData" CSV file."""
        ret: list[np.ndarray] = []

        with open(self.csv_path_name_ext, "r") as fin:
            row_idx = -1

            for i in range(self.nimages):
                image = np.zeros((self.height, self.width), dtype=np.uint8)

                for y in range(self.height):
                    row_idx += 1
                    row = fin.readline()
                    row = row.strip()
                    svals = row.split(",")
                    if len(svals) != self.width:
                        lt.error_and_raise(
                            RuntimeError,
                            f"Unexpected error in PixelData.parse_images() on line {row_idx} of {self.csv_path_name_ext}",
                        )

                    for x, sval in enumerate(svals):
                        if sval == "":
                            lt.error_and_raise(
                                RuntimeError,
                                f"Unexpected error in PixelData.parse_images() on line {row_idx} of {self.csv_path_name_ext}",
                            )
                        val = int(sval)
                        if x == 0 and val == 70:
                            pass
                        if val < 0 or val > 255:
                            lt.error_and_raise(
                                RuntimeError,
                                f"Unexpected error in PixelData.parse_images() on line {row_idx} of {self.csv_path_name_ext}",
                            )
                        image[y, x] = val

                row_idx += 1
                image_break_row = fin.readline()
                image_break_row = image_break_row.strip()
                if not image_break_row.startswith("70,"):
                    lt.error_and_raise(
                        RuntimeError,
                        "Error in PixelData.parse_images(): "
                        + f"expected the image break row {row_idx} to start with \"70,\", "
                        + f"but instead it starts with \"{image_break_row[:10]}\"",
                    )
                image_break_row_zeros = image_break_row[3:]
                empty_image_break_row = image_break_row_zeros.replace("0,", "")
                if empty_image_break_row != "0":
                    lt.error_and_raise(
                        RuntimeError,
                        "Error in PixelData.parse_images(): "
                        + f"expected the image break row {row_idx} to end with \"0,...,0\", "
                        + f"but it is instead \"{image_break_row}\"",
                    )

                ret.append(image)

        return ret

    def subtract_from_raw(self, image_name_ext: str, image: np.ndarray) -> npt.NDArray[np.int32]:
        """Subtracts the given image from the matching image_name_ext in the
        "Raw Images" directory and returns the result."""
        raw_dir = ft.join(ft.path_components(self.csv_path_name_ext)[0], "Raw Images")
        raw_img = Image.open(ft.join(raw_dir, image_name_ext))
        raw_image = np.array(raw_img)
        return raw_image.astype(np.int32) - image.astype(np.int32)

    def convert_file(self, delete_after_conversion=False):
        """Converts the CSV file to .png image files, and possibly deletes the CSV file once complete."""
        max_allowed_raw_images_diff = 255

        # print simple stats
        csv_path, csv_name, csv_ext = ft.path_components(self.csv_path_name_ext)
        # lt.info(f"{csv_name}: {self.nrows=}, {self.ncols=}, {self.nimages=}, {self.width=}, {self.height=}")
        lt.info(f"Saving {self.nimages} images for {self.csv_path_name_ext}...", end="")

        # parse and save all images from the file
        out_dir = ft.join(csv_path, "PixelData Images")
        ft.create_directories_if_necessary(out_dir)
        image_names = self.raw_image_names()
        for image_name_ext, image in zip(image_names, self.parse_images()):
            # verify the image isn't too different from it's "raw" counterpart
            comparison: np.ndarray = self.subtract_from_raw(image_name_ext, image)
            diff = np.max(np.abs(comparison))
            if diff > max_allowed_raw_images_diff:
                lt.error_and_raise(
                    RuntimeError,
                    "Error in convert_pixel_data_files.py: "
                    + f"{max_allowed_raw_images_diff=} but the maximum diff for {image_name_ext} is {diff}",
                )

            # save the image
            _, image_name, image_ext = ft.path_components(image_name_ext)
            img = Image.fromarray(image)
            png_path_name_ext = ft.join(out_dir, image_name + ".png")
            if ft.file_exists(png_path_name_ext):
                ft.delete_file(png_path_name_ext)
            img.save(png_path_name_ext)

        # compare the before/after size
        before_size = ft.file_size(self.csv_path_name_ext)
        before_size += 4096  # don't know size on disk, assume one extra block
        after_size = 0
        for png_name_ext in ft.files_in_directory(out_dir, files_only=True):
            after_size += ft.file_size(ft.join(out_dir, png_name_ext))
            after_size += 4096  # don't know size on disk, assume one extra block
        lt.info(f" [{int(np.round(after_size / before_size * 100)):02d}%]")

        # delete the original file to save space on disk
        ft.delete_file(self.csv_path_name_ext)


def find_pixel_data_files(search_dir: str):
    ret: list[str] = []
    pd_pattern = re.compile(r"^[0-9]{8}_[0-9]{6}_PixelData.csv$")

    for fn in ft.files_in_directory(search_dir):
        file_path_name_ext = ft.join(search_dir, fn)
        if os.path.isdir(file_path_name_ext):
            ret += find_pixel_data_files(file_path_name_ext)
        elif os.path.isfile(file_path_name_ext):
            if pd_pattern.match(fn) is not None:
                ret.append(file_path_name_ext)

    return ret


if __name__ == "__main__":
    search_dir = ft.join(opencsp_settings["opencsp_root_path"]["collaborative_dir"], "NSTTF_Optics/Experiments")
    files = find_pixel_data_files(search_dir)

    # Comment out the following two lines to enable this script.
    lt.error(
        f"This will convert and delete all pixel data files in {search_dir}. "
        + f"If you want to continue, please comment out this line {__file__}{sys._getframe(1).f_lineno} and the next."
    )
    sys.exit(1)

    for csv_path_name_ext in files:
        pd = NSTTFPixelDataConverter(csv_path_name_ext)
        pd.convert_file(delete_after_conversion=True)
