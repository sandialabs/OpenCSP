"""
Utilities for image processing.



"""

import sys
from typing import Callable, TypeVar

import exiftool
import numpy as np
from PIL import Image

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

# disable auto formatting
# fmt: off
# from https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
pil_image_formats_rw = ["blp", "bmp", "dds", "dib", "eps", "gif", "icns", "ico", "im", "jpg", "jpeg",
                        "msp", "pcx", "png", "apng", "pbm", "pgm", "ppm", "pnm", "sgi", "spider", "tga", "tiff", "webp", "xbm"]
""" A list of all image image formats that can be read and written by the Python Imaging Library (PIL) """
pil_image_formats_readable = pil_image_formats_rw + ["cur", "dcx", "fits", "fli", "flc", "fpx", "ftex", "gbr",
                                                     "gd", "imt", "iptc", "naa", "mcidas", "mic", "mpo", "pcd", "pixar", "psd", "sun", "wal", "wmf", "emf", "xpm"]
""" A list of all image image formats that can be read by the Python Imaging Library (PIL). Note that not all of these formats can be written by PIL. """
pil_image_formats_writable = pil_image_formats_rw + ["palm", "pdf", "xv"]
""" A list of all image image formats that can be written by the Python Imaging Library (PIL). Note that not all of these formats can be read by PIL. """
pil_image_formats_supporting_exif = ["jpg", "jpeg", "png", "tiff", "webp"]
# fmt: on


T = TypeVar('T')


def numpy_to_image(arr: np.ndarray, rescale_or_clip='rescale', rescale_max=-1):
    """Convert the numpy representation of an image to a Pillow Image.

    Coverts the given arr to an Image. The array is converted to an integer
    type, as necessary. The color information is then rescaled/clipd to fit
    within an 8-bit color depth.

    In theory, images can be saved with higher bit-depth information using
    opencv imwrite('12bitimage.png', arr), but I (BGB) haven't tried very hard
    and haven't had any luck getting this to work.

    Parameters
    ----------
    arr : np.ndarray
        The array to be converted.
    rescale_or_clip : str, optional
        Whether to rescale the value in the array to fit within 0-255, or to
        clip the values so that anything over 255 is set to 255. By default
        'rescale'.
    rescale_max : int, optional
        The maximum value expected in the input arr, which will be set to 255.
        When less than 0, the maximum of the input array is used. Only
        applicable when rescale_or_clip='rescale'. By default -1.

    Returns
    -------
    image: PIL.Image
        The image representation of the input array.
    """
    allowed_int_types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]

    # get the current integer size, and convert to integer type
    if not np.issubdtype(arr.dtype, np.integer):
        maxval = np.max(arr)
        for int_type in allowed_int_types:
            if np.iinfo(int_type).max >= maxval:
                break
        arr = arr.astype(int_type)
    else:
        int_type = arr.dtype

    # rescale down to 8-bit if bitdepth is too large
    if np.iinfo(int_type).max > 255:
        if rescale_or_clip == "rescale":
            if rescale_max < 0:
                rescale_max = np.max(arr)
            scale = 255 / rescale_max
            arr = arr * scale
            arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255)
            arr = arr.astype(np.uint8)

    img = Image.fromarray(arr)
    return img


def images_are_identical(image_1: np.ndarray, image_2: np.ndarray, tolerance_pixel: int):
    """Checks if two images are identical.

    Args:
        - image_1 (matrix): The first image to compare, such as from cv.imread().
        - image_2 (matrix): The second image to compare, such as from cv.imread().
        - tolerance_pixel (int): How many pixels are allowed to be different and the images are still considered identical.

    Returns:
        - bool: True if identical, False otherwise
    """
    if image_1.shape != image_2.shape:
        # Different shapes.
        return False
    else:
        # Same shape.
        abs_diff = abs(image_1 - image_2)
        if abs_diff.max() <= tolerance_pixel:
            return True
        else:
            return False


def dims_and_nchannels(image: np.ndarray):
    """Get the (y,x) dimensions of the image, and the number of color channels.

    Raises:
    -------
    ValueError
        If the given array doesn't have the correct number of dimensions (2 or 3)."""
    shape = image.shape

    nchannels = 1
    if len(shape) == 2:
        nchannels = 1
    elif len(shape) == 3:
        nchannels = shape[2]
    else:
        lt.error_and_raise(
            ValueError,
            f"Error in image_tools.dims_and_nchannels(): expected image to have 2-3 dimensions (x, y, and color), but {shape=}!",
        )

    ydim = shape[0]
    xdim = shape[1]
    return (ydim, xdim), nchannels


def min_max_colors(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the minimum and maximum values for each of the color channels in the given image.

    Parameters
    ----------
    image : np.ndarray
        The image to get the minimum and maximum color values for. Must have 2-3 dimensions.

    Returns
    -------
    min_colors: ndarray
        The minimum values for each color channel. If there is only 1 color
        channel, then this array will have shape=(1).
    max_colors: ndarray
        The maximum values for each color channel. If there is only 1 color
        channel, then this array will have shape=(1).
    """
    _, nchannels = dims_and_nchannels(image)
    if nchannels == 1:
        min_color: np.ndarray = np.array([np.min(image)])
        max_color: np.ndarray = np.array([np.max(image)])
        if max_color.ndim != 1 or max_color.shape[0] != 1:
            lt.error_and_raise(
                RuntimeError,
                "Programmer error in image_tools.min_max(): "
                + f"Returned value should have shape=(1), but is instead {max_color.shape=}",
            )
        return min_color, max_color
    else:
        if image.ndim != 3:
            lt.error_and_raise(
                RuntimeError,
                "Programmer error in image_tools.min_max(): "
                + f"I really expected the image array to have 3 dimensions, but found {image.ndim=}!",
            )
        min_per_row: np.ndarray = np.min(image, axis=1)
        max_per_row: np.ndarray = np.max(image, axis=1)
        min_colors: np.ndarray = np.min(min_per_row, axis=0)
        max_colors: np.ndarray = np.max(max_per_row, axis=0)
        if max_colors.ndim != 1 or max_colors.shape[0] != nchannels or max_colors.shape[0] <= 1:
            lt.error_and_raise(
                RuntimeError,
                "Programmer error in image_tools.min_max(): "
                + f"Returned value should have shape=(3), but is instead {max_colors.shape=}",
            )
        return min_colors, max_colors


def range_for_threshold(image: np.ndarray, threshold: int) -> tuple[int, int, int, int]:
    """
    Get the start (inclusive) and end (exclusive) range for which the given image is >= the given threshold.

    Parameters
    ----------
    image : np.ndarray
        The 2d numpy array to be searched, row major (y axis 0, x axis 1).
    threshold : int
        The cutoff value that the returned region should have pixels greater than.

    Returns
    -------
    start_y, end_y, start_x, end_x: tuple[int, int, int, int]
        The start (inclusive) and end (exclusive) matching range. Returns the full image size if there are no
        matching pixels.
    """
    ret: list[int] = []

    for axis in [0, 1]:
        # If we want the maximum value for all rows, then we need to accumulate across columns.
        # If we want the maximum value for all columns, then we need to accumulate across rows.
        perpendicular_axis = 0 if axis == 1 else 1

        # find matches
        img_matching = np.max(image, perpendicular_axis) >= threshold
        match_idxs = np.argwhere(img_matching)

        # find the range
        if match_idxs.size > 0:
            start, end = match_idxs[0][0], match_idxs[-1][0] + 1
        else:
            start, end = 0, image.shape[axis]

        ret.append(start)
        ret.append(end)

    return tuple(ret)


def image_files_in_directory(dir: str, allowable_extensions: list[str] = None, recursive=False) -> list[str]:
    """
    Get a list of all image files in the given directory, as determined by the file extension.

    Parameters
    ----------
    dir : str
        The directory to get files from.
    allowable_extensions : list[str], optional
        The allowed extensions, such as ["png"]. By default pil_image_formats_rw.
    recursive: bool
        If true, then walk through all files in the given directory and all
        subdirectories. Does not follow symbolic links to directories. Default
        False.

    Returns
    -------
    image_file_names_exts: list[str]
        A list of the name.ext for each image file in the given directory.
    """
    # normalize input
    if allowable_extensions is None:
        allowable_extensions = pil_image_formats_rw
    for i, ext in enumerate(allowable_extensions):
        if ext.startswith("."):
            continue
        else:
            allowable_extensions[i] = "." + ext

    # get all matching files
    files_per_ext = ft.files_in_directory_by_extension(dir, allowable_extensions, recursive=recursive)

    # condense into a single list
    files: list[str] = []
    for ext in files_per_ext:
        files += files_per_ext[ext]

    # sort
    files = sorted(files)

    return files


def getsizeof_approx(img: Image) -> int:
    """
    Get the number of bytes of memory used by the given image.

    Note that this value is approximate. It should be close based on the basic
    assumptions of uncompressed data in memory, and one byte per pixel per color
    channel.

    Parameters
    ----------
    img : Image
        The image to get the size of.

    Returns
    -------
    int
        Number of bytes of memory that the image object + image data together occupy.
    """
    # Get the size of the image object
    object_size = sys.getsizeof(img)

    # Calculate the approximate size of the image data
    if img is None:
        image_data_size = 0
    else:
        width, height = img.size
        mode_size = len(img.getbands())  # Number of bytes per pixel
        image_data_size = width * height * mode_size

    return object_size + image_data_size


def get_exif_value(
    data_dir: str, image_path_name_exts: str | list[str], exif_val: str = "EXIF:ISO", parser: Callable[[str], T] = int
) -> T | list[T]:
    """Returns the exif_val Exif information on the given images, if they have such
    information. If not, then None is returned for those images."""
    # build the list of files
    if isinstance(image_path_name_exts, str):
        files = [ft.join(data_dir, image_path_name_exts)]
    else:
        files = [ft.join(data_dir, image_path_name_ext) for image_path_name_ext in image_path_name_exts]

    # verify the files exist
    for file in files:
        if not ft.file_exists(file):
            lt.error_and_raise(
                FileNotFoundError, "Error in image_tools.get_exif_value: " + f"image file {file} does not exist!"
            )

    # get the exif tags
    with exiftool.ExifToolHelper() as et:
        tags = et.get_tags(files, tags=[exif_val])

    # parse the value
    has_val = [exif_val in image_tags for image_tags in tags]
    parsed_exif_values: list[T | None] = []
    for i in range(len(has_val)):
        if has_val[i]:
            exif_str = tags[i][exif_val]
            try:
                parsed = parser(exif_str)
            except Exception as ex:
                lt.error(
                    "Error in image_tools.get_exif_value: "
                    + f"{type(ex)} encountered while parsing exif value {exif_str} for image {files[i]}"
                )
                raise
            parsed_exif_values.append(parsed)
        else:
            parsed_exif_values.append(None)

    # return a singular value if we were given a singular value
    if isinstance(image_path_name_exts, str):
        return parsed_exif_values[0]
    return parsed_exif_values


def set_exif_value(data_dir: str, image_path_name_ext: str, exif_val: str, exif_name: str = "EXIF:ISO"):
    """
    Set the specified EXIF tag value for an image file.

    This function uses the ExifTool to set the value of a specified EXIF tag
    (defaulting to "EXIF:ISO") for a given image file located in the specified
    directory. It attempts to overwrite the original image file with the new
    EXIF data. If the initial attempt fails due to an execution error, the
    function is designed to handle the error gracefully (though the error
    handling code is currently commented out and requires implementation).

    Parameters:
    -----------
    data_dir : str
        The directory where the image file is located.

    image_path_name_ext : str
        The name of the image file, including its extension.

    exif_val : str
        The value to set for the specified EXIF tag.

    exif_name : str, optional
        The name of the EXIF tag to set (default is "EXIF:ISO").

    Raises:
    -------
    exiftool.exceptions.ExifToolExecuteError
        If the ExifTool encounters an error while attempting to set the EXIF tag.

    Notes:
    ------
    The function currently contains commented-out code for handling poorly
    formatted images by rewriting them using the Pillow library. This
    functionality is intended to be implemented in the future.
    """
    # "ChatGPT 4o" assisted with generating this docstring.
    # TODO is calling set_tags twice intended and necessary?
    with exiftool.ExifToolHelper() as et:
        et.set_tags(
            ft.join(data_dir, image_path_name_ext),
            tags={exif_name: str(exif_val)},
            params=["-P", "-overwrite_original"],
        )

    with exiftool.ExifToolHelper() as et:
        try:
            et.set_tags(
                ft.join(data_dir, image_path_name_ext),
                tags={exif_name: str(exif_val)},
                params=["-P", "-overwrite_original"],
            )
        except exiftool.exceptions.ExifToolExecuteError:

            # # The image may have been poorly formatted the first time around
            # TODO
            # for image_pne in image_path_name_exts:

            #     # Save the image to a new file with a trusted program (Pillow)
            #     p, n, e = ft.path_components(image_pne)
            #     rewrite = ft.join(p, n + " - rewrite" + e)
            #     Image.open(image_pne).save(rewrite)
            #     shutil.copystat(image_pne, rewrite)
            #     ft.delete_file(image_pne)
            #     ft.rename_file(rewrite, image_pne)

            #     # Try to set the gain EXIF information again
            #     et.set_tags(image_pne, tags={"EXIF:ISO": str(gain)}, params=["-P", "-overwrite_original"])
            raise
