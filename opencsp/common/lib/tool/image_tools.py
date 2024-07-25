"""
Utilities for image processing.



"""

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
# fmt: on


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
        if rescale_or_clip == 'rescale':
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
    """Get the (x,y) dimensions of the image, and the number of color channels.

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

    xdim = shape[0]
    ydim = shape[1]
    return (xdim, ydim), nchannels


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


def image_files_in_directory(dir: str, allowable_extensions: list[str] = None) -> list[str]:
    """
    Get a list of all image files in the given directory, as determined by the file extension.

    Parameters
    ----------
    dir : str
        The directory to get files from.
    allowable_extensions : list[str], optional
        The allowed extensions, such as ["png"]. By default pil_image_formats_rw.

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
    files_per_ext = ft.files_in_directory_by_extension(dir, allowable_extensions)

    # condense into a single list
    files: list[str] = []
    for ext in files_per_ext:
        files += files_per_ext[ext]

    # sort
    files = sorted(files)

    return files
