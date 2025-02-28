import matplotlib.pyplot as plt
import numpy as np
import os

import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.unit_conversion as uc


def construct_target_image(image_width, image_height, dpm):  # Meters  # Meters  # Dots per meter
    """
    Construct a target image with specified dimensions and dots per meter.

    Parameters
    ----------
    image_width : float
        The width of the image in meters.
    image_height : float
        The height of the image in meters.
    dpm : float
        The number of dots per meter (resolution) for the image.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array representing the constructed image, initialized to zeros (black image),
        with shape (image_rows, image_cols, 3) where 3 corresponds to the RGB color channels.
    """
    image_cols = round(image_width * dpm)
    image_rows = round(image_height * dpm)
    img = np.uint8(
        np.zeros([image_rows, image_cols, 3])
    )  # ?? SCAFFOLDING RCB -- IS THIS THE CORRECT PLACE TO DO THIS?  WILL WE LOSE IMAGE PRECISION INTERMEDIATE CALCULATIONS?
    # print('In construct_target_image(), image shape =', img.shape)  # ?? SCAFFOLDING RCB -- TEMPORARY
    return img


def rows_cols(img):
    """
    Get the number of rows and columns in an image.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D numpy array representing the image, expected to have shape (n_rows, n_cols, n_bands).

    Returns
    -------
    tuple
        A tuple containing the number of rows (int) and the number of columns (int) in the image.

    Raises
    ------
    AssertionError
        If the number of bands in the image is not equal to 3.
    """
    n_rows = img.shape[0]
    n_cols = img.shape[1]
    n_bands = img.shape[2]
    if n_bands != 3:
        print("ERROR: Number of input image bands is not 3.")
        assert False
    return n_rows, n_cols


def save_image(img, output_dpm, output_dir, output_file_body, output_ext):
    """
    Save an image to a specified directory with a given resolution.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D numpy array representing the image to be saved, expected to have shape (n_rows, n_cols, 3).
    output_dpm : float
        The desired output resolution in dots per meter.
    output_dir : str
        The directory where the image will be saved.
    output_file_body : str
        The base name for the output file (without extension).
    output_ext : str
        The file extension for the output image (e.g., '.png', '.jpg').

    Returns
    -------
    str
        The full path to the saved image file.

    Notes
    -----
    This function creates the necessary directories if they do not exist and saves the image using
    the specified resolution converted to dots per inch (DPI).
    """
    ft.create_directories_if_necessary(output_dir)
    output_file_dir_body_ext = os.path.join(output_dir, output_file_body + output_ext)
    print("Saving file:", output_file_dir_body_ext)
    output_dpi = round(uc.dpm_to_dpi(output_dpm))
    plt.imsave(output_file_dir_body_ext, img, dpi=output_dpi)
    print("Done.")
    return output_file_dir_body_ext
