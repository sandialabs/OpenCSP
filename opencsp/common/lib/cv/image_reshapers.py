"""
A collection of functions that operate on images to facilitate computer vision.
Each function accepts at least one image, and returns at least one image, but
makes no guarantees that the output image has either the same shape or data
type.

For filters that don't modify the image shape or data type, see
image_filters.py.
"""

import cv2 as cv
import numpy as np
from typing import Literal

import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


def _map_jet_large_rgb(input_color: int):
    """Like the opencv jet false color map, except that this covers a
    larger color range.

    Parameters
    ----------
    input_color : int
        A grayscale color in the range of 0-1530.

    Returns
    -------
    rgb: int
        An rgb color, where the bits 23:16 are red, bits 15:8 are green, and
        bits 7:0 are blue (assuming bit 31 is the most significant bit).
    """
    if input_color <= 255:  # black to blue
        ret = [0, 0, input_color]
    elif input_color <= 255 * 2:  # blue to cyan
        ret = [0, input_color - 255, 255]
    elif input_color <= 255 * 3:  # cyan to green
        ret = [0, 255, 255 * 3 - input_color]
    elif input_color <= 255 * 4:  # green to yellow
        ret = [input_color - 255 * 3, 255, 0]
    elif input_color <= 255 * 5:  # yellow to red
        ret = [255, 255 * 5 - input_color, 0]
    else:  # red to white
        ret = [255, input_color - 255 * 5, input_color - 255 * 5]
    return (ret[0] << 16) + (ret[1] << 8) + ret[2]


def _map_jet_human_rgb(input_color: int):
    """Like _map_jet_large_rgb, but with more limited red and green values.

    Parameters
    ----------
    input_color : int
        A grayscale color in the range of 0-893.

    Returns
    -------
    rgb: int
        An rgb color, where the bits 23:16 are red, bits 15:8 are green, and
        bits 7:0 are blue (assuming bit 31 is the most significant bit).
    """
    if input_color <= 128:  # black to blue
        ret = [0, 0, input_color * 2]
    elif input_color <= 255 + 128:  # blue to cyan
        ret = [0, input_color - 128, 255]
    elif input_color <= 255 * 2:  # cyan to green
        ret = [0, 255, 2 * (255 * 2 - input_color)]
    elif input_color <= 255 * 2 + 127:  # green to yellow
        ret = [2 * (input_color - (255 * 2)), 255, 0]
    elif input_color <= 255 * 3:  # yellow to red
        ret = [255, 2 * ((255 * 3) - input_color), 0]
    else:  # red to white
        ret = [255, 2 * (input_color - (255 * 3)), 2 * (input_color - (255 * 3))]
    return (ret[0] << 16) + (ret[1] << 8) + ret[2]


def nchannels_reshaper(from_image: np.ndarray, new_nchannels: int):
    """
    Reshape the number of channels in an image.

    This function takes an input image and reshapes it to have a specified
    number of channels. It supports conversions between single-channel (grayscale),
    three-channel (RGB), and four-channel (RGBA) images. The function handles
    broadcasting for single-channel images and converts RGBA images to RGB or
    grayscale as needed.

    Parameters
    ----------
    from_image : np.ndarray
        The input image array. It can be a 2D array (height, width) for
        single-channel images or a 3D array (height, width, channels) for
        multi-channel images.

    new_nchannels : int
        The desired number of channels for the output image. Must be one of
        the following: 1 (grayscale), 3 (RGB), or 4 (RGBA).

    Returns
    -------
    np.ndarray
        A new image array with the specified number of channels. The output
        will have the shape (height, width, new_nchannels).

    Raises
    ------
    ValueError
        If the conversion from the old number of channels to the new number
        of channels is not supported.

    Notes
    -----
    - If the input image has fewer than 3 dimensions, it will be expanded
      to have a third dimension.
    - If the input image is in RGBA format and the new number of channels
      is set to 1, the image will be converted to grayscale.
    - If the input image is in RGBA format and the new number of channels
      is set to 3, the alpha channel will be discarded.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(100, 100, 4)  # Example RGBA image
    >>> reshaped_img = nchannels_reshaper(img, 3)  # Convert to RGB
    >>> reshaped_img.shape
    (100, 100, 3)

    >>> gray_img = nchannels_reshaper(img, 1)  # Convert to grayscale
    >>> gray_img.shape
    (100, 100, 1)
    """
    # ChatGPT 4o-mini assisted with generating this docstring
    (height, width), old_nchannels = it.dims_and_nchannels(from_image)
    new_image = from_image

    if new_image.ndim < 3:
        new_image = np.expand_dims(new_image, axis=2)

    if old_nchannels != new_nchannels:
        if old_nchannels == 1:
            new_image = np.broadcast_to(new_image, (height, width, new_nchannels))
        elif old_nchannels == 4:
            # assumed old image is RGBA
            new_image = new_image[:, :, :3]
            if new_nchannels == 1:
                new_image = cv.cvtColor(new_image, cv.COLOR_RGB2GRAY)
            elif new_nchannels == 3:
                pass
            else:
                lt.error_and_raise(
                    ValueError,
                    "Error in image_reshapers.nchannels_reshaper(): "
                    + f"unknown conversion from {old_nchannels=} to {new_nchannels=}",
                )

    new_image = np.copy(new_image)
    return new_image


def false_color_reshaper(
    from_image: np.ndarray,
    input_max_value: int = 255,
    map_name: str = 'jet',
    map_type: Literal['large'] | Literal['human'] = 'human',
) -> np.ndarray:
    """Updates the primary image to use the jet color map plus black and
    white (black->blue->cyan->green->yellow->red->white).

    This larger version of the opencv color map can represent either 893 or
    1530 different grayscale colors (compared to 256 colors with
    opencv.applyColorMap()). However, this takes ~0.28 seconds for a 1626 x 1236
    pixel image.

    Parameters
    ----------
    from_image: np.ndarray
        The image to convert to false color. Should be grayscale (aka from_image.ndim == 2).
    input_max_value: int
        The value to map to the maximum of the color range.
    map_name: str, optional
        The color map to use. Currently just 'jet' is implemented.
    map_type: str, optional
        Whether to use the full map space 'large' or a smaller map space that is easier to distinguish 'human'.

    Returns
    -------
    to_image: np.ndarray
        An image with the same width and height as the input image, but with a
        third dimension for color. The dtype will be from_image.dtype.
    """
    # rescale to the number of representable colors
    # black_to_blue = 128/255
    # blue_to_cyan = 255
    # cyan_to_green = 128/255
    # green_to_yellow = 127/255
    # yellow_to_red = 128/255
    # red_to_white = 127/255
    representable_colors = 255 * 6 if map_type == 'large' else 893
    new_image: np.ndarray = from_image * ((representable_colors - 1) / input_max_value)
    new_image = np.clip(new_image, 0, representable_colors - 1).astype(np.int32)
    if len(new_image.shape) == 3:
        new_image = np.squeeze(new_image, axis=2)

    # add extra color channels, as necessary
    color_image = nchannels_reshaper(new_image, 3).astype(np.uint32)
    assert it.dims_and_nchannels(color_image)[1] == 3

    # apply the mapping
    map_func = _map_jet_large_rgb if map_type == 'large' else _map_jet_human_rgb
    mapping = {k: map_func(k) for k in range(representable_colors)}
    new_image = np.vectorize(mapping.__getitem__)(new_image)
    color_image[:, :, 0] = new_image >> 16
    color_image[:, :, 1] = (new_image >> 8) & 255
    color_image[:, :, 2] = new_image & 255
    ret = color_image.astype(from_image.dtype)

    # Other methods I've tried:
    # new_image = np.vectorize(_map_jet_large_rgb)(new_image)
    #     ~1.61 s/image
    # np.apply_along_axis(_map_jet_large, axis=2, arr=color_image)
    #     ~14.09 s/image
    # color_image[np.where(new_image==k)] = _map_jet_large(k)
    #     ~9.18 s/image

    return ret
