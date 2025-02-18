"""
A collection of functions that operate on images to facilitate computer vision.
Each filter accepts at least one image, and returns at least one image, where
the input and output image have the same shape and data type. In addition,
filters may accept or return additional argument or values.

For reshapers that filter images without any guarantee of maintaining shape or
type, see image_reshapers.py.
"""

import numpy as np
import scipy.ndimage

import opencsp.common.lib.tool.log_tools as lt


def percentile_filter(image: np.ndarray, percentile: int = 50, filter_shape: int | tuple = 3) -> np.ndarray:
    """
    Finds the hotest pixel at the given percentile within a neighborhood of the
    given filter_shape. Can also be used as a minimum, median, or maximum filter by
    setting percentile to 0, 50, or 100, accordingly.

    This is the same as a sliding window function for capturing the
    percentile pixel within the window. For example::

        a = np.array([ [1, 2, 3, 2, 1],
                       [2, 3, 4, 3, 2],
                       [3, 4, 5, 4, 3],
                       [2, 3, 4, 3, 2],
                       [1, 2, 3, 2, 1] ])

        results = percentile_filter(a, percentile=0)
        # result:
        # array([ [1, 1, 2, 1, 1],
        #         [1, 1, 2, 1, 1],
        #         [2, 2, 3, 2, 2],
        #         [1, 1, 2, 1, 1],
        #         [1, 1, 2, 1, 1] ])

        results = percentile_filter(a, percentile=100)
        # result:
        # array([ [3, 4, 4, 4, 3],
        #         [4, 5, 5, 5, 4],
        #         [4, 5, 5, 5, 4],
        #         [4, 5, 5, 5, 4],
        #         [3, 4, 4, 4, 3] ])

    Parameters
    ----------
    image : np.ndarray
        The image to be filtered
    percentile : int, optional
        The percentile pixel of the neighborhood to capture, by default 50
    filter_shape : int | tuple, optional
        The width and height (and depth etc) of the neighborhood to apply.

    Returns
    -------
    filtered_image: np.ndarray
        The resulting image with the same shape and type as the input image.
    """
    # validate the input
    if percentile < -100 or percentile > 100:
        lt.error_and_raise(
            ValueError,
            "Error in image_filters.percentile_filter(): "
            + f"percentile should be between 0 and 100 but is {percentile} "
            + "(negative values between -100 and 0 will be wrapped to be within the 0-100 range).",
        )
    if isinstance(filter_shape, tuple):
        if len(filter_shape) != image.ndim:
            image = image.squeeze()  # maybe we just have extra 0-length or 1-length dimensions
        if len(filter_shape) != image.ndim:
            lt.error_and_raise(
                ValueError,
                "Error in image_filters.percentile_filter(): "
                + f"window shape should have the same number of dimensions as the image, but {filter_shape=} and {image.ndim=}",
            )
    else:
        if filter_shape < 0 or filter_shape % 2 == 0:
            lt.error_and_raise(
                ValueError,
                "Error in image_filters.percentile_filter(): "
                + f"window size should be positive and odd, but is {filter_shape}",
            )

    # normalize input
    if percentile < 0:
        percentile = 100 + percentile

    # apply the filter
    if percentile == 0:
        filtered_image = scipy.ndimage.minimum_filter(image, size=filter_shape, mode="nearest")
    elif percentile == 100:
        filtered_image = scipy.ndimage.maximum_filter(image, size=filter_shape, mode="nearest")
    else:
        filtered_image = scipy.ndimage.percentile_filter(
            image, percentile=percentile, size=filter_shape, mode="nearest"
        )

    return filtered_image
