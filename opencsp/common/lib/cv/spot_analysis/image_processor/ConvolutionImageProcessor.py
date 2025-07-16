import dataclasses
from typing import Callable

import numpy as np
import numpy.typing as npt
import scipy.ndimage
import scipy.signal

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.log_tools as lt


class ConvolutionImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Convolves an image by the given kernel

    Example use cases include reducing the effects of noise, and finding the average value for a larger area.
    """

    def __init__(self, kernel="gaussian", diameter=3):
        """
        Parameters
        ----------
        kernel : str, optional
            The type of kernel to apply. Options are "gaussian" or "box". By default "gaussian".
        diameter : int, optional
            The size of the kernel to be applied, by default 3
        """
        super().__init__()

        # validate inputs
        if kernel not in self._kernels:
            lt.error_and_raise(
                ValueError,
                "Error in ConvolutionImageProcessor(): "
                + f"the kernel for convolution must be one of {list(self._kernels.keys())}, but is instead '{kernel}'",
            )
        if diameter < 1:
            lt.error_and_raise(
                ValueError, "Error in ConvolutionImageProcessor(): " + f"the diameter must be >= 1, but is {diameter}"
            )
        if diameter % 2 == 0:
            lt.error_and_raise(
                ValueError, "Error in ConvolutionImageProcessor(): " + f"diameter must be odd, but is {diameter}"
            )

        # register parameters
        self.kernel_name = kernel
        self.diameter = diameter

        # internal values
        self.kernel = lambda img: self._kernels[self.kernel_name](img)
        self.radius = diameter / 2
        self.iradius = int(self.radius)

    @property
    def _kernels(self) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
        return {"box": self._box_filter, "gaussian": self._gaussian_filter}

    def _box_filter(self, image: npt.NDArray[np.int_]):
        """
        Convolve the image with a simple box filter, where all pixels in a neighborhood are weighted equaly.

        For example, with a diameter of 3, the image will be convolved with the following array::

            kernel = np.array([[1/9, 1/9, 1/9],
                               [1/9, 1/9, 1/9],
                               [1/9, 1/9, 1/9]])
        """
        orig_type = image.dtype
        image = image.astype(np.float64)

        # evaluate the filter
        mode = "same"  # shape is max(image, kernel)
        boundary = "symm"  # edges are reflected, ie image[-1] = image[0], image[-2] = image[1], etc...
        kernel = np.ones((self.diameter, self.diameter)) / (self.diameter**2)
        ret = scipy.signal.convolve2d(image, kernel, mode, boundary)

        ret = np.round(ret)
        ret = ret.astype(orig_type)
        return ret

    def _gaussian_filter(self, image: npt.NDArray[np.int_]):
        """
        Convolves the image with a gaussian filter with sigma 1.
        """
        orig_type = image.dtype
        image = image.astype(np.float64)

        # evaluate the filter
        ret = scipy.ndimage.gaussian_filter(image, sigma=1, radius=self.iradius)

        ret = np.round(ret)
        ret = ret.astype(orig_type)
        return ret

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # validate input
        input_image = operable.primary_image.nparray
        if (self.diameter > input_image.shape[0]) or (self.diameter > input_image.shape[1]):
            lt.error_and_raise(
                RuntimeError,
                "Error in ConvolutionImageProcessor._box_filter(): "
                + "although scipy.signal.convolve2d supports convolutions with a kernel that is larger than "
                + "the array being convolved, we don't currently support that use case. Consider adding that functionality.",
            )

        # evaluate the kernel
        filtered_image = self.kernel(input_image)

        # create the returned operable
        cacheable = CacheableImage(filtered_image)
        new_operable = dataclasses.replace(operable, primary_image=cacheable)

        return [new_operable]
