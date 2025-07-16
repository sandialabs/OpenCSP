import dataclasses

import cv2
import numpy as np

import opencsp.common.lib.cv.image_reshapers as reshapers
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class FalseColorImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Image processor to produce color gradient images from grayscale
    images, for better contrast and legibility by humans.
    """

    def __init__(self, map_type="human", opencv_map=cv2.COLORMAP_JET):
        """
        Parameters
        ----------
        map_type : str, optional
            This determines the number of visible colors. Options are 'opencv'
            (256), 'human' (1020), 'large' (1530). Large has the most possible
            colors. Human reduces the number of greens and reds, since those are
            difficult to discern. Default is 'human'.
        opencv_map : opencv map type, optional
            Which color pallete to use with the OpenCV color mapper. Default is
            cv2.COLORMAP_JET.
        """
        super().__init__(self.__class__.__name__)
        self.map_type = map_type
        self.opencv_map = opencv_map

    def apply_mapping_jet_custom(self, operable: SpotAnalysisOperable) -> SpotAnalysisOperable:
        """
        Updates the primary image with a false color map ('human' or
        'large'). This has a much larger range of colors that get applied but is
        also much slower than the OpenCV version.

        See also :py:meth:`image_reshapers.false_color_reshaper`

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable whose primary image to apply the false color to.

        Returns
        -------
        SpotAnalysisOperable
            A copy of the input operable with the primary image replaced with
            the false color image.
        """
        max_value = operable.max_popf
        from_image = operable.primary_image.nparray
        ret = reshapers.false_color_reshaper(from_image, max_value, map_type=self.map_type)
        return dataclasses.replace(operable, primary_image=ret)

    @staticmethod
    def apply_mapping_jet(operable: SpotAnalysisOperable, opencv_map) -> SpotAnalysisOperable:
        """
        Updates the primary image with a false color map. Opencv maps can
        represent 256 different grayscale colors and only takes ~0.007s for a
        1626 x 1236 pixel image.

        Parameters
        ----------
        operable : SpotAnalysisOperable
            The operable whose primary image to apply the false color to.

        Returns
        -------
        SpotAnalysisOperable
            A copy of the input operable with the primary image replaced with
            the false color image.
        """
        # rescale to the number of representable colors
        representable_colors = 256
        max_value = operable.max_popf
        new_image: np.ndarray = operable.primary_image.nparray * ((representable_colors - 1) / max_value)
        new_image = np.clip(new_image, 0, representable_colors - 1)
        new_image = new_image.astype(np.uint8)

        # apply the mapping
        ret = cv2.applyColorMap(new_image, opencv_map)

        return dataclasses.replace(operable, primary_image=ret)

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray

        # verify that this is a grayscale image
        dims, nchannels = it.dims_and_nchannels(image)
        if nchannels > 1:
            lt.error_and_raise(
                ValueError,
                f"Error in {self.name}._execute(): "
                + f"image should be in grayscale, but {nchannels} color channels were found ({image.shape=})!",
            )

        # apply the false color mapping
        if self.map_type == "large" or self.map_type == "human":
            ret = [self.apply_mapping_jet_custom(operable)]
        else:
            ret = [self.apply_mapping_jet(operable, self.opencv_map)]

        return ret
