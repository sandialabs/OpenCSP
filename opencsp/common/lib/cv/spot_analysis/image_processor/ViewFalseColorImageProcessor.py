import copy
import dataclasses
from typing import Callable

import cv2
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
import opencsp.common.lib.cv.image_reshapers as reshapers
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class ViewFalseColorImageProcessor(AbstractVisualizationImageProcessor):
    """
    Image processor to produce color gradient images from grayscale
    images, for better contrast and legibility by humans.
    """

    def __init__(
        self,
        map_type='human',
        opencv_map=cv2.COLORMAP_JET,
        interactive: bool | Callable[[SpotAnalysisOperable], bool] = False,
        base_image_selector: str | ImageType = None,
    ):
        """
        Parameters
        ----------
        map_type : str, optional
            This determines the number of visible colors. Options are 'opencv'
            (256), 'human' (893), 'large' (1530). Large has the most possible
            colors. Human reduces the number of greens and reds, since those are
            difficult to discern. Default is 'human'.
        opencv_map : opencv map type, optional
            Which color pallete to use with the OpenCV color mapper. Default is
            cv2.COLORMAP_JET.
        """
        super().__init__(interactive, base_image_selector)

        self.map_type = map_type
        self.opencv_map = opencv_map

        self.axis_control = rca.image(grid=False)
        self.view_spec = vs.view_spec_im()
        self.figure: rcfr.RenderControlFigureRecord = None

    @property
    def num_figures(self) -> int:
        return 1

    def init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        self.figure = fm.setup_figure(
            render_control_fig,
            self.axis_control,
            self.view_spec,
            equal=False,
            title=f"{self.name}",
            code_tag=f"{__file__}.init_figure_records",
        )
        return [self.figure]

    def apply_mapping_jet_custom(self, operable: SpotAnalysisOperable, image: CacheableImage) -> CacheableImage:
        """
        Updates the primary image with a false color map ('human' or
        'large'). This has a much larger range of colors that get applied but is
        also much slower than the OpenCV version.

        See also :py:meth:`image_reshapers.false_color_reshaper`

        Parameters
        ----------
        operable: SpotAnalysisOperable
            The operable that the given image came from.
        image : CacheableImage
            The image to apply the false color to.

        Returns
        -------
        image
            A new image with RGB color channels and the input grayscale values
            mapped to the jet color scheme.
        """
        max_value = operable.max_popf
        from_image = image.nparray

        # apply the mapping
        false_color_image = reshapers.false_color_reshaper(from_image, max_value, map_type=self.map_type)

        return CacheableImage.from_single_source(false_color_image)

    def apply_mapping_jet(self, operable: SpotAnalysisOperable, image: CacheableImage) -> CacheableImage:
        """
        Updates the primary image with a false color map. Opencv maps can
        represent 256 different grayscale colors and only takes ~0.007s for a
        1626 x 1236 pixel image.

        Parameters
        ----------
        operable: SpotAnalysisOperable
            The operable that the given image came from.
        image : CacheableImage
            The image to apply the false color to.

        Returns
        -------
        image
            A new image with RGB color channels and the input grayscale values
            mapped to the jet color scheme.
        """
        # rescale to the number of representable colors
        representable_colors = 256
        max_value = operable.max_popf
        new_image: np.ndarray = image.nparray * ((representable_colors - 1) / max_value)
        new_image = np.clip(new_image, 0, representable_colors - 1)
        new_image = new_image.astype(np.uint8)

        # apply the mapping
        false_color_image = cv2.applyColorMap(new_image, self.opencv_map)

        return CacheableImage.from_single_source(false_color_image)

    def visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool, base_image: CacheableImage
    ) -> list[CacheableImage | rcfr.RenderControlFigureRecord]:
        # verify that this is a grayscale image
        (height, width), nchannels = it.dims_and_nchannels(base_image.nparray)
        if nchannels > 1:
            lt.error_and_raise(
                ValueError,
                f"Error in {self.name}.visualize_operable(): "
                + f"image should be in grayscale, but {nchannels} color channels were found ({base_image.shape=})!",
            )

        # apply the false color mapping
        if self.map_type == 'large' or self.map_type == 'human':
            ret = [self.apply_mapping_jet_custom(operable, base_image)]
        else:
            ret = [self.apply_mapping_jet(operable, base_image)]

        return ret

    def close_figures(self):
        if self.figure is not None:
            self.figure.close()
            self.figure = None
