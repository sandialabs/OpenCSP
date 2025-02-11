import dataclasses
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.fiducials.PointFiducials import PointFiducials
import opencsp.common.lib.cv.image_reshapers as ir
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class ViewHighlightImageProcessor(AbstractVisualizationImageProcessor):
    """
    Shows the white and/or black parts of the image in a different color, in
    order to highlight them.
    """

    def __init__(
        self,
        black_highlight_color: color.Color | str | tuple = None,
        white_highlight_color: color.Color | str | tuple = None,
        thresh=0,
        interactive: bool | Callable[[SpotAnalysisOperable], bool] = False,
        base_image_selector: str | ImageType = None,
    ):
        """
        Parameters
        ----------
        black_highlight_color: color.Color | str | tuple, optional
            The color to replace all black pixels with, or 'none' to not
            highlight black pixels. Default is magenta.
        white_highlight_color: color.Color | str | tuple, optional
            The color to replace all white pixels with, or 'none' to not
            highlight white pixels. Default is cyan.
        thresh: int, optional
            How close the color has to be to white or black to be replaced. 0
            means it must be exactly white or exactly black. Default is 0.
        """
        super().__init__(interactive, base_image_selector)

        # normalize defaults
        if black_highlight_color is None:
            black_highlight_color = color.magenta()
        if white_highlight_color is None:
            white_highlight_color = color.cyan()

        self.black_highlight_color = black_highlight_color
        self.white_highlight_color = white_highlight_color
        self.thresh = thresh

        self.axis_control = rca.image(grid=False)
        self.view_spec = vs.view_spec_im()
        self.figure: rcfr.RenderControlFigureRecord

    @property
    def num_figures(self) -> int:
        return 1

    def init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        self.figure = fm.setup_figure(
            render_control_fig,
            self.axis_control,
            self.view_spec,
            equal=True,
            title=f"{self.name}",
            code_tag=f"{__file__}.init_figure_records()",
        )
        return [self.figure]

    def visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool, base_image: CacheableImage
    ) -> list[CacheableImage | rcfr.RenderControlFigureRecord]:
        old_image = ir.nchannels_reshaper(base_image.nparray, 3)
        new_image = np.copy(old_image)

        # replace black
        if isinstance(self.black_highlight_color, str) and self.black_highlight_color.lower() == 'none':
            pass
        else:
            highlight_color = color.Color.from_generic(self.black_highlight_color)
            black_selector = (
                (old_image[:, :, 0] <= self.thresh)
                & (old_image[:, :, 1] <= self.thresh)
                & (old_image[:, :, 2] <= self.thresh)
            )
            new_image[black_selector] = highlight_color.rgb_255()

        # replace white
        if isinstance(self.white_highlight_color, str) and self.white_highlight_color.lower() == 'none':
            pass
        else:
            highlight_color = color.Color.from_generic(self.white_highlight_color)
            white_selector = (
                (old_image[:, :, 0] >= 255 - self.thresh)
                & (old_image[:, :, 1] >= 255 - self.thresh)
                & (old_image[:, :, 2] >= 255 - self.thresh)
            )
            new_image[white_selector] = highlight_color.rgb_255()

        # show the visualization
        self.figure.clear()
        self.figure.view.imshow(new_image)
        self.figure.view.show(block=False)

        # build the return value
        cacheable_image = CacheableImage(new_image)

        return [cacheable_image]

    def close_figures(self):
        if self.figure is not None:
            self.figure.close()
            self.figure = None
