import dataclasses
from typing import Callable, Type

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.fiducials.PointFiducials import PointFiducials
from opencsp.common.lib.cv.fiducials.AbstractFiducials import AbstractFiducials
from opencsp.common.lib.cv.spot_analysis.ImageType import ImageType
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class ViewAnnotationsImageProcessor(AbstractVisualizationImageProcessor):
    """
    Draws annotations on top of the input image. The annotations drawn are those in operable.given_fiducials,
    operable.found_fiducials, and operable.annotations.
    """

    def __init__(
        self,
        annotations_filter: list[Type[AbstractFiducials]] = None,
        interactive: bool | Callable[[SpotAnalysisOperable], bool] = False,
        base_image_selector: str | ImageType = None,
    ):
        """
        Parameters
        ----------
        annotations_filter : list[Type[AbstractFiducials]], optional
            If set, then the annotations that get rendered are only those that
            match this filter.
        """
        super().__init__(interactive, base_image_selector)

        self.annotations_filter = annotations_filter

        self.axis_control = rca.image(grid=False)
        self.view_spec = vs.view_spec_im()
        self.figure: rcfr.RenderControlFigureRecord

    @property
    def num_figures(self) -> int:
        """
        How many figure windows this instance intends to create. Must be
        available at all times after this instance has been initialized.
        """
        return 1

    def init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        """
        Initializes the figure windows (via figure_management.setup_figure*) for
        this instance and returns the list of initialized figures. The length of
        this list ideally should match what was previously returned for
        num_figures.

        Parameters
        ----------
        render_control_fig : rcf.RenderControlFigure
            The render controller to use during figure setup.

        Returns
        -------
        figures: list[rcfr.RenderControlFigureRecord]
            The list of newly created figure windows.
        """
        self.figure = fm.setup_figure(
            render_control_fig,
            self.axis_control,
            self.view_spec,
            equal=True,
            title=f"{self.name}",
            code_tag=f"{__file__}.init_figure_records()",
        )
        return [self.figure]

    def _annotations_match_filter(self, annotations: AbstractFiducials) -> bool:
        if self.annotations_filter is None:
            return True

        else:
            for annotations_type in self.annotations_filter:
                if isinstance(annotations, annotations_type):
                    return True

        return False

    def visualize_operable(
        self, operable: SpotAnalysisOperable, is_last: bool, base_image: CacheableImage
    ) -> list[CacheableImage | rcfr.RenderControlFigureRecord]:
        """
        Updates the figures for this instance with the data from the given operable.
        """
        image = base_image.nparray

        # get a list of the fiducials to be rendered
        to_render: list[AbstractFiducials] = []
        to_render += list(filter(self._annotations_match_filter, operable.given_fiducials))
        to_render += filter(self._annotations_match_filter, operable.found_fiducials)
        to_render += filter(self._annotations_match_filter, operable.annotations)

        # initialize the figure
        self.figure.clear()
        self.figure.view.imshow(image)

        # render
        include_label = len(to_render) > 1
        for fiducials in to_render:
            fiducials.render_to_figure(self.figure, image, include_label)

        # show the visualization
        self.figure.view.show(block=False, legend=include_label)

        return [self.figure]

    def close_figures(self):
        """
        Closes all visualization windows created by this instance.
        """
        if self.figure is not None:
            self.figure.close()
            self.figure = None


if __name__ == "__main__":
    import os

    indir = ft.norm_path(
        os.path.join(
            orp.opencsp_scratch_dir(),
            "solar_noon/dev/2023-05-12_SpringEquinoxMidSummerSolstice/2_Data/BCS_data/Measure_01/processed_images",
        )
    )
    image_file = ft.norm_path(os.path.join(indir, "20230512_113032.81 5W01_000_880_2890 Raw_Testing_Peak_Flux.png"))

    style = rcps.RenderControlPointSeq(markersize=10)
    fiducials = PointFiducials(style, points=p2.Pxy(np.array([[0, 643, 1000], [0, 581, 1000]])))
    operable = SpotAnalysisOperable(CacheableImage(source_path=image_file), given_fiducials=[fiducials])

    processor = ViewAnnotationsImageProcessor()
    result = processor.process_operable(operable)[0]
    img = result.primary_image.nparray

    plt.figure()
    plt.imshow(img)
    plt.show(block=True)
