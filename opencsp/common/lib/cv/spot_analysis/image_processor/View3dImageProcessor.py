from typing import Callable

import cv2 as cv
import matplotlib.axes
import matplotlib.backend_bases
import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlSurface as rcs
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.image_tools as it


class View3dImageProcessor(AbstractVisualizationImageProcessor):
    """
    Interprets the current image as a 3D surface plot and either displays it, or if interactive it displays the surface
    and waits on the next press of the "enter" key.
    """

    def __init__(
        self,
        label: str | rca.RenderControlAxis = 'Light Intensity',
        interactive: bool | Callable[[SpotAnalysisOperable], bool] = False,
        max_resolution: tuple[int, int] | None = None,
        crop_to_threshold: int | None = None,
    ):
        """
        Parameters
        ----------
        label : str | rca.RenderControlAxis, optional
            The label to use for the window title, by default 'Light Intensity'
        interactive : bool | Callable[[SpotAnalysisOperable], bool], optional
            If True then the spot analysis pipeline is paused until the user presses the "enter" key, by default False
        max_resolution : tuple[int, int] | None, optional
            Limits the resolution along the x and y axes to the given values. No limit if None. By default None.
        crop_to_threshold : int | None, optional
            Crops the image on the x and y axis to the first/last value >= the given threshold. None to not crop the
            image. Useful when trying to inspect hot spots on images with very concentrated values. By default None.
        """
        super().__init__(self.__class__.__name__, interactive)

        self.max_resolution = max_resolution
        self.crop_to_threshold = crop_to_threshold

        # intialize certain visualization values
        if isinstance(label, str):
            self.rca = rca.RenderControlAxis(z_label=label)
        else:
            self.rca = label
        self.rcs = rcs.RenderControlSurface(alpha=1.0, color=None, contour='xyz')

        # declare future values
        self.fig_record: rcfr.RenderControlFigureRecord
        self.view: v3d.View3d
        self.axes: matplotlib.axes.Axes

    @property
    def num_figures(self) -> int:
        return 1

    def _init_figure_records(self, render_control_fig: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        self.fig_record = fm.setup_figure_for_3d_data(
            render_control_fig,
            self.rca,
            equal=False,
            number_in_name=False,
            name=self.rca.z_label,
            code_tag=f"{__file__}.__init__()",
        )
        self.view = self.fig_record.view
        self.axes = self.fig_record.figure.gca()

        return [self.fig_record]

    def _visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool):
        image = operable.primary_image.nparray

        # reduce data based on threshold
        y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
        if self.crop_to_threshold is not None:
            y_start, y_end, x_start, x_end = it.range_for_threshold(image, self.crop_to_threshold)
            image = image[y_start:y_end, x_start:x_end]

        # reduce data based on max_resolution
        if self.max_resolution is not None:
            width = np.min([x_end - x_start, self.max_resolution[0]])
            height = np.min([y_end - y_start, self.max_resolution[1]])
            image = cv.resize(image, (height, width), interpolation=cv.INTER_AREA)

        # Clear the previous data
        self.fig_record.clear()

        # Update the title
        self.fig_record.title = operable.best_primary_nameext

        # Draw the new data
        if self.crop_to_threshold is None and self.max_resolution is None:
            self.view.draw_xyz_surface(image, self.rcs)
        else:
            width = image.shape[1]
            height = image.shape[0]
            x_arr = (np.arange(0, width) * (x_end - x_start) / width) + x_start
            y_arr = (np.arange(0, height) * (y_end - y_start) / height) + y_start
            x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
            self.view.draw_xyz_surface_customshape(x_mesh, y_mesh, image, self.rcs)

        # draw
        self.view.show(block=False)

    def _close_figures(self):
        with et.ignored(Exception):
            self.view.close()

        self.fig_record = None
        self.view = None
        self.axes = None
