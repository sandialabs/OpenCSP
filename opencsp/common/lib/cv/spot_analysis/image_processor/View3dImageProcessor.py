from typing import Callable

import cv2 as cv
import matplotlib.backend_bases
import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlSurface as rcs
import opencsp.common.lib.tool.file_tools as ft
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
        super().__init__(self.__class__.__name__)

        self.interactive = interactive
        self.enter_pressed = False
        self.closed = False
        self.max_resolution = max_resolution
        self.crop_to_threshold = crop_to_threshold

        self.rcf = rcf.RenderControlFigure(tile=False)
        if isinstance(label, str):
            self.rca = rca.RenderControlAxis(z_label=label)
        else:
            self.rca = label
        self.rcs = rcs.RenderControlSurface(alpha=1.0, color=None, contour='xyz')

        self._init_figure_record()

    def _init_figure_record(self):
        self.fig_record = fm.setup_figure_for_3d_data(
            self.rcf,
            self.rca,
            equal=False,
            number_in_name=False,
            name=self.rca.z_label,
            code_tag=f"{__file__}.__init__()",
        )
        self.view = self.fig_record.view
        self.axes = self.fig_record.figure.gca()

        self.enter_pressed = False
        self.closed = False
        self.fig_record.figure.canvas.mpl_connect('close_event', self.on_close)
        self.fig_record.figure.canvas.mpl_connect('key_release_event', self.on_key_release)

    def on_key_release(self, event: matplotlib.backend_bases.KeyEvent):
        if event.key == "enter" or event.key == "return":
            self.enter_pressed = True

    def on_close(self, event: matplotlib.backend_bases.CloseEvent):
        self.closed = True

    def _visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool):
        image = operable.primary_image.nparray

        # check if the view has been closed
        if self.closed:
            # UI design decision: it feels more natural to me (Ben) for the plot to not be shown again when it has
            # been closed instead of being reinitialized and popping back up.
            return
            # self._init_figure_record()

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
        self.fig_record.view.clear()

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

        # wait for the user to press enter
        wait_for_enter_key = self.interactive if isinstance(self.interactive, bool) else self.interactive(operable)
        if wait_for_enter_key:
            self.enter_pressed = False
            while True:
                if self.enter_pressed or self.closed:
                    break
                self.fig_record.figure.waitforbuttonpress(0.1)
