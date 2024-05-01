from typing import Callable

import cv2 as cv
import matplotlib.backend_bases
import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


class ViewCrossSectionImageProcessor(AbstractSpotAnalysisImagesProcessor):
    """
    Interprets the current image as a 2D cross section and either displays it,
    or if interactive it displays the plot and waits on the next press of the
    "enter" key.
    """

    def __init__(
        self,
        cross_section_location: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]],
        label: str | rca.RenderControlAxis = 'Light Intensity',
        interactive: bool | Callable[[SpotAnalysisOperable], bool] = False,
        crop_to_threshold: int | None = None,
    ):
        """
        Parameters
        ----------
        cross_section_location : tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]]
            The (x, y) pixel location to take cross sections through.
        label : str | rca.RenderControlAxis, optional
            The label to use for the window title, by default 'Cross Section at [cross_section_location]'
        interactive : bool | Callable[[SpotAnalysisOperable], bool], optional
            If True then the spot analysis pipeline is paused until the user presses the "enter" key, by default False
        crop_to_threshold : int | None, optional
            Crops the image on the x and y axis to the first/last value >= the given threshold. None to not crop the
            image. Useful when trying to inspect hot spots on images with very concentrated values. By default None.
        """
        super().__init__(self.__class__.__name__)

        self.cross_section_location = cross_section_location
        self.label = label
        self.interactive = interactive
        self.enter_pressed = False
        self.closed = False
        self.crop_to_threshold = crop_to_threshold

        self.rcf = rcf.RenderControlFigure(tile=True, tile_array=(2, 1))
        self.rca = rca.RenderControlAxis()
        self.horizontal_style = rcps.RenderControlPointSeq(color='red', marker='None')
        self.vertical_style = rcps.RenderControlPointSeq(color='blue', marker='None')

        self._init_figure_record()

    def _init_figure_record(self):
        self.vs = vs.view_spec_xy()
        self.fig_record = fm.setup_figure(
            self.rcf,
            self.rca,
            self.vs,
            equal=False,
            number_in_name=False,
            name=self.label,
            title="",
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

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray

        # check if the view has been closed
        if self.closed:
            # UI design decision: it feels more natural to me (Ben) for the plot to not be shown again when it has
            # been closed instead of being reinitialized and popping back up.
            return [operable]
            # self._init_figure_record()

        # get the cross section pixel location
        if isinstance(self.cross_section_location, tuple):
            cs_loc_x, cs_loc_y = self.cross_section_location
        else:
            cs_loc_x, cs_loc_y = self.cross_section_location(operable)

        # subselect a piece of the image based on the crop threshold
        if self.crop_to_threshold is not None:
            y_start, y_end, x_start, x_end = it.range_for_threshold(image, self.crop_to_threshold)

            # check that this cropped range contains the cross section target
            if cs_loc_x >= x_start and cs_loc_x < x_end:
                if cs_loc_y >= y_start and cs_loc_y < y_end:
                    image = image[y_start:y_end, x_start:x_end]
                    cs_loc_x, cs_loc_y = cs_loc_x - x_start, cs_loc_y - y_start

        # Get the cross sections
        v_cross_section = image[:, cs_loc_x : cs_loc_x + 1].squeeze().tolist()
        v_p_list = list(range(len(v_cross_section)))
        h_cross_section = image[cs_loc_y : cs_loc_y + 1, :].squeeze().tolist()
        h_p_list = list(range(len(h_cross_section)))

        # Align the cross sections so that the intersect point overlaps
        if cs_loc_x < cs_loc_y:
            diff = cs_loc_x - cs_loc_y
            v_p_list = [i + diff for i in v_p_list]
        if cs_loc_y < cs_loc_x:
            diff = cs_loc_y - cs_loc_x
            h_p_list = [i + diff for i in h_p_list]

        # Clear the previous plot
        self.fig_record.view.clear()

        # Update the title
        self.fig_record.title = operable.best_primary_nameext

        # Draw the new plot using the same axes
        self.view.draw_pq_list(
            zip(v_p_list, v_cross_section), style=self.vertical_style, label="Vertical Cross Section"
        )
        self.view.draw_pq_list(
            zip(h_p_list, h_cross_section), style=self.horizontal_style, label="Horizontal Cross Section"
        )

        # draw
        self.view.show(legend=True, block=False)

        # wait for the user to press enter
        wait_for_enter_key = self.interactive if isinstance(self.interactive, bool) else self.interactive(operable)
        if wait_for_enter_key:
            self.enter_pressed = False
            while True:
                if self.enter_pressed or self.closed:
                    break
                self.fig_record.figure.waitforbuttonpress(0.1)

        return [operable]
