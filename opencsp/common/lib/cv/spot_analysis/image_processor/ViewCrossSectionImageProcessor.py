from typing import Callable

import matplotlib.axes
import matplotlib.backend_bases
import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractVisualizationImageProcessor import (
    AbstractVisualizationImageProcessor,
)
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcf
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.image_tools as it


class ViewCrossSectionImageProcessor(AbstractVisualizationImageProcessor):
    """
    Interprets the current image as a 2D cross section and either displays it,
    or if interactive it displays the plot and waits on the next press of the
    "enter" key.

    This visualization uses either one or two windows to display the cross
    sections, depending on the initialization parameters.
    """

    def __init__(
        self,
        cross_section_location: tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]],
        label: str | rca.RenderControlAxis = 'Light Intensity',
        single_plot: bool = True,
        interactive: bool | Callable[[SpotAnalysisOperable], bool] = False,
        crop_to_threshold: int | None = None,
    ):
        """
        Parameters
        ----------
        cross_section_location : tuple[int, int] | Callable[[SpotAnalysisOperable], tuple[int, int]]
            The (x, y) pixel location to take cross sections through.
        label : str | rca.RenderControlAxis, optional
            The label to use for the window title, by default 'Cross Section at
            [cross_section_location]'
        single_plot : bool, optional
            If True, then draw both the horizational and vertical cross section
            graphs on the same plot. If False, then use two separate plots.
            Default is True.
        interactive : bool | Callable[[SpotAnalysisOperable], bool], optional
            If True then the spot analysis pipeline is paused until the user
            presses the "enter" key, by default False
        crop_to_threshold : int | None, optional
            Crops the image on the x and y axis to the first/last value >= the
            given threshold. None to not crop the image. Useful when trying to
            inspect hot spots on images with very concentrated values. By
            default None.
        """
        super().__init__(self.__class__.__name__, interactive)

        self.cross_section_location = cross_section_location
        self.label = label
        self.single_plot = single_plot
        self.crop_to_threshold = crop_to_threshold

        # initialize certain visualization values
        self.horizontal_style = rcps.RenderControlPointSeq(color='red', marker='None')
        self.vertical_style = rcps.RenderControlPointSeq(color='blue', marker='None')

        # declare future values
        self.view_specs: list[dict]
        self.rc_axises: list[rca.RenderControlAxis]
        self.fig_records: list[rcfr.RenderControlFigureRecord]
        self.views: list[v3d.View3d]
        self.axes: list[matplotlib.axes.Axes]
        self.plot_titles: list[str]

    @property
    def num_figures(self) -> int:
        if self.single_plot:
            return 1
        else:
            return 2

    def _init_figure_records(self, render_control_figure: rcf.RenderControlFigure) -> list[rcfr.RenderControlFigureRecord]:
        self.view_specs = []
        self.rc_axises = []
        self.fig_records = []
        self.views = []
        self.axes = []
        self.plot_titles = []

        if self.single_plot:
            plot_titles = [""]
        else:
            plot_titles = ["Horizontal CS: ", "Vertical CS: "]

        for plot_title in plot_titles:
            if self.single_plot:
                rc_axis = rca.RenderControlAxis(x_label='index', y_label='value')
                name_suffix = ""
            else:
                if "Horizontal" in plot_title:
                    rc_axis = rca.RenderControlAxis(x_label='x', y_label='value')
                    name_suffix = " (Horizontal)"
                else:
                    rc_axis = rca.RenderControlAxis(x_label='y', y_label='value')
                    name_suffix = " (Vertical)"

            view_spec = vs.view_spec_xy()
            fig_record = fm.setup_figure(
                render_control_figure,
                rc_axis,
                view_spec,
                equal=False,
                number_in_name=False,
                name=self.label + name_suffix,
                title="",
                code_tag=f"{__file__}.__init__()",
            )
            view = fig_record.view
            axes = fig_record.figure.gca()

            self.view_specs.append(view_spec)
            self.rc_axises.append(rc_axis)
            self.fig_records.append(fig_record)
            self.views.append(view)
            self.axes.append(axes)
            self.plot_titles.append(plot_title)

        return self.fig_records

    def _visualize_operable(self, operable: SpotAnalysisOperable, is_last: bool):
        image = operable.primary_image.nparray

        # get the cross section pixel location
        if isinstance(self.cross_section_location, tuple):
            cs_loc_x, cs_loc_y = self.cross_section_location
        else:
            cs_loc_x, cs_loc_y = self.cross_section_location(operable)

        # subselect a piece of the image based on the crop threshold
        y_start, y_end, x_start, x_end = 0, image.shape[0], 0, image.shape[1]
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

        if self.single_plot:
            # Align the cross sections so that the intersect point overlaps
            if cs_loc_x < cs_loc_y:
                diff = cs_loc_x - cs_loc_y
                v_p_list = [i + diff for i in v_p_list]
            if cs_loc_y < cs_loc_x:
                diff = cs_loc_y - cs_loc_x
                h_p_list = [i + diff for i in h_p_list]
        else:
            # Translate the cross sections plots to their actual locations
            v_p_list = [i + y_start for i in v_p_list]
            h_p_list = [i + x_start for i in h_p_list]

        # Clear the previous plot
        for fig_record in self.fig_records:
            fig_record.clear()

        # Update the title
        for plot_title_prefix, fig_record in zip(self.plot_titles, self.fig_records):
            fig_record.title = plot_title_prefix + operable.best_primary_nameext

        # Draw the new plot using the same axes
        v_view = self.views[0]
        h_view = self.views[0]
        if not self.single_plot:
            v_view = self.views[1]
        v_view.draw_pq_list(zip(v_p_list, v_cross_section), style=self.vertical_style, label="Vertical Cross Section")
        h_view.draw_pq_list(
            zip(h_p_list, h_cross_section), style=self.horizontal_style, label="Horizontal Cross Section"
        )

        # draw
        for view in self.views:
            view.show(block=False, legend=self.single_plot)

    def _close_figures(self):
        for view in self.views:
            with et.ignored(Exception):
                view.close()

        self.view_specs.clear()
        self.rc_axises.clear()
        self.fig_records.clear()
        self.views.clear()
        self.axes.clear()
        self.plot_titles.clear()


if __name__ == "__main__":
    from opencsp.common.lib.cv.CacheableImage import CacheableImage

    row = np.arange(100)
    rows = np.repeat(row, 100, axis=0).reshape(100, 100)
    # array([[ 0,  0,  0, ...,  0,  0,  0],
    #        [ 1,  1,  1, ...,  1,  1,  1],
    #        [ 2,  2,  2, ...,  2,  2,  2],
    #        ...,
    #        [97, 97, 97, ..., 97, 97, 97],
    #        [98, 98, 98, ..., 98, 98, 98],
    #        [99, 99, 99, ..., 99, 99, 99]])
    cacheable_rows = CacheableImage(rows, source_path=__file__)

    processor = ViewCrossSectionImageProcessor((50, 50), single_plot=False, interactive=True, crop_to_threshold=20)
    processor.process_image(SpotAnalysisOperable(cacheable_rows))
