import copy
import dataclasses
from typing import Callable

import cv2 as cv
import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.annotations.HotspotAnnotation import HotspotAnnotation
import opencsp.common.lib.cv.image_filters as filters
import opencsp.common.lib.cv.image_reshapers as reshapers
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class HotspotImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Adds an annotation marker to images to indicate at which pixel the
    brightest part of the image is.

    We want to be able to determine the point of peak intensity within an image,
    within a certain window size. The window size is chosen manually and should
    reflect the level of uncertainty in heliostat pointing. A local minimum
    filter is applied to find the overall hottest part of the image. The filter
    starts large and gradually shrinks until it fits the desired window size.

    Note: this is NOT the same as the brightest pixel, which can be trivially
    found with np.max(image). Likewise, it is NOT the centroid.
    """

    starting_max_factor: float = 2
    """ The factor to multiple desired_shape by to start the search """
    iteration_reduction_px: int = 10  # must be even
    """ The amount to subtract from the previous search shape by for each iteration """

    def __init__(
        self,
        desired_shape: int | tuple,
        style: rcps.RenderControlPointSeq = None,
        draw_debug_view: bool | Callable[[SpotAnalysisOperable], bool] = False,
        record_visualization=False,
        record_debug_view: bool | int = False,
    ):
        """
        Parameters
        ----------
        desired_shape : int | tuple, optional
            The window size used to determine the brightest area, in pixels. If
            an integer, then the same value is used for all dimensions. If a
            tuple, then it must have the same number of values as dimensions in
            the image. Must be odd.
        style : rcps.RenderControlPointSeq, optional
            The style used to render the hotspot point with
            View3d.draw_pq_list(). By default ('x', red).
        draw_debug_view : bool | Callable[[SpotAnalysisOperable], bool], optional
            True to show the iterative process used to converge on the hotspot.
            By default False.
        record_visualization: bool
            True to add the visualization of the hotspot on top of the input
            image to the visualization images. By default False.
        record_debug_view : bool | int, optional
            True to record the debug views to the visualization images. If an
            integer, than up to that number of debug images will be recorded. If
            draw_debug_view is False then this option does nothing.
        """
        super().__init__()

        # normalize the input
        if not isinstance(desired_shape, tuple):
            desired_shape = int(desired_shape)  # force floats etc to be ints

        # validate the input
        # if valid, then percentile_filter won't raise any issues
        if isinstance(desired_shape, tuple):
            test_img = np.zeros(desired_shape, dtype='uint8')
        else:
            test_img = np.zeros((desired_shape, desired_shape))
        filters.percentile_filter(test_img, 100, desired_shape)

        # register values
        self.desired_shape = desired_shape
        self.draw_debug_view = draw_debug_view
        self.record_debug_view = record_debug_view
        self.style = style
        self.record_visualization = record_visualization

        # determine a good iteration amount
        desired_shape_min = desired_shape if isinstance(desired_shape, int) else np.min(desired_shape)
        desired_shape_iter = int(np.round(desired_shape_min / 3))
        iter_sub = np.min([self.iteration_reduction_px, desired_shape_iter])
        if iter_sub % 2 == 1:
            iter_sub = np.max([iter_sub - 1, 1])

        # internal variables
        self.iter_sub = iter_sub
        self.has_scikit_image = None

        # build the shapes that we're planning on iterating through
        if isinstance(desired_shape, tuple):
            self.internal_shapes = self._build_windows_tuple(desired_shape)
        else:  # isinstance(desired_shape, int)
            self.internal_shapes = self._build_windows_int(desired_shape)

    def _build_windows_tuple(self, desired_shape: tuple[int, int]) -> list[tuple[int, int]]:
        """
        Constructs the list of window sizes to iterate through while isolating the hot spot.

        The window sizes will be in order from largest to smallest, with the
        intention that each successive (smaller) window will search over the
        matched area from the previous (larger) window.

        This method works with non-square window tuples.

        Parameters
        ----------
        desired_shape: tuple[int, int]
            The final shape for the window with the desired size in the x and y axes, in pixels.
        """
        ret: list[tuple[int, int]] = []

        # determine the starting shape
        starting_shape_list: list[int] = []
        for i, v in enumerate(desired_shape):
            v = int(np.ceil(v * self.starting_max_factor))
            if v % 2 == 0:
                v += 1
            starting_shape_list.append(v)
        starting_shape = tuple(starting_shape_list)
        ret.append(starting_shape)

        curr_shape = starting_shape
        while True:
            # reduce each dimension by iter_sub
            reduced_shape: list[int] = []
            for i, v in enumerate(curr_shape):
                v = np.max([v - self.iter_sub, desired_shape[i]])
                reduced_shape.append(int(v))

            # prepare for next iteration
            if curr_shape == reduced_shape:
                # we must have reached the desired shape
                break
            curr_shape = reduced_shape

            # register the reduced shape
            ret.append(tuple(reduced_shape))

        return ret

    def _build_windows_int(self, desired_shape: int) -> list[int]:
        """
        See _build_windows_tuple() for a full description.

        This method works with square windows.

        Parameters
        ----------
        desired_shape: int
            The final shape for the window with the desired size both axes, in pixels.
        """
        ret: list[int] = []

        # determine the starting shape
        starting_size = int(np.ceil(desired_shape * self.starting_max_factor))
        if starting_size % 2 == 0:
            starting_size += 1
        curr_size = starting_size
        ret.append(starting_size)

        while curr_size > desired_shape:
            # reduce by iter_sub
            reduced_size = int(curr_size - self.iter_sub)
            reduced_size = np.max([reduced_size, desired_shape])

            # prepare for next iteration
            curr_size = reduced_size

            # register the reduced size
            ret.append(reduced_size)

        return ret

    def _draw_debug_image(
        self, image: np.ndarray, fig_rec: fm.RenderControlFigureRecord, x1: int, x2: int, y1: int, y2: int
    ):
        """
        Draws the given img and a rectangle at the given location.

        Parameters
        ----------
        img : np.ndarray
            The image to be drawn.
        fig_rec : fm.RenderControlFigureRecord
            The figure to use to draw the image and rectangle.
        x1 : int
            Pixel index of the left side of the rectangle.
        x2 : int
            Pixel index of the right side of the rectangle.
        y1 : int
            Pixel index of the top side of the rectangle.
        y2 : int
            Pixel index of the bottom side of the rectangle.
        """
        # draw the image
        image = reshapers.false_color_reshaper(image, 255)
        (height, width), nchannels = it.dims_and_nchannels(image)
        fig_rec.view.imshow(image)

        # sanitize the input
        x1 = np.clip(x1, 0, width - 1)
        x2 = np.clip(x2, x1, width - 1)
        y1 = np.clip(y1, 0, height - 1)
        y2 = np.clip(y2, y1, height - 1)

        # matplotlib puts the origin in the bottom left instead of the top left
        y1 = height - y1
        y2 = height - y2

        # draw the rectangle
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        fig_rec.view.draw_pq_list(corners, close=True, style=self.style)

        return fig_rec

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # The algorithm here iteratively zooms in on the hotest pixel in the
        # image by starting with the largest window and slowly reducing the
        # window size and image size. The steps are:
        #
        # 1. apply the filter for the current window size
        # 2. find the hottest pixel(s)
        # 3. get the new window size
        # 4. verify that the hottest regions are continuous
        # 5. reduce the image size to fit the new window size, reduce the window size, go to either step 1 or 6
        # 6. label the most central hottest pixel as the hotspot
        image = operable.primary_image.nparray
        total_start_y = 0
        total_start_x = 0

        # prepare the debug view
        show_debug_view = (
            self.draw_debug_view if isinstance(self.draw_debug_view, bool) else self.draw_debug_view(operable)
        )
        draw_debug_view = (show_debug_view) or (self.record_debug_view != False)
        if draw_debug_view:
            (height, width), nchannels = it.dims_and_nchannels(image)
            fig_size = rcfg.RenderControlFigure.pixel_resolution_inches(width, height)
            axis_control = rca.image(grid=False)
            fig_control = rcfg.RenderControlFigure(tile=True, tile_array=(4, 2), figsize=fig_size, grid=False)
            view_spec_image = vs.view_spec_im()
            view_spec_pq = vs.view_spec_pq()
            debug_fig_recs: list[rcfr.RenderControlFigureRecord] = []
            ppt_fig_images: list[np.array] = []

        for shape_idx, shape in enumerate(self.internal_shapes):
            if isinstance(shape, tuple):
                width, height = shape[1], shape[0]
            else:
                width, height = shape, shape

            # 1. apply the current window size
            filtered_image = filters.percentile_filter(image, percentile=0, filter_shape=shape)

            # 2. find the hottest pixel(s)
            maxval = np.max(filtered_image)
            match_idxs = np.argwhere(filtered_image == maxval)
            match_where = np.where(filtered_image == maxval)

            # 3. get the new window size
            # The new window should be centered around the hottest pixels.
            start_y, end_y = match_where[0][0], match_where[0][-1] + 1
            start_x, end_x = match_where[1][0], match_where[1][-1] + 1
            # Get ending values that include the window around the hottest pixels
            if (end_x - start_x) < width:
                mid_x = start_x + ((end_x - start_x) / 2)
                start_x, end_x = mid_x - (width / 2), mid_x + (width / 2)
                start_x, end_x = np.max([start_x, 0]), np.min([end_x, image.shape[1]])
                start_x, end_x = int(start_x), int(end_x)
            if (end_y - start_y) < height:
                mid_y = start_y + ((end_y - start_y) / 2)
                start_y, end_y = mid_y - (height / 2), mid_y + (height / 2)
                start_y, end_y = np.max([start_y, 0]), np.min([end_y, image.shape[0]])
                start_y, end_y = int(start_y), int(end_y)
            total_start_y += start_y
            total_start_x += start_x

            # draw the debug view
            if draw_debug_view:
                setup_figure = lambda title, view_spec: fm.setup_figure(
                    fig_control,
                    axis_control,
                    view_spec,
                    equal=False,
                    name=operable.best_primary_nameext,
                    title=title,
                    code_tag=f"{__file__}._execute()",
                )

                if shape_idx == 0:
                    fig_rec = setup_figure("original", view_spec_image)
                    fig_rec.view.imshow(reshapers.false_color_reshaper(image, 255))
                    debug_fig_recs.append(fig_rec)

                if show_debug_view:
                    fig_rec = setup_figure(str(shape), view_spec_pq)
                    self._draw_debug_image(filtered_image, fig_rec, start_x, end_x, start_y, end_y)
                    debug_fig_recs.append(fig_rec)

                fig_rec = setup_figure(str(shape), view_spec_pq)
                self._draw_debug_image(image, fig_rec, start_x, end_x, start_y, end_y)
                ppt_fig_images.append(fig_rec.to_array())
                fig_rec.close()

            # 4. verify that the hottest regions are continuous
            # TODO do we want to include scikit-image in requirements.txt?
            if self.has_scikit_image is None:
                try:
                    import skimage.morphology

                    self.has_scikit_image = True

                    continuity_image = np.zeros(image.shape, 'uint8')
                    continuity_image[filtered_image == maxval] = 2
                    flooded_image = skimage.morphology.flood_fill(continuity_image, tuple(match_idxs[0]), 1)
                    if np.max(flooded_image) > 1:
                        lt.warning(
                            "Warning in HotspotImageProcessor._execute(): "
                            + f"There are at least 2 regions in '{operable.best_primary_nameext}', "
                            + f"area [{total_start_y}:{total_start_y+end_y}, {total_start_x}:{total_start_x+end_x}] "
                            + "that share the hottest pixel value."
                        )

                except ImportError as ex:
                    self.has_scikit_image = False

                    lt.debug(
                        "In HotspotImageProcessor._execute(): "
                        + f"can't import scikit-image ({repr(ex)}), and so can't determine if the matching region is continuous"
                    )

            # 5. reduce the image size to fit the new window size
            # This is both key to the algorithm and also an optimization
            image = image[start_y:end_y, start_x:end_x]

        # 6. label the center of the hotspot region as the hotspot
        regional_center = p2.Pxy([int(image.shape[1] / 2), int(image.shape[0] / 2)])
        absolute_hotspot = regional_center + p2.Pxy([total_start_x, total_start_y])
        hotspot = HotspotAnnotation(self.style, absolute_hotspot)

        # wait for the interactive session
        if show_debug_view:
            fig_rec.view.show(block=True)
        if draw_debug_view:
            for fig_rec in debug_fig_recs:
                fig_rec.close()

        # build the return value
        annotations = copy.copy(operable.annotations)
        annotations.append(hotspot)
        new_operable = dataclasses.replace(operable, annotations=annotations)

        # add the debug view of this step to the visualization images
        if self.record_debug_view is not False:
            to_draw_count = len(ppt_fig_images)
            if isinstance(self.record_debug_view, int):
                to_draw_count = np.min([to_draw_count, self.record_debug_view])
            ppt_fig_images = ppt_fig_images[-to_draw_count:]
            algorithm_images = copy.copy(new_operable.algorithm_images)
            algorithm_images[self] = [CacheableImage(img) for img in ppt_fig_images]
            new_operable = dataclasses.replace(new_operable, algorithm_images=algorithm_images)

        # add the visualization of this step to the visualization images
        if self.record_visualization:
            visualized = hotspot.render_to_image(operable.primary_image.nparray)
            visualization_images = copy.copy(new_operable.visualization_images)
            if self not in visualization_images:
                visualization_images[self] = []
            visualization_images[self] += [CacheableImage(visualized)]
            new_operable = dataclasses.replace(new_operable, visualization_images=visualization_images)

        return [new_operable]
