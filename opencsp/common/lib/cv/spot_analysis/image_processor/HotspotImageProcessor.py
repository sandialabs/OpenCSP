import copy
import dataclasses

import numpy as np

from opencsp.common.lib.cv.fiducials.HotspotFiducial import HotspotFiducial
import opencsp.common.lib.cv.image_filters as filters
import opencsp.common.lib.cv.image_reshapers as reshapers
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.FalseColorImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class HotspotImageProcessor(AbstractSpotAnalysisImagesProcessor):
    starting_max_factor: float = 2.5
    """ The factor to multiple desired_shape by to start the search """
    iteration_reduction_px: int = 10  # must be even
    """ The amount to subtract from the previous search shape by for each iteration """

    def __init__(self, desired_shape: int | tuple = 39, style: rcps.RenderControlPointSeq = None, draw_debug_view=False):
        """
        """
        super().__init__(self.__class__.__name__)

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
        self.style = style

        # determine a good iteration amount
        desired_shape_min = desired_shape if isinstance(desired_shape, int) else np.min(desired_shape)
        desired_shape_iter = int(np.round(desired_shape_min / 3))
        iter_sub = np.min([self.iteration_reduction_px, desired_shape_iter])
        if iter_sub % 2 == 1:
            iter_sub = np.max([iter_sub-1, 1])

        # internal variables
        self.has_warned_about_imports = False
        self.iter_sub = iter_sub

        # build the shapes that we're planning on iterating through
        if isinstance(desired_shape, tuple):
            self.internal_shapes = self._build_windows_tuple(desired_shape)
        else:  # isinstance(desired_shape, int)
            self.internal_shapes = self._build_windows_int(desired_shape)

    def _build_windows_tuple(self, desired_shape: tuple[int, int]) -> list[tuple[int, int]]:
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
                v = np.max([v-self.iter_sub, desired_shape[i]])
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

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        # The algorithm here iteratively zooms in on the hotest pixel in the
        # image by starting with the largest window and slowly reducing the
        # window size and image size. The steps are:
        #
        # 1. apply the filter for the current window size
        # 2. find the hottest pixel(s)
        # 3. verify that the hottest regions are continuous
        # 4. get the new window size
        # 5. reduce the image size to fit the new window size, reduce the window size, go to either step 1 or 6
        # 6. label the most central hottest pixel as the hotspot
        image = operable.primary_image.nparray
        _, image_name, _ = ft.path_components(operable.primary_image_source_path)
        total_start_y = 0
        total_start_x = 0

        # prepare the debug view
        if self.draw_debug_view:
            axis_control = rca.image(grid=False)
            fig_control = rcfg.RenderControlFigure(tile=True, tile_array=(4, 2), grid=False)
            view_spec = vs.view_spec_im()

            def imshow(img_to_show: np.ndarray, title: str):
                fig_rec = fm.setup_figure(fig_control, axis_control, view_spec, equal=False,
                                          name=image_name, title=title, code_tag=f"{__file__}._execute()")
                view = fig_rec.view
                view.imshow(reshapers.false_color_reshaper(img_to_show, 255))
                return view

        for shape_idx, shape in enumerate(self.internal_shapes):
            # 1. apply the current window size
            filtered_image = filters.percentile_filter(image, percentile=0, filter_shape=shape)

            # 2. find the hottest pixel(s)
            maxval = np.max(filtered_image)
            match_idxs = np.argwhere(filtered_image == maxval)

            # 3. verify that the hottest regions are continuous
            # TODO do we want to include scikit-image in the requirements?
            try:
                import skimage.morphology
                continuity_image = np.zeros(image.shape, 'uint8')
                continuity_image[filtered_image == maxval] = 2
                flooded_image = skimage.morphology.flood_fill(continuity_image, tuple(match_idxs[0]), 1)
                if np.max(flooded_image) > 1:
                    lt.error_and_raise("Error in PercentileFilterImageProcessor._execute(): " +
                                       f"There are at least 2 regions in {image_name} that share the hottest pixel value.")
            except ImportError as ex:
                if not self.has_warned_about_imports:
                    lt.debug("In PercentileFilterImageProcessor._execute(): " +
                             f"can't import scikit-image ({repr(ex)}), and so can't determine if the matching region is continuous")
                    self.has_warned_about_imports = True

            # 4. get the new window size
            start_y, end_y = match_idxs[0][0], match_idxs[-1][0] + 1
            start_x, end_x = match_idxs[0][1], match_idxs[-1][1] + 1
            if isinstance(shape, tuple):
                width, height = shape[1], shape[0]
            else:
                width, height = shape, shape
            start_y = np.max([0, start_y - height])
            end_y = np.min([image.shape[0], end_y + height])
            start_x = np.max([0, start_x - width])
            end_x = np.min([image.shape[1], end_x + width])

            # 5. reduce the image size to fit the new window size
            # This is both key to the algorithm and also an optimization
            image = image[start_y:end_y, start_x:end_x]
            total_start_y += start_y
            total_start_x += start_x

            # draw the debug view
            if self.draw_debug_view:
                if shape_idx == 0:
                    view = imshow(image, "original")
                filtered_image_part = filtered_image[start_y:end_y, start_x:end_x]
                imshow(filtered_image_part, str(shape))

        # 6. label the most central hottest pixel as the hotspot
        maxval = np.max(filtered_image)
        match_idxs = np.argwhere(filtered_image == maxval)
        center = p2.Pxy([int(image.shape[0]/2), int(image.shape[1]/2)])
        min_dist = 10e6
        central_match = None
        for i in range(match_idxs.shape[0]):
            x = match_idxs[i][1]
            y = match_idxs[i][0]
            idx = p2.Pxy([x, y])
            dist = (idx - center).magnitude()
            if dist < min_dist:
                min_dist = dist
                central_match = p2.Pxy([x + total_start_x, y + total_start_y])
        hotspot = HotspotFiducial(self.style, central_match)

        # wait for the interactive session
        if self.draw_debug_view:
            view.show(block=True)

        # return
        found_fiducials = copy.copy(operable.found_fiducials)
        found_fiducials.append(hotspot)
        new_operable = dataclasses.replace(operable, found_fiducials=found_fiducials)
        return [new_operable]
