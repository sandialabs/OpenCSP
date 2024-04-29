import copy
import dataclasses

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.fiducials.HotspotFiducial import HotspotFiducial
from opencsp.common.lib.cv.fiducials.PointFiducials import PointFiducials
import opencsp.common.lib.cv.image_filters as imgf
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


class HotspotImageProcessor(AbstractSpotAnalysisImagesProcessor):
    starting_max_factor: float = 2.5
    """ The factor to multiple desired_shape by to start the search """
    iteration_reduction_px: int = 6  # must be even
    """ The amount to subtract from the previous search shape by for each iteration """

    def __init__(self, desired_shape: int | tuple = 39, draw_debug_view=False):
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
        imgf.percentile_filter(test_img, 100, desired_shape)

        # register values
        self.desired_shape = desired_shape
        self.draw_debug_view = draw_debug_view

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
            # reduce each dimension by iteration_reduction_px
            reduced_shape: list[int] = []
            for i, v in enumerate(curr_shape):
                v = np.max([v-self.iteration_reduction_px, desired_shape[i]])
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
            # reduce by iteration_reduction_px
            reduced_size = int(curr_size - self.iteration_reduction_px)
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

        for shape in self.internal_shapes:
            # 1. apply the current window size
            filtered_image = imgf.percentile_filter(image, percentile=100, filter_shape=shape)

            # 2. find the hottest pixel(s)
            maxval = np.max(filtered_image)
            match_idxs = np.argwhere(filtered_image == maxval)

            # 3. verify that the hottest regions are continuous
            # TODO do we want to include scikit-learn in the requirements?
            try:
                import skimage.morphology
                continuity_image = np.zeros(image.shape, 'uint8')
                continuity_image[filtered_image == maxval] = 2
                flooded_image = skimage.morphology.flood_fill(continuity_image, match_idxs[0], 1)
                if np.max(flooded_image) > 1:
                    lt.error_and_raise("Error in PercentileFilterImageProcessor._execute(): " +
                                       f"There are at least 2 regions in {image_name} that share the hottest pixel value.")
            except ImportError as ex:
                lt.error("In PercentileFilterImageProcessor._execute(): " +
                         f"can't import scikit-learn ({repr(ex)}), and so can't determine if the matching region is continuous")

            # 4. get the new window size
            start_y, end_y = match_idxs[0][0], match_idxs[-1][0] + 1
            start_x, end_x = match_idxs[0][1], match_idxs[-1][1] + 1
            if isinstance(shape, tuple):
                start_y = np.max([0, start_y - shape[0]])
                end_y = np.min([image.shape[0], end_y + shape[0]])
                start_x = np.max([0, start_x - shape[1]])
                end_x = np.min([image.shape[1], end_x + shape[1]])

            # 5. reduce the image size to fit the new window size
            # This is both key to the algorithm and also an optimization
            image = image[start_y:end_y, start_x:end_x]
            total_start_y += start_y
            total_start_x += start_x

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
        hotspot = HotspotFiducial(self.hotspot_style, central_match)

        # return
        found_fiducials = copy.copy(operable.found_fiducials)
        found_fiducials.append(hotspot)
        new_operable = dataclasses.replace(operable, found_fiducials=found_fiducials)
        return [new_operable]


if __name__ == "__main__":
    import os

    from PIL import Image
    import numpy as np

    from opencsp.common.lib.cv.spot_analysis.image_processor import FalseColorImageProcessor
    import opencsp.common.lib.opencsp_path.opencsp_root_path as orp
    import opencsp.common.lib.render.figure_management as fm
    import opencsp.common.lib.render.view_spec as vs
    import opencsp.common.lib.render_control.RenderControlAxis as rca
    import opencsp.common.lib.render_control.RenderControlFigure as rcfg
    import opencsp.common.lib.tool.file_tools as ft

    dir = ft.norm_path(
        os.path.join(
            orp.opencsp_scratch_dir(),
            "solar_noon/dev/2023-05-12_SpringEquinoxMidSummerSolstice/2_Data/BCS_data/Measure_01/raw_images",
        )
    )
    file = "20230512_113036.31 5W01_000_880_2890 Raw.JPG"
    image_path_name_ext = ft.norm_path(os.path.join(dir, file))

    img = np.array(Image.open(image_path_name_ext))
    img = img[550:800, 600:1000]
    cimg = CacheableImage(img, source_path=image_path_name_ext)
    operable = SpotAnalysisOperable(cimg, primary_image_source_path=image_path_name_ext)
    fc_processor = FalseColorImageProcessor()
    fcimg = fc_processor.process_image(operable, False)[0].primary_image.nparray

    # draw the original image
    axis_control = rca.image(grid=False)
    figure_control = rcfg.RenderControlFigure(tile=True, tile_array=(3, 2))
    view_spec_2d = vs.view_spec_im()
    fig_record_orig = fm.setup_figure(
        figure_control, axis_control, view_spec_2d, title="Original", code_tag=f"{__file__}", equal=False
    )
    fig_record_orig.view.imshow(fcimg)
    fig_record_orig.view.show()

    for i, winsize in enumerate([9, 19, 29, 39, 49]):
        # get the modified image
        processor = PercentileImageProcessor(percentile=0, size=winsize)
        intermediate = processor.process_image(operable, False)[0]
        result = fc_processor.process_image(intermediate, False)[0]

        # draw the modified image
        fig_record_min = fm.setup_figure(
            figure_control, axis_control, view_spec_2d, title=f"Minimum, Window Size {winsize}", code_tag=f"{__file__}", equal=False
        )
        fig_record_min.view.imshow(result.primary_image.nparray)
        fig_record_min.view.show(block=(i == 4))
