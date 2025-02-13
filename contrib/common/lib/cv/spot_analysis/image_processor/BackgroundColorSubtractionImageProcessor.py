import copy
import dataclasses

import cv2 as cv
import numpy as np
from PIL import Image
import scipy.interpolate
import scipy.spatial.transform

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable, ImageType
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.geometry.LineXY as l2
import contrib.common.lib.geometry.RectXY as r2
import opencsp.common.lib.geometry.RegionXY as reg2
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class BackgroundColorSubtractionImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    A simple background color detection and removal image processor.
    """

    def __init__(self, color_detection_method: str = "solid", constant_color_value: tuple[int] = (0)):
        """
        Parameters
        ----------
        color_detection_method : str, optional
            The color detection method. By default 'auto'. Can be one of the following:

            - constant:
              The user provides a constant_color_value that will be subtracted.
            - solid:
              The most common color value is used as the value to be subtracted.
            - plane:
              Starts with a solid color, then adjusts the colors at each of
              three points until the interpolated 2D gradient most closely
              matches the image.
        constant_color_value : tuple[int], optional
            The value to subtract in the case that color_detection_method is
            'constant'. Should have as many values as there are color channels
            (usually this means 1 value for grayscale images or 3 values for RGB
            images). By default (0).
        """
        # The gradient option is not recommended. Suggestion is to use plane instead.
        # - gradient:
        #     Starts with a solid color, then adjusts the colors at each corner
        #     until the interpolated 2D gradient most closely matches the image.
        super().__init__()

        allowed_color_detection_methods = ["constant", "solid", "gradient", "plane"]
        if color_detection_method not in allowed_color_detection_methods:
            lt.error_and_raise(
                ValueError,
                "Error in BackgroundColorSubtractionImageProcessor(): "
                + f"unknown color detection method {color_detection_method}, "
                + f"must be one of {allowed_color_detection_methods}.",
            )

        self.color_detection_method = color_detection_method
        self.constant_color_value = constant_color_value

    def find_solid_background_color(self, primary_image: np.ndarray) -> tuple[int]:
        if primary_image.ndim > 2:
            ret: list[int] = []
            for channel in range(primary_image.shape[2]):
                ret.append(self.find_solid_background_color(primary_image[:, :, channel])[0])
            return tuple(ret)

        # get the range of colors
        color_range = [np.min(primary_image), np.max(primary_image)]
        hist = cv.calcHist([primary_image], [0], None, [color_range[1] - color_range[0]], color_range)
        color_cnt = {i + color_range[0]: v for i, v in enumerate(hist)}

        # choose the colors that make up the top 20% of the image
        cnts: list[int] = []
        colors: list[int] = []
        height, width = it.dims_and_nchannels(primary_image)[0]
        while sum(cnts) < (height * width * 0.2):
            next_cnt = max(color_cnt.values())
            for color in color_cnt:
                if color_cnt[color] == next_cnt:
                    colors.append(color)
                    cnts.append(int(next_cnt))
                    del color_cnt[color]
                    break

        # discard colors that stray too far from the other colors
        max_allowed_spread_radius = 10
        if max(colors) - min(colors) > max_allowed_spread_radius * 2:
            if len(colors) > 1:
                keep_cnts: list[int] = []
                keep_colors: list[int] = []
                avg_color = sum([colors[i] * (cnts[i] / sum(cnts)) for i in range(len(colors))])
                for i in range(len(colors)):
                    if np.abs(avg_color - colors[i]) < max_allowed_spread_radius:
                        keep_cnts.append(cnts[i])
                        keep_colors.append(cnts[i])

                cnts = keep_cnts
                colors = keep_colors

        # use the average of these colors
        avg_color = sum([colors[i] * (cnts[i] / sum(cnts)) for i in range(len(colors))])
        avg_color_int = int(np.round(avg_color))

        return tuple([avg_color_int])

    def fit_four_corners(self, image: np.ndarray, solid_background_color: tuple[int]) -> tuple[tuple[int]]:
        """
        Fits a 4-corner interpolated background to the given image,
        using the given solid_background_color as the starting point.

        Steps:
            1. subtract the current four-corner fit
            2. isolate no-spot pixels from spot pixels
            3. increase the value for quadrants that have >50% no-spot pixels
               visible, and decrease the value for quadrants that have <50%
               no-spot pixels visible
            4. repeat steps 1-3 until values stabilize or we reach a maximum
               number of iterations
        """
        max_iterations = 255
        max_int = min(np.iinfo(image.dtype).max, 2**12)
        noise_threshold = int(np.round(5 / 255 * max_int))
        (height, width), nchannels = it.dims_and_nchannels(image)

        # build the corner data
        half_height, half_width = int(height / 2), int(width / 2)
        corner_colors_tl_tr_bl_br: list[tuple[int, int, int, int]] = []
        for channel in range(nchannels):
            channel_color = solid_background_color[channel]
            channel_colors = [channel_color for corner_index in range(4)]
            corner_colors_tl_tr_bl_br.append(channel_colors)
        corner_selection_tl_tr_bl_br = [
            lambda arr: arr[: height - half_height, : width - half_width, ...],
            lambda arr: arr[: height - half_height, half_width:, ...],
            lambda arr: arr[half_height:, : width - half_width, ...],
            lambda arr: arr[half_height:, half_width:, ...],
        ]
        previously_seen_corner_colors = [copy.deepcopy(corner_colors_tl_tr_bl_br)]

        # build the initial fit
        four_corner_fit = self.build_background_image(image, gradient_tl_tr_bl_br=corner_colors_tl_tr_bl_br)

        # determine the no-spot pixels
        nospot_pixels_image = np.zeros_like(image)
        initial_sub_image = image.astype(np.int64) - four_corner_fit.astype(np.int64)
        nospot_pixels_image[np.where(initial_sub_image <= noise_threshold)] = 1

        for i in range(max_iterations):
            # step 1: subtract current fit
            residual_image = image.astype(np.int64) - four_corner_fit.astype(np.int64)

            colors_changed = False
            for corner_index, corner_selection in enumerate(corner_selection_tl_tr_bl_br):
                corner_pixels = corner_selection(residual_image)
                for channel in range(nchannels):
                    corner_channel_pixels = corner_pixels
                    if nchannels > 1:
                        corner_channel_pixels = corner_pixels[:, :, channel]

                    # step 2: isolate no-spot pixels
                    nospot_pixels = np.where(corner_selection(nospot_pixels_image))
                    num_nospot_pixels = nospot_pixels[0].size

                    # step 3: increase or decrease values
                    ndark = np.where(corner_channel_pixels[nospot_pixels] < 0)[0].size
                    nzero = np.where(corner_channel_pixels[nospot_pixels] == 0)[0].size
                    nbright = np.where(corner_channel_pixels[nospot_pixels] > 0)[0].size

                    # no change? increase? decrease?
                    if (nbright == ndark) or (nzero >= num_nospot_pixels * 0.45):
                        pass
                    elif nbright > ndark:
                        if nbright > num_nospot_pixels * 0.4:
                            corner_colors_tl_tr_bl_br[channel][corner_index] += 1
                            colors_changed = True
                    elif ndark > nbright:
                        if ndark > num_nospot_pixels * 0.4:
                            corner_colors_tl_tr_bl_br[channel][corner_index] -= 1
                            colors_changed = True

                    if (
                        corner_colors_tl_tr_bl_br[channel][corner_index] == 0
                        or corner_colors_tl_tr_bl_br[channel][corner_index] == 255
                    ):
                        pass

            # step 4: repeat?
            if (colors_changed) and (corner_colors_tl_tr_bl_br not in previously_seen_corner_colors):
                four_corner_fit = self.build_background_image(image, gradient_tl_tr_bl_br=corner_colors_tl_tr_bl_br)
                previously_seen_corner_colors.append(copy.deepcopy(corner_colors_tl_tr_bl_br))
            else:
                break

        return corner_colors_tl_tr_bl_br

    def _remove_duplicate_points(self, points: list[p2.Pxy]) -> list[p2.Pxy]:
        points = list(copy.copy(points))

        to_remove: list[p2.Pxy] = []
        for n in range(len(points)):
            m = (n + 1) % len(points)
            pt_n, pt_m = points[n], points[m]
            if (np.abs(pt_n.x[0] - pt_m.x[0]) < 0.1) and (np.abs(pt_n.y[0] - pt_m.y[0]) < 0.1):
                to_remove.append(pt_m)

        for pt in to_remove:
            points.remove(pt)

        return points

    def _get_three_point_regions(self, image: np.ndarray) -> tuple[p2.Pxy, list[reg2.RegionXY]]:
        (height, width), nchannels = it.dims_and_nchannels(image)

        # Check if the image is oriented so that the long axis is horizontal,
        # and transpose the image as necessary. We transpose it to remove one of
        # the complications in generating our 3-point regions.
        if width < height:
            transposed_image = image.transpose()
            three_points_transposed, three_points_regions = self._get_three_point_regions(transposed_image)
            three_points = three_points_transposed.transpose()
            for i in range(3):
                three_points_regions[i].loops[0].flip_in_place()
            return three_points, three_points_regions

        # Determine the three points that will define our plane.
        # The three points are in top-left, top-right, bottom-middle order.
        half_width, half_height = int(width / 2), int(height / 2)
        bounds = r2.RectXY(p2.Pxy([[0], [0]]), p2.Pxy([[width], [height]]))
        xs = [0, width, half_width]
        ys = [0, 0, height]
        three_points = p2.Pxy((xs, ys))

        # Define two division lines. One between the top-left point and the
        # bottom-middle point, and one between the top-right point and the
        # bottom-middle point.
        left_midpoint = p2.Pxy([[int(half_width / 2)], [half_height]])
        left_division = l2.LineXY.from_location_angle(left_midpoint, -np.pi / 4)
        right_midpoint = p2.Pxy([[int((width - half_width) / 2) + half_width], [half_height]])
        right_division = l2.LineXY.from_location_angle(right_midpoint, np.pi / 4)

        # Find the intersection point between the two division lines
        division_intersection = left_division.intersect_with(right_division)
        middle_pt = p2.Pxy([[int(np.round(division_intersection.x[0]))], [int(np.round(division_intersection.y[0]))]])
        assert np.abs(middle_pt.x[0] - half_width) < 2

        # Use a VERY big rectangle to define our three trirants that define our
        # image regions
        left_outer_edge = l2.LineXY.from_location_angle(p2.Pxy([[-width - 10], [0]]), np.pi / 2)
        right_outer_edge = l2.LineXY.from_location_angle(p2.Pxy([[width * 2 + 10], [0]]), np.pi / 2)
        top_outer_edge = l2.LineXY.from_location_angle(p2.Pxy([[0], [-width - 10]]), 0)
        middle_vertical = l2.LineXY.from_location_angle(middle_pt, np.pi / 2)

        # Build the list of regions
        left_region_pts = [
            middle_pt,
            left_outer_edge.intersect_with(left_division),
            top_outer_edge.intersect_with(middle_vertical),
        ]
        right_region_pts = [
            middle_pt,
            right_outer_edge.intersect_with(right_division),
            top_outer_edge.intersect_with(middle_vertical),
        ]
        middle_region_pts = [middle_pt, right_region_pts[1], left_region_pts[1]]
        regions = [
            reg2.RegionXY.from_vertices(p2.Pxy.from_list(left_region_pts)),
            reg2.RegionXY.from_vertices(p2.Pxy.from_list(right_region_pts)),
            reg2.RegionXY.from_vertices(p2.Pxy.from_list(middle_region_pts)),
        ]

        # Limit regions by their overlapping area with the image bounds.
        for i in range(3):
            loop1 = regions[i].loops[0]
            loop2 = bounds.loops[0]
            loop_intersect = loop1.intersect_loop(loop2)
            regions[i] = reg2.RegionXY(loop_intersect)

        return three_points, regions

    def fit_background_plane(self, image: np.ndarray, solid_background_color: tuple[int]) -> tuple[int]:
        """
        Fits a 3-corner interpolated background to the given image,
        using the given solid_background_color as the starting point.

        Steps:
            1. subtract the current three-corner fit
            2. isolate no-spot pixels from spot pixels
            3. increase the value for tridrants that have >50% no-spot pixels
               visible, and decrease the value for tridrants that have <50%
               no-spot pixels visible
            4. repeat steps 1-3 until values stabilize or we reach a maximum
               number of iterations
        """
        max_iterations = 255
        noise_threshold = np.percentile(image, 40)
        (height, width), nchannels = it.dims_and_nchannels(image)

        # get the corner regions
        corners, corner_regions = self._get_three_point_regions(image)
        vx, vy = np.arange(width), np.arange(height)
        corner_masks: list[np.array] = []
        corner_bboxes_lrbt: list[list[int]] = []
        for corner_region in corner_regions:
            corner_mask = corner_region.loops[0].as_mask(vx, vy)
            corner_masks.append(corner_mask.astype(np.uint8))
            corner_bbox = corner_region.axis_aligned_bounding_box()
            corner_bboxes_lrbt.append([int(np.round(corner_bbox[i])) for i in range(4)])

        # build the corner data
        corner_colors_tl_tr_bm: list[tuple[int, int, int]] = []
        for channel in range(nchannels):
            channel_color = solid_background_color[channel]
            channel_colors = [channel_color for corner_index in range(3)]
            corner_colors_tl_tr_bm.append(channel_colors)
        _corner_tridrant_selector = lambda arr, lrbt: arr[lrbt[2] : lrbt[3], lrbt[0] : lrbt[1], ...]
        corner_tridrant_selectors = [
            lambda arr: _corner_tridrant_selector(arr, corner_bboxes_lrbt[0]),
            lambda arr: _corner_tridrant_selector(arr, corner_bboxes_lrbt[1]),
            lambda arr: _corner_tridrant_selector(arr, corner_bboxes_lrbt[2]),
        ]
        previously_seen_corner_colors = [copy.deepcopy(corner_colors_tl_tr_bm)]

        # build the initial fit
        three_corner_fit = self.build_background_image(image, solid_background_color=solid_background_color)

        # determine the no-spot pixels
        nospot_pixels_image = np.zeros_like(image)
        initial_sub_image = image.astype(np.int64) - three_corner_fit.astype(np.int64)
        nospot_pixels_image[np.where(initial_sub_image <= noise_threshold)] = 1

        for step_size in [5, 1]:
            for i in range(max_iterations):
                # step 1: subtract current fit
                residual_image = image.astype(np.int64) - three_corner_fit.astype(np.int64)

                colors_changed = False
                for corner_index, (corner_tridrant_selector, corner_mask) in enumerate(
                    zip(corner_tridrant_selectors, corner_masks)
                ):
                    corner_pixels_tridrant = corner_tridrant_selector(residual_image)
                    corner_mask_tridrant = corner_tridrant_selector(corner_mask)

                    for channel in range(nchannels):
                        corner_channel_pixels_tridrant = corner_pixels_tridrant
                        if nchannels > 1:
                            corner_channel_pixels_tridrant = corner_pixels_tridrant[:, :, channel]

                        # step 2: isolate no-spot pixels
                        nospot_pixels_image_tridrant = corner_tridrant_selector(nospot_pixels_image)
                        nospot_pixels = np.where(corner_mask_tridrant * nospot_pixels_image_tridrant)
                        corner_channel_pixels_tridrant = corner_channel_pixels_tridrant[nospot_pixels]
                        num_nospot_pixels = nospot_pixels[0].size

                        # step 3: increase or decrease values
                        ndark = np.where(corner_channel_pixels_tridrant < 0)[0].size
                        nzero = np.where(corner_channel_pixels_tridrant == 0)[0].size
                        nbright = np.where(corner_channel_pixels_tridrant > 0)[0].size

                        # no change? increase? decrease?
                        if (nbright == ndark) or (nzero >= num_nospot_pixels * 0.45):
                            pass
                        elif nbright > ndark:
                            if nbright > num_nospot_pixels * 0.4:
                                corner_colors_tl_tr_bm[channel][corner_index] += step_size
                                corner_colors_tl_tr_bm[channel][corner_index] = np.clip(
                                    corner_colors_tl_tr_bm[channel][corner_index], 0, 255
                                )
                                colors_changed = True
                        elif ndark > nbright:
                            if ndark > num_nospot_pixels * 0.4:
                                corner_colors_tl_tr_bm[channel][corner_index] -= step_size
                                corner_colors_tl_tr_bm[channel][corner_index] = np.clip(
                                    corner_colors_tl_tr_bm[channel][corner_index], 0, 255
                                )
                                colors_changed = True

                # step 4: repeat?
                if (colors_changed) and (corner_colors_tl_tr_bm not in previously_seen_corner_colors):
                    three_corner_fit = self.build_background_image(
                        image, background_plane_tl_tr_bm=corner_colors_tl_tr_bm
                    )
                    previously_seen_corner_colors.append(copy.deepcopy(corner_colors_tl_tr_bm))
                else:
                    break

        return corner_colors_tl_tr_bm

    def build_background_image(
        self,
        primary_image: np.ndarray,
        solid_background_color: tuple[int] = None,
        gradient_tl_tr_bl_br: tuple[tuple[int]] = None,
        background_plane_tl_tr_bm: tuple[tuple[int]] = None,
    ) -> np.ndarray:
        (height, width), nchannels = it.dims_and_nchannels(primary_image)
        ret = np.zeros_like(primary_image)

        if nchannels > 1:
            for channel in range(primary_image.shape[2]):
                channel_color = None if solid_background_color is None else tuple([solid_background_color[channel]])
                channel_gradient_tl_tr_bl_br = (
                    None if gradient_tl_tr_bl_br is None else tuple([gradient_tl_tr_bl_br[channel]])
                )
                channel_background_plane = (
                    None if background_plane_tl_tr_bm is None else tuple([background_plane_tl_tr_bm[channel]])
                )
                ret[:, :, channel] = self.build_background_image(
                    primary_image[:, :, channel], channel_color, channel_gradient_tl_tr_bl_br, channel_background_plane
                )
            return ret

        if background_plane_tl_tr_bm is not None:
            tl, tr, bm = background_plane_tl_tr_bm[0]
            lt.info(f"Building plane image for corner colors {tl=}, {tr=}, {bm=}")
            gradient_tl_tr_bl_br = [[tl, tr, 0, 0]]
            corners, corner_regions = self._get_three_point_regions(primary_image)
            tl_corner, tr_corner, bm_corner = corners[0], corners[1], corners[2]
            bl_corner, br_corner = p2.Pxy([[0], [height]]), p2.Pxy([[width], [height]])
            gradient_color_params: list[dict] = [
                {"gi": 3, "top": tl_corner, "opp": tr_corner, "tar": br_corner, "topc": tl, "oppc": tr},
                {"gi": 2, "top": tr_corner, "opp": tl_corner, "tar": bl_corner, "topc": tr, "oppc": tl},
            ]
            for gc in gradient_color_params:
                gradient_idx, top_corner, opposite_corner, target_corner, top_color, opposite_color = tuple(gc.values())
                connecting_line = l2.LineXY.from_two_points(top_corner, bm_corner)
                cross_line = l2.LineXY.from_two_points(opposite_corner, target_corner)
                cross_intersection = connecting_line.intersect_with(cross_line)
                cross_color_slope = (top_color - bm) / top_corner.distance(bm_corner)
                cross_color = (cross_color_slope * bm_corner.distance(cross_intersection)) + top_color
                target_color_slope = (opposite_color - cross_color) / opposite_corner.distance(cross_intersection)
                target_color = target_color_slope * bl_corner.distance(tr_corner) + opposite_color
                gradient_tl_tr_bl_br[0][gradient_idx] = int(np.round(target_color))
            return self.build_background_image(primary_image, gradient_tl_tr_bl_br=gradient_tl_tr_bl_br)
        elif gradient_tl_tr_bl_br is not None:
            height, width = it.dims_and_nchannels(primary_image)[0]
            x, y = np.array([0, width]), np.array([0, height])
            tl, tr, bl, br = gradient_tl_tr_bl_br[0]
            lt.info(f"Building gradient image for corner colors {tl=}, {tr=}, {bl=}, {br=}")
            data = np.array([[tl, tr], [bl, br]])
            interp = scipy.interpolate.RegularGridInterpolator((y, x), data)
            yg, xg = np.meshgrid(range(height), range(width), indexing="ij")
            ret = interp((yg, xg))
        elif solid_background_color is not None:
            ret[:, :] = solid_background_color[0]
        else:
            lt.error_and_raise(
                "Error in BackgroundColorSubtractionImageProcessor.build_background_image(): "
                + "one of the color, gradient or plane inputs is required, but all are None."
            )

        ret = np.round(np.clip(ret, 0, 255)).astype(primary_image.dtype)
        return ret

    def subtract_background(self, primary_image: np.ndarray, background_image: np.ndarray):
        # We convert the primary image to type int64 so that we have negative
        # values available. If left as a uint8, then subtracting below 0 would
        # cause values to wrap around, instead.
        background_image = background_image.astype(np.int64)
        new_primary_image = primary_image.astype(np.int64) - background_image
        new_primary_image = np.clip(new_primary_image, 0, np.max(primary_image))
        new_primary_image = new_primary_image.astype(primary_image.dtype)
        return new_primary_image

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray.copy()

        # find the background color
        background_colors = None
        background_plane = None
        background_gradients = None
        if self.color_detection_method == "constant":
            background_colors = self.constant_color_value
        elif self.color_detection_method == "solid":
            background_colors = self.find_solid_background_color(image)
        elif self.color_detection_method == "gradient":
            background_colors = self.find_solid_background_color(image)
            background_gradients = self.fit_four_corners(image, background_colors)
        elif self.color_detection_method == "plane":
            background_colors = self.find_solid_background_color(image)
            background_plane = self.fit_background_plane(image, background_colors)
        else:
            lt.error_and_raise(
                ValueError,
                "Error in BackgroundColorSubtractionImageProcessor._execute(): "
                + f"unknown color_detection_method {self.color_detection_method}",
            )

        # generate the background to be subtracted
        background_image = self.build_background_image(image, background_colors, background_gradients, background_plane)

        # subtract the background
        subtracted_image = self.subtract_background(image, background_image)

        ret = dataclasses.replace(operable, primary_image=CacheableImage.from_single_source(subtracted_image))

        if ImageType.NULL not in ret.supporting_images:
            supporting_images = copy.copy(operable.supporting_images)
            supporting_images[ImageType.NULL] = CacheableImage.from_single_source(background_image)
            ret = dataclasses.replace(ret, supporting_images=supporting_images)

        algorithm_images = copy.copy(operable.algorithm_images)
        if self not in algorithm_images:
            algorithm_images[self] = []
        algorithm_images[self].append(CacheableImage.from_single_source(background_image))
        algorithm_images[self].append(ret.primary_image)
        ret = dataclasses.replace(ret, algorithm_images=algorithm_images)

        return [ret]
