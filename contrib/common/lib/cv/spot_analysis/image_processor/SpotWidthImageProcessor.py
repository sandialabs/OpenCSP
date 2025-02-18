import copy
import dataclasses
import json

import cv2 as cv
import numpy as np

from contrib.common.lib.cv.annotations.SpotWidthAnnotation import SpotWidthAnnotation
from contrib.common.lib.cv.annotations.MomentsAnnotation import MomentsAnnotation
from opencsp.common.lib.cv.CacheableImage import CacheableImage
import opencsp.common.lib.cv.image_reshapers as ir
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImageProcessor,
)
import opencsp.common.lib.geometry.angle as geo_angle
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render_control.RenderControlSpotSize as rcss
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class SpotWidthImageProcessor(AbstractSpotAnalysisImageProcessor):
    """
    Calculates the spot width using a specified technique.
    """

    def __init__(self, spot_width_technique="fwhm", style: rcss.RenderControlSpotSize = None):
        """
        Parameters
        ----------
        spot_width_technique : str, optional
            The technique used to find the spot width, by default "fwhm". Must be one of the following:

            "average_radius_x2":
                First the pixels that are at half-maximum are found, then the
                centroid of those pixels is determined, and the average of
                half-maximum pixel distances from that centroid is produced as
                the average radius. Result is average_radius*2.
            "fwhm":
                First the pixels that are at half-maximum are found, then the
                centroid of those pixels is determined, then the maximum
                distance is found by doing a sweep of all angles through the
                centroid.
        style: RenderControlSpotSize, optional
            The style applied to the SpotWidthAnnotation. By default
            SpotWidthAnnotation.default().
        """
        super().__init__()

        # validate input
        allowed_techniques = ["average_radius_x2", "fwhm"]
        if spot_width_technique not in allowed_techniques:
            lt.error_and_raise(
                ValueError,
                "Error in SpotWithImageProcessor(): "
                + f"spot_width_technique must be one of {allowed_techniques}, but is {spot_width_technique}",
            )

        self.spot_width_technique = spot_width_technique
        self.style = style

    def locate_half_max_pixels(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns the Y and X coordinates for half-maximum pixels in the given image."""
        maxval = np.max(image)
        half_max = int(np.round(maxval / 2))
        return np.where(image == half_max)

    def find_centroid(self, coords: tuple[np.ndarray, np.ndarray]) -> p2.Pxy:
        # create an image with 1s at the specified coordinates
        y_max = np.max(coords[0])
        x_max = np.max(coords[1])
        tmp_img = np.zeros((y_max + 1, x_max + 1), dtype=np.uint8)
        tmp_img[coords] = 1

        # get the moments
        moments = cv.moments(tmp_img)
        annotation = MomentsAnnotation(moments)

        return annotation.centroid

    def average_radius_x2(self, image: np.ndarray) -> tuple[p2.Pxy, float]:
        half_max_pixel_coords = self.locate_half_max_pixels(image)
        half_max_pixel_locations = p2.Pxy.from_numpy_coords(half_max_pixel_coords)
        centroid = self.find_centroid(half_max_pixel_coords)
        radii = half_max_pixel_locations.distance(centroid)
        average_radius = np.average(radii)

        lt.debug(f"Full-width-half-maximum pixels centroid is at {centroid.astuple()}")
        lt.debug(f"Full-width-half-maximum average radius is {average_radius}")

        return centroid, average_radius * 2

    def _find_closest_coordinate_to_angle(self, target_angle: float, angles: np.ndarray, coordinates: p2.Pxy):
        angle_diffs = np.abs(angles - target_angle)
        angle_diffs = geo_angle.normalize(angle_diffs)
        idx = np.argmin(angle_diffs)
        return coordinates[idx]

    def color_angles(
        self, image: np.ndarray, yx_coords: tuple[np.ndarray, np.ndarray], xy_pts: p2.Pxy, angles: tuple[float]
    ):
        # initialize the circle image, magnitude image, and radians image
        ref_hsv = np.zeros_like(image)

        # generate the magnitudes and colors
        for pnt_idx in range(len(xy_pts)):
            x = int(xy_pts[pnt_idx].x[0])
            y = int(xy_pts[pnt_idx].y[0])
            ang = angles[pnt_idx]

            # convert to HSV space
            hue = int(ang / (2 * np.pi) * 179)
            ref_hsv[y, x, :] = (hue, 255, 255)

        # convert to RGB space
        circle_rgb = cv.cvtColor(ref_hsv, cv.COLOR_HSV2RGB)

        # copy onto the return image
        ret = np.copy(image)
        ret[yx_coords] = circle_rgb[yx_coords]

        return ret

    def fwhm(self, image_name: str, image: np.ndarray) -> tuple[p2.Pxy, float, float, float, np.ndarray]:
        half_max_pixel_coords = self.locate_half_max_pixels(image)
        half_max_pixel_locations = p2.Pxy.from_numpy_coords(half_max_pixel_coords)
        centroid = self.find_centroid(half_max_pixel_coords)
        angles = half_max_pixel_locations.angle_from(centroid)

        # Create the explainer image.
        algorithm_image = np.copy(image)
        algorithm_image = ir.nchannels_reshaper(algorithm_image, 3)

        # sort coordinates by their angle
        sorted_order = np.argsort(angles)
        angles = angles[sorted_order]
        half_max_pixel_locations = p2.Pxy(
            [half_max_pixel_locations.x[sorted_order], half_max_pixel_locations.y[sorted_order]]
        )
        new_angles = half_max_pixel_locations.angle_from(centroid)
        if not np.array_equal(angles, new_angles):
            lt.error_and_raise(RuntimeError, "")

        # find the full width
        angle_data: dict[float, tuple[float, float, p2.Pxy]] = {}
        coord_pairs_seen: list[tuple[p2.Pxy, p2.Pxy]] = []
        has_warned, has_infoed = False, False
        for i in range(len(half_max_pixel_locations)):
            coord1 = half_max_pixel_locations[i]
            coord2 = self._find_closest_coordinate_to_angle(
                coord1.angle_from(centroid)[0] + np.pi, angles, half_max_pixel_locations
            )
            if ((coord1, coord2) in coord_pairs_seen) or ((coord2, coord1) in coord_pairs_seen):
                continue
            coord_pairs_seen.append((coord1, coord2))

            angle_diff = np.abs(coord1.angle_from(centroid)[0] - coord2.angle_from(centroid)[0])
            info_diff_epsilon = np.pi / 4
            warn_diff_epsilon = np.pi / 3
            err_diff_epsilon = np.pi / 2
            err_msg = (
                f"SpotWidthImageProcessor.fwhm() for image {image_name}: "
                + f"expected all coordinates to have an opposite coordinate at ~180 degrees ({np.pi} radians), "
                + f"but {angle_diff=} for coordinates {coord1.astuple()}, {coord2.astuple}"
            )
            if np.abs(angle_diff - np.pi) > err_diff_epsilon:
                lt.error("Error in " + err_msg)
            elif np.abs(angle_diff - np.pi) > warn_diff_epsilon:
                if not has_warned:
                    lt.warn("Warning in " + err_msg)
                    has_warned = True
            elif np.abs(angle_diff - np.pi) > info_diff_epsilon:
                if not has_infoed:
                    lt.info("In " + err_msg)
                    has_infoed = True

            width = coord1.distance(coord2)[0]
            angle = coord1.angle_from(centroid)[0]
            coord_center = ((coord1 - coord2) / 2) + coord2
            angle_data[angle] = (angle, width, coord_center)

            # Continue the algorithm image.
            # Draw lines between several example pairs of points.
            if i % int(len(half_max_pixel_locations) / 10) == 0:
                hue = angle / (2 * np.pi)
                rgb = color.Color.from_hsv(hue, 1, 1, "Coordinate Color", "Coordinate Color").rgb_255()
                algorithm_image = cv.line(
                    algorithm_image, (int(coord1.x[0]), int(coord1.y[0])), (int(coord2.x[0]), int(coord2.y[0])), rgb
                )

        long_axis_idx = np.argmax([width for ang, width, cc in angle_data.values()])
        long_axis_rotation = list(angle_data.keys())[long_axis_idx]
        _, long_axis_width, long_axis_center = angle_data[long_axis_rotation]

        # find the orthogonal width
        orthogonal_rotation = long_axis_rotation + np.pi / 2
        coord1 = self._find_closest_coordinate_to_angle(orthogonal_rotation, angles, half_max_pixel_locations)
        coord2 = self._find_closest_coordinate_to_angle(orthogonal_rotation + np.pi, angles, half_max_pixel_locations)
        orthogonal_axis_width = coord1.distance(coord2)[0]

        # Finish the algorithm image.
        # Highlight half-max pixels with a color for their angle.
        # Highlight the centroid in matplotib purple.
        algorithm_image = self.color_angles(algorithm_image, half_max_pixel_coords, half_max_pixel_locations, angles)
        centroid_coord = (int(centroid.x[0]), int(centroid.y[0]))
        algorithm_image = cv.circle(algorithm_image, centroid_coord, 10, color.plot_colors["purple"].rgb_255(), 2)

        return centroid, long_axis_width, long_axis_rotation, long_axis_center, orthogonal_axis_width, algorithm_image

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray
        annotations = copy.copy(operable.annotations)
        notes = copy.copy(operable.image_processor_notes)
        algorithm_images = copy.copy(operable.algorithm_images)

        if self.spot_width_technique == "average_radius_x2":
            centroid, spot_width = self.average_radius_x2(image)
            annotations.append(SpotWidthAnnotation(self.spot_width_technique, centroid, spot_width, style=self.style))
            notes.append(
                (self.name, json.dumps({"spot_width": spot_width, "spot_width_technique": self.spot_width_technique}))
            )
        elif self.spot_width_technique == "fwhm":
            centroid, spot_width, long_axis_rotation, long_axis_center, orthogonal_axis_width, algorithm_image = (
                self.fwhm(operable.best_primary_pathnameext, image)
            )
            annotations.append(
                SpotWidthAnnotation(
                    self.spot_width_technique,
                    centroid,
                    spot_width,
                    long_axis_rotation,
                    long_axis_center,
                    orthogonal_axis_width,
                    style=self.style,
                )
            )
            notes.append(
                (
                    self.name,
                    json.dumps(
                        {
                            "long_axis_center": long_axis_center.astuple(),
                            "long_axis_rotation": long_axis_rotation,
                            "orthogonal_axis_width": orthogonal_axis_width,
                            "spot_width": spot_width,
                            "spot_width_technique": self.spot_width_technique,
                        }
                    ),
                )
            )
            algorithm_image = CacheableImage.from_single_source(algorithm_image)
            if self not in algorithm_images:
                algorithm_images[self] = []
            algorithm_images[self].append(algorithm_image)

        ret = dataclasses.replace(operable, annotations=annotations, image_processor_notes=notes)

        return [ret]
