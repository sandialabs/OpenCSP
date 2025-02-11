import cv2 as cv
import numpy as np

import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.tool.log_tools as lt


class PerspectiveTransform:
    def __init__(self, pixel_coordinates: p2.Pxy | list[p2.Pxy], meters_coordinates: p2.Pxy | list[p2.Pxy]):
        """
        Parameters
        ----------
        pixel_coordinates : p2.Pxy | list[p2.Pxy]
            The coordinates of the pixels that mark the vertices of a region. If
            a quadrilatural, then these should be in top-left, top-right,
            bottom-right, bottom-left order.
        meters_coordinates : p2.Pxy | list[p2.Pxy]
            The coordinates that the pixels region should map to in real-world
            units of meters. If a quadrilatural, then these should be in
            top-left, top-right, bottom-right, bottom-left order.
        """
        # normalize input
        if isinstance(pixel_coordinates, list):
            pixel_coordinates = p2.Pxy.from_list(pixel_coordinates)
        if isinstance(meters_coordinates, list):
            meters_coordinates = p2.Pxy.from_list(meters_coordinates)

        # validate input
        if len(pixel_coordinates) != len(meters_coordinates):
            lt.error_and_raise(
                ValueError,
                "Error in PerspectiveTransform(): "
                + "there must be the same number of pixel and meter coordinates, "
                + f"but there are {len(pixel_coordinates)} (pixels) and {len(meters_coordinates)} (meters).",
            )

        # register inputs
        self.pixel_coordinates: p2.Pxy = pixel_coordinates
        """ Image coordinates in pixels, in clockwise order (tl, tr, br, bl) """
        self.meters_coordinates: p2.Pxy = meters_coordinates
        """ Real-world coordinates in meters, in clockwise order (tl, tr, br, bl) """

        # convert to millimeters for higher accuracy
        self.millimeters_coordinates = p2.Pxy(self.meters_coordinates.data * 1000.0)

        # transform matrices
        self.meters_to_pixels_transform: np.ndarray = None
        self.millimeters_to_pixels_transform: np.ndarray = None
        self.pixels_to_meters_transform: np.ndarray = None
        self.pixels_to_millimeters_transform: np.ndarray = None

        self._find_transforms()

    def _find_transforms(self):
        # get all the coordinates
        px_xy_coords: p2.Pxy = self.pixel_coordinates
        m_xy_coords: p2.Pxy = self.meters_coordinates
        mm_xy_coords: p2.Pxy = self.millimeters_coordinates

        # convert from a 2x4 array to a 4x2 array
        px_xy: np.ndarray = np.array([[px_xy_coords.x[i], px_xy_coords.y[i]] for i in range(4)])
        m_xy: np.ndarray = np.array([[m_xy_coords.x[i], m_xy_coords.y[i]] for i in range(4)])
        mm_xy: np.ndarray = np.array([[mm_xy_coords.x[i], mm_xy_coords.y[i]] for i in range(4)])

        # convert to opencv expected data type
        px_xy = px_xy.astype(np.float32)
        m_xy = m_xy.astype(np.float32)
        mm_xy = mm_xy.astype(np.float32)

        # find the transforms
        self.meters_to_pixels_transform = cv.getPerspectiveTransform(m_xy, px_xy)
        self.millimeters_to_pixels_transform = cv.getPerspectiveTransform(mm_xy, px_xy)
        self.pixels_to_meters_transform = cv.getPerspectiveTransform(px_xy, m_xy)
        self.pixels_to_millimeters_transform = cv.getPerspectiveTransform(px_xy, mm_xy)

    @property
    def width_meters(self) -> float:
        """
        The widest length of the meters coordinates. If a quadrilatural, then
        this is computed from the (tl,tr) and (bl,br) coordinate pairs.
        Otherwise the maximum x-distance between two coordinates is used.
        """
        measure_horizontal: bool = len(self.meters_coordinates) == 4
        return self._max_distance(self.meters_coordinates.x, horizontal_distance=measure_horizontal)

    @property
    def height_meters(self) -> float:
        """
        The tallest length of the meters coordinates. If a quadrilatural, then
        this is computed from the (tl,bl) and (tr,br) coordinate pairs.
        Otherwise the maximum y-distance between two coordinates is used.
        """
        measure_vertical: bool = len(self.meters_coordinates) == 4
        return self._max_distance(self.meters_coordinates.y, vertical_distance=measure_vertical)

    def _max_distance(
        self, data: np.ndarray, horizontal_distance: bool = False, vertical_distance: bool = False
    ) -> float:
        """
        Returns the maximum distance between two of the values in the given 1d
        data array. If horizontal_distance or vertical_distance is true, then it
        assumes that the data represents the tl, tr, br, bl corners of a
        rectangle.

        Parameters
        ----------
        data : np.ndarray
            1D array. Should have four values if horizontal_distance is true.
        horizontal_distance : bool
            True if data represents the four corners of a rectangle and you're
            retrieving the width. False otherwise.
        horizontal_distance : bool
            True if data represents the four corners of a rectangle and you're
            retrieving the height. False otherwise.

        Returns
        -------
        float
            The maximum distance between two of the values in data.
        """
        if horizontal_distance and vertical_distance:
            lt.error_and_raise(
                ValueError,
                "Error in PerspectiveTransform._max_distance(): "
                + "only one of horizontal_distance and vertical_distance can be true at the same time",
            )

        if horizontal_distance:
            # 0: tl, 1: tr, 2: br, 3: bl
            d1 = np.abs(data[0] - data[1])
            d2 = np.abs(data[2] - data[3])
            return np.max([d1, d2])

        elif vertical_distance:
            # 0: tl, 1: tr, 2: br, 3: bl
            d1 = np.abs(data[3] - data[0])
            d2 = np.abs(data[2] - data[1])
            return np.max([d1, d2])

        else:
            distances: list[float] = []
            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    distances.append(np.abs(data[i] - data[j]))
            return np.max(distances)

    def transform_image(self, image: np.ndarray, buffer_width_px: int = 0, full_image=False) -> np.ndarray:
        if buffer_width_px != 0:
            xs = self.pixel_coordinates.x.copy()
            ys = self.pixel_coordinates.y.copy()
            xs[[0, 3]] -= buffer_width_px
            xs[[1, 2]] += buffer_width_px
            ys[[0, 1]] -= buffer_width_px
            ys[[2, 3]] += buffer_width_px
            buffer_coords_px = p2.Pxy((xs, ys))
            transform = PerspectiveTransform(buffer_coords_px, self.meters_coordinates)
            ret = transform.transform_image(image)

        elif not full_image:
            width, height = self.width_meters * 1000.0, self.height_meters * 1000.0
            ret = cv.warpPerspective(image, self.pixels_to_millimeters_transform, (int(width), int(height)))

        else:
            xs = [0, image.shape[1], image.shape[1], 0]
            ys = [0, 0, image.shape[0], image.shape[0]]
            full_pixels_coordinates = p2.Pxy((xs, ys))
            full_meters_coordinates = self.pixels_to_meters(full_pixels_coordinates)
            transform = PerspectiveTransform(full_pixels_coordinates, full_meters_coordinates)
            ret = transform.transform_image(image)

        return ret

    def _a_to_b(self, a_coordinate: p2.Pxy, transform: np.ndarray) -> p2.Pxy:
        a_x_vals, a_y_vals = a_coordinate.x, a_coordinate.y
        b_x_vals, b_y_vals = np.zeros_like(a_x_vals), np.zeros_like(a_y_vals)

        for i in range(len(a_x_vals)):
            ti_b_x, ti_b_y, ti = np.matmul(transform, np.array([a_x_vals[i], a_y_vals[i], 1]))
            b_x_vals[i], b_y_vals[i] = ti_b_x / ti, ti_b_y / ti

        return p2.Pxy((b_x_vals, b_y_vals))

    def pixels_to_meters(self, pixel_coordinate: p2.Pxy) -> p2.Pxy:
        meters_coordinate = self._a_to_b(pixel_coordinate, self.pixels_to_meters_transform)
        return p2.Pxy(meters_coordinate.data)

    def meters_to_pixels(self, meter_coordinate: p2.Pxy) -> p2.Pxy:
        meters_coordinate = p2.Pxy(meter_coordinate.data)
        return self._a_to_b(meters_coordinate, self.meters_to_pixels_transform)
