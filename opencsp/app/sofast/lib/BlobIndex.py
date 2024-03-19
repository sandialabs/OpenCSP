from typing import Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np

from opencsp.common.lib.geometry.Vxy import Vxy


class BlobIndex:
    """Class containing blob indexing algorithms to assign indices to blobs in a rough grid pattern.
    X/Y axes correspond to image axes; +x is to right, +y is down. Class takes in points (in units
    of pixels) that have been previously found with a blob detector and attempts to assign all found
    xy pixel points with a blob index.

    Attributes
    ----------
    search_thresh : float

    search_perp_axis_ratio : float
        Ratio of point distances: (perpendicular to axis) / (along axis) used to
        search for points.
    apply_filter : bool
        To filter bad points (experimental)
    verbose : bool
        Deprecated
    """

    def __init__(
        self, points: Vxy, x_min: int, x_max: int, y_min: int, y_max: int
    ) -> 'BlobIndex':
        """Instantiates BlobIndex class

        Parameters
        ----------
        points : Vxy
            Points of all possible blobs found with a blob detector, pixels
        origin : Vxy
            Origin point assumed to have blob index coordinates (0, 0), pixels
        x_min/x_max/y_min/y_max : int
            Expected min/max of blob indices in x/y directions
        """
        self._points = points

        self._num_pts = len(points)
        self._point_indices = np.arange(self._num_pts)

        self._idx_x = np.zeros(self._num_pts) * np.nan
        self._idx_y = np.zeros(self._num_pts) * np.nan
        self._is_assigned = np.zeros(self._num_pts, dtype=bool)
        self._neighbor_dists = (
            np.zeros((self._num_pts, 4)) * np.nan
        )  # left, right, up, down

        self.search_thresh = 5.0  # pixels
        self.search_perp_axis_ratio = 3.0
        self.apply_filter = False
        self.verbose = True

        self._offset_x = -x_min  # index
        self._offset_y = -y_min  # index
        idx_x_vec = np.arange(x_min, x_max + 1)  # index
        idx_y_vec = np.arange(y_min, y_max + 1)  # index
        self._idx_x_mat, self._idx_y_mat = np.meshgrid(idx_x_vec, idx_y_vec)  # index
        self._points_mat = (
            np.zeros((y_max - y_min + 1, x_max - x_min + 1, 2)) * np.nan
        )  # pixels
        self._point_indices_mat = (
            np.zeros((y_max - y_min + 1, x_max - x_min + 1)) * np.nan  # index
        )

    def _get_assigned_point_indices(self) -> np.ndarray[int]:
        """Returns found point indices"""
        return self._point_indices[self._is_assigned]

    def _get_unassigned_point_indices(self) -> np.ndarray[int]:
        """Returns not found point indices"""
        return self._point_indices[np.logical_not(self._is_assigned)]

    def _get_assigned_points(self) -> Vxy:
        """Returns assigned xy points"""
        return self._points[self._is_assigned]

    def _get_unassigned_points(self) -> Vxy:
        """Returns unassigned xy points"""
        return self._points[np.logical_not(self._is_assigned)]

    def _nearest_unassigned_idx_from_xy_point(self, point: Vxy) -> tuple[int, float]:
        """Returns the point index and distance of unassigned point nearest to given xy point"""
        points = self._get_unassigned_points()
        idxs = self._get_unassigned_point_indices()
        dists = (points - point).magnitude()
        idx = np.argmin(dists)

        return idxs[idx], dists[idx]

    def _nearest_unassigned_idx_from_xy_point_direction(
        self, pt_cur: Vxy, pt_exp: Vxy
    ) -> tuple[bool, tuple[int, float]]:
        """Returns the point index and distance of unassigned point nearest to given xy
        point in direction form current to expected point.

        Parameters
        ----------
        pt_cur : Vxy
            Current point, pixels
        pt_exp : Vxy
            Expected next point, pixels

        Returns
        -------
        tuple[int, float]
            Point index (indexing self._points) and distance from expected point
        """
        points = self._get_unassigned_points()
        idxs = self._get_unassigned_point_indices()
        # Calculate xy deltas for expected/current point
        points_rel = points - pt_cur  # Vectors, current point to all points
        v_search = pt_exp - pt_cur  # Vector, from current point to expected point
        v_perp = v_search.rotate(
            np.array([[0, -1], [1, 0]])
        )  # Vector, perpendicular to search axis
        dists_axis = v_search.dot(points_rel)  # Distance of points along search axis
        dists_perp = np.abs(v_perp.dot(points_rel))  # Distance of points from line
        # Make mask of valid points
        mask = np.logical_and(
            dists_axis > 0, dists_perp / dists_axis <= self.search_perp_axis_ratio
        )
        # Check there are points to find
        if mask.sum() == 0:
            return False, (None, None)
        # Find nearest point to current point
        idx = np.argmin(points_rel[mask].magnitude())
        point = points[mask][idx]  # pixels
        dist_exp = (point - pt_exp).magnitude()[0]
        idx = idxs[mask][idx]

        return True, (idx, dist_exp)

    def _point_index_from_xy_index(self, idx_x: int, idx_y: int) -> tuple[bool, int]:
        """Returns point index from xy index"""
        idx = self._point_indices_mat[idx_y + self._offset_y, idx_x + self._offset_x]
        return (not np.isnan(idx)), int(idx)

    def _assign(self, idx_pt: int, idx_x: int, idx_y: int) -> None:
        """Assigns given blob index an xy index

        Parameters
        ----------
        idx_pt : int
            Index of point (indexing self._points)
        idx_x : int
            X blob index
        idx_y : int
            Y blob index
        """
        # Assign vectors
        self._idx_x[idx_pt] = idx_x
        self._idx_y[idx_pt] = idx_y
        self._is_assigned[idx_pt] = True
        # Assign matrices
        self._points_mat[idx_y + self._offset_y, idx_x + self._offset_x] = self._points[
            idx_pt
        ].data.squeeze()
        self._point_indices_mat[idx_y + self._offset_y, idx_x + self._offset_x] = idx_pt

        if self.verbose:
            print(f'Blob number {idx_pt:d} was assigned ({idx_x:d}, {idx_y:d})')

    def _unassign(self, idx_pt: int) -> None:
        """Unassigns a point index

        Parameters
        ----------
        idx_pt : int
            Index of point (indexing self._points)
        """
        # Unassign matrices
        idx_mat_x = self._idx_x[idx_pt] + self._offset_x
        idx_mat_y = self._idx_y[idx_pt] + self._offset_y

        if np.isnan(idx_mat_x) or np.isnan(idx_mat_y):
            return

        self._points_mat[int(idx_mat_y), int(idx_mat_x)] = self._points[
            idx_pt
        ].data.squeeze()
        self._point_indices_mat[int(idx_mat_y), int(idx_mat_x)] = idx_pt

        # Unassign vectors
        self._idx_x[idx_pt] = np.nan
        self._idx_y[idx_pt] = np.nan
        self._is_assigned[idx_pt] = False

        if self.verbose:
            print(f'Blob number {idx_pt:d} was unassigned')

    def _assign_center(self, pt_origin: Vxy) -> None:
        """Assigns the center point to (0, 0)

        Parameters
        ----------
        pt_origin : Vxy
            Location of origin, pixels
        """
        idx, dist = self._nearest_unassigned_idx_from_xy_point(pt_origin)
        if dist > self.search_thresh:
            warn(
                f'Assigning point {idx:d} to index (0, 0) resulted in {dist:.2f} pixels error.'
            )
        self._assign(idx, 0, 0)

    def _find_nearest_in_direction(
        self, idx_pt: int, direction: Literal['right', 'left', 'up', 'down']
    ) -> tuple[int, int, int]:
        """Finds the directly nearest point index to given point
        in left, right, up, down direction. Can return already found points

        Parameters
        ----------
        idx_pt : int
            Index of point (indexing self._points) from which to start searching
        direction : Literal['right', 'left', 'up', 'down']
            The direction to search from starting point

        Returns
        -------
        point_index
            Index of found point (indexing self._points)
        index_x
            X blob index
        index_y
            Y blob index
        """
        # Find possible points
        unassigned_points = self._get_unassigned_points()
        unassigned_deltas = unassigned_points - self._points[idx_pt]
        unassigned_idxs = self._get_unassigned_point_indices()
        idx_x = self._idx_x[idx_pt]
        idx_y = self._idx_y[idx_pt]

        if direction == 'right':
            mask = np.logical_and(
                unassigned_deltas.x > 0,
                unassigned_deltas.x > (2 * np.abs(unassigned_deltas.y)),
            )
            idx_x_out = idx_x + 1
            idx_y_out = idx_y
        elif direction == 'left':
            mask = np.logical_and(
                unassigned_deltas.x < 0,
                -unassigned_deltas.x > (2 * np.abs(unassigned_deltas.y)),
            )
            idx_x_out = idx_x - 1
            idx_y_out = idx_y
        elif direction == 'up':
            mask = np.logical_and(
                unassigned_deltas.y > 0,
                unassigned_deltas.y > (2 * np.abs(unassigned_deltas.x)),
            )
            idx_x_out = idx_x
            idx_y_out = idx_y + 1
        # Down
        elif direction == 'down':
            mask = np.logical_and(
                unassigned_deltas.y < 0,
                -unassigned_deltas.y > (2 * np.abs(unassigned_deltas.x)),
            )
            idx_x_out = idx_x
            idx_y_out = idx_y - 1

        # Find closest point
        possible_idxs = unassigned_idxs[mask]
        possible_pts = unassigned_points[mask]
        dists = (possible_pts - self._points[idx_pt]).magnitude()
        idx_out = np.argmin(dists)
        return possible_idxs[idx_out], int(idx_x_out), int(idx_y_out)

    def _num_unassigned(self) -> int:
        """Returns number of unassigned points (referencing self._points)"""
        return np.logical_not(self._is_assigned).sum()

    def _find_3x3_center_block(self) -> None:
        """Finds the center 3x3 block around center point using nearest in direction method
        a  b  c
        d  e  f
        g  h  i
        """
        ret, idx_e = self._point_index_from_xy_index(0, 0)
        if not ret:
            warn('Could not find 3x3 center block. Could not find point index (0, 0).')
        # Right
        idx_f, x, y = self._find_nearest_in_direction(idx_e, 'right')
        self._assign(idx_f, x, y)
        # Left
        idx_d, x, y = self._find_nearest_in_direction(idx_e, 'left')
        self._assign(idx_d, x, y)
        # Up
        idx_b, x, y = self._find_nearest_in_direction(idx_e, 'up')
        self._assign(idx_b, x, y)
        # Down
        idx_h, x, y = self._find_nearest_in_direction(idx_e, 'down')
        self._assign(idx_h, x, y)
        # Up/right
        idx_c, x, y = self._find_nearest_in_direction(idx_f, 'up')
        self._assign(idx_c, x, y)
        # Up/left
        idx_a, x, y = self._find_nearest_in_direction(idx_d, 'up')
        self._assign(idx_a, x, y)
        # Down/right
        idx_i, x, y = self._find_nearest_in_direction(idx_f, 'down')
        self._assign(idx_i, x, y)
        # Down/left
        idx_g, x, y = self._find_nearest_in_direction(idx_d, 'down')
        self._assign(idx_g, x, y)

    def _extend_data(self, direction: Literal['x', 'y'], step: Literal[1, -1]) -> None:
        """Extends found blob rows/collumns in given direction

        Steps in the given axis, a or b

        Parameters
        ----------
        direction : Literal['x', 'y']
            Axis to search
        step : Literal[1, -1]
            Direction to search
            -  1 = right/down
            - -1 = left/up
        """
        if step not in [-1, 1]:
            raise ValueError(f'Step must be -1 or 1, not {step}')

        if direction == 'x':
            idxs_a = self._idx_y
            idxs_b = self._idx_x
        elif direction == 'y':
            idxs_a = self._idx_x
            idxs_b = self._idx_y
        else:
            raise ValueError(
                f'Given "direction" must be either "x" or "y", not {direction}'
            )

        # Step through direction
        # TODO Can speed up with matrix data storage
        for idx_a in np.unique(idxs_a[self._get_assigned_point_indices()]):
            # Get points on axis
            mask = idxs_a == idx_a
            pts = self._points[mask]  # points on axis
            is_b = idxs_b[mask]  # indices of points on axis
            # Step through all points on axis
            for i_b in is_b:
                if not i_b + step in is_b:  # If adjacent point is not assigned, find it
                    idx_b_prev = i_b - step  # Index used for slope calc
                    if idx_b_prev in is_b:  # If history exists, find points
                        for idx_b_next in range(500):
                            # First iteration, use previously assigned points
                            if idx_b_next == 0:
                                pt_cur = pts[is_b == i_b]
                                pt_prev = pts[is_b == idx_b_prev]
                                if (len(pt_cur) > 1) or (len(pt_prev) > 1):
                                    raise ValueError(
                                        f'Point index {idx_a:.0f}, {i_b:.0f} '
                                        'was assigned more than once. '
                                        'Try tightening dot search settings.'
                                    )
                            else:  # Next iterations, use new points
                                pt_prev = pt_cur
                                pt_cur = self._points[idx_new]
                            # Calculate deltas
                            pt_exp = self._exp_pt_from_pt_pair(pt_cur, pt_prev)
                            success, (idx_new, dist) = (
                                self._nearest_unassigned_idx_from_xy_point_direction(
                                    pt_cur, pt_exp
                                )
                            )
                            if not success:
                                break
                            # Assign point
                            if dist < self.search_thresh:
                                if direction == 'x':
                                    idx_x = int(i_b + step + (idx_b_next * step))
                                    idx_y = int(idx_a)
                                else:
                                    idx_x = int(idx_a)
                                    idx_y = int(i_b + step + (idx_b_next * step))
                                self._assign(idx_new, idx_x, idx_y)
                            else:
                                break

    def _exp_pt_from_pt_pair(self, pt_cur: Vxy, pt_prev: Vxy) -> Vxy:
        """Calculates the expected point from a given current and previous point pair

        Parameters
        ----------
        pt_cur : Vxy
            Current point, pixels
        pt_prev : Vxy
            Previous point, pixels

        Returns
        -------
        Vxy
            Refined expected point location
        """
        del_y = (pt_cur.y - pt_prev.y)[0]  # pixels
        del_x = (pt_cur.x - pt_prev.x)[0]  # pixels
        return pt_cur + Vxy((del_x, del_y))  # pixels

    def _filter_bad_points(self) -> None:
        """Filters erroneous assigned points"""
        del_1_x = np.diff(self._points_mat, axis=1)
        del_1_y = np.diff(self._points_mat, axis=0)
        del_2_x = np.diff(del_1_x, axis=1)
        del_2_y = np.diff(del_1_y, axis=0)

        thresh = 3
        ny = self._points_mat.shape[0]
        nx = self._points_mat.shape[1]
        mask_bad_pixels_x = np.abs(del_2_x) > thresh
        mask_bad_pixels_y = np.abs(del_2_y) > thresh

        # Calculate mask of bad pixels using x and y derivatives
        mask_bad_pixels_x = np.concatenate(
            (np.zeros((ny, 2, 2), dtype=bool), mask_bad_pixels_x), axis=1
        )
        mask_bad_pixels_y = np.concatenate(
            (np.zeros((2, nx, 2), dtype=bool), mask_bad_pixels_y), axis=0
        )

        # Combine into one mask
        mask_bad_pixels = (mask_bad_pixels_x + mask_bad_pixels_y).max(2)

        # Unassign bad points
        idxs_bad_pts = self._point_indices_mat[mask_bad_pixels].astype(int)
        for idx in idxs_bad_pts:
            self._unassign(idx)

    def run(self, pt_origin: Vxy) -> None:
        """Runs blob indexing sequence

        Parameters
        ----------
        pt_origin : Vxy
            Location of origin point with blob index of (0, 0), pixels
        """
        # Assign center point
        self._assign_center(pt_origin)
        # Find 3x3 core point block
        self._find_3x3_center_block()
        # Extend rows
        prev_num_unassigned = self._num_unassigned()
        for idx in range(100):
            self._extend_data('x', -1)
            self._extend_data('x', 1)
            if self.apply_filter:
                self._filter_bad_points()
            self._extend_data('y', -1)
            self._extend_data('y', 1)
            if self.apply_filter:
                self._filter_bad_points()

            # Check if no progress has been made
            cur_num_unassigned = self._num_unassigned()
            if prev_num_unassigned == cur_num_unassigned:
                if self.verbose:
                    print(f'All possible points found in {idx + 1:d} iterations.')
                break
            else:
                prev_num_unassigned = cur_num_unassigned

    def plot_points_labels(self, labels: bool = False) -> None:
        """Plots points and labels

        Parameters
        ----------
        labels : bool, optional
            To include xy blob index labels, by default False
        """
        plt.scatter(*self._points[self._is_assigned].data, color='blue')
        if labels:
            for x, y, pt in zip(
                self._idx_x[self._is_assigned],
                self._idx_y[self._is_assigned],
                self._points[self._is_assigned],
            ):
                plt.text(*pt.data, f'({x:.0f}, {y:.0f})')

    def plot_points_connections(self, labels: bool = False) -> None:
        """Plots points and connections for rows/collumns

        Parameters
        ----------
        labels : bool, optional
            To add point index (indexing self._points) labels, by default False
        """
        for row_idx in np.unique(self._idx_y):
            # Get all points in row
            mask_all = self._idx_y == row_idx
            pts = self._points[mask_all]
            idxs_pts = self._point_indices[mask_all]
            xs = pts.x
            ys = pts.y
            idxs = np.argsort(xs)
            plt.plot(xs[idxs], ys[idxs], marker='.', alpha=0.5)

            # Add point indices
            if labels:
                for x, y, idx in zip(xs[idxs], ys[idxs], idxs_pts[idxs]):
                    plt.text(x, y, str(idx))

    def get_data(self) -> tuple[Vxy, Vxy]:
        """Returns found points and indices

        Returns
        -------
        points : Vxy
            Lenght N vector, located points xy locations, pixels
        indices_xy : Vxy
            Length N vector, located points xy blob indices, int
        """
        # Get points as vectors
        x_pts_mat = self._points_mat[..., 0]
        y_pts_mat = self._points_mat[..., 1]
        x_pts = x_pts_mat.flatten()
        y_pts = y_pts_mat.flatten()

        # Get mask
        mask_unassigned = np.logical_or(np.isnan(x_pts), np.isnan(y_pts))
        mask_assigned = np.logical_not(mask_unassigned)

        # Get indices as vectors
        idx_x_mat = self._idx_x_mat
        idx_y_mat = self._idx_y_mat
        idx_x = idx_x_mat.flatten()
        idx_y = idx_y_mat.flatten()

        indices = Vxy((idx_x[mask_assigned], idx_y[mask_assigned]), int)
        points = Vxy((x_pts[mask_assigned], y_pts[mask_assigned]))
        return points, indices

    def get_data_mat(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns found points and indices in matrix form

        Returns
        -------
        points : Vxy
            Lenght N vector, located points xy locations, pixels
        indices_xy : Vxy
            Length N vector, located points xy blob indices, int
        """
        # Get indices as vectors
        idx_x_mat = self._idx_x_mat
        idx_y_mat = self._idx_y_mat
        idx_mat = np.concatenate((idx_x_mat[..., None], idx_y_mat[..., None]), 2)

        return self._points_mat.copy(), idx_mat
