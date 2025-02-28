"""Find the four corners of Aruco markers in an image, using OpenCV. The
upper-left corner is the "origin" of the marker (idx=0). The ImageMarker
class can hold information about just the origin point of the markers, or
it can be converted to a four point model which uses all the corners. By default,
the model loads an image using the one point model. In the one point model, the
point ID is the aruco marker ID. In the four point model, the point ID is the
Aruco marker ID * 4 plus the corner index (ranging from 0 to 4).
"""

from warnings import warn

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from opencsp.common.lib.camera.Camera import Camera
import opencsp.common.lib.photogrammetry.photogrammetry as ph
from opencsp.common.lib.tool.hdf5_tools import save_hdf5_datasets
import opencsp.common.lib.tool.log_tools as lt


class ImageMarker:
    """Class to hold images of Aruco markers. Contains methods
    to process locations of Aruco markers."""

    def __init__(self, image: ndarray, point_ids: ndarray[int], pts_im_xy: ndarray, img_id: int, camera: Camera):
        """
        Instantiates ImageMarker class.

        Parameters
        ----------
        image : ndarray
            2D image array.
        point_ids : ndarray
            1d array of point IDs.
        pts_im_xy : ndarray
            Nx2 array of point locations in image.
        img_id : int
            ID of image.
        camera : opencsp.common.lib.camera.Camera.Camera
            Camera object of camera that captured image.
        """
        # Perform checks
        if point_ids.size != pts_im_xy.shape[0]:
            raise ValueError("Length of given point_ids and pts_im_xy must be equal.")
        if np.ndim(pts_im_xy) != 2 or pts_im_xy.shape[1] != 2:
            raise ValueError("Points must be Nx2 array.")
        if not isinstance(point_ids.dtype.type, type(np.integer)):
            raise TypeError("Input IDs dtype must be int")

        # Save data
        self.image = image
        self.img_id = img_id
        self.camera = camera
        self.four_corner_model = False

        # Image point data
        self.point_ids = point_ids
        self.pts_im_xy = pts_im_xy
        self.located_markers_mask = np.zeros(self.pts_im_xy.shape[0], dtype=bool)
        self.pts_obj_xyz = np.zeros((self.num_markers, 3)) * np.nan

        # Camera pose data
        self.pose_known = False
        self.rvec = np.zeros(3) * np.nan
        self.tvec = np.zeros(3) * np.nan

    def __repr__(self):
        return f"Image {self.img_id:d}"

    def plot_image_with_points(self) -> None:
        """Plots captured image with image point and reprojected point locations"""
        # Calculate reprojected points
        pts_obj_known = self.pts_obj_xyz[self.located_markers_mask]
        pts_reproj = self.camera.project_mat(pts_obj_known, self.rvec, self.tvec)  # Nx2 points

        # Plot
        ax = plt.axes()
        ax.imshow(self.image)
        ax.scatter(*pts_reproj.T, marker="o", facecolor="none", edgecolor="red", label="Reprojected")
        ax.scatter(*self.pts_im_xy.T, marker=".", color="blue", label="Image points")
        ax.legend()

    @classmethod
    def load_aruco_origin(cls, file: str, img_id: int, camera: Camera, **kwargs) -> "ImageMarker":
        """Loads an image file, finds Aruco markers, saves the origin point.

        Parameters
        ----------
        file : str
            File path to image
        img_id : int
            Image index to save to image.
        camera : opencsp.common.lib.camera.Camera.Camera
            Calibrated camera object

        Returns
        -------
        ImageMarker

        """
        # Load image
        img_gray = ph.load_image_grayscale(file)

        # Find aruco markers
        ids_marker, pts_list = ph.find_aruco_marker(img_gray, **kwargs)

        # Save only origin point
        pts_mat = np.vstack([pts[0] for pts in pts_list])

        return cls(img_gray, ids_marker, pts_mat, img_id, camera)

    def convert_to_four_corner(self, **kwargs) -> None:
        """Converts from using only origin point to all four marker corners"""
        if self.four_corner_model:
            warn("Image is already a four corner model", UserWarning, stacklevel=2)
            return
        self.four_corner_model = True

        # Find all aruco corners in image
        point_ids, pts_list = ph.find_aruco_marker(self.image, **kwargs)

        # Concatenate all points
        pts_im_xy = np.vstack(pts_list)
        num_markers = len(point_ids)

        point_ids = np.array(point_ids) * 4
        point_ids = np.repeat(point_ids, 4)
        point_ids += np.array([0, 1, 2, 3] * num_markers)

        # Update
        self.point_ids = point_ids
        self.pts_im_xy = pts_im_xy

        mask = np.array([1, 0, 0, 0] * num_markers)
        self.located_markers_mask = (np.repeat(self.located_markers_mask, 4) * mask).astype(bool)

        mask = np.array([1, np.nan, np.nan, np.nan] * num_markers)
        self.pts_obj_xyz = np.repeat(self.pts_obj_xyz, 4, axis=0) * mask[:, None]

    def attempt_calculate_pose(self) -> int:
        """Calculates pose of camera if enough points in image are located

        Returns
        -------
        int
            0=already known, 1=calculated successfully, -1=not calculated successfully
        """
        # Check if pose already known
        if self.pose_known:
            return 0

        # Check if too few points to calculate pose
        if self.located_markers_mask.sum() <= 5:
            lt.debug(f"Camera pose {self.img_id:d} has too few points to solve")
            return -1

        # Get object and image points
        pts_obj = self.located_marker_points_object
        pts_img = self.located_marker_points_image

        # Use SolvePNP
        ret, rvec, tvec = cv.solvePnP(pts_obj, pts_img, self.camera.intrinsic_mat, self.camera.distortion_coef)
        rvec: ndarray = rvec.squeeze()
        tvec: ndarray = tvec.squeeze()

        # Check if CV succeeded
        if not ret:
            lt.debug("Camera pose {self.img_id:d} did not solve properly")
            return -1
        lt.debug(f"Camera pose {self.img_id:d} solved")

        # Check if pose is valid
        valid = ph.valid_camera_pose(self.camera, rvec, tvec, pts_img, pts_obj)
        if not valid:
            lt.debug(f"Camera pose {self.img_id:d} not valid")
            return -1

        # Pose is valid
        self.pose_known = True
        self.set_pose(rvec, tvec)
        return 1

    def set_point_id_located(self, id_: int, pt: ndarray) -> None:
        """Sets given point ID as located_markers_mask"""
        mask = self.point_ids == id_
        self.located_markers_mask[mask] = True
        self.pts_obj_xyz[mask] = pt

    def set_point_id_unlocated(self, id_: int) -> None:
        """Sets given point ID as unlocated"""
        mask = self.point_ids == id_
        self.located_markers_mask[mask] = False
        self.pts_obj_xyz[mask] = np.zeros(3) * np.nan

    def set_pose(self, rvec: ndarray, tvec: ndarray) -> None:
        """Sets the pose of the camera known

        Parameters
        ----------
        rvec : ndarray
            Shape (3,) rotation vector
        tvec : ndarray
            Shape (3,) translation vector
        """
        lt.debug(f"Camera pose {self.img_id:d} set")
        self.pose_known = True
        self.rvec = rvec
        self.tvec = tvec

    def unset_pose(self) -> None:
        """Removes the pose of the camera"""
        lt.debug(f"Camera pose {self.img_id:d} unset")
        self.pose_known = False
        self.rvec = np.zeros(3) * np.nan
        self.tvec = np.zeros(3) * np.nan

    @property
    def num_markers(self) -> int:
        """Returns number of total markers"""
        return len(self.point_ids)

    @property
    def unlocated_point_ids(self) -> ndarray:
        """Returns only unlocated marker IDs"""
        return self.point_ids[np.logical_not(self.located_markers_mask)]

    @property
    def num_located_markers(self) -> int:
        """Returns number of located markes"""
        return self.located_markers_mask.sum()

    @property
    def located_point_ids(self) -> ndarray:
        """Returns only located_markers_mask marker IDs"""
        return self.point_ids[self.located_markers_mask]

    @property
    def located_marker_points_object(self) -> ndarray:
        """Returns only located_markers_mask object points of markers"""
        return self.pts_obj_xyz[self.located_markers_mask]

    @property
    def located_marker_points_image(self) -> ndarray:
        """Returns only located_markers_mask image points of markers"""
        return self.pts_im_xy[self.located_markers_mask]

    def save_as_hdf(self, file: str) -> None:
        """Writes all information to given HDF file"""
        datasets = [
            "point_ids",
            "located_point_ids",
            "unlocated_point_ids",
            "pts_im_xy",
            "pts_obj_xyz",
            "located_marker_mask",
            "image_index",
        ]
        data = [
            self.point_ids,
            self.located_point_ids,
            self.unlocated_point_ids,
            self.pts_im_xy,
            self.pts_obj_xyz,
            self.located_markers_mask,
            self.img_id,
        ]
        save_hdf5_datasets(data, datasets, file)

    def calc_reprojection_errors(self) -> ndarray:
        """Calculates reprojection error for all located points

        Returns
        -------
        ndarray
            Shape (N,) array of reprojection errors. N=number of image points

        """
        # Reproject points
        pts_world = self.pts_obj_xyz[self.located_markers_mask]  # Nx3
        pts_im_reproj = self.camera.project_mat(pts_world, self.rvec, self.tvec)
        # Calculate error
        return np.sqrt(np.sum((self.pts_im_xy[self.located_markers_mask] - pts_im_reproj) ** 2, 1))
