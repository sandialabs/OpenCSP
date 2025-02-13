"""Photogrammetric reconstruction class based on images of Aruco markers
"""

from glob import glob
from os.path import join
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.photogrammetry.ImageMarker import ImageMarker
import opencsp.common.lib.photogrammetry.bundle_adjustment as ba
import opencsp.common.lib.photogrammetry.photogrammetry as ph
import opencsp.common.lib.tool.log_tools as lt


class SceneReconstruction:
    """Class containing methods and analysis algorithms to reconstruct a 3d
    scene of Aruco markers.

    Attributes
    ----------
    make_figures : bool
        To make output figures, by default, False.
    intersect_threshold : float
        Maximum point to ray distance to be considered an intersection during
        triangulation, by default 0.02 meters.
    """

    def __init__(self, camera: Camera, known_point_locations: ndarray, image_filter_path: str) -> "SceneReconstruction":
        """Instantiates SceneReconstruction class

        Parameters
        ----------
        camera : Camera
            Camera object
        known_point_locations : ndarray
            Shape (N, 4) array containing rough locations of some initial points. This is
            needed to give the algorithm a starting point. Be sure several camera locations
            can be calculated from these. Columns are [Marker ID, X, Y, Z]
        image_filter_path : str
            Glob-like file path search string to locations of images with Aruco markers

        """
        self.intersect_threshold = 0.02  # meters

        # Save data
        self.camera = camera
        self.known_point_locations = known_point_locations
        self.image_paths = glob(image_filter_path)
        self.image_paths.sort()

        # Declare attributes
        self.images: list[ImageMarker]  # Loaded image marker objects
        self.unique_point_ids: ndarray[int]  # Unique point IDs in scene
        self.unique_marker_ids: ndarray[int]  # Unique marker IDs in scene
        self.num_markers: int  # Total number of markers
        self.num_points: int  # Total number of points
        self.num_poses: int  # Total number of camera poses (number of images)
        self.points_xyz: ndarray  # Points locations
        self.located_point_ids: ndarray[int]  # IDs of located points
        self.located_point_mask: ndarray[bool]  # Mask of located points

        # Save figures
        self.make_figures = False
        self.figures: list[plt.Figure] = []

    @property
    def located_camera_idxs(self) -> ndarray:
        """Returns image indices of cameras with known poses"""
        idxs = []
        for image in self.images:
            if image.pose_known:
                idxs.append(image.img_id)
        return np.array(idxs)

    @property
    def all_image_rvecs(self) -> ndarray:
        """Returns all rvecs for all images in Nx3 array"""
        return np.vstack([image.rvec for image in self.images])

    @property
    def all_image_tvecs(self) -> ndarray:
        """Returns all tvecs for all images in Nx3 array"""
        return np.vstack([image.tvec for image in self.images])

    @property
    def unlocated_marker_ids(self) -> ndarray:
        """Returns all unlocated marker IDs"""
        return np.unique(np.hstack([image.unlocated_point_ids for image in self.images]))

    def convert_to_four_corners(self) -> None:
        """Converts all images to four corner images instead of single points"""
        for image in self.images:
            image.convert_to_four_corner()

        self.unique_point_ids = np.unique(np.hstack([im.point_ids for im in self.images]))
        self.unique_marker_ids = np.floor(self.unique_point_ids.astype(float) / 4).astype(int)

        mask_nan = np.array([1, np.nan, np.nan, np.nan] * self.num_points)
        mask_zero = np.array([1, 0, 0, 0] * self.num_points)

        self.num_points = self.unique_point_ids.size
        self.points_xyz = np.repeat(self.points_xyz, 4, axis=0) * mask_nan[:, None]
        self.located_point_ids = self.located_point_ids * 4
        self.located_point_mask = (np.repeat(self.located_point_mask, 4) * mask_zero).astype(bool)

    def set_id_known(self, id_: int, pt: np.ndarray) -> None:
        """Sets given ID as known in all images

        Parameters
        ----------
        id_ : int
            Marker ID to set
        pt : np.ndarray
            Shape (3,) ndarray xyz point location
        """
        lt.debug(f"Point ID {id_:.0f} located.")
        # Update master array
        mask = self.unique_point_ids == id_
        self.points_xyz[mask] = pt
        self.located_point_mask += mask
        self.located_point_ids = self.unique_point_ids[self.located_point_mask]
        # Save in all images
        for image in self.images:
            image.set_point_id_located(id_, pt)

    def set_ids_known(self, ids: Iterable[int], pts: ndarray) -> None:
        """Sets multiple IDs known in all images

        Parameters
        ----------
        ids : Iterable[int]
            Marker IDs to set
        pts : ndarray
            Nx3 ndarray of marker ID locations
        """
        for id_, pt in zip(ids, pts):
            self.set_id_known(id_, pt)

    def save_ids_known(self) -> None:
        """Loads known marker IDs and their locations"""
        # Load data
        marker_ids = self.known_point_locations[:, 0]
        pts_xyz_marker = self.known_point_locations[:, 1:4]
        # Set known IDs
        self.set_ids_known(marker_ids, pts_xyz_marker)

    def load_images(self) -> None:
        """Saves loaded dataset in class"""
        self.images: list[ImageMarker] = []
        for idx, file in enumerate(tqdm(self.image_paths, desc="Loading marker images")):
            self.images.append(ImageMarker.load_aruco_origin(file, idx, self.camera))

        # Save unique markers
        self.unique_point_ids = np.unique(np.hstack([im.point_ids for im in self.images]))
        self.unique_marker_ids = self.unique_point_ids.copy()
        self.num_markers = self.unique_point_ids.size
        self.num_points = self.unique_point_ids.size
        self.num_poses = len(self.images)
        self.points_xyz = np.zeros((self.num_points, 3)) * np.nan
        self.located_point_ids = np.array([])
        self.located_point_mask = np.zeros(self.num_points, dtype=bool)

    def located_images_with_view_of_marker(self, id_: int) -> list[ImageMarker]:
        """Returns list of located images that have view of given marker

        Parameters
        ----------
        id_ : int
            Marker ID to find

        Returns
        -------
        list[ImageMarker]
        """
        images = []
        for image in self.images:
            if image.pose_known and (id_ in image.point_ids.tolist()):
                images.append(image)
        return images

    def attempt_all_camera_pose_calculation(self) -> None:
        """Attempt to calculate poses of all cameras"""
        lt.debug("Solving for camera poses with SolvePNP")
        for image in self.images:
            image.attempt_calculate_pose()

    def attempt_all_points_triangulation(self, intersect_thres: float = 0.02) -> None:
        """Attemps to calculate position of all unknown points using ray intersection

        Parameters
        ----------
        intersect_thres : float, optional
            Maximum point to ray distance to be considered an intersection, by default 0.02
        """
        lt.debug("Solving for marker locations by intersecing rays")
        for id_ in self.unlocated_marker_ids:
            # Get images with view of marker
            images = self.located_images_with_view_of_marker(id_)
            # Check if enough views
            if len(images) < 2:
                lt.debug(f"Not enough camera views to locate marker ID {id_:d}")
                continue
            # Get list of cameras, pts, rots, tvecs
            cameras = []
            rots = []
            tvecs = []
            pts_img = []
            for im in images:
                idx = im.point_ids.tolist().index(id_)  # Index of point in current image
                pts_img.append(im.pts_im_xy[idx])  # Location of 2d point in image
                rots.append(Rotation.from_rotvec(im.rvec))
                tvecs.append(im.tvec)
                cameras.append(self.camera)
            tvecs = Vxyz(np.array(tvecs).T)
            pts_img = Vxy(np.array(pts_img).T)
            # Triangulate
            pt_int, dists = ph.triangulate(cameras, rots, tvecs, pts_img)
            if dists.max() < intersect_thres:
                self.set_id_known(id_, pt_int.data.squeeze())
            else:
                lt.debug(f"Too high of intersecion error to locate marker ID {id_:d}")

    def refine_located_poses_and_points(self) -> None:
        """Performs bundle adjustment on located points and poses"""
        lt.debug("Refining point and camera locations")

        # Locations of all cameras and marker points
        rvecs_all = np.nan_to_num(self.all_image_rvecs)  # (Nim, 3)
        tvecs_all = np.nan_to_num(self.all_image_tvecs)  # (Nim, 3)
        obj_pts_all = np.nan_to_num(self.points_xyz)  # (Npts, 3)

        point_indices = []
        camera_idxs = []
        points2d = []
        for image in self.images:
            if not image.pose_known:
                # Skip if pose is not known
                continue
            # Camera index for every point observation that has been located
            camera_idxs.append([image.img_id] * image.num_located_markers)
            # Point (0-based) index for every point observation that has been located
            pt_ids = image.located_point_ids  # ID number
            pt_idxs = [self.unique_point_ids.tolist().index(pt_id) for pt_id in pt_ids]  # index
            point_indices.append(pt_idxs)
            # 2d image points of every point observation
            points2d.append(image.pts_im_xy[image.located_markers_mask])

        camera_idxs = np.hstack(camera_idxs).astype(int)  # (Nobs, )
        point_indices = np.hstack(point_indices).astype(int)  # (Nobs, )
        points2d = np.vstack(points2d)  # (Nobs, 2)

        (
            rvecs_all_opt,  # optimized rvecs
            tvecs_all_opt,  # optimized tvecs
            pts_marker_opt,  # optimized marker locations
        ) = ba.bundle_adjust(
            rvecs_all,  # (Nim, 3)
            tvecs_all,  # (Nim, 3)
            obj_pts_all,  # (Npts, 3)
            camera_idxs,  # (Nobs,)
            point_indices,  # (Nobs,)
            points2d,  # (Nobs, 2)
            self.camera.intrinsic_mat,
            self.camera.distortion_coef,
            "both",
            1,
        )

        # Update 3d marker points
        self.set_ids_known(self.located_point_ids, pts_marker_opt[self.located_point_mask])

        # Update rvec/tvec
        for idx in self.located_camera_idxs:
            self.images[idx].set_pose(rvecs_all_opt[idx], tvecs_all_opt[idx])

    def calculate_mean_reproj_errors(self) -> ndarray:
        """Returns array of reprojection errors. For each located camera, saves mean reprojection
        errors of all located points. NaN if pose is unlocated."""
        errors = np.zeros(self.num_poses) * np.nan
        for idx, image in enumerate(self.images):
            if image.pose_known:
                errors[idx] = image.calc_reprojection_errors().mean()
        return np.array(errors)

    def scale_points(self, point_pairs: ndarray[int], distances: ndarray[float]) -> None:
        """Scales point locations to match measured distances between pairs
        of Aruco marker origin points (corner 0).
        (Must be applied after conversion to the four point model)

        Parameters
        ----------
        point_pairs : ndarray
            Nx2 array of point pairs indices (MARKER IDs)
        distances : ndarray
            (N,) array of distances between point pairs
        """
        # Calculate scales
        scales = ph.scale_points(Vxyz(self.points_xyz.T), self.unique_point_ids, point_pairs * 4, distances)

        lt.info("Point cloud scaling summary:")
        lt.info(f"Calculated average point cloud scale: {scales.mean():.4f}.")
        lt.info(f"STDEV of point cloud scales (N={scales.size}): {scales.std():.4f}")

        # Apply scale to points
        self.points_xyz *= scales.mean()

    def align_points(self, marker_ids: ndarray[int], alignment_values: Vxyz, apply_scale: bool = False) -> None:
        """Aligns selected markers origin points (corner index 0) within
        point cloud to match given alignment values. Set to NAN for floating.
        Points are aligned FIRST, then transformed.
        (Must be applied after conversion to the four point model)

        See align_points() of photogrammetry.py for more information.

        Parameters
        ----------
        marker_ids : ndarray[int]
            (N,) array of MARKER IDs
        alignment_values : Vxyz
            (N, 3) array of optimal point locations of marker origin point (corner 0)
        apply_scales : bool
            To scale points or just align.
        """
        # Gather points with matching marker IDs
        pts_obj = []
        for id_ in marker_ids:
            mask = self.unique_point_ids == (id_ * 4)
            pts_obj.append(self.points_xyz[mask])
        pts_obj = Vxyz(np.array(pts_obj).T)

        # Perform alignment
        trans, scale, errors = ph.align_points(pts_obj, alignment_values, apply_scale)

        # Apply to points
        self.points_xyz = trans.apply(Vxyz(self.points_xyz.T) * scale).data.T

        # Log summary output
        lt.info("Point cloud alignment summary:")
        lt.info(f"Scale factor: {scale:.4f}")
        lt.info(f"Rotation: {trans.R.magnitude():.4f} radians")
        lt.info(f"Translation: {trans.V.magnitude()[0]:.4f} meters")
        lt.info(f"Average alignment errors: {errors.mean():.4f} meters")
        lt.info(f"STDEV of alignment errors (N={errors.size:d}): {errors.std():.4f} meters")

    def log_reprojection_error_summary(self) -> None:
        """Logs reprojection error summary"""
        errors = self.calculate_mean_reproj_errors()
        lt.info("Reprojection error summary:")
        lt.info(f"Average per-camera reprojection error: {np.nanmean(errors):.2f} pixels")
        lt.info(f"Min per-camera reprojection error: {np.nanmin(errors):.2f} pixels")
        lt.info(f"Max per-camera reprojection error: {np.nanmax(errors):.2f} pixels")
        lt.info(
            "STDEV of per-camera reprojection error "
            f"(N={np.logical_not(np.isnan(errors)).size:d}): {np.nanstd(errors):.2f} pixels"
        )

    def log_located_points_cameras(self) -> None:
        """Logs currently located cameras and markers"""
        lt.debug(f"Located camera indices: {self.located_camera_idxs}")
        lt.debug(f"Located point IDs: {self.located_point_ids}")

    def log_point_location_summary(self) -> None:
        """Logs short summary of current progress"""
        lt.debug(f"Number of located cameras: {len(self.located_camera_idxs):d}")
        lt.debug(f"Number of located points: {len(self.located_point_ids):d}")

    def plot_point_camera_summary(self) -> None:
        """Plots situational summary: Camera poses and point locations"""
        # Create axes
        fig = plt.figure("Scene_Reconstruction_Summary")
        ax = fig.add_subplot(projection="3d")
        self.figures.append(fig)

        # Plot camera locations with direction needle
        for image in self.images:
            if not image.pose_known:
                continue
            tvec = image.tvec
            rvec = image.rvec
            rot_cam_to_world = Rotation.from_rotvec(rvec).inv()
            tvec_world = rot_cam_to_world.apply(-tvec[None, :])
            z_vec = rot_cam_to_world.apply(np.array([[0.0, 0.0, 1.0]]))
            ax.scatter(*tvec_world.T, color="orange")
            ax.plot(
                [tvec_world[0, 0], tvec_world[0, 0] + z_vec[0, 0]],
                [tvec_world[0, 1], tvec_world[0, 1] + z_vec[0, 1]],
                [tvec_world[0, 2], tvec_world[0, 2] + z_vec[0, 2]],
                color="black",
            )

        # Plot located points
        ax.scatter(*self.points_xyz.T)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.axis("equal")
        ax.set_title("Point and Camera Positions")

    def plot_reprojection_errors(self) -> None:
        """Plots mean reprojection error magnitude vs camera pose"""
        fig = plt.figure("Scene_Reconstruction_ReprojectionError")
        ax = fig.gca()
        self.figures.append(fig)

        errors = self.calculate_mean_reproj_errors()
        ax.plot(errors)
        ax.set_xlabel("Camera Pose")
        ax.set_ylabel("Mean Reprojecion Error (pixels)")
        ax.grid()
        ax.set_title("Reprojection Errors")

    def save_all_as_hdf(self, directory: str) -> None:
        """Saves all ImageMarkers as HDF files in given directory"""
        for idx, image in enumerate(self.images):
            image.save_as_hdf(join(directory, f"{idx:03d}.h5"))

    def save_data_as_csv(self, file: str) -> None:
        """Saves the location of all marker points in given CSV file. Columns are
        [Marker ID, Corner ID, x, y, z]
        """
        np.savetxt(
            file,
            self.get_data(),
            delimiter=",",
            header="Marker ID,Corner ID,x (meter),y (meter),z (meter)",
            encoding="utf-8",
            fmt=["%.0f", "%.0f", "%.8f", "%.8f", "%.8f"],
        )

    def optimize(self, intersect_thresh: float = 0.02) -> None:
        """Runs point location optimization routine

        Parameters
        ----------
        intersect_thres : float, optional
            Maximum point to ray distance to be considered an intersection, by default 0.02
        """
        # Count number of unlocated points
        num_unlocated_pts = self.unlocated_marker_ids.size

        # Update status
        lt.debug("SceneReconstruction initial state")
        self.log_point_location_summary()

        for idx_attempt in range(100):
            # Solve for camera positions using initial points only
            self.attempt_all_camera_pose_calculation()

            # Solve for unknown points
            self.attempt_all_points_triangulation(intersect_thresh)

            # Update status
            lt.debug(f"SceneReconstruction entering iteration {idx_attempt:d}")
            self.log_point_location_summary()

            # Bundle adjust
            self.refine_located_poses_and_points()

            # Check if complete
            if num_unlocated_pts == self.unlocated_marker_ids.size:
                break
            num_unlocated_pts = self.unlocated_marker_ids.size

        # Check that all markers have been found
        if self.unlocated_marker_ids.size != 0:
            lt.warn(
                f"{self.unlocated_marker_ids.size:d} markers remain unlocated. " "More camera images may be needed."
            )

        # Convert to 4-corner model
        lt.debug("SceneReconstruction entering final 4 point refinement phase")
        self.convert_to_four_corners()

        # Intersect rays
        self.attempt_all_points_triangulation(intersect_thresh)

        # Bundle adjust
        self.refine_located_poses_and_points()

    def run_calibration(self) -> None:
        """Runs the calibration sequence"""
        # Run calibration
        self.load_images()
        self.save_ids_known()
        self.optimize(self.intersect_threshold)
        self.calculate_mean_reproj_errors()

        # Log reprojection errors
        self.log_reprojection_error_summary()

        # Plot figures
        if self.make_figures:
            self.plot_point_camera_summary()
            self.plot_reprojection_errors()

    def get_data(self) -> ndarray:
        """Returns marker IDs, point IDs, and point locations in one array

        Returns
        -------
        Nx4 ndarray.
            Marker ID, point ID, X, Y, Z

        """
        return np.hstack((self.unique_marker_ids[:, None], self.unique_point_ids[:, None], self.points_xyz))
