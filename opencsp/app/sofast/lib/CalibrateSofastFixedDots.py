"""Fixed pattern dot location calibration.
"""

import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy import ndarray
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.common.lib.camera.Camera import Camera
from opencsp.app.sofast.lib.BlobIndex import BlobIndex
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.photogrammetry.photogrammetry as ph
from opencsp.common.lib.photogrammetry.ImageMarker import ImageMarker
import opencsp.common.lib.tool.log_tools as lt


class CalibrateSofastFixedDots:
    """Class to handle calibration of physical fixed pattern display dot locations

    Assumes the camera and screen are rotated about the y axis relative to each other. (ONLY the
    x indices are flipped)

    Attributes
    ----------
    plot : bool
        To create output plots
    intersection_threshold : float
        Threshold to consider a ray intersection a success, by default 0.002 meters.
    figures : list
        List of figures produced
    blob_search_threshold : float
        Search radius to use when searching for blobs, by default 20 pixels.
    blob_detector : cv.SimpleBlobDetector_Params
        Blob detetion settings used to detect blobs in image. By default:
        - minDistBetweenBlobs = 2
        - filterByArea = True
        - minArea = 50
        - maxArea = 1000
        - filterByCircularity = True
        - minCircularity = 0.8
        - filterByConvexity = False
        - filterByInertia = False
    """

    def __init__(
        self,
        files_images: list[str],
        origin_pts: Vxy,
        camera: Camera,
        pts_xyz_corners: Vxyz,
        pts_ids_corners: npt.NDArray[np.int_],
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        xy_pts_known: tuple[tuple[int, int]] = None,
    ) -> "CalibrateSofastFixedDots":
        """Instantiates the calibration class

        Parameters
        ----------
        files_images : list[str]
            File paths to images of Aruco markers and dots
        origin_pts : Vxy
            Points on the images corresponding to index `xy_pts_known`
        camera : Camera
            Camera calibration parameters of camera used to take images
        pts_xyz_corners : Vxyz
            XYZ locations of Aruco marker corners in entire scene
        pts_ids_corners : ndarray
            Corner ID (marker ID * 4 + corner index) for corresponding corner xyz points
        x_min/x_max : int
            Expected min/max x index values (follows screen x axis)
        y_min/y_max : int
            Expected min/max y index values (follows screen y axis)
        xy_pts_known : tuple[tuple[int, int]]
            The origin point xy indices. One xy point for each input image.
            For example, for two images using (0, 0) - xy_pts_known = ((0, 0), (0, 0)).
            If None, defaults to all zeros.
        """
        # Load images
        self._images: list[ImageMarker] = []
        for idx, file in enumerate(files_images):
            # Load image and find aruco marker corners (pixels)
            im = ImageMarker.load_aruco_origin(file, idx, camera)
            im.convert_to_four_corner()
            self._images.append(im)
            # Assign xyz corner locations in images (meters)
            for pt_xyz, pt_id in zip(pts_xyz_corners, pts_ids_corners):
                im.set_point_id_located(pt_id, pt_xyz.data.squeeze())

        if xy_pts_known is None:
            xy_pts_known = ((0, 0),) * len(origin_pts)

        # Save data
        self._origin_pts = origin_pts
        self._xy_pts_known = xy_pts_known
        self._camera = camera
        self._pts_xyz_corners = pts_xyz_corners
        self._pts_ids_corners = pts_ids_corners
        self._x_min = x_min
        self._y_min = y_min
        self._x_max = x_max
        self._y_max = y_max

        self._num_images = len(self._images)

        # Settings
        self.plot = False
        """To plot output figures"""
        self.intersection_threshold = 0.002  # meters
        """Ray intersection threshold to consider rays intersected, meters"""
        self.figures: list[plt.Figure] = []
        """List of output figures. Empty list if figures not generated."""
        self.blob_search_threshold = 20  # pixels
        """Location threshold when finding expected dots, pixels."""

        # Blob detection parameters
        self.blob_detector = cv.SimpleBlobDetector_Params()
        """OpenCV SimpleBlobDetector object used to find blobs in images"""
        self.blob_detector.minDistBetweenBlobs = 2
        self.blob_detector.filterByArea = True
        self.blob_detector.minArea = 50
        self.blob_detector.maxArea = 1000
        self.blob_detector.filterByCircularity = True
        self.blob_detector.minCircularity = 0.8
        self.blob_detector.filterByConvexity = False
        self.blob_detector.filterByInertia = False

        # Attributes
        self._dot_image_points_xy: list[Vxy] = []
        self._dot_image_points_indices: Vxy
        self._dot_image_points_indices_x: ndarray
        self._dot_image_points_indices_y: ndarray
        self._dot_points_xyz_mat = np.ndarray((y_max - y_min + 1, x_max - x_min + 1, 3)) * np.nan
        self._num_dots: int
        self._rots_cams: list[Rotation] = []
        self._vecs_cams: list[Vxyz] = []
        self._dot_intersection_dists: ndarray

    def _find_dots_in_images(self) -> None:
        """Finds dot locations for several camera poses"""
        dot_image_points_xy_mat = []
        masks_unassigned = []
        for idx_image, origin_pt in enumerate(self._origin_pts):
            lt.info(f"Finding dots in image: {idx_image:d}")

            # Find blobs
            pts = ip.detect_blobs(self._images[idx_image].image, self.blob_detector)

            # Index all found points
            blob_index = BlobIndex(pts, -self._x_max, -self._x_min, self._y_min, self._y_max)
            blob_index.search_thresh = self.blob_search_threshold
            blob_index.run(
                origin_pt, x_known=self._xy_pts_known[idx_image][0], y_known=self._xy_pts_known[idx_image][1]
            )
            points, indices = blob_index.get_data_mat()

            # Save points and indices
            dot_image_points_xy_mat.append(points)
            masks_unassigned.append(np.isnan(points).max(axis=2))

        # Flip ONLY x indices
        indices[..., 0] *= -1

        # Calculate common dots between all camera views
        mask_some_unassigned = np.array(masks_unassigned).sum(axis=0)
        mask_all_assigned = np.logical_not(mask_some_unassigned)
        self._num_dots = mask_all_assigned.sum()

        # Save common xy points
        for idx in range(self._num_images):
            dot_image_points_x = dot_image_points_xy_mat[idx][mask_all_assigned, 0]
            dot_image_points_y = dot_image_points_xy_mat[idx][mask_all_assigned, 1]
            self._dot_image_points_xy.append(Vxy((dot_image_points_x, dot_image_points_y)))

        # Save common indices as vector
        indices_x = indices[mask_all_assigned, 0]
        indices_y = indices[mask_all_assigned, 1]
        self._dot_image_points_indices = Vxy((indices_x, indices_y), dtype=int)

        # Save all indices as matrix
        self._dot_image_points_indices_x = np.arange(self._x_min, self._x_max + 1)
        self._dot_image_points_indices_y = np.arange(self._y_min, self._y_max + 1)

    def _calculate_camera_poses(self) -> None:
        """Calculates 3d camera poses"""
        for cam_idx in range(self._num_images):
            # Calculate camera pose
            ret = self._images[cam_idx].attempt_calculate_pose()
            if ret == -1:
                lt.critical_and_raise(ValueError, f"Camera pose {cam_idx:d} not calculated successfully")

            self._rots_cams.append(Rotation.from_rotvec(self._images[cam_idx].rvec))
            self._vecs_cams.append(Vxyz(self._images[cam_idx].tvec))

            # Calculate reproj error
            errors = self._images[cam_idx].calc_reprojection_errors()
            # Log errors
            lt.info(f"Camera {cam_idx:d} mean corner reprojection error: {errors.mean():.2f} pixels")
            lt.info(f"Camera {cam_idx:d} min corner reprojection error: {errors.min():.2f} pixels")
            lt.info(f"Camera {cam_idx:d} max corner reprojection error: {errors.mean():.2f} pixels")
            lt.info(f"Camera {cam_idx:d} STDEV corner reprojection error: {errors.mean():.2f} pixels")

    def _intersect_rays(self) -> None:
        """Intersects camera rays to find dot xyz locations"""
        points_xyz = []
        int_dists = []
        for dot_idx in tqdm(range(self._num_dots), desc="Intersecting rays"):
            dot_image_pts_xy = [pt[dot_idx] for pt in self._dot_image_points_xy]
            point, dists = ph.triangulate(
                [self._camera] * self._num_images, self._rots_cams, self._vecs_cams, dot_image_pts_xy
            )
            points_xyz.append(point)
            int_dists.append(dists)
            # Save xyz point in matrix
            if dists.mean() <= self.intersection_threshold:
                indices = self._dot_image_points_indices[dot_idx]
                idx_x = indices.x[0] - self._x_min
                idx_y = indices.y[0] - self._y_min
                self._dot_points_xyz_mat[idx_y, idx_x, :] = point.data.squeeze()

        self._dot_intersection_dists = np.array(int_dists)
        lt.info(
            "Dot ray intersections mean intersection error: " f"{self._dot_intersection_dists.mean() * 1000:.1f} mm"
        )
        lt.info("Dot ray intersections min intersection error: " f"{self._dot_intersection_dists.min() * 1000:.1f} mm")
        lt.info("Dot ray intersections max intersection error: " f"{self._dot_intersection_dists.max() * 1000:.1f} mm")
        lt.info(
            "Dot ray intersections STDEV of intersection error: " f"{self._dot_intersection_dists.std() * 1000:.1f} mm"
        )

    def _plot_common_dots(self) -> None:
        """Plots common dots on images"""
        for idx_image in range(self._num_images):
            fig = plt.figure(f"image_{idx_image:d}_annotated_dots")
            plt.imshow(self._images[idx_image].image, cmap="gray")
            plt.scatter(*self._dot_image_points_xy[idx_image].data, marker=".", color="red")
            self.figures.append(fig)

    def _plot_marker_corners(self) -> None:
        """Plots images with annotated marker corners"""
        for idx_image in range(self._num_images):
            fig = plt.figure(f"image_{idx_image:d}_annotated_marker_corners")
            ax = fig.gca()
            ax.imshow(self._images[idx_image].image, cmap="gray")
            Vxy(self._images[idx_image].pts_im_xy.T).draw(ax=ax)
            self.figures.append(fig)

    def _plot_located_cameras_and_points(self) -> None:
        """Plots all input xyz points and located cameras"""
        fig = plt.figure("cameras_and_points")
        ax = fig.add_subplot(111, projection="3d")
        ph.plot_pts_3d(ax, self._pts_xyz_corners.data.T, self._rots_cams, self._vecs_cams)
        ax.set_xlabel("x (meter)")
        ax.set_ylabel("y (meter)")
        ax.set_zlabel("z (meter)")
        ax.set_title("Located Cameras and Marker Corners")
        self.figures.append(fig)

    def _plot_intersection_distances(self) -> None:
        """Plots mean intersection distances"""
        fig = plt.figure("dot_ray_mean_intersection_distances")
        plt.plot(self._dot_intersection_dists.mean(axis=1) * 1000)
        plt.axhline(self.intersection_threshold * 1000, color="k")
        plt.xlabel("Dot Number")
        plt.ylabel("Mean Intersection Distance (mm)")
        plt.grid("on")
        self.figures.append(fig)

    def _plot_xyz_surface(self) -> None:
        """Plots xyz dot structure"""
        fig = plt.figure("xyz_dot_map")
        xs = self._dot_points_xyz_mat[..., 0]
        ys = self._dot_points_xyz_mat[..., 1]
        zs = self._dot_points_xyz_mat[..., 2]
        plt.scatter(xs, ys, c=zs, marker="o")
        cb = plt.colorbar()
        cb.set_label("Z height (meter)")
        plt.xlabel("x (meter)")
        plt.ylabel("y (meter)")
        plt.title("XYZ Dot Map")
        self.figures.append(fig)

    def _plot_xyz_indices(self) -> None:
        """Plots z value per dot on grid"""
        fig = plt.figure("dot_index_map")
        plt.imshow(
            self._dot_points_xyz_mat[..., 2],
            extent=(self._x_min - 0.5, self._x_max + 0.5, self._y_min - 0.5, self._y_max + 0.5),
            origin="lower",
        )
        cb = plt.colorbar()
        cb.set_label("Z height (meter)")
        plt.xlabel("x index")
        plt.ylabel("y index")
        plt.title("Dot Index Map")
        self.figures.append(fig)

    def get_data(self) -> tuple[ndarray, ndarray, ndarray]:
        """Returns calibration dot xyz locations

        Returns
        -------
        ndarray
            1d array, X index axis
        ndarray
            1d array, Y index axis
        ndarray
            (N, M, 3) array of dot xyz locations
        """
        return (self._dot_image_points_indices_x, self._dot_image_points_indices_y, self._dot_points_xyz_mat)

    def get_dot_location_object(self) -> DotLocationsFixedPattern:
        """Returns DotLocationsFixedPattern object with calibrated data"""
        return DotLocationsFixedPattern(
            self._dot_image_points_indices_x, self._dot_image_points_indices_y, self._dot_points_xyz_mat
        )

    def save_figures(self, dir_save: str) -> None:
        """Saves figures in given directory"""
        for fig in self.figures:
            file = os.path.join(dir_save, fig.get_label() + ".png")
            lt.info(f"Saving figure to file: {file:s}")
            fig.savefig(file)

    def run(self) -> None:
        """Runs full calibration sequence"""
        self._find_dots_in_images()
        self._calculate_camera_poses()
        self._intersect_rays()

        if self.plot:
            self._plot_common_dots()
            self._plot_marker_corners()
            self._plot_located_cameras_and_points()
            self._plot_intersection_distances()
            self._plot_xyz_surface()
            self._plot_xyz_indices()
