"""Fixed pattern dot location calibration.
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from opencsp.app.fixed_pattern_deflectometry.lib.FixedPatternDotLocations import (
    FixedPatternDotLocations,
)
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.BlobIndex import BlobIndex
import opencsp.common.lib.deflectometry.image_processing as ip
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.photogrammetry.photogrammetry as ph


class FixedPatternSetupCalibrate:
    """Class to handle calibration of physical fixed pattern display dot locations

    Assumes the camera and screen are rotated about the y axis relative to each other. (ONLY the
    x indices are flipped)

    Attributes
    ----------
    - verbose : 0=no output, 1=print only output, 2=print and plot output, 3=only plot output
    - intersection_threshold : threshold to consider a ray intersection a success or not (meters)
    - figures : list of figures produced
    - marker_detection_params : parameters used in detecting Aruco markers in images
    - blob_detector : cv.SimpleBlobDetector_Params, used to detect blobs in image
    - blob_search_threshold : Search radius to use when searching for blobs (pixels)
    """

    def __init__(
        self,
        images: list[ndarray],
        origin_pts: Vxy,
        camera: Camera,
        pts_xyz_corners: Vxyz,
        pts_ids_corners: ndarray,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
    ) -> 'FixedPatternSetupCalibrate':
        """Instantiates the calibration class

        Parameters
        ----------
        images : list[ndarray]
            Images of Aruco markers and dots
        origin_pts : Vxy
            Points on the images corresponding to index (0, 0)
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
        """
        # Save data
        self._images = images
        self._origin_pts = origin_pts
        self._camera = camera
        self._pts_xyz_corners = pts_xyz_corners
        self._pts_ids_corners = pts_ids_corners
        self._x_min = x_min
        self._y_min = y_min
        self._x_max = x_max
        self._y_max = y_max

        self._num_images = len(images)

        # Settings
        self.verbose = 0
        self.intersection_threshold = 0.002  # meters
        self.figures = []
        self.marker_detection_params = {
            'adaptive_thresh_constant': 10,
            'min_marker_perimeter_rate': 0.01,
        }
        self.blob_search_threshold = 20  # pixels

        # Blob detection parameters
        self.blob_detector = cv.SimpleBlobDetector_Params()
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
        self._dot_points_xyz_mat = (
            np.ndarray((x_max - x_min + 1, y_max - y_min + 1, 3)) * np.nan
        )
        self._num_dots: int
        self._marker_ids: list[ndarray] = []
        self._marker_corner_ids: list[ndarray] = []
        self._marker_corners_xy: list[Vxy] = []
        self._marker_corners_xyz: list[Vxyz] = []
        self._rots_cams: list[ndarray] = []
        self._vecs_cams: list[ndarray] = []
        self._dot_intersection_dists: ndarray

    def _find_dots_in_images(self) -> None:
        """Finds dot locations for several camera poses"""
        dot_image_points_xy_mat = []
        masks_unassigned = []
        for idx, (image, origin_pt) in enumerate(zip(self._images, self._origin_pts)):
            if self.verbose in [1, 2]:
                print(f'Finding dots in image: {idx:d}')

            # Find blobs
            pts = ip.detect_blobs(image, self.blob_detector)

            # Index all found points
            blob_index = BlobIndex(
                pts, -self._x_max, -self._x_min, self._y_min, self._y_max
            )
            blob_index.search_thresh = self.blob_search_threshold
            blob_index.verbose = False
            blob_index.run(origin_pt)
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
            self._dot_image_points_xy.append(
                Vxy((dot_image_points_x, dot_image_points_y))
            )

        # Save common indices as vector
        indices_x = indices[mask_all_assigned, 0]
        indices_y = indices[mask_all_assigned, 1]
        self._dot_image_points_indices = Vxy((indices_x, indices_y), dtype=int)

        # Save all indices as matrix
        self._dot_image_points_indices_x = np.arange(self._x_min, self._x_max + 1)
        self._dot_image_points_indices_y = np.arange(self._y_min, self._y_max + 1)

    def _find_markers_in_images(self) -> None:
        """Finds Aruco marker corners in images and assigns xyz points"""
        ids_add = np.array([0, 1, 2, 3])
        for idx, image in enumerate(self._images):
            if self.verbose in [1, 2]:
                print(f'Finding marker corners in image: {idx:d}')

            # Find markers in image
            ids, pts = ph.find_aruco_marker(
                image,
                self.marker_detection_params['adaptive_thresh_constant'],
                self.marker_detection_params['min_marker_perimeter_rate'],
            )
            # Save point locations and IDs
            marker_ids = np.repeat(ids, 4)
            marker_corner_ids = np.repeat(ids * 4, 4) + np.tile(ids_add, ids.size)
            marker_corners_xy = Vxy(np.concatenate(pts, 0).T)
            # Save xyz locations
            point_idxs = []
            for marker_corner_id in marker_corner_ids:
                point_idxs.append(
                    np.where(self._pts_ids_corners == marker_corner_id)[0][0]
                )

            self._marker_ids.append(marker_ids)
            self._marker_corner_ids.append(marker_corner_ids)
            self._marker_corners_xy.append(marker_corners_xy)
            self._marker_corners_xyz.append(self._pts_xyz_corners[point_idxs])

    def _calculate_camera_poses(self) -> None:
        """Calculates 3d camera poses"""
        for cam_idx in range(self._num_images):
            if self.verbose in [0, 2]:
                print(
                    f'Calculating camera {cam_idx:d} pose with {len(self._marker_corners_xyz[cam_idx]):d} points'
                )

            # Attempt to solve for camera pose
            ret, rvec, tvec = cv.solvePnP(
                self._marker_corners_xyz[cam_idx].data.T,
                self._marker_corners_xy[cam_idx].data.T,
                self._camera.intrinsic_mat,
                self._camera.distortion_coef,
            )
            if not ret:
                raise ValueError(
                    f'Camera calibration was not successful for image {cam_idx:d}'
                )

            self._rots_cams.append(Rotation.from_rotvec(rvec.squeeze()))
            self._vecs_cams.append(Vxyz(tvec))

    def _intersect_rays(self) -> None:
        """Intersects camera rays to find dot xyz locations"""
        points_xyz = []
        int_dists = []
        for dot_idx in tqdm(range(self._num_dots), desc='Intersecting rays'):
            dot_image_pts_xy = [pt[dot_idx] for pt in self._dot_image_points_xy]
            point, dists = ph.triangulate(
                [self._camera] * self._num_images,
                self._rots_cams,
                self._vecs_cams,
                dot_image_pts_xy,
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

    def _plot_common_dots(self) -> None:
        """Plots common dots on images"""
        for idx, (image, pts) in enumerate(
            zip(self._images, self._dot_image_points_xy)
        ):
            fig = plt.figure(f'image_{idx:d}_annotated_dots')
            plt.imshow(image, cmap='gray')
            plt.scatter(*pts.data, marker='.', color='red')
            self.figures.append(fig)

    def _plot_marker_corners(self) -> None:
        """Plots images with annotated marker corners"""
        for idx_cam, image in enumerate(self._images):
            fig = plt.figure(f'image_{idx_cam:d}_annotated_marker_corners')
            plt.imshow(image, cmap='gray')
            self._marker_corners_xy[idx_cam].draw()
            self.figures.append(fig)

    def _plot_located_cameras_and_points(self) -> None:
        """Plots all input xyz points and located cameras"""
        fig = plt.figure('cameras_and_points')
        ax = fig.add_subplot(111, projection='3d')
        ph.plot_pts_3d(
            ax, self._pts_xyz_corners.data.T, self._rots_cams, self._vecs_cams
        )
        ax.set_xlabel('x (meter)')
        ax.set_ylabel('y (meter)')
        ax.set_zlabel('z (meter)')
        ax.set_title('Located Cameras and Marker Corners')
        self.figures.append(fig)

    def _plot_intersection_distances(self) -> None:
        """Plots mean intersection distances"""
        fig = plt.figure('dot_ray_mean_intersection_distances')
        plt.plot(self._dot_intersection_dists.mean(axis=1) * 1000)
        plt.axhline(self.intersection_threshold * 1000, color='k')
        plt.xlabel('Dot Number')
        plt.ylabel('Mean Intersection Distance (mm)')
        plt.grid('on')
        self.figures.append(fig)

    def _plot_xyz_surface(self) -> None:
        """Plots xyz dot structure"""
        fig = plt.figure('xyz_dot_map')
        xs = self._dot_points_xyz_mat[..., 0]
        ys = self._dot_points_xyz_mat[..., 1]
        zs = self._dot_points_xyz_mat[..., 2]
        plt.scatter(xs, ys, c=zs, marker='o')
        cb = plt.colorbar()
        cb.set_label('Z height (meter)')
        plt.xlabel('x (meter)')
        plt.ylabel('y (meter)')
        plt.title('XYZ Dot Map')
        self.figures.append(fig)

    def _plot_xyz_indices(self) -> None:
        """Plots z value per dot on grid"""
        fig = plt.figure('dot_index_map')
        plt.imshow(
            self._dot_points_xyz_mat[..., 2],
            extent=(
                self._x_min - 0.5,
                self._x_max + 0.5,
                self._y_min - 0.5,
                self._y_max + 0.5,
            ),
            origin='lower',
        )
        cb = plt.colorbar()
        cb.set_label('Z height (meter)')
        plt.xlabel('x index')
        plt.ylabel('y index')
        plt.title('Dot Index Map')
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
        return (
            self._dot_image_points_indices_x,
            self._dot_image_points_indices_y,
            self._dot_points_xyz_mat,
        )

    def get_dot_location_object(self) -> FixedPatternDotLocations:
        """Returns FixedPatternDotLocations object with calibrated data"""
        return FixedPatternDotLocations(
            self._dot_image_points_indices_x,
            self._dot_image_points_indices_y,
            self._dot_points_xyz_mat,
        )

    def run(self) -> None:
        """Runs full calibration sequence"""
        self._find_dots_in_images()
        self._find_markers_in_images()
        self._calculate_camera_poses()
        self._intersect_rays()

        if self.verbose in [2, 3]:
            self._plot_common_dots()
            self._plot_marker_corners()
            self._plot_located_cameras_and_points()
            self._plot_intersection_distances()
            self._plot_xyz_surface()
            self._plot_xyz_indices()
