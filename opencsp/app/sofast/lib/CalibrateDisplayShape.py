"""Class containing all screen distortion calibration routines.
Saves distortion data and calibrated markers for camera position calibration.
"""

from dataclasses import dataclass

import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from numpy import ndarray
from scipy import interpolate
from scipy.signal import medfilt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from opencsp.app.sofast.lib.DisplayShape import DisplayShape
import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.ImageProjection import CalParams, ImageProjectionData
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.photogrammetry.photogrammetry as ph
import opencsp.common.lib.photogrammetry.bundle_adjustment as ba
import opencsp.common.lib.tool.log_tools as lt


@dataclass
class DataInput:
    """Data class storing input data for a CalibrateDisplayShape class

    Parameters:
    -----------
    corner_ids : ndarray
        Corner IDs (marker ID * 4 + corner index). Does not have to be continuous
    screen_cal_point_pairs : ndarray
        Two column ndarray [Screen calibration fiducial ID, marker ID]
    resolution_xy : tuple[int, int]
        Resolution of output screen map
    pts_xyz_marker : Vxyz
        Aruco marker corners, meters
    camera : Camera
        Camera object used to capture screen distortion data
    image_projection_data : ImageProjectionData
        Image projection parameters
    measurements_screen : list[MeasurementSofastFringe]
        Screen shape Sofast measurement objects
    assume_located_points : bool
        To assume that points are located accuratly, does not optimize point location, by default True.
    ray_intersection_threshold : float
        Threshold to consider a valid intersection, meters, by default 0.001.
    """

    corner_ids: ndarray
    screen_cal_point_pairs: ndarray
    resolution_xy: tuple[int, int]
    pts_xyz_marker: Vxyz
    camera: Camera
    image_projection_data: ImageProjectionData
    measurements_screen: list[MeasurementSofastFringe]
    assume_located_points: bool = True
    ray_intersection_threshold: float = 0.001


@dataclass
class DataCalculation:
    """Data class storing calculation data"""

    pts_screen_frac_x: ndarray = None
    pts_screen_frac_y: ndarray = None
    cal_pattern_params: CalParams = None
    pts_uv_pixel_orientation: list[Vxy] = None
    pts_uv_pixel: list[Vxy] = None
    num_points_screen: int = None
    num_poses: int = None
    rvecs: list[Rotation] = None
    tvecs: list[Vxyz] = None
    pts_xyz_screen_orientation: Vxyz = None
    pts_xyz_screen_aligned: Vxyz = None
    intersection_dists_mean: ndarray = None
    intersection_points_mask: ndarray = None
    im_x_screen_pts: Vxyz = None
    im_y_screen_pts: Vxyz = None
    im_z_screen_pts: Vxyz = None
    masks: list[ndarray] = None
    x_phase_list: list[ndarray] = None
    y_phase_list: list[ndarray] = None


class CalibrateDisplayShape:
    """Class containing methods to calibrate screen distortion/position

    Attributes
    ----------
    data_input: : DataInput
        DataInput class with input data
    make_figures : bool
        Set to True to output plots
    data_calculation : DataCalculation
        DataCalculation class for storing calculated data
    figures : list[plt.Figure]
        List containing generated figures, if not figures generated, empty list.
    """

    def __init__(self, data_input: DataInput) -> 'CalibrateDisplayShape':
        """Instantiates CalibrateDisplayShape object

        Parameters
        ----------
        data_input : DataInput
            A DataInput class with all fields defined
        """
        self.data_input = data_input
        self.make_figures = False
        self.xyz_screen_map_clim_mm = 3

        # Load cal params
        cal_pattern_params = CalParams(
            self.data_input.image_projection_data.active_area_size_x,
            self.data_input.image_projection_data.active_area_size_y,
        )

        # Initialize calculation data structure
        self.data_calculation = DataCalculation(
            pts_screen_frac_x=np.linspace(0.01, 0.99, data_input.resolution_xy[0]),
            pts_screen_frac_y=np.linspace(0.01, 0.99, data_input.resolution_xy[1]),
            cal_pattern_params=cal_pattern_params,
        )

        # Save figures
        self.figures: list[plt.Figure] = []

    def interpolate_camera_pixel_positions(self) -> None:
        """Interpolates the XY position of each camera pixel position for
        regular grids of screen fractions"""
        # Create interpolation objects for each camera pose
        pts_uv_pixel_ori: list[Vxy] = []
        pts_uv_pixel_full: list[Vxy] = []
        self.data_calculation.x_phase_list = []
        self.data_calculation.y_phase_list = []
        self.data_calculation.masks = []
        for idx, meas in enumerate(self.data_input.measurements_screen):
            lt.debug(f'Processing measurement: {idx:d}')

            # Calculate mask
            mask1 = ip.calc_mask_raw(meas.mask_images, hist_thresh=0.2)
            mask2 = ip.keep_largest_mask_area(mask1)
            mask = np.logical_and(mask1, mask2)
            self.data_calculation.masks.append(mask)

            # Unwrap x phase
            signal = meas.fringe_images_x[mask].T
            ps = meas.fringe_periods_x
            x_pos = ip.unwrap_phase(signal, ps)
            self.data_calculation.x_phase_list.append(x_pos)

            # Unwrap y phase
            signal = meas.fringe_images_y[mask].T
            ps = meas.fringe_periods_y
            y_pos = ip.unwrap_phase(signal, ps)
            self.data_calculation.y_phase_list.append(y_pos)

            # Assemble data in 2D array
            im_x = np.zeros(mask.shape) * np.nan
            im_x[mask] = x_pos

            im_y = np.zeros(mask.shape) * np.nan
            im_y[mask] = y_pos

            # Interpolate orientation points
            pts_uv_pixel_ori.append(
                interp_xy_screen_positions(
                    im_x,
                    im_y,
                    self.data_calculation.cal_pattern_params.x_screen_axis,
                    self.data_calculation.cal_pattern_params.y_screen_axis,
                )
            )

            # Interpolate full resolution points
            pts_uv_pixel_full.append(
                interp_xy_screen_positions(
                    im_x, im_y, self.data_calculation.pts_screen_frac_x, self.data_calculation.pts_screen_frac_y
                )
            )

        # Save interpolated points
        self.data_calculation.pts_uv_pixel_orientation = pts_uv_pixel_ori
        self.data_calculation.pts_uv_pixel = pts_uv_pixel_full

        # Save number of points
        self.data_calculation.num_points_screen = (
            self.data_calculation.pts_screen_frac_x.size * self.data_calculation.pts_screen_frac_y.size
        )
        self.data_calculation.num_poses = len(pts_uv_pixel_ori)

    def locate_camera_positions(self) -> None:
        """Finds location of cameras in space using orientation points"""
        # Define initial guess of object points (Nx3 array)
        pts_obj_ori_0 = []
        for marker_id in self.data_input.screen_cal_point_pairs[:, 1]:
            mask = self.data_input.corner_ids == (marker_id * 4)
            pts_obj_ori_0.append(self.data_input.pts_xyz_marker.data.T[mask])
        pts_obj_ori_0 = np.vstack(pts_obj_ori_0)

        # Roughly locate cameras with SolvePNP
        rvecs_0 = []
        tvecs_0 = []
        pts_used_idxs = self.data_input.screen_cal_point_pairs[:, 0]
        for pts_img in self.data_calculation.pts_uv_pixel_orientation:
            pts_img_used = pts_img[pts_used_idxs]
            ret, rvec, tvec = cv.solvePnP(
                pts_obj_ori_0,
                pts_img_used.data.T,
                self.data_input.camera.intrinsic_mat,
                self.data_input.camera.distortion_coef,
            )
            if not ret:
                lt.error_and_raise(ValueError, 'Camera position did not solve correctly.')
            rvecs_0.append(rvec.squeeze())
            tvecs_0.append(tvec.squeeze())
        rvecs_0 = np.array(rvecs_0)
        tvecs_0 = np.array(tvecs_0)

        # Format data for optimization
        point_indices = np.tile(np.arange(pts_used_idxs.size), self.data_calculation.num_poses)
        camera_indices = np.repeat(np.arange(self.data_calculation.num_poses), pts_used_idxs.size)
        points_2d = np.vstack([vec[pts_used_idxs].data.T for vec in self.data_calculation.pts_uv_pixel_orientation])

        # Calculate error after rough camera alignment
        errors_0 = ph.reprojection_errors(
            rvecs_0, tvecs_0, pts_obj_ori_0, self.data_input.camera, camera_indices, point_indices, points_2d
        )
        error_0 = np.sqrt(np.mean(errors_0**2))
        lt.info(f'Reprojection error stage 1 rough alignment: {error_0:.2f} pixels')

        # Bundle adjustment optimizing points and camera poses
        if self.data_input.assume_located_points:
            type_ = 'camera'  # only optimize camera location
        else:
            type_ = 'both'  # optimize camera and point locations
        rvecs_1, tvecs_1, pts_obj_ori_1 = ba.bundle_adjust(
            rvecs_0,
            tvecs_0,
            pts_obj_ori_0,
            camera_indices,
            point_indices,
            points_2d,
            self.data_input.camera.intrinsic_mat,
            self.data_input.camera.distortion_coef,
            type_,
            True,
        )

        # Calculate error
        errors_1 = ph.reprojection_errors(
            rvecs_1, tvecs_1, pts_obj_ori_1, self.data_input.camera, camera_indices, point_indices, points_2d
        )
        error_1 = np.sqrt(np.mean(errors_1**2))
        lt.info(f'Reprojection error stage 2 bundle adjustment: {error_1:.2f} pixels')

        # Save data in class
        self.data_calculation.rvecs = [Rotation.from_rotvec(v).inv() for v in rvecs_1]
        self.data_calculation.tvecs = Vxyz(tvecs_1.T)
        self.data_calculation.pts_xyz_screen_orientation = Vxyz(pts_obj_ori_1.T)

    def calculate_3d_screen_points(self) -> None:
        """Calculates 3d screen points by intersecting camera rays"""
        # Find camera location in screen coordinates
        v_screen_cam_screen = []
        for vec, rot in zip(self.data_calculation.tvecs, self.data_calculation.rvecs):
            v_cam_screen_screen = vec.rotate(rot)
            v_screen_cam_screen.append(-v_cam_screen_screen)
        v_screen_cam_screen = [v.data for v in v_screen_cam_screen]
        v_screen_cam_screen = Vxyz(np.concatenate(v_screen_cam_screen, 1))

        # Find pointing vectors for points in screen coordinates
        u_cam_pt_screen_mat = np.zeros((self.data_calculation.num_points_screen, self.data_calculation.num_poses, 3))

        # Loop through all camera poses
        for idx_pose, (pts, rot) in enumerate(zip(self.data_calculation.pts_uv_pixel, self.data_calculation.rvecs)):
            # Get camera pointing vectors in camera coordinates
            v_cam_pt_cam = self.data_input.camera.vector_from_pixel(pts)

            # Convert to unit vectors in unit vector in screen coordinates
            u_cam_pt_screen = v_cam_pt_cam.rotate(rot).normalize()

            # Store in array
            u_cam_pt_screen_mat[:, idx_pose, :] = u_cam_pt_screen.data.T

        # Calculate high-res intersection points
        v_screen_pt_screen_mat = np.zeros((self.data_calculation.num_points_screen, 3))
        intersection_dists = np.zeros((self.data_calculation.num_points_screen, self.data_calculation.num_poses))

        for idx_pt in tqdm(range(self.data_calculation.num_points_screen), desc='Intersecting rays'):
            # Intersect points
            pt, dists = ph.nearest_ray_intersection(
                v_screen_cam_screen, Vxyz(u_cam_pt_screen_mat[idx_pt].T)  # length N  # length N
            )

            v_screen_pt_screen_mat[idx_pt] = pt.data.squeeze()
            intersection_dists[idx_pt] = dists

        # Create mask of accurate intersections
        dist_error_mean = intersection_dists.mean(1)

        # Save data
        self.data_calculation.pts_xyz_screen_aligned = Vxyz(v_screen_pt_screen_mat.T)
        self.data_calculation.intersection_dists_mean = dist_error_mean
        self.data_calculation.intersection_points_mask = dist_error_mean < self.data_input.ray_intersection_threshold

    def assemble_xyz_data_into_images(self) -> None:
        """Assembles data into 2d arrays"""
        nx = self.data_calculation.pts_screen_frac_x.size
        ny = self.data_calculation.pts_screen_frac_y.size

        im_x = self.data_calculation.pts_xyz_screen_aligned.x.reshape((ny, nx)).copy()
        im_y = self.data_calculation.pts_xyz_screen_aligned.y.reshape((ny, nx)).copy()
        im_z = self.data_calculation.pts_xyz_screen_aligned.z.reshape((ny, nx)).copy()
        im_mask = self.data_calculation.intersection_points_mask.reshape((ny, nx))

        im_x[np.logical_not(im_mask)] = np.nan
        im_y[np.logical_not(im_mask)] = np.nan
        im_z[np.logical_not(im_mask)] = np.nan

        self.data_calculation.im_x_screen_pts = im_x
        self.data_calculation.im_y_screen_pts = im_y
        self.data_calculation.im_z_screen_pts = im_z

    def get_data(self) -> dict:
        """Returns dictionary with screen distortion data with fields:
        - pts_xy_screen_fraction: Vxy
        - pts_xyz_screen_coords: Vxyz
        """
        pts_x_screen_frac, pts_y_screen_frac = np.meshgrid(
            self.data_calculation.pts_screen_frac_x, self.data_calculation.pts_screen_frac_y
        )  # Screen fractions
        pts_y_screen_frac = np.flip(pts_y_screen_frac, axis=0)  # Flip y coordinate only
        pts_xy_screen_fraction = Vxy(
            np.array((pts_x_screen_frac.flatten(), pts_y_screen_frac.flatten()))[
                :, self.data_calculation.intersection_points_mask
            ]
        )
        pts_xyz_screen = Vxyz(
            self.data_calculation.pts_xyz_screen_aligned.data[:, self.data_calculation.intersection_points_mask]
        )

        return {'xy_screen_fraction': pts_xy_screen_fraction, 'xyz_screen_coords': pts_xyz_screen}

    def as_DisplayShape(self, name: str) -> DisplayShape:
        """Returns calibrated DisplayShape object.

        Parameters
        ----------
        name : str
            Name of DisplayShape.
        """
        grid_data = self.get_data()
        grid_data.update({'screen_model': 'distorted3D'})
        return DisplayShape(grid_data, name)

    def visualize_located_cameras(self) -> None:
        """Plots cameras and alignment points"""
        # Visualize located cameras
        fig = plt.figure('CalibrationScreenShape_Located_Cameras')
        self.figures.append(fig)
        ax = fig.add_subplot(111, projection='3d')

        ph.plot_pts_3d(
            ax,
            self.data_calculation.pts_xyz_screen_orientation.data.T,
            self.data_calculation.rvecs,
            self.data_calculation.tvecs,
            2,
        )
        plt.title('Calculated Camera Positions')
        ax.axis('equal')

    def visualize_annotated_camera_images(self) -> None:
        """Annotates images of screen with screen points"""
        # Visualize each camera pose
        for idx_pose in range(self.data_calculation.num_poses):
            fig = plt.figure(f'CalibrationScreenShape_Annotated_Camera_{idx_pose:d}_Images')
            self.figures.append(fig)
            ax = fig.gca()
            # Get measurement
            meas = self.data_input.measurements_screen[idx_pose]
            # Plot all white mask image
            ax.imshow(meas.mask_images[..., 1], cmap='gray')
            # Plot points
            for idx_point, pt in zip(
                self.data_calculation.cal_pattern_params.index, self.data_calculation.pts_uv_pixel_orientation[idx_pose]
            ):
                if idx_point in self.data_input.screen_cal_point_pairs[:, 0]:
                    color = 'blue'
                else:
                    color = 'black'
                ax.scatter(*pt.data, color=color, s=20)
                ax.text(*(pt + Vxy([5, -10])).data, idx_point, size=8, color='white')
            ax.scatter([], [], color='blue', label='Point used for orientation')
            ax.scatter([], [], color='black', label='Point unused for orientation')
            ax.legend()

    def plot_ray_intersection_errors(self) -> None:
        """Plots camera ray intersection errors"""
        fig = plt.figure('CalibrationScreenShape_Ray_Intersection_Errors', figsize=(9, 3))
        self.figures.append(fig)
        ax = fig.gca()

        ax.plot(self.data_calculation.intersection_dists_mean * 1000)
        ax.set_xlabel('Point Index')
        ax.set_ylabel('Average intersection \n error (mm)')
        ax.set_title('Intersection Errors')
        ax.axhline(self.data_input.ray_intersection_threshold * 1000)
        ax.set_ylim(0, self.data_input.ray_intersection_threshold * 1000 * 5)
        ax.grid()
        plt.subplots_adjust(bottom=0.21)

    def visualize_final_scenario(self) -> None:
        """Plots alignment points and screen points"""
        fig = plt.figure('CalibrationScreenShape_Scenario_Summary', figsize=(7, 5))
        self.figures.append(fig)
        ax = fig.gca()

        # Full screen point set
        mask_int_pts = self.data_calculation.intersection_points_mask
        pts_screen = self.data_calculation.pts_xyz_screen_aligned

        # Alignment screen point set
        pts_screen_orientation = self.data_calculation.pts_xyz_screen_orientation

        x = pts_screen[mask_int_pts].x
        y = pts_screen[mask_int_pts].y

        ax.scatter(*pts_screen[mask_int_pts].data[:2], marker='.', c='r', s=1, alpha=0.3)  # Intersection points
        ax.scatter(
            *pts_screen_orientation.data[:2], marker='s', s=20, label='Calibration Points'
        )  # Screen calibration points
        ax.set_title('Screen Points Summary')
        ax.grid()
        ax.set_xlim(x.max() * 1.05, x.min() * 1.05)
        ax.set_ylim(y.max() * 1.05, y.min() * 1.05)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.axis('equal')
        ax.legend(loc=2, bbox_to_anchor=(1, 1))
        fig.tight_layout()

    def visualize_xyz_screen_maps(self) -> None:
        """Visualizes screen xyz coordinates as images"""
        mask_int_pts = self.data_calculation.intersection_points_mask
        x = self.data_calculation.pts_xyz_screen_aligned[mask_int_pts].x
        y = self.data_calculation.pts_xyz_screen_aligned[mask_int_pts].y
        extent = (np.nanmax(x), np.nanmin(x), np.nanmax(y), np.nanmin(y))

        def format_image(axis: plt.Axes, im):
            axis.set_xlabel('X (meters)')
            axis.set_ylabel('Y (meters)')
            axis.axis('image')
            axis.set_xlim(np.nanmax(x), np.nanmin(x))
            axis.set_ylim(np.nanmax(y), np.nanmin(y))
            plt.colorbar(im)

        # Show processed x image
        fig = plt.figure('CalibrationScreenShape_Screen_Map_X')
        self.figures.append(fig)
        ax = fig.gca()
        im = ax.imshow(self.data_calculation.im_x_screen_pts, extent=extent, cmap='jet')
        format_image(ax, im)
        ax.set_title('X (m)')

        # Show processed y image
        fig = plt.figure('CalibrationScreenShape_Screen_Map_Y')
        self.figures.append(fig)
        ax = fig.gca()
        im = ax.imshow(self.data_calculation.im_y_screen_pts, extent=extent, cmap='jet')
        format_image(ax, im)
        ax.set_title('Y (m)')

        # Show processed z image
        fig = plt.figure('CalibrationScreenShape_Screen_Map_Z')
        self.figures.append(fig)
        ax = fig.gca()
        im = ax.imshow(self.data_calculation.im_z_screen_pts * 1000, extent=extent, cmap='jet')
        format_image(ax, im)
        im.set_clim(-self.xyz_screen_map_clim_mm, self.xyz_screen_map_clim_mm)
        ax.set_title('Z (mm)')

    def visualize_unwrapped_phase(self) -> None:
        """Visualizes x/y unwrapped phase for all poses"""
        for idx_pose in range(self.data_calculation.num_poses):
            # Get phase/mask data
            im_x = self.data_input.measurements_screen[idx_pose].mask_images[..., 1].copy()
            im_y = self.data_input.measurements_screen[idx_pose].mask_images[..., 1].copy()
            mask = self.data_calculation.masks[idx_pose]
            vals_x = self.data_calculation.x_phase_list[idx_pose]
            vals_y = self.data_calculation.y_phase_list[idx_pose]

            # Make x/y images RGB
            im_x = im_x.astype(float) / im_x.max()
            im_y = im_y.astype(float) / im_y.max()

            im_x = np.stack([im_x] * 3, axis=2)
            im_y = np.stack([im_y] * 3, axis=2)

            # Add active pixels as colored pixels
            cm = colormaps.get_cmap('jet')
            vals_x_jet = cm(vals_x)[:, :3]  # remove alpha channel
            vals_y_jet = cm(vals_y)[:, :3]  # remove alpha channel
            im_x[mask, :] = vals_x_jet
            im_y[mask, :] = vals_y_jet

            # Plot and save to figures list
            fig = plt.figure(f'CalibrationScreenShape_Unwrapped_pose_{idx_pose:d}_x_phase.png')
            self.figures.append(fig)
            ax = fig.gca()
            ax.imshow(im_x)
            ax.set_title(f'Unwrapped X Phase: Pose {idx_pose:d}')
            # Plot orientation points
            ax.scatter(*self.data_calculation.pts_uv_pixel_orientation[idx_pose].data, color='white')
            # Label orientation points
            for idx_pt, pt in enumerate(self.data_calculation.pts_uv_pixel_orientation[idx_pose]):
                plt.text(pt.x + mask.shape[1] * 0.02, pt.y, str(idx_pt), color='white')

            fig = plt.figure(f'CalibrationScreenShape_Unwrapped_pose_{idx_pose:d}_y_phase.png')
            self.figures.append(fig)
            ax = fig.gca()
            ax.imshow(im_y)
            ax.set_title(f'Unwrapped Y Phase: Pose {idx_pose:d}')
            # Plot orientation points
            ax.scatter(*self.data_calculation.pts_uv_pixel_orientation[idx_pose].data, color='white')
            # Label orientation points
            for idx_pt, pt in enumerate(self.data_calculation.pts_uv_pixel_orientation[idx_pose]):
                plt.text(pt.x + mask.shape[1] * 0.02, pt.y, str(idx_pt), color='white')

    def run_calibration(self) -> None:
        """Runs a complete calibration"""
        # Run calibration
        self.interpolate_camera_pixel_positions()
        self.locate_camera_positions()
        self.calculate_3d_screen_points()
        self.assemble_xyz_data_into_images()

        # Plot figures
        if self.make_figures:
            self.visualize_located_cameras()
            self.visualize_annotated_camera_images()
            self.plot_ray_intersection_errors()
            self.visualize_final_scenario()
            self.visualize_xyz_screen_maps()
            self.visualize_unwrapped_phase()


def interp_xy_screen_positions(im_x: np.ndarray, im_y: np.ndarray, x_sc: np.ndarray, y_sc: np.ndarray) -> Vxy:
    """
    Calculates the interpolated XY screen positions given X/Y fractional
    screen maps and X/Y interpolation vectors.

    Parameters
    ----------
    im_x : np.ndarray
        2D ndarray. X screen fraction image (fractional screens).
    im_y : np.ndarray
        2D ndarray. Y screen fraction image (fractional screens).
    x_sc : np.ndarray
        1D length N ndarray. X axis for output interpolated image.
        (fractionalscreens)
    y_sc : np.ndarray
        1D length M ndarray. Y axis for output interpolated image.
        (fractional screens)

    Returns
    -------
    Vxy
        Length (M * N) image coordinates corresponding to input interpolation axes (pixels).
    """
    # Set up interpolation parameters
    x_px = np.arange(im_x.shape[1]) + 0.5  # image pixels
    y_px = np.arange(im_y.shape[0]) + 0.5  # image pixels

    # Interpolate in X direction for every pixel row of image
    x_px_y_px_x_sc = np.zeros((y_px.size, x_sc.size)) * np.nan  # x pixel data, (y pixel, x screen) size array
    y_px_y_px_x_sc = np.zeros((y_px.size, x_sc.size)) * np.nan  # y pixel data, (y pixel, x screen) size array
    for idx_y in range(y_px.size):
        # Get x slices of x and y position values from images
        x_sc_vals = im_x[idx_y, :]  # x screen fractions
        y_sc_vals = im_y[idx_y, :]  # y screen fractions

        # Define active area of current row
        mask_row = np.logical_not(np.isnan(x_sc_vals))

        # Skip if not enough active pixels
        if mask_row.sum() <= 1:
            continue

        # Get active pixel locations (remove nans)
        x_sc_vals = x_sc_vals[mask_row]  # x screen fractions
        y_sc_vals = y_sc_vals[mask_row]  # y screen fractions
        x_px_vals = x_px[mask_row]  # x pixel locations

        # Smooth to reduce noise
        if x_sc_vals.size > 15:
            med_row = medfilt(x_sc_vals, 11)
            mask_noise = np.abs(x_sc_vals - med_row) < 0.05
        else:
            std_row = x_sc_vals.std()
            mask_noise = np.abs(x_sc_vals - x_sc_vals.mean()) < (3 * std_row)

        # Skip if not enough active pixels
        if mask_noise.sum() <= 2:
            continue

        x_sc_vals = x_sc_vals[mask_noise]
        y_sc_vals = y_sc_vals[mask_noise]
        x_px_vals = x_px_vals[mask_noise]

        # Interpolate x pixel coordinate
        f = interpolate.interp1d(x_sc_vals, x_px_vals, bounds_error=False, fill_value=np.nan)
        row = f(x_sc)  # x pixel coordinate
        x_px_y_px_x_sc[idx_y, :] = row

        # Interpolate y screen fraction value
        f = interpolate.interp1d(x_sc_vals, y_sc_vals, bounds_error=False, fill_value=np.nan)
        row = f(x_sc)
        y_px_y_px_x_sc[idx_y, :] = row

    # Interpolate in Y direction for every x-screen sample point column of image
    x_px_y_sc_x_sc = np.zeros((y_sc.size, x_sc.size))  # x pixel data, (y screen, x screen) size array
    y_px_y_sc_x_sc = np.zeros((y_sc.size, x_sc.size))  # y pixel data, (y screen, x screen) size array
    for idx_x in range(x_sc.size):
        # Get active pixel locations
        y_sc_vals = y_px_y_px_x_sc[:, idx_x]
        mask_col = np.logical_not(np.isnan(y_sc_vals))

        # Get interpolation vectors over active range (remove nans)
        y_sc_vals = y_sc_vals[mask_col]
        x_px_vals = x_px_y_px_x_sc[mask_col, idx_x]
        y_px_vals = y_px[mask_col]

        # Smooth to reduce noise
        if y_sc_vals.size > 15:
            med_row = medfilt(y_sc_vals, 11)
            mask_noise = np.abs(y_sc_vals - med_row) < 0.05
        else:
            std_row = y_sc_vals.std()
            mask_noise = np.abs(y_sc_vals - y_sc_vals.mean()) < (3 * std_row)

        y_sc_vals = y_sc_vals[mask_noise]
        x_px_vals = x_px_vals[mask_noise]
        y_px_vals = y_px_vals[mask_noise]

        # Interpolate x pixel coordinate
        f = interpolate.interp1d(y_sc_vals, x_px_vals, bounds_error=False, fill_value=np.nan)
        col = f(y_sc)
        x_px_y_sc_x_sc[:, idx_x] = col

        # Interpolate y pixel coordinate
        f = interpolate.interp1d(y_sc_vals, y_px_vals, bounds_error=False, fill_value=np.nan)
        col = f(y_sc)
        y_px_y_sc_x_sc[:, idx_x] = col

    # Return screen points
    if np.any(np.isnan(y_px_y_sc_x_sc)):
        raise ValueError('Nans present in y pixel interpolation array')
    if np.any(np.isnan(x_px_y_sc_x_sc)):
        raise ValueError('Nans present in x pixel interpolation array')

    return Vxy((x_px_y_sc_x_sc.flatten(), y_px_y_sc_x_sc.flatten()))
