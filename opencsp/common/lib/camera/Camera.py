"""
Model of machine vision camera.

    ***  NOTE: THIS IS CAMERA CLASS DEVELOPED FOR SOFAST, SHOULD BE MERGED WITH HELIO_SCAN VERSION OF CAMERA.  ***

"""

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.tool import hdf5_tools


class Camera:
    """
    Calibrated machine vision camera representation.

    Parameters
    ----------
    intrinsic_mat : np.ndarray
    distortion_coef : np.ndarray
        1d array, distortion coefficients.
    image_shape_xy : tuple(int)
        (x, y), image size in pixels.
    name : str
        Name of camera/lens combination.

    """

    def __init__(
        self, intrinsic_mat: np.ndarray, distortion_coef: np.ndarray, image_shape_xy: tuple[int, int], name: str
    ):
        if intrinsic_mat.shape[0] != 3 or intrinsic_mat.shape[1] != 3 or np.ndim(intrinsic_mat) != 2:
            raise ValueError('Input intrinsic_mat must be a 3x3 ndarray.')

        self.intrinsic_mat = intrinsic_mat
        self.distortion_coef = distortion_coef
        self.image_shape_xy = image_shape_xy
        self.name = name

    def __repr__(self):
        """Returns the defined camera name"""
        return 'Camera: { ' + str(self.name) + ' }'

    @property
    def image_shape_yx(self):
        return self.image_shape_xy[::-1]

    def vector_from_pixel(self, pixels: Vxy) -> Uxyz:
        """
        Calculates pointing directions for given pixel

        Parameters
        ----------
        pixels : Vxy[float32 or float64]
            Pixel locations on camera sensor

        Returns
        -------
        Uxyz
            Poining direction for each input pixel

        """
        pointing = cv.undistortPoints(pixels.data, self.intrinsic_mat, self.distortion_coef)
        pointing = pointing[:, 0, :].T
        z = np.ones((1, pointing.shape[1]), dtype=pointing.dtype)
        pointing = np.concatenate((pointing, z), axis=0)
        return Uxyz(pointing)

    def project(self, P_object: Vxyz, R_object_cam: Rotation, V_cam_object_cam: Vxyz) -> Vxy:
        """
        Projects points in 3D space to the camera sensor.

        Parameters
        ----------
        P_object : Vxyz, float32 or float64
            3D points to project, object coordinates.
        R_object_cam : Rotation
            Rotation from object to camera coordinates.
        V_cam_object_cam : Vxyz
            Translation from camera to object in camera coordinates.

        Returns
        -------
        Vxy.
            Projected points, pixels.

        """
        pixels = self.project_mat(P_object.data.T, R_object_cam.as_rotvec(), V_cam_object_cam.data.squeeze())
        return Vxy(pixels.T)

    def project_mat(self, pts_object: np.ndarray, rot_vec: np.ndarray, v_cam_object_cam: np.ndarray) -> np.ndarray:
        """
        Identical to project but points in matrix form.

        Parameters
        ----------
        pts_object : ndarray
            Nx3 3D points to project, object coordinates.
        rot_vec : ndarray
            Rotation vector from object to camera coordinates.
        v_cam_object_cam : ndarray
            1D size (3,) translation vector from camera to object origin in
            camera coordinates.

        Returns
        -------
        ndarray
            Nx2 array of projected points.

        """
        pixels = cv.projectPoints(pts_object, rot_vec, v_cam_object_cam, self.intrinsic_mat, self.distortion_coef)[0]
        return pixels[:, 0, :]

    @classmethod
    def load_from_hdf(cls, file: str):
        """
        Loads from HDF5 file

        Parameters
        ----------
        file : string
            HDF5 file to load

        """
        datasets = ['Camera/intrinsic_mat', 'Camera/distortion_coef', 'Camera/image_shape_xy', 'Camera/name']
        kwargs = hdf5_tools.load_hdf5_datasets(datasets, file)

        return cls(**kwargs)

    def save_to_hdf(self, file: str):
        """
        Saves to HDF5 file

        Parameters
        ----------
        file : string
            HDF5 file to save

        """
        datasets = ['Camera/intrinsic_mat', 'Camera/distortion_coef', 'Camera/image_shape_xy', 'Camera/name']
        data = [self.intrinsic_mat, self.distortion_coef, self.image_shape_xy, self.name]
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
