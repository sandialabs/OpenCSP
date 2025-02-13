from dataclasses import dataclass
from warnings import warn

import numpy as np
from numpy import ndarray
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.tool import hdf5_tools


@dataclass
class CalculationDataGeometryGeneral(hdf5_tools.HDF5_SaveAbstract):
    """Data class used in deflectometry measurments. Saves general geometry and
    orientation data associated with a measurement."""

    v_cam_optic_centroid_cam_exp: Vxyz = None
    r_optic_cam_exp: Rotation = None
    v_cam_optic_cam_exp: Vxyz = None
    r_optic_cam_refine_1: Rotation = None
    r_optic_cam_refine_2: Rotation = None
    v_cam_optic_cam_refine_1: Vxyz = None
    v_cam_optic_cam_refine_2: Vxyz = None
    v_cam_optic_cam_refine_3: Vxyz = None

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given HDF5 file. Data is stored in PREFIX + CalculationDataGeometryGeneral/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.v_cam_optic_centroid_cam_exp,
            self.r_optic_cam_exp,
            self.v_cam_optic_cam_exp,
            self.r_optic_cam_refine_1,
            self.r_optic_cam_refine_2,
            self.v_cam_optic_cam_refine_1,
            self.v_cam_optic_cam_refine_2,
            self.v_cam_optic_cam_refine_3,
        ]
        datasets = [
            prefix + "CalculationDataGeometryGeneral/v_cam_optic_centroid_cam_exp",
            prefix + "CalculationDataGeometryGeneral/r_optic_cam_exp",
            prefix + "CalculationDataGeometryGeneral/v_cam_optic_cam_exp",
            prefix + "CalculationDataGeometryGeneral/r_optic_cam_refine_1",
            prefix + "CalculationDataGeometryGeneral/r_optic_cam_refine_2",
            prefix + "CalculationDataGeometryGeneral/v_cam_optic_cam_refine_1",
            prefix + "CalculationDataGeometryGeneral/v_cam_optic_cam_refine_2",
            prefix + "CalculationDataGeometryGeneral/v_cam_optic_cam_refine_3",
        ]
        _save_data_in_file(data, datasets, file)


@dataclass
class CalculationDataGeometryFacet(hdf5_tools.HDF5_SaveAbstract):
    """Data class used in deflectometry calculations of a single facet. Holds
    geometric/orientation data of a single facet measurement setup.
    """

    u_cam_measure_point_facet: Uxyz = None
    measure_point_screen_distance: float = None
    spatial_orientation: SpatialOrientation = None
    v_align_point_facet: Vxyz = None
    v_screen_points_fractional_screens: Vxy = None
    v_screen_points_screen: Vxyz = None
    u_pixel_pointing_facet: Uxyz = None
    v_screen_points_facet: Vxyz = None

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given HDF5 file. Data is stored in PREFIX + CalculationDataGeometryFacet/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.u_cam_measure_point_facet,
            self.measure_point_screen_distance,
            self.v_align_point_facet,
            self.v_screen_points_fractional_screens,
            self.v_screen_points_screen,
            self.u_pixel_pointing_facet,
            self.v_screen_points_facet,
        ]
        datasets = [
            prefix + "CalculationDataGeometryFacet/u_cam_measure_point_facet",
            prefix + "CalculationDataGeometryFacet/measure_point_screen_distance",
            prefix + "CalculationDataGeometryFacet/v_align_point_facet",
            prefix + "CalculationDataGeometryFacet/v_screen_points_fractional_screens",
            prefix + "CalculationDataGeometryFacet/v_screen_points_screen",
            prefix + "CalculationDataGeometryFacet/u_pixel_pointing_facet",
            prefix + "CalculationDataGeometryFacet/v_screen_points_facet",
        ]
        self.spatial_orientation.save_to_hdf(file, prefix + "CalculationDataGeometryFacet/")
        _save_data_in_file(data, datasets, file)


@dataclass
class CalculationError(hdf5_tools.HDF5_SaveAbstract):
    """Data class used in deflectometry calculations. Holds data on measurement/calculation
    errors during deflectometry calculations.
    """

    error_dist_optic_screen_1: float = None
    error_dist_optic_screen_2: float = None
    error_dist_optic_screen_3: float = None
    error_reprojection_1: float = None
    error_reprojection_2: float = None
    error_reprojection_3: float = None

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given HDF5 file. Data is stored in PREFIX + CalculationError/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.error_dist_optic_screen_1,
            self.error_dist_optic_screen_2,
            self.error_dist_optic_screen_3,
            self.error_reprojection_1,
            self.error_reprojection_2,
            self.error_reprojection_3,
        ]
        datasets = [
            prefix + "CalculationError/error_dist_optic_screen_1",
            prefix + "CalculationError/error_dist_optic_screen_2",
            prefix + "CalculationError/error_dist_optic_screen_3",
            prefix + "CalculationError/error_reprojection_1",
            prefix + "CalculationError/error_reprojection_2",
            prefix + "CalculationError/error_reprojection_3",
        ]
        _save_data_in_file(data, datasets, file)


@dataclass
class CalculationImageProcessingFacet(hdf5_tools.HDF5_SaveAbstract):
    """Data class used in deflectometry calculations of a single facet. Holds
    image processing data of a single facet measurement.
    """

    loop_facet_image_refine: LoopXY = None
    mask_fitted: ndarray = None
    mask_processed: ndarray = None
    v_facet_corners_image_exp: Vxy = None
    v_facet_centroid_image_exp: Vxy = None
    mask_bad_pixels: ndarray = None

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given HDF5 file. Data is stored in PREFIX + CalculationImageProcessingFacet/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.loop_facet_image_refine,
            self.mask_fitted,
            self.mask_processed,
            self.v_facet_corners_image_exp,
            self.v_facet_centroid_image_exp,
            self.mask_bad_pixels,
        ]
        datasets = [
            prefix + "CalculationImageProcessingFacet/loop_facet_image_refine",
            prefix + "CalculationImageProcessingFacet/mask_fitted",
            prefix + "CalculationImageProcessingFacet/mask_processed",
            prefix + "CalculationImageProcessingFacet/v_facet_corners_image_exp",
            prefix + "CalculationImageProcessingFacet/v_facet_centroid_image_exp",
            prefix + "CalculationImageProcessingFacet/mask_bad_pixels",
        ]
        _save_data_in_file(data, datasets, file)


@dataclass
class CalculationImageProcessingGeneral(hdf5_tools.HDF5_SaveAbstract):
    """Data class used in deflectometry calculations. Holds general image processing
    calculations from a deflectometry measurement.
    """

    mask_raw: ndarray = None
    v_edges_image: Vxyz = None
    v_mask_centroid_image: Vxy = None
    loop_optic_image_exp: LoopXY = None
    loop_optic_image_refine: LoopXY = None

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given HDF5 file. Data is stored in PREFIX + CalculationImageProcessingGeneral/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.mask_raw,
            self.v_edges_image,
            self.v_mask_centroid_image,
            self.loop_optic_image_exp,
            self.loop_optic_image_refine,
        ]
        datasets = [
            prefix + "CalculationImageProcessingGeneral/mask_raw",
            prefix + "CalculationImageProcessingGeneral/v_edges_image",
            prefix + "CalculationImageProcessingGeneral/v_mask_centroid_image",
            prefix + "CalculationImageProcessingGeneral/loop_optic_image_exp",
            prefix + "CalculationImageProcessingGeneral/loop_optic_image_refine",
        ]
        _save_data_in_file(data, datasets, file)


@dataclass
class CalculationBlobAssignment(hdf5_tools.HDF5_SaveAbstract):
    """Data class for holding calculated parameters from Sofast Fixed blob assignment"""

    pts_image: Vxy = None
    """Positions in the measured image that correspond to blobs (units of image pixels from upper-left corner)"""
    pts_index_xy: Vxy = None
    """XY indices relative to user-defined origin point (0, 0) corresponding to positions in the image (pts_image)"""
    active_point_mask: ndarray[bool] = None
    """2d ndarray, mask of active points"""

    def save_to_hdf(self, file: str, prefix: str = "") -> None:
        """Saves data to given HDF5 file. Data is stored in PREFIX + CalculationBlobAssignment/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [self.pts_image.data, self.pts_index_xy.data, self.active_point_mask]
        datasets = [
            prefix + "CalculationBlobAssignment/pts_image",
            prefix + "CalculationBlobAssignment/pts_index_xy.data",
            prefix + "CalculationBlobAssignment/active_point_mask",
        ]
        _save_data_in_file(data, datasets, file)


@dataclass
class CalculationFacetEnsemble(hdf5_tools.HDF5_SaveAbstract):
    """Data class used in deflectometry calculations. Holds calculations
    relating to facet ensembles.
    """

    trans_facet_ensemble: TransformXYZ = None
    slopes_ensemble_xy: npt.NDArray[np.float_] = None
    v_surf_points_ensemble: Vxyz = None
    v_facet_pointing_ensemble: Vxyz = None

    def save_to_hdf(self, file: str, prefix: str = ""):
        """Saves data to given HDF5 file. Data is stored in PREFIX + CalculationEnsemble/...

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str
            Prefix to append to folder path within HDF file (folders must be separated by "/")
        """
        data = [
            self.trans_facet_ensemble.matrix,
            self.slopes_ensemble_xy,
            self.v_surf_points_ensemble.data,
            self.v_facet_pointing_ensemble.data,
        ]
        datasets = [
            prefix + "CalculationEnsemble/trans_facet_ensemble",
            prefix + "CalculationEnsemble/slopes_ensemble_xy",
            prefix + "CalculationEnsemble/v_surf_points_ensemble",
            prefix + "CalculationEnsemble/v_facet_pointing_ensemble",
        ]
        _save_data_in_file(data, datasets, file)


def _save_data_in_file(data_in: list, datasets_in: list, file: str) -> None:
    # Save only data that is not None
    data = []
    datasets = []
    for d, ds in zip(data_in, datasets_in):
        if d is not None:
            # Convert to array for known data types
            if isinstance(d, Vxyz) or isinstance(d, Vxy):
                data.append(d.data)
            elif isinstance(d, LoopXY):
                data.append(d.vertices.data)
            elif isinstance(d, Rotation):
                data.append(d.as_rotvec())
            elif isinstance(d, float) or isinstance(d, int) or isinstance(d, ndarray):
                data.append(d)
            else:
                raise ValueError(f"Unrecognized data type {type(d)} could not be saved.")
            datasets.append(ds)

    if len(data) > 0:
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
    else:
        warn(f'Length 0 dataset was not saved to file "{file:s}"', UserWarning, stacklevel=2)
