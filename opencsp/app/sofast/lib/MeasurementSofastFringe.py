"""Measurement class for SofastFringe
"""

import datetime as dt

import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
import opencsp.app.sofast.lib.AbstractMeasurementSofast as ams
import opencsp.app.sofast.lib.DistanceOpticScreen as osd
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools


class MeasurementSofastFringe(ams.AbstractMeasurementSofast):
    """SofastFringe measurement data class that contains captured images
    and metadata about the measurement.
    """

    def __init__(
        self,
        mask_images: np.ndarray,
        fringe_images: np.ndarray,
        fringe_periods_x: np.ndarray,
        fringe_periods_y: np.ndarray,
        dist_optic_screen_measure: osd.DistanceOpticScreen,
        date: dt.datetime,
        name: str = '',
    ) -> 'MeasurementSofastFringe':
        """
        A measurement contains 2 (MxN) mask images, n (MxN) images of
        horizontal/vertical fringes.

        Parameters
        ----------
        mask_images : ndarray
            MxNx2 frame array.
        fringe_images : ndarray
            MxNxN frame array. Y fringes first
        fringe_periods_x/y : 1d array
            Periods used to generate x/y fringes, fractional screens.
        dist_optic_screen_measure : MeasurementDistance
            Optic-screen distance measurement.
        date : datetime
            Collection date/time.
        name : str
            Name or serial number of measurement.

        """
        super().__init__(dist_optic_screen_measure, date, name)

        # Check mask image size
        if mask_images.shape[2] != 2 or np.ndim(mask_images) != 3:
            raise ValueError(f'Two mask images needed, but {mask_images.shape[2]} given.')

        # Save input measurement data
        self.mask_images = mask_images
        self.fringe_images = fringe_images
        self.fringe_periods_x = fringe_periods_x
        self.fringe_periods_y = fringe_periods_y

        # Save calculations
        self.image_shape_xy = np.flip(self.mask_images.shape[:2])
        self.image_shape_yx = self.mask_images.shape[:2]
        self.phase_shifts = int(4)
        self.num_y_ims = self.fringe_periods_y.size * self.phase_shifts
        self.num_x_ims = self.fringe_periods_x.size * self.phase_shifts
        self.num_fringe_ims = self.fringe_images.shape[2]
        # Check number of input fringes
        if (self.num_y_ims + self.num_x_ims) != self.num_fringe_ims or np.ndim(fringe_images) != 3:
            raise ValueError(f'Incorrect number of fringe images given. Fringe images shape = {fringe_images.shape}.')

        # Instantiate calibration objected fringes
        self._fringe_images_calibrated = None

    @property
    def fringe_images_y(self) -> np.ndarray:
        """Returns raw y-only fringes"""
        return self.fringe_images[..., : self.num_y_ims]

    @property
    def fringe_images_x(self) -> np.ndarray:
        """Returns raw x-only fringes"""
        return self.fringe_images[..., self.num_y_ims :]

    @property
    def fringe_images_calibrated(self) -> np.ndarray:
        """Returns calibrated fringes"""
        if self._fringe_images_calibrated is None:
            raise ValueError('Fringe images have not been calibrated.')

        return self._fringe_images_calibrated

    @property
    def fringe_images_y_calibrated(self) -> np.ndarray:
        """Returns calibrated y-only fringes"""
        return self.fringe_images_calibrated[..., : self.num_y_ims]

    @property
    def fringe_images_x_calibrated(self) -> np.ndarray:
        """Returns calibrated x-only fringes"""
        return self.fringe_images_calibrated[..., self.num_y_ims :]

    def calibrate_fringe_images(self, calibration: ImageCalibrationAbstract, **kwargs) -> None:
        """
        Performs brightness level calibration on the raw captured fringes.

        Parameters
        ----------
        calibration : ImageCalibrationAbstract
            Image Calibration object.
        **kwargs
            Other keyword arguments to pass into ImageCalibration object
            "apply_to_images" method.

        """
        if not isinstance(calibration, ImageCalibrationAbstract):
            raise ValueError('Input calibration must be instance of ImageCalibrationAbstract.')

        self._fringe_images_calibrated = calibration.apply_to_images(self, **kwargs)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = '') -> 'MeasurementSofastFringe':
        """
        Loads data from given file. Assumes data is stored as: PREFIX + MeasurementSofastFringe/Field_1

        Parameters
        ----------
        file : string
            HDF file to load
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        # Load grid data
        datasets = [
            prefix + 'MeasurementSofastFringe/mask_images',
            prefix + 'MeasurementSofastFringe/fringe_images',
            prefix + 'MeasurementSofastFringe/fringe_periods_x',
            prefix + 'MeasurementSofastFringe/fringe_periods_y',
        ]
        kwargs = hdf5_tools.load_hdf5_datasets(datasets, file)
        kwargs.update(super()._load_from_hdf(file, prefix + 'MeasurementSofastFringe/'))

        return cls(**kwargs)

    def save_to_hdf(self, file: str, prefix: str = '') -> None:
        """
        Saves data to given file. Data is stored as: PREFIX + MeasurementSofastFringe/Field_1

        Parameters
        ----------
        file : string
            HDF file to save
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        datasets = [
            prefix + 'MeasurementSofastFringe/mask_images',
            prefix + 'MeasurementSofastFringe/fringe_images',
            prefix + 'MeasurementSofastFringe/fringe_periods_x',
            prefix + 'MeasurementSofastFringe/fringe_periods_y',
        ]
        data = [self.mask_images, self.fringe_images, self.fringe_periods_x, self.fringe_periods_y]

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
        super()._save_to_hdf(file, prefix + 'MeasurementSofastFringe/')
