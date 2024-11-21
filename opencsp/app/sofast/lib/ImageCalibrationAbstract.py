from abc import abstractmethod, ABC
from warnings import warn

from numpy import ndarray
import numpy as np
from scipy import interpolate

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.lib.AbstractPlotHandler as aph
import opencsp.common.lib.tool.hdf5_tools as hdf5_tools
import opencsp.common.lib.tool.log_tools as lt


class ImageCalibrationAbstract(hdf5_tools.HDF5_IO_Abstract, aph.AbstractPlotHandler, ABC):
    def __init__(self, camera_values: ndarray, display_values: ndarray):
        """
        ImageCalibration object used for calibrating fringe images. Creates a
        "response_function" that converts camera values to effecive display
        values.

        Parameters
        ----------
        camera_values : ndarray
            1D array, camera digital numbers.
        display_values : ndarray
            1D array, corresponding display digital numbers.

        """
        super().__init__()

        self.camera_values = camera_values
        self.display_values = display_values

        self._create_response_function()

    @staticmethod
    def get_cal_options() -> dict[str, type['ImageCalibrationAbstract']]:
        """Available calibration objects that can be instantiated with load_from_hdf_guess_type()"""
        # import here to avoid circular references
        from opencsp.app.sofast.lib.ImageCalibrationGlobal import ImageCalibrationGlobal
        from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling

        return {'Global': ImageCalibrationGlobal, 'Scaling': ImageCalibrationScaling}

    @staticmethod
    @abstractmethod
    def get_calibration_name() -> str:
        """The name of the calibration class type (for saving/loading data)"""

    @abstractmethod
    def apply_to_images(self, measurement) -> ndarray:
        """
        Performs camera-projector brightness values calibration.

        Parameters
        ----------
        fringe_images : ndarray
            Measurement fringe images.

        Returns
        -------
        ndarray
            Calibrated fringe images, float.
        """
        pass

    def _create_response_function(self) -> None:
        """
        Creates response function (interpolation object) to convert from camera
        to display values.

        """
        # Calculate display values range
        display_min = self.display_values[0]
        display_max = self.display_values[-1]

        # Remove non-monotonically increasing values
        idxs = np.where(np.diff(self.camera_values) <= 0)[0]
        camera_values_clip = np.delete(self.camera_values, idxs)
        display_values_clip = np.delete(self.display_values, idxs)

        # Create interpolation function
        self.response_function = interpolate.interp1d(
            camera_values_clip, display_values_clip, bounds_error=False, fill_value=(display_min, display_max)
        )

    @classmethod
    def from_data(
        cls, images_cal: ndarray, display_values: ndarray, mask: ndarray | None = None, num_samps: int = 1000
    ) -> 'ImageCalibrationAbstract':
        """
        Calculates camera values from calibration images. Returns
        ImageCalibration object.

        Parameters
        ----------
        images_cal : ndarray
            Shape (n, m, N) ndarray. Shape of image = (n, m). N = number of
            calibration images.
        display_values : ndarray
            1D ndarray of display values.
        mask : [ndarray | None], optional
            Shape (n, m) boolean ndarray. Mask defining which pixels to
            consider. The default is None.
        num_samps : int, optional
            Calculate using num_samps of the brightest pixels. The default is
            1000.

        Returns
        -------
        ImageCalibrationAbstract

        """
        # Apply mask if given
        if mask is not None:
            images_cal = images_cal * mask[..., None]

        # Get brightest image
        im_1 = images_cal[..., -1].astype(int)

        # Get index of brightest pixel and N brightest pixels below
        N = im_1.size
        idx_1 = int(N * 0.99)
        idx_0 = idx_1 - num_samps
        if idx_0 < 0:
            idx_0 = 0
            warn(f'Number of samples smaller than n_samps. Using {idx_1:d} samples instead.', stacklevel=2)

        # Get brightness values corresponding to indices
        vals_sort = np.sort(im_1.flatten())
        value_1 = vals_sort[idx_1]
        value_0 = vals_sort[idx_0]

        # Get image indices of pixels of interest
        (y, x) = np.where((im_1 >= value_0) * (im_1 <= value_1))

        # Get ensemble of calibration curves
        camera_values = images_cal[y, x, :].astype(int)

        # Calculate calibration curve as mean of brightest curves
        camera_values = camera_values.mean(0)

        return cls(camera_values.astype(float), display_values.astype(float))

    def calculate_min_display_camera_values(self, derivative_thresh: float = 0.4) -> tuple[float, float]:
        """
        Calculates the minimum display and camera brightness values to be used
        in a valid calibration. Values lower than these values are too close to
        the noise floor of the camera.

        Parameters
        ----------
        derivative_thresh : float, optional
            Threshold of derivative to determine valid calibration range. The
            default is 0.4.

        Returns
        -------
        tuple[float, float]
            (display_min_value, camera_min_value).

        """
        # Calculate normalized differential
        camera_values_norm = self.camera_values.astype(float) / self.camera_values.astype(float).max()
        display_values_norm = self.display_values.astype(float) / self.display_values.astype(float).max()
        dy_dx = np.diff(camera_values_norm) / np.diff(display_values_norm)

        # Calculate data points that are below threshold
        mask = dy_dx < derivative_thresh
        if mask.sum() == 0:
            idx = 0
        else:
            idx = np.where(mask)[0][-1] + 1

        # Calculate minimum values
        camera_min_value = self.camera_values[idx]
        display_min_value = self.display_values[idx]
        return (display_min_value, camera_min_value)

    def plot_gray_levels(self) -> None:
        """
        Shows plot of gray levels calibration data. When the close() method of this instance is called (or this
        instance is destructed), the plot will be closed automatically."""
        title = 'Projector-Camera Calibration Curve'

        # Plot figure
        fig = fm.mpl_pyplot_figure()
        ax = fig.gca()
        ax.plot(self.display_values, self.camera_values)
        ax.set_xlabel('Display Values')
        ax.set_ylabel('Camera Values')
        ax.grid(True)
        ax.set_title('Projector-Camera Calibration Curve')

        # Track this figure, to be closed eventually
        self._register_plot(fig)

        # TODO use RenderControlFigureRecord:
        # def plot_gray_levels_cal(self, fig_record: fm.RenderControlFigureRecord = None) -> fm.RenderControlFigureRecord:
        # # Create the plot
        # if fig_record is None:
        #     fig_record = fm.setup_figure(
        #         rcfg.RenderControlFigure(),
        #         rca.image(grid=False),
        #         vs.view_spec_im(),
        #         title=title,
        #         code_tag=f"{__file__}",
        #         equal=False)

        # # Plot the gray levels
        # ax = fig_record.axis
        # ax.plot(self.display_values, self.camera_values)
        # ax.set_xlabel('Display Values')
        # ax.set_ylabel('Camera Values')
        # ax.grid(True)

        # return fig_record

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = '') -> 'ImageCalibrationAbstract':
        """
        Loads from HDF5 file

        Parameters
        ----------
        file : string
            HDF5 file to load

        """
        # Check calibration type
        datasets = [prefix + 'ImageCalibration/calibration_type']
        data = hdf5_tools.load_hdf5_datasets(datasets, file)
        calibration_name = cls.get_calibration_name()

        if data['calibration_type'] != calibration_name:
            raise ValueError(f'ImageCalibration file is not of type {calibration_name:s}')

        # Load grid data
        datasets = [prefix + 'ImageCalibration/camera_values', prefix + 'ImageCalibration/display_values']
        kwargs = hdf5_tools.load_hdf5_datasets(datasets, file)

        return cls(**kwargs)

    @staticmethod
    def load_from_hdf_guess_type(hdf5_file_path_name_ext: str, prefix: str = '') -> 'ImageCalibrationAbstract':
        """Loads saved results of a projector-camera intensity calibration, returning a calibration instance whose type
        is based on information stored in the given file.

        Raises:
        -------
        KeyError:
            If the necessary typing information isn't available in the given file
        ValueError:
            If the type of the calibration instance in the given file is unknown"""
        # Load file
        datasets = [prefix + 'ImageCalibration/calibration_type']
        cal_type = hdf5_tools.load_hdf5_datasets(datasets, hdf5_file_path_name_ext)['calibration_type']

        # import here to avoid circular references
        from opencsp.app.sofast.lib.ImageCalibrationGlobal import ImageCalibrationGlobal
        from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling

        # build a return value based on the information found in the file
        if cal_type == 'ImageCalibrationGlobal':
            calibration = ImageCalibrationGlobal.load_from_hdf(hdf5_file_path_name_ext, prefix)
        elif cal_type == 'ImageCalibrationScaling':
            calibration = ImageCalibrationScaling.load_from_hdf(hdf5_file_path_name_ext, prefix)
        else:
            lt.error_and_raise(ValueError, f'Selected calibration type, {cal_type}, not supported.')

        return calibration

    def save_to_hdf(self, file: str, prefix: str = '') -> None:
        """
        Saves to HDF file

        Parameters
        ----------
        file : string
            HDF5 file to save

        """
        datasets = [
            prefix + 'ImageCalibration/camera_values',
            prefix + 'ImageCalibration/display_values',
            prefix + 'ImageCalibration/calibration_type',
        ]
        data = [self.camera_values, self.display_values, self.get_calibration_name()]

        # Save data
        hdf5_tools.save_hdf5_datasets(data, datasets, file)
