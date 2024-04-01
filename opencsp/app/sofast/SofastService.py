""" Network API to control SOFAST image projection, image acquisition,
and data capture. Can capture datasets and return HDF5 files.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition as ImageAcquisition_DCAM
from opencsp.common.lib.camera.ImageAcquisition_DCAM_color import ImageAcquisition as ImageAcquisition_DCAM_color
from opencsp.common.lib.camera.ImageAcquisition_MSMF import ImageAcquisition as ImageAcquisition_MSMF
from opencsp.app.sofast.lib.ImageCalibrationAbstract import ImageCalibrationAbstract
from opencsp.app.sofast.lib.ImageCalibrationGlobal import ImageCalibrationGlobal
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
import opencsp.app.sofast.lib.SofastServiceCallback as ssc
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
import opencsp.common.lib.tool.exception_tools as et
import opencsp.common.lib.tool.hdf5_tools as h5
import opencsp.common.lib.tool.log_tools as lt


class SofastService:
    """Class that interfaces with SOFAST to run data acquisition and process results"""

    cam_options: dict[str, type[ImageAcquisitionAbstract]] = {
        'DCAM Mono': ImageAcquisition_DCAM,
        'DCAM Color': ImageAcquisition_DCAM_color,
        'MSMF Mono': ImageAcquisition_MSMF,
    }
    """ Defines camera objects to choose from (camera description, python type) """
    cal_options: dict[str, type[ImageCalibrationAbstract]] = {
        'Global': ImageCalibrationGlobal,
        'Scaling': ImageCalibrationScaling,
    }
    """ Available calibration objects to use """

    def __init__(self, callback: ssc.SofastServiceCallback = None) -> 'SofastService':
        self.callback = callback if callback else ssc.SofastServiceCallback()

        # Set defaults
        self._image_projection: ImageProjection = None
        self._image_acquisition: ImageAcquisitionAbstract = None
        self._calibration: ImageCalibrationAbstract = None
        self._system: SystemSofastFringe = None

    def __del__(self):
        with et.ignored(Exception):
            self.image_acquisition = None
        with et.ignored(Exception):
            self.image_projection = None
        with et.ignored(Exception):
            self._system.close_all()
        with et.ignored(Exception):
            self._system = None

    @property
    def system(self) -> SystemSofastFringe:
        """Get the sofast system object.

        Checks if the system object has been instantiated, or if the system
        object can be instantiated.

        Raises:
        -------
        RuntimeError:
            If the system instance hasn't been and can't be instantiated yet"""
        if self._system is None:
            if not self._load_system_elements():
                lt.error_and_raise(RuntimeError, 'Both ImageAcquisiton and ImageProjection must both be loaded.')

        return self._system

    @system.setter
    def system(self, val: SystemSofastFringe):
        if val is None:
            old = self._system
            self._system = None
            self.callback.on_system_unset(old)
        else:
            self._system = val
            self.callback.on_system_set(val)

    @property
    def image_projection(self) -> ImageProjection:
        return self._image_projection

    @image_projection.setter
    def image_projection(self, val: ImageProjection):
        if val is None:
            if self._image_projection is not None:
                old_ip = self._image_projection
                self._image_projection = None
                self.callback.on_image_projection_unset(old_ip)
                old_ip.close()
            self.system = None
        else:
            self._image_projection = val
            self.callback.on_image_projection_set(val)
            self._load_system_elements()

    @property
    def image_acquisition(self) -> ImageAcquisitionAbstract:
        return self._image_acquisition

    @image_acquisition.setter
    def image_acquisition(self, val: ImageAcquisitionAbstract):
        if val is None:
            if self._image_acquisition is not None:
                old_ia = self._image_acquisition
                self._image_acquisition = None
                self.callback.on_image_acquisition_unset(old_ia)
                old_ia.close()
            self.system = None
        else:
            self._image_acquisition = val
            self.callback.on_image_acquisition_set(val)
            self._load_system_elements()

    @property
    def calibration(self) -> ImageCalibrationAbstract:
        return self._calibration

    @calibration.setter
    def calibration(self, val: ImageCalibrationAbstract):
        if val is None:
            old = self._calibration
            self._calibration = None
            self.callback.on_calibration_unset(old)
        else:
            self._calibration = val
            self.callback.on_calibration_set(val)

    def get_frame(self) -> np.ndarray:
        """
        Captures frame from camera

        Returns
        -------
        frame : ndarray
            2D image from camera.

        """
        # Check camera is loaded
        if self.image_acquisition is None:
            lt.error_and_raise(RuntimeError, 'Camera is not connected.')
            return

        # Get frame
        frame = self.image_acquisition.get_frame()

        return frame

    def _load_system_elements(self) -> bool:
        """Loads the system instance, as appropriate.

        Checks if System object can be instantiated (if both
        ImageAcquisition and ImageProjection classes are loaded)
        """
        if self.image_acquisition is not None and self.image_projection is not None:
            self.system = SystemSofastFringe(self.image_projection, self.image_acquisition)
            return True
        return False

    def run_measurement(self, fringes: Fringes, on_done: Callable) -> None:
        """Runs data collect and saved data."""

        # Get minimum display value from calibration
        min_disp_value = self.calibration.calculate_min_display_camera_values()[0]

        # Load fringes
        self.system.load_fringes(fringes, min_disp_value)

        # Capture images
        self.system.capture_mask_and_fringe_images(on_done)

    def run_exposure_cal(self) -> None:
        """Runs camera exposure calibration"""
        if self.image_acquisition is None:
            lt.error_and_raise(RuntimeError, 'Camera must be connected.')

        # If only camera is loaded
        if self.image_projection is None:
            lt.info('Running calibration without displayed white image.')
            self.image_acquisition.calibrate_exposure()
        else:
            lt.info('Running calibration with displayed white image.')
            run_next = self.image_projection.show_crosshairs
            self.system.run_camera_exposure_calibration(run_next)

    def load_gray_levels_cal(self, hdf5_file_path_name_ext: str) -> None:
        """Loads saved results of a projector-camera intensity calibration"""
        # Load file
        datasets = ['ImageCalibration/calibration_type']
        cal_type = h5.load_hdf5_datasets(datasets, hdf5_file_path_name_ext)['calibration_type']

        if cal_type == 'ImageCalibrationGlobal':
            self.calibration = ImageCalibrationGlobal.load_from_hdf(hdf5_file_path_name_ext)
        elif cal_type == 'ImageCalibrationScaling':
            self.calibration = ImageCalibrationScaling.load_from_hdf(hdf5_file_path_name_ext)
        else:
            lt.error_and_raise(ValueError, f'Selected calibration type, {cal_type}, not supported.')

    def plot_gray_levels_cal(self) -> None:
        """Shows plot of gray levels calibration data"""
        title = 'Projector-Camera Calibration Curve'

        # Plot figure
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(self.calibration.display_values, self.calibration.camera_values)
        ax.set_xlabel('Display Values')
        ax.set_ylabel('Camera Values')
        ax.grid(True)
        ax.set_title('Projector-Camera Calibration Curve')

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
        # ax.plot(self.calibration.display_values, self.calibration.camera_values)
        # ax.set_xlabel('Display Values')
        # ax.set_ylabel('Camera Values')
        # ax.grid(True)

        # return fig_record

    def get_exposure(self) -> float | None:
        if self.image_acquisition is None:
            return None
        return self.image_acquisition.exposure_time

    def set_exposure(self, new_exp: float) -> None:
        """Sets camera exposure time value to the given value"""
        self.image_acquisition.exposure_time = new_exp

    def close(self) -> None:
        """
        Closes all windows

        """
        # Close image projection
        with et.ignored(Exception):
            self.image_projection = None

        # Close image acquisition
        with et.ignored(Exception):
            self.image_acquisition = None
