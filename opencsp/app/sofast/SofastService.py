""" Network API to control SOFAST image projection, image acquisition,
and data capture. Can capture datasets and return HDF5 files.
"""

import copy
from typing import Callable

import matplotlib.backend_bases
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

import opencsp.app.sofast.lib.MeasurementSofastFringe as msf
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
import opencsp.common.lib.render.figure_management as fm
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
        """Service object for standard callibration, measurement, and analysis with the SOFAST tool.

        The goal of this class is to provide a standard interface for interactive use of SOFAST, such as from the SOFAST
        GUI or from a RESTful interface. This class also tracks instances of other necessary classes, such as for a
        camera or projector. When these instances are free'd they will have their own 'close()' methods evaluated as
        well. The necessary classes are free'd in several situations, including at least:
         - in SofastService.close()
         - when the attribute is unset
         - when SofastService is destructed

        There are currently two SOFAST systems 'fringe' and 'fixed'. So far this server interacts with the 'fringe'
        system, with possible extension to use the 'fixed' system in the future.

        Params:
        -------
        callback : SofastServiceCallback
            Callbacks for this instance, for when attributes get set or unset.
        """
        self.callback = callback if callback else ssc.SofastServiceCallback()

        # Set defaults
        self._image_projection: ImageProjection = None
        self._image_acquisition: ImageAcquisitionAbstract = None
        self._calibration: ImageCalibrationAbstract = None
        self._system: SystemSofastFringe = None

        # Track plots that will need to be eventually closed
        self._open_plots: list[matplotlib.figure.Figure] = []

    def __del__(self):
        with et.ignored(Exception):
            self.close()

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
        """Set or unset the system instance. Does NOT call system.close() when being unset."""
        if val is None:
            old = self._system
            self._system = None
            self.callback.on_system_unset(old)
            # 'close_all()' also closes the acquisition and projection instances, which might not be desired
            # old.close_all()
        else:
            self._system = val
            self.callback.on_system_set(val)

    @property
    def image_projection(self) -> ImageProjection | None:
        """The image projection (projector) instance. None if not yet set."""
        return self._image_projection

    @image_projection.setter
    def image_projection(self, val: ImageProjection):
        """Set or unset the image projection (projector) instance. If being unset, then the instance's 'close()' method
        will be evaluated."""
        if val is None:
            if self._image_projection is not None:
                old_ip = self._image_projection
                self._image_projection = None
                self.callback.on_image_projection_unset(old_ip)
                old_ip.close()  # release system resources
            self.system = None
        else:
            self._image_projection = val
            self.callback.on_image_projection_set(val)
            self._load_system_elements()

    @property
    def image_acquisition(self) -> ImageAcquisitionAbstract | None:
        """The image acquisition (camera) instance. None if not yet set."""
        return self._image_acquisition

    @image_acquisition.setter
    def image_acquisition(self, val: ImageAcquisitionAbstract):
        """Set or unset the image acquisition (camera) instance. If being unset, then the instance's 'close()' method
        will be evaluated."""
        if val is None:
            if self._image_acquisition is not None:
                old_ia = self._image_acquisition
                self._image_acquisition = None
                self.callback.on_image_acquisition_unset(old_ia)
                old_ia.close()  # release system resources
            self.system = None
        else:
            self._image_acquisition = val
            self.callback.on_image_acquisition_set(val)
            self._load_system_elements()

    @property
    def calibration(self) -> ImageCalibrationAbstract | None:
        """The grayscale calibration instance. None if not yet set"""
        return self._calibration

    @calibration.setter
    def calibration(self, val: ImageCalibrationAbstract):
        """Set or unset the system instance. Does NOT call calibration.close() when being unset."""
        if val is None:
            old = self._calibration
            self._calibration = None
            self.callback.on_calibration_unset(old)
            # old.close() # no such method
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

    def run_measurement(self, fringes: Fringes, on_done: Callable = None) -> None:
        """Runs data collection with the given fringes.

        Once the data has been captured, the images will be available from the system instance. Similarly, the data can
        then be processed with the system instance, such as with SystemSofastFringe.get_measurements().
        """
        # Get minimum display value from calibration
        if self.calibration is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in SofastService.run_measurement(): "
                + "must run or provide calibration before starting a measurement.",
            )
        min_disp_value = self.calibration.calculate_min_display_camera_values()[0]

        # Load fringes
        self.system.load_fringes(fringes, min_disp_value)

        # Capture images
        self.system.capture_mask_and_fringe_images(on_done)

    def run_exposure_cal(self) -> None:
        """Runs camera exposure calibration. This adjusts the exposure time of the camera to keep the pixels from being
        under or over saturated."""
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
        """Loads saved results of a projector-camera intensity calibration, which can then be accessed via the
        'calibration' instance."""
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
        """Shows plot of gray levels calibration data. When the close() method of this instance is called (or this
        instance is destructed), the plot will be closed automatically."""
        title = 'Projector-Camera Calibration Curve'

        # Check that we have a calibration
        if self.calibration is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in SofastService.plot_gray_levels_cal(): "
                + "must run or provide calibration before trying to plot calibration.",
            )

        # Plot figure
        fig = fm._mpl_pyplot_figure()
        ax = fig.gca()
        ax.plot(self.calibration.display_values, self.calibration.camera_values)
        ax.set_xlabel('Display Values')
        ax.set_ylabel('Camera Values')
        ax.grid(True)
        ax.set_title('Projector-Camera Calibration Curve')

        # Track this figure, to be closed eventually
        self._open_plots.append(fig)
        fig.canvas.mpl_connect('close_event', self._on_plot_closed)

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

    def _on_plot_closed(self, event: matplotlib.backend_bases.CloseEvent):
        """Stop tracking plots that are still open when the plots get closed."""
        to_remove = None
        for fig in self._open_plots:
            if fig.canvas == event.canvas:
                to_remove = fig
                break
        if to_remove is not None:
            self._open_plots.remove(to_remove)

    def get_exposure(self) -> float | None:
        """Returns the exposure time of the camera (seconds)."""
        if self.image_acquisition is None:
            lt.error_and_raise(
                RuntimeError,
                "Error in SofastService.get_exposure(): "
                + "must initialize image acquisition (camera) before attempting to get the exposure.",
            )
        return self.image_acquisition.exposure_time

    def set_exposure(self, new_exp: float) -> None:
        """Sets camera exposure time value to the given value (seconds)"""
        self.image_acquisition.exposure_time = new_exp

    def close(self) -> None:
        """
        Closes all windows and releases all contained objects. This includes:
         - image_projection
         - image_acquisition
         - system
         - plots
        """
        # Note that this method is called from the destructor, and so must be resilient to all the possible errors
        # therein. More information can be found in the python documentation at
        # https://docs.python.org/3/reference/datamodel.html#object.__del__

        # Close image projection
        with et.ignored(Exception):
            self.image_projection = None

        # Close image acquisition
        with et.ignored(Exception):
            self.image_acquisition = None

        # Close system
        with et.ignored(Exception):
            self._system.close_all()
        self.system = None

        # Close plots
        for fig in copy.copy(self._open_plots):
            with et.ignored(Exception):
                plt.close(fig)
