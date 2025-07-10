"""
Sofast Interface Module for Optical Measurement Processing.

This module provides the `SofastInterface` class, which facilitates the
configuration and execution of optical measurements using the SOFAST framework.
It includes functionality for initializing measurement systems, processing
fringe and fixed measurements, and handling user interactions. The module
also supports saving and loading calibration data, displaying camera images,
and managing the overall workflow of optical measurements.

Usage:

To use this module, create an instance of the `SofastInterface` class
with an image acquisition object, and then call the appropriate methods
to run measurements and process data. See /example/ for more information.

TODO:

Refactor so that common code is separate from end-of-file input/execution block, move this to common.

Make this a "kitchen sink" file, which includes all aspects:

    1. Data collection:

    - Fringe measurement
    - Fixed measurement with projector
    - Fixed measurement with printed target in ambient lght

    2. Data analysis -- finding the best-fit instance of the class of shapes.

    3. Fitting to a desired reference optical shape.  (Make this an enhancement issue added to SOFAST.  Then, another file?)

    4. Plotting/ray tracing

    5. (Suggest puttting calibration in another file.)

    - This file contains 1 and 2.

"""

import glob
import os.path
import os
from dataclasses import dataclass
import datetime as dt
import sys

import matplotlib.pyplot as plt

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DisplayShape import DisplayShape
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.app.sofast.lib.SystemSofastFringe import SystemSofastFringe
from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.camera.ImageAcquisitionAbstract import ImageAcquisitionAbstract
from opencsp.common.lib.camera.image_processing import highlight_saturation
from opencsp.common.lib.camera.LiveView import LiveView
from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.tool.time_date_tools import current_date_time_string_forfile as timestamp


@dataclass
class SofastFixedRunData:
    """Data class holding data required to run Sofast Fixed"""

    origin: Vxy
    """The xy pixel location in the camera image of the center of the xy_known dot"""
    xy_known: tuple[int, int] = (0, 0)
    """The xy index of a known dot location seen by the camera"""
    pattern_width = 3
    """Fixed pattern dot width, pixels"""
    pattern_spacing = 6
    """Fixed pattern dot spacing between edges of neighboring dots, pixels."""


@dataclass
class SofastCommonRunData:
    """Data class holding data required to run Sofast Fixed and Fringe"""

    measure_point_optic: Vxyz
    """The measurement point in 3D space for the optical system."""
    dist_optic_screen: float
    """The distance from the optic to the screen, in meters."""
    name_optic: str
    """A descriptive name for the optical system."""


@dataclass
class SofastCommonProcessData:
    """Data class holding data common to processing captured Sofast Fixed or Fringe measurements"""

    facet_definition: DefinitionFacet
    """The definition of the optical facet being measured."""
    camera: Camera
    """The camera calibration parameters for the camera used during measurements."""
    spatial_orientation: SpatialOrientation
    """The spatial orientation of the measurement setup."""


@dataclass
class SofastFringeProcessData:
    """Data class holding data required to process captured Sofast Fringe measurements"""

    display_shape: DisplayShape
    """The display shape used in the fringe measurement."""
    surface_2d: Surface2DParabolic
    """The 2D surface model used for processing fringe data."""


@dataclass
class SofastFixedProcessData:
    """Data class holding data required to process captured Sofast Fixed measurements"""

    fixed_pattern_dot_locs: DotLocationsFixedPattern
    """The locations of fixed pattern dots used in measurements."""
    surface_2d: Surface2DParabolic
    """The 2D surface model used for processing fixed data."""


@dataclass
class _Paths:
    dir_save_fringe: str = ""
    """Location to save Sofast Fringe measurement data"""
    dir_save_fixed: str = ""
    """Location to save Sofast Fixed measurement data"""
    dir_save_fringe_calibration: str = ""
    """Location to save Sofast Fringe calibration data"""


class SofastInterface:
    """
    Interface for managing SOFAST optical measurement systems.

    The `SofastInterface` class provides methods to initialize and run
    measurements using the SOFAST framework. It manages the configuration
    of fringe and fixed measurement systems, processes measurement data,
    and handles user interactions. The class also supports saving and
    loading calibration data and displaying camera images.

    Parameters
    ----------
    image_acquisition : ImageAcquisitionAbstract
        Object responsible for acquiring images from the camera.
    """

    def __init__(self, image_acquisition: ImageAcquisitionAbstract) -> "SofastInterface":
        """Instantiates SofastInterface

        Parameters
        ----------
        image_acquisition : ImageAcquisitionAbstract
            Image acquisition object connected to camera.

        Returns
        -------
        SofastInterface
        """
        # Common parameters
        self.image_acquisition = image_acquisition
        self.image_projection = ImageProjection.instance()
        self.plotting: StandardPlotOutput = StandardPlotOutput()

        # Sofast system objects
        self.system_fixed: SystemSofastFixed = None
        self.system_fringe: SystemSofastFringe = None
        self.timestamp: dt.datetime = None
        self._process_sofast_fixed: ProcessSofastFixed = None

        # User input objects
        self.data_sofast_common_run: SofastCommonRunData = None
        self.data_sofast_fixed_run: SofastFixedRunData = None
        self.data_sofast_common_proccess: SofastCommonProcessData = None
        self.data_sofast_fringe_process: SofastFringeProcessData = None
        self.data_sofast_fixed_process: SofastFixedProcessData = None
        self.paths = _Paths()

    @property
    def file_timestamp(self) -> str:
        """Returns current run timestamp in string format"""
        return self.timestamp.strftime(r"%Y-%m-%d_%H_%M_%S_%f")

    def run_cli(self) -> None:
        """Runs command line Sofast"""
        self._func_user_input()

    def initialize_sofast_fringe(self, fringes: Fringes) -> None:
        """Initializes sofast fringe system

        Parameters
        ----------
        fringes : Fringes
            Input fringe object to initialize Sofast Fringe system
        """
        self.system_fringe = SystemSofastFringe(self.image_acquisition)
        self.system_fringe.set_fringes(fringes)

    def initialize_sofast_fixed(self) -> None:
        """Initializes sofast fixed system"""
        if self._check_fixed_run_data_loaded():
            self.system_fixed = SystemSofastFixed(self.image_acquisition)
            self.system_fixed.set_pattern_parameters(
                self.data_sofast_fixed_run.pattern_width, self.data_sofast_fixed_run.pattern_spacing
            )

    def func_run_fringe_measurement(self) -> None:
        """Runs sofast fringe measurement"""
        lt.info(f"{timestamp():s} Starting Sofast Fringe measurement")

        def _on_done():
            lt.info(f"{timestamp():s} Completed Sofast Fringe measurement")
            self.system_fringe.run_next_in_queue()

        self.system_fringe.run_measurement(_on_done)

    def func_run_fixed_measurement(self) -> None:
        """Runs Sofast Fixed measurement"""
        lt.info(f"{timestamp():s} Starting Sofast Fixed measurement")

        def _f1():
            lt.info(f"{timestamp():s} Completed Sofast Fixed measurement")
            self.system_fixed.run_next_in_queue()

        self.system_fixed.prepend_to_queue([self.system_fixed.run_measurement, _f1])
        self.system_fixed.run_next_in_queue()

    def func_show_crosshairs_fringe(self) -> None:
        """Shows crosshairs and run next in Sofast fringe queue after a 0.2s wait"""
        self.image_projection.show_crosshairs()
        self.system_fringe.root.after(200, self.system_fringe.run_next_in_queue)

    def func_process_sofast_fringe_data(self) -> None:
        """Processes Sofast Fringe data"""
        lt.info(f"{timestamp():s} Starting Sofast Fringe data processing")

        # Get Measurement object
        measurement = self.system_fringe.get_measurements(
            self.data_sofast_common_run.measure_point_optic,
            self.data_sofast_common_run.dist_optic_screen,
            self.data_sofast_common_run.name_optic,
        )[0]

        # Calibrate fringe images
        measurement.calibrate_fringe_images(self.system_fringe.calibration)

        # Instantiate ProcessSofastFringe
        sofast = ProcessSofastFringe(
            measurement,
            self.data_sofast_common_proccess.spatial_orientation,
            self.data_sofast_common_proccess.camera,
            self.data_sofast_fringe_process.display_shape,
        )

        # Process
        sofast.process_optic_singlefacet(
            self.data_sofast_common_proccess.facet_definition, self.data_sofast_fringe_process.surface_2d
        )

        lt.info(f"{timestamp():s} Completed Sofast Fringe data processing")

        # Get optic
        mirror = sofast.get_optic().mirror

        # Plot optic
        lt.debug(f"{timestamp():s} Plotting Sofast Fringe data")
        output_dir = os.path.join(self.paths.dir_save_fringe, self.file_timestamp)
        os.makedirs(output_dir, exist_ok=True)
        self.plotting.optic_measured = mirror
        self.plotting.options_file_output.output_dir = output_dir
        self.plotting.plot()

        # Save processed sofast data
        sofast.save_to_hdf(f"{self.paths.dir_save_fringe:s}/{self.file_timestamp:s}_data_sofast_fringe.h5")
        lt.debug(f"{timestamp():s} Sofast Fringe data saved to HDF5")

        # Continue
        self.system_fringe.run_next_in_queue()

    def func_process_sofast_fixed_data(self) -> None:
        """Process Sofast Fixed data"""
        lt.info(f"{timestamp():s} Starting Sofast Fixed data processing")
        # Instantiate sofast processing object
        process_sofast_fixed = ProcessSofastFixed(
            self.data_sofast_common_proccess.spatial_orientation,
            self.data_sofast_common_proccess.camera,
            self.data_sofast_fixed_process.fixed_pattern_dot_locs,
        )

        # Get Measurement object
        measurement = self.system_fixed.get_measurement(
            self.data_sofast_common_run.measure_point_optic,
            self.data_sofast_common_run.dist_optic_screen,
            self.data_sofast_fixed_run.origin,
            name=self.data_sofast_common_run.name_optic,
        )
        process_sofast_fixed.load_measurement_data(measurement)

        # Process
        process_sofast_fixed.process_single_facet_optic(
            self.data_sofast_common_proccess.facet_definition,
            self.data_sofast_fixed_process.surface_2d,
            self.data_sofast_fixed_run.origin,
            xy_known=self.data_sofast_fixed_run.xy_known,
        )

        lt.info(f"{timestamp():s} Completed Sofast Fixed data processing")

        # Get optic
        mirror = process_sofast_fixed.get_optic()

        # Plot optic
        lt.debug(f"{timestamp():s} Plotting Sofast Fixed data")
        output_dir = os.path.join(self.paths.dir_save_fringe, self.file_timestamp)
        os.makedirs(output_dir, exist_ok=True)
        self.plotting.optic_measured = mirror
        self.plotting.options_file_output.output_dir = output_dir
        self.plotting.plot()

        # Save processed sofast data
        process_sofast_fixed.save_to_hdf(f"{self.paths.dir_save_fixed:s}/{self.file_timestamp:s}_data_sofast_fixed.h5")
        lt.debug(f"{timestamp():s} Sofast Fixed data saved to HDF5")

        # Continue
        self.system_fixed.run_next_in_queue()

    def func_save_measurement_fringe(self) -> None:
        """Saves measurement to HDF file"""
        measurement = self.system_fringe.get_measurements(
            self.data_sofast_common_run.measure_point_optic,
            self.data_sofast_common_run.dist_optic_screen,
            self.data_sofast_common_run.name_optic,
        )[0]
        file = f"{self.paths.dir_save_fringe:s}/{self.file_timestamp:s}_measurement_fringe.h5"
        measurement.save_to_hdf(file)
        self.system_fringe.calibration.save_to_hdf(file)
        self.system_fringe.run_next_in_queue()

    def func_save_measurement_fixed(self) -> None:
        """Save fixed measurement files"""
        measurement = self.system_fixed.get_measurement(
            self.data_sofast_common_run.measure_point_optic,
            self.data_sofast_common_run.dist_optic_screen,
            self.data_sofast_fixed_run.origin,
            name=self.data_sofast_common_run.name_optic,
        )
        measurement.save_to_hdf(f"{self.paths.dir_save_fixed:s}/{self.file_timestamp:s}_measurement_fixed.h5")
        self.system_fixed.run_next_in_queue()

    def func_load_last_sofast_fringe_image_cal(self) -> None:
        """Loads last ImageCalibration object"""
        # Find file
        files = glob.glob(os.path.join(self.paths.dir_save_fringe_calibration, "image_calibration_scaling*.h5"))
        files.sort()

        if len(files) == 0:
            lt.error(f"No previous calibration files found in {self.paths.dir_save_fringe_calibration}")
            return

        # Get latest file and set
        file = files[-1]
        image_calibration = ImageCalibrationScaling.load_from_hdf(file)
        self.system_fringe.set_calibration(image_calibration)
        lt.info(f"{timestamp()} Loaded image calibration file: {file}")

    def func_gray_levels_cal(self) -> None:
        """Runs gray level calibration sequence"""
        file = os.path.join(self.paths.dir_save_fringe_calibration, f"image_calibration_scaling_{timestamp():s}.h5")
        self.system_fringe.run_gray_levels_cal(
            ImageCalibrationScaling,
            file,
            on_processed=self._func_user_input,
            on_processing=self.func_show_crosshairs_fringe,
        )

    def show_cam_image(self) -> None:
        """Shows a camera image"""
        image = self.image_acquisition.get_frame()
        image_proc = highlight_saturation(image, self.image_acquisition.max_value)
        plt.imshow(image_proc)
        plt.show()

    def show_live_view(self) -> None:
        """Shows live view window"""
        LiveView(self.image_acquisition)

    def _check_fixed_run_data_loaded(self) -> bool:
        if self.data_sofast_fixed_run is None:
            lt.error(f"{timestamp()} Sofast Fixed processing data not loaded")
            return False
        return True

    def _check_common_run_data_loaded(self) -> bool:
        if self.data_sofast_common_run is None:
            lt.error(f"{timestamp()} Sofast Fringe processing data not loaded")
            return False
        return True

    def _check_fringe_system_loaded(self) -> bool:
        if self.system_fringe is None:
            lt.error(f"{timestamp()} Sofast Fringe system not initizlized")
            return False
        if not self._check_common_run_data_loaded():
            return False
        return True

    def _check_fixed_system_loaded(self) -> bool:
        if self.system_fixed is None:
            lt.error(f"{timestamp()} Sofast Fixed system not initialized")
            return False
        if not self._check_common_run_data_loaded():
            return False
        if not self._check_fixed_run_data_loaded():
            return False
        return True

    def _func_user_input(self) -> None:
        # Get user input
        retval = input("> ")
        self.timestamp = dt.datetime.now()
        lt.debug(f"{timestamp():s} user input: {retval:s}")

        try:
            self._run_given_input(retval)
        except Exception as error:
            lt.error(repr(error))

    def _run_given_input(self, retval: str) -> None:
        # Run fringe measurement and process/save
        if retval == "help":
            print("\n")
            print("Value      Command")
            print("------------------")
            print("mrp        run Sofast Fringe measurement and process/save")
            print("mrs        run Sofast Fringe measurement and save only")
            print("mip        run Sofast Fixed measurement and process/save")
            print("mis        run Sofast Fixed measurement and save only")
            print("ce         calibrate camera exposure")
            print("cr         calibrate camera-projector response")
            print("lr         load most recent camera-projector response calibration file")
            print("q          quit and close all")
            print("im         show image from camera.")
            print("lv         shows camera live view")
            print("cross      show crosshairs")
            self._func_user_input()
        elif retval == "mrp":
            lt.info(f"{timestamp()} Running Sofast Fringe measurement and processing/saving data")
            if self._check_fringe_system_loaded():
                funcs = [
                    self.func_run_fringe_measurement,
                    self.func_show_crosshairs_fringe,
                    self.func_process_sofast_fringe_data,
                    self.func_save_measurement_fringe,
                    self._func_user_input,
                ]
                self.system_fringe.set_queue(funcs)
                self.system_fringe.run()
            else:
                self._func_user_input()
        # Run fringe measurement and save
        elif retval == "mrs":
            lt.info(f"{timestamp()} Running Sofast Fringe measurement and saving data")
            if self._check_fringe_system_loaded():
                funcs = [
                    self.func_run_fringe_measurement,
                    self.func_show_crosshairs_fringe,
                    self.func_save_measurement_fringe,
                    self._func_user_input,
                ]
                self.system_fringe.set_queue(funcs)
                self.system_fringe.run()
            else:
                self._func_user_input()
        # Run fixed measurement and process/save
        elif retval == "mip":
            lt.info(f"{timestamp()} Running Sofast Fixed measurement and processing/saving data")
            if self._check_fixed_system_loaded():
                funcs = [
                    self.func_run_fixed_measurement,
                    self.func_process_sofast_fixed_data,
                    self.func_save_measurement_fixed,
                    self._func_user_input,
                ]
                self.system_fixed.set_queue(funcs)
                self.system_fixed.run()
            else:
                self._func_user_input()
        # Run fixed measurement and save
        elif retval == "mis":
            lt.info(f"{timestamp()} Running Sofast Fixed measurement and saving data")
            if self._check_fixed_system_loaded():
                funcs = [self.func_run_fixed_measurement, self.func_save_measurement_fixed, self._func_user_input]
                self.system_fixed.set_queue(funcs)
                self.system_fixed.run()
            else:
                self._func_user_input()
        # Calibrate exposure time
        elif retval == "ce":
            lt.info(f"{timestamp()} Calibrating camera exposure")
            self.image_acquisition.calibrate_exposure()
            self._func_user_input()
        # Calibrate response
        elif retval == "cr":
            lt.info(f"{timestamp()} Calibrating camera-projector response")
            if self._check_fringe_system_loaded():
                funcs = [self.func_gray_levels_cal]
                self.system_fringe.set_queue(funcs)
                self.system_fringe.run()
            else:
                self._func_user_input()
        # Load last fringe calibration file
        elif retval == "lr":
            lt.info(f"{timestamp()} Loading response calibration")
            if self._check_fringe_system_loaded():
                self.func_load_last_sofast_fringe_image_cal()
            self._func_user_input()
        # Quit
        elif retval == "q":
            lt.info(f"{timestamp():s} quitting")
            if self.system_fringe is not None:
                self.system_fringe.close_all()
            if self.system_fixed is not None:
                self.system_fixed.close_all()
            sys.exit(0)
        # Show single camera image
        elif retval == "im":
            self.show_cam_image()
            self._func_user_input()
        # Show camera live view
        elif retval == "lv":
            self.show_live_view()
            self._func_user_input()
        # Project crosshairs
        elif retval == "cross":
            self.image_projection.show_crosshairs()
            self._func_user_input()
        else:
            lt.error(f"{timestamp()} Command, {retval}, not recognized")
            self._func_user_input()
