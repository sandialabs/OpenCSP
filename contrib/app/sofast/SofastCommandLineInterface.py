"""A script that runs SOFAST in a command-line manner. We recommend copying this file into
your working directory and modifying it there. To run SOFAST, perform the following steps:

1. Navigate to the bottom of this file and fill in all user-input data specific to your SOFAST run.
2. Type "help" to display help message
3. Run SOFAST. Collect only measurement data, or (optionally) process data.

NOTE: To update any of the parameters in the bottom of the file, the CLI must be
restarted for changes to take effect.

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

(Suggest puttting calibration in another file.)

This file contains 1 and 2.

In output logs, include user input.
"""

import glob
from os.path import join, dirname, abspath

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
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition
from opencsp.common.lib.camera.image_processing import highlight_saturation
from opencsp.common.lib.camera.LiveView import LiveView
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.log_tools as lt
from opencsp.common.lib.tool.time_date_tools import current_date_time_string_forfile as timestamp


class SofastCommandLineInterface:
    """Sofast Command Line Interface class."""

    def __init__(self) -> "SofastCommandLineInterface":
        # Common sofast parameters
        self.image_acquisition: ImageAcquisition = None
        self.camera: Camera = None
        self.facet_definition: DefinitionFacet = None
        self.spatial_orientation: SpatialOrientation = None
        self.measure_point_optic: Vxyz = None
        self.dist_optic_screen: float = None
        self.name_optic: str = None

        self.image_projection = ImageProjection.instance()
        self.res_plot = 0.002  # meters
        """Resolution of slope map image (meters)"""
        self.colorbar_limit = 7  # mrad
        """Colorbar limits in slope magnitude plot (mrad)"""

        # Sofast fringe specific
        self.system_fringe: SystemSofastFringe = None
        self.display_shape: DisplayShape = None
        self.surface_fringe: Surface2DParabolic = None
        self.timestamp_fringe_measurement = None

        # Sofast fixed specific
        self.system_fixed: SystemSofastFixed = None
        self.process_sofast_fixed: ProcessSofastFixed = None
        self.fixed_pattern_dot_locs: DotLocationsFixedPattern = None
        self.surface_fixed: Surface2DParabolic = None
        self.origin: Vxy = None
        self.timestamp_fixed_measurement = None
        self.pattern_width = 3
        """Fixed pattern dot width, pixels"""
        self.pattern_spacing = 6
        """Fixed pattern dot spacing, pixels"""

        # Save directories
        self.dir_save_fringe: str = ""
        """Location to save Sofast Fringe measurement data"""
        self.dir_save_fixed: str = ""
        """Location to save Sofast Fixed measurement data"""
        self.dir_save_fringe_calibration: str = ""
        """Location to save Sofast Fringe calibration data"""

    def run(self) -> None:
        """Runs command line Sofast"""
        self.func_user_input()

    def set_common_data(
        self,
        image_acquisition: ImageAcquisition,
        image_projection: ImageProjection,
        camera: Camera,
        facet_definition: DefinitionFacet,
        spatial_orientation: SpatialOrientation,
        measure_point_optic: Vxyz,
        dist_optic_screen: float,
        name_optic: str,
    ) -> None:
        """Sets common parametes for Sofast Fringe and Fixed

        Parameters
        ----------
        image_acquisition : ImageAcquisition
            ImageAcquisition object
        image_projection : ImageProjection
            Image projection object
        camera : Camera
            Camera calibration object
        facet_definition : DefinitionFacet
            Facet definition object
        spatial_orientation : SpatialOrientation
            Calibrated SpatialOrientation object
        measure_point_optic : Vxyz
            Measure point location on optic
        dist_optic_screen : float
            Distance from measure point to screen center
        name_optic : str
            Name of optic
        """
        self.image_acquisition = image_acquisition
        self.image_projection = image_projection
        self.camera = camera
        self.facet_definition = facet_definition
        self.spatial_orientation = spatial_orientation
        self.measure_point_optic = measure_point_optic
        self.dist_optic_screen = dist_optic_screen
        self.name_optic = name_optic

    def set_sofast_fringe_data(
        self, display_shape: DisplayShape, fringes: Fringes, surface_fringe: Surface2DParabolic
    ) -> None:
        """Loads Sofast Fringe specific objects

        Parameters
        ----------
        display_shape : DisplayShape
            Calibrated DisplayShape object
        fringes : Fringes
            Fringe objects to display
        surface_fringe : Surface2DParabolic
            Surface to use when processing Sofast Fringe data
        """
        self.system_fringe = SystemSofastFringe(self.image_acquisition)
        self.system_fringe.set_fringes(fringes)
        self.display_shape = display_shape
        self.surface_fringe = surface_fringe

    def set_sofast_fixed_data(
        self, fixed_pattern_dot_locs: DotLocationsFixedPattern, origin: Vxy, surface_fixed: Surface2DParabolic
    ) -> None:
        """Loads Sofast Fixed specific objects

        Parameters
        ----------
        fixed_pattern_dot_locs : DotLocationsFixedPattern
            Calibrated dot locations object
        origin : Vxy
            Origin dot location in image, pixels
        surface_fixed : Surface2DParabolic
            Surface to use when processing Sofast Fringe data
        """
        self.system_fixed = SystemSofastFixed(self.image_acquisition)
        self.system_fixed.set_pattern_parameters(self.pattern_width, self.pattern_spacing)
        self.fixed_pattern_dot_locs = fixed_pattern_dot_locs
        self.origin = origin
        self.surface_fixed = surface_fixed

        self.process_sofast_fixed = ProcessSofastFixed(self.spatial_orientation, self.camera, fixed_pattern_dot_locs)

    def func_run_fringe_measurement(self) -> None:
        """Runs sofast fringe measurement"""
        lt.info(f"{timestamp():s} Starting Sofast Fringe measurement")
        self.timestamp_fringe_measurement = timestamp()

        def _on_done():
            lt.info(f"{timestamp():s} Completed Sofast Fringe measurement")
            self.system_fringe.run_next_in_queue()

        self.system_fringe.run_measurement(_on_done)

    def func_run_fixed_measurement(self) -> None:
        """Runs Sofast Fixed measurement"""
        lt.info(f"{timestamp():s} Starting Sofast Fixed measurement")
        self.timestamp_fixed_measurement = timestamp()

        def _f1():
            lt.info(f"{timestamp():s} Completed Sofast Fixed measurement")
            self.system_fixed.run_next_in_queue()

        self.system_fixed.prepend_to_queue([self.system_fixed.run_measurement, _f1])
        self.system_fixed.run_next_in_queue()

    def func_show_crosshairs_fringe(self):
        """Shows crosshairs and run next in Sofast fringe queue after a 0.2s wait"""
        self.image_projection.show_crosshairs()
        self.system_fringe.root.after(200, self.system_fringe.run_next_in_queue)

    def func_process_sofast_fringe_data(self):
        """Processes Sofast Fringe data"""
        lt.info(f"{timestamp():s} Starting Sofast Fringe data processing")

        # Get Measurement object
        measurement = self.system_fringe.get_measurements(
            self.measure_point_optic, self.dist_optic_screen, self.name_optic
        )[0]

        # Calibrate fringe images
        measurement.calibrate_fringe_images(self.system_fringe.calibration)

        # Instantiate ProcessSofastFringe
        sofast = ProcessSofastFringe(measurement, self.spatial_orientation, self.camera, self.display_shape)

        # Process
        sofast.process_optic_singlefacet(self.facet_definition, self.surface_fringe)

        lt.info(f"{timestamp():s} Completed Sofast Fringe data processing")

        # Plot optic
        mirror = sofast.get_optic().mirror

        lt.debug(f"{timestamp():s} Plotting Sofast Fringe data")
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()
        fig_record = fm.setup_figure(figure_control, axis_control_m, title="")
        mirror.plot_orthorectified_slope(self.res_plot, clim=self.colorbar_limit, axis=fig_record.axis)
        fig_record.save(self.dir_save_fringe, f"{self.timestamp_fringe_measurement:s}_slope_magnitude_fringe", "png")
        fig_record.close()

        # Save processed sofast data
        sofast.save_to_hdf(f"{self.dir_save_fringe:s}/{self.timestamp_fringe_measurement:s}_data_sofast_fringe.h5")
        lt.debug(f"{timestamp():s} Sofast Fringe data saved to HDF5")

        # Continue
        self.system_fringe.run_next_in_queue()

    def func_process_sofast_fixed_data(self):
        """Process Sofast Fixed data"""
        lt.info(f"{timestamp():s} Starting Sofast Fixed data processing")
        # Get Measurement object
        measurement = self.system_fixed.get_measurement(
            self.measure_point_optic, self.dist_optic_screen, self.origin, name=self.name_optic
        )
        self.process_sofast_fixed.load_measurement_data(measurement)

        # Process
        xy_known = (0, 0)
        self.process_sofast_fixed.process_single_facet_optic(
            self.facet_definition, self.surface_fixed, self.origin, xy_known=xy_known
        )

        lt.info(f"{timestamp():s} Completed Sofast Fixed data processing")

        # Plot optic
        mirror = self.process_sofast_fixed.get_optic()

        lt.debug(f"{timestamp():s} Plotting Sofast Fixed data")
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()
        fig_record = fm.setup_figure(figure_control, axis_control_m, title="")
        mirror.plot_orthorectified_slope(self.res_plot, clim=self.colorbar_limit, axis=fig_record.axis)
        fig_record.save(self.dir_save_fixed, f"{self.timestamp_fixed_measurement:s}_slope_magnitude_fixed", "png")

        # Save processed sofast data
        self.process_sofast_fixed.save_to_hdf(
            f"{self.dir_save_fixed:s}/{self.timestamp_fixed_measurement:s}_data_sofast_fixed.h5"
        )
        lt.debug(f"{timestamp():s} Sofast Fixed data saved to HDF5")

        # Continue
        self.system_fixed.run_next_in_queue()

    def func_save_measurement_fringe(self):
        """Saves measurement to HDF file"""
        measurement = self.system_fringe.get_measurements(
            self.measure_point_optic, self.dist_optic_screen, self.name_optic
        )[0]
        file = f"{self.dir_save_fringe:s}/{self.timestamp_fringe_measurement:s}_measurement_fringe.h5"
        measurement.save_to_hdf(file)
        self.system_fringe.calibration.save_to_hdf(file)
        self.system_fringe.run_next_in_queue()

    def func_save_measurement_fixed(self):
        """Save fixed measurement files"""
        measurement = self.system_fixed.get_measurement(
            self.measure_point_optic, self.dist_optic_screen, self.origin, name=self.name_optic
        )
        measurement.save_to_hdf(f"{self.dir_save_fixed:s}/{self.timestamp_fixed_measurement:s}_measurement_fixed.h5")
        self.system_fixed.run_next_in_queue()

    def func_load_last_sofast_fringe_image_cal(self):
        """Loads last ImageCalibration object"""
        # Find file
        files = glob.glob(join(self.dir_save_fringe_calibration, "image_calibration_scaling*.h5"))
        files.sort()

        if len(files) == 0:
            lt.error(f"No previous calibration files found in {self.dir_save_fringe_calibration}")
            return

        # Get latest file and set
        file = files[-1]
        image_calibration = ImageCalibrationScaling.load_from_hdf(file)
        self.system_fringe.set_calibration(image_calibration)
        lt.info(f"{timestamp()} Loaded image calibration file: {file}")

    def func_gray_levels_cal(self):
        """Runs gray level calibration sequence"""
        file = join(self.dir_save_fringe_calibration, f"image_calibration_scaling_{timestamp():s}.h5")
        self.system_fringe.run_gray_levels_cal(
            ImageCalibrationScaling,
            file,
            on_processed=self.func_user_input,
            on_processing=self.func_show_crosshairs_fringe,
        )

    def show_cam_image(self):
        """Shows a camera image"""
        image = self.image_acquisition.get_frame()
        image_proc = highlight_saturation(image, self.image_acquisition.max_value)
        plt.imshow(image_proc)
        plt.show()

    def show_live_view(self):
        """Shows live vieew window"""
        LiveView(self.image_acquisition)

    def func_user_input(self):
        """Waits for user input"""
        retval = input("Input: ")

        lt.debug(f"{timestamp():s} user input: {retval:s}")

        try:
            self._run_given_input(retval)
        except Exception as error:
            lt.error(repr(error))
            self.func_user_input()

    def _check_fringe_system_loaded(self) -> bool:
        if self.system_fringe is None:
            lt.error(f"{timestamp()} No Sofast Fringe system loaded")
            return False
        return True

    def _check_fixed_system_loaded(self) -> bool:
        if self.system_fixed is None:
            lt.error(f"{timestamp()} No Sofast Fixed system loaded")
            return False
        return True

    def _run_given_input(self, retval: str) -> None:
        """Runs the given command"""
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
            self.func_user_input()
        elif retval == "mrp":
            lt.info(f"{timestamp()} Running Sofast Fringe measurement and processing/saving data")
            if self._check_fringe_system_loaded():
                funcs = [
                    self.func_run_fringe_measurement,
                    self.func_show_crosshairs_fringe,
                    self.func_process_sofast_fringe_data,
                    self.func_save_measurement_fringe,
                    self.func_user_input,
                ]
                self.system_fringe.set_queue(funcs)
                self.system_fringe.run()
            else:
                self.func_user_input()
        # Run fringe measurement and save
        elif retval == "mrs":
            lt.info(f"{timestamp()} Running Sofast Fringe measurement and saving data")
            if self._check_fringe_system_loaded():
                funcs = [
                    self.func_run_fringe_measurement,
                    self.func_show_crosshairs_fringe,
                    self.func_save_measurement_fringe,
                    self.func_user_input,
                ]
                self.system_fringe.set_queue(funcs)
                self.system_fringe.run()
            else:
                self.func_user_input()
        # Run fixed measurement and process/save
        elif retval == "mip":
            lt.info(f"{timestamp()} Running Sofast Fixed measurement and processing/saving data")
            if self._check_fixed_system_loaded():
                funcs = [
                    self.func_run_fixed_measurement,
                    self.func_process_sofast_fixed_data,
                    self.func_save_measurement_fixed,
                    self.func_user_input,
                ]
                self.system_fixed.set_queue(funcs)
                self.system_fixed.run()
            else:
                self.func_user_input()
        # Run fixed measurement and save
        elif retval == "mis":
            lt.info(f"{timestamp()} Running Sofast Fixed measurement and saving data")
            if self._check_fixed_system_loaded():
                funcs = [self.func_run_fixed_measurement, self.func_save_measurement_fixed, self.func_user_input]
                self.system_fixed.set_queue(funcs)
                self.system_fixed.run()
            else:
                self.func_user_input()
        # Calibrate exposure time
        elif retval == "ce":
            lt.info(f"{timestamp()} Calibrating camera exposure")
            self.image_acquisition.calibrate_exposure()
            self.func_user_input()
        # Calibrate response
        elif retval == "cr":
            lt.info(f"{timestamp()} Calibrating camera-projector response")
            if self._check_fringe_system_loaded():
                funcs = [self.func_gray_levels_cal]
                self.system_fringe.set_queue(funcs)
                self.system_fringe.run()
            else:
                self.func_user_input()
        # Load last fringe calibration file
        elif retval == "lr":
            lt.info(f"{timestamp()} Loading response calibration")
            if self._check_fringe_system_loaded():
                self.func_load_last_sofast_fringe_image_cal()
            self.func_user_input()
        # Quit
        elif retval == "q":
            lt.info(f"{timestamp():s} quitting")
            if self.system_fringe is not None:
                self.system_fringe.close_all()
            if self.system_fixed is not None:
                self.system_fixed.close_all()
            return
        # Show single camera image
        elif retval == "im":
            self.show_cam_image()
            self.func_user_input()
        # Show camera live view
        elif retval == "lv":
            self.show_live_view()
            self.func_user_input()
        # Project crosshairs
        elif retval == "cross":
            self.image_projection.show_crosshairs()
            self.func_user_input()
        else:
            lt.error(f"{timestamp()} Command, {retval}, not recognized")
            self.func_user_input()


# Start program
if __name__ == "__main__":
    # Define main directory in which to save captured/processed data
    # ==============================================================

    # Define upper level save direcory
    dir_save = abspath(join(dirname(__file__), "../../../../sofast_cli"))

    # Define logger directory and set up logger
    dir_log = join(dir_save, "logs")
    lt.logger(join(dir_log, f"log_{timestamp():s}.txt"), lt.log.INFO)

    # Define the file locations of all SOFAST calibration data
    # ========================================================

    # Define directory containing Sofast calibration files
    dir_cal = abspath(join(dirname(__file__), "../../../../sofast_calibration_files"))

    # Define data files
    file_facet_definition_json = join(dir_cal, "facet_NSTTF.json")
    file_spatial_orientation = join(dir_cal, "spatial_orientation_optics_lab_landscape.h5")
    file_display = join(dir_cal, "display_shape_optics_lab_landscape_square_distorted_3d_100x100.h5")
    file_camera = join(dir_cal, "camera_sofast_optics_lab_landscape.h5")
    file_image_projection = join(dir_cal, "image_projection_optics_lab_landscape_square.h5")
    file_dot_locs = join(dir_cal, "dot_locations_optics_lab_landscape_square_width3_space6.h5")

    # Instantiate Sofast Command Line Interface
    # =========================================
    sofast_cli = SofastCommandLineInterface()

    # Define specific sofast save directories
    # =======================================
    sofast_cli.dir_save_fringe = join(dir_save, "sofast_fringe")
    sofast_cli.dir_save_fringe_calibration = join(sofast_cli.dir_save_fringe, "calibration")
    sofast_cli.dir_save_fixed = join(dir_save, "sofast_fixed")

    # Define camera (ImageAcquisition) parameters
    # ===========================================
    image_acquisition_in = ImageAcquisition(instance=0)  # First camera instance found
    image_acquisition_in.frame_size = (1626, 1236)  # Set frame size
    image_acquisition_in.gain = 230  # Set gain (higher=faster/more noise, lower=slower/less noise)

    # Define projector/display (ImageProjection) parameters
    # =====================================================
    image_projection_in = ImageProjection.load_from_hdf(file_image_projection)
    image_projection_in.display_data.image_delay_ms = 200  # define projector-camera delay

    # Define measurement-specific inputs
    # ==================================

    # NOTE: to update any of these fields, change them here and restart the SOFAST CLI
    measure_point_optic_in = Vxyz((0, 0, 0))  # Measure point on optic, meters
    dist_optic_screen_in = 10.158  # Measured optic-screen distance, meters
    name_optic_in = "Test optic"  # Optic name

    # Load all other calibration files
    # ================================
    camera_in = Camera.load_from_hdf(file_camera)  # SOFAST camera definition
    facet_definition_in = DefinitionFacet.load_from_json(file_facet_definition_json)  # Facet definition
    spatial_orientation_in = SpatialOrientation.load_from_hdf(file_spatial_orientation)  # Spatial orientation

    sofast_cli.set_common_data(
        image_acquisition_in,
        image_projection_in,
        camera_in,
        facet_definition_in,
        spatial_orientation_in,
        measure_point_optic_in,
        dist_optic_screen_in,
        name_optic_in,
    )

    sofast_cli.colorbar_limit = 2  # Update plotting colorbar limit, mrad

    # Set Sofast Fringe parameters
    # ============================
    display_shape_in = DisplayShape.load_from_hdf(file_display)
    fringes_in = Fringes.from_num_periods(4, 4)
    surface_fringe_in = Surface2DParabolic((100.0, 100.0), False, 10)

    sofast_cli.set_sofast_fringe_data(display_shape_in, fringes_in, surface_fringe_in)

    # Set Sofast Fixed parameters
    # ===========================
    # NOTE: to get the value of "origin_in," the user must first start the CLI, bring up the
    # camera live view, then manually find the location of the (0, 0) fixed pattern dot.
    fixed_pattern_dot_locs_in = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
    origin_in = Vxy((1100, 560))  # pixels, location of (0, 0) dot in camera image
    surface_fixed_in = Surface2DParabolic((100.0, 100.0), False, 1)

    sofast_cli.set_sofast_fixed_data(fixed_pattern_dot_locs_in, origin_in, surface_fixed_in)

    # Run
    # ===
    sofast_cli.run()
