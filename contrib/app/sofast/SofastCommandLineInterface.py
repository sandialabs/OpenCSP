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
    """
    Sofast Command Line Interface class.
    """

    def __init__(self) -> 'SofastCommandLineInterface':
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

        # Sofast fringe specific
        self.system_fringe: SystemSofastFringe = None
        self.display_shape: DisplayShape = None
        self.fringes: Fringes = None
        self.surface_fringe: Surface2DParabolic = None

        # Sofast fixed specific
        self.system_fixed: SystemSofastFixed = None
        self.process_sofast_fixed: ProcessSofastFixed = None
        self.fixed_pattern_dot_locs: DotLocationsFixedPattern = None
        self.surface_fixed: Surface2DParabolic = None
        self.origin: Vxy = None
        self.pattern_width = 3
        """Fixed pattern dot width, pixels"""
        self.pattern_spacing = 6
        """Fixed pattern dot spacing, pixels"""

        # Save directories
        self.dir_save_fringe: str = ''
        """Location to save Sofast Fringe measurement data"""
        self.dir_save_fixed: str = ''
        """Location to save Sofast Fixed measurement data"""
        self.dir_save_fringe_calibration: str = ''
        """Location to save Sofast Fringe calibration data"""

    def run(self) -> None:
        """Runs command line Sofast"""
        self.func_user_input()

    def set_common_data(
        self,
        image_acquisition: ImageAcquisition,
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
        self.display_shape = display_shape
        self.fringes = fringes
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

        self.process_sofast_fixed = ProcessSofastFixed(
            self.spatial_orientation, self.camera, fixed_pattern_dot_locs, self.facet_definition
        )

    def func_show_crosshairs(self):
        """Shows crosshairs"""
        self.image_projection.show_crosshairs()
        self.func_user_input()

    def func_process_sofast_fringe_data(self):
        """Processes Sofast Fringe data"""
        lt.debug(f'{timestamp():s} Processing Sofast Fringe data')

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

        # Plot optic
        mirror = sofast.get_optic().mirror

        lt.debug(f'{timestamp():s} Plotting Sofast Fringe data')
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()
        fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
        mirror.plot_orthorectified_slope(self.res_plot, clim=7, axis=fig_record.axis)
        fig_record.save(self.dir_save_fringe, f'{timestamp():s}_slope_magnitude_fringe', 'png')
        fig_record.close()

        # Save processed sofast data
        sofast.save_to_hdf(f'{self.dir_save_fringe:s}/{timestamp():s}_data_sofast_fringe.h5')
        lt.debug(f'{timestamp():s} Sofast Fringe data saved to HDF5')

        # Continue
        self.system_fringe.run_next_in_queue()

    def func_process_sofast_fixed_data(self):
        """Process Sofast Fixed data"""
        # Get Measurement object
        measurement = self.system_fixed.get_measurement(
            self.measure_point_optic, self.dist_optic_screen, self.origin, name=self.name_optic
        )
        self.process_sofast_fixed.load_measurement_data(measurement)

        # Process
        self.process_sofast_fixed.process_single_facet_optic(self.surface_fixed)

        # Plot optic
        mirror = self.process_sofast_fixed.get_mirror()

        lt.debug(f'{timestamp():s} Plotting Sofast Fixed data')
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()
        fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
        mirror.plot_orthorectified_slope(self.res_plot, clim=7, axis=fig_record.axis)
        fig_record.save(self.dir_save_fixed, f'{timestamp():s}_slope_magnitude_fixed', 'png')

        # Save processed sofast data
        self.process_sofast_fixed.save_to_hdf(f'{self.dir_save_fixed:s}/{timestamp():s}_data_sofast_fixed.h5')
        lt.debug(f'{timestamp():s} Sofast Fixed data saved to HDF5')

        # Continue
        self.system_fixed.run_next_in_queue()

    def func_save_measurement_fixed(self):
        """Save fixed measurement files"""
        measurement = self.system_fixed.get_measurement(
            self.measure_point_optic, self.dist_optic_screen, self.origin, name=self.name_optic
        )
        measurement.save_to_hdf(f'{self.dir_save_fixed:s}/{timestamp():s}_measurement_fixed.h5')
        self.system_fixed.run_next_in_queue()

    def func_save_measurement_fringe(self):
        """Saves measurement to HDF file"""
        measurement = self.system_fringe.get_measurements(
            self.measure_point_optic, self.dist_optic_screen, self.name_optic
        )[0]
        measurement.save_to_hdf(f'{self.dir_save_fringe:s}/{timestamp():s}_measurement_fringe.h5')
        self.system_fringe.run_next_in_queue()

    def func_pause(self):
        """Pauses for 200 ms"""
        self.system_fringe.root.after(200, self.system_fringe.run_next_in_queue)

    def func_load_last_sofast_fringe_image_cal(self):
        """Loads last ImageCalibration object"""
        files = glob.glob(join(self.dir_save_fringe_calibration, 'image_calibration_scaling*.h5'))
        files.sort()
        file = files[-1]
        image_calibration = ImageCalibrationScaling.load_from_hdf(file)
        self.system_fringe.calibration = image_calibration
        self.system_fringe.set_fringes(self.fringes)
        lt.info(f'{timestamp()} Loaded image calibration file: {file}')

    def func_gray_levels_cal(self):
        """Runs gray level calibration sequence"""
        self.system_fringe.run_display_camera_response_calibration(
            res=10, run_next=self.system_fringe.run_next_in_queue
        )

    def func_show_cam_image(self):
        """Shows a camera image"""
        image = self.image_acquisition.get_frame()
        plt.imshow(image)
        plt.show()

    def func_user_input(self):
        """Waits for user input"""
        print('\n')
        print('Value      Command')
        print('------------------')
        print('mrp        run Sofast Fringe measurement and process/save')
        print('mrs        run Sofast Fringe measurement and save only')
        print('mip        run Sofast Fixed measurement and process/save')
        print('mis        run Sofast Fixed measurement and save only')
        print('ce         calibrate camera exposure')
        print('cr         calibrate camera-projector response')
        print('lr         load most recent camera-projector response calibration file')
        print('q          quit and close all')
        print('im         show image from camera.')
        print('cross      show crosshairs')
        retval = input('Input: ')

        lt.debug(f'{timestamp():s} user input: {retval:s}')

        if retval == 'mrp':
            lt.info(f'{timestamp()} Running Sofast Fringe measurement and processing/saving data')
            funcs = [
                lambda: self.system_fringe.run_measurement(self.system_fringe.run_next_in_queue),
                self.func_show_crosshairs,
                self.func_pause,
                self.func_process_sofast_fringe_data,
                self.func_save_measurement_fringe,
                self.func_user_input,
            ]
            self.system_fringe.set_queue(funcs)
            self.system_fringe.run()
        elif retval == 'mrs':
            lt.info(f'{timestamp()} Running Sofast Fringe measurement and saving data')
            funcs = [
                lambda: self.system_fringe.run_measurement(self.system_fringe.run_next_in_queue),
                self.func_show_crosshairs,
                self.func_pause,
                self.func_save_measurement_fringe,
                self.func_user_input,
            ]
            self.system_fringe.set_queue(funcs)
            self.system_fringe.run()
        elif retval == 'mip':
            lt.info(f'{timestamp()} Running Sofast Fixed measurement and processing/saving data')
            funcs = [
                self.system_fixed.run_measurement,
                self.func_process_sofast_fixed_data,
                self.func_save_measurement_fixed,
                self.func_user_input,
            ]
            self.system_fixed.set_queue(funcs)
            self.system_fixed.run()
        elif retval == 'mis':
            lt.info(f'{timestamp()} Running Sofast Fixed measurement and saving data')
            funcs = [self.system_fixed.run_measurement, self.func_save_measurement_fixed, self.func_user_input]
            self.system_fixed.set_queue(funcs)
            self.system_fixed.run()
        elif retval == 'ce':
            lt.info(f'{timestamp()} Calibrating camera exposure')
            self.image_acquisition.calibrate_exposure()
            self.func_user_input()
        elif retval == 'cr':
            lt.info(f'{timestamp()} Calibrating camera-projector response')
            funcs = [self.func_gray_levels_cal, self.func_show_crosshairs, self.func_user_input]
            self.system_fringe.set_queue(funcs)
            self.system_fringe.run()
        elif retval == 'lr':
            lt.info(f'{timestamp()} Loading response calibration')
            self.func_load_last_sofast_fringe_image_cal()
            self.func_user_input()
        elif retval == 'q':
            lt.info(f'{timestamp():s} quitting')
            self.system_fringe.close_all()
            self.system_fixed.close_all()
            return
        elif retval == 'im':
            self.func_show_cam_image()
            self.func_user_input()
        elif retval == 'cross':
            self.image_projection.show_crosshairs()
            self.func_user_input()
        else:
            lt.error(f'{timestamp()} Command, {retval}, not recognized')
            self.func_user_input()


# Start program
if __name__ == '__main__':
    # Define upper level save direcory
    dir_save = abspath(join(dirname(__file__), '../../../../sofast_cli'))

    # Define logger directory and set up logger
    dir_log = join(dir_save, 'logs')
    lt.logger(join(dir_log, f'log_{timestamp():s}.txt'), lt.log.INFO)

    # Define directory containing Sofast calibration files
    dir_cal = abspath(join(dirname(__file__), '../../../../sofast_calibration_files'))

    # Define data files
    file_facet_definition_json = join(dir_cal, 'facet_NSTTF.json')
    file_spatial_orientation = join(dir_cal, 'spatial_orientation_optics_lab_landscape.h5')
    file_display = join(dir_cal, 'display_shape_optics_lab_landscape_square_distorted_3d_100x100.h5')
    file_camera = join(dir_cal, 'camera_sofast_optics_lab_landscape.h5')
    file_image_projection = join(dir_cal, 'image_projection_optics_lab_landscape_square.h5')
    file_dot_locs = join(dir_cal, 'dot_locations_optics_lab_landscape_square_width3_space6.h5')

    # Define image projection
    image_projection = ImageProjection.load_from_hdf_and_display(file_image_projection)
    image_projection.display_data['image_delay'] = 200

    # Instantiate Sofast Command Line Interface
    sofast_cli = SofastCommandLineInterface()

    # Define sofast and calibration file save directories
    sofast_cli.dir_save_fringe = join(dir_save, 'sofast_fringe')
    sofast_cli.dir_save_fringe_calibration = join(sofast_cli.dir_save_fringe, 'calibration')
    sofast_cli.dir_save_fixed = join(dir_save, 'sofast_fixed')

    # Load common data
    image_acquisition = ImageAcquisition(0)
    image_acquisition.frame_size = (1626, 1236)
    image_acquisition.gain = 230

    facet_definition = DefinitionFacet.load_from_json(file_facet_definition_json)
    spatial_orientation = SpatialOrientation.load_from_hdf(file_spatial_orientation)
    camera = Camera.load_from_hdf(file_camera)
    name_optic = 'Test optic'
    measure_point_optic = Vxyz((0, 0, 0))  # meters
    dist_optic_screen = 10.263  # meters

    sofast_cli.set_common_data(
        image_acquisition,
        camera,
        facet_definition,
        spatial_orientation,
        measure_point_optic,
        dist_optic_screen,
        name_optic,
    )

    # Load Sofast Fringe data
    display_shape = DisplayShape.load_from_hdf(file_display)
    fringes = Fringes.from_num_periods(4, 4)
    surface_fringe = Surface2DParabolic((100.0, 100.0), False, 10)

    sofast_cli.set_sofast_fringe_data(display_shape, fringes, surface_fringe)

    # Load Sofast Fixed data
    fixed_pattern_dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)
    origin = Vxy((993, 644))  # pixels
    surface_fixed = Surface2DParabolic((100.0, 100.0), False, 1)

    sofast_cli.set_sofast_fixed_data(fixed_pattern_dot_locs, origin, surface_fixed)

    # Run
    sofast_cli.run()
