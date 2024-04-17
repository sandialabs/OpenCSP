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


def main(
    image_acquisition: ImageAcquisition,  # Common imputs
    image_projection: ImageProjection,
    spatial_orientation: SpatialOrientation,
    camera: Camera,
    facet_definition: DefinitionFacet,
    measure_point_optic: Vxyz,
    dist_optic_screen: float,
    name_optic: str,
    display_shape: DisplayShape,  # Sofast Fringe inputs
    fringes: Fringes,
    surface_fringe: Surface2DParabolic,
    fixed_pattern_dot_locs: DotLocationsFixedPattern,  # Sofast fixed inputs
    origin: Vxy,
    surface_fixed: Surface2DParabolic,
    dir_save_fringe_calibration: str,  # Saving inputs
    dir_save_fixed: str,
    dir_save_fringe: str,
    res_plot: float,
) -> None:
    """Main system_fringe runner

    Parameters
    ----------
    image_acquisition : ImageAcquisition
        Image acquisition object
    spatial_orientation : SpatialOrientation
        Spatial orientation object
    camera : Camera
        Camera object
    facet_definition : DefinitionFacet
        Facet definition
    measure_point_optic : Vxyz
        Measure point on optic
    dist_optic_screen : float
        Distance from measure point to screen
    name_optic : str
        Name of optic
    display_shape : DisplayShape
        Display shape object
    surface_fringe : Surface2DParabolic
        Surface definition for Sofast fringe
    fixed_pattern_dot_locs : DotLocationsFixedPattern
        Dot location definition object
    surface_fixed : Surface2DParabolic
        Surface definition of Sofast Fixed
    dir_save_fringe_calibration : str
        Save path for Sofast Fringe calibration files
    dir_save_fringe : str
        Save path for Sofast Fringe measurement files
    res_plot : float
        Save path for Sofast Fixed measurement files
    """

    system_fringe = SystemSofastFringe(image_acquisition)

    system_fixed = SystemSofastFixed(image_acquisition)
    system_fixed.set_pattern_parameters(3, 6)

    sofast_fixed = ProcessSofastFixed(spatial_orientation, camera, fixed_pattern_dot_locs, facet_definition)

    def func_process_gray_levels_cal():
        """Processes the grey levels calibration data"""
        calibration_images = system_fringe.get_calibration_images()[0]
        calibration = ImageCalibrationScaling.from_data(calibration_images, system_fringe._calibration_display_values)
        calibration.save_to_hdf(join(dir_save_fringe_calibration, f'image_calibration_scaling_{timestamp():s}.h5'))
        system_fringe.calibration = calibration
        system_fringe.set_fringes(fringes)
        lt.info(f'{timestamp()} ImageCalibration data loaded into SystemSofastFringe')
        system_fringe.run_next_in_queue()

    def func_show_crosshairs():
        """Shows crosshairs"""
        im_proj_instance = image_projection.instance()
        im_proj_instance.show_crosshairs()
        system_fringe.run_next_in_queue()

    def func_process_sofast_fringe_data():
        """Processes Sofast Fringe data"""
        lt.debug(f'{timestamp():s} Processing Sofast Fringe data')

        # Get Measurement object
        measurement = system_fringe.get_measurements(measure_point_optic, dist_optic_screen, name_optic)[0]

        # Calibrate fringe images
        measurement.calibrate_fringe_images(system_fringe.calibration)

        # Instantiate ProcessSofastFringe
        sofast = ProcessSofastFringe(measurement, spatial_orientation, camera, display_shape)

        # Process
        sofast.process_optic_singlefacet(facet_definition, surface_fringe)

        # Plot optic
        mirror = sofast.get_optic().mirror

        lt.debug(f'{timestamp():s} Plotting Sofast Fringe data')
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()
        fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
        mirror.plot_orthorectified_slope(res_plot, clim=7, axis=fig_record.axis)
        fig_record.save(dir_save_fringe, f'{timestamp():s}_slope_magnitude_fringe', 'png')
        fig_record.close()

        # Save processed sofast data
        sofast.save_to_hdf(f'{dir_save_fringe:s}/{timestamp():s}_data_sofast_fringe.h5')
        lt.debug(f'{timestamp():s} Sofast Fringe data saved to HDF5')

        # Continue
        system_fringe.run_next_in_queue()

    def func_process_sofast_fixed_data():
        """Process Sofast Fixed data"""
        # Get Measurement object
        measurement = system_fixed.get_measurement(measure_point_optic, dist_optic_screen, origin, name=name_optic)
        sofast_fixed.load_measurement_data(measurement)

        # Process
        sofast_fixed.process_single_facet_optic(surface_fixed)

        # Plot optic
        mirror = sofast_fixed.get_mirror()

        lt.debug(f'{timestamp():s} Plotting Sofast Fixed data')
        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()
        fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
        mirror.plot_orthorectified_slope(res_plot, clim=7, axis=fig_record.axis)
        fig_record.save(dir_save_fixed, f'{timestamp():s}_slope_magnitude_fixed', 'png')

        # Save processed sofast data
        sofast_fixed.save_to_hdf(f'{dir_save_fixed:s}/{timestamp():s}_data_sofast_fixed.h5')
        lt.debug(f'{timestamp():s} Sofast Fixed data saved to HDF5')

        # Continue
        system_fixed.run_next_in_queue()

    def func_save_measurement_fixed():
        """Save fixed measurement files"""
        measurement = system_fixed.get_measurement(measure_point_optic, dist_optic_screen, origin, name=name_optic)
        measurement.save_to_hdf(f'{dir_save_fixed:s}/{timestamp():s}_measurement_fixed.h5')
        system_fixed.run_next_in_queue()

    def func_save_measurement_fringe():
        """Saves measurement to HDF file"""
        measurement = system_fringe.get_measurements(measure_point_optic, dist_optic_screen, name_optic)[0]
        measurement.save_to_hdf(f'{dir_save_fringe:s}/{timestamp():s}_measurement_fringe.h5')
        system_fringe.run_next_in_queue()

    def func_pause():
        """Pauses for 200 ms"""
        system_fringe.root.after(200, system_fringe.run_next_in_queue)

    def func_load_last_sofast_fringe_image_cal():
        """Loads last ImageCalibration object"""
        files = glob.glob(join(dir_save_fringe_calibration, 'image_calibration_scaling*.h5'))
        files.sort()
        file = files[-1]
        image_calibration = ImageCalibrationScaling.load_from_hdf(file)
        system_fringe.calibration = image_calibration
        system_fringe.set_fringes(fringes)
        lt.info(f'{timestamp()} Loaded image calibration file: {file}')
        system_fringe.run_next_in_queue()

    def func_gray_levels_cal():
        """Runs gray level calibration sequence"""
        system_fringe.run_display_camera_response_calibration(res=10, run_next=system_fringe.run_next_in_queue)

    def func_show_cam_image():
        """Shows a camera image"""
        image = image_acquisition.get_frame()
        plt.imshow(image)
        plt.show()

    def func_user_input():
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
                lambda: system_fringe.run_measurement(system_fringe.run_next_in_queue),
                func_show_crosshairs,
                func_pause,
                func_process_sofast_fringe_data,
                func_save_measurement_fringe,
                func_user_input,
            ]
            system_fringe.set_queue(funcs)
            system_fringe.run()
        elif retval == 'mrs':
            lt.info(f'{timestamp()} Running Sofast Fringe measurement and saving data')
            funcs = [
                lambda: system_fringe.run_measurement(system_fringe.run_next_in_queue),
                func_show_crosshairs,
                func_pause,
                func_save_measurement_fringe,
                func_user_input,
            ]
            system_fringe.set_queue(funcs)
            system_fringe.run()
        elif retval == 'mip':
            lt.info(f'{timestamp()} Running Sofast Fixed measurement and processing/saving data')
            funcs = [
                system_fixed.run_measurement,
                func_process_sofast_fixed_data,
                func_save_measurement_fixed,
                func_user_input,
            ]
            system_fixed.set_queue(funcs)
            system_fixed.run()
        elif retval == 'mis':
            lt.info(f'{timestamp()} Running Sofast Fixed measurement and saving data')
            funcs = [system_fixed.run_measurement, func_save_measurement_fixed, func_user_input]
            system_fixed.set_queue(funcs)
            system_fixed.run()
        elif retval == 'ce':
            lt.info(f'{timestamp()} Calibrating camera exposure')
            image_acquisition.calibrate_exposure()
            func_user_input()
        elif retval == 'cr':
            lt.info(f'{timestamp()} Calibrating camera-projector response')
            funcs = [func_gray_levels_cal, func_show_crosshairs, func_process_gray_levels_cal, func_user_input]
            system_fringe.set_queue(funcs)
            system_fringe.run()
        elif retval == 'lr':
            lt.info(f'{timestamp()} Loading response calibration')
            funcs = [func_load_last_sofast_fringe_image_cal, func_user_input]
            system_fringe.set_queue(funcs)
            system_fringe.run()
        elif retval == 'q':
            lt.info(f'{timestamp():s} quitting')
            system_fringe.close_all()
            system_fixed.close_all()
            return
        elif retval == 'im':
            func_show_cam_image()
            system_fringe.set_queue([func_user_input])
            system_fringe.run()
        elif retval == 'cross':
            image_projection.show_crosshairs()
            func_user_input()
        else:
            lt.error(f'{timestamp()} Command, {retval}, not recognized')
            funcs = [func_user_input]
            system_fringe.set_queue(funcs)
            system_fringe.run()

    # Run system_fringe
    system_fringe.set_queue([func_user_input])
    system_fringe.run()


# Start program
if __name__ == '__main__':
    # Define upper level save direcory
    dir_save = abspath(join(dirname(__file__), '../../../../sofast_cli'))

    # Define logger directory and set up logger
    dir_log = join(dir_save, 'logs')
    lt.logger(join(dir_log, f'log_{timestamp():s}.txt'), lt.log.INFO)

    # Define sofast and calibration file save directories
    dir_save_fringe_in = join(dir_save, 'sofast_fringe')
    dir_save_fringe_calibration_in = join(dir_save_fringe_in, 'calibration')
    dir_save_fixed_in = join(dir_save, 'sofast_fixed')

    # Define directory containing Sofast calibration files
    dir_cal = abspath(join(dirname(__file__), '../../../../sofast_calibration_files'))

    # Define common values
    file_facet_definition_json = join(dir_cal, 'facet_NSTTF.json')
    file_spatial_orientation = join(dir_cal, 'spatial_orientation_optics_lab_landscape.h5')
    file_display = join(dir_cal, 'display_shape_optics_lab_landscape_square_distorted_3d_100x100.h5')
    file_camera = join(dir_cal, 'camera_sofast_optics_lab_landscape.h5')
    file_image_projection = join(dir_cal, 'image_projection_optics_lab_landscape_square.h5')
    file_dot_locs = join(dir_cal, 'dot_locations_optics_lab_landscape_square_width3_space6.h5')

    # Load common data
    facet_definition_in = DefinitionFacet.load_from_json(file_facet_definition_json)
    spatial_orientation_in = SpatialOrientation.load_from_hdf(file_spatial_orientation)
    display_shape_in = DisplayShape.load_from_hdf(file_display)
    camera_in = Camera.load_from_hdf(file_camera)
    fixed_pattern_dot_locs_in = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)

    # Define common parameters
    name_optic_in = 'Test optic'
    res_plot_in = 0.002  # meters
    measure_point_optic_in = Vxyz((0, 0, 0))  # meters
    dist_optic_screen_in = 10.263  # meters

    # Define Sofast Fringe parameters
    fringes_in = Fringes.from_num_periods(4, 4)
    surface_fringe_in = Surface2DParabolic((100.0, 100.0), False, 10)

    # Define Sofast Fixed parameters
    surface_fixed_in = Surface2DParabolic((100.0, 100.0), False, 1)
    origin_in = Vxy((993, 644))  # pixels

    # Define image projection
    image_projection_in = ImageProjection.load_from_hdf_and_display(file_image_projection)
    image_projection_in.display_data['image_delay'] = 200

    # Setup image acquisition
    image_acquisition_in = ImageAcquisition(0)
    image_acquisition_in.frame_size = (1626, 1236)
    image_acquisition_in.gain = 230

    kwargs = {
        'image_acquisition': image_acquisition_in,
        'image_projection': image_projection_in,
        'spatial_orientation': spatial_orientation_in,
        'camera': camera_in,
        'facet_definition': facet_definition_in,
        'measure_point_optic': measure_point_optic_in,
        'dist_optic_screen': dist_optic_screen_in,
        'name_optic': name_optic_in,
        'display_shape': display_shape_in,
        'fringes': fringes_in,
        'surface_fringe': surface_fringe_in,
        'fixed_pattern_dot_locs': fixed_pattern_dot_locs_in,
        'origin': origin_in,
        'surface_fixed': surface_fixed_in,
        'dir_save_fringe_calibration': dir_save_fringe_calibration_in,
        'dir_save_fixed': dir_save_fixed_in,
        'dir_save_fringe': dir_save_fringe_in,
        'res_plot': res_plot_in,
    }

    main(**kwargs)
