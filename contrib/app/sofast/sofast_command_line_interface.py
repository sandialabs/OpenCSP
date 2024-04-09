import datetime as dt
import glob as glob
from os.path import join, dirname

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


def _timestamp() -> str:
    return dt.datetime.now().isoformat().replace(":", "_")


def main():
    """Main system_fringe runner"""
    # Define save direcory and set up logger
    dir_save = join(dirname(__file__), '../../../../sofast_cli')
    lt.logger(join(dir_save, f'log_{_timestamp():s}.txt'), lt.log.INFO)

    # Define directory containing Sofast calibration files
    dir_cal = join(dirname(__file__), '../../../../sofast_calibration_files')

    # Define common values
    file_facet_definition_json = join(dir_cal, 'facet_NSTTF.json')
    file_spatial_orientation = join(dir_cal, 'spatial_orientation_optics_lab_landscape.h5')
    file_display = join(dir_cal, 'display_shape_optics_lab_landscape_square_distorted_3d_100x100.h5')
    file_camera = join(dir_cal, 'camera_sofast_optics_lab_landscape.h5')
    file_image_projection = join(dir_cal, 'image_projection_optics_lab_landscape_square.h5')
    file_dot_locs = join(dir_cal, 'dot_locations_optics_lab_landscape_square_width3_space6.h5')

    facet_definition = DefinitionFacet.load_from_json(file_facet_definition_json)
    spatial_orientation = SpatialOrientation.load_from_hdf(file_spatial_orientation)
    display = DisplayShape.load_from_hdf(file_display)
    camera = Camera.load_from_hdf(file_camera)
    fixed_pattern_dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_locs)

    periods_x = [0.9, 4., 16.]  # , 64.]
    periods_y = [0.9, 4., 16.]  # , 64.]
    fringes = Fringes(periods_x, periods_y)
    surface_fixed = Surface2DParabolic((100.0, 100.0), False, 10)
    surface_fringe = Surface2DParabolic((100.0, 100.0), False, 10)
    measure_point_optic = Vxyz((0, 0, 0))  # meters
    dist_optic_screen = 10.263  # meters
    origin = Vxy((993, 628))  # pixels
    name_optic = 'Test optic'
    res_plot = 0.002  # meters

    # Define setup (NSTTF Optics Lab)
    image_projection = ImageProjection.load_from_hdf_and_display(file_image_projection)

    # Set image delay
    image_projection.display_data['image_delay'] = 200

    image_acquisition = ImageAcquisition(0)
    image_acquisition.frame_size = (1626, 1236)
    image_acquisition.gain = 230

    system_fringe = SystemSofastFringe(image_acquisition)

    system_fixed = SystemSofastFixed(image_acquisition)
    system_fixed.set_pattern_parameters(3, 6)

    sofast_fixed = ProcessSofastFixed(spatial_orientation, camera, fixed_pattern_dot_locs, facet_definition)
    # sofast_fixed.params.geometry_data_debug.debug_active = True

    def func_process_gray_levels_cal():
        """Processes the grey levels calibration data"""
        lt.debug(f'{_timestamp()} Processing gray level data')
        calibration_images = system_fringe.get_calibration_images()[0]
        calibration = ImageCalibrationScaling.from_data(calibration_images, system_fringe._calibration_display_values)
        calibration.save_to_hdf(join(dir_save, f'image_calibration_scaling_{_timestamp():s}.h5'))
        system_fringe.calibration = calibration
        system_fringe.set_fringes(fringes)  # TODO this should be set earlier
        lt.info(f'{_timestamp()} ImageCalibration loaded')
        system_fringe.run_next_in_queue()

    def func_show_crosshairs():
        """Shows crosshairs"""
        im_proj_instance = image_projection.instance()
        im_proj_instance.show_crosshairs()
        system_fringe.run_next_in_queue()

    def func_process_sofast_fringe_data():
        """Processes Sofast Fringe data"""
        lt.debug(f'{_timestamp():s} Processing Sofast Fringe data')

        # Get Measurement object
        measurement = system_fringe.get_measurements(measure_point_optic, dist_optic_screen, name_optic)[0]

        # Calibrate measurement
        measurement.calibrate_fringe_images(system_fringe.calibration)

        # Instantiate ProcessSofastFringe
        sofast = ProcessSofastFringe(measurement, spatial_orientation, camera, display)

        # Process
        sofast.process_optic_singlefacet(facet_definition, surface_fringe)

        # Plot optic
        mirror = sofast.get_optic().mirror

        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()
        fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
        mirror.plot_orthorectified_slope(res_plot, clim=7, axis=fig_record.axis)
        fig_record.save(dir_save, f'{_timestamp():s}_slope_magnitude_fringe', 'png')

        # Save processed sofast data
        sofast.save_to_hdf(f'{dir_save:s}/{_timestamp():s}_data_sofast_fringe.h5')

        # Continue
        system_fringe.run_next_in_queue()

    def func_process_sofast_fixed_data():
        """Process Sofast Fixed data"""
        lt.debug(f'{_timestamp():s} Processing Sofast Fixed data')

        # Get Measurement object
        measurement = system_fixed.get_measurement(measure_point_optic, dist_optic_screen, origin, name=name_optic)
        sofast_fixed.load_measurement_data(measurement)

        # Process
        sofast_fixed.process_single_facet_optic(surface_fixed)

        # Plot optic
        mirror = sofast_fixed.get_mirror()

        figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
        axis_control_m = rca.meters()
        fig_record = fm.setup_figure(figure_control, axis_control_m, title='')
        mirror.plot_orthorectified_slope(res_plot, clim=7, axis=fig_record.axis)
        fig_record.save(dir_save, f'{_timestamp():s}_slope_magnitude_fixed', 'png')

        # Save processed sofast data
        sofast_fixed.save_to_hdf(f'{dir_save:s}/{_timestamp():s}_data_sofast_fixed.h5')

        # Continue
        system_fixed.run_next_in_queue()

    def func_save_measurement_fixed():
        """Save fixed measurement files"""
        measurement = system_fixed.get_measurement(measure_point_optic, dist_optic_screen, origin, name=name_optic)
        measurement.save_to_hdf(f'{dir_save:s}/{_timestamp():s}_measurement_fixed.h5')
        system_fixed.run_next_in_queue()

    def func_save_measurement_fringe():
        """Saves measurement to HDF file"""
        measurement = system_fringe.get_measurements(measure_point_optic, dist_optic_screen, name_optic)[0]
        measurement.save_to_hdf(f'{dir_save:s}/{_timestamp():s}_measurement_fringe.h5')
        system_fringe.run_next_in_queue()

    def func_calibrate_camera_exposure():
        """Calibrates camera exposure"""
        system_fringe.image_acquisitions[0].calibrate_exposure()
        system_fringe.run_next_in_queue()

    def func_pause():
        """Pauses for 200 ms"""
        system_fringe.root.after(200, system_fringe.run_next_in_queue)

    def func_load_last_sofast_fringe_image_cal():
        """Loads last ImageCalibration object"""
        files = glob.glob(join(dir_save, 'image_calibration_scaling*.h5'))
        files.sort()
        file = files[-1]
        image_calibration = ImageCalibrationScaling.load_from_hdf(file)
        system_fringe.calibration = image_calibration
        system_fringe.set_fringes(fringes)  # TODO this should be set earlier
        lt.info(f'{_timestamp()} Loaded image calibration file: {file}')
        system_fringe.run_next_in_queue()

    def func_gray_levels_cal():
        """Runs gray level calibration sequence"""
        system_fringe.run_display_camera_response_calibration(
            res=10,
            run_next=system_fringe.run_next_in_queue
        )

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
        retval = input('Input: ')

        lt.debug(f'{_timestamp():s} user input: {retval:s}')

        if retval == 'mrp':
            lt.info(f'{_timestamp()} Running Sofast Fringe measurement and processing/saving data')
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
            lt.info(f'{_timestamp()} Running Sofast Fringe measurement and saving data')
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
            lt.info(f'{_timestamp()} Running Sofast Fixed measurement and processing/saving data')
            funcs = [
                system_fixed.run_measurement,
                func_process_sofast_fixed_data,
                func_save_measurement_fixed,
                func_user_input,
            ]
            system_fixed.set_queue(funcs)
            system_fixed.run()
        elif retval == 'mis':
            lt.info(f'{_timestamp()} Running Sofast Fixed measurement and saving data')
            funcs = [
                system_fixed.run_measurement,
                func_save_measurement_fixed,
                func_user_input,
            ]
            system_fixed.set_queue(funcs)
            system_fixed.run()
        elif retval == 'ce':
            lt.info(f'{_timestamp()} Calibrating camera exposure')
            funcs = [
                func_calibrate_camera_exposure,
                func_user_input,
            ]
            system_fringe.set_queue(funcs)
            system_fringe.run()
        elif retval == 'cr':
            lt.info(f'{_timestamp()} Calibrating camera-projector response')
            funcs = [
                func_gray_levels_cal,
                func_show_crosshairs,
                func_process_gray_levels_cal,
                func_user_input,
            ]
            system_fringe.set_queue(funcs)
            system_fringe.run()
        elif retval == 'lr':
            lt.info(f'{_timestamp()} Loading response calibration')
            funcs = [
                func_load_last_sofast_fringe_image_cal,
                func_user_input,
            ]
            system_fringe.set_queue(funcs)
            system_fringe.run()
        elif retval == 'q':
            lt.info(f'{_timestamp():s} quitting')
            system_fringe.close_all()
            system_fixed.close_all()
            return
        elif retval == 'im':
            func_show_cam_image()
            system_fringe.set_queue([func_user_input])
            system_fringe.run()
        else:
            lt.error(f'{_timestamp()} Command, {retval}, not recognized')
            funcs = [func_user_input]
            system_fringe.set_queue(funcs)
            system_fringe.run()

    # Run system_fringe
    system_fringe.set_queue([func_user_input])
    system_fringe.run()


# Start program
if __name__ == '__main__':
    main()
