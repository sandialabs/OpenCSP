"""Example script that runs SOFAST on a single facet with one camera

- Runs SOFAST collecion
- Characterizes optic
- Plots slope map

NOTE: must be run with a computer connected to a working SOFAST system. This includes a camera,
mirror, screen, and system layout calibration files.
"""
import os

import numpy as np

from opencsp.common.lib.deflectometry.Display import Display
from opencsp.common.lib.deflectometry.FacetData import FacetData
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.System import System
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition
from opencsp.common.lib.geometry.Pxyz import Pxyz


def main():
    # Define file locations
    data_dir = '../../sofast_2_system_calibration_files/'

    file_image_projection = os.path.join(
        data_dir, 'Image_Projection_optics_lab_landscape_rectangular.h5'
    )
    file_display = os.path.join(data_dir, 'Display_optics_lab_landscape_distorted2D.h5')
    file_camera = os.path.join(data_dir, 'Camera_optics_lab_landscape.h5')
    file_facet = os.path.join(data_dir, 'Facet_NSTTF.json')
    output_file = None

    # Define measurement parameters
    optic_screen_dist = 10.516  # meter
    optic_name = 'NSTTF Facet'
    optic_measure_point = Pxyz((0, 0, 0))  # meter

    # Load data
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    facet_data = FacetData.load_from_json(file_facet)

    # Define surface fitting parameters
    surface_data = dict(
        surface_type='parabolic',
        initial_focal_lengths_xy=(100.0, 100.0),
        robust_least_squares=False,
        downsample=10,
    )

    # Create fringe object
    periods_x = [0.9, 4.0, 16.0, 64.0]
    periods_y = [0.9, 4.0, 16.0, 64.0]
    fringes = Fringes(periods_x, periods_y)

    # Load ImageProjection
    im_proj = ImageProjection.load_from_hdf_and_display(file_image_projection)

    # Initialize variables
    calibrations: list[ImageCalibrationScaling] = []

    # Load ImageAcquisition
    im_acqu = ImageAcquisition(0)
    im_acqu.frame_size = (1626, 1236)
    im_acqu.gain = 230
    im_acqu.exposure_time = 230025
    im_acqu.frame_rate = np.max([0.014902, 1.05 / im_acqu.exposure_time])

    # Create System
    system = System(im_proj, im_acqu)

    # Calibrate camera exposure
    def func_calibrate_exposure():
        print('Calibrating camera exposure')
        system.run_camera_exposure_calibration(run_next=system.run_next_in_queue)

    # Capture calibration data
    def func_capture_calibration_frames():
        print('Capturing display-camera response calibration data')
        system.run_display_camera_response_calibration(
            res=10, run_next=system.run_next_in_queue
        )

    # Process calibration data
    def func_process_calibration_data():
        print('Processing calibration data')
        calibration_images = system.get_calibration_images()
        calibrations.append(
            ImageCalibrationScaling.from_data(
                calibration_images[0], system.calibration_display_values
            )
        )
        system.run_next_in_queue()

    # Load fringe object
    def func_load_fringes():
        print('Loading fringes')
        # Use the largest minimum display values
        min_disp_value = calibrations[0].calculate_min_display_camera_values()[0]

        # Load fringe objects with min display value
        system.load_fringes(fringes, min_disp_value)

        system.run_next_in_queue()

    # Capture mask and fringe images
    def func_capture_fringes():
        print('Capturing mask and fringe images')
        system.capture_mask_and_fringe_images(run_next=system.run_next_in_queue)

    # Close all windows and cameras
    def func_close_all():
        print('Closing all.')
        system.close_all()

    # Set process queue
    funcs = [
        func_calibrate_exposure,
        func_capture_calibration_frames,
        func_process_calibration_data,
        func_load_fringes,
        func_capture_fringes,
        func_close_all,
    ]
    system.set_queue(funcs)

    # Run system
    system.run()

    # Save data
    print('Saving Data')

    # Get Measurement object
    measurement = system.get_measurements(
        optic_measure_point, optic_screen_dist, optic_name
    )[0]
    calibration = calibrations[0]

    # Process data
    print('Processing data')

    # Calibrate measurements
    measurement.calibrate_fringe_images(calibration)

    # Instantiate SOFAST objects
    sofast = Sofast(measurement, camera, display)

    # Process using optic data
    sofast.process_optic_singlefacet(facet_data, surface_data)

    # Save processed data, calibration, and measurement data
    if output_file is not None:
        calibration.save_to_hdf(output_file)
        measurement.save_to_hdf(output_file)
        sofast.save_to_hdf(output_file)


# Start program
if __name__ == '__main__':
    main()
