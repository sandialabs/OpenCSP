"""Example script that performs automatic fixed pattern deflectometry data capture and analysis
using displayed dots on a screen system. The user is asked to perform a measurement, or perform
a camera exposure calibration via command line. The resulting slope magnitude map is
displayed.

NOTE: The user must have a complete deflectometry setup in place. This includes a camera,
dot grid, and system calibration files.
"""
import os

import matplotlib.pyplot as plt

from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition
from opencsp.common.lib.deflectometry.FacetData import FacetData
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.deflectometry.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


def calibrate_camera_exposure(image_acquisition: ImageAcquisition) -> None:
    """Automatically calibrates the camera exposure"""
    image_acquisition.calibrate_exposure()

    print('Gain:', image_acquisition.gain, '\n')


def process(
    fixed_pattern_dot_locs: DotLocationsFixedPattern,
    orientation: SpatialOrientation,
    camera: Camera,
    facet_data: FacetData,
    measurement: MeasurementSofastFixed,
    surface_data: dict,
) -> None:
    """Performs processing of measured dot data"""
    # Process fixed pattern
    fixed_pattern = ProcessSofastFixed(
        orientation, camera, fixed_pattern_dot_locs, facet_data
    )
    fixed_pattern.load_measurement_data(measurement)
    fixed_pattern.process_single_facet_optic(surface_data)

    # Print focal lengths from best fit paraboloid
    surf_coefs = fixed_pattern.data_slope_solver.surf_coefs_facet
    focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
    print('Parabolic fit focal lengths:')
    print(f'  X {focal_lengths_xy[0]:.3f} m')
    print(f'  Y {focal_lengths_xy[1]:.3f} m')

    # Plot slope image
    mirror = fixed_pattern.get_mirror()
    plt.figure()
    mirror.plot_orthorectified_slope(res=0.002, clim=7)
    plt.show()


def run() -> None:
    """Runs measurement/characterization process"""
    # connect camera
    image_acquisition = ImageAcquisition()

    # Define inputs
    dir_calibration = 'path/to/calibration/files'
    v_measure_point_facet = Vxyz((0, 0, 0))
    dist_optic_screen = 10.008  # m
    pt_origin = Vxy((947, 649))

    # Load data
    file_camera = os.path.join(dir_calibration, "Camera_optics_lab_landscape.h5")
    file_fixed_pattern_dot = os.path.join(
        dir_calibration, "fixed_pattern_display_w3_s6.h5"
    )
    file_spatial_orientation = os.path.join(
        dir_calibration, "fixed_pattern_display_w3_s6.h5"
    )
    file_facet_data = os.path.join(dir_calibration, "Facet_NSTTF.json")
    file_image_projection = os.path.join(
        dir_calibration, 'Image_Projection_optics_lab_landscape_square.h5'
    )

    # Load ImageProjection and fixed pattern parameters
    image_projection = ImageProjection.load_from_hdf_and_display(file_image_projection)
    fixed_pattern = SystemSofastFixed(
        image_projection.size_x,
        image_projection.size_y,
        width_pattern=3,
        spacing_pattern=6,
    )
    image = fixed_pattern.get_image('uint8', 255)
    image_projection.display_image_in_active_area(image)

    # Load other components
    camera = Camera.load_from_hdf(file_camera)
    spatial_orientation = SpatialOrientation.load_from_hdf(file_spatial_orientation)
    fixed_pattern_dot_locs = DotLocationsFixedPattern.load_from_hdf(
        file_fixed_pattern_dot
    )

    # Define facet measurement setup
    facet_data = FacetData.load_from_json(file_facet_data)
    surface_data = dict(
        surface_type='parabolic',
        initial_focal_lengths_xy=(150.0, 150),
        robust_least_squares=False,
        downsample=1,
    )

    def run_next():
        resp = input(
            'Measure (m), calibrate camera exposure (c), or stop (any other key): '
        )
        if resp == 'm':
            # Capture image
            frame = image_acquisition.get_frame()
            # Process
            measurement = MeasurementSofastFixed(
                frame,
                v_measure_point_facet,
                dist_optic_screen,
                pt_origin,
                name='NSTTF Facet',
            )
            process(
                fixed_pattern_dot_locs,
                spatial_orientation,
                camera,
                facet_data,
                measurement,
                surface_data,
            )
            # Continue or exit
            image_projection.root.after(200, run_next)
        elif resp == 'c':
            calibrate_camera_exposure(image_acquisition)
            image_projection.root.after(200, run_next)
        else:
            image_acquisition.close()
            image_projection.close()

    # Run image projection sequence
    image_projection.root.after(200, run_next)
    image_projection.run()


if __name__ == '__main__':
    run()
