"""Script that performs a SofastFixed data capture and analysis using displayed dots
on a screen system. The user is asked to perform a measurement, or perform
a camera exposure calibration via command line. The resulting slope magnitude map is
displayed.

NOTE: The user must have a complete deflectometry setup in place. This includes a camera, mirror to
measure, and system calibration files.
"""

from os.path import join, abspath

import matplotlib.pyplot as plt

from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.DotLocationsFixedPattern import DotLocationsFixedPattern
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
import opencsp.app.sofast.lib.DistanceOpticScreen as osd
from opencsp.app.sofast.lib.ProcessSofastFixed import ProcessSofastFixed
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.app.sofast.lib.SystemSofastFixed import SystemSofastFixed
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.camera.ImageAcquisition_DCAM_mono import ImageAcquisition
from opencsp.common.lib.deflectometry.ImageProjection import ImageProjection
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.log_tools as lt
import opencsp.common.lib.tool.time_date_tools as tdt


def calibrate_camera_exposure(image_acquisition: ImageAcquisition) -> None:
    """Automatically calibrates the camera exposure"""
    image_acquisition.calibrate_exposure()
    lt.info(f"Camera gain: {image_acquisition.gain:.2f}")


def process(
    fixed_pattern_dot_locs: DotLocationsFixedPattern,
    orientation: SpatialOrientation,
    camera: Camera,
    facet: DefinitionFacet,
    measurement: MeasurementSofastFixed,
    surface: Surface2DParabolic,
    dir_save: str,
) -> None:
    """Performs processing of measured dot data"""
    # Process fixed pattern
    system_sofast_fixed = ProcessSofastFixed(orientation, camera, fixed_pattern_dot_locs, facet)
    system_sofast_fixed.load_measurement_data(measurement)
    system_sofast_fixed.process_single_facet_optic(surface)

    # Print focal lengths from best fit paraboloid
    surf_coefs = system_sofast_fixed.data_slope_solver.surf_coefs_facet
    focal_lengths_xy = [1 / 4 / surf_coefs[2], 1 / 4 / surf_coefs[5]]
    lt.info(f"Parabolic fit focal xy lengths: ({focal_lengths_xy[0]:.3f}, {focal_lengths_xy[1]:.3f}) m")

    time_str = tdt.current_date_time_string_forfile()

    # Save data as HDF5 file
    file_save = join(dir_save, "sofast_fixed_data_" + time_str + ".h5")
    system_sofast_fixed.save_to_hdf(file_save)
    fixed_pattern_dot_locs.save_to_hdf(file_save)
    orientation.save_to_hdf(file_save)
    camera.save_to_hdf(file_save)
    facet.save_to_hdf(file_save)
    measurement.save_to_hdf(file_save)
    surface.save_to_hdf(file_save)

    # Plot slope image
    mirror = system_sofast_fixed.get_mirror(interpolation_type="bilinear")

    figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
    axis_control_m = rca.meters()
    fig_record = fm.setup_figure(figure_control, axis_control_m, title="")
    mirror.plot_orthorectified_slope(res=0.002, clim=7, axis=fig_record.axis)
    fig_record.save(dir_save, "slope_magnitude_" + time_str, "png", close_after_save=False)
    plt.show()


def run() -> None:
    """Runs measurement/characterization process"""
    # Define user inputs
    v_measure_point_facet = Vxyz((0, 0, 0))
    dist_optic_screen = 10.263  # m
    pt_origin = Vxy((993, 644))  # pixels
    name = "NSTTF Facet"
    width_pattern = 3  # pixels
    spacing_pattern = 6  # pixels
    dir_cal_files = join(opencsp_code_dir(), "../../sofast_calibration_files")
    dir_save = abspath("../../")

    # Start logging
    lt.logger(join(dir_save, f"log_{tdt.current_date_time_string_forfile():s}.txt"))

    # Connect camera
    image_acquisition = ImageAcquisition()

    # Load data
    file_camera = join(dir_cal_files, "Camera_optics_lab_landscape.h5")
    file_dot_location = join(dir_cal_files, f"fixed_pattern_dot_locations_w{width_pattern:d}_s{spacing_pattern:d}.h5")
    file_spatial_orientation = join(dir_cal_files, "spatial_orientation_optics_lab.h5")
    file_facet_data = join(dir_cal_files, "Facet_NSTTF.json")
    file_image_projection = join(dir_cal_files, "Image_Projection_optics_lab_landscape_square.h5")

    # Load ImageProjection and SystemSofastFixed objects
    image_projection = ImageProjection.load_from_hdf_and_display(file_image_projection)
    system_sofast_fixed = SystemSofastFixed(
        image_projection.size_x, image_projection.size_y, width_pattern=width_pattern, spacing_pattern=spacing_pattern
    )

    # Display fixed pattern image
    image = system_sofast_fixed.get_image("uint8", 255)
    image_projection.display_image_in_active_area(image)

    # Load other components
    camera = Camera.load_from_hdf(file_camera)
    spatial_orientation = SpatialOrientation.load_from_hdf(file_spatial_orientation)
    fixed_pattern_dot_locs = DotLocationsFixedPattern.load_from_hdf(file_dot_location)

    # Define facet measurement setup
    facet = DefinitionFacet.load_from_json(file_facet_data)
    surface = Surface2DParabolic(initial_focal_lengths_xy=(150.0, 150.0), robust_least_squares=False, downsample=1)

    def run_next():
        resp = input("Measure (m), calibrate camera exposure (c), or stop (any other key): ")
        if resp == "m":
            # Capture image
            frame = image_acquisition.get_frame()
            # Process
            dist_optic_screen_measure = osd.DistanceOpticScreen(v_measure_point_facet, dist_optic_screen)
            measurement = MeasurementSofastFixed(frame, dist_optic_screen_measure, pt_origin, name)
            process(fixed_pattern_dot_locs, spatial_orientation, camera, facet, measurement, surface, dir_save)
            # Continue or exit
            image_projection.root.after(200, run_next)
        elif resp == "c":
            calibrate_camera_exposure(image_acquisition)
            image_projection.root.after(200, run_next)
        else:
            image_acquisition.close()
            image_projection.close()

    # Run image projection sequence
    image_projection.root.after(200, run_next)
    image_projection.run()


if __name__ == "__main__":
    run()
