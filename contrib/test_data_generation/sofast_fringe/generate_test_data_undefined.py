"""Generates test data from measurement file for mirror type 'undefined'.
"""

from os.path import join, dirname, exists

import matplotlib.pyplot as plt
import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


def generate_dataset(
    file_measurement: str, file_camera: str, file_display: str, file_calibration: str, file_dataset_out: str
):
    """Generates and saves dataset"""
    # Check output file exists
    if not exists(dirname(file_dataset_out)):
        raise FileNotFoundError(f"Output directory {file_dataset_out:s} does not exist.")

    # Load components
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    measurement = Measurement.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)

    # Calibrate fringes
    measurement.calibrate_fringe_images(calibration)

    # Creates sofast object
    sofast = Sofast(measurement, camera, display)

    # Update mask calculation options
    sofast.params.mask_keep_largest_area = True

    # Define surface data
    surface_data = dict(
        surface_type="parabolic", initial_focal_lengths_xy=(100.0, 100.0), robust_least_squares=False, downsample=10
    )

    # Process optic data
    sofast.process_optic_undefined(surface_data)

    # Save testing data
    sofast.save_to_hdf(file_dataset_out)
    display.save_to_hdf(file_dataset_out)
    camera.save_to_hdf(file_dataset_out)
    calibration.save_to_hdf(file_dataset_out)
    print(f"All data saved to: {file_dataset_out:s}")

    # Show slope map
    mask = sofast.data_image_processing_facet[0].mask_processed
    slopes_xy = sofast.data_calculation_facet[0].slopes_facet_xy
    slopes = np.sqrt(np.sum(slopes_xy**2, 0))
    image = np.zeros(mask.shape) * np.nan
    image[mask] = slopes
    plt.imshow(image, cmap="jet")
    plt.title("Slope Magnitude")
    plt.show()


if __name__ == "__main__":
    # Generate measurement set 1 data
    base_dir = join(opencsp_code_dir(), "test/data/measurements_sofast_fringe")

    generate_dataset(
        file_measurement=join(base_dir, "measurement_facet.h5"),
        file_camera=join(base_dir, "camera.h5"),
        file_display=join(base_dir, "display_distorted_2d.h5"),
        file_calibration=join(base_dir, "image_calibration.h5"),
        file_dataset_out=join(base_dir, "calculations_undefined_mirror/data.h5"),
    )
