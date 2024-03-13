"""Generates test data from measurement file for mirror type 'multi_facet'.
"""
from os.path import join, dirname, exists

import matplotlib.pyplot as plt
import numpy as np

from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe as Measurement
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.common.lib.deflectometry.Display import Display
from opencsp.common.lib.deflectometry.EnsembleData import EnsembleData
from opencsp.common.lib.deflectometry.FacetData import FacetData
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir


def generate_dataset(
    file_measurement: str,
    file_camera: str,
    file_display: str,
    file_calibration: str,
    file_facet: str,
    file_ensemble: str,
    file_dataset_out: str,
):
    """Generates and saves test data"""
    # Check output file exists
    if not exists(dirname(file_dataset_out)):
        raise FileNotFoundError(f'Output directory {file_dataset_out:s} does not exist.')

    # Load components
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    measurement = Measurement.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
    ensemble_data = EnsembleData.load_from_json(file_ensemble)
    facet_data = [FacetData.load_from_json(file_facet)] * ensemble_data.num_facets

    # Calibrate fringes
    measurement.calibrate_fringe_images(calibration)

    # Create sofast object
    sofast = Sofast(measurement, camera, display)

    # Update image processing parameters
    sofast.params.mask_hist_thresh = 0.83
    sofast.params.geometry_params.perimeter_refine_perpendicular_search_dist = 10.0
    sofast.params.geometry_params.facet_corns_refine_frac_keep = 1.0
    sofast.params.geometry_params.facet_corns_refine_perpendicular_search_dist = 3.0
    sofast.params.geometry_params.facet_corns_refine_step_length = 5.0

    # Define surface data
    surface_data = [
        dict(
            surface_type='parabolic',
            initial_focal_lengths_xy=(100.0, 100.0),
            robust_least_squares=False,
            downsample=10,
        )
    ] * ensemble_data.num_facets

    # Process optic data
    sofast.process_optic_multifacet(facet_data, ensemble_data, surface_data)

    # Save data
    sofast.save_to_hdf(file_dataset_out)
    display.save_to_hdf(file_dataset_out)
    camera.save_to_hdf(file_dataset_out)
    calibration.save_to_hdf(file_dataset_out)
    print(f'Data saved to: {file_dataset_out:s}')

    # Show slope map
    mask = sofast.data_image_processing_general.mask_raw
    image = np.zeros(mask.shape) * np.nan
    for idx in range(sofast.num_facets):
        mask = sofast.data_image_processing_facet[idx].mask_processed
        slopes_xy = sofast.data_characterization_facet[idx].slopes_facet_xy
        slopes = np.sqrt(np.sum(slopes_xy**2, 0))
        image[mask] = slopes
    plt.figure()
    plt.imshow(image, cmap='jet')
    plt.title('Slope Magnitude')
    plt.show()


if __name__ == '__main__':
    # Generate measurement set 1
    base_dir = join(opencsp_code_dir(), 'test/data/sofast_measurements')

    generate_dataset(
        file_measurement=join(base_dir, 'measurement_ensemble.h5'),
        file_camera=join(base_dir, 'camera.h5'),
        file_display=join(base_dir, 'display_distorted_2d.h5'),
        file_calibration=join(base_dir, 'calibration.h5'),
        file_facet=join(base_dir, 'Facet_lab_6x4.json'),
        file_ensemble=join(base_dir, 'Ensemble_lab_6x4.json'),
        file_dataset_out=join(base_dir, 'calculations_facet_ensemble/data.h5'),
    )
