import os
from   os.path import join

from   opencsp.common.lib.deflectometry.Display import Display
from   opencsp.common.lib.deflectometry.EnsembleData import EnsembleData
from   opencsp.common.lib.deflectometry.FacetData import FacetData
from   opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from   opencsp.app.sofast.lib.Measurement import Measurement
from   opencsp.app.sofast.lib.Sofast import Sofast
from   opencsp.common.lib.camera.Camera import Camera


def gen_data_multi_facet(file_measurement: str,
                         file_camera: str,
                         file_display: str,
                         file_calibration: str,
                         file_facet: str,
                         file_ensemble: str,
                         file_dataset_out: str):
    """Generates full-resolution dataset of SOFAST characterization data for a multi-facet
    Sofast data collection.
    """
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
    S = Sofast(measurement, camera, display)

    # Define surface data
    surface_data = [dict(surface_type='parabolic',
                         initial_focal_lengths_xy=(1000., 1000.),
                         robust_least_squares=False,
                         downsample=10)] * ensemble_data.num_facets

    # Process optic data
    S.process_optic_multifacet(facet_data, ensemble_data, surface_data)

    # Save data
    S.save_data_to_hdf(file_dataset_out)
    display.save_to_hdf(file_dataset_out)
    camera.save_to_hdf(file_dataset_out)
    calibration.save_to_hdf(file_dataset_out)
    print(f'Multi-facet data saved to: {os.path.abspath(file_dataset_out):s}')


def gen_data_single_facet(file_measurement: str,
                          file_camera: str,
                          file_display: str,
                          file_calibration: str,
                          file_facet: str,
                          file_dataset_out: str):
    """Generates and saves dataset"""
    # Load components
    camera = Camera.load_from_hdf(file_camera)
    display = Display.load_from_hdf(file_display)
    measurement = Measurement.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)
    facet_data = FacetData.load_from_json(file_facet)

    # Calibrate fringes
    measurement.calibrate_fringe_images(calibration)

    # Creates sofast object
    S = Sofast(measurement, camera, display)

    # Define surface data
    surface_data = dict(surface_type='parabolic',
                        initial_focal_lengths_xy=(100., 100.),
                        robust_least_squares=True,
                        downsample=10)

    # Process optic data
    S.process_optic_singlefacet(facet_data, surface_data)

    # Check output file exists
    if not os.path.exists(os.path.dirname(file_dataset_out)):
        os.mkdir(os.path.dirname(file_dataset_out))

    # Save testing data
    S.save_data_to_hdf(file_dataset_out)
    display.save_to_hdf(file_dataset_out)
    camera.save_to_hdf(file_dataset_out)
    calibration.save_to_hdf(file_dataset_out)
    print(f'Single facet data saved to: {os.path.abspath(file_dataset_out):s}')


def example_driver():
    """Generates characterized SOFAST files on full-resolution SOFAST sample data.
    Saves the data files in the sample data directory.
    """

    # Define location of sample data
    sample_data_dir = os.path.join(os.path.dirname(__file__), '../../../sample_data/sofast')
    print(f'Processing data from sample data directory: {os.path.abspath(sample_data_dir):s}')

    # Generate multi-facet data for measurement set 1
    gen_data_multi_facet(
        file_measurement=join(sample_data_dir, 'measurement_set_1/measurement_ensemble.h5'),
        file_camera=join(sample_data_dir, 'measurement_set_1/camera.h5'),
        file_display=join(sample_data_dir, 'measurement_set_1/display_distorted_2d.h5'),
        file_calibration=join(sample_data_dir, 'measurement_set_1/calibration.h5'),
        file_facet=join(sample_data_dir, 'measurement_set_1/Facet_lab_6x4.json'),
        file_ensemble=join(sample_data_dir, 'measurement_set_1/Ensemble_lab_6x4.json'),
        file_dataset_out=join(sample_data_dir, 'measurement_set_1/calculations_facet_ensemble/data.h5')
    )

    # Generate single-facet data for measurement set 1
    gen_data_single_facet(
        file_measurement=join(sample_data_dir, 'measurement_set_1/measurement_facet.h5'),
        file_camera=join(sample_data_dir, 'measurement_set_1/camera.h5'),
        file_display=join(sample_data_dir, 'measurement_set_1/display_distorted_2d.h5'),
        file_calibration=join(sample_data_dir, 'measurement_set_1/calibration.h5'),
        file_facet=join(sample_data_dir, 'measurement_set_1/Facet_NSTTF.json'),
        file_dataset_out=join(sample_data_dir, 'measurement_set_1/calculations_facet/data.h5')
    )

    # Generate multi-facet data for measurement set 2
    gen_data_multi_facet(
        file_measurement=join(sample_data_dir, 'measurement_set_2/measurement_ensemble.h5'),
        file_camera=join(sample_data_dir, 'measurement_set_2/camera.h5'),
        file_display=join(sample_data_dir, 'measurement_set_2/display.h5'),
        file_calibration=join(sample_data_dir, 'measurement_set_2/calibration.h5'),
        file_facet=join(sample_data_dir, 'measurement_set_2/Facet_NSTTF.json'),
        file_ensemble=join(sample_data_dir, 'measurement_set_2/Ensemble_NSTTF_heliostat_spherical_80m.json'),
        file_dataset_out=join(sample_data_dir, 'measurement_set_2/calculations_facet_ensemble/data.h5')
    )

if __name__ == '__main__':
    example_driver()