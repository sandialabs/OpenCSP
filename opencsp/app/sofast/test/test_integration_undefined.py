"""Integration test. Testing processing of an 'undefined' type optic.
"""
import os

import numpy as np

import opencsp
from   opencsp.common.lib.deflectometry.Display import Display
from   opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from   opencsp.app.sofast.lib.Measurement import Measurement
from   opencsp.app.sofast.lib.Sofast import Sofast
from   opencsp.common.lib.camera.Camera import Camera
from   opencsp.common.lib.tool.hdf5_tools import load_hdf5_datasets


def test_undefined():
    # Get test data location
    base_dir = os.path.join(os.path.dirname(opencsp.__file__), 'test/data/sofast_measurements')

    # Directory Setup
    file_dataset = os.path.join(base_dir, 'calculations_undefined_mirror/data.h5')
    file_measurement = os.path.join(base_dir, 'measurement_facet.h5')

    # Load data
    camera = Camera.load_from_hdf(file_dataset)
    display = Display.load_from_hdf(file_dataset)
    measurement = Measurement.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_dataset)

    # Calibrate measurement
    measurement.calibrate_fringe_images(calibration)

    # Load calculation/user data
    datasets = [
        'DataSofastCalculation/facet/facet_000/slopes_facet_xy',
        'DataSofastCalculation/facet/facet_000/slope_coefs_facet',
        'DataSofastInput/surface_params/facet_000/initial_focal_lengths_xy',
        'DataSofastInput/surface_params/facet_000/robust_least_squares',
        'DataSofastInput/surface_params/facet_000/downsample',
        'DataSofastInput/surface_params/facet_000/surface_type',
    ]
    data = load_hdf5_datasets(datasets, file_dataset)

    # Load sofast params
    datasets = [
        'DataSofastInput/sofast_params/mask_hist_thresh',
        'DataSofastInput/sofast_params/mask_filt_width',
        'DataSofastInput/sofast_params/mask_filt_thresh',
        'DataSofastInput/sofast_params/mask_thresh_active_pixels',
        'DataSofastInput/sofast_params/mask_keep_largest_area',
        'DataSofastInput/sofast_params/perimeter_refine_axial_search_dist',
        'DataSofastInput/sofast_params/perimeter_refine_perpendicular_search_dist',
        'DataSofastInput/sofast_params/facet_corns_refine_step_length',
        'DataSofastInput/sofast_params/facet_corns_refine_perpendicular_search_dist',
        'DataSofastInput/sofast_params/facet_corns_refine_frac_keep',
    ]
    params = load_hdf5_datasets(datasets, file_dataset)

    # Instantiate sofast object
    sofast = Sofast(measurement, camera, display)

    # Update parameters
    sofast.params.mask_hist_thresh = params['mask_hist_thresh']
    sofast.params.mask_filt_width = params['mask_filt_width']
    sofast.params.mask_filt_thresh = params['mask_filt_thresh']
    sofast.params.mask_thresh_active_pixels = params['mask_thresh_active_pixels']
    sofast.params.mask_keep_largest_area = params['mask_keep_largest_area']
    sofast.params.geometry_params.perimeter_refine_axial_search_dist = params['perimeter_refine_axial_search_dist']
    sofast.params.geometry_params.perimeter_refine_perpendicular_search_dist = params['perimeter_refine_perpendicular_search_dist']
    sofast.params.geometry_params.facet_corns_refine_step_length = params['facet_corns_refine_step_length']
    sofast.params.geometry_params.facet_corns_refine_perpendicular_search_dist = params['facet_corns_refine_perpendicular_search_dist']
    sofast.params.geometry_params.facet_corns_refine_frac_keep = params['facet_corns_refine_frac_keep']

    # Define surface data
    if data['surface_type'] == 'parabolic':
        surface_data = dict(surface_type=data['surface_type'],
                            initial_focal_lengths_xy=data['initial_focal_lengths_xy'],
                            robust_least_squares=bool(data['robust_least_squares']),
                            downsample=data['downsample'])
    else:
        surface_data = dict(surface_type=data['surface_type'],
                            robust_least_squares=bool(data['robust_least_squares']),
                            downsample=data['downsample'])

    # Run SOFAST
    sofast.process_optic_undefined(surface_data)

    # Test
    slopes = sofast.data_characterization_facet[0].slopes_facet_xy
    slope_coefs = sofast.data_characterization_facet[0].slope_coefs_facet

    np.testing.assert_allclose(data['slopes_facet_xy'], slopes, atol=1e-7, rtol=0)
    np.testing.assert_allclose(data['slope_coefs_facet'], slope_coefs, atol=1e-8, rtol=0)


if __name__ == '__main__':
    test_undefined()
    print('All tests run.')
