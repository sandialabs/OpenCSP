"""Example script that finds blobs in image and saves annotated image.
"""
import os
from os.path import join, dirname, exists

import cv2 as cv

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.app.sofast.lib.MeasurementFixedPattern import MeasurementFixedPattern
from opencsp.common.lib.deflectometry.image_processing import detect_blobs_annotate


def example_find_blobs_in_image():
    """Finds blobs in image, annotates, and saves image"""
    file_meas = join(
        opencsp_code_dir(),
        '../../sample_data/deflectometry/sandia_lab/fixed_pattern/measurement_screen_square_width3_space6.h5',
    )
    file_save = join(
        dirname(__file__), 'data/output/blob_detection/image_with_detected_blobs.png'
    )

    if not exists(dirname(file_save)):
        os.makedirs(dirname(file_save))

    # Load image
    measurement = MeasurementFixedPattern.load_from_hdf(file_meas)
    image = measurement.image

    # Detect blobs
    params = cv.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 2
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = 30
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    image_annotate = detect_blobs_annotate(image, params)

    cv.imwrite(file_save, image_annotate)


if __name__ == '__main__':
    example_find_blobs_in_image()
