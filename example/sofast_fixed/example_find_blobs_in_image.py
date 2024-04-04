"""Example script that finds blobs in image and saves annotated image.
"""

from os.path import join, dirname

import cv2 as cv

from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.app.sofast.lib.image_processing import detect_blobs_annotate
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_find_blobs_in_image():
    """Example script that finds blobs in image, annotates image, and saves"""
    # General Setup
    dir_save = join(dirname(__file__), 'data/output/find_blobs_in_image')
    ft.create_directories_if_necessary(dir_save)

    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # Load image from measurement file
    file_meas = join(opencsp_code_dir(), 'test/data/sofast_fixed/data_measurement/measurement_facet.h5')
    measurement = MeasurementSofastFixed.load_from_hdf(file_meas)
    image = measurement.image

    # Load image from existing image file
    # image = cv2.imread(file_jpg, cv2.IMREAD_GRAYSCALE)
    # image = imageio.imread(file_png)

    # Detect blobs and annotate image
    params = cv.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 2
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = 30
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    image_annotate = detect_blobs_annotate(image, params)

    # Save image
    cv.imwrite(join(dir_save, 'annotated_blobs.png'), image_annotate)


if __name__ == '__main__':
    example_find_blobs_in_image()
