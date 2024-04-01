from os.path import join, dirname

import cv2 as cv

import opencsp.app.sofast.lib.image_processing as ip
from opencsp.app.sofast.lib.MeasurementSofastFixed import MeasurementSofastFixed
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.photogrammetry.photogrammetry as ph
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


def example_create_measurement_file_from_png():
    """Example that creates a SofastFixed measurement file from a PNG image. The PNG image has
    a point LED light near origin dot.
    1. Load image
    2. Define measurement parameters
    3. Find location of origin point
    4. Create measurement object
    5. Save measurement as HDF5 file
    6. Plot image with annotated origin dot
    """
    # General setup
    # =============
    dir_save = join(dirname(__file__), 'data/output/measurement_file')
    ft.create_directories_if_necessary(dir_save)
    lt.logger(join(dir_save, 'log.txt'), lt.log.INFO)

    # 1. Load image
    # =============
    file_image = join(
        opencsp_code_dir(), 'test/data/sofast_fixed/data_measurement/measurement_image.png'
    )
    image = ph.load_image_grayscale(file_image)

    # 2. Define measurement parameters
    # ================================
    v_measure_point_facet = Vxyz((0, 0, 0))  # meters
    dist_optic_screen = 10.008  # meters
    name = 'NSTTF Facet'

    # 3. Find location of origin point
    # ================================

    # Find point light
    params = cv.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 2
    params.filterByArea = True
    params.minArea = 3
    params.maxArea = 30
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = False
    params.filterByInertia = False
    origin = ip.detect_blobs_inverse(image, params)

    # Check only one origin point was found
    if len(origin) != 1:
        lt.error_and_raise(ValueError, f'Expected 1 origin point, found {len(origin):d}.')

    # 4. Create measurement object
    # ============================
    measurement = MeasurementSofastFixed(image, v_measure_point_facet, dist_optic_screen, origin, name=name)

    # 5. Save measurement as HDF5 file
    # ================================
    measurement.save_to_hdf(join(dir_save, 'measurement.h5'))

    # 6. Plot image with annotated origin dot
    # =======================================
    image_annotated = ip.detect_blobs_inverse_annotate(image, params)

    figure_control = rcfg.RenderControlFigure(tile_array=(1, 1), tile_square=True)
    axis_control = rca.image()
    fig_record = fm.setup_figure(figure_control, axis_control, title='Annotated Origin Point')
    fig_record.axis.imshow(image_annotated)
    fig_record.save(dir_save, 'annotated_image_with_origin_point', 'png')


if __name__ == '__main__':
    example_create_measurement_file_from_png()
