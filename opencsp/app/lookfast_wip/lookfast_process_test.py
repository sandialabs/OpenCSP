import json
import os
from logging import DEBUG

import imageio.v3 as imageio
import cv2
import numpy as np
import matplotlib.pyplot as plt

from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.SofastConfiguration import SofastConfiguration
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.deflectometry.Surface2DParabolic import Surface2DParabolic
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt


import opencsp.app.sofast.lib.image_processing as imgp
import opencsp.app.lookback.lookback_tools as lbt

# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_lookback_main_debug.txt",
    log_type=DEBUG,
)


def create_lookfast_data_structure():

    primary_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025"
    video_name = "DSC_0025.MOV"
    checkpoint_folder = os.path.join(primary_folder, "0_checkpoints")
    checkpoint_main_name = "lookfast_test_checkpoint.json"

    # Define save dir
    dir_save = os.path.join(primary_folder, "lookfast_test")
    ft.create_directories_if_necessary(dir_save)

    # Define sample data directory
    dir_data_sofast = os.path.join(opencsp_code_dir(), "test/data/sofast_fringe")
    dir_data_common = os.path.join(opencsp_code_dir(), "test/data/sofast_common")

    # Directory Setup
    file_measurement = os.path.join(dir_data_sofast, "data_measurement/measurement_facet.h5")
    file_camera = os.path.join(dir_data_common, "camera_sofast_downsampled.h5")
    file_display = os.path.join(dir_data_common, "display_distorted_2d.h5")
    file_orientation = os.path.join(dir_data_common, "spatial_orientation.h5")
    file_calibration = os.path.join(dir_data_sofast, "data_measurement/image_calibration.h5")
    file_facet = os.path.join(dir_data_common, "Facet_NSTTF.json")

    # =================================================
    camera = Camera.load_from_hdf(file_camera)  # Calibrate the Normal Way
    orientation = SpatialOrientation()

    # orientation
    light_image = cv2.imread(os.path.join(primary_folder, "light_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    dark_image = cv2.imread(os.path.join(primary_folder, "dark_mask_test.png"), cv2.IMREAD_GRAYSCALE)

    mask_raw = imgp.calc_mask_raw(
        np.concatenate((dark_image[:, :, np.newaxis], light_image[:, :, np.newaxis]), axis=2),
        hist_thresh=0.5,
        filt_width=9,
        filt_thresh=4,
        thresh_active_pixels=0.01,
    )
    mask = imgp.keep_largest_mask_area(mask_raw)
    v_mask_centroid_image = imgp.centroid_mask(mask)
    v_edges_image = imgp.edges_from_mask(mask)

    display = Display.load_from_hdf(file_display)  # TODO How to approach the screen

    # orientation = SpatialOrientation.load_from_hdf(file_orientation)

    measurement = MeasurementSofastFringe.load_from_hdf(file_measurement)
    calibration = ImageCalibrationScaling.load_from_hdf(file_calibration)  # Projector to Camera Calibration Curves
    facet_data = DefinitionFacet.load_from_json(file_facet)


if __name__ == "__main__":
    create_lookfast_data_structure()
