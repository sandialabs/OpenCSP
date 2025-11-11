import configparser
import os

import cv2 as cv
import numpy as np

from contrib.common.lib.cv.spot_analysis.image_processor import *
from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis
from opencsp.common.lib.cv.annotations.HotspotAnnotation import HotspotAnnotation
from contrib.common.lib.cv.annotations.MomentsAnnotation import MomentsAnnotation
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it


def main(input_images: list[str], results_dir: str, experiment_name: str):
    def centroid_pixel_locator(operable: SpotAnalysisOperable) -> tuple[int, int]:
        """Returns the x/y pixel location of the centroid center"""
        moments = filter(lambda a: isinstance(a, MomentsAnnotation), operable.annotations)
        mom_annotation: MomentsAnnotation = list(moments)[-1]
        ret = mom_annotation.centroid.astuple()
        return (int(ret[0]), int(ret[1]))

    def hotspot_pixel_locator(operable: SpotAnalysisOperable) -> tuple[int, int]:
        """Returns the x/y pixel location of the hotspot center"""
        hotspots = filter(lambda a: isinstance(a, HotspotAnnotation), operable.annotations)
        hs_annotation: HotspotAnnotation = list(hotspots)[-1]
        ret = hs_annotation.origin.astuple()
        return (int(ret[0]), int(ret[1]))

    def rgb2gray(operable: SpotAnalysisOperable) -> np.ndarray:
        grayscale_image = cv.cvtColor(operable.primary_image.nparray, cv.COLOR_RGB2GRAY)
        return grayscale_image

    image_processors = {
        "EchoEcho": EchoImageProcessor(),
        "Original": ConvolutionImageProcessor(diameter=1),
        "Rgb2Gray": CustomSimpleImageProcessor(rgb2gray),
        "PopStats": PopulationStatisticsImageProcessor(),
        "Centroid": MomentsImageProcessor(
            include_visualization=True, centroid_style=rcps.default(color=color.cyan(), markersize=20)
        ),
        "VFalseCl": ViewFalseColorImageProcessor(),
        "VCentOrg": ViewAnnotationsImageProcessor(base_image_selector='visualization'),
        "CropCent": CroppingImageProcessor(centered_location=centroid_pixel_locator, width_height=(1500, 1500)),
        "HotSpotS": HotspotImageProcessor(
            21, draw_debug_view=False, record_visualization=False, record_debug_view=False
        ),
        "VFalseC2": ViewFalseColorImageProcessor(),
        "VHotspot": ViewAnnotationsImageProcessor([HotspotAnnotation], base_image_selector='visualization'),
        "CropHots": CroppingImageProcessor(centered_location=hotspot_pixel_locator, width_height=(250, 250)),
        "Centrod2": MomentsImageProcessor(
            include_visualization=True, centroid_style=rcps.default(color=color.cyan(), markersize=20)
        ),
        "SpotSize": SpotWidthImageProcessor(spot_width_technique="fwhm"),
        "VFalseC3": ViewFalseColorImageProcessor(),
        "VOverExp": ViewHighlightImageProcessor(black_highlight_color=(70, 0, 70), white_highlight_color=(70, 70, 0)),
        "VAnnotat": ViewAnnotationsImageProcessor(base_image_selector='visualization'),
        "VCrosSec": ViewCrossSectionImageProcessor(
            centroid_pixel_locator, single_plot=False, y_range=(0, 255), base_image_selector='visualization'
        ),
        'Ve3d': View3dImageProcessor(max_resolution=(100, 100)),
    }
    _p = image_processors
    # fmt: off
    processors_per_slide = [
        [
            (_p["Original"], "Original"),
            (_p["Rgb2Gray"], "Grayscale"),
            (_p["Centroid"], "Centroid & Principle Axis"),
            (_p["CropCent"], "Cropped to Centroid"),
        ],
        [
            (_p["VFalseC2"], "Cropped to Centroid"),
            (_p["VHotspot"], "Hotspot"),
            (_p["CropHots"], "Cropped to Hotspot"),
        ],
        [
            (_p["CropHots"], "Cropped Image"),
            _p["VAnnotat"],
            _p["VCrosSec"],
            _p["Ve3d"]
        ],
    ]
    # fmt: on
    image_processors["PowerPnt"] = PowerpointImageProcessor(
        results_dir, experiment_name, processors_per_slide=processors_per_slide
    )
    image_processors_list = list(image_processors.values())

    spot_analysis = SpotAnalysis(experiment_name, image_processors_list, save_dir=results_dir)
    spot_analysis.set_primary_images(input_images)

    for result in spot_analysis:
        pass


if __name__ == "__main__":
    experiment_settings_file = ft.join(ft.path_components(__file__)[0], "experiment_settings.ini")
    experiment_settings = configparser.ConfigParser()
    experiment_settings.read(experiment_settings_file)

    experiment_name = experiment_settings["DEFAULT"]["experiment_name"]
    input_dir = experiment_settings["DEFAULT"]["input_dir"]
    process_dir = experiment_settings["DEFAULT"]["process_dir"]

    input_images = [ft.join(input_dir, fn) for fn in it.image_files_in_directory(input_dir)]

    if ft.directory_exists(process_dir):
        if True:
            ft.delete_files_in_directory(process_dir, "*.pptx")
            ft.delete_files_in_directory(process_dir, "*.jpg")
            ft.delete_files_in_directory(process_dir, "*.png")
            ft.delete_files_in_directory(process_dir, "*.txt")
    ft.create_directories_if_necessary(process_dir)

    main(input_images, process_dir, experiment_name)
