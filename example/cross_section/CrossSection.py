# Cross section spot analysis example by Sandia National Laboratories.
#
# This example:
#  - loads images taken of the sun
#  - locates the sun spot within the image
#  - measures the size of the sun spot
#  - produces a cross section of the sun spot
#  - produces additional visualizations
#
# To run this example, copy "experiment_settings_example.ini" to
# "experiment_settings.ini" and modify the values in the ini file to point to
# the example data on your computer.

import configparser

import cv2 as cv
import numpy as np

from contrib.common.lib.cv.spot_analysis.image_processor import *
from opencsp.common.lib.cv.SpotAnalysis import SpotAnalysis
from opencsp.common.lib.cv.annotations.HotspotAnnotation import HotspotAnnotation
from contrib.common.lib.cv.annotations.MomentsAnnotation import MomentsAnnotation
from opencsp.common.lib.cv.fiducials.AbstractFiducials import AbstractFiducials
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


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
        """Converts the input image to grayscale using the OpenCV cvtColor method."""
        grayscale_image = cv.cvtColor(operable.primary_image.nparray, cv.COLOR_RGB2GRAY)
        return grayscale_image

    image_processors = {
        # This image processor just prints out the image name to the terminal,
        # to let us know that the operations have started for the image.
        "EchoEcho": EchoImageProcessor(),
        # This 1-pixel convolution image processor doesn't actually apply any
        # changes to the image, but it does give us a reference that we can
        # use to view the original image later in the Powerpoint deck.
        "Original": ConvolutionImageProcessor(diameter=1),
        # Convert the input image to grayscale and determine the range of
        # parameters for all input images.
        "Rgb2Gray": CustomSimpleImageProcessor(rgb2gray),
        "PopStats": PopulationStatisticsImageProcessor(),
        # Find the centroid of the image. Because the sun is so much brighter
        # than anything else in the sky this should get us pretty close to the
        # sun's location in the image. Crop the image down to this location.
        "Centroid": MomentsImageProcessor(
            include_visualization=True, centroid_style=rcps.default(color=color.cyan(), markersize=20)
        ),
        "VFalseCl": ViewFalseColorImageProcessor(),
        "VCentOrg": ViewAnnotationsImageProcessor(base_image_selector='visualization'),
        "CropCent": CroppingImageProcessor(centered_location=centroid_pixel_locator, width_height=(1500, 1500)),
        "VFalseC2": ViewFalseColorImageProcessor(),
        "VCentCrp": ViewAnnotationsImageProcessor([MomentsAnnotation], base_image_selector='visualization'),
        # Use the hotspot locator to find the actual sun in the image. We use
        # this locator instead of a centroid because lens reflections will
        # cause a centroid to produce an invalid location. For this example
        # a simple 'brightest pixel' locator would also work but would be less
        # robust. Using the hotspot also makes this example applicable to
        # other less bright light sources such as a reflection, flashlight, or
        # laser pointer.
        "HotSpotS": HotspotImageProcessor(
            21, draw_debug_view=False, record_visualization=False, record_debug_view=False
        ),
        "VFalseC3": ViewFalseColorImageProcessor(),
        "VHotspot": ViewAnnotationsImageProcessor([HotspotAnnotation], base_image_selector='visualization'),
        # We know approximately the size of the sun in the image. Crop down to
        # that size to exclude lens reflections.
        "CropHots": CroppingImageProcessor(centered_location=hotspot_pixel_locator, width_height=(250, 250)),
        "VFalseC4": ViewFalseColorImageProcessor(),
        "VHotCrop": ViewAnnotationsImageProcessor([HotspotAnnotation], base_image_selector='visualization'),
        # Now we can use a centroid to find the center of the sun.
        "Centrod2": MomentsImageProcessor(
            include_visualization=True, centroid_style=rcps.default(color=color.green(), markersize=20)
        ),
        # Get the size of the sun in the image with the full width half maximum
        # technique, which works well for light sources that are roughly
        # gaussian.
        "SpotSize": SpotWidthImageProcessor(spot_width_technique="fwhm"),
        # Visualize the sun spot, including over and under exposed pixels, the
        # centroid and hotspot location, a cross section of the sun spot, and a
        # 3D view of the spot.
        "VOverEx2": ViewHighlightImageProcessor(black_highlight_color=(70, 0, 70), white_highlight_color=(70, 70, 0)),
        "VAnnotat": ViewAnnotationsImageProcessor(base_image_selector='visualization'),
        "VCrosSec": ViewCrossSectionImageProcessor(
            centroid_pixel_locator, single_plot=False, y_range=(0, 255), base_image_selector='visualization'
        ),
        "V3dImage": View3dImageProcessor(max_resolution=(100, 100)),
    }

    # Create the PowerpointImageProcessor which will generate the Powerpoint
    # deck with the results from the image processors.
    #
    # In this example, there will be three slides per input image, and each
    # slide will contain multiple results. If the referenced image processor
    # is a visualization image processor (name starts with "View"), then all
    # visualization images from that processor will be included.
    #
    # For each image processor included in a slide, a secondary parameter can
    # be provided for a caption to be applied to the images for that processor.
    # If no caption is provided, then a default caption is applied.
    _p = image_processors  # shortened variable name for readability
    # fmt: off
    processors_per_slide = [
        [
            (_p["Original"], "Original"),
            (_p["Rgb2Gray"], "Grayscale"),
            (_p["Centroid"], "Centroid & Principle Axis"),
            (_p["VCentCrp"], "Cropped to Centroid"),
        ],
        [
            (_p["VCentCrp"], "Cropped to Centroid"),
            (_p["VHotspot"], "Hotspot"),
            (_p["CropHots"], "Cropped to Hotspot"),
            (_p["VHotCrop"], "Cropped to Hotspot"),
        ],
        [
            (_p["CropHots"], "Cropped to Hotspot"),
            (_p["VAnnotat"], "Cropped Annotations"),
            _p["VCrosSec"],
            _p["V3dImage"]
        ],
    ]
    # fmt: on
    image_processors["PowerPnt"] = PowerpointImageProcessor(
        results_dir, experiment_name, processors_per_slide=processors_per_slide
    )

    # Create the SpotAnalysis instance that will coordinate and evaluate all
    # the spot analysis image processors.
    image_processors_list = list(image_processors.values())
    spot_analysis = SpotAnalysis(experiment_name, image_processors_list, save_dir=results_dir)
    spot_analysis.set_primary_images(input_images)

    for result in spot_analysis:
        pass

    ppt: PowerpointImageProcessor = image_processors["PowerPnt"]
    lt.info(f"Powerpoint deck saved to {ppt.dest_path_name_ext}")


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
