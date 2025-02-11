"""
Script that generates Aruco marker PNG files of given size and number
"""

from os.path import dirname, join

import cv2 as cv
import cv2.aruco as aruco
import imageio.v3 as imageio
import numpy as np

import opencsp.common.lib.tool.file_tools as ft


def example_make_aruco_images():
    """Example script that makes aruco markers images and saves as PNG files. The
    Aruco marker images are square images with a white border. The Aruco marker ID
    is printed on the bottom left corner of the image. These PNG images can be
    printed to make physical aruco markers.

    This example, when run as-is, will save 10 Aruco marker images with an active aruco
    marker size of 500x500 pixels with 150 pixels of white padding around the edge for a
    final size of 800x800 pixels. The images are saved in:
    /example/scene_reconstruction/data/output/aruco_markers/.

    The following variables can be updated to make markers with different properties:
    - number : the number of Aruco markers to make
    - marker_active_size : the width/height (in pixels) of the entire Aruco marker active
        area (not including the white padding around the edge)
    - padding : the width of the padding (in pixels) around the perimeter of the active
        marker area. NOTE: if the padding is too small, the text label will not be visible.
    - font_thickness : the thickness of the Aruco ID label font on the lower left corner
    - save_path : the save path for generated Aruco marker PNG files
    """
    ### UPDATE THESE PARAMETERS TO MAKE NEW MARKERS ###
    number = 10
    marker_active_size = 500
    padding = 150
    font_thickness = 2
    save_path = join(dirname(__file__), 'data/output/aruco_markers')
    ###################################################

    ft.create_directories_if_necessary(save_path)

    # Define marker parameters
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    for id_ in range(number):
        # Define marker image
        img = np.ones((marker_active_size, marker_active_size), dtype='uint8') * 255

        # Create marker image
        aruco.generateImageMarker(dictionary, id_, marker_active_size, img[-marker_active_size:, :])

        # Add padding
        img = np.pad(img, ((padding, padding), (padding, padding)), constant_values=255)

        # Add text
        id_string = f'ID: {id_:d}'
        cv.putText(
            img,
            id_string,
            (20, marker_active_size + 2 * padding - 20),
            fontFace=0,
            fontScale=1.5,
            color=0,
            thickness=font_thickness,
        )

        # Save image
        imageio.imwrite(join(f'{save_path:s}', f'{id_:03d}.png'), img)


if __name__ == '__main__':
    example_make_aruco_images()
