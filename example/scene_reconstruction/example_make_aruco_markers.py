"""
Script that generates Aruco marker PNG files of given size and number
"""

from os.path import dirname, join

import cv2 as cv
import cv2.aruco as aruco
import imageio.v3 as imageio
import numpy as np

import opencsp.common.lib.tool.file_tools as ft


def make_aruco_images(save_path: str, number: str, size: int = 500, padding: int = 50):
    """Generates aruco marker images and saves images as PNG files

    Parameters
    ----------
    save_path : str
        Path to save marker images
    number : str
        Number of markers to make, starting at ID=0
    size : int, optional
        Size of active area, pixels, by default 500
    padding : int, optional
        Size of white border, pixels, by default 50
    """
    # Define marker parameters
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    side_pixels = size  # Side width
    font_thickness = 2
    pad_width = padding

    for id_ in range(number):
        # Define marker image
        img = np.ones((side_pixels, side_pixels), dtype='uint8') * 255

        # Create marker image
        aruco.drawMarker(dictionary, id_, side_pixels, img[-side_pixels:, :])

        # Add padding
        img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width)), constant_values=255)

        # Add text
        id_string = f'ID: {id_:d}'
        cv.putText(img, id_string, (20, side_pixels + 2 * pad_width - 4), 0, 1, 0, font_thickness)

        # Save image
        imageio.imwrite(join(f'{save_path:s}', f'{id_:03d}.png'), img)


def example_make_aruco_images():
    """Example script that makes aruco markers images and saves as PNG files.
    These can be printed to make physical aruco markers.
    """
    # Create save path
    path = join(dirname(__file__), 'data/output/aruco_markers')
    ft.create_directories_if_necessary(path)
    # Define number of markers
    num_markers = 10

    make_aruco_images(path, num_markers)


if __name__ == '__main__':
    example_make_aruco_images()
