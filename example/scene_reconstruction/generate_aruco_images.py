"""
Script that generates Aruco marker PNG files of given size and number
"""
import argparse
import os

import cv2 as cv
import imageio.v3 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        prog='generate_aruco_images',
        description='Generates desired number of Aruco markers and saves as PNG files. NOTE: The "save_path" must use python type slashes (/)',
    )
    parser.add_argument('save_path', type=str, help='Location to save marker PNGs.')
    parser.add_argument(
        'number', type=int, help='Number of images to generate (1-1000).'
    )
    parser.add_argument(
        '-s',
        '--size',
        metavar='size',
        default=500,
        type=int,
        help='Side length in pixels of generated images not including padding (default=500).',
    )
    parser.add_argument(
        '-p',
        '--padding',
        metavar='padding',
        default=50,
        type=int,
        help='Size of white border to pad image in pixels (default=50)',
    )
    args = parser.parse_args()

    # Define marker parameters
    dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_1000)
    side_pixels = args.size  # Side width
    font_thickness = 2
    pad_width = args.padding

    for id_ in range(args.number):
        # Define marker image
        img = np.ones((side_pixels, side_pixels), dtype='uint8') * 255

        # Create marker image
        cv.aruco.drawMarker(dictionary, id_, side_pixels, img[-side_pixels:, :])

        # Add padding
        img = np.pad(
            img, ((pad_width, pad_width), (pad_width, pad_width)), constant_values=255
        )

        # Add text
        id_string = f'ID: {id_:d}'
        cv.putText(
            img,
            id_string,
            (20, side_pixels + 2 * pad_width - 4),
            0,
            1,
            0,
            font_thickness,
        )

        # Save image
        imageio.imwrite(os.path.join(f'{args.save_path:s}', f'{id_:03d}.png'), img)


if __name__ == '__main__':
    main()
