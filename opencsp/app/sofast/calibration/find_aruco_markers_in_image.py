"""
Finds all aruco markes and annotates image with labeled IDs and aruco marker corners.
Saves all images to destination folder.
"""
from glob import glob
import os

import argparse
import cv2 as cv
import imageio
import numpy as np

import opencsp.common.lib.photogrammetry.photogrammetry as pg


def main(
    source_pattern: str,
    save_dir: str,
    line_width: int = 1,
    font_thickness: int = 2,
    font_scale: float = 1.25,
):
    """Finds aruco markers, annotates edges, labels, and saves into destination folder

    Parameters
    ----------
    source_pattern : str
        Glob-like search string
    save_dir : str
        Save directory
    line_width : int, optional
        Width of annotation line, by default 1
    font_thickness : int, optional
        Thickness of label font, by default 2
    font_scale : float, optional
        Scale of label font, by default 1.25
    """
    # Define files to search through
    files = glob(source_pattern)

    # Check number of found files
    num_ims = len(files)
    if num_ims == 0:
        print('No files match given pattern')
        return
    else:
        print(f'{num_ims:d} images found.')

    # Check save dir
    if not os.path.exists(save_dir):
        print(f'Save directory, {save_dir:s} does not exist:')
        return

    # Define text parameters
    font_type = cv.FONT_HERSHEY_SIMPLEX

    # Annotate image
    for file in files:
        # Load image
        img_gray = pg.load_image_grayscale(file)
        img_rgb = np.concatenate([img_gray[..., None]] * 3, 2)

        # Find markers
        ids, pts_img = pg.find_aruco_marker(img_gray)

        # Annotate image
        for id_, pt_img in zip(ids, pts_img):
            # Add line
            for i1 in range(4):
                i2 = np.mod(i1 + 1, 4)
                pt1 = pt_img[i1].astype(int)
                pt2 = pt_img[i2].astype(int)
                cv.line(img_rgb, pt1, pt2, (255, 0, 0), line_width)

            # Add text
            orig = (int(pt_img[:, 0].min()), int(pt_img[:, 1].min() - 5))
            cv.putText(
                img_rgb,
                str(id_),
                orig,
                font_type,
                font_scale,
                (0, 0, 255),
                font_thickness,
            )

        # Save image
        save_name = os.path.basename(file)
        print(save_name)
        imageio.imwrite(os.path.join(save_dir, save_name), img_rgb)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Finds all aruco markers in files matching the given source file pattern. Saves annotated images to destination folder.'
    )
    parser.add_argument(
        'source_pattern', metavar='src', type=str, help='Source file pattern.'
    )
    parser.add_argument(
        'save_dir',
        metavar='dst',
        type=str,
        help='Destination folder to save annotated images.',
    )
    parser.add_argument(
        '-w',
        '--line-width',
        metavar='width',
        type=int,
        default=1,
        help='Width of annotation lines in pixels (default=1)',
    )
    parser.add_argument(
        '-t',
        '--font-thickness',
        metavar='thickness',
        type=int,
        default=6,
        help='Thickness of font in pixels (default=6)',
    )
    parser.add_argument(
        '-s',
        '--font-scale',
        metavar='scale',
        type=float,
        default=5,
        help='Font size scale value (default=5)',
    )
    args = parser.parse_args()

    # Annotate images
    main(
        args.source_pattern,
        args.save_dir,
        args.line_width,
        args.font_thickness,
        args.font_scale,
    )
