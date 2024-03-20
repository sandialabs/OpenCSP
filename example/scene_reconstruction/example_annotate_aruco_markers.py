from glob import glob
from os.path import join, exists, basename, dirname

import cv2 as cv
import imageio
import numpy as np

import opencsp.common.lib.photogrammetry.photogrammetry as pg
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir
import opencsp.common.lib.tool.file_tools as ft


def annotate_aruco_markers(
    source_pattern: str,
    save_dir: str,
    line_width: int = 1,
    font_thickness: int = 2,
    font_scale: float = 2,
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
        raise ValueError('No files match given pattern')
    else:
        print(f'{num_ims:d} images found.')

    # Check save dir
    if not exists(save_dir):
        raise FileNotFoundError(f'Save directory, {save_dir:s} does not exist:')

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
        save_name = basename(file)
        print(save_name)
        imageio.imwrite(join(save_dir, save_name), img_rgb)


def example_annotate_aruco_markers():
    """Example script that annotates aruco markers found in imput images matching
    the given source_pattern. Markers are outlined in red and labeled in blue text.
    """
    source_pattern = join(
        opencsp_code_dir(),
        'app/scene_reconstruction/test/data',
        'data_measurement/aruco_marker_images/DSC0365*.JPG',
    )
    save_dir = join(dirname(__file__), 'data/output/annotated_aruco_markers')

    ft.create_directories_if_necessary(save_dir)

    # Annotate images
    annotate_aruco_markers(source_pattern, save_dir)


if __name__ == "__main__":
    example_annotate_aruco_markers()
