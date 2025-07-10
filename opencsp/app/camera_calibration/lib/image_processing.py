"""Library of image processing functions used for camera calibration"""

import cv2 as cv
import numpy as np

from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz


def find_checkerboard_corners(npts: tuple[int, int], img: np.ndarray) -> tuple[Vxyz, Vxy]:
    """
    Finds checkerboard corners in given image.

    Parameters
    ----------
    npts : tuple (x, y)
        Number of corners to find. Corners must be surrounded on all four
        sides by complete black/white squares that are NOT on the edge of
        the pattern.
    img : 2D numpy array, uint8
        Image containing checkerboard image. Should be opened using
        cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    Returns
    -------
    p_object : Vxyz
        Location of corners in target grid coordinates
    p_corners_refined : Vxy
        Location of corners in camera pixels

    """
    # Find corners
    chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv.findChessboardCorners(img, npts, chessboard_flags)

    # Check corners were found
    if ret is not True:
        return None, None

    # Process corners
    p_corners = Vxy(corners[:, 0, :].T, dtype=np.float32)

    # Refine the corners
    p_corners_refined = refine_checkerboard_corners(img, p_corners, window_size=(11, 11))

    # Define object points (x,y,z) as in: (0,0,0), (1,0,0), (2,0,0), ... (6,5,0)
    objp = np.zeros((npts[0] * npts[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0 : npts[0], 0 : npts[1]].T.reshape(-1, 2)
    p_object = Vxyz(objp.T, dtype=np.float32)

    return p_object, p_corners_refined


def refine_checkerboard_corners(
    image: np.ndarray,
    p_image: Vxy,
    window_size: tuple[int, int] = (10, 10),
    max_iterations: int = 40,
    precision: float = 0.001,
) -> Vxy:
    """
    Refines rough locations of checkerboard corners

    Parameters
    ----------
    image : np.ndarray
        Image to refine with, uint8.
    p_image : Vxy
        Rough u/v corner locations, float32.
    window_size : tuple[int, int], optional
        Search window half-width. The default is (10, 10).
    max_iterations : int, optional
        Max number of search iterations. The default is 40.
    precision : float, optional
        Desired precision in pixels. The default is 0.001.

    Returns
    -------
    Vxy
        Refined image points.

    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, max_iterations, precision)
    imgpts_in = p_image.data.T.copy()
    imgpts_refined = cv.cornerSubPix(image, imgpts_in, window_size, (-1, -1), criteria)
    return Vxy(imgpts_refined.T, dtype=np.float32)


def annotate_found_corners(npts: tuple[int, int], img: np.ndarray, img_points: Vxy) -> None:
    """
    Updates an image with found checkerboard corner annotations.

    Parameters
    ----------
    npts : tuple (x, y), int
        Number of corners in image to annotate.
    img : ndarray (RGB or grayscale)
        The checkerboard image. This is updated with annotations.
    img_points : Vxy
        Points found in image. Output from cv.findChessboardCorners.

    """
    cv.drawChessboardCorners(img, npts, img_points.data.T, True)
