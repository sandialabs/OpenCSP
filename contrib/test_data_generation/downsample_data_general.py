"""Library of functions to downsample high-res image and other data
"""

import numpy as np
from scipy.signal import convolve2d

from opencsp.common.lib.camera.Camera import Camera


def downsample_camera(file: str, n: int) -> Camera:
    """Returns downsamples camera file

    Parameters
    ----------
    file : str
        High res camera file
    n : int
        Downsample factor

    Returns
    -------
    Camera
    """
    # Load camera
    camera_orig = Camera.load_from_hdf(file)

    # Downsample camera
    return Camera(
        intrinsic_mat=camera_orig.intrinsic_mat / float(n),
        distortion_coef=camera_orig.distortion_coef,
        image_shape_xy=np.floor(camera_orig.image_shape_xy / float(n)).astype(int),
        name=camera_orig.name,
    )


def downsample_images(images: np.ndarray, n: int) -> np.ndarray:
    """Returns downsampled version of input images

    Parameters
    ----------
    images : np.ndarray
        Input m x n( x N) image array
    n : int
        Downsample factor

    Returns
    -------
    np.ndarray
        Downsampled m x n( x N) image array as uint8
    """
    ker = np.ones((n, n), float) / float(n**2)
    images_ds_list = []

    if np.ndim(images) == 2:  # one, single 2d image
        n_images = 1
        images = images[:, :, None]
    else:  # multiple 2d images
        n_images = images.shape[2]

    for idx_im in range(n_images):
        images_ds_list.append(convolve2d(images[..., idx_im], ker, mode="valid")[::n, ::n, None].astype("uint8"))

    images_out = np.concatenate(images_ds_list, 2)

    if n_images == 1:
        images_out = images_out[..., 0]

    return images_out
