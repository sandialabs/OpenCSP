import cv2 as cv
import numpy as np


def encode_RG_to_RGB(image: np.ndarray) -> np.ndarray:
    """
    Converts raw data from a Bayer patterned camera to NxMx3 RGB image.
    For each 2x2 pixel block, simply averages green pixels and uses R and
    B pixels as-is.

    Parameters
    ----------
    image : np.ndarray
        Input 2d image directly from RG Bayer pattern sensor. Pixels are
        arranged as:

            R  G  ...
            G  B  ...
            .  .  ...

    Returns
    -------
    np.ndarray
        NxMx3 RGB image
    """
    # Create kernels
    ker_r = np.array([[0.0, 0.0], [0.0, 1.0]])
    ker_g = np.array([[0.0, 0.5], [0.5, 0.0]])
    ker_b = np.array([[1.0, 0.0], [0.0, 0.0]])

    im_r = cv.filter2D(image, -1, ker_r)[::2, ::2, None]
    im_g = cv.filter2D(image, -1, ker_g)[::2, ::2, None]
    im_b = cv.filter2D(image, -1, ker_b)[::2, ::2, None]

    return np.concatenate((im_r, im_g, im_b), 2)


def highlight_saturation(image: np.ndarray, saturation_value: int | float) -> np.ndarray:
    """
    Highlights saturated pixels red. Image can be 2d or 3d, a 3d
    image is returned.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image array
    saturation_value : int | float
        Value that defines saturation in the image

    Returns
    -------
    rgb : ndarray
        3D image array converted to scaled float with range [0, 1].

    """
    # Add red channel if grayscale
    if np.ndim(image) == 2:
        rgb = np.concatenate([image[:, :, np.newaxis]] * 3, 2)
    elif np.ndim(image) == 3:
        rgb = image
    else:
        raise ValueError(f'Input image must have 1 or 3 channels, but image has shape: {image.shape}')

    # Mask saturated pixels
    mask = (rgb >= saturation_value).max(2)
    rgb[mask, 0] = saturation_value
    rgb[mask, 1:] = 0

    # Scale to 0-1 float
    rgb = rgb.astype(np.float32)
    rgb /= np.float32(saturation_value)

    return rgb
