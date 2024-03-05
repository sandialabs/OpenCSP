"""Library of functions used for Sofast screen distortion calculations
"""
import numpy as np
from scipy import interpolate
from scipy.signal import medfilt

from opencsp.common.lib.geometry.Vxy import Vxy


def interp_xy_screen_positions(
    im_x: np.ndarray, im_y: np.ndarray, x_sc: np.ndarray, y_sc: np.ndarray
) -> Vxy:
    """
    Calculates the interpolated XY screen positions given X/Y fractional
    screen maps and X/Y interpolation vectors.

    Parameters
    ----------
    im_x : np.ndarray
        2D ndarray. X screen fraction image (fractional screens).
    im_y : np.ndarray
        2D ndarray. Y screen fraction image (fractional screens).
    x_sc : np.ndarray
        1D length N ndarray. X axis for output interpolated image.
        (fractionalscreens)
    y_sc : np.ndarray
        1D length M ndarray. Y axis for output interpolated image.
        (fractional screens)

    Returns
    -------
    Vxy
        Length (M * N) image coordinates corresponding to input interpolation axes (pixels).

    """
    # Set up interpolation parameters
    x_px = np.arange(im_x.shape[1]) + 0.5  # image pixels
    y_px = np.arange(im_y.shape[0]) + 0.5  # image pixels

    # Interpolate in X direction for every pixel row of image
    x_px_y_px_x_sc = (
        np.zeros((y_px.size, x_sc.size)) * np.nan
    )  # x pixel data, (y pixel, x screen) size array
    y_px_y_px_x_sc = (
        np.zeros((y_px.size, x_sc.size)) * np.nan
    )  # y pixel data, (y pixel, x screen) size array
    for idx_y in range(y_px.size):
        # Get x slices of x and y position values from images
        x_sc_vals = im_x[idx_y, :]  # x screen fractions
        y_sc_vals = im_y[idx_y, :]  # y screen fractions

        # Define active area of current row
        mask_row = np.logical_not(np.isnan(x_sc_vals))

        # Skip if not enough active pixels
        if mask_row.sum() <= 1:
            continue

        # Get active pixel locations (remove nans)
        x_sc_vals = x_sc_vals[mask_row]  # x screen fractions
        y_sc_vals = y_sc_vals[mask_row]  # y screen fractions
        x_px_vals = x_px[mask_row]  # x pixel locations

        # Smooth to reduce noise
        if x_sc_vals.size > 15:
            med_row = medfilt(x_sc_vals, 11)
            mask_noise = np.abs(x_sc_vals - med_row) < 0.05
        else:
            std_row = x_sc_vals.std()
            mask_noise = np.abs(x_sc_vals - x_sc_vals.mean()) < (3 * std_row)

        # Skip if not enough active pixels
        if mask_noise.sum() <= 2:
            continue

        x_sc_vals = x_sc_vals[mask_noise]
        y_sc_vals = y_sc_vals[mask_noise]
        x_px_vals = x_px_vals[mask_noise]

        # Interpolate x pixel coordinate
        f = interpolate.interp1d(
            x_sc_vals, x_px_vals, bounds_error=False, fill_value=np.nan
        )
        row = f(x_sc)  # x pixel coordinate
        x_px_y_px_x_sc[idx_y, :] = row

        # Interpolate y screen fraction value
        f = interpolate.interp1d(
            x_sc_vals, y_sc_vals, bounds_error=False, fill_value=np.nan
        )
        row = f(x_sc)
        y_px_y_px_x_sc[idx_y, :] = row

    # Interpolate in Y direction for every x-screen sample point column of image
    x_px_y_sc_x_sc = np.zeros(
        (y_sc.size, x_sc.size)
    )  # x pixel data, (y screen, x screen) size array
    y_px_y_sc_x_sc = np.zeros(
        (y_sc.size, x_sc.size)
    )  # y pixel data, (y screen, x screen) size array
    for idx_x in range(x_sc.size):
        # Get active pixel locations
        y_sc_vals = y_px_y_px_x_sc[:, idx_x]
        mask_col = np.logical_not(np.isnan(y_sc_vals))

        # Get interpolation vectors over active range (remove nans)
        y_sc_vals = y_sc_vals[mask_col]
        x_px_vals = x_px_y_px_x_sc[mask_col, idx_x]
        y_px_vals = y_px[mask_col]

        # Smooth to reduce noise
        if y_sc_vals.size > 15:
            med_row = medfilt(y_sc_vals, 11)
            mask_noise = np.abs(y_sc_vals - med_row) < 0.05
        else:
            std_row = y_sc_vals.std()
            mask_noise = np.abs(y_sc_vals - y_sc_vals.mean()) < (3 * std_row)

        y_sc_vals = y_sc_vals[mask_noise]
        x_px_vals = x_px_vals[mask_noise]
        y_px_vals = y_px_vals[mask_noise]

        # Interpolate x pixel coordinate
        f = interpolate.interp1d(
            y_sc_vals, x_px_vals, bounds_error=False, fill_value=np.nan
        )
        col = f(y_sc)
        x_px_y_sc_x_sc[:, idx_x] = col

        # Interpolate y pixel coordinate
        f = interpolate.interp1d(
            y_sc_vals, y_px_vals, bounds_error=False, fill_value=np.nan
        )
        col = f(y_sc)
        y_px_y_sc_x_sc[:, idx_x] = col

    # Return screen points
    if np.any(np.isnan(y_px_y_sc_x_sc)):
        raise ValueError('Nans present in y pixel interpolation array')
    if np.any(np.isnan(x_px_y_sc_x_sc)):
        raise ValueError('Nans present in x pixel interpolation array')

    return Vxy((x_px_y_sc_x_sc.flatten(), y_px_y_sc_x_sc.flatten()))
