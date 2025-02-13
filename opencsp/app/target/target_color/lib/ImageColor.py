"""
Tool for matching pixels between two images and viewing results.

"""

import cv2 as cv
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import numpy as np
import rawpy

from opencsp.common.lib.geometry.LoopXY import LoopXY


class ImageColor:
    """
    Class containing image processing and color matching methods on RGB images.

    """

    def __init__(self, image: np.ndarray) -> "ImageColor":
        """
        Provide input 3D image

        Parameters
        ----------
        image : ndarray
            NxMx3 color image to process

        """
        # Save image
        self.image = image.astype(np.float32)

        # Normalize image
        self.image_norm, self.rgbs, self.shape = self._get_normalized_image_data()

    @classmethod
    def from_file(cls, file: str) -> "ImageColor":
        """Creates instance by directly loading image file.

        Parameters
        ----------
        file : str
            File name
        """
        if file.split(".")[-1] in ["NEF", "RAW", "nef", "raw"]:
            # Load image if raw
            with rawpy.imread(file) as raw:
                im_array = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)
        else:
            # Load image if not raw
            im_array = imageio.imread(file)

        return cls(im_array)

    def _get_normalized_image_data(self) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        """
        Returns normalized image, RGB values as Nx3 ndarray, and XY shape of image
        """
        # Normalize image
        image_mag = np.linalg.norm(self.image, axis=2)[..., None]
        image_mag[image_mag == 0] = np.nan
        image_norm = self.image.astype(float) / image_mag

        # Get pixel RGB values
        rgb_vals = image_norm.reshape((-1, 3))

        return image_norm, rgb_vals, image_norm.shape[:2]

    def match_indices(self, rgb: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Calls returns the indices of pixels matching given color in form (ys, xs).

        Parameters
        ----------
        rgb : ndarray
            Length 3 vector to match in current image
        thresh : float
            Threshold, in radians, to consider a match

        Returns
        -------
        ys/xs : ndarray
            Y and X pixel indices that match input RGB vector within given threshold

        """
        return np.where(self.match_mask(rgb, thresh))

    def match_mask(self, rgb: np.ndarray, thresh: float) -> np.ndarray:
        """Returns a mask of all pixels whos RGB vectors are within given
        threshold (in radians) to the given RGB value.

        Parameters
        ----------
        rgb : ndarray
            Length 3 color vector to match in current image
        thresh : float
            Threshold, in radians, to consider a match

        Returns
        -------
        np.ndarray
            Mask of matching pixels.

        Raises
        ------
        ValueError
            If input vector is not size 3
        """
        if rgb.size != 3:
            raise ValueError(f"rgb must be size 3, not shape {rgb.shape}")

        # Normalize input rgb
        rgb = rgb / np.sqrt(np.sum(rgb**2))
        rgb = rgb.squeeze()

        # Calculate angle between input and image RGB vectors
        cross = np.cross(self.rgbs, rgb)
        angles = np.linalg.norm(cross, axis=1)

        # Threshold
        mask = np.abs(angles) <= thresh

        # Return image
        return mask.reshape(self.shape)

    def crop_image(self, region: LoopXY) -> None:
        """
        Masks image according to given region.

        Parameters
        ----------
        region : LoopXY
            Active pixel region

        """
        # Create mask and apply to image
        vx = np.arange(self.image.shape[1])
        vy = np.arange(self.image.shape[0])
        mask = region.as_mask(vx, vy)

        # Find bounding box and crop image
        bbox = region.axis_aligned_bounding_box()
        bbox = list(map(int, bbox))

        # Mask image
        self.image = self.image[bbox[2] : bbox[3], bbox[0] : bbox[1]]
        mask_out = mask[bbox[2] : bbox[3], bbox[0] : bbox[1]]

        # Fill croped areas with NANs
        self.image[np.logical_not(mask_out), :] = np.nan

        # Update normalized image
        self.image_norm, self.rgbs, self.shape = self._get_normalized_image_data()

    def smooth_image(self, ker: np.ndarray) -> np.ndarray:
        """Smoothes image using OpenCV's filter2D.

        Parameters
        ----------
        image : ndarray
            NxMx3 image to smooth
        ker : ndarray
            2d smoothing kernel

        Returns
        -------
        ndarray
            NxMx3 smoothed image

        """
        # Remove nans from image
        mask_nan_pixel = np.isnan(self.image).max(2)
        im_to_smooth = np.nan_to_num(self.image)

        # Smooth image
        self.image = cv.filter2D(im_to_smooth, -1, ker)
        self.image[mask_nan_pixel, :] = np.nan

        # Update normalized image
        self.image_norm, self.rgbs, self.shape = self._get_normalized_image_data()

    def plot_normalized(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """Plots normalized RGB image

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on, by default None. If None, plt.gca() is used.

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        if ax is None:
            ax = plt.gca()

        im_plot = np.nan_to_num(self.image_norm, nan=1, copy=True)
        ax.imshow(im_plot, **kwargs)

        return ax

    def plot_unprocessed(self, ax: plt.Axes = None, **kwargs) -> plt.Axes:
        """Plots unprocessed RGB image

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot on, by default None. If None, plt.gca() is used.

        Returns
        -------
        plt.Axes
            Matplotlib axes
        """
        if ax is None:
            ax = plt.gca()

        # Scale image to max of 1 (float) with 2% of pixels saturated
        im_plot = np.nan_to_num(self.image, copy=True).astype(float)
        im_plot /= float(np.percentile(im_plot, 98))
        im_plot[np.isnan(self.image)] = 1
        im_plot[im_plot > 1] = 1

        ax.imshow(im_plot, **kwargs)

        return ax
