import cv2
import dataclasses
import numpy as np

from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractSpotAnalysisImageProcessor import (
    AbstractSpotAnalysisImagesProcessor,
)
import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.log_tools as lt


class FalseColorImageProcessor(AbstractSpotAnalysisImagesProcessor):
    def __init__(self, map_type='human', opencv_map=cv2.COLORMAP_JET):
        """Image processor to produce color gradient images from grayscale
        images, for better contrast and legibility by humans.

        Parameters
        ----------
        map_type : str, optional
            This determines the number of visible colors. Options are 'opencv'
            (256), 'human' (1020), 'large' (1530). Large has the most possible
            colors. Human reduces the number of greens and reds, since those are
            difficult to discern. Default is 'human'.
        opencv_map : opencv map type, optional
            Which color pallete to use with the OpenCV color mapper. Default is
            cv2.COLORMAP_JET.
        """
        super().__init__(self.__class__.__name__)
        self.map_type = map_type
        self.opencv_map = opencv_map

    @staticmethod
    def _map_jet_large_rgb(input_color: int):
        """Like the opencv jet false color map, except that this covers a
        larger color range.

        Parameters
        ----------
        input_color : int
            A grayscale color in the range of 0-1530.

        Returns
        -------
        rgb: int
            An rgb color, where the bits 23:16 are red, bits 15:8 are green, and
            bits 7:0 are blue (assuming bit 31 is the most significant bit).
        """
        if input_color <= 255:  # black to blue
            ret = [0, 0, input_color]
        elif input_color <= 255 * 2:  # blue to cyan
            ret = [0, input_color - 255, 255]
        elif input_color <= 255 * 3:  # cyan to green
            ret = [0, 255, 255 * 3 - input_color]
        elif input_color <= 255 * 4:  # green to yellow
            ret = [input_color - 255 * 3, 255, 0]
        elif input_color <= 255 * 5:  # yellow to red
            ret = [255, 255 * 5 - input_color, 0]
        else:  # red to white
            ret = [255, input_color - 255 * 5, input_color - 255 * 5]
        return (ret[0] << 16) + (ret[1] << 8) + ret[2]

    @staticmethod
    def _map_jet_human_rgb(input_color: int):
        """Like _map_jet_large_rgb, but with more limited red and green values.

        Parameters
        ----------
        input_color : int
            A grayscale color in the range of 0-1020.

        Returns
        -------
        rgb: int
            An rgb color, where the bits 23:16 are red, bits 15:8 are green, and
            bits 7:0 are blue (assuming bit 31 is the most significant bit).
        """
        if input_color <= 255:  # black to blue
            ret = [0, 0, input_color]
        elif input_color <= 255 * 2:  # blue to cyan
            ret = [0, input_color - 255, 255]
        elif input_color <= 255 * 2 + 128:  # cyan to green
            ret = [0, 255, 2 * (255 * 2 + 128 - input_color)]
        elif input_color <= 255 * 3:  # green to yellow
            ret = [2 * (input_color - (255 * 2 + 128)), 255, 0]
        elif input_color <= 255 * 3 + 128:  # yellow to red
            ret = [255, 2 * ((255 * 3 + 128) - input_color), 0]
        else:  # red to white
            ret = [255, 2 * (input_color - (255 * 3 + 128)), 2 * (input_color - (255 * 3 + 128))]
        return (ret[0] << 16) + (ret[1] << 8) + ret[2]

    def apply_mapping_jet_custom(self, operable: SpotAnalysisOperable, map_type: str):
        """Updates the primary image to use the jet color map plus black and
        white (black->blue->cyan->green->yellow->red->white). This larger
        version of the opencv color map can represent either 1020 or 1530
        different grayscale colors (compared to 256 colors with
        opencv.applyColorMap()). However, this takes ~0.28 seconds for a
        1626 x 1236 pixel image."""
        # rescale to the number of representable colors
        # black_to_blue = 255
        # blue_to_cyan = 255
        # cyan_to_green = 128/255
        # green_to_yellow = 127/255
        # yellow_to_red = 128/255
        # red_to_white = 127/255
        representable_colors = 255 * 6 if map_type == 'large' else 255 * 4
        max_value = operable.max_popf
        new_image: np.ndarray = operable.primary_image.nparray * ((representable_colors - 1) / max_value)
        new_image = np.clip(new_image, 0, representable_colors - 1).astype(np.int32)
        if len(new_image.shape) == 3:
            new_image = np.squeeze(new_image, axis=2)

        # add extra color channels
        dims, nchannels = it.dims_and_nchannels(new_image)
        color_image = np.expand_dims(new_image, axis=2)
        color_image = np.broadcast_to(color_image, (dims[0], dims[1], 3))
        color_image = np.array(color_image.astype(np.uint32))
        assert it.dims_and_nchannels(color_image)[1] == 3

        # apply the mapping
        map_func = self._map_jet_large_rgb if map_type == 'large' else self._map_jet_human_rgb
        mapping = {k: map_func(k) for k in range(representable_colors)}
        new_image = np.vectorize(mapping.__getitem__)(new_image)
        color_image[:, :, 0] = new_image >> 16
        color_image[:, :, 1] = (new_image >> 8) & 255
        color_image[:, :, 2] = new_image & 255
        ret = color_image

        # Other methods I've tried:
        # new_image = np.vectorize(self._map_jet_large_rgb)(new_image)
        #     ~1.61 s/image
        # np.apply_along_axis(self._map_jet_large, axis=2, arr=color_image)
        #     ~14.09 s/image
        # color_image[np.where(new_image==k)] = self._map_jet_large(k)
        #     ~9.18 s/image

        return dataclasses.replace(operable, primary_image=ret)

    def apply_mapping_jet(self, operable: SpotAnalysisOperable):
        """Updates the primary image with a false color map. Opencv maps can
        represent 256 different grayscale colors and only takes ~0.007s for a
        1626 x 1236 pixel image."""
        # rescale to the number of representable colors
        representable_colors = 256
        max_value = operable.max_popf
        new_image: np.ndarray = operable.primary_image.nparray * ((representable_colors - 1) / max_value)
        new_image = np.clip(new_image, 0, representable_colors - 1)
        new_image = new_image.astype(np.uint8)

        # apply the mapping
        ret = cv2.applyColorMap(new_image, self.opencv_map)

        return dataclasses.replace(operable, primary_image=ret)

    def _execute(self, operable: SpotAnalysisOperable, is_last: bool) -> list[SpotAnalysisOperable]:
        image = operable.primary_image.nparray

        # verify that this is a grayscale image
        dims, nchannels = it.dims_and_nchannels(image)
        if nchannels > 1:
            lt.error_and_raise(
                ValueError,
                f"Error in {self.name}._execute(): "
                + f"image should be in grayscale, but {nchannels} color channels were found ({image.shape=})!",
            )

        # apply the false color mapping
        if self.map_type == 'large' or self.map_type == 'human':
            ret = [self.apply_mapping_jet_custom(operable, self.map_type)]
        else:
            ret = [self.apply_mapping_jet(operable)]

        return ret
