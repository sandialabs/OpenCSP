from abc import ABC
import numpy as np

import opencsp.common.lib.geometry.Vxy as v2
import opencsp.common.lib.geometry.Vxyz as v3
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg


class AbstractFiducial(ABC):
    """A marker (such as an ArUco board) that is used to orient the camera
    relative to observed objects in the scene. Note that each implementing class
    must also implement a matching locate_instances() method."""

    @property
    def bounding_box(self) -> reg.RegionXY:
        """The X/Y bounding box of this instance, in pixels."""

    @property
    def unit_vector(self) -> v3.Vxyz:
        """Returns a vector representing the origin, orientation, and scale of this instance."""
        pass

    @property
    def origin(self) -> p2.Pxy:
        """The origin point of this instance, in pixels."""

    @property
    def orientation(self) -> v3.Vxyz:
        """The orientation of this instance, in radians. This is relative to
        the source image, where x is positive to the right, y is positive down,
        and z is positive in (away from the camera)."""

    @property
    def size(self) -> float:
        """The scale of this fiducial, in pixels, relative to its longest axis.
        For example, if the fiducial is a square QR-code and is oriented tangent
        to the camera, then the scale will be the number of pixels from one
        corner to the other."""  # TODO is this a good definition?

    @property
    def scale(self) -> float:
        """The scale of this fiducial, in meters, relative to its longest axis.
        This can be used to determine the distance and orientation of the
        fiducial relative to the camera."""

    @classmethod
    def locate_instances(
        self, img: np.ndarray, anticipated_unit_vector: v3.Vxyz = None
    ) -> list["AbstractFiducial"]:
        """For the given input image, find and report any regions that strongly match this fiducial type.

        Parameters:
        -----------
            - img (ndarray): The image to search for fiducials within.
            - anticipated_unit_vector (Vxyz): Where the fiducial is expected to
              be, based on some outsize knowledge. If None, then there isn't
              enough information to make an informed guess. Default None.
        """
