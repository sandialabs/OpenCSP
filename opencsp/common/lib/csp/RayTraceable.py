from abc import abstractmethod

from scipy.spatial.transform import Rotation
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz


class RayTraceable:
    """Abstract class inherited by objects that can be raytraced"""

    @abstractmethod
    def survey_of_points(self, resolution: Resolution) -> tuple[Pxyz, Vxyz]:
        """Returns a set of points sampled from inside the optic region in the optic's
        global coordinate reference frame.

        Parameters
        ----------
        resolution : Resolution
            container of the list of points to survey over. If the Resolution is unresolved
            then it will be resolved in the bounding box of self.

        Returns
        -------
        A tuple of the points (Pxyz) and normals at the respective points (Vxyz) in
        the object's global coordinate reference frame.
        """

    @abstractmethod
    def most_basic_ray_tracable_objects(self) -> list["RayTraceable"]:
        """Return the list of the smallest Ray Traceable that makes up the larger object."""
        pass
