from abc import abstractmethod

from scipy.spatial.transform import Rotation
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz


class RayTraceable:
    """Abstract class inherited by objects that can be raytraced"""
    @abstractmethod
    def survey_of_points(self, resolution: int, resolution_type: str = 'pixelX', random_seed: int | None = None) -> tuple[Pxyz, Vxyz]:
        """Returns a set of points sampled from inside the optic region in the optic's
        parent coordinate reference frame.

        Parameters
        ----------
        resolution : int
            Number of sample points to generate. Depends on input resolution_type.
            If, 'pixelX' or 'pixelY', the number of x sample points on the sample
            grid. The y sample points are chosen to have the same spacing. For
            'random', the total number of random points to generate.
        resolution_type : str, optional
            {'random', 'pixelY', 'pixelX'}, by default 'pixelX'
        random_seed : int | None, optional
            Random seed for random sample point generation. If None, no seed is
            used. If resolution_type is not "random", this is ignored, by default None.

        Returns
        -------
        A tuple of the points (Pxyz) and normals at the respective points (Vxyz) in
        the object's parent coordinate reference frame.
        """

    @abstractmethod
    def set_position_in_space(self, translation: Pxyz | Vxyz, rotation: Rotation) -> None:
        """Sets the optic's base coordinate reference frame location
        relative to the parent reference frame. When combined into a 3d
        transformation, this converts base coordinates into parent coordinates.

        Parameters
        ----------
        translation : Pxyz
            Translation from parent to base coordinate reference frame
        rotation : Rotation
            Rotation from parent to base coordinate reference frame
        """

    @abstractmethod
    def most_basic_ray_tracable_objects(self) -> list['RayTraceable']:
        """Return the list of the smallest Ray Traceable that makes up the larger object.
        """
        pass
