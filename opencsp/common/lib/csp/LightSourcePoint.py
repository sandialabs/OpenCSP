from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.LightSource import LightSource


class LightSourcePoint(LightSource):
    """
    A class representing a point light source in 3D space.

    This class defines a light source located at a specific point in space,
    characterized by its position and the ability to generate incident rays
    towards a specified point.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self, location_in_space: Pxyz) -> None:
        """
        Initializes a LightSourcePoint with the specified location in space.

        Parameters
        ----------
        location_in_space : Pxyz
            The position of the light source in 3D space.

        Raises
        ------
        TypeError
            If the input location_in_space is not a subclass of Vxyz.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if not isinstance(location_in_space, Vxyz):
            raise TypeError(f"Input location_in_space must be subclass of {Vxyz} but is {type(location_in_space)}")
        self.location_in_space = location_in_space

    def get_incident_rays(self, point: Pxyz) -> list[LightPath]:
        """
        Generates a list of incident rays from the light source to a specified point.

        Parameters
        ----------
        point : Pxyz
            The target point in 3D space to which the incident rays are directed.

        Returns
        -------
        list[LightPath]
            A list of LightPath objects representing the incident rays from the light source to the specified point.

        Raises
        ------
        TypeError
            If the input point is not a subclass of Vxyz.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Check inputs
        if not isinstance(point, Vxyz):
            raise TypeError(f"Input point must be subclass of {Vxyz} but is {type(point)}")

        init_vector = Uxyz.normalize(point - self.location_in_space)
        return [LightPath(self.location_in_space, Vxyz([0, 0, 0]), init_vector)]
