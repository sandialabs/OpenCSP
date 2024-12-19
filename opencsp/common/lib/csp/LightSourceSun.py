import datetime
from warnings import warn

import numpy as np
import pysolar
from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp import sun_position as sun_pos
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.LightSource import LightSource


class LightSourceSun(LightSource):
    """
    A class representing a point light source that simulates sunlight.

    This class models the sun as a top-hat function in space, allowing for the generation
    of incident rays based on the sun's position in the sky.

    Attributes
    ----------
    incident_rays : list[LightPath]
        A list of LightPath objects representing the rays incident from the sun.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self) -> None:
        """
        Initializes a LightSourceSun object with an empty list of incident rays.

        Parameters
        ----------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.incident_rays: list[LightPath] = []

    def get_incident_rays(self, point: Pxyz) -> list[LightPath]:
        """
        Retrieves the incident rays from the light source.

        Parameters
        ----------
        point : Pxyz
            The point in space for which the incident rays are requested.

        Returns
        -------
        list[LightPath]
            A list of LightPath objects representing the incident rays.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return self.incident_rays

    @classmethod
    def from_given_sun_position(
        cls, sun_pointing: Uxyz, resolution: int, sun_dia: float = 0.009308, verbose=False
    ) -> 'LightSourceSun':
        """
        Creates a LightSourceSun object initialized from a given sun pointing direction.

        Represents the sun as a top-hat function in space.

        Parameters
        ----------
        sun_pointing : Uxyz
            The pointing direction of the sun.
        resolution : int
            The number of points in each direction that will be sampled.
        sun_dia : float, optional
            The angular diameter of the sun in radians (default is 0.009308).
        verbose : bool, optional
            If True, prints updates during initialization (default is False).

        Returns
        -------
        LightSourceSun
            A LightSourceSun object initialized with the specified sun pointing direction.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Calculate sun ray cone pointing down (z=-1)
        sun_rays = cls._calc_sun_ray_cone(resolution, sun_dia, verbose)

        # Rotate the cone of sun rays
        center_pointing = Vxyz([0, 0, -1])
        rotation_from_sun_position = center_pointing.align_to(sun_pointing)
        sun_rays = sun_rays.rotate(rotation_from_sun_position)
        if verbose:
            print("Sun rays are initialized")

        # Create object
        obj = cls()
        obj.incident_rays = LightPath.many_rays_from_many_vectors(None, sun_rays)
        return obj

    @classmethod
    def from_location_time(
        cls,
        loc: tuple[float, float],
        time: datetime.datetime,
        resolution: int,
        sun_dia: float = 0.009308,
        verbose=False,
    ) -> 'LightSourceSun':
        """
        Creates a LightSourceSun object initialized from a given latitude/longitude and time.

        Represents the sun as a top-hat function in space.

        Parameters
        ----------
        loc : tuple[float, float]
            The location of the scene in the form (latitude, longitude) in degrees, WGS84.
        time : datetime.datetime
            A datetime object representing the time. Must have timezone set.
        resolution : int
            The number of points in each direction that will be sampled.
        sun_dia : float, optional
            The angular diameter of the sun in radians (default is 0.009308).
        verbose : bool, optional
            If True, prints updates during initialization (default is False).

        Returns
        -------
        LightSourceSun
            A LightSourceSun object initialized based on the specified location and time.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Calculate direction of sun pointing
        longitude, latitude = loc
        alt = pysolar.solar.get_altitude(latitude, longitude, time)
        azm = pysolar.solar.get_azimuth(latitude, longitude, time)
        sun_pointing = -Vxyz((0, 1, 0)).rotate(Rotation.from_euler('xz', [alt, -azm], degrees=True))

        # Calculate sun ray cone pointing down (z=-1)
        sun_rays = cls._calc_sun_ray_cone(resolution, sun_dia, verbose)

        # Rotate the cone of sun rays
        center_pointing = Vxyz([0, 0, -1])
        rotation_from_sun_position = center_pointing.align_to(sun_pointing)
        sun_rays = sun_rays.rotate(rotation_from_sun_position)
        if verbose:
            print("Sun rays are initialized")

        # Create object
        obj = cls()
        obj.incident_rays = LightPath.many_rays_from_many_vectors(None, sun_rays)
        return obj

    @staticmethod
    def _calc_sun_ray_cone(resolution: int, sun_dia: float, verbose: bool = False) -> Vxyz:
        # Calculate sun radius
        sun_radius = sun_dia / 2

        # Handle special cases
        if resolution >= 3:
            xs = ys = np.linspace(-sun_radius, sun_radius, resolution)
        elif resolution == 2:
            xs = ys = np.array([-sun_radius / 3, sun_radius / 3])
        elif resolution == 1:
            xs = ys = np.zeros(1)
        else:
            raise ValueError("Illegal Resolution. Resolution must be at least 1.")

        # Create sun rays
        sun_rays = Vxyz.empty()
        for i, x in enumerate(xs):
            for y in ys:
                # Only keep points in the circle defined by sun_radius
                if np.sqrt(x**2 + y**2) > sun_radius:
                    continue
                # Calculate rotation of current ray from z=-1
                x_pt = np.sin(x)
                y_pt = np.sin(y)
                z = np.sqrt(1 - x_pt**2 + y_pt**2)
                ray_cur_pointing = Vxyz((x, y, -z))
                sun_rays = sun_rays.concatenate(ray_cur_pointing)
            if verbose and (i % 100 == 0):
                print(f"{i / resolution * 100}% sun rays initalized")
        if verbose:
            print(r'100% sun rays initialized')

        return sun_rays

    def set_incident_rays(
        self, loc: tuple[float, float], time: tuple, resolution: int, sun_dia: float = 0.009308, verbose=False
    ) -> None:
        """
        Defines the rays that will be used from this light source for ray tracing.

        Sets them to self.incident_rays.

        Parameters
        ----------
        loc : tuple[float, float]
            A tuple representing the location of the scene that will see the sun rays in the form (longitude, latitude).
        time : tuple
            A tuple in the ymdhmsz convention, representing (year, month, day, hour, minute, second, time zone).
        resolution : int
            The number of points in each direction that will be sampled.
        sun_dia : float, optional
            The angular diameter of the sun in radians (default is 0.009308).
        verbose : bool, optional
            If True, prints updates on how many rays have been generated to the console (default is False).

        Raises
        ------
        DeprecationWarning
            If this method is called, indicating it is deprecated.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Function is deprecated
        warn(
            'LightSourceSun.set_incident_rays is deprecated. Use initialize_from_solar_position instead.',
            DeprecationWarning,
            stacklevel=2,
        )

        real_center_vector = -sun_pos.sun_position(loc, time)
        # self.incident_rays = [LightPath([], real_center_vector)]
        # return
        center = Vxyz([0, 0, -1])
        sun_radius = sun_dia / 2

        if resolution >= 3:
            # defines a square of points, corners cut off later
            xs = ys = np.linspace(-sun_radius, sun_radius, resolution)
        elif resolution == 2:
            xs = ys = np.array([-sun_radius / 3, sun_radius / 3])
        elif resolution == 1:
            xs = ys = np.zeros(1)
        else:
            raise ValueError("Illegal Resolution. Resolution must be at least 1.")

        sun_rays = Vxyz.empty()
        for i, x in enumerate(xs):
            x_rotation = Rotation.from_euler('x', x)
            for y in ys:
                if np.sqrt(x**2 + y**2) > sun_radius:
                    continue  # only runs on points in the circle defined by sun_radius
                y_rotation = Rotation.from_euler('y', y)
                full_rot = x_rotation * y_rotation
                sun_rays = sun_rays.concatenate(center.rotate(full_rot))
            if verbose:
                print(f"{i}/{resolution} of the sun rays have been initalized")

        # rotate the cone of sun rays
        print("Rotating sun rays...")
        angle: float = np.arccos(np.dot(np.array([0, 0, -1]), real_center_vector))
        cross_prod = np.cross(real_center_vector, np.array([0, 0, 1]))  # angle to rotate the rays
        axis_of_rotation = cross_prod / np.linalg.norm(cross_prod)  # axis to rotate around
        rotation_from_sun_position = Rotation.from_rotvec(angle * axis_of_rotation)
        # print(f"rot from sun: {rotation_from_sun_position}")
        # print(f"SUN RAYS: {sun_rays}") # TODO
        rotated_sun_rays = sun_rays.rotate(rotation_from_sun_position)
        self.incident_rays = LightPath.many_rays_from_many_vectors(None, rotated_sun_rays)
        print("Sun rays are initialized\n")
