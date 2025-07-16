from hashlib import new

import numpy as np

import opencsp.common.lib.render.lib.Drawable as dw
import opencsp.common.lib.render_control.RenderControlLightPath as rclp
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d


class LightPath(dw.Drawable):
    """
    The LightPath will represent the path of a photon in a light beam.
    There are two ways to think about the photon for this class.

    1. The photon originates at an unknown point, and only has an original known direction
    (init_direction [OldVxyz]) and then has bounce off a point(s)
    and now is continuing in the direction of current_direction.

    2. The photon originated at the first point in the list of points it has passed through.
    In this case the init_direction sould be [0, 0, 0].

    In either case, to represent a photon that no longer exists
    (i.e. hits a wall and did not reflect) simply set the current_dirrection to [0, 0, 0].
    """

    def __init__(
        self,
        points_list: Pxyz,
        init_direction: Uxyz,
        current_direction: Uxyz = None,
        color: tuple[float, float, float] = None,
        intensity: list[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        points_list -- list of points that the light has passed through/reflected at (list[OldPxyz])

        init_direction -- the initial direction the light was traveling when we started tracking, if we assume the light started at
            the first point in the points_list then this should be np.array([0, 0, 0]) (OldUxyz)

        current_direction -- the current direction the light is traveling. Will be automatically set to init_direction if the
            no value is given. (OldUxyz)

        color -- 3-tuple for RGB color (tuple[float, float, float])

        intensity -- the intesity at each part of the light's journey. if the light started going in the
            direction v1, passed through [p1, p2, p3], and is currently going in the direction v2, then the
            intensity would be [i@v1, i_after_p1, i_after_p2 , i_after_p3 or i@v2]. (list[float])
        """
        if len(init_direction) != 1:
            raise ValueError(f"Initial direction argument should be a single vector. Given value was {init_direction}.")
        self.points_list = points_list
        # TODO: assert -1e-6 < np.linalg.norm(init_direction) - 1 < 1e-6
        self.init_direction = init_direction
        self.current_direction = init_direction if current_direction is None else current_direction
        # TODO: assert 1e-6 < np.linalg.norm(self.current_direction) - 1 < 1e-6
        self.color = color
        self.intensity = intensity

    def __repr__(self) -> str:
        return f"{self.init_direction} --> \n{self.points_list} --> \n{self.current_direction}"

    def __str__(self) -> str:  # TODO TJL:make a more useful string representation
        return f"{self.init_direction} --> \n{self.points_list} --> \n{self.current_direction}"

    def many_rays_from_many_vectors(
        many_points_lists: list[Pxyz], many_init_directions: Vxyz, many_current_directions: Vxyz = []
    ) -> list["LightPath"]:
        """
        Creates a list of LightPaths from vectors
        If the many_points_lists is None then the function will infer that they are
        all just the current vectors and have no history.
        """
        if many_points_lists == None:  # None implies there are no recorded points at all
            many_points_lists = [Pxyz.empty()] * len(many_init_directions)
        elif len(many_points_lists) > 0 and len(many_points_lists) != len(many_init_directions):
            raise ValueError(f"The number of points lists and initial vectors must be the same.")

        diff_vectors = len(many_init_directions) - len(many_current_directions)
        many_current_directions = (
            many_current_directions + [None] * diff_vectors
        )  # pads the end of diff vectors with Nones so the LightPath constructor can deal with it
        res = [
            LightPath(points_list, init_direction.normalize(), current_direction)
            for points_list, init_direction, current_direction in zip(
                many_points_lists, many_init_directions, many_current_directions
            )
        ]
        # print(f"MANY INPUT : {many_points_lists}")
        # print(f"MANY RESULT : {res}")
        return res

    def draw(self, view: View3d, path_style: rclp.RenderControlLightPath = rclp.default_path()) -> None:
        # print("drawing ray") #TODO TJL:print for debug
        # print(f"Points: \n{self.points_list}") # TODO TJL:debug print
        points_array = list(self.points_list.data.T)
        init_direction_array = self.init_direction.data.T[0]
        current_direction_array = self.current_direction.data.T[0]

        if path_style.end_at_plane == None:
            view.draw_xyz_list(
                [points_array[0] - init_direction_array * path_style.init_length]  # initial direction
                + points_array  # each point passed through
                + [points_array[-1] + current_direction_array * path_style.current_length],  # current direction
                style=path_style.line_render_control,
            )

        # else
        if path_style.end_at_plane != None:
            plane_point, plane_normal_vector = path_style.end_at_plane

            plane_normal_vector = plane_normal_vector.as_Vxyz()  # cannot have the directions be Uxyz objects
            current_direction = self.current_direction.as_Vxyz()

            d: float = Vxyz.dot(plane_normal_vector, current_direction)
            # filter out rays that miss the target and NAN values from the raytrace
            if np.abs(d) < 10e-6:
                return  # draw nothing if it does not intersect

            p0 = self.points_list[-1]  # most recent point in light path
            w = p0 - plane_point
            fac: float = (-Vxyz.dot(plane_normal_vector, w) / d)[0]  # test without -
            v = current_direction * fac
            intersection = p0 + v

            curr_length = (Pxyz(points_array[-1]) - intersection).magnitude()

            view.draw_xyz_list(
                [points_array[0] - init_direction_array * path_style.init_length]  # initial direction
                + points_array  # each point passed through
                + [points_array[-1] + current_direction_array * curr_length],  # current direction
                style=path_style.line_render_control,
            )

    def add_step(self, point: Pxyz, new_direction: Uxyz, new_intensity: float = None):
        """
        Add a step (aka a reflection) to this light path.

        Parameters
        ----------
        point : Pxyz
            Where the reflection takes place.
        new_direction : Uxyz
            The direction that the bounced light is traveling in.
        new_intensity : float, optional
            Reserved

        Raises
        ------
        ValueError
            If the given point or direction is not singular.
        """
        if not (issubclass(type(point), Vxyz) and issubclass(new_direction, Vxyz)):
            raise TypeError(
                f"LightPath.add_step expects two parameters of the Vxyz class family\n \
                            Parameter 1 is of type {type(point)} and parameter 2 is of type {type(new_direction)}."
            )
        if len(point) != 1:
            raise ValueError(
                f"parameter is not a single point. LightPath.add_step can only add one step at a time. \
                             \nThe Pxyz given in parameter 1 was of length {len(point)}."
            )
        if len(new_direction) != 1:
            raise ValueError(
                f"parameter is not a single new direction. LightPath.add_step can only add one step at a time. \
                             \nThe Vxyz given in paremeter 2 was of length {len(new_direction)}."
            )
        self.points_list = self.points_list.concatenate(point)
        self.current_direction = new_direction
        self.points_list = self.points_list.concatenate(point)
        self.current_direction = new_direction
