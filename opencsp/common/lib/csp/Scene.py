import numpy as np
from opencsp.common.lib.csp.LightSource import LightSource
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.render.View3d import View3d


class Scene(OpticOrientationAbstract):
    """
    A class representing a scene containing optical elements and light sources.

    This class manages a collection of objects that can be ray-traced and light sources
    that illuminate the scene. It also handles the spatial orientation of these elements.

    Attributes
    ----------
    objects : list[RayTraceable]
        A list of objects in the scene that can be ray-traced.
    light_sources : list[LightSource]
        A list of light sources present in the scene.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.
    def __init__(self) -> None:
        """
        Initializes a Scene object with empty lists for objects and light sources.

        Parameters
        ----------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.objects: list[RayTraceable] = []
        self.light_sources: list[LightSource] = []
        OpticOrientationAbstract.__init__(self)

    def add_object(self, new_object: OpticOrientationAbstract) -> None:
        """
        Adds a new object to the scene.

        Parameters
        ----------
        new_object : OpticOrientationAbstract
            The object to be added to the scene.

        Returns
        -------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.add_child(new_object)

    def add_light_source(self, new_light_source: LightSource) -> None:
        """
        Adds a new light source to the scene.

        Parameters
        ----------
        new_light_source : LightSource
            The light source to be added to the scene.

        Returns
        -------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        self.light_sources.append(new_light_source)

    def set_position_in_space(self, child: OpticOrientationAbstract, transform: TransformXYZ) -> None:
        """
        Sets the spatial position of a child object within the scene.

        Parameters
        ----------
        child : OpticOrientationAbstract
            The child object whose position is to be set.
        transform : TransformXYZ
            The transformation to apply to the child object.

        Raises
        ------
        ValueError
            If the child is not a member of the scene.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if child not in self.children:
            raise ValueError(f"{child} is not a child of the Scene.")
        child._self_to_parent_transform = transform

    # override from OpticOrientationAbstract
    @property
    def children(self) -> list[OpticOrientationAbstract]:
        """
        Retrieves the list of child objects in the scene.

        Returns
        -------
        list[OpticOrientationAbstract]
            A list of child objects in the scene.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return self.objects

    # override from OpticOrientationAbstract
    def _add_child_helper(self, new_child: OpticOrientationAbstract):
        if isinstance(new_child, type(self)):
            raise ValueError(f"cannot add a Scene object {new_child} to a scene.")
        self.objects.append(new_child)

    def draw_objects(self, view: View3d, render_controls: dict = None):
        """
        Draws all objects in the scene using the specified rendering controls.

        Parameters
        ----------
        view : View3d
            The view in which to draw the objects.
        render_controls : dict[type, RenderControl], optional
            A dictionary mapping object types to their corresponding render control settings (default is None).

        Returns
        -------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # TODO this can be improved
        if render_controls is None:
            render_controls = {}

        render_controls.setdefault(None, None)

        for optic in self.objects:
            style = render_controls[type(optic)]
            optic.draw(view, style)
