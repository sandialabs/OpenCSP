import numpy as np
from opencsp.common.lib.csp.LightSource import LightSource
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.render.View3d import View3d


class Scene(OpticOrientationAbstract):
    def __init__(self) -> None:
        self.objects: list[RayTraceable] = []
        self.light_sources: list[LightSource] = []
        OpticOrientationAbstract.__init__(self)

    def add_object(self, new_object: OpticOrientationAbstract) -> None:
        self.add_child(new_object)

    def add_light_source(self, new_light_source: LightSource) -> None:
        self.light_sources.append(new_light_source)

    def set_position_in_space(self, child: OpticOrientationAbstract,
                              transform: TransformXYZ) -> None:
        """
        Allows for the spacial placement of children inside the scene.
        """
        if child not in self.children:
            raise ValueError(f"{child} is not a child of the Scene.")
        child._self_to_parent_transform = transform

    # override from OpticOrientationAbstract
    @property
    def children(self) -> list[OpticOrientationAbstract]:
        return self.objects

    # override from OpticOrientationAbstract
    def _add_child_helper(self, new_child: OpticOrientationAbstract):
        if isinstance(new_child, type(self)):
            raise ValueError(f"cannot add a Scene object {new_child} to a scene.")
        self.objects.append(new_child)

    def draw_objects(self, view: View3d, render_controls: dict = None):
        """
        Will draw every `OpticOrientationAbstract` object in the scene. 
        It determines the render control by taking the type of the object and 
        looking for that type in the dictionary of render controls.

        Parameters
        ----------
        view: View3d
            - the view to draw the objects inside
        
        render_controls: dict[type, RenderControl]
            - The render control objects that correspond to specific types 
            - TODO this can be improved
        """
        if render_controls is None:
            render_controls = {}

        render_controls.setdefault(None, None)

        for optic in self.objects:
            style = render_controls[type(optic)]
            optic.draw(view, style)