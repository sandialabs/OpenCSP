import numpy as np
from opencsp.common.lib.csp.LightSource import LightSource
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ


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
        if child not in self.children:
            raise ValueError(f"{child} is not a child of the Scene.")
        child._self_to_parent_transform = transform

    @property
    def children(self) -> list[OpticOrientationAbstract]:
        return self.objects

    def _add_child_helper(self, new_child: OpticOrientationAbstract):
        if isinstance(new_child, type(self)):
            raise ValueError(f"cannot add a Scene object {new_child} to a scene.")
        self.objects.append(new_child)
