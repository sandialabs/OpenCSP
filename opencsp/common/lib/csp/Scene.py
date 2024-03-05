import numpy as np
from opencsp.common.lib.csp.LightSource import LightSource
from opencsp.common.lib.csp.RayTraceable import RayTraceable


class Scene():
    def __init__(self) -> None:
        self.objects: list[RayTraceable] = []
        self.light_sources: list[LightSource] = []

    def add_object(self, new_object: RayTraceable) -> None:
        self.objects.append(new_object)

    def add_light_source(self, new_light_source: LightSource) -> None:
        self.light_sources.append(new_light_source)
