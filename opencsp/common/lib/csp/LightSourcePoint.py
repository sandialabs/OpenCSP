from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.csp.LightSource import LightSource


class LightSourcePoint(LightSource):
    def __init__(self, location_in_space: Pxyz) -> None:
        if not isinstance(location_in_space, Vxyz):
            raise TypeError(f"Input location_in_space must be subclass of {Vxyz} but is {type(location_in_space)}")
        self.location_in_space = location_in_space

    def get_incident_rays(self, point: Pxyz) -> list[LightPath]:
        # Check inputs
        if not isinstance(point, Vxyz):
            raise TypeError(f"Input point must be subclass of {Vxyz} but is {type(point)}")

        init_vector = Uxyz.normalize(point - self.location_in_space)
        return [LightPath(self.location_in_space, Vxyz([0, 0, 0]), init_vector)]
