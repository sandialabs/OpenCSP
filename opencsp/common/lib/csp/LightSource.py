from abc import abstractmethod, ABC

from opencsp.common.lib.csp.LightPath import LightPath
from opencsp.common.lib.geometry.Pxyz import Pxyz


class LightSource(ABC):
    """interface for objects that can be light sources"""

    @abstractmethod
    def get_incident_rays(self, point: Pxyz) -> list[LightPath]:
        """Returns the rays originating from this light source incident to the point."""
