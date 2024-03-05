import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz


class RenderControlLightPath():
    def __init__(self,
                 init_length: float = 1,
                 current_length: float = 1,
                 end_at_plane: tuple[Pxyz, Uxyz] = None,
                 line_render_control: float = rcps.thin(),
                 ) -> None:
        self.init_length = init_length
        self.current_length = current_length
        self.end_at_plane = end_at_plane
        self.line_render_control: rcps.RenderControlPointSeq = line_render_control

# Common Configurations


def default_path() -> RenderControlLightPath:
    return RenderControlLightPath()


def end_at_plane(plane: tuple[Pxyz, Uxyz]):
    return RenderControlLightPath(end_at_plane=plane)
