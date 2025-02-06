"""

"""

import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
import opencsp.common.lib.render_control.RenderControlMirror as rcm


# Constants
DEFAULT_SURFACE_NORMAL_LENGTH = 2  # m
DEFAULT_CORNER_NORMAL_LENGTH = 1  # m


class RenderControlFacet:
    """
    Render control for heliostat facets.
    """

    def __init__(
        self,
        draw_centroid=False,
        centroid_style=rcps.marker(),
        draw_outline=True,
        outline_style=rcps.outline(),
        draw_surface_normal=False,
        surface_normal_length=4,
        surface_normal_style=rcps.outline(),
        surface_normal_base_style=rcps.marker(),
        # draw_surface_normal_at_corners=False, # Unimplemented
        # corner_normal_length=2,               # Unimplemented
        # corner_normal_style=rcps.outline(),   # Unimplemented
        # corner_normal_base_style=rcps.marker(), # Unimplemented
        draw_name=False,
        name_style=rctxt.default(color='k'),
        draw_mirror_curvature=False,
        mirror_styles=rcm.RenderControlMirror(),
    ):

        super(RenderControlFacet, self).__init__()

        self.draw_centroid = draw_centroid
        self.centroid_style = centroid_style
        self.draw_outline = draw_outline
        self.outline_style = outline_style
        self.draw_surface_normal = draw_surface_normal
        self.surface_normal_length = surface_normal_length
        self.surface_normal_style = surface_normal_style
        self.surface_normal_base_style = surface_normal_base_style
        # self.draw_surface_normal_at_corners = draw_surface_normal_at_corners
        # self.corner_normal_length = corner_normal_length
        # self.corner_normal_style = corner_normal_style
        # self.corner_normal_base_style = corner_normal_base_style
        self.draw_name = draw_name
        self.name_style = name_style
        self.draw_mirror_curvature = draw_mirror_curvature
        self.mirror_styles = mirror_styles

    def style(self, imput_name: str):
        return self


# COMMON CASES


def default():
    return RenderControlFacet()


def outline(color='k'):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_name=False,
    )


def outline_thin(color='k', linewidth=0.5):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color, linewidth=linewidth),
        draw_surface_normal=False,
        draw_name=False,
    )


def outline_name(color='k'):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_name=True,
        name_style=rctxt.default(color=color),
    )


def normal_mirror_surface(color='k'):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=False,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_name=False,
        name_style=rctxt.default(color=color),
        draw_mirror_curvature=True,
        mirror_styles=rcm.normal_mirror(),
    )


def normal_outline(color='k'):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=True,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_name=False,
    )


def corner_normals_outline_NOTWORKING(color='k'):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=True,
        corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        draw_name=False,
    )


def corner_normals_outline_name_NOTWORKING(color='k'):
    return RenderControlFacet(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=True,
        corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        draw_name=True,
        name_style=rctxt.default(color=color),
    )


def highlight_NOTWORKING(color='b'):
    return RenderControlFacet(
        centroid_style=rcps.marker(color=color),
        outline_style=rcps.outline(color=color),
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        name_style=rctxt.default(color=color),
    )
