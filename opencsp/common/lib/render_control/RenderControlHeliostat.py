"""


"""

import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlMirror as rcm
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt

# Constants
DEFAULT_SURFACE_NORMAL_LENGTH = 4  # m
DEFAULT_CORNER_NORMAL_LENGTH = 2  # m


class RenderControlHeliostat:
    """
    Render control for heliostats.
    """

    def __init__(
        self,
        draw_centroid=True,
        centroid_style=rcps.marker(),
        draw_outline=True,
        outline_style=rcps.outline(),
        draw_surface_normal=True,
        surface_normal_length=4,
        surface_normal_style=rcps.outline(),
        surface_normal_base_style=rcps.marker(),
        draw_surface_normal_at_corners=True,
        corner_normal_length=2,
        corner_normal_style=rcps.outline(),
        corner_normal_base_style=rcps.marker(),
        draw_facets=False,
        facet_styles=rce.RenderControlEnsemble(rcf.outline()),
        draw_name=False,
        name_style=rctxt.default(color='k'),
    ):
        super(RenderControlHeliostat, self).__init__()

        self.draw_centroid = draw_centroid
        self.centroid_style = centroid_style
        self.draw_outline = draw_outline
        self.outline_style = outline_style
        self.draw_surface_normal = draw_surface_normal
        self.surface_normal_length = surface_normal_length
        self.surface_normal_style = surface_normal_style
        self.surface_normal_base_style = surface_normal_base_style
        self.draw_surface_normal_at_corners = draw_surface_normal_at_corners
        self.corner_normal_length = corner_normal_length
        self.corner_normal_style = corner_normal_style
        self.corner_normal_base_style = corner_normal_base_style
        self.draw_facets = draw_facets
        self.facet_styles = facet_styles
        self.draw_name = draw_name
        self.name_style = name_style

    def style(self, any):
        """ "style" is a method commonly used by RenderControlEnsemble.
        We add this method here so that RenderControlHeliostat can be used similarly to RenderControlEnsemble.
        """
        return self


# COMMON CASES


def default():
    return RenderControlHeliostat()


def blank():
    # Nothing.  (Used for cases where heliostats are added as special cases.)
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def name(color='k', fontsize='medium'):
    # Name only.
    return RenderControlHeliostat(
        draw_centroid=True,  # Draw a tiny point to ensure that things will draw.
        centroid_style=rcps.marker(color=color, marker='.', markersize=0.1),
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(color=color, fontsize=fontsize),
    )


def centroid(color='k'):
    # Centroid only.
    return RenderControlHeliostat(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def centroid_name(color='k'):
    # Centroid and name.
    return RenderControlHeliostat(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(color=color, horizontalalignment='left'),
    )


def centroid_name_outline(
    color='k',
    horizontalalignment='center',  # center, right, left
    verticalalignment='center',
):  # center, top, bottom, baseline, center_baseline
    # Name and overall outline.
    return RenderControlHeliostat(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(
            color=color,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
        ),
    )


def outline(color='k'):
    # Overall outline only.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def name_outline(
    color='k',
    horizontalalignment='center',  # center, right, left
    verticalalignment='center',
):  # center, top, bottom, baseline, center_baseline
    # Name and overall outline.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(
            color=color,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
        ),
    )


def facet_outlines(color='k'):
    # Facet outline only.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline(color=color)),
        draw_name=False,
    )


def facet_outlines_names(color='k'):
    # Facet outlines and facet name labels.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline_name(color=color)),
        draw_name=False,
    )


def facet_outlines(color='k'):
    # Facet outline only.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline(color=color)),
        draw_name=False,
    )


def normal(color='k', surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH):
    # Overall surface normal only.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=True,
        surface_normal_length=surface_normal_length,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def normal_outline(color='k', surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH):
    # Overall surface normal and overall outline.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=True,
        surface_normal_length=surface_normal_length,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=False,
        draw_name=False,
    )


def normal_facet_outlines_names(color='k'):
    # Facet outlines and facet name labels.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=True,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline_name(color=color)),
        draw_name=False,
    )


def mirror_surfaces(color='k'):
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.normal_mirror_surface(color=color)),
        draw_name=False,
    )


def corner_normals_outline(color='k'):
    # Overall outline, and overall surface normal drawn at each overall corner.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=True,
        corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        draw_facets=False,
        draw_name=False,
    )


def normal_facet_outlines(color='k'):
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=True,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline(color=color)),
        draw_name=False,
    )


def normal_length_facet_outlines(color='k'):
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=True,
        surface_normal_length=330,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.outline(color=color)),
        draw_name=False,
    )


def facet_outlines_normals(color='k'):
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.normal_outline(color=color)),
        draw_name=False,
    )


def facet_outlines_corner_normals(color='k'):
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=False,
        draw_surface_normal=False,
        surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        draw_surface_normal_at_corners=False,
        draw_facets=True,
        facet_styles=rce.RenderControlEnsemble(rcf.corner_normals_outline(color=color)),
        draw_name=False,
    )


def highlight(color='b'):
    return RenderControlHeliostat(
        centroid_style=rcps.marker(color=color),
        outline_style=rcps.outline(color=color),
        surface_normal_style=rcps.outline(color=color),
        surface_normal_base_style=rcps.marker(color=color),
        corner_normal_style=rcps.outline(color=color),
        corner_normal_base_style=rcps.marker(color=color),
        draw_facets=False,
        facet_styles=rce.RenderControlEnsemble(rcf.outline()),
        name_style=rctxt.default(color=color),
    )


def low_res_heliostat():
    return RenderControlHeliostat(
        facet_styles=rce.RenderControlEnsemble(
            default_style=rcf.RenderControlFacet(
                draw_outline=False,
                draw_centroid=False,
                draw_name=False,
                draw_surface_normal_at_corners=False,
                draw_mirror_curvature=True,
                surface_normal_length=60,
                mirror_styles=rcm.low_res_mirror(),
            )
        ),
        draw_facets=True,
        draw_centroid=False,
        draw_surface_normal_at_corners=False,
        draw_outline=True,
    )
