"""


"""

import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlFacet as rcf
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt


# Constants
DEFAULT_SURFACE_NORMAL_LENGTH = 4 # m
DEFAULT_CORNER_NORMAL_LENGTH  = 2 # m


class RenderControlDeflectometryInstrument():
    """
    Render control for deflectometry instruments.
    """

    def __init__(self, 
                 draw_centroid = True,
                 centroid_style = rcps.marker(),
                 draw_outline = True,
                 outline_style = rcps.outline(),
                 draw_surface_normal = True,
                 surface_normal_length = 4,
                 surface_normal_style = rcps.outline(),
                 surface_normal_base_style = rcps.marker(),
                 draw_surface_normal_at_corners = True,
                 corner_normal_length = 2,
                 corner_normal_style = rcps.outline(),
                 corner_normal_base_style = rcps.marker(),
                 draw_facets=False,
                 facet_styles=rce.RenderControlEnsemble(rcf.outline()),
                 draw_name=False,
                 name_style=rctxt.default(color='k'),
                 ):

        super(RenderControlDeflectometryInstrument, self).__init__()
        
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


# COMMON CASES

def default():
    return RenderControlDeflectometryInstrument()


def blank():
    # Nothing.  (Used for cases where heliostats are added as special cases.)
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = False,
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=False)


def name(color='k', fontsize='medium'):
    # Name only.
    return RenderControlDeflectometryInstrument(draw_centroid = True,  # Draw a tiny point to ensure that things will draw.
                                   centroid_style = rcps.marker(color=color, marker='.', markersize=0.1),
                                   draw_outline = False,
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=True,
                                   name_style=rctxt.RenderControlText(color=color, fontsize=fontsize))


def centroid(color='k'):
    # Centroid only.
    return RenderControlDeflectometryInstrument(draw_centroid = True,
                                   centroid_style = rcps.marker(color=color),
                                   draw_outline = False,
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=False)


def centroid_name(color='k'):
    # Centroid and name.
    return RenderControlDeflectometryInstrument(draw_centroid = True,
                                   centroid_style = rcps.marker(color=color),
                                   draw_outline = False,
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=True,
                                   name_style=rctxt.RenderControlText(color=color, horizontalalignment='left'))


def centroid_name_outline(color='k',
                          horizontalalignment='center', # center, right, left
                          verticalalignment='center'):   # center, top, bottom, baseline, center_baseline
    # Name and overall outline.
    return RenderControlDeflectometryInstrument(draw_centroid = True,
                                   centroid_style = rcps.marker(color=color),
                                   draw_outline = True,
                                   outline_style = rcps.outline(color=color),
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=True,
                                   name_style=rctxt.RenderControlText(color=color,
                                                                      horizontalalignment=horizontalalignment,
                                                                      verticalalignment=verticalalignment))


def outline(color='k'):
    # Overall outline only.
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = True,
                                   outline_style = rcps.outline(color=color),
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=False)


def name_outline(color='k',
                 horizontalalignment='center', # center, right, left
                 verticalalignment='center'):   # center, top, bottom, baseline, center_baseline
    # Name and overall outline.
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = True,
                                   outline_style = rcps.outline(color=color),
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=True,
                                   name_style=rctxt.RenderControlText(color=color,
                                                                      horizontalalignment=horizontalalignment,
                                                                      verticalalignment=verticalalignment))


def facet_outlines(color='k'):
    # Facet outline only.
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = False,
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=True,
                                   facet_styles=rce.RenderControlEnsemble(rcf.outline(color=color)),
                                   draw_name=False)


def facet_outlines_names(color='k'):
    # Facet outlines and facet name labels.
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = False,
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=True,
                                   facet_styles=rce.RenderControlEnsemble(rcf.outline_name(color=color)),
                                   draw_name=False)


def normal(color='k', surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH):
    # Overall surface normal only.
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = False,
                                   draw_surface_normal = True,
                                   surface_normal_length=surface_normal_length,
                                   surface_normal_style = rcps.outline(color=color),
                                   surface_normal_base_style = rcps.marker(color=color),
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=False)


def normal_outline(color='k', surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH):
    # Overall surface normal and overall outline.
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = True,
                                   outline_style = rcps.outline(color=color),
                                   draw_surface_normal = True,
                                   surface_normal_length=surface_normal_length,
                                   surface_normal_style = rcps.outline(color=color),
                                   surface_normal_base_style = rcps.marker(color=color),
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=False,
                                   draw_name=False)


def normal_facet_outlines_names(color='k'):
    # Facet outlines and facet name labels.
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = False,
                                   draw_surface_normal = True,
                                   surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
                                   surface_normal_style = rcps.outline(color=color),
                                   surface_normal_base_style = rcps.marker(color=color),
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=True,
                                   facet_styles=rce.RenderControlEnsemble(rcf.outline_name(color=color)),
                                   draw_name=False)


def corner_normals_outline(color='k'):
    # Overall outline, and overall surface normal drawn at each overall corner.
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = True,
                                   outline_style = rcps.outline(color=color),
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = True,
                                   corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
                                   corner_normal_style = rcps.outline(color=color),
                                   corner_normal_base_style = rcps.marker(color=color),
                                   draw_facets=False,
                                   draw_name=False)


def normal_facet_outlines(color='k'):
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = False,
                                   draw_surface_normal = True,
                                   surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
                                   surface_normal_style = rcps.outline(color=color),
                                   surface_normal_base_style = rcps.marker(color=color),
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=True,
                                   facet_styles=rce.RenderControlEnsemble(rcf.outline(color=color)),
                                   draw_name=False)


def facet_outlines_normals(color='k'):
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = False,
                                   draw_surface_normal = False,
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=True,
                                   facet_styles=rce.RenderControlEnsemble(rcf.normal_outline(color=color)),
                                   draw_name=False)


def facet_outlines_corner_normals(color='k'):
    return RenderControlDeflectometryInstrument(draw_centroid = False,
                                   draw_outline = False,
                                   draw_surface_normal = False,
                                   surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
                                   surface_normal_style = rcps.outline(color=color),
                                   surface_normal_base_style = rcps.marker(color=color),
                                   draw_surface_normal_at_corners = False,
                                   draw_facets=True,
                                   facet_styles=rce.RenderControlEnsemble(rcf.corner_normals_outline(color=color)),
                                   draw_name=False)


def highlight(color='b'):
    return RenderControlDeflectometryInstrument(centroid_style = rcps.marker(color=color),
                               outline_style = rcps.outline(color=color),
                               surface_normal_style = rcps.outline(color=color),
                               surface_normal_base_style = rcps.marker(color=color),
                               corner_normal_style = rcps.outline(color=color),
                               corner_normal_base_style = rcps.marker(color=color),
                               draw_facets=False,
                               facet_styles=rce.RenderControlEnsemble(rcf.outline()),
                               name_style=rctxt.default(color=color))

# name
# centroid
# centroid_name
## outline
## normal_outline
# normal_corners_outline
# facet_outlines
## normal_facet_outlines
# facet_outlines_normals
