"""


"""

import opencsp.common.lib.render_control.RenderControlFacetEnsemble as rcfe
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

    # TODO: decide if some of these should be in heliostat and ensemble or just ensemble
    def __init__(
        self,
        draw_centroid=False,
        centroid_style=rcps.marker(),
        #  draw_outline=False,
        #  outline_style=rcps.outline(),
        #  draw_surface_normal=True, # unimplmeneted
        #  surface_normal_length=4, # unimplmeneted
        #  surface_normal_style=rcps.outline(),  # unimplmeneted
        #  surface_normal_base_style=rcps.marker(),  # unimplmeneted
        #  draw_surface_normal_at_corners=False,  # unimplmeneted
        #  corner_normal_length=2,  # unimplmeneted
        #  corner_normal_style=rcps.outline(),  # unimplmeneted
        #  corner_normal_base_style=rcps.marker(),  # unimplmeneted
        draw_facet_ensemble=True,
        facet_ensemble_style=rcfe.RenderControlFacetEnsemble(rcf.outline()),
        draw_name=False,  # unimplmeneted
        name_style=rctxt.default(color='k'),  # unimplmeneted
        post=0,  # by default there is no post
    ):

        super(RenderControlHeliostat, self).__init__()

        self.draw_centroid = draw_centroid
        self.centroid_style = centroid_style
        # self.draw_outline = draw_outline
        # self.outline_style = outline_style
        # self.draw_surface_normal = draw_surface_normal
        # self.surface_normal_length = surface_normal_length
        # self.surface_normal_style = surface_normal_style
        # self.surface_normal_base_style = surface_normal_base_style
        # self.draw_surface_normal_at_corners = draw_surface_normal_at_corners
        # self.corner_normal_length = corner_normal_length
        # self.corner_normal_style = corner_normal_style
        # self.corner_normal_base_style = corner_normal_base_style

        self.draw_facet_ensemble = draw_facet_ensemble
        self.facet_ensemble_style = facet_ensemble_style
        self.draw_name = draw_name
        self.name_style = name_style
        self.post = post


# SPECIAL CASES


# COMMON CASES


def default():
    return RenderControlHeliostat()


def blank():
    # Nothing.  (Used for cases where heliostats are added as special cases.)
    return RenderControlHeliostat(draw_centroid=False, draw_facet_ensemble=False)


def name(color='k', fontsize='medium'):
    # Name only.
    return RenderControlHeliostat(
        draw_centroid=True,  # Draw a tiny point to ensure that things will draw.
        centroid_style=rcps.marker(color=color, marker='.', markersize=0.1),
        draw_facet_ensemble=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(color=color, fontsize=fontsize),
    )


def centroid(color='k'):
    # Centroid only.
    return RenderControlHeliostat(
        draw_centroid=True, centroid_style=rcps.marker(color=color), draw_facet_ensemble=False
    )


def centroid_name(color='k'):
    # Centroid and name.
    return RenderControlHeliostat(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_facet_ensemble=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(color=color, horizontalalignment='left'),
    )


def centroid_name_outline(
    color='k', horizontalalignment='center', verticalalignment='center'  # center, right, left
):  # center, top, bottom, baseline, center_baseline
    # Name and overall outline.
    return RenderControlHeliostat(
        draw_centroid=True,
        centroid_style=rcps.marker(color=color),
        draw_facet_ensemble=True,
        facet_ensemble_style=rcfe.only_outline(color=color),
        draw_name=True,
        name_style=rctxt.RenderControlText(
            color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
        ),
    )


def outline(color='k'):
    # Overall outline only.
    return RenderControlHeliostat(
        draw_centroid=False, draw_facet_ensemble=True, facet_ensemble_style=rcfe.only_outline(color=color)
    )


def name_outline(
    color='k', horizontalalignment='center', verticalalignment='center'  # center, right, left
):  # center, top, bottom, baseline, center_baseline
    # Name and overall outline.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_outline=True,
        outline_style=rcps.outline(color=color),
        draw_surface_normal=False,
        draw_surface_normal_at_corners=False,
        draw_facet_ensemble=False,
        draw_name=True,
        name_style=rctxt.RenderControlText(
            color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
        ),
    )


def facet_outlines_names(color='k'):
    # Facet outlines and facet name labels.
    return RenderControlHeliostat(
        draw_centroid=False,
        draw_facet_ensemble=True,
        facet_ensemble_style=rcfe.RenderControlFacetEnsemble(rcf.outline_name(color=color)),
        draw_name=False,
    )


def normal(color='k', normal_vector_length=DEFAULT_SURFACE_NORMAL_LENGTH):
    # Overall surface normal only.
    return RenderControlHeliostat(
        draw_centroid=True,
        #   draw_normal_vector=True,
        #   normal_vector_length=normal_vector_length,
        #   normal_vector_style=rcps.outline(color=color),
        #   normal_vector_base_style=rcps.marker(color=color),
        draw_facet_ensemble=True,
        facet_ensemble_style=rcfe.normal_only(color=color, normal_vector_length=normal_vector_length),
    )


def normal_outline(color='k', normal_vector_length=DEFAULT_SURFACE_NORMAL_LENGTH, **kwargs):
    # Overall surface normal and overall outline.
    fe_style = rcfe.facet_ensemble_outline(color=color, normal_vector_length=normal_vector_length)
    return RenderControlHeliostat(
        draw_centroid=False, facet_ensemble_style=fe_style, draw_facet_ensemble=True, draw_name=False, **kwargs
    )


def normal_facet_outlines_names(color='k'):
    # Facet outlines and facet name labels.
    fe_style = rcfe.RenderControlFacetEnsemble(
        draw_normal_vector=True,
        normal_vector_length=DEFAULT_SURFACE_NORMAL_LENGTH,
        normal_vector_style=rcps.outline(color=color),
        normal_vector_base_style=rcps.marker(color=color),
        default_style=rcf.outline_name(color=color),
    )
    return RenderControlHeliostat(
        draw_centroid=False, draw_facet_ensemble=True, facet_ensemble_style=fe_style, draw_name=False
    )


def mirror_surfaces(color='k', **kwargs):
    fe_style = rcfe.RenderControlFacetEnsemble(rcf.normal_mirror_surface(color=color))
    return RenderControlHeliostat(
        draw_centroid=False, draw_facet_ensemble=True, facet_ensemble_style=fe_style, **kwargs
    )


# TODO implement the normals at corners
# def corner_normals_outline(color='k'):
#     # Overall outline, and overall surface normal drawn at each overall corner.
#     return RenderControlHeliostat(draw_centroid=False,
#                                 #   draw_outline=True,
#                                 #   outline_style=rcps.outline(color=color),
#                                 #   draw_surface_normal_at_corners=True # unimplemented
#                                   corner_normal_length=DEFAULT_CORNER_NORMAL_LENGTH,
#                                   corner_normal_style=rcps.outline(color=color),
#                                   corner_normal_base_style=rcps.marker(color=color),
#                                   draw_facet_ensemble=True,
#                                   draw_name=False)


def normal_facet_outlines(color='k'):
    return RenderControlHeliostat(facet_ensemble_style=rcfe.normal_facet_outlines(color=color))


def facet_outlines(color='k', **kwargs):
    # Facet outline only.
    return RenderControlHeliostat(facet_ensemble_style=rcfe.facet_outlines(color=color))


def facet_outlines_normals(color='k'):
    fe_style = rcfe.RenderControlFacetEnsemble(default_style=rcf.normal_outline(color=color))
    return RenderControlHeliostat(draw_facet_ensemble=True, facet_ensemble_style=fe_style)


# TODO implement the normals at corners
# def facet_outlines_corner_normals(color='k'):
#     return RenderControlHeliostat(draw_centroid=False,
#                                   draw_outline=False,
#                                   draw_surface_normal=False,
#                                   surface_normal_length=DEFAULT_SURFACE_NORMAL_LENGTH,
#                                   surface_normal_style=rcps.outline(color=color),
#                                   surface_normal_base_style=rcps.marker(color=color),
#                                   draw_surface_normal_at_corners=False,
#                                   draw_facet_ensemble=True,
#                                   facet_ensemble_style=rcfe.RenderControlFacetEnsemble(rcf.corner_normals_outline(color=color)),
#                                   draw_name=False)


# TODO this is not working, check if it should be fixed
# def highlight(color='b'):
#     return RenderControlHeliostat(centroid_style=rcps.marker(color=color),
#                                   outline_style=rcps.outline(color=color),
#                                   surface_normal_style=rcps.outline(color=color),
#                                   surface_normal_base_style=rcps.marker(color=color),
#                                   corner_normal_style=rcps.outline(color=color),
#                                   corner_normal_base_style=rcps.marker(color=color),
#                                   draw_facet_ensemble=False,
#                                   facet_ensemble_style=rcfe.RenderControlFacetEnsemble(rcf.outline()),
#                                   name_style=rctxt.default(color=color))


def low_res_heliostat():
    return RenderControlHeliostat(
        facet_ensemble_style=rcfe.RenderControlFacetEnsemble(
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
        draw_facet_ensemble=True,
        draw_centroid=False,
    )
