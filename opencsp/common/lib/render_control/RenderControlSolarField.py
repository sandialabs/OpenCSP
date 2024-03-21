"""


"""

import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt


class RenderControlSolarField:
    """
    Render control for solar fields.
    """

    def __init__(
        self,
        draw_outline=True,
        outline_style=rcps.outline(),
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.outline()),
        draw_name=False,
        name_style=rctxt.default(color='k'),
    ):
        super(RenderControlSolarField, self).__init__()

        self.draw_outline = draw_outline
        self.outline_style = outline_style
        self.draw_heliostats = draw_heliostats
        self.heliostat_styles = heliostat_styles
        self.draw_name = draw_name
        self.name_style = name_style


# COMMON CASES


def default():
    return RenderControlSolarField()


def outline(color='k'):
    # Overall field outline only.
    return RenderControlSolarField(
        draw_outline=True, outline_style=rcps.outline(color=color), draw_heliostats=False, draw_name=False
    )


def heliostat_blanks(color='k'):
    # Draw nothing.  Heliostats will be added as special rendering categories.
    return RenderControlSolarField(
        draw_outline=False,
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.blank()),
        draw_name=False,
    )


def heliostat_names(color='k'):
    return RenderControlSolarField(
        draw_outline=False,
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.name(color=color)),
        draw_name=False,
    )


def heliostat_centroids(color='k'):
    return RenderControlSolarField(
        draw_outline=False,
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.centroid(color=color)),
        draw_name=False,
    )


def heliostat_centroids_names(color='k'):
    return RenderControlSolarField(
        draw_outline=False,
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.centroid_name(color=color)),
        draw_name=False,
    )


def heliostat_outlines(color='k'):
    return RenderControlSolarField(
        draw_outline=False,
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.outline(color=color)),
        draw_name=False,
    )


def heliostat_normals_outlines(color='k'):
    return RenderControlSolarField(
        draw_outline=False,
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.normal_outline(color=color)),
        draw_name=False,
    )


def heliostat_outlines_names(color='k'):
    return RenderControlSolarField(
        draw_outline=False,
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.name_outline(color=color)),
        draw_name=False,
    )


def heliostat_centroids_outlines_names(
    color='k', horizontalalignment='center', verticalalignment='center'  # center, right, left
):  # center, top, bottom, baseline, center_baseline
    return RenderControlSolarField(
        draw_outline=False,
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(
            rch.centroid_name_outline(
                color=color, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment
            )
        ),
        draw_name=False,
    )


def heliostat_vector_field(color='k', vector_length=9):
    return RenderControlSolarField(
        draw_outline=False,
        outline_style=rcps.outline(color=color),
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(rch.normal(color=color, surface_normal_length=vector_length)),
        draw_name=False,
    )


def heliostat_vector_field_outlines(color='k', vector_length=9):
    return RenderControlSolarField(
        draw_outline=False,
        outline_style=rcps.outline(color=color),
        draw_heliostats=True,
        heliostat_styles=rce.RenderControlEnsemble(
            rch.normal_outline(color=color, surface_normal_length=vector_length)
        ),
        draw_name=False,
    )
