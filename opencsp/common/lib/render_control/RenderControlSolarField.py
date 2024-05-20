"""
Copyright (c) 2021 Sandia National Laboratories.

"""

from warnings import warn
import opencsp.common.lib.render_control.RenderControlEnsemble as rce
import opencsp.common.lib.render_control.RenderControlHeliostat as rch
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt
from opencsp.common.lib.tool.typing_tools import strict_types


class RenderControlSolarField():
    """
    Render control for solar fields.
    """

    def __init__(self,
                 #  draw_outline=True, # unimplemented
                 #  outline_style=rcps.outline(), # umimplemented
                 draw_origin=False,
                 draw_heliostats=True,
                 heliostat_styles=rch.RenderControlHeliostat(),
                 draw_name=False,
                 name_style=rctxt.default(color='k'),
                 special_styles: dict[str, rch.RenderControlHeliostat] = None
                 ):

        super(RenderControlSolarField, self).__init__()

        # self.draw_outline = draw_outline
        # self.outline_style = outline_style
        self.draw_origin = draw_origin
        self.draw_heliostats = draw_heliostats
        self.heliostat_styles = heliostat_styles
        self.draw_name = draw_name
        self.name_style = name_style
        if special_styles is None:
            special_styles = {}
        self.special_styles = special_styles

    # def get_heliostat_style(self, heliostat_name: str):
    #     style = self.heliostat_styles
    #     if heliostat_name in self.special_styles:
    #         style = self.special_styles[heliostat_name]
    #     return style

    # def add_special_names(self, heliostat_name: str, heliostat_style: rch.RenderControlHeliostat):
    #     if heliostat_name is None:
    #         warn("heliostats with name=None should not have special styles in RenderControlheliostatEnsemble.")
    #     self.special_styles[heliostat_name] = heliostat_style

    def get_special_style(self, heliostat_name: str):
        style = self.heliostat_styles
        if heliostat_name in self.special_styles:
            style = self.special_styles[heliostat_name]
        return style

    @strict_types
    def add_special_names(self, heliostat_names: str | list, heliostat_style: rch.RenderControlHeliostat):
        if type(heliostat_names) is list and isinstance(heliostat_names[0], str):
            for name in heliostat_names:
                self.add_special_names(name, heliostat_style)
            return
        if heliostat_names is None:
            warn("Heliostats with name=None should not have special styles in RenderControlSolarField.")
        self.special_styles[heliostat_names] = heliostat_style


# COMMON CASES

def default():
    return RenderControlSolarField()


def outline(color='k'):
    # Overall field outline only.
    return RenderControlSolarField(draw_outline=True,
                                   outline_style=rcps.outline(color=color),
                                   draw_heliostats=False,
                                   draw_name=False)


def heliostat_blanks(color='k'):
    # Draw nothing.  Heliostats will be added as special rendering categories.
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=(rch.blank()),
                                   draw_name=False)


def heliostat_names(color='k'):
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=rch.name(color=color),
                                   draw_name=False)


def heliostat_centroids(color='k'):
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=(rch.centroid(color=color)),
                                   draw_name=False)


def heliostat_centroids_names(color='k'):
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=(rch.centroid_name(color=color)),
                                   draw_name=False)


def heliostat_outlines(color='k'):
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=(rch.outline(color=color)),
                                   draw_name=False)


def heliostat_normals_outlines(color='k'):
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=(rch.normal_outline(color=color)),
                                   draw_name=False)


def heliostat_outlines_names(color='k'):
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=(rch.name_outline(color=color)),
                                   draw_name=False)


def heliostat_centroids_outlines_names(color='k',
                                       horizontalalignment='center',  # center, right, left
                                       verticalalignment='center'):   # center, top, bottom, baseline, center_baseline
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=(rch.centroid_name_outline(color=color,
                                                                               horizontalalignment=horizontalalignment,
                                                                               verticalalignment=verticalalignment)),
                                   draw_name=False)


def heliostat_vector_field(color='k', vector_length=9):
    return RenderControlSolarField(draw_heliostats=True,
                                   heliostat_styles=(rch.normal(color=color,
                                                                normal_vector_length=vector_length)),
                                   draw_name=False)


def heliostat_vector_field_outlines(color='k', vector_length=9):
    return RenderControlSolarField(
        # outline_style=rcps.outline(color=color),
        draw_heliostats=True,
        heliostat_styles=(rch.normal_outline(color=color,
                                             normal_vector_length=vector_length)),
        draw_name=False)
