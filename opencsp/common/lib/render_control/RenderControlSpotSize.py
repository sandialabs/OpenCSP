import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class RenderControlSpotSize:
    """
    Controls how an enclosed energy plot will be drawn, as from
    :py:class:`StandardPlotOutput` or :py:class:`SpotWidthImageProcessor`.
    """

    def __init__(
        self,
        center_style: rcps.RenderControlPointSeq | str = None,
        full_width_style: rcps.RenderControlPointSeq | str = None,
        bounding_box_style: rcps.RenderControlPointSeq | str = 'None',
    ) -> None:
        """
        Parameters
        ----------
        center_style : rcps.RenderControlPointSeq, optional
            The style used to render the center of the area considered for the
            spot size, or 'None' to not render the center point. By default
            (yellow, marker='^').
        full_width_style : rcps.RenderControlPointSeq, optional
            The style used to render the ellipse around the spot, or 'None' to
            not render the ellipse. By default center_style.
        bounding_box_style : rcps.RenderControlPointSeq, optional
            The style used to render the rectangular region around the spot, or
            'None' to not render the bounding box. Can also be specified as
            'center_style' or 'full_width_style'. By default 'None'.
        """
        # set defaults
        if center_style is None:
            center_style = rcps.default(marker='+', color='yellow')
        if full_width_style is None:
            full_width_style = center_style
        if bounding_box_style is None:
            bounding_box_style = 'None'

        # inherit styles
        if bounding_box_style == 'center_style':
            bounding_box_style = center_style
        elif bounding_box_style == 'full_width_style':
            bounding_box_style = full_width_style

        self.center_style = center_style
        self.full_width_style = full_width_style
        self.bounding_box_style = bounding_box_style


# Common Configurations


def default() -> RenderControlSpotSize:
    return RenderControlSpotSize()
