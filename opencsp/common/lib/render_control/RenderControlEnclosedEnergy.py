import opencsp.common.lib.render_control.RenderControlPointSeq as rcps


class RenderControlEnclosedEnergy:
    """
    Controls how an enclosed energy plot will be drawn, as from
    :py:class:`StandardPlotOutput` or :py:class:`EnclosedEnergyImageProcessor`.
    """

    def __init__(
        self, measured: rcps.RenderControlPointSeq = None, theoretical: rcps.RenderControlPointSeq = None
    ) -> None:
        """
        Parameters
        ----------
        measured : rcps.RenderControlPointSeq, optional
            The measured enclosed energy draw style, by default a solid black line.
        theoretical : rcps.RenderControlPointSeq, optional
            The theoretical enclosed energy draw style, by default a dashed black line.
        """
        if measured is None:
            measured = rcps.RenderControlPointSeq(linestyle='-', color='k', marker='None')
        if theoretical is None:
            theoretical = rcps.RenderControlPointSeq(linestyle='--', color='k', marker='None')

        self.measured = measured
        self.theoretical = theoretical


# Common Configurations


def default() -> RenderControlEnclosedEnergy:
    return RenderControlEnclosedEnergy()
