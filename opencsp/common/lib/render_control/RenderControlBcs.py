from opencsp.common.lib.render_control.RenderControlPointSeq import RenderControlPointSeq


class RenderControlBcs(RenderControlPointSeq):
    def __init__(
        self,
        linestyle: str | None = "-",
        linewidth: float = 1,
        color: str = "b",
        marker: str | None = ".",
        markersize: float = 8,
        markeredgecolor: str | None = None,
        markeredgewidth: float | None = None,
        markerfacecolor: str | None = None,
    ):
        """
        Render control for the Beam Characterization System target.

        Controls style of the point marker and circle marker of the BCS.

        Parameters
        ----------
        linestyle : str, optional
            How to draw the line for the circle around the BCS. One of '-', '--', '-.', ':', '' or None (see RenderControlPointSeq for a description). By default '-'
        linewidth : int, optional
            Width of the line for the circle around the BCS. By default 1
        color : str, optional
            Color for the circle around the BCS. One of bgrcmykw (see RenderControlPointSeq for a description). By default 'b'
        marker : str, optional
            Shape of the center BCS marker. One of .,ov^<>12348sp*hH+xXDd|_ or None. By default '.'
        markersize : int, optional
            Size of the center BCS marker. By default 8
        markeredgecolor : str, optional
            Defaults to color above if not set. By default None
        markeredgewidth : float, optional
            Defaults to linewidth if not set. By default None
        markerfacecolor : str, optional
            Defaults to color above if not set. By default None
        """
        super().__init__(
            linestyle=linestyle,
            linewidth=linewidth,
            color=color,
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
            markeredgewidth=markeredgewidth,
            markerfacecolor=markerfacecolor,
        )


# COMMON CASES


def default(marker=".", color="b", linewidth=1, markersize=8) -> RenderControlBcs:
    """
    What to draw if no particular preference is expressed.
    """
    return RenderControlBcs(linewidth=linewidth, color=color, marker=marker, markersize=markersize)


def thin(marker=".", color="b", linewidth=0.3, markersize=5) -> RenderControlBcs:
    return RenderControlBcs(color=color, marker=marker, linewidth=linewidth, markersize=markersize)
