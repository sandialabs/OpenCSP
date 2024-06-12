class RenderControlFunctionXY:
    def __init__(
        self,
        draw_heatmap: bool = True,
        draw_contours: bool = False,
        cmap: str = "jet",
        colorbar: bool = False,
        colorbar_min_max: tuple[float, float] = None,
        bounds: tuple[float, float, float, float] = None,  # unverified
    ) -> None:
        self.draw_heatmap = draw_heatmap
        self.draw_contours = draw_contours
        self.cmap = cmap
        self.colorbar = colorbar
        self.colorbar_min_max = colorbar_min_max
        self.bounds = bounds  # unverified


def countours(**kwargs):
    kwargs["draw_heatmap"] = False
    kwargs["draw_contours"] = True
    return RenderControlFunctionXY(**kwargs)


def heatmap(**kwargs):
    kwargs["draw_heatmap"] = True
    return RenderControlFunctionXY(**kwargs)


def heatmap_and_contours(**kwargs):
    kwargs["draw_contours"] = True
    kwargs["draw_heatmap"] = True
    return RenderControlFunctionXY(**kwargs)
