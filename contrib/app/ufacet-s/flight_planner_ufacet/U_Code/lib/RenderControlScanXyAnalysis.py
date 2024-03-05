"""


"""


class RenderControlScanXyAnalysis:
    """
    Render control for plot axes.
    """

    def __init__(self, draw_xy_segment_analysis=True, draw_xy_segment_result=True):
        super(RenderControlScanXyAnalysis, self).__init__()

        self.draw_xy_segment_analysis = draw_xy_segment_analysis
        self.draw_xy_segment_result = draw_xy_segment_result
