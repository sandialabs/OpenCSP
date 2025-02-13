"""


"""

import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render_control.RenderControlText as rctxt


class RenderControlScanPass:
    """
    Render control for a scan pass within a UAS flight.
    """

    def __init__(
        self,
        draw_core_segment=True,
        core_segment_style=rcps.outline(color="c", linewidth=4),
        draw_segment_of_interest=True,
        segment_of_interest_style=rcps.outline(color="brown"),
        draw_idx=False,
        idx_style=rctxt.default(color="k"),
    ):
        super(RenderControlScanPass, self).__init__()

        self.draw_core_segment = draw_core_segment
        self.core_segment_style = core_segment_style
        self.draw_segment_of_interest = draw_segment_of_interest
        self.segment_of_interest_style = segment_of_interest_style
        self.draw_idx = draw_idx
        self.idx_style = idx_style


# COMMON CASES


def default():
    return RenderControlScanPass()
