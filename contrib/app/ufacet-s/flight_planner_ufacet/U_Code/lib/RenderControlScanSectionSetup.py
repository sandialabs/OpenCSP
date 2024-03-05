"""


"""

class RenderControlScanSectionSetup():
    """
    Render control for plot axes.
    """

    def __init__(self,
                 draw_section_setup=True,
                 highlight_candidate_heliostats=False,
                 highlight_selected_heliostats=True,
                 highlight_rejected_heliostats=False,
                 ):

        super(RenderControlScanSectionSetup, self).__init__()
        
        self.draw_section_setup             = draw_section_setup
        self.highlight_candidate_heliostats = highlight_candidate_heliostats
        self.highlight_selected_heliostats  = highlight_selected_heliostats
        self.highlight_rejected_heliostats  = highlight_rejected_heliostats
