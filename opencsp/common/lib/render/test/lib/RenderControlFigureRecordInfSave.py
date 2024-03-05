import opencsp.common.lib.render_control.RenderControlFigureRecord as rcfr

class RenderControlFigureRecordInfSave(rcfr.RenderControlFigureRecord):
    """ A subclass of RenderControlFigureRecord that never finishes saving, for testing save timeouts. """
    def __init__(self, *vargs, **kwargs):
        super().__init__(*vargs, **kwargs)
    
    def save(self, *vargs, **kwargs):
        import time
        while (True):
            time.sleep(1)