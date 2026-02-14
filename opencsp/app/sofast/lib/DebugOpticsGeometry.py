"""Class holding debug information for proces_optics_geometry calculations"""


class DebugOpticsGeometry:
    """Class for holding debug data for used in 'process_optics_geometry"""

    def __init__(self):
        self.debug_active: bool = False
        """To activate geometry debugging. Default False"""
        self.figures: list = []
        """List to hold figure objects once created."""
        self.save_dir = None
        """Where to save figure objects."""
        self.figure_idx = 0
        """Incrementing figure index, so figures appear in order created."""
