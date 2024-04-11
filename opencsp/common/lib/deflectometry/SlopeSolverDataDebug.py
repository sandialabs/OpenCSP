"""Class that holds information for debugging a SlopeSolver calculation."""

from typing import Any


class SlopeSolverDataDebug:
    """Class to hold data for debugging SlopeSolver calculations"""

    def __init__(self):
        self.debug_active: bool = False
        self.optic_data: Any = None
        self.slope_solver_figures: list = []
        self.slope_solver_camera_rays_length: float = 0.0
        self.slope_solver_plot_camera_screen_points: bool = False
        self.slope_solver_point_downsample: int = 50
        self.slope_solver_single_plot: bool = False
