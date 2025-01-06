"""Class that holds information for debugging a SlopeSolver calculation."""

from typing import Any


class SlopeSolverDataDebug:
    """Class to hold data for debugging SlopeSolver calculations"""

    def __init__(self):
        self.debug_active: bool = False
        """To activate slope solver debugging. (Default False)"""
        self.optic_data: Any = None
        """Representation of optic (Facet/Mirror) being solved for.
        The geometry data in this object is used to create visualization plots.
        This information is updated automatically during SOFAST execution and will
        overwrite any previously user-given values. (Default None)"""
        self.slope_solver_figures: list = []
        """List to hold figure objects once created."""
        self.slope_solver_camera_rays_length: float = 0.0
        """The length (meters) of camera rays to draw when plotting the 3d slope solving scenario plot. (Default 0.0)"""
        self.slope_solver_plot_camera_screen_points: bool = False
        """To include scatter plot of xyz screen point locations seen by camera in slope solving scenario plot. (Default False)"""
        self.slope_solver_point_downsample: int = 50
        """The downsample factor (to save computing resources) to apply to screen points
        Only applicable if plotting screen points is enabled with the
        `SlopeSolverDataDebug.slope_solver_plot_camera_screen_points` flag). (Default 50)"""
        self.slope_solver_single_plot: bool = False
        """Flag to plot all iterations of the slope solving algorithm on one plot (True) or create a separate
        plot for each iteration (False). Default False (new plot for each iteration)"""
