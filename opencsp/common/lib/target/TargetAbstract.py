from abc import ABC, abstractmethod

import numpy as np

from opencsp.common.lib.geometry.Pxy import Pxy
from opencsp.common.lib.geometry.RegionXY import RegionXY
import opencsp.common.lib.render.Color as cl
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror
import opencsp.common.lib.target.target_image as ti
import opencsp.common.lib.tool.unit_conversion as uc


class TargetAbstract(ABC):
    """
    Abstract class inherited by all target classes

    NOTE -- for a mirror subclass to be defined as a mirror one must be able to get the z value at a point and the surface normal at a point.
    """

    def __init__(
        self, image_width: float, image_height: float, dpm: float  # Meters  # Meters  # dots per meter
    ) -> None:
        super().__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.dpm = dpm
        self.comments = ["Target Comments:"]
        # Construct image object.
        self.image = ti.construct_target_image(self.image_width, self.image_height, self.dpm)
        # Set initial pattern.
        # ?? SCAFFOLDING RCB -- RENAME THIS VARIABLE TO "NAME"?  SEE splice_targets_above_below() FOR MAYBE REASON WHY
        self.pattern_description = "blank"

    # ACCESS

    def rows_cols(self):
        """
        Returns the number of rows and columns in the image.

        Returns
        -------
        n_rows : int
            The number of rows in the image.
        n_cols : int
            The number of columns in the image.
        """
        n_rows = self.image.shape[0]
        n_cols = self.image.shape[1]
        return n_rows, n_cols

    def rows_cols_bands(self):
        """
        Returns the number of rows, columns, and (color?) bands in the image.

        Returns
        -------
        n_rows : int
            The number of rows in the image.
        n_cols : int
            The number of columns in the image.
        n_bands : int
            The number of (color?) bands in the image.
        """
        n_rows = self.image.shape[0]
        n_cols = self.image.shape[1]
        n_bands = self.image.shape[2]
        return n_rows, n_cols, n_bands

    def image_size_str_meter(self) -> str:
        """
        Returns a string describing the image size in meters.

        Returns
        -------
        size_str : str
            A string in the format 'w{width:.3f}m_h{height:.3f}m_{dpm:.1f}dpm'.
        """
        return "w{w:.3f}m_h{h:.3f}m_{dpm:.1f}dpm".format(w=self.image_width, h=self.image_height, dpm=round(self.dpm))

    def image_size_str_inch(self) -> str:
        """
        Returns a string describing the image size in inches.

        Returns
        -------
        size_str : str
            A string in the format 'w{width:.3f}in_h{height:.3f}in_{dpi:d}dpi'.
        """
        return "w{w:.3f}in_h{h:.3f}in_{dpi:d}dpi".format(
            w=uc.meter_to_inch(self.image_width),
            h=uc.meter_to_inch(self.image_height),
            dpi=round(uc.dpm_to_dpi(self.dpm)),
        )

    def description_meter(self) -> str:
        """
        Returns a string describing the target in meters.

        Returns
        -------
        description : str
            A string in the format '{pattern_description}__w{width}m_h{height}m_{dpm}dpm'.
        """
        return self.pattern_description + "__" + self.image_size_str_meter()

    def description_inch(self) -> str:
        """
        Returns a string describing the target in inches.

        Returns
        -------
        description : str
            A string in the format '{pattern_description}__w{width}in_h{height}in_{dpi}dpi'.
        """
        return self.pattern_description + "__" + self.image_size_str_inch()

    # MODIFICATION

    def set_pattern_description(self, description: str) -> None:
        """
        Sets the pattern description.

        For the abstract class base implementation the value "blank" is always used.

        Parameters
        ----------
        description : str
            The new pattern description.
        """
        # TODO BGB use the description value
        self.pattern_description = "blank"

    # ?? SCAFFOLDING RCB -- ASK TRISTAN ABOUT THIS
    # @abstractmethod   # ?? SCAFFOLDING RCB -- FILL THIS IN
    # def color_at(self, p:Pxy) -> cl.Color:
    #     """
    #     Gives the color of the point on the target

    #     Parameters
    #     ----------
    #     x : float
    #         horizontal coordinate of the point
    #     y : float
    #         vertical coordinate of the point

    #     Returns
    #     -------
    #     color : cl.Color
    #         Image content at the point (x, y)
    #     """
    #     ...

    # ?? SCAFFOLDING RCB -- FILL THIS IN WITH IMAGE DRAWING
    # def draw(self, view:View3d, mirror_style:RenderControlMirror) -> None:
    #     """
    #     Draws a mirror onto the View3d that was input.

    #     Parameters:
    #     -----------
    #         view: A view 3d object that holds the figure.
    #         mirror_styles: A RenderControlMirror object that holds attibutes about the graph.
    #     """
    #     resolution = mirror_style.resolution

    #     edge_values = self.region.edge_sample(resolution)
    #     inner_values = self.region.points_sample(resolution, 'pixelX')

    #     domain = edge_values.concatenate(inner_values)
    #     p_space = self.location_in_space(domain)
    #     X = p_space.x
    #     Y = p_space.y
    #     Z = p_space.z

    #     tri = Triangulation(domain.x, domain.y)
    #     view.draw_xyz_trisurface(X, Y, Z, surface_style=mirror_style.surface_style, triangles=tri.triangles)

    #     if mirror_style.surface_normals:
    #         points, normals = self.survey_of_points(mirror_style.norm_res)
    #         xyzdxyz = [[point.data, normal.data * mirror_style.norm_len] for point, normal in zip(points, normals)]
    #         view.draw_xyzdxyz_list(xyzdxyz, close=False, style=mirror_style.norm_base_style)

    # ?? SCAFFOLDING RCB -- FILL THIS IN
    # def in_bounds(self, p:Pxy) -> np.ndarray[bool]:
    #     """
    #     Determines what points are valid points on the mirror.

    #     Parameters
    #     -----------
    #     p:Pxy is the set of points in the 2D view of the mirror.

    #     Returns
    #     --------
    #     np.ndarray[bool] is the return type. The length of the output is equal to the length
    #     of the input p. If the ith element of p is in the region defining the mirror, then the
    #     ith element of the ouput will be True. Otherwise the output will be false.

    #     """
    #     mask = self.region.is_inside_or_on_border(p)
    #     return mask

    # ?? SCAFFOLDING RCB -- FILL THIS IN
    # def set_position_in_space(self, fac_origin:Pxyz, fac_rotation:Rotation) -> None:
    #     """
    #     Sets the new origin in space and the rotation around that origin.
    #     Should take into account the position of heliostat, roation of the heliostat, the origin offset
    #     of the facets on the heliostat, and the canting of the facets.
    #     """
    #     # Sets facet's position given heliostat configuration.
    #     # self.origin = np.array(fac_origin) + fac_rotation.dot(fac_origin)
    #     if not issubclass(type(fac_origin), Vxyz):
    #         fac_origin = Pxyz(fac_origin)
    #     self.origin = fac_origin
    #     self.rotation = fac_rotation
