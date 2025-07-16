""" """

import copy
import csv

import numpy as np

import opencsp.common.lib.csp.HeliostatConfiguration as hc
from opencsp.common.lib.csp.HeliostatAbstract import HeliostatAbstract
from opencsp.common.lib.csp.MirrorParametricRectangular import MirrorParametricRectangular
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.csp.OpticOrientationAbstract import OpticOrientationAbstract
from opencsp.common.lib.csp.RayTraceable import RayTraceable
from opencsp.common.lib.geometry.FunctionXYContinuous import FunctionXYContinuous
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.RegionXY import Resolution
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.render.View3d import View3d
from opencsp.common.lib.render_control.RenderControlSolarField import RenderControlSolarField


class SolarField(RayTraceable, OpticOrientationAbstract):
    """
    Representation of a heliostat solar field.

    This class manages a collection of heliostats and their configurations, allowing for
    spatial placement and rendering of the solar field.
    """

    # "ChatGPT 4o-mini" assisted with generating this docstring.

    # CONSTRUCTION

    def __init__(
        self,
        heliostats: list[HeliostatAbstract],
        origin_lon_lat: list[float] | tuple[float, float] = None,
        name: str = None,
        short_name: str = None,
    ) -> None:
        """
        Initializes a SolarField object with the specified heliostats and location.

        Parameters
        ----------
        heliostats : list[HeliostatAbstract]
            A list of heliostat objects that make up the solar field.
        origin_lon_lat : list[float] | tuple[float, float], optional
            The longitude and latitude of the solar field's location (default is None).
        name : str, optional
            The name of the solar field (default is None).
        short_name : str, optional
            A short name for the solar field (default is None).
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # assert isinstance(heliostats[0], HeliostatAbstract)

        self.heliostats = heliostats
        OpticOrientationAbstract.__init__(self)

        # Input parameters.
        self.name = name
        self.short_name = short_name
        self.origin_lon_lat = origin_lon_lat

        # # Constructed members.
        # # self.heliostats, self.heliostat_dict = self.heliostats_read_file(heliostat_file, facet_centroids_file)
        # self.heliostat_dict = {heliostat.name: i for i, heliostat in enumerate(heliostats)}
        # self.num_heliostats = len(self.heliostats)
        # self.heliostat_origin_xyz_list = [h.origin for h in self.heliostats]
        # self.heliostat_origin_fit_plane = g3d.best_fit_plane(self.heliostat_origin_xyz_list)  # 3-d plane fit through heliostat origins.
        # self._aimpoint_xyz = None   # (x,y,y) in m. Do not access this member externally; use aimpoint_xyz() function instead.
        # self._when_ymdhmsz = None   # (y,m,d,h,m,s,z). Do not access this member externally; use when_ymdhmsz() function instead.

        # self.set_position_in_space(self.origin, self.rotation)

    def set_heliostat_positions(self, positions: list[Pxyz]):
        """
        Places the heliostats at the specified positions.

        Parameters
        ----------
        positions : list[Pxyz]
            A list of Pxyz objects representing the positions for each heliostat.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the number of positions does not match the number of heliostats.

        Notes
        -----
        The Pxyzs should appear in the same order as the heliostats in `self.heliostats`.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        for heliostat, point in zip(self.heliostats, positions):
            heliostat_position_transform = TransformXYZ.from_V(point)
            heliostat._self_to_parent_transform = heliostat_position_transform

    @classmethod
    def from_csv_files(
        cls,
        long_lat: list[float] | tuple[float, float],
        heliostat_attributes_csv: str,
        facet_attributes_csv: str,
        name=None,
    ):
        """
        Creates a SolarField object from CSV files containing heliostat and facet attributes.

        Parameters
        ----------
        long_lat : list[float] | tuple[float, float]
            The (longitude, latitude) pair defining the location of the solar field.
        heliostat_attributes_csv : str
            The file path to the CSV file containing heliostat attributes.
        facet_attributes_csv : str
            The file path to the CSV file describing the facets of the heliostats.
        name : str, optional
            An optional name for the solar field (default is None).

        Returns
        -------
        SolarField
            A SolarField object initialized with the specified attributes.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        with open(heliostat_attributes_csv) as h_csv:
            h_reader = csv.reader(h_csv)
            h_headers = next(h_reader)
            heliostat_attributes = {
                row[0]: {h_headers[i]: float(attribute) for i, attribute in enumerate(row[1:-1], start=1)}
                for row in h_reader
            }

        with open(facet_attributes_csv) as f_csv:
            f_reader = csv.reader(f_csv)
            f_headers = next(f_reader)
            f_reader_T = np.array(list(f_reader)).T
            f_ids = f_reader_T[0]
            f_positions = Pxyz(f_reader_T[1:])

        flat_function = FunctionXYContinuous("x * y * 0")

        heliostats: list[HeliostatAzEl] = []
        locations: list[Pxyz] = []

        for heliostat_name in heliostat_attributes:

            this_heliostat_attributes = heliostat_attributes[heliostat_name]

            width = this_heliostat_attributes["Facet Width"]
            height = this_heliostat_attributes["Facet Height"]
            mirror_template = MirrorParametricRectangular(flat_function, (width, height))

            heliostat_location = Pxyz(
                [this_heliostat_attributes["X"], this_heliostat_attributes["Y"], this_heliostat_attributes["Z"]]
            )

            heliostat = HeliostatAzEl.from_attributes(
                int(this_heliostat_attributes["Num. Facets"]),
                f_positions,
                mirror_template,
                heliostat_name,
                f_ids,
                this_heliostat_attributes["Pivot Offset"],
            )

            heliostats.append(heliostat)
            locations.append(heliostat_location)

        sf = SolarField(heliostats, long_lat, name=name)
        sf.set_heliostat_positions(locations)
        return sf

    # OVERRIDES

    # override from OpticOrientationAbstract
    @property
    def children(self):
        """
        Retrieves the list of child objects (heliostats) in the solar field.

        Returns
        -------
        list[RayTraceable]
            A list of heliostat objects in the solar field.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        return self.heliostats

    # override OpticOrientationAbstract
    def _add_child_helper(self, new_child: OpticOrientationAbstract):
        raise NotImplementedError("SolarField does not accept new children.")

    # ACCESS

    def heliostat_name_list(self) -> list[str]:
        """
        Returns a list of all the names of heliostats in the solar field.

        The order is the same as the order the heliostats are stored.

        Returns
        -------
        list[str]
            A list of heliostat names.
        """
        name_list = [h.name for h in self.heliostats]
        return name_list

    def lookup_heliostat(self, heliostat_name: str) -> HeliostatAbstract:
        """
        Returns the first HeliostatAbstract in the solar field that matches the given name.

        Parameters
        ----------
        heliostat_name : str
            The name of the heliostat to look up.

        Returns
        -------
        HeliostatAbstract
            The matching heliostat object.

        Raises
        ------
        KeyError
            If no heliostat with the specified name exists in the solar field.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Check input.
        matching_heliostats = filter(lambda heliostat: heliostat.name == heliostat_name, self.heliostats)
        first_matching = next(matching_heliostats, None)
        if first_matching is None:
            raise KeyError(f"No heliostat with the name '{heliostat_name}' appears in the SolarField {self}.")
        # Return.
        return first_matching

    def heliostat_bounding_box_xy(self):
        """
        Returns an axis-aligned bounding box that only takes into account the heliostat origins.

        Returns
        -------
        list[list[float]]
            A list containing the minimum and maximum coordinates of the bounding box in the XY plane.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        heliostat_origins = Pxyz.merge([h._self_to_parent_transform.apply(Pxyz.origin()) for h in self.heliostats])
        x_min = min(heliostat_origins.x)
        x_max = max(heliostat_origins.x)
        y_min = min(heliostat_origins.y)
        y_max = max(heliostat_origins.y)
        xy_min: list[float] = [x_min, y_min]
        xy_max: list[float] = [x_max, y_max]
        return [xy_min, xy_max]

    def heliostat_field_regular_grid_xy(self, n_x: int, n_y: int):
        """
        Generates a regular grid of points in the XY plane based on the bounding box of the heliostats.
        The grid is created by evenly spacing points within the bounding box defined by the heliostat origins.

        The grid is evenly spaced across the entire field.

        Parameters
        ----------
        n_x : int
            The number of points along the X-axis.
        n_y : int
            The number of points along the Y-axis.

        Returns
        -------
        list[list[float]]
            A list of [x, y] coordinates representing the points in the regular grid.
        """
        bbox_xy = self.heliostat_bounding_box_xy()
        xy_min = bbox_xy[0]
        x_min = xy_min[0]
        y_min = xy_min[1]
        xy_max = bbox_xy[1]
        x_max = xy_max[0]
        y_max = xy_max[1]
        grid_xy: list[list[float]] = []
        for x in np.linspace(x_min, x_max, n_x):
            for y in np.linspace(y_min, y_max, n_y):
                grid_xy.append([x, y])
        return grid_xy

    # MODIFICATION

    def set_full_field_tracking(self, aimpoint_xyz: Pxyz, when_ymdhmsz: tuple):
        """
        Configures all heliostats in the solar field to track a specified aim point.

        Parameters
        ----------
        aimpoint_xyz : Pxyz
            The 3D coordinates of the aim point that the heliostats will track, in field-relative coordinates.
        when_ymdhmsz : tuple
            A tuple representing the time (year, month, day, hour, minute, second) when the tracking is set.

        Notes
        -----
        This method updates the internal state of the solar field and each heliostat.
        """
        # Save tracking command.
        self._aimpoint_xyz = aimpoint_xyz
        self._when_ymdhmsz = when_ymdhmsz
        # Set each heliostat tracking.
        for heliostat in self.heliostats:
            heliostat.set_tracking_configuration(aimpoint_xyz, self.origin_lon_lat, when_ymdhmsz)

    def set_full_field_stow(self):
        """
        Configures all heliostats in the solar field to a stowed position.

        See also :py:meth:`HeliostatConfiguration.NSTTF_stow`
        """
        for heliostat in self.heliostats:
            heliostat.set_orientation(hc.NSTTF_stow())

    def set_full_field_face_up(self):
        """
        Configures all heliostats in the solar field to a stowed position.

        See also :py:meth:`HeliostatConfiguration.face_up`
        """
        for heliostat in self.heliostats:
            heliostat.set_orientation(hc.face_up())

    # RENDERING

    def draw(
        self,
        view: View3d,
        solar_field_style: RenderControlSolarField = RenderControlSolarField(),
        transform: TransformXYZ = None,
    ) -> None:
        """
        Draws the solar field in a 3D view.

        Parameters
        ----------
        view : View3d
            The 3D view in which to draw the solar field.
        solar_field_style : RenderControlSolarField, optional
            The style settings for rendering the solar field (default is a new RenderControlSolarField object).
        transform : TransformXYZ, optional
            A transformation to apply to the solar field when drawing (default is None).

        Returns
        -------
        None
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        if transform is None:
            transform = self.self_to_global_tranformation

        origin = transform.apply(Pxyz.origin())

        # Draw Origin
        if solar_field_style.draw_origin:
            origin.draw_points(view)

        # Heliostats.
        if solar_field_style.draw_heliostats:
            for heliostat in self.heliostats:
                transform_heliostat = transform * heliostat._self_to_parent_transform
                heliostat_style = solar_field_style.get_special_style(heliostat.name)
                heliostat.draw(view, heliostat_style, transform_heliostat)

        # Name.
        if solar_field_style.draw_name:
            view.draw_xyz_text(origin.data.T[0], self.name)

    def survey_of_points(self, resolution: Resolution) -> tuple[Pxyz, Vxyz]:
        """
        Returns a grid of equispaced points and the normal vectors at those points.

        Parameters
        ----------
        resolution : Resolution
            The rectangular resolution of the points gathered.

        Returns
        -------
        tuple[Pxyz, Vxyz]
            A tuple containing the sampled points and their corresponding normal vectors.
        """
        # "ChatGPT 4o-mini" assisted with generating this docstring.
        # Get sample point locations (z=0 plane in "child" reference frame)
        return self._survey_of_points_helper(resolution, TransformXYZ.identity())

    def _survey_of_points_helper(
        self, given_resolution: Resolution, frame_transform: TransformXYZ
    ) -> tuple[Pxyz, Vxyz]:
        points, normals = [], []
        for heliostat in self.heliostats:
            heliostat_points, heliostat_normals = heliostat._survey_of_points_helper(
                copy.deepcopy(given_resolution), heliostat._self_to_parent_transform.inv()
            )
            points.append(heliostat_points)
            normals.append(heliostat_normals)

        return Pxyz.merge(points), Vxyz.merge(normals)
