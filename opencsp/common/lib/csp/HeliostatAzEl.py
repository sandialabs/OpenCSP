import copy
import csv
import numpy as np

from scipy.spatial.transform import Rotation

from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.HeliostatAbstract import HeliostatAbstract
from opencsp.common.lib.csp.HeliostatConfiguration import HeliostatConfiguration
from opencsp.common.lib.csp.MirrorAbstract import MirrorAbstract
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.geometry.Vxyz import Vxyz

EAST = Vxyz([1, 0, 0])
WEST = -EAST
NORTH = Vxyz([0, 1, 0])
SOUTH = -NORTH
UP = Vxyz([0, 0, 1])
DOWN = -UP
DEG2RAD = np.pi / 180


class HeliostatAzEl(HeliostatAbstract):
    """
    Child class of HeliostatAbstract. HeliostatAzEl instances have motion characterized by
    a motor that rotates the heliostat in the azumuth direction (an angle measured
    clockwise from North in UNE cooridinates or from y in XYZ coordinates) and a motor that
    rotates up from the East-North plane (XY plane).
    """

    def __init__(self, facet_ensemble: FacetEnsemble, name: str = None) -> None:
        HeliostatAbstract.__init__(self, facet_ensemble, name)

        # TODO TJL:determine the values an az-el heliostat needs, are these enough
        # TODO TJL:should these be in the constructor?
        # self.az_origin = Pxyz.origin()  # TODO tjlaki: we might want to make these @properties
        # self.el_origin = Pxyz.origin()
        # self.az_pivot = 0
        # self.el_pivot = 0

        # self._default_direction = TransformXYZ.from_R(Rotation.from_euler("x", 90, degrees=True))  # TODO TJL:add default dirction instead of assuming the heliostat points SOUTH
        self._default_direction = TransformXYZ.from_R(Rotation.from_euler("xz", [90 * DEG2RAD, 180 * DEG2RAD]))
        self._pivot = 0
        self._az = 0
        self._el = 0
        self.set_orientation_from_az_el(0, 0)

    # HELIOSTAT AZ-EL ATTRIBUTES

    @property
    def default_direction(self):
        return self._default_direction

    @default_direction.setter
    def default_direction(self, new_default_direction: Rotation):
        self._default_direction = new_default_direction
        self.set_transform_from_az_el(0, 0)

    @property
    def pivot(self):
        return self._pivot

    @pivot.setter
    def pivot(self, new_pivot):
        self._pivot = new_pivot
        self.set_orientation_from_az_el(self._az, self._el)

    @classmethod
    def from_attributes(
        cls,
        number_of_facets: int,
        facet_positions: Pxyz,
        mirror_template: MirrorAbstract,
        name: str = None,
        facet_names: list[str] = None,
        pivot: float = 0,
    ) -> 'HeliostatAzEl':
        """
        Creates a Heliostat of identical mirrors as given by the facet_template.
        Positions the facets as given by the attributes
        """
        if len(facet_positions) != number_of_facets:
            raise ValueError("number_of_facets and length of facet_positions must match")
        if (facet_names is not None) and (len(facet_names) != number_of_facets):
            raise ValueError("number_of_facets and length of facet_names must match")
        if facet_names is None:
            facet_names = [None for _ in range(number_of_facets)]

        mirrors = [copy.deepcopy(mirror_template) for _ in range(number_of_facets)]
        facets = [Facet(mirror, facet_name) for mirror, facet_name in zip(mirrors, facet_names)]
        facet_ensemble = FacetEnsemble(facets)
        facet_ensemble.set_facet_positions(facet_positions)
        heliostat = HeliostatAzEl(facet_ensemble, name)
        heliostat.pivot = pivot
        return heliostat

    @classmethod
    def from_csv_files(
        cls,
        heliostat_name: str,
        heliostat_attributes_csv: str,
        facet_attributes_csv: str,
        mirror_template: MirrorAbstract,
    ) -> tuple['HeliostatAzEl', Pxyz]:
        """returns the heliostat that is requested based on the given information

        Paramters
        ---------
        helisotat_name: str
            the name of the heliostat as given in the csv file
        heliostat_attribute_csv: str
            filepath to the csv file that contains information about the desired heliostat.
        facet_attricute_csv: str
            filepath ot the csv file that describes how the facets in the desired heliost
            will be positoned and named.
        mirror_template: MirrorAbstract
            the desired heliostat will have uniform mirrors, this is the teplate that
            all mirrors will be based on

        Returns
        -------
        `tuple[HeliostatAzEl, Pxyz]`

        HeliostatAzEl
            the helistat that is defined by the csv files
        Pxyz
            the position of the heliostat as given by the heliostat_attributes_csv
        """
        with open(heliostat_attributes_csv) as h_csv:
            h_reader = csv.reader(h_csv)
            h_headers = next(h_reader)
            heliostat_attributes = {
                row[0]: {h_headers[i]: float(attribute) for i, attribute in enumerate(row[1:-1], start=1)}
                for row in h_reader
            }
            # heliostat_attributes = {}
            # for row in h_reader:
            #     heliostat_attributes[row[0]] = {}
            #     for i, attribute in enumerate(row[1:], start=1):
            #         print(attribute)
            #         heliostat_attributes[row[0]][h_headers[i]] = float(attribute)

        if heliostat_name not in heliostat_attributes:
            raise ValueError(f"{heliostat_name} is not a valid heliostat " f"name in {heliostat_attributes_csv}")
        with open(facet_attributes_csv) as f_csv:
            f_reader = csv.reader(f_csv)
            f_headers = next(f_reader)
            f_reader_T = np.array(list(f_reader)).T
            f_ids = f_reader_T[0]
            f_positions = Pxyz(f_reader_T[1:])

        this_heliostat_attributes = heliostat_attributes[heliostat_name]
        heliostat = cls.from_attributes(
            int(this_heliostat_attributes["Num. Facets"]),
            f_positions,
            mirror_template,
            heliostat_name,
            f_ids,
            this_heliostat_attributes["Pivot Offset"],
        )

        heliostat_location = Pxyz(
            [this_heliostat_attributes["X"], this_heliostat_attributes["Y"], this_heliostat_attributes["Z"]]
        )
        return (heliostat, heliostat_location)

    # override from HelistatAbstract
    def from_pointing_vector_to_configuration(self, pointing_vector: Vxyz) -> HeliostatConfiguration:
        # Extract surface normal coordinates.
        n_x = pointing_vector.x[0]
        n_y = pointing_vector.y[0]
        n_z = pointing_vector.z[0]

        # Convert heliostat surface normal to (az,el) coordinates.
        #   Elevation is measured up from horizontal,
        #   Azimuth is measured clockwise from north (compass headings).
        #
        # elevation
        n_xy_norm = np.sqrt((n_x * n_x) + (n_y * n_y))
        el = np.arctan2(n_z, n_xy_norm)
        # azimuth
        # nu is the angle to the projection of the surface normal onto the (x,y) plane, measured ccw from the x axis.
        nu = np.arctan2(n_y, n_x)
        az = (np.pi / 2) - nu  # Measured cw from the y axis.

        return HeliostatConfiguration('az-el', az, el)

    # override from HelistatAbstract
    def movement_transform(self, config: HeliostatConfiguration):
        az_angle, el_angle = config.get_values()
        return self.transform_from_az_el(az_angle, el_angle)

    # override from HelistatAbstract
    @property
    def current_configuration(self) -> HeliostatConfiguration:
        return HeliostatConfiguration('az-el', az=self._az, el=self._el)

    # override from HelistatAbstract
    @current_configuration.setter
    def current_configuration(self, new_current_configuration: HeliostatConfiguration):
        az, el = new_current_configuration.get_values()
        self._az = az
        self._el = el

    def transform_from_az_el(self, az_angle: float, el_angle: float):
        """movement_transform for an azimuth and elevation based heliostat."""
        rotation_about_x = (np.pi / 2) - el_angle
        rotation_about_z = np.pi - az_angle
        pivot_transform = TransformXYZ.from_V(UP * self.pivot)
        reorient_from_default_direction = self._default_direction

        # rotation_about_z = -az_angle
        # rotation_about_x = -el_angle

        el_rotation = Rotation.from_euler('x', rotation_about_x, degrees=False)
        transform_el = TransformXYZ.from_R(el_rotation)

        az_rotation = Rotation.from_euler('z', rotation_about_z, degrees=False)
        transform_az = TransformXYZ.from_R(az_rotation)

        # el_rotation_axis: np.ndarray = UP.rotate(az_rotation).data.T[0]
        # el_rotation = Rotation.from_rotvec(el_angle * el_rotation_axis, degrees=degrees)
        # transform_el = TransformXYZ.from_R(el_rotation)

        composite_transform = (
            transform_az
            * transform_el
            *
            #    reorient_from_default_direction *
            pivot_transform
        )
        return composite_transform

    def set_orientation_from_az_el(self, az_angle: float, el_angle: float):
        self._az = az_angle
        self._el = el_angle
        transform = self.transform_from_az_el(az_angle, el_angle)
        self.facet_ensemble._self_to_parent_transform = transform
        config = HeliostatConfiguration('az-el', az=az_angle, el=el_angle)
        self.set_orientation(config)

    pass  # end of HeliostatAzEl
