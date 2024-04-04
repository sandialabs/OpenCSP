from dataclasses import dataclass, field

from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as h5


@dataclass
class DistanceOpticScreen(h5.HDF5_IO_Abstract):
    """Represents a distance measurement between the optic and the center of the screen. This measurement is typically
    achieved by displaying the crosshairs from the SofastGUI and measuring from the origin of the optic
    (measurement_point 0,0,0) to the center of the crosshairs.

    The optic's origin is typically the center of the optic for on-axis optics, such as with spherical or flat mirrors
    that are symetric around their midpoint."""

    v_measure_point_facet: Vxyz = field(default_factory=lambda: Vxyz((0.0, 0.0, 0.0)))
    """ Location of measure point, meters. """
    dist_optic_screen: float = 0
    """ Optic-screen distance, meters. """

    def save_to_hdf(self, file: str, prefix: str) -> None:
        """Saves data to given file. Data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        datasets = [prefix + '/v_measure_point_facet', prefix + '/dist_optic_screen']
        data = [self.v_measure_point_facet.data.squeeze(), self.dist_optic_screen]

        # Save data
        h5.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str):
        """Loads data from given file. Assumes data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        groups, file_names_and_shapes = h5.get_groups_and_datasets(file)
        file_names = [name for name, shape in file_names_and_shapes]
        datasets = []
        for s in ['/measure_point', '/v_measure_point_facet', '/dist_optic_screen', '/optic_screen_dist']:
            dataset = prefix + s
            if dataset in file_names:
                datasets.append(dataset)

        # Load grid data
        # datasets = [prefix + '/measure_point', prefix + '/dist_optic_screen']
        kwargs = h5.load_hdf5_datasets(datasets, file)

        # TODO update all existing HDF5 files to use consistent naming
        if 'measure_point' in kwargs:
            kwargs['v_measure_point_facet'] = kwargs['measure_point']
            del kwargs['measure_point']
        if 'optic_screen_dist' in kwargs:
            kwargs['dist_optic_screen'] = kwargs['optic_screen_dist']
            del kwargs['optic_screen_dist']

        kwargs['v_measure_point_facet'] = Vxyz(kwargs['v_measure_point_facet'])

        return cls(**kwargs)
