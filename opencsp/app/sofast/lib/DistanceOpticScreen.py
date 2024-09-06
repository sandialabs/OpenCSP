from dataclasses import dataclass, field

from opencsp.common.lib.geometry.Vxyz import Vxyz
import opencsp.common.lib.tool.hdf5_tools as h5


@dataclass
class DistanceOpticScreen(h5.HDF5_IO_Abstract):
    """Represents a distance measurement between the optic and the origin of the screen coordinate system.
    This measurement is typically achieved by displaying the crosshairs from the SofastGUI and measuring
    from a point on the optic (measurement_point x,y,z) to the center of the crosshairs.

    The measure point is typically the center of the optic for on-axis optics, such as with spherical or flat
    mirrors that are symmetric around their midpoint.
    """

    v_measure_point_facet: Vxyz = field(default_factory=lambda: Vxyz((0.0, 0.0, 0.0)))
    """ Location of measure point, meters. """
    dist_optic_screen: float = 0
    """ Optic-screen distance, meters. """

    def save_to_hdf(self, file: str, prefix: str = '') -> None:
        """Saves data to given file. Data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to save to
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        datasets = [prefix + 'v_measure_point_facet', prefix + 'dist_optic_screen']
        data = [self.v_measure_point_facet.data.squeeze(), self.dist_optic_screen]

        # Save data
        h5.save_hdf5_datasets(data, datasets, file)

    @classmethod
    def load_from_hdf(cls, file: str, prefix: str = ''):
        """Loads data from given file. Assumes data is stored as: PREFIX + Folder/Field_1

        Parameters
        ----------
        file : str
            HDF file to load from
        prefix : str, optional
            Prefix to append to folder path within HDF file (folders must be separated by "/").
            Default is empty string ''.
        """
        datasets = [prefix + 'v_measure_point_facet', prefix + 'dist_optic_screen']

        # Load data
        kwargs = h5.load_hdf5_datasets(datasets, file)
        kwargs['v_measure_point_facet'] = Vxyz(kwargs['v_measure_point_facet'])

        return cls(**kwargs)
